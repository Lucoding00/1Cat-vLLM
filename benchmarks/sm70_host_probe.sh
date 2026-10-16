#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Root-free host probe for MoE expert tiering on multi-GPU V100 boxes.
#
# Needs only bash, coreutils, gcc (with OpenMP) and the NVIDIA driver
# (libcuda.so.1, nvidia-smi).  No root, numactl, nvbandwidth, nvcc, Python
# or torch.
#
# Measures:
#   * CPU / NUMA / GPU topology from /sys and nvidia-smi
#   * DRAM bandwidth per NUMA node (1 / 4 / all physical cores, first-touch
#     local memory) plus cross-node and whole-machine runs
#   * Host<->GPU bandwidth with pinned memory allocated by a thread bound to
#     the GPU's NUMA node: single GPU, GPU pairs, per-node groups, all GPUs
#   * Optionally, how much pinned memory each NUMA node can lock
#
# Usage:
#   bash benchmarks/sm70_host_probe.sh [--dram-mib 256] [--dram-iters 5]
#       [--pcie-mib 256] [--pcie-iters 20] [--pin-gib 0] [--skip-dram]
#       [--skip-pcie] [--out DIR]
#
# Stop GPU workloads (vLLM server) first, otherwise the numbers are skewed.

set -uo pipefail

DRAM_MIB=256
DRAM_ITERS=5
PCIE_MIB=256
PCIE_ITERS=20
PIN_GIB=0
SKIP_DRAM=0
SKIP_PCIE=0
OUT=""
SYS=${SM70_PROBE_SYSFS:-/sys}  # override only for testing

usage() { sed -n '5,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dram-mib) DRAM_MIB=$2; shift 2 ;;
    --dram-iters) DRAM_ITERS=$2; shift 2 ;;
    --pcie-mib) PCIE_MIB=$2; shift 2 ;;
    --pcie-iters) PCIE_ITERS=$2; shift 2 ;;
    --pin-gib) PIN_GIB=$2; shift 2 ;;
    --skip-dram) SKIP_DRAM=1; shift ;;
    --skip-pcie) SKIP_PCIE=1; shift ;;
    --out) OUT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$OUT" ]] || OUT="sm70_host_probe_$(hostname -s 2>/dev/null || echo host)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
exec > >(tee -a "$OUT/report.txt") 2>&1

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

section() { printf '\n==== %s ====\n' "$*"; }

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# "0-3,8,10-11" -> one cpu per line
expand_cpulist() {
  local part
  local IFS=,
  for part in $1; do
    if [[ $part == *-* ]]; then seq "${part%-*}" "${part#*-}"; else echo "$part"; fi
  done
}

node_ids() {
  if compgen -G "$SYS/devices/system/node/node[0-9]*" >/dev/null; then
    for d in "$SYS"/devices/system/node/node[0-9]*; do echo "${d##*node}"; done | sort -n
  else
    echo 0
  fi
}

node_cpulist() {
  local f=$SYS/devices/system/node/node$1/cpulist
  if [[ -r $f ]]; then cat "$f"; else cat "$SYS/devices/system/cpu/online"; fi
}

# First hardware thread of every physical core on node $1, comma separated.
node_physical_cores() {
  local cpu sib first
  for cpu in $(expand_cpulist "$(node_cpulist "$1")"); do
    sib=$(cat "$SYS/devices/system/cpu/cpu$cpu/topology/thread_siblings_list" 2>/dev/null || echo "$cpu")
    first=${sib%%[,-]*}
    [[ $first == "$cpu" ]] && echo "$cpu"
  done | paste -sd, -
}

first_n() { echo "$1" | tr , '\n' | head -n "$2" | paste -sd, -; }
count_list() { echo "$1" | tr , '\n' | grep -c .; }

# ---------------------------------------------------------------------------
# embedded C sources
# ---------------------------------------------------------------------------

cat >"$WORK/dram.c" <<'EOF'
#define _GNU_SOURCE
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

/* usage: dram MIB ITERS MEMNODE(-1 = first-touch local) CPU[,CPU...] */
int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s MIB ITERS MEMNODE CPULIST\n", argv[0]);
    return 2;
  }
  size_t bytes = (size_t)atol(argv[1]) << 20;
  int iters = atoi(argv[2]);
  int memnode = atoi(argv[3]);
  static int cpus[8192];
  int n = 0;
  char *s = strdup(argv[4]);
  for (char *tok = strtok(s, ","); tok && n < 8192; tok = strtok(NULL, ","))
    cpus[n++] = atoi(tok);
  size_t cnt = bytes / sizeof(double);
  double best_read = 1e30, best_copy = 1e30, tstart = 0;
  volatile double sink = 0;
  int failures = 0;

  omp_set_dynamic(0);
#pragma omp parallel num_threads(n) reduction(+ : failures)
  {
    int tid = omp_get_thread_num();
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpus[tid], &set);
    if (sched_setaffinity(0, sizeof set, &set)) failures++;
    if (memnode >= 0) {
      unsigned long mask[16] = {0};
      mask[memnode / 64] |= 1UL << (memnode % 64);
      /* MPOL_BIND = 2; per-thread policy, no root needed */
      if (syscall(SYS_set_mempolicy, 2, mask, 16 * 64)) failures++;
    }
    double *a = aligned_alloc(64, bytes);
    double *b = aligned_alloc(64, bytes);
    if (!a || !b) {
      failures++;
    } else {
      for (size_t i = 0; i < cnt; i++) { a[i] = 1.0 + (double)(i & 7); b[i] = 0; }
      for (int k = 0; k < 2; k++) {
        for (int it = 0; it < iters; it++) {
#pragma omp barrier
#pragma omp master
          tstart = now();
#pragma omp barrier
          if (k == 0) {
            double acc = 0;
            for (size_t i = 0; i < cnt; i++) acc += a[i];
            if (acc == -1.0) sink += acc;
          } else {
            memcpy(b, a, bytes);
          }
#pragma omp barrier
#pragma omp master
          {
            double dt = now() - tstart;
            if (k == 0) { if (dt < best_read) best_read = dt; }
            else if (dt < best_copy) best_copy = dt;
          }
        }
      }
    }
#pragma omp barrier
    free(a);
    free(b);
  }
  /* read GB/s, copy GB/s (STREAM convention: read + write bytes), failures */
  printf("%.2f %.2f %d\n", n * (double)bytes / best_read / 1e9,
         2.0 * n * (double)bytes / best_copy / 1e9, failures);
  return 0;
}
EOF

cat >"$WORK/pcie.c" <<'EOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* CUDA driver API, declared by hand so neither CUDA headers nor nvcc are
 * needed; libcuda.so.1 ships with the NVIDIA driver. */
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;

static CUresult (*p_cuInit)(unsigned);
static CUresult (*p_cuDeviceGetByPCIBusId)(CUdevice *, const char *);
static CUresult (*p_cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
static CUresult (*p_cuCtxSetCurrent)(CUcontext);
static CUresult (*p_cuMemHostAlloc)(void **, size_t, unsigned);
static CUresult (*p_cuMemFreeHost)(void *);
static CUresult (*p_cuMemAlloc)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemFree)(CUdeviceptr);
static CUresult (*p_cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
static CUresult (*p_cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
static CUresult (*p_cuStreamCreate)(CUstream *, unsigned);
static CUresult (*p_cuStreamSynchronize)(CUstream);
static CUresult (*p_cuStreamDestroy)(CUstream);
static CUresult (*p_cuGetErrorString)(CUresult, const char **);

static void *lib;
static void load(void **fn, const char *sym) {
  *fn = dlsym(lib, sym);
  if (!*fn) { fprintf(stderr, "libcuda: missing symbol %s\n", sym); exit(3); }
}

static void check(CUresult r, const char *what, const char *who) {
  if (r) {
    const char *msg = "?";
    if (p_cuGetErrorString) p_cuGetErrorString(r, &msg);
    fprintf(stderr, "%s failed for %s: %s (%d)\n", what, who, msg, r);
    exit(4);
  }
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static void parse_cpulist(const char *s, cpu_set_t *set) {
  CPU_ZERO(set);
  char *dup = strdup(s);
  for (char *tok = strtok(dup, ","); tok; tok = strtok(NULL, ",")) {
    int a, b;
    if (sscanf(tok, "%d-%d", &a, &b) == 2) {
      for (int c = a; c <= b; c++) CPU_SET(c, set);
    } else {
      CPU_SET(atoi(tok), set);
    }
  }
  free(dup);
}

/* spec = LABEL/PCI_BUS_ID/CPULIST */
typedef struct {
  char label[32], bus[64];
  cpu_set_t cpus;
  double h0, h1, d0, d1;
} job_t;

static size_t g_bytes;
static int g_iters;
static pthread_barrier_t g_bar;

static void parse_spec(const char *spec, job_t *j) {
  char cpus[4096];
  memset(j, 0, sizeof *j);
  if (sscanf(spec, "%31[^/]/%63[^/]/%4095s", j->label, j->bus, cpus) != 3) {
    fprintf(stderr, "bad spec %s\n", spec);
    exit(2);
  }
  parse_cpulist(cpus, &j->cpus);
}

static CUcontext open_ctx(job_t *j) {
  CUdevice dev;
  CUcontext ctx;
  sched_setaffinity(0, sizeof j->cpus, &j->cpus);
  check(p_cuDeviceGetByPCIBusId(&dev, j->bus), "cuDeviceGetByPCIBusId", j->label);
  check(p_cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain", j->label);
  check(p_cuCtxSetCurrent(ctx), "cuCtxSetCurrent", j->label);
  return ctx;
}

static void *bw_worker(void *arg) {
  job_t *j = arg;
  open_ctx(j);
  void *host;
  CUdeviceptr dev;
  CUstream st;
  /* allocated after binding, so pages land on the GPU's NUMA node */
  check(p_cuMemHostAlloc(&host, g_bytes, 0), "cuMemHostAlloc", j->label);
  memset(host, 1, g_bytes);
  check(p_cuMemAlloc(&dev, g_bytes), "cuMemAlloc", j->label);
  check(p_cuStreamCreate(&st, 1), "cuStreamCreate", j->label);
  check(p_cuMemcpyHtoDAsync(dev, host, g_bytes, st), "warmup H2D", j->label);
  check(p_cuMemcpyDtoHAsync(host, dev, g_bytes, st), "warmup D2H", j->label);
  check(p_cuStreamSynchronize(st), "sync", j->label);

  pthread_barrier_wait(&g_bar);
  j->h0 = now();
  for (int i = 0; i < g_iters; i++)
    check(p_cuMemcpyHtoDAsync(dev, host, g_bytes, st), "H2D", j->label);
  check(p_cuStreamSynchronize(st), "sync", j->label);
  j->h1 = now();

  pthread_barrier_wait(&g_bar);
  j->d0 = now();
  for (int i = 0; i < g_iters; i++)
    check(p_cuMemcpyDtoHAsync(host, dev, g_bytes, st), "D2H", j->label);
  check(p_cuStreamSynchronize(st), "sync", j->label);
  j->d1 = now();

  p_cuStreamDestroy(st);
  p_cuMemFree(dev);
  p_cuMemFreeHost(host);
  return NULL;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr,
            "usage: %s bw MIB ITERS SPEC...\n"
            "       %s pin GIB CHUNK_GIB SPEC\n"
            "SPEC = LABEL/PCI_BUS_ID/CPULIST\n", argv[0], argv[0]);
    return 2;
  }
  lib = dlopen("libcuda.so.1", RTLD_NOW);
  if (!lib) { fprintf(stderr, "cannot load libcuda.so.1: %s\n", dlerror()); return 3; }
  load((void **)&p_cuInit, "cuInit");
  load((void **)&p_cuDeviceGetByPCIBusId, "cuDeviceGetByPCIBusId");
  load((void **)&p_cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
  load((void **)&p_cuCtxSetCurrent, "cuCtxSetCurrent");
  load((void **)&p_cuMemHostAlloc, "cuMemHostAlloc");
  load((void **)&p_cuMemFreeHost, "cuMemFreeHost");
  load((void **)&p_cuMemAlloc, "cuMemAlloc_v2");
  load((void **)&p_cuMemFree, "cuMemFree_v2");
  load((void **)&p_cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
  load((void **)&p_cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
  load((void **)&p_cuStreamCreate, "cuStreamCreate");
  load((void **)&p_cuStreamSynchronize, "cuStreamSynchronize");
  load((void **)&p_cuStreamDestroy, "cuStreamDestroy_v2");
  load((void **)&p_cuGetErrorString, "cuGetErrorString");
  check(p_cuInit(0), "cuInit", "driver");

  if (strcmp(argv[1], "bw") == 0) {
    g_bytes = (size_t)atol(argv[2]) << 20;
    g_iters = atoi(argv[3]);
    int n = argc - 4;
    job_t *jobs = calloc(n, sizeof *jobs);
    pthread_t *th = calloc(n, sizeof *th);
    pthread_barrier_init(&g_bar, NULL, n);
    for (int i = 0; i < n; i++) parse_spec(argv[4 + i], &jobs[i]);
    for (int i = 0; i < n; i++) pthread_create(&th[i], NULL, bw_worker, &jobs[i]);
    for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
    double moved = (double)g_bytes * g_iters;
    double hmin = 1e30, hmax = 0, dmin = 1e30, dmax = 0;
    for (int i = 0; i < n; i++) {
      job_t *j = &jobs[i];
      printf("RESULT gpu=%s h2d=%.2f d2h=%.2f\n", j->label,
             moved / (j->h1 - j->h0) / 1e9, moved / (j->d1 - j->d0) / 1e9);
      if (j->h0 < hmin) hmin = j->h0;
      if (j->h1 > hmax) hmax = j->h1;
      if (j->d0 < dmin) dmin = j->d0;
      if (j->d1 > dmax) dmax = j->d1;
    }
    printf("RESULT aggregate h2d=%.2f d2h=%.2f\n", n * moved / (hmax - hmin) / 1e9,
           n * moved / (dmax - dmin) / 1e9);
    return 0;
  }

  if (strcmp(argv[1], "pin") == 0) {
    int gib = atoi(argv[2]), chunk = atoi(argv[3]);
    job_t j;
    parse_spec(argv[4], &j);
    open_ctx(&j);
    void **held = calloc(gib / chunk + 2, sizeof *held);
    int got = 0;
    const char *err = "";
    double t0 = now();
    while (got * chunk < gib) {
      CUresult r = p_cuMemHostAlloc(&held[got], (size_t)chunk << 30, 0);
      if (r) { p_cuGetErrorString(r, &err); break; }
      got++;
    }
    double dt = now() - t0;
    printf("RESULT pin node=%s pinned_gib=%d seconds=%.1f error=%s\n", j.label,
           got * chunk, dt, err);
    for (int i = 0; i < got; i++) p_cuMemFreeHost(held[i]);
    return 0;
  }
  fprintf(stderr, "unknown mode %s\n", argv[1]);
  return 2;
}
EOF

# ---------------------------------------------------------------------------
# 1. topology
# ---------------------------------------------------------------------------

section "system"
echo "date: $(date -Is)"
echo "host: $(hostname)  kernel: $(uname -r)"
lscpu 2>/dev/null | grep -E '^(Model name|Socket|Core|Thread|CPU\(s\)|NUMA|L3|CPU max MHz)' || true
echo -n "AVX-512 flags: "
grep -m1 -o 'avx512[a-z_0-9]*' /proc/cpuinfo | sort -u | paste -sd' ' - || echo none
echo "F16C: $(grep -m1 -qw f16c /proc/cpuinfo && echo yes || echo no)  FMA: $(grep -m1 -qw fma /proc/cpuinfo && echo yes || echo no)"
echo "ulimit -l (memlock): $(ulimit -l)"
echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
lscpu >"$OUT/lscpu.txt" 2>/dev/null || true

section "NUMA nodes"
declare -A NODE_CORES
NODES=$(node_ids)
for n in $NODES; do
  NODE_CORES[$n]=$(node_physical_cores "$n")
  mem=$(awk '/MemTotal/ {printf "%.1f GiB total", $4/1048576} /MemFree/ {printf ", %.1f GiB free", $4/1048576}' \
        "$SYS/devices/system/node/node$n/meminfo" 2>/dev/null)
  echo "node$n: cpus=$(node_cpulist "$n")  physical=$(count_list "${NODE_CORES[$n]}")  ${mem:-}"
done
dist=$SYS/devices/system/node/node0/distance
[[ -r $dist ]] && for n in $NODES; do echo "distance node$n: $(cat "$SYS/devices/system/node/node$n/distance")"; done

section "GPUs"
GPU_IDX=(); GPU_BUS=(); GPU_NODE=()
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,memory.used \
    --format=csv 2>/dev/null
  while IFS=, read -r idx bus; do
    idx=${idx// /}; bus=${bus// /}
    sysbus=$(echo "${bus: -12}" | tr 'A-F' 'a-f')
    node=$(cat "$SYS/bus/pci/devices/$sysbus/numa_node" 2>/dev/null || echo -1)
    [[ $node -lt 0 ]] && node=$(echo "$NODES" | head -n1)
    GPU_IDX+=("$idx"); GPU_BUS+=("${bus: -12}"); GPU_NODE+=("$node")
    echo "GPU$idx bus=${bus: -12} numa_node=$node"
  done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null)
  nvidia-smi topo -m >"$OUT/nvidia_topo.txt" 2>&1 && cat "$OUT/nvidia_topo.txt"
  nvidia-smi -q >"$OUT/nvidia_smi_q.txt" 2>&1 || true
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [[ $busy -gt 0 ]] && echo "WARNING: $busy GPU compute process(es) running; stop them for clean numbers"
else
  echo "nvidia-smi not found; PCIe tests skipped"
fi
echo "note: pcie.link.gen.current drops to Gen1 when idle; check gen.max / measured GB/s"

# ---------------------------------------------------------------------------
# 2. DRAM bandwidth
# ---------------------------------------------------------------------------

MIN_NODE_READ=""
if [[ $SKIP_DRAM -eq 0 ]]; then
  section "DRAM bandwidth (GB/s, ${DRAM_MIB} MiB per array per core)"
  if gcc -O3 -march=native -ffast-math -fopenmp "$WORK/dram.c" -o "$WORK/dram" 2>"$WORK/dram.log"; then
    run_dram() {  # label memnode cpulist
      local r
      r=$("$WORK/dram" "$DRAM_MIB" "$DRAM_ITERS" "$2" "$3") || { echo "$1: failed"; return; }
      read -r rd cp fail <<<"$r"
      printf '%-34s cores=%-3s read=%8s copy=%8s%s\n' "$1" "$(count_list "$3")" "$rd" "$cp" \
        "$([[ $fail != 0 ]] && echo "  (binding failures: $fail)")"
      echo "RESULT dram label=\"$1\" cores=$(count_list "$3") read=$rd copy=$cp" >>"$OUT/results.txt"
      LAST_READ=$rd
    }
    ALL_CORES=""
    for n in $NODES; do
      cores=${NODE_CORES[$n]}
      total=$(count_list "$cores")
      ALL_CORES=${ALL_CORES:+$ALL_CORES,}$cores
      for k in 1 4 8; do
        [[ $k -lt $total ]] && run_dram "node$n local, $k core(s)" -1 "$(first_n "$cores" "$k")"
      done
      run_dram "node$n local, all cores" -1 "$cores"
      if [[ -z $MIN_NODE_READ ]] || awk "BEGIN{exit !($LAST_READ < $MIN_NODE_READ)}"; then
        MIN_NODE_READ=$LAST_READ
      fi
    done
    node_count=$(echo "$NODES" | wc -w)
    if [[ $node_count -gt 1 ]]; then
      first=$(echo "$NODES" | head -n1); other=$(echo "$NODES" | sed -n 2p)
      run_dram "node$first cpus -> node$other memory" "$other" "${NODE_CORES[$first]}"
      run_dram "all nodes, local memory" -1 "$ALL_CORES"
    fi
  else
    echo "gcc/OpenMP build failed:"; cat "$WORK/dram.log"
  fi
fi

# ---------------------------------------------------------------------------
# 3. Host <-> GPU bandwidth
# ---------------------------------------------------------------------------

MIN_ALL_H2D=""
if [[ $SKIP_PCIE -eq 0 && ${#GPU_IDX[@]} -gt 0 ]]; then
  section "Host<->GPU pinned bandwidth (GB/s, ${PCIE_MIB} MiB x ${PCIE_ITERS})"
  if gcc -O2 -pthread "$WORK/pcie.c" -o "$WORK/pcie" -ldl 2>"$WORK/pcie.log"; then
    spec() {  # position in GPU arrays -> LABEL/BUS/CPULIST
      echo "${GPU_IDX[$1]}/${GPU_BUS[$1]}/$(node_cpulist "${GPU_NODE[$1]}")"
    }
    run_pcie() {  # label positions...
      local label=$1; shift
      local specs=() p out
      for p in "$@"; do specs+=("$(spec "$p")"); done
      out=$("$WORK/pcie" bw "$PCIE_MIB" "$PCIE_ITERS" "${specs[@]}" 2>&1) || { echo "$label: failed: $out"; return; }
      local per agg
      per=$(echo "$out" | awk '/RESULT gpu=/ {sub("gpu=","",$2); sub("h2d=","",$3); sub("d2h=","",$4); printf "GPU%s %s/%s  ", $2, $3, $4}')
      agg=$(echo "$out" | awk '/RESULT aggregate/ {sub("h2d=","",$3); sub("d2h=","",$4); printf "%s/%s", $3, $4}')
      printf '%-22s total h2d/d2h=%-13s | %s\n' "$label" "$agg" "$per"
      echo "$out" | sed "s/^RESULT /RESULT pcie scenario=\"$label\" /" >>"$OUT/results.txt"
      LAST_MIN_H2D=$(echo "$out" | awk '/RESULT gpu=/ {sub("h2d=","",$3); if (m == "" || $3 < m) m = $3} END {print m}')
    }
    G=${#GPU_IDX[@]}
    for ((i = 0; i < G; i++)); do run_pcie "single GPU${GPU_IDX[$i]}" "$i"; done
    for ((i = 0; i + 1 < G; i += 2)); do run_pcie "pair ${GPU_IDX[$i]}+${GPU_IDX[$((i + 1))]}" "$i" $((i + 1)); done
    for ((i = 0; i + 2 < G; i++)); do
      (( i % 4 < 2 )) && run_pcie "pair ${GPU_IDX[$i]}+${GPU_IDX[$((i + 2))]}" "$i" $((i + 2))
    done
    for n in $NODES; do
      group=()
      for ((i = 0; i < G; i++)); do [[ ${GPU_NODE[$i]} == "$n" ]] && group+=("$i"); done
      [[ ${#group[@]} -gt 1 && ${#group[@]} -lt $G ]] && run_pcie "node$n GPUs (${#group[@]})" "${group[@]}"
    done
    run_pcie "all GPUs ($G)" $(seq 0 $((G - 1)))
    MIN_ALL_H2D=${LAST_MIN_H2D:-}

    if [[ $PIN_GIB -gt 0 ]]; then
      section "pinned capacity (target ${PIN_GIB} GiB per NUMA node)"
      for n in $NODES; do
        pos=""
        for ((i = 0; i < G; i++)); do [[ ${GPU_NODE[$i]} == "$n" ]] && { pos=$i; break; }; done
        [[ -z $pos ]] && pos=0
        out=$("$WORK/pcie" pin "$PIN_GIB" 4 "$n/${GPU_BUS[$pos]}/$(node_cpulist "$n")" 2>&1)
        echo "$out" | tee -a "$OUT/results.txt"
      done
    fi
  else
    echo "pcie build failed:"; cat "$WORK/pcie.log"
  fi
fi

# ---------------------------------------------------------------------------
# 4. summary
# ---------------------------------------------------------------------------

section "recommended cost-model inputs"
echo "cpu_bw_gbs (weakest node, all cores, read): ${MIN_NODE_READ:-n/a}"
echo "pcie_gbs_per_gpu (weakest GPU, all GPUs concurrent, H2D): ${MIN_ALL_H2D:-n/a}"
echo
echo "python benchmarks/sm70_moe_expert_tiering.py analyze glm_routes.npz --moe-layers 3-44 \\"
echo "    --tp 4 --pp 2 --cpu-bw-gbs ${MIN_NODE_READ:-100} --pcie-gbs-per-gpu ${MIN_ALL_H2D:-6}"
echo
echo "full report: $OUT/report.txt  (raw: results.txt, lscpu.txt, nvidia_topo.txt, nvidia_smi_q.txt)"
