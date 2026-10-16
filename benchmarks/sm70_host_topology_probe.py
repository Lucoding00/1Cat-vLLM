# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe host memory and PCIe bandwidth for SM70 MoE expert tiering.

Needs neither root, ``numactl`` nor ``nvbandwidth``:

- topology comes from ``/sys`` (NUMA nodes, physical cores, GPU NUMA nodes);
- DRAM bandwidth uses one process per physical core, each pinned with
  ``sched_setaffinity`` and touching its own arrays first, so pages land on
  the local node;
- host/device bandwidth uses PyTorch, with every GPU's pinned buffer allocated
  by a thread bound to that GPU's NUMA node.

Stop any server using the GPUs first. The results feed
``benchmarks/sm70_moe_expert_tiering.py analyze``: ``--cpu-bw-gbs`` and
``--pcie-gbs-per-gpu``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import resource
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

SYS_NODE = Path("/sys/devices/system/node")
SYS_CPU = Path("/sys/devices/system/cpu")


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------


def parse_cpulist(text: str) -> list[int]:
    """Parse a kernel cpulist such as ``"0-17,36-53"``."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = (int(v) for v in part.split("-", 1))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return cpus


def numa_nodes(sys_node: Path = SYS_NODE) -> dict[int, list[int]]:
    nodes = {}
    for path in sorted(sys_node.glob("node[0-9]*")):
        cpulist = (path / "cpulist").read_text()
        cpus = parse_cpulist(cpulist)
        if cpus:
            nodes[int(path.name[4:])] = cpus
    if not nodes:
        nodes[0] = sorted(os.sched_getaffinity(0))
    return nodes


def physical_cores(cpus: list[int], sys_cpu: Path = SYS_CPU) -> list[int]:
    """Keep the first hyperthread of every core, preserving order."""
    chosen: list[int] = []
    seen: set[tuple[int, ...]] = set()
    for cpu in cpus:
        siblings_file = sys_cpu / f"cpu{cpu}" / "topology" / "thread_siblings_list"
        try:
            siblings = tuple(parse_cpulist(siblings_file.read_text()))
        except OSError:
            siblings = (cpu,)
        if siblings not in seen:
            seen.add(siblings)
            chosen.append(cpu)
    return chosen


def node_meminfo(node: int, sys_node: Path = SYS_NODE) -> dict[str, float]:
    info = {}
    try:
        text = (sys_node / f"node{node}" / "meminfo").read_text()
    except OSError:
        return info
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2] in ("MemTotal:", "MemFree:"):
            info[parts[2][:-1]] = int(parts[3]) / 2**20  # kB -> GiB
    return info


def cpu_summary() -> dict:
    model = ""
    flags: set[str] = set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and not model:
                model = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and not flags:
                flags = set(line.split(":", 1)[1].split())
    except OSError:
        pass
    wanted = sorted(
        f
        for f in flags
        if f.startswith("avx512") or f in {"avx2", "fma", "f16c", "avx_vnni"}
    )
    memlock = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    thp = Path("/sys/kernel/mm/transparent_hugepage/enabled")
    return {
        "model": model,
        "simd": wanted,
        "memlock_soft_hard": [
            "unlimited" if v == resource.RLIM_INFINITY else v for v in memlock
        ],
        "transparent_hugepage": thp.read_text().strip() if thp.exists() else None,
        "kernel": platform.release(),
    }


def gpu_numa_nodes() -> list[dict]:
    """GPU index, PCI address and NUMA node from ``nvidia-smi`` and ``/sys``."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id,name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        index, bus_id, name = (v.strip() for v in line.split(",", 2))
        domain, rest = bus_id.lower().split(":", 1)
        address = f"{int(domain, 16):04x}:{rest}"
        try:
            node = int(Path(f"/sys/bus/pci/devices/{address}/numa_node").read_text())
        except OSError:
            node = -1
        gpus.append(
            {"index": int(index), "pci": address, "name": name, "numa": max(node, 0)}
        )
    return gpus


# --------------------------------------------------------------------------
# DRAM bandwidth (one process per physical core)
# --------------------------------------------------------------------------


def _dram_worker(cpu, mib, iters, barrier, results, slot):
    os.sched_setaffinity(0, {cpu})
    n = mib * 2**20 // 8
    a = np.ones(n, dtype=np.float64)  # first touch on the local node
    b = np.empty_like(a)
    b.fill(0.0)
    a.sum()
    np.copyto(b, a)
    barrier.wait()
    start = time.perf_counter()
    for _ in range(iters):
        a.sum()
    read_s = time.perf_counter() - start
    barrier.wait()
    start = time.perf_counter()
    for _ in range(iters):
        np.copyto(b, a)
    copy_s = time.perf_counter() - start
    results[slot] = (read_s, copy_s)


def measure_dram(cpus: list[int], mib: int, iters: int) -> dict[str, float]:
    """Aggregate read and STREAM-copy bandwidth of processes pinned to ``cpus``."""
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(len(cpus))
    manager = ctx.Manager()
    results = manager.list([None] * len(cpus))
    procs = [
        ctx.Process(target=_dram_worker, args=(cpu, mib, iters, barrier, results, i))
        for i, cpu in enumerate(cpus)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    timings = list(results)
    manager.shutdown()
    if any(t is None for t in timings):
        raise RuntimeError("A DRAM worker failed; reduce --dram-mib.")
    total = len(cpus) * mib * 2**20 * iters
    read_s = max(t[0] for t in timings)
    copy_s = max(t[1] for t in timings)
    return {
        "processes": len(cpus),
        "read_gbs": total / read_s / 1e9,
        "copy_gbs": 2 * total / copy_s / 1e9,  # STREAM convention: read + write
    }


def dram_thread_counts(num_cores: int, per_rank: int) -> list[int]:
    return sorted({1, min(per_rank, num_cores), num_cores})


# --------------------------------------------------------------------------
# Host <-> device bandwidth
# --------------------------------------------------------------------------


def pcie_scenarios(num_gpus: int, gpu_nodes: dict[int, int]) -> list[list[int]]:
    """Singles, adjacent and non-adjacent pairs, each NUMA node, then all GPUs."""
    scenarios: list[list[int]] = [[i] for i in range(num_gpus)]
    scenarios += [[i, i + 1] for i in range(0, num_gpus - 1, 2)]
    scenarios += [[i, i + 2] for i in range(0, num_gpus - 2, 4)]
    by_node: dict[int, list[int]] = {}
    for gpu in range(num_gpus):
        by_node.setdefault(gpu_nodes.get(gpu, 0), []).append(gpu)
    scenarios += [g for g in by_node.values() if 1 < len(g) < num_gpus]
    if num_gpus > 1:
        scenarios.append(list(range(num_gpus)))
    unique: list[list[int]] = []
    for s in scenarios:
        if s not in unique:
            unique.append(s)
    return unique


def measure_pcie(
    gpu_nodes: dict[int, int],
    nodes: dict[int, list[int]],
    mib: int,
    iters: int,
) -> list[dict]:
    import torch

    num_gpus = torch.cuda.device_count()
    host, device, streams = {}, {}, {}

    def setup(gpu):
        os.sched_setaffinity(0, set(nodes.get(gpu_nodes.get(gpu, 0), [0])))
        torch.cuda.set_device(gpu)
        host[gpu] = torch.empty(mib * 2**20, dtype=torch.uint8, pin_memory=True)
        device[gpu] = torch.empty_like(host[gpu], device=f"cuda:{gpu}")
        streams[gpu] = torch.cuda.Stream(device=gpu)
        torch.cuda.synchronize(gpu)

    threads = [threading.Thread(target=setup, args=(g,)) for g in range(num_gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    def run(gpus, direction):
        barrier = threading.Barrier(len(gpus))
        elapsed = {}

        def copy(gpu):
            os.sched_setaffinity(0, set(nodes.get(gpu_nodes.get(gpu, 0), [0])))
            torch.cuda.set_device(gpu)
            src, dst = (
                (host[gpu], device[gpu])
                if direction == "h2d"
                else (device[gpu], host[gpu])
            )
            with torch.cuda.stream(streams[gpu]):
                dst.copy_(src, non_blocking=True)
                streams[gpu].synchronize()
                barrier.wait()
                start = time.perf_counter()
                for _ in range(iters):
                    dst.copy_(src, non_blocking=True)
                streams[gpu].synchronize()
                elapsed[gpu] = time.perf_counter() - start

        workers = [threading.Thread(target=copy, args=(g,)) for g in gpus]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        nbytes = mib * 2**20 * iters
        per_gpu = {g: nbytes / elapsed[g] / 1e9 for g in gpus}
        return {
            "gpus": gpus,
            "direction": direction,
            "per_gpu_gbs": per_gpu,
            "min_gbs": min(per_gpu.values()),
            "aggregate_gbs": len(gpus) * nbytes / max(elapsed.values()) / 1e9,
        }

    results = []
    for scenario in pcie_scenarios(num_gpus, gpu_nodes):
        results.append(run(scenario, "h2d"))
        if len(scenario) == 1 or len(scenario) == num_gpus:
            results.append(run(scenario, "d2h"))
    return results


def measure_pinned_capacity(
    nodes: dict[int, list[int]], gib: int, chunk_gib: int = 4
) -> dict[str, dict]:
    """Try to pin ``gib`` GiB per NUMA node in chunks; report what succeeded.

    Expert tiering needs about 80 GiB of pinned memory per node for
    GLM-5.3-Flash, and a non-root account cannot raise ``ulimit -l``.
    """
    import torch

    results: dict[str, dict] = {}
    for node, cpus in nodes.items():
        outcome: dict = {}

        def pin(node_cpus=cpus, outcome=outcome):
            os.sched_setaffinity(0, set(node_cpus))
            held = []
            start = time.perf_counter()
            try:
                while len(held) * chunk_gib < gib:
                    held.append(
                        torch.empty(
                            chunk_gib * 2**30, dtype=torch.uint8, pin_memory=True
                        )
                    )
            except RuntimeError as exc:
                outcome["error"] = str(exc).splitlines()[0]
            outcome["pinned_gib"] = len(held) * chunk_gib
            outcome["seconds"] = time.perf_counter() - start
            held.clear()

        thread = threading.Thread(target=pin)
        thread.start()
        thread.join()
        results[str(node)] = outcome
    return results


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def recommend(report: dict) -> dict:
    """Pick the analyze inputs: weakest node, all-core read; weakest GPU, all-8."""
    rec = {}
    full = [runs[-1]["read_gbs"] for runs in report.get("dram", {}).values() if runs]
    if full:
        rec["cpu_bw_gbs"] = round(min(full), 1)
    pcie = report.get("pcie", [])
    if pcie:
        widest = max(len(r["gpus"]) for r in pcie)
        h2d = [r for r in pcie if r["direction"] == "h2d" and len(r["gpus"]) == widest]
        rec["pcie_gbs_per_gpu"] = round(min(r["min_gbs"] for r in h2d), 1)
    return rec


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dram-mib", type=int, default=128, help="per array/process")
    parser.add_argument("--dram-iters", type=int, default=10)
    parser.add_argument("--cores-per-rank", type=int, default=4)
    parser.add_argument("--pcie-mib", type=int, default=512)
    parser.add_argument("--pcie-iters", type=int, default=10)
    parser.add_argument(
        "--pin-test-gib",
        type=int,
        default=0,
        help="Try pinning this many GiB per NUMA node (0: skip); e.g. 96",
    )
    parser.add_argument("--skip-dram", action="store_true")
    parser.add_argument("--skip-pcie", action="store_true")
    parser.add_argument("--json-out", default="sm70_host_topology.json")
    args = parser.parse_args(argv)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # match nvidia-smi
    nodes = numa_nodes()
    gpus = gpu_numa_nodes()
    report: dict = {
        "cpu": cpu_summary(),
        "nodes": {
            str(n): {
                "cpus": len(c),
                "physical_cores": len(physical_cores(c)),
                **node_meminfo(n),
            }
            for n, c in nodes.items()
        },
        "gpus": gpus,
    }
    print(json.dumps({k: report[k] for k in ("cpu", "nodes", "gpus")}, indent=2))

    if not args.skip_dram:
        report["dram"] = {}
        for node, cpus in nodes.items():
            cores = physical_cores(cpus)
            runs = []
            for count in dram_thread_counts(len(cores), args.cores_per_rank):
                run = measure_dram(cores[:count], args.dram_mib, args.dram_iters)
                runs.append(run)
                print(
                    f"node{node} {count:>2} cores: read {run['read_gbs']:6.1f} GB/s"
                    f"  copy {run['copy_gbs']:6.1f} GB/s"
                )
            report["dram"][str(node)] = runs

    if not args.skip_pcie and gpus:
        gpu_nodes = {g["index"]: g["numa"] for g in gpus}
        report["pcie"] = measure_pcie(gpu_nodes, nodes, args.pcie_mib, args.pcie_iters)
        for r in report["pcie"]:
            per = " ".join(f"{g}:{v:.1f}" for g, v in r["per_gpu_gbs"].items())
            print(
                f"{r['direction']} gpus={r['gpus']}  aggregate"
                f" {r['aggregate_gbs']:6.1f} GB/s  per-GPU {per}"
            )

    if args.pin_test_gib > 0:
        report["pinned_capacity"] = measure_pinned_capacity(nodes, args.pin_test_gib)
        for node, outcome in report["pinned_capacity"].items():
            print(
                f"node{node} pinned {outcome['pinned_gib']} GiB in"
                f" {outcome['seconds']:.1f} s {outcome.get('error', '')}"
            )

    report["recommended_analyze_args"] = recommend(report)
    print("recommended:", report["recommended_analyze_args"])
    Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
