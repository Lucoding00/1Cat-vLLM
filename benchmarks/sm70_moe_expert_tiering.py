# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-0 calibration for SM70 MoE expert tiering.

Expert tiering keeps the hottest routed experts resident in GPU memory, keeps
every expert in pinned host memory, lets the CPU compute small-batch misses,
and streams missing experts over PCIe for large prefill chunks. Whether this
pays off depends on how skewed the real routing is. This tool measures that
before any runtime code is written.

``collect`` sends a representative workload to a running server started with
``--enable-return-routed-experts`` and stores the per-token routing.
``analyze`` builds static GPU placements from a calibration split, evaluates
them on a held-out split, and applies a simple V100 cost model.

Run calibration with prefix caching and speculative decoding disabled. Cached
prefix tokens are not forwarded, and rejected draft tokens are not part of the
final sequence, so either would bias the measured routing.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

NVFP4_BYTES_PER_PARAM = 0.5 + 1.0 / 16.0  # E2M1 codes + one E4M3 scale per 16


# --------------------------------------------------------------------------
# Routing data helpers
# --------------------------------------------------------------------------


def decode_routes_b64(payload: str) -> np.ndarray:
    """Decode the server's base64 ``.npy`` payload to ``[tokens, layers, k]``."""
    routes = np.load(io.BytesIO(base64.b64decode(payload)), allow_pickle=False)
    if routes.ndim != 3:
        raise ValueError(f"Expected [tokens, layers, top_k], got {routes.shape}.")
    return routes


def parse_layer_spec(spec: str) -> list[int]:
    """Parse ``"3-44,46"`` into a sorted list of unique layer indices."""
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(v) for v in part.split("-", 1))
            if end < start:
                raise ValueError(f"Invalid layer range {part!r}.")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def detect_moe_layers(routes_list: list[np.ndarray]) -> list[int]:
    """Return layers whose captured routing actually varies.

    Dense layers, and layers the capturer never wrote (for example layers that
    live on another pipeline stage), stay zero or constant. A routed layer
    selects more than ``top_k`` distinct experts across the workload.
    """
    if not routes_list:
        return []
    num_layers = routes_list[0].shape[1]
    top_k = routes_list[0].shape[2]
    layers = []
    for layer in range(num_layers):
        seen: set[int] = set()
        for routes in routes_list:
            seen.update(np.unique(routes[:, layer, :]).tolist())
            if len(seen) > top_k:
                break
        if len(seen) > top_k:
            layers.append(layer)
    return layers


def split_requests(num_requests: int, holdout: float) -> tuple[list[int], list[int]]:
    """Deterministically split requests into calibration and evaluation sets."""
    if num_requests == 0:
        return [], []
    if num_requests == 1 or holdout <= 0.0:
        return list(range(num_requests)), list(range(num_requests))
    stride = max(2, round(1.0 / holdout))
    evaluation = [i for i in range(num_requests) if i % stride == stride - 1]
    if not evaluation:
        evaluation = [num_requests - 1]
    held_out = set(evaluation)
    calibration = [i for i in range(num_requests) if i not in held_out]
    return calibration, evaluation


def _phase_rows(routes: np.ndarray, prompt_len: int, phase: str) -> np.ndarray:
    # Row i is the forward pass of token i: rows [0, prompt_len) are prefill.
    if phase == "prefill":
        return routes[:prompt_len]
    if phase == "decode":
        return routes[prompt_len:]
    if phase == "all":
        return routes
    raise ValueError(f"Unknown phase {phase!r}.")


def expert_counts(
    routes_list: list[np.ndarray],
    prompt_lens: list[int],
    layers: list[int],
    num_experts: int,
    phase: str,
) -> np.ndarray:
    """Count expert selections per MoE layer: ``[len(layers), num_experts]``."""
    counts = np.zeros((len(layers), num_experts), dtype=np.int64)
    for routes, prompt_len in zip(routes_list, prompt_lens):
        rows = _phase_rows(routes, prompt_len, phase)
        for i, layer in enumerate(layers):
            ids = rows[:, layer, :].reshape(-1).astype(np.int64)
            counts[i] += np.bincount(ids, minlength=num_experts)[:num_experts]
    return counts


# --------------------------------------------------------------------------
# Static placement
# --------------------------------------------------------------------------


def _top_mask(scores: np.ndarray, count: int) -> np.ndarray:
    mask = np.zeros(scores.shape, dtype=bool)
    if count <= 0:
        return mask
    # Stable ordering: higher score first, then lower expert ID.
    order = np.lexsort((np.arange(scores.size), -scores.reshape(-1)))
    mask.reshape(-1)[order[:count]] = True
    return mask


def plan_uniform(counts: np.ndarray, fraction: float) -> np.ndarray:
    """Cache the same number of hottest experts in every layer."""
    num_layers, num_experts = counts.shape
    per_layer = min(num_experts, max(0, round(fraction * num_experts)))
    mask = np.zeros(counts.shape, dtype=bool)
    for layer in range(num_layers):
        mask[layer] = _top_mask(counts[layer], per_layer)
    return mask


def plan_global(counts: np.ndarray, fraction: float) -> np.ndarray:
    """Spend one global budget on the most frequently selected (layer, expert).

    Every layer sees the same number of selections, so raw counts are directly
    comparable; skewed layers receive fewer slots than flat layers.
    """
    total = counts.size
    budget = min(total, max(0, round(fraction * total)))
    return _top_mask(counts.astype(np.float64), budget)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeStats:
    tokens: int
    hit_rate: float
    misses_per_layer: list[float]  # mean CPU experts per token, per MoE layer
    mean_misses_per_token_layer: float
    p99_misses_per_token_layer: float


def decode_hit_stats(
    routes_list: list[np.ndarray],
    prompt_lens: list[int],
    layers: list[int],
    mask: np.ndarray,
) -> DecodeStats:
    miss_sums = np.zeros(len(layers), dtype=np.float64)
    miss_samples: list[np.ndarray] = []
    tokens = 0
    selections = 0
    hits = 0
    for routes, prompt_len in zip(routes_list, prompt_lens):
        rows = _phase_rows(routes, prompt_len, "decode")
        if rows.shape[0] == 0:
            continue
        tokens += rows.shape[0]
        for i, layer in enumerate(layers):
            ids = rows[:, layer, :].astype(np.int64)
            cached = mask[i][ids]
            misses = (~cached).sum(axis=1)
            miss_sums[i] += misses.sum()
            miss_samples.append(misses)
            hits += int(cached.sum())
            selections += cached.size
    if tokens == 0:
        return DecodeStats(0, float("nan"), [], float("nan"), float("nan"))
    samples = np.concatenate(miss_samples)
    return DecodeStats(
        tokens=tokens,
        hit_rate=hits / selections,
        misses_per_layer=(miss_sums / tokens).tolist(),
        mean_misses_per_token_layer=float(samples.mean()),
        p99_misses_per_token_layer=float(np.percentile(samples, 99)),
    )


def prefill_uncached_touched(
    routes_list: list[np.ndarray],
    prompt_lens: list[int],
    layers: list[int],
    mask: np.ndarray,
    chunk_tokens: int,
) -> list[float] | None:
    """Mean number of non-resident experts one full prefill chunk touches.

    Returns one value per MoE layer, or ``None`` when no prompt contains a full
    chunk of the requested size.
    """
    sums = np.zeros(len(layers), dtype=np.float64)
    chunks = 0
    for routes, prompt_len in zip(routes_list, prompt_lens):
        prefill = _phase_rows(routes, prompt_len, "prefill")
        for start in range(0, prefill.shape[0] - chunk_tokens + 1, chunk_tokens):
            chunk = prefill[start : start + chunk_tokens]
            chunks += 1
            for i, layer in enumerate(layers):
                touched = np.unique(chunk[:, layer, :])
                sums[i] += int((~mask[i][touched.astype(np.int64)]).sum())
    if chunks == 0:
        return None
    return (sums / chunks).tolist()


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """First-order V100 tiering model; every input is an explicit assumption."""

    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    tp: int = 4
    pp: int = 2
    # Usable DRAM bandwidth of the socket serving the active stage. With TP4/PP2
    # each stage maps to one NUMA node; single-request stages run in turn.
    cpu_bw_gbs: float = 100.0
    pcie_gbs_per_gpu: float = 6.0  # H2D per GPU while its switch peer also copies
    handoff_us_per_layer: float = 40.0  # D2H route + CPU wake + H2D partial
    baseline_tpot_ms: float = 18.86  # measured GLM-5.3 TP4/PP2 no-MTP decode

    @property
    def shard_bytes(self) -> float:
        params = 3 * self.hidden_size * self.moe_intermediate_size / self.tp
        return params * NVFP4_BYTES_PER_PARAM

    def stage_layers(self, num_layers: int) -> list[list[int]]:
        return [
            [i for i in range(num_layers) if i * self.pp // num_layers == stage]
            for stage in range(self.pp)
        ]

    def vram_freed_per_gpu(self, mask: np.ndarray) -> list[float]:
        """Freed expert bytes on one GPU of each pipeline stage."""
        uncached = (~mask).sum(axis=1)
        return [
            float(uncached[stage].sum() * self.shard_bytes)
            for stage in self.stage_layers(mask.shape[0])
        ]

    def decode_extra_ms(self, misses_per_layer: list[float]) -> float:
        """Added per-token latency, assuming no GPU/CPU overlap (conservative).

        Each TP rank computes its own intermediate shard of every missed expert
        on CPU, so one stage reads ``tp * misses * shard`` bytes from DRAM.
        """
        misses = np.asarray(misses_per_layer, dtype=np.float64)
        total_s = 0.0
        for stage in self.stage_layers(misses.size):
            stage_bytes = self.tp * misses[stage].sum() * self.shard_bytes
            total_s += stage_bytes / (self.cpu_bw_gbs * 1e9)
            total_s += len(stage) * self.handoff_us_per_layer * 1e-6
        return total_s * 1e3

    def prefill_stream_s(self, touched_per_layer: list[float]) -> float:
        """H2D seconds per chunk for the slowest stage (one GPU's shard)."""
        touched = np.asarray(touched_per_layer, dtype=np.float64)
        return max(
            touched[stage].sum() * self.shard_bytes / (self.pcie_gbs_per_gpu * 1e9)
            for stage in self.stage_layers(touched.size)
        )


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _post_json(url: str, body: dict, api_key: str | None, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _collect_one(args: argparse.Namespace, record: dict) -> tuple[np.ndarray, int]:
    base = args.base_url.rstrip("/")
    common = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    if "messages" in record:
        url, body = "/v1/chat/completions", {**common, "messages": record["messages"]}
    elif "prompt" in record:
        url, body = "/v1/completions", {**common, "prompt": record["prompt"]}
    else:
        raise ValueError("Each JSONL record needs 'messages' or 'prompt'.")
    data = _post_json(base + url, body, args.api_key, args.timeout)
    payload = data["choices"][0].get("routed_experts")
    if payload is None:
        raise RuntimeError(
            "The server returned no routed_experts; start it with "
            "--enable-return-routed-experts."
        )
    return decode_routes_b64(payload), int(data["usage"]["prompt_tokens"])


def cmd_collect(args: argparse.Namespace) -> None:
    records = [
        json.loads(line)
        for line in Path(args.prompts).read_text().splitlines()
        if line.strip()
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda r: _collect_one(args, r), records))
    arrays = {f"routes_{i}": routes for i, (routes, _) in enumerate(results)}
    np.savez_compressed(
        args.output,
        prompt_lens=np.asarray([p for _, p in results], dtype=np.int64),
        **arrays,
    )
    tokens = sum(r.shape[0] for r, _ in results)
    print(f"Saved {len(results)} requests / {tokens} routed tokens to {args.output}")


def load_routes(paths: list[str]) -> tuple[list[np.ndarray], list[int]]:
    routes_list: list[np.ndarray] = []
    prompt_lens: list[int] = []
    for path in paths:
        with np.load(path) as data:
            lens = data["prompt_lens"].tolist()
            for i, prompt_len in enumerate(lens):
                routes_list.append(data[f"routes_{i}"])
                prompt_lens.append(int(prompt_len))
    return routes_list, prompt_lens


def analyze(
    routes_list: list[np.ndarray],
    prompt_lens: list[int],
    *,
    layers: list[int],
    num_experts: int,
    fractions: list[float],
    chunk_sizes: list[int],
    holdout: float,
    rank_by: str,
    model: CostModel,
) -> dict:
    calibration, evaluation = split_requests(len(routes_list), holdout)
    calib_routes = [routes_list[i] for i in calibration]
    calib_lens = [prompt_lens[i] for i in calibration]
    eval_routes = [routes_list[i] for i in evaluation]
    eval_lens = [prompt_lens[i] for i in evaluation]
    counts = expert_counts(calib_routes, calib_lens, layers, num_experts, rank_by)
    full_bytes = len(layers) * num_experts * model.shard_bytes / model.pp
    rows = []
    for strategy, planner in (("uniform", plan_uniform), ("global", plan_global)):
        for fraction in fractions:
            mask = planner(counts, fraction)
            decode = decode_hit_stats(eval_routes, eval_lens, layers, mask)
            extra_ms = (
                model.decode_extra_ms(decode.misses_per_layer)
                if decode.tokens
                else float("nan")
            )
            prefill = {}
            for chunk in chunk_sizes:
                touched = prefill_uncached_touched(
                    eval_routes, eval_lens, layers, mask, chunk
                )
                if touched is None:
                    prefill[str(chunk)] = None
                    continue
                stream_s = model.prefill_stream_s(touched)
                prefill[str(chunk)] = {
                    "stream_s_per_chunk": stream_s,
                    "stream_bound_tok_s": chunk / stream_s if stream_s else math.inf,
                }
            freed = model.vram_freed_per_gpu(mask)
            rows.append(
                {
                    "strategy": strategy,
                    "fraction": fraction,
                    "cached_fraction": float(mask.mean()),
                    "vram_freed_gib_per_gpu_min": min(freed) / 2**30,
                    "decode_hit_rate": decode.hit_rate,
                    "mean_misses_per_token_layer": decode.mean_misses_per_token_layer,
                    "p99_misses_per_token_layer": decode.p99_misses_per_token_layer,
                    "decode_extra_ms": extra_ms,
                    "est_decode_tok_s": 1e3 / (model.baseline_tpot_ms + extra_ms),
                    "prefill": prefill,
                }
            )
    return {
        "requests": {"calibration": len(calibration), "evaluation": len(evaluation)},
        "moe_layers": layers,
        "num_experts": num_experts,
        "expert_shard_mib": model.shard_bytes / 2**20,
        "resident_expert_gib_per_gpu": full_bytes / 2**30,
        "cost_model": asdict(model),
        "rows": rows,
    }


def _print_report(report: dict) -> None:
    print(
        f"requests calib/eval: {report['requests']['calibration']}/"
        f"{report['requests']['evaluation']}  MoE layers: {len(report['moe_layers'])}"
        f"  expert shard: {report['expert_shard_mib']:.2f} MiB"
        f"  all experts per GPU: {report['resident_expert_gib_per_gpu']:.1f} GiB"
    )
    chunks = sorted({c for row in report["rows"] for c in row["prefill"]}, key=int)
    header = (
        f"{'plan':<8}{'cache%':>7}{'freed GiB/GPU':>14}{'hit%':>7}"
        f"{'miss/tok/L':>11}{'+ms/tok':>9}{'decode tok/s':>13}"
        + "".join(f"{'pf' + c + ' s/tok/s':>18}" for c in chunks)
    )
    print(header)
    for row in report["rows"]:
        cells = [
            f"{row['strategy']:<8}",
            f"{100 * row['cached_fraction']:>7.1f}",
            f"{row['vram_freed_gib_per_gpu_min']:>14.2f}",
            f"{100 * row['decode_hit_rate']:>7.1f}",
            f"{row['mean_misses_per_token_layer']:>11.2f}",
            f"{row['decode_extra_ms']:>9.2f}",
            f"{row['est_decode_tok_s']:>13.1f}",
        ]
        for chunk in chunks:
            item = row["prefill"].get(chunk)
            cells.append(
                f"{'-':>18}"
                if item is None
                else f"{item['stream_s_per_chunk']:>11.2f}/"
                f"{item['stream_bound_tok_s']:<6.0f}"
            )
        print("".join(cells))


def cmd_analyze(args: argparse.Namespace) -> None:
    routes_list, prompt_lens = load_routes(args.inputs)
    layers = (
        parse_layer_spec(args.moe_layers)
        if args.moe_layers
        else detect_moe_layers(routes_list)
    )
    if not layers:
        sys.exit("No routed layers found; check the capture and --moe-layers.")
    num_experts = args.num_experts or int(
        max(int(r[:, layers, :].max()) for r in routes_list) + 1
    )
    model = CostModel(
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        tp=args.tp,
        pp=args.pp,
        cpu_bw_gbs=args.cpu_bw_gbs,
        pcie_gbs_per_gpu=args.pcie_gbs_per_gpu,
        handoff_us_per_layer=args.handoff_us,
        baseline_tpot_ms=args.baseline_tpot_ms,
    )
    report = analyze(
        routes_list,
        prompt_lens,
        layers=layers,
        num_experts=num_experts,
        fractions=[float(v) for v in args.fractions.split(",")],
        chunk_sizes=[int(v) for v in args.chunk_tokens.split(",")],
        holdout=args.holdout,
        rank_by=args.rank_by,
        model=model,
    )
    _print_report(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Record routing from a running server.")
    collect.add_argument("--base-url", default="http://127.0.0.1:8000")
    collect.add_argument("--model", required=True)
    collect.add_argument("--prompts", required=True, help="JSONL: messages or prompt")
    collect.add_argument("--output", required=True, help="Output .npz path")
    collect.add_argument("--max-tokens", type=int, default=512)
    collect.add_argument("--temperature", type=float, default=0.6)
    collect.add_argument("--concurrency", type=int, default=1)
    collect.add_argument("--api-key")
    collect.add_argument("--timeout", type=float, default=1800.0)
    collect.set_defaults(func=cmd_collect)

    ana = sub.add_parser("analyze", help="Evaluate static GPU expert placements.")
    ana.add_argument("inputs", nargs="+", help=".npz files written by collect")
    ana.add_argument("--moe-layers", help="e.g. 3-44 for GLM-5.3; default: detect")
    ana.add_argument("--num-experts", type=int, default=0, help="default: detect")
    ana.add_argument("--fractions", default="0.3,0.4,0.5,0.6,0.7,0.8")
    ana.add_argument("--chunk-tokens", default="2048,8192,16384")
    ana.add_argument("--holdout", type=float, default=0.25)
    ana.add_argument("--rank-by", choices=("decode", "prefill", "all"), default="all")
    ana.add_argument("--hidden-size", type=int, default=4096)
    ana.add_argument("--moe-intermediate-size", type=int, default=2048)
    ana.add_argument("--tp", type=int, default=4)
    ana.add_argument("--pp", type=int, default=2)
    ana.add_argument("--cpu-bw-gbs", type=float, default=100.0)
    ana.add_argument("--pcie-gbs-per-gpu", type=float, default=6.0)
    ana.add_argument("--handoff-us", type=float, default=40.0)
    ana.add_argument("--baseline-tpot-ms", type=float, default=18.86)
    ana.add_argument("--json-out")
    ana.set_defaults(func=cmd_analyze)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
