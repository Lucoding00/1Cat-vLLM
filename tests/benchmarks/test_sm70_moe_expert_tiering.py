# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import io

import numpy as np
import pytest

from benchmarks.sm70_moe_expert_tiering import (
    CostModel,
    analyze,
    decode_hit_stats,
    decode_routes_b64,
    detect_moe_layers,
    expert_counts,
    load_routes,
    parse_layer_spec,
    plan_global,
    plan_uniform,
    prefill_uncached_touched,
    split_requests,
)


def _zipf_routes(
    rng, tokens, num_layers, moe_layers, num_experts, top_k, skew, seed=123
):
    """Synthetic routing: dense layers stay zero, MoE layers follow a Zipf law.

    Each layer has its own hot set, fixed by ``seed`` so that every request of
    one workload shares it.
    """
    routes = np.zeros((tokens, num_layers, top_k), dtype=np.uint16)
    probs = 1.0 / np.arange(1, num_experts + 1) ** skew
    probs /= probs.sum()
    perm_rng = np.random.default_rng(seed)
    for layer in moe_layers:
        perm = perm_rng.permutation(num_experts)
        for t in range(tokens):
            routes[t, layer] = perm[
                rng.choice(num_experts, size=top_k, replace=False, p=probs)
            ]
    return routes


def test_decode_routes_round_trip():
    routes = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    buf = io.BytesIO()
    np.save(buf, routes)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")

    np.testing.assert_array_equal(decode_routes_b64(payload), routes)


def test_parse_layer_spec():
    assert parse_layer_spec("3-5, 7,5") == [3, 4, 5, 7]
    with pytest.raises(ValueError):
        parse_layer_spec("5-3")


def test_detect_moe_layers_skips_dense_and_uncaptured_layers():
    rng = np.random.default_rng(0)
    routes = _zipf_routes(rng, 64, 6, [2, 3, 5], 32, 4, 1.0)
    routes[:, 4] = np.array([1, 2, 3, 4])  # constant: never written / dense

    assert detect_moe_layers([routes]) == [2, 3, 5]


def test_split_requests_is_deterministic_and_disjoint():
    calibration, evaluation = split_requests(8, 0.25)

    assert evaluation == [3, 7]
    assert calibration == [0, 1, 2, 4, 5, 6]
    assert split_requests(1, 0.25) == ([0], [0])


def test_expert_counts_separates_prefill_and_decode():
    routes = np.array([[[0, 1]], [[0, 2]], [[3, 1]]], dtype=np.uint16)

    prefill = expert_counts([routes], [2], [0], 4, "prefill")
    decode = expert_counts([routes], [2], [0], 4, "decode")

    np.testing.assert_array_equal(prefill, [[2, 1, 1, 0]])
    np.testing.assert_array_equal(decode, [[0, 1, 0, 1]])


def test_plans_respect_budget_and_pick_hot_experts():
    counts = np.array([[10, 0, 5, 1], [3, 3, 3, 3]])

    uniform = plan_uniform(counts, 0.5)
    global_ = plan_global(counts, 0.5)

    np.testing.assert_array_equal(uniform[0], [True, False, True, False])
    assert uniform.sum(axis=1).tolist() == [2, 2]
    # The flat layer competes for a single global budget of four slots.
    assert global_.sum() == 4
    assert global_[0, 0] and global_[0, 2]


def test_decode_hit_stats_counts_misses_per_layer():
    routes = np.array([[[0, 1]], [[0, 2]], [[3, 1]]], dtype=np.uint16)
    mask = np.array([[True, True, False, False]])

    stats = decode_hit_stats([routes], [1], [0], mask)

    assert stats.tokens == 2
    assert stats.hit_rate == pytest.approx(0.5)
    assert stats.misses_per_layer == pytest.approx([1.0])


def test_prefill_uncached_touched_counts_unique_experts_per_chunk():
    routes = np.array(
        [[[0, 2]], [[2, 3]], [[1, 0]], [[3, 3]], [[9, 9]]], dtype=np.uint16
    )
    mask = np.zeros((1, 10), dtype=bool)
    mask[0, 0] = True

    touched = prefill_uncached_touched([routes], [4], [0], mask, 2)

    # Chunk [0,2,3] -> {2,3}; chunk [1,0,3] -> {1,3}; decode row ignored.
    assert touched == pytest.approx([2.0])
    assert prefill_uncached_touched([routes], [4], [0], mask, 8) is None


def test_cost_model_matches_glm_tp4_arithmetic():
    model = CostModel(tp=4, pp=2, cpu_bw_gbs=100.0, handoff_us_per_layer=0.0)

    # 3 * 4096 * 2048 / 4 params at 4.5 bits.
    assert model.shard_bytes == pytest.approx(3_538_944.0)
    freed = model.vram_freed_per_gpu(np.zeros((42, 288), dtype=bool))
    assert freed[0] == pytest.approx(21 * 288 * model.shard_bytes)
    # One miss per layer: 4 ranks * 42 layers * shard bytes at 100 GB/s.
    assert model.decode_extra_ms([1.0] * 42) == pytest.approx(
        4 * 42 * model.shard_bytes / 100e9 * 1e3
    )


def test_analyze_skewed_routing_beats_flat_routing(tmp_path):
    rng = np.random.default_rng(1)
    kwargs = dict(
        tokens=96, num_layers=4, moe_layers=[1, 2, 3], num_experts=64, top_k=4
    )
    skewed = [_zipf_routes(rng, skew=1.2, **kwargs) for _ in range(8)]
    flat = [_zipf_routes(rng, skew=0.0, **kwargs) for _ in range(8)]
    lens = [64] * 8
    np.savez(
        tmp_path / "skewed.npz",
        prompt_lens=np.asarray(lens),
        **{f"routes_{i}": r for i, r in enumerate(skewed)},
    )
    loaded, loaded_lens = load_routes([str(tmp_path / "skewed.npz")])

    common = dict(
        layers=[1, 2, 3],
        num_experts=64,
        fractions=[0.25],
        chunk_sizes=[32],
        holdout=0.25,
        rank_by="all",
        model=CostModel(pp=1),
    )
    skewed_report = analyze(loaded, loaded_lens, **common)
    flat_report = analyze(flat, lens, **common)

    skewed_hit = skewed_report["rows"][0]["decode_hit_rate"]
    flat_hit = flat_report["rows"][0]["decode_hit_rate"]
    assert skewed_hit > flat_hit + 0.2
    assert skewed_report["rows"][0]["prefill"]["32"]["stream_s_per_chunk"] > 0
