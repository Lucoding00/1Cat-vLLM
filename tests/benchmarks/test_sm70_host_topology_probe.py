# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from benchmarks.sm70_host_topology_probe import (
    dram_thread_counts,
    measure_dram,
    numa_nodes,
    parse_cpulist,
    pcie_scenarios,
    physical_cores,
    recommend,
)


def test_parse_cpulist_matches_kernel_format():
    assert parse_cpulist("0-2,36-37\n") == [0, 1, 2, 36, 37]
    assert parse_cpulist("5") == [5]


def test_numa_nodes_and_physical_cores_from_sysfs(tmp_path):
    node_root = tmp_path / "node"
    cpu_root = tmp_path / "cpu"
    for node, cpulist in ((0, "0-1,4-5"), (1, "2-3,6-7")):
        (node_root / f"node{node}").mkdir(parents=True)
        (node_root / f"node{node}" / "cpulist").write_text(cpulist)
    for cpu in range(8):
        topo = cpu_root / f"cpu{cpu}" / "topology"
        topo.mkdir(parents=True)
        core = cpu % 4
        topo.joinpath("thread_siblings_list").write_text(f"{core},{core + 4}")

    nodes = numa_nodes(node_root)

    assert nodes == {0: [0, 1, 4, 5], 1: [2, 3, 6, 7]}
    assert physical_cores(nodes[0], cpu_root) == [0, 1]
    assert physical_cores(nodes[1], cpu_root) == [2, 3]


def test_pcie_scenarios_for_two_socket_eight_gpu_box():
    gpu_nodes = {g: 0 if g < 4 else 1 for g in range(8)}

    scenarios = pcie_scenarios(8, gpu_nodes)

    assert scenarios[:8] == [[g] for g in range(8)]
    assert [0, 1] in scenarios and [6, 7] in scenarios  # likely switch peers
    assert [0, 2] in scenarios and [4, 6] in scenarios  # different switches
    assert [0, 1, 2, 3] in scenarios and [4, 5, 6, 7] in scenarios
    assert scenarios[-1] == list(range(8))


def test_dram_thread_counts():
    assert dram_thread_counts(18, 4) == [1, 4, 18]
    assert dram_thread_counts(2, 4) == [1, 2]


def test_measure_dram_runs_pinned_workers():
    cpus = sorted(os.sched_getaffinity(0))[:2]

    result = measure_dram(cpus, mib=8, iters=2)

    assert result["processes"] == len(cpus)
    assert result["read_gbs"] > 0 and result["copy_gbs"] > 0


def test_recommend_uses_weakest_node_and_all_gpu_h2d():
    report = {
        "dram": {
            "0": [{"read_gbs": 10.0}, {"read_gbs": 98.0}],
            "1": [{"read_gbs": 11.0}, {"read_gbs": 92.5}],
        },
        "pcie": [
            {"gpus": [0], "direction": "h2d", "min_gbs": 11.8},
            {"gpus": list(range(8)), "direction": "h2d", "min_gbs": 5.7},
            {"gpus": list(range(8)), "direction": "d2h", "min_gbs": 4.0},
        ],
    }

    assert recommend(report) == {"cpu_bw_gbs": 92.5, "pcie_gbs_per_gpu": 5.7}
