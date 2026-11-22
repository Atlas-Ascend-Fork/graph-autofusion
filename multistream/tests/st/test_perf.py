# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Performance and recursion-budget checks (test plan §4.11 / §10).

The perf-marked tests use a short warm-up followed by a few measured runs and
write JSON suitable for CI artifacts.  Thresholds are intentionally advisory
unless ``MULTISTREAM_PERF_STRICT=1`` is set, so shared builders are not made flaky
by machine load.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time
import warnings

import pytest

import multistream


WARMUP_RUNS = int(os.environ.get("MULTISTREAM_PERF_WARMUP", "3"))
MEASURED_RUNS = int(os.environ.get("MULTISTREAM_PERF_RUNS", "10"))
REPORT_PATH = Path(os.environ.get("MULTISTREAM_PERF_JSON", "multistream_perf.json"))
STRICT = os.environ.get("MULTISTREAM_PERF_STRICT", "0") == "1"
RECORDS = []


def _percentile(samples, fraction):
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _measure(callable_under_test):
    for _ in range(max(0, WARMUP_RUNS)):
        callable_under_test()
    samples = []
    for _ in range(max(1, MEASURED_RUNS)):
        started = time.perf_counter()
        callable_under_test()
        samples.append(time.perf_counter() - started)
    return {
        "runs": len(samples),
        "median": statistics.median(samples),
        "p95": _percentile(samples, 0.95),
        "max": max(samples),
        "min": min(samples),
    }


def _report(name, stats, budget=None):
    record = {"case": name, **stats}
    if budget:
        record["budget"] = budget
    RECORDS.append(record)
    # Persist after every case as well as at session end.  Pytest only treats
    # hooks in plugin/conftest modules as hooks; keeping the write here makes
    # the artifact reliable when this standalone test module is invoked.
    _write_report()
    line = f"[perf] {name}: median={stats['median'] * 1000:.2f}ms p95={stats['p95'] * 1000:.2f}ms max={stats['max'] * 1000:.2f}ms"
    if budget:
        line += f" (预算 median<{budget['median'] * 1000:.0f}ms p95<{budget['p95'] * 1000:.0f}ms)"
    print(line)
    if not budget:
        return
    breaches = [key for key in ("median", "p95") if stats[key] >= budget[key]]
    if not breaches:
        return
    message = f"{name} 超出性能预算: " + "; ".join(breaches)
    if STRICT:
        pytest.fail(message)
    warnings.warn("[告警，未冻结门禁] " + message, RuntimeWarning, stacklevel=2)


def _write_report():
    if not RECORDS:
        return
    payload = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "records": RECORDS,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pytest_sessionfinish(session, exitstatus):  # pragma: no cover - pytest hook
    _write_report()


def _large_graph(n=100):
    graph = multistream.DAGGraph(f"perf_{n}")
    nodes = [graph.add_node(f"n{i}", "Op") for i in range(n)]
    # Fixed fan-in keeps construction and allocation deterministic and acyclic.
    for index in range(1, n):
        graph.add_edge(nodes[index - 1], 0, nodes[index], 0)
        if index > 1 and index % 3 == 0:
            graph.add_edge(nodes[index - 2], 1, nodes[index], 1)
    return graph


@pytest.mark.perf
@pytest.mark.parametrize("node_count", [100, 500])
def test_plan_stays_inside_its_time_budget(node_count):
    graph = _large_graph(node_count)
    for strategy in multistream.MergeStrategy:
        stats = _measure(
            lambda: multistream.plan(graph, max_stream_num=8, merge_strategy=strategy)
        )
        _report(
            f"plan/{node_count}/{strategy.name.lower()}",
            stats,
            {"median": 0.5, "p95": 1.0},
        )


@pytest.mark.perf
def test_building_and_reading_back_a_thousand_node_plan():
    graph = _large_graph(1000)

    def build_and_read():
        plan = multistream.plan(graph, max_stream_num=8)
        assert all(plan.get_stream_id(node) >= 0 for node in graph.get_nodes())

    stats = _measure(build_and_read)
    _report("build_and_readback/1000", stats, {"median": 1.0, "p95": 2.0})


@pytest.mark.perf
def test_a_deep_chain_does_not_hit_the_recursion_limit(deep_chain_graph):
    graph = deep_chain_graph(1000)
    depth_probe = {"max": 0}
    original = sys.getrecursionlimit()

    def tracer(frame, event, arg):
        if event == "call":
            depth_probe["max"] = max(depth_probe["max"], traceback_stack(frame))

    def traceback_stack(frame):
        depth = 0
        while frame is not None:
            depth += 1
            frame = frame.f_back
        return depth

    sys.setprofile(tracer)
    try:
        plan = multistream.plan(graph, max_stream_num=8)
    finally:
        sys.setprofile(None)
    assert plan.get_required_stream_count() == 1
    assert all(plan.get_stream_id(node) == 0 for node in graph.get_nodes())
    assert original - depth_probe["max"] > 0
    RECORDS.append(
        {
            "case": "deep_chain/1000",
            "max_stack_depth": depth_probe["max"],
            "recursion_limit": original,
            "headroom": original - depth_probe["max"],
        }
    )
    _write_report()


def test_gate_finishes_in_seconds(large_graph):
    graph = large_graph(100)
    stats = _measure(lambda: multistream.plan(graph))
    _report("gate_sample/100", stats, {"median": 0.5, "p95": 1.0})
