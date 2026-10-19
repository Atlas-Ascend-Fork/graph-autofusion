"""Installed multistream end-to-end planning scenarios (MS-ST-02 through 13)."""

import csv
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import random
import subprocess
import sys
import time
import tracemalloc

import pytest

import multistream as ms

pytestmark = [pytest.mark.st, pytest.mark.runtime]


def graph(name, names, edges=()):
    result = ms.DAGGraph(name)
    nodes = {name: result.add_node(name, "Op") for name in names}
    for src, dst in edges:
        result.add_edge(nodes[src], 0, nodes[dst], 0)
    return result, nodes


def mapping(plan, nodes):
    return tuple(plan.get_stream_id(node) for node in nodes.values())


def profile(path, rows, headers=None):
    headers = headers or (
        "Op Name",
        "Task Type",
        "Task Duration(us)",
        "Block Num",
        "Mix Block Num",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def route(caplog, implementation):
    assert any(
        "used implementation=%s," % implementation in record.getMessage()
        for record in caplog.records
    )


def check_regular(plan, nodes, base, limit):
    ids = mapping(plan, nodes)
    assert all(base <= stream_id < base + limit for stream_id in ids)
    assert len(set(ids)) == plan.get_required_stream_count()
    return ids


def test_ms_st_02_runtime_stub_boundary(stub_runtime, caplog, tmp_path):
    caplog.set_level(logging.INFO, logger="multistream")
    dag, nodes = graph("runtime", ("one",))
    plain = ms.plan(dag)
    sample = profile(tmp_path / "sample.csv", [("one", "AI_CORE", 1, 1, 0)])
    weighted = ms.plan(dag, profiling_path=sample)
    assert mapping(plain, nodes) == mapping(weighted, nodes) == (0,)
    queries = [
        record.getMessage()
        for record in caplog.records
        if "queried core limits from Runtime:" in record.getMessage()
    ]
    assert len(queries) == 2
    assert queries[0] == queries[1]
    match = re.search(r"ai_core_cnt=(\d+), vector_core_cnt=(\d+)", queries[0])
    assert match
    observed = tuple(map(int, match.groups()))
    assert observed == stub_runtime.counts
    assert len(stub_runtime.calls) == 4
    assert {key for _label, key, _size in stub_runtime.calls} == {
        b"ai_core_cnt",
        b"vector_core_cnt",
    }
    stub_runtime.counts = (32, 0)
    caplog.clear()
    ms.plan(dag)
    assert any(
        "ai_core_cnt=32, vector_core_cnt=0" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "shape", ("empty", "single", "chain", "diamond", "disconnected")
)
def test_ms_st_03_public_graphs(shape):
    if shape == "empty":
        dag, nodes = graph(shape, ())
    elif shape == "single":
        dag, nodes = graph(shape, ("a",))
    elif shape == "chain":
        names = tuple("n%d" % i for i in range(8))
        dag, nodes = graph(shape, names, tuple(zip(names, names[1:])))
    elif shape == "diamond":
        dag, nodes = graph(
            shape,
            ("a", "b", "c", "d"),
            (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")),
        )
    else:
        dag, nodes = graph(shape, ("a", "b", "c", "d"), (("a", "b"), ("c", "d")))
    before = [
        (node.get_name(), node.get_type(), node.get_node_cost())
        for node in nodes.values()
    ]
    result = ms.plan(dag)
    ids = mapping(result, nodes)
    assert dag.get_node_count() == len(nodes)
    assert [
        (node.get_name(), node.get_type(), node.get_node_cost())
        for node in nodes.values()
    ] == before
    assert len(ids) == len(nodes)
    if shape == "empty":
        assert result.get_required_stream_count() == 0
    elif shape in ("single", "chain"):
        assert result.get_required_stream_count() == 1
        assert ids == (0,) * len(nodes)
    else:
        assert ids == ((0, 0, 0, 0) if shape == "diamond" else (0, 0, 1, 1))


@pytest.mark.parametrize("strategy", list(ms.MergeStrategy))
def test_ms_st_04_strategy_limit_and_offset(strategy, caplog):
    caplog.set_level(logging.INFO, logger="multistream")
    dag, nodes = graph("branches", ("a", "b", "c", "d"), (("a", "b"),))
    for limit in (1, 2, 8, 64, None):
        size = 8 if limit is None else limit
        plain = ms.plan(dag, max_stream_num=limit, merge_strategy=strategy)
        shifted = ms.plan(
            dag, max_stream_num=limit, base_stream_id=7, merge_strategy=strategy
        )
        plain_ids = check_regular(plain, nodes, 0, size)
        shifted_ids = check_regular(shifted, nodes, 7, size)
        assert shifted_ids == tuple(item + 7 for item in plain_ids)
        if size == 1:
            assert plain_ids == (0, 0, 0, 0)
    route(caplog, strategy.value)


@pytest.mark.parametrize("limit", (1, 2))
def test_ms_st_05_serial_streams(limit):
    dag, nodes = graph("serial", ("ordinary", "s1", "s2", "s3"))
    for name, label in (("s1", "same"), ("s2", "same"), ("s3", "other")):
        nodes[name].set_stream_label(label)
    nodes["s2"].set_node_cost(ms.NodeCost(execution_time=0))
    result = ms.plan(dag, max_stream_num=limit, base_stream_id=7)
    ids = dict(zip(nodes, mapping(result, nodes)))
    assert ids["ordinary"] == 7
    assert ids["s1"] == ids["s2"] == 8
    assert ids["s3"] == 9
    assert result.get_required_stream_count() == 3
    all_serial, serial_nodes = graph("all_serial", ("x", "y"))
    serial_nodes["x"].set_stream_label("x")
    serial_nodes["y"].set_stream_label("y")
    assert mapping(
        ms.plan(all_serial, max_stream_num=1, base_stream_id=7), serial_nodes
    ) == (7, 8)


@pytest.mark.parametrize("strategy", list(ms.MergeStrategy))
def test_ms_st_06_exclusion_and_dependencies(strategy):
    dag, nodes = graph(
        "residual",
        ("a", "zero", "serial", "b", "Data", "NetOutput"),
        (
            ("a", "zero"),
            ("zero", "b"),
            ("a", "serial"),
            ("serial", "b"),
            ("Data", "NetOutput"),
        ),
    )
    nodes["zero"].set_node_cost(ms.NodeCost(execution_time=0))
    nodes["serial"].set_node_cost(ms.NodeCost(execution_time=0))
    nodes["serial"].set_stream_label("serial")
    dag.add_edge(nodes["a"], -1, nodes["b"], -1)
    result = ms.plan(dag, max_stream_num=2, merge_strategy=strategy)
    assert mapping(result, nodes) == (0, -1, 2, 0, 1, 1)
    assert dag.get_node_count() == 6


@pytest.mark.parametrize("strategy", list(ms.MergeStrategy))
def test_ms_st_07_profiling_overrides_strategy(strategy, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="multistream")
    dag, nodes = graph("profiled", ("a", "b", "c", "d"), (("a", "b"),))
    before = [node.get_node_cost() for node in nodes.values()]
    plain = ms.plan(dag, max_stream_num=2, merge_strategy=strategy)
    route(caplog, strategy.value)
    caplog.clear()
    csv_path = profile(
        tmp_path / "op_summary.csv",
        [
            (name, "AI_CORE", duration, 1, 0)
            for name, duration in zip(nodes, (1, 1, 1, 2))
        ],
    )
    sampled = ms.plan(
        dag, max_stream_num=2, merge_strategy=strategy, profiling_path=csv_path
    )
    route(caplog, "WeightedLoadBalance")
    assert mapping(sampled, nodes) == (0, 0, 0, 1)
    assert mapping(plain, nodes)[0:2] == (0, 0)
    assert [node.get_node_cost() for node in nodes.values()] == before


@pytest.mark.parametrize("strategy", list(ms.MergeStrategy))
def test_ms_st_08_profiling_fallback(strategy, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="multistream")
    dag, nodes = graph("fallback", ("a", "b", "zero", "serial"), (("a", "b"),))
    nodes["zero"].set_node_cost(ms.NodeCost(execution_time=0))
    nodes["serial"].set_stream_label("serial")
    baseline = ms.plan(dag, max_stream_num=2, merge_strategy=strategy)
    variants = (
        None,
        "",
        profile(tmp_path / "header.csv", []),
        profile(tmp_path / "miss.csv", [("unknown", "AI_CORE", 1, 1, 0)]),
        profile(
            tmp_path / "excluded.csv",
            [("zero", "AI_CORE", 1, 1, 0), ("serial", "AI_CORE", 1, 1, 0)],
        ),
    )
    for path in variants:
        caplog.clear()
        result = ms.plan(
            dag, max_stream_num=2, merge_strategy=strategy, profiling_path=path
        )
        route(caplog, strategy.value)
        assert mapping(result, nodes) == mapping(baseline, nodes)
        assert result.get_stream_id(nodes["zero"]) == -1
    hit = profile(tmp_path / "hit.csv", [("a", "AI_CORE", 3, 1, 0)])
    caplog.clear()
    ms.plan(dag, max_stream_num=2, merge_strategy=strategy, profiling_path=hit)
    route(caplog, "WeightedLoadBalance")


def test_ms_st_09_csv_compatibility(tmp_path, caplog):
    dag, nodes = graph("csv", ("scope/a", "b", "c", "d"))
    csv_path = profile(
        tmp_path / "real_shape.csv",
        [
            ("AI_CORE", "a", 1, 1, 0),
            ("AI_CORE", "scope/a", 2, 2, 0),
            ("AI_CORE", "scope/a", 3, 3, 0),
            ("AI_VECTOR_CORE", "b", 1, 2, 0),
            ("MIX_AIC", "c", 1, 1, 2),
            ("MIX_AIV", "d", 1, 1, 2),
            ("AI_CORE", "bad", "nan", 1, 0),
        ],
        headers=(
            "Task Type",
            "Op Name",
            "Task Duration(us)",
            "Block Num",
            "Mix Block Num",
            "Extra",
        ),
    )
    caplog.set_level(logging.INFO, logger="multistream")
    result = ms.plan(dag, profiling_path=csv_path)
    assert len(mapping(result, nodes)) == 4
    messages = [record.getMessage() for record in caplog.records]
    for fragment in (
        "scope/a: time=3.000000, cube=3, vec=0",
        "b: time=1.000000, cube=0, vec=2",
        "c: time=1.000000, cube=1, vec=2",
        "d: time=1.000000, cube=2, vec=1",
    ):
        assert any(fragment in message for message in messages)
    for invalid in (
        tmp_path / "missing.csv",
        profile(tmp_path / "missing_column.csv", [], headers=("Op Name",)),
    ):
        with pytest.raises(ValueError):
            ms.plan(dag, profiling_path=invalid)


def test_ms_st_10_core_type_and_limits(tmp_path):
    dag, nodes = graph("cores", tuple(item.name for item in ms.CoreType))
    before = {}
    for name, node in nodes.items():
        node.set_core_type(ms.CoreType[name])
        node.set_node_cost(ms.NodeCost(cube_core_num_limit=2, vec_core_num_limit=3))
        before[name] = (node.get_core_type(), node.get_node_cost())
    csv_path = profile(
        tmp_path / "cores.csv", [(name, "AI_CORE", 1, 1, 0) for name in nodes]
    )
    for path in (None, csv_path):
        result = ms.plan(dag, profiling_path=path)
        for name, node in nodes.items():
            assert (node.get_core_type(), node.get_node_cost()) == before[name]
            assert (
                result.get_core_num(node, "AIC")
                == result.get_core_num(node, "AIV")
                == 0
            )
        with pytest.raises(ValueError):
            result.get_core_num(next(iter(nodes.values())), "MIX_AIC")


def test_ms_st_11_reentrant_parallel_and_hash_seed(tmp_path):
    dag, nodes = graph(
        "repeat", ("a", "b", "c", "d"), (("a", "b"), ("a", "c"), ("c", "d"))
    )
    nodes["d"].set_stream_label("serial")
    before = [node.get_node_cost() for node in nodes.values()]
    csv_path = profile(tmp_path / "repeat.csv", [("a", "AI_CORE", 1, 1, 0)])
    first = ms.plan(dag)
    first_map = mapping(first, nodes)
    ms.plan(dag, profiling_path=csv_path)
    assert mapping(ms.plan(dag), nodes) == mapping(first, nodes) == first_map
    assert [node.get_node_cost() for node in nodes.values()] == before
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: mapping(ms.plan(dag), nodes), range(16)))
    assert all(result == first_map for result in results)
    script = (
        "from common import runtime_stub; runtime_stub.install(); "
        "import multistream as m; g=m.DAGGraph('hash'); "
        "ns=[g.add_node(x,'Op') for x in ('a','b','c')]; "
        "g.add_edge(ns[0],0,ns[1],0); p=m.plan(g); "
        "print(tuple(p.get_stream_id(n) for n in ns))"
    )
    outputs = []
    for seed in ("1", "2"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script], env=env, cwd=str(tmp_path), timeout=30
            ).strip()
        )
    assert outputs[0] == outputs[1]


def test_ms_st_12_errors_and_recovery(tmp_path):
    dag, nodes = graph("valid", ("a", "b"), (("a", "b"),))
    first = ms.plan(dag)
    original = mapping(first, nodes)
    for value in (0, 65):
        with pytest.raises(ValueError):
            ms.plan(dag, max_stream_num=value)
    with pytest.raises(ValueError):
        ms.plan(dag, merge_strategy="unknown")
    with pytest.raises(ValueError):
        ms.plan(dag, profiling_path=tmp_path / "absent.csv")
    foreign, other = graph("foreign", ("x",))
    with pytest.raises(ValueError):
        dag.add_edge(nodes["a"], 0, other["x"], 0)
    cycle, cycle_nodes = graph("cycle", ("x", "y"), (("x", "y"), ("y", "x")))
    with pytest.raises(RuntimeError, match="cycle"):
        ms.plan(cycle)
    assert mapping(first, nodes) == original
    assert mapping(ms.plan(dag), nodes) == original
    assert dag.get_node_count() == 2


@pytest.mark.perf
@pytest.mark.parametrize("shape", ("chain", "wide", "sparse"))
def test_ms_st_13_scale(shape):
    count = 100 if shape == "wide" else 1000
    dag, nodes = graph(shape, tuple("n%d" % i for i in range(count)))
    ordered = list(nodes.values())
    if shape == "chain":
        edges = ((index - 1, index) for index in range(1, count))
    elif shape == "sparse":
        rng = random.Random(17)
        edges = ((rng.randrange(index), index) for index in range(1, count))
    else:
        edges = ()
    for src, dst in edges:
        dag.add_edge(ordered[src], 0, ordered[dst], 0)
    started = time.monotonic()
    tracemalloc.start()
    result = ms.plan(dag)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ids = mapping(result, nodes)
    assert len(ids) == count
    assert all(0 <= stream_id < 8 for stream_id in ids)
    print(
        "MS-ST-13 %s nodes=%d elapsed=%.3fs peak_bytes=%d"
        % (shape, count, time.monotonic() - started, peak)
    )
