# Multistream

## Overview

Multistream is a pure Python stream planning module based on MiniDAG. The caller describes operators and their dependencies as a directed acyclic graph. The module generates logical streams through path covering, merges them into a limited number of physical streams, and returns each node's stream ID and the required stream count. The caller is responsible for creating streams, scheduling tasks, and synchronizing across streams according to the plan.

The module supports two base merge strategies: `LOAD_BALANCE` and `MAIN_STREAM`. When profiling data contains records for nodes participating in ordinary path merging, the module automatically uses weighted load balancing to assign streams based on execution time and core usage. Performance gains must be measured on the target device.

## Environment and Installation

`pyproject.toml` requires Python 3.7 or later. The package has no third-party Python runtime dependencies and contains no C/C++ extensions of its own. Run the following command from the repository root:

```bash
python3 -m pip install ./multistream
```

Importing the module and creating a graph do not load CANN Runtime. Before calling `plan()`, the application must configure the CANN Runtime environment so that the system can locate `libruntime.so`, and initialize the runtime or configure an offline target SoC. Planning queries the current target's core counts through `rtGetSocSpec`. The module does not set the target itself or use default core counts when the query fails.

## Basic Usage

Run this example after preparing the Runtime environment described above:

```python
import multistream as ms

graph = ms.DAGGraph("branches")
source = graph.add_node("source", "Op")
left = graph.add_node("left", "Op")
right = graph.add_node("right", "Op")
join = graph.add_node("join", "Op")

graph.add_edge(source, 0, left, 0)
graph.add_edge(source, 1, right, 0)
graph.add_edge(left, 0, join, 0)
graph.add_edge(right, 0, join, 1)

result = ms.plan(
    graph,
    max_stream_num=2,
    base_stream_id=0,
    merge_strategy=ms.MergeStrategy.LOAD_BALANCE,
)

print("required streams:", result.get_required_stream_count())
for node in graph.get_nodes():
    print(node.get_name(), result.get_stream_id(node))
```

`add_node()` requires unique node names within a graph and returns `None` for a duplicate name. `find_node()` also returns `None` when no matching node exists. Both nodes passed to `add_edge(src, src_port, dst, dst_port)` must belong to the current graph. Ports accept nonnegative integers or `-1`. `plan()` checks the graph for cycles.

## API Reference

The module exposes `DAGGraph`, `DAGNode`, `NodeCost`, `CoreType`, `StreamPlan`, `MergeStrategy`, and `plan`. Create nodes through `DAGGraph.add_node()` and read results through the query methods of `StreamPlan`.

### Planning Parameters and Results

| Parameter | Default | Description |
|-----------|---------|-------------|
| `graph` | Required | A `DAGGraph` object |
| `max_stream_num` | `None`, treated as 8 | Maximum physical stream count for ordinary nodes, from 1 to 64 |
| `base_stream_id` | `0` | Nonnegative integer specifying the starting stream ID |
| `merge_strategy` | `MergeStrategy.LOAD_BALANCE` | Base merge strategy; also supports `MergeStrategy.MAIN_STREAM` |
| `profiling_path` | `None` | Path to a profiling CSV file; an empty string also disables profiling |

All parameters except `graph` are keyword-only.

- `get_stream_id(node)`: Returns the stream ID of a node in the original graph. Excluded ordinary nodes return `ms.INVALID_STREAM_ID` (`-1`).
- `get_required_stream_count()`: Returns the total number of ordinary streams and additional serial streams, excluding the `base_stream_id` offset.
- `get_core_num(node, "AIC")` / `get_core_num(node, "AIV")`: Returns the suggested core count. The current version always returns `0`, indicating that the caller should retain its original configuration. These queries do not accept `MIX_AIC` or `MIX_AIV`.

Planning does not modify the input graph or its original costs. Subsequent planning calls do not overwrite existing results. Query a result using nodes from the original graph of that planning call.

### Node Duration, Resource Usage, and Serial Labels

Use `node.set_node_cost(ms.NodeCost(...))` to set node duration and resource usage. `NodeCost` accepts keyword-only arguments and is immutable. To change a cost, create a new `NodeCost` object and assign it to the node.

| Field | Default | Description |
|-------|---------|-------------|
| `execution_time` | `-1.0` | Unknown duration; use microseconds when combining it with profiling. Stored as float32 |
| `memory_usage`, `bandwidth_usage` | `0` | Memory and bandwidth usage |
| `cube_block_num`, `vec_block_num` | `0` | Actual Cube and Vector core usage of the operator |
| `cube_core_num_limit`, `vec_core_num_limit` | `0` | User-configured core limits; `0` means unlimited |

Duration must be finite, and the other fields must be nonnegative integers. An ordinary node is not assigned a stream when the absolute value of its duration for the current plan is at most `1e-6`. The default duration of `-1.0` still participates in stream assignment. Profiling does not override a duration manually set to `0.0`.

`node.set_core_type()` accepts `CoreType.AIC`, `MIX_AIC`, `AIV`, `MIX_AIV`, or `UNKNOWN` to describe the node's execution engine. The default is `UNKNOWN`.

`node.set_stream_label("serial-group")` assigns a node to a serial group. Nodes with the same nonempty label share one additional stream. Streams for distinct labels are appended after the ordinary streams in order of first appearance. Labeled nodes receive a stream even when their duration is zero, so the total stream count may exceed `max_stream_num`. Allocate resources according to `get_required_stream_count()`.

### Using Profiling

`profiling_path` accepts a UTF-8 CSV file, for example:

```csv
Op Name,Task Type,Task Duration(us),Block Num,Mix Block Num
left,AI_CORE,10.0,8,0
right,AI_VECTOR_CORE,6.0,8,0
```

Add `profiling_path="op_summary.csv"` to the `ms.plan(...)` call in the basic example. All five column headers are required. `Task Duration(us)` must be finite and positive, and both Block Num fields must be within the uint32 range.

| Task Type | Cube Core Source | Vector Core Source |
|-----------|------------------|--------------------|
| `AI_CORE` | `Block Num` | 0 |
| `AI_VECTOR_CORE` | 0 | `Block Num` |
| `MIX_AIC` | `Block Num` | `Mix Block Num` |
| `MIX_AIV` | `Mix Block Num` | `Block Num` |

Matching first uses the full node name, then the part after the last `/` in the node name. For valid records with the same name, the last record takes precedence. Invalid rows are skipped with a warning. Planning fails if the file cannot be read or a required header is missing.

Profiling duration and actual core usage apply only to the current plan. They do not modify the original node information or user-configured core limits. If a valid record matches a node participating in ordinary path merging, planning uses weighted load balancing even when `MAIN_STREAM` is specified. The specified base strategy remains in use when only `NodeCost` is set, or when profiling matches only serial nodes or nodes excluded from stream assignment.

## Directory Structure

```text
multistream/
  algorithm/       # Path covering, stream merging, profiling parsing, and planning entry point
  common/          # Logging, internal errors, and Runtime core queries
  minidag/         # Graphs, nodes, cost types, and planning results
  docs/            # Developer documentation
  scripts/         # UT/ST coverage configuration
  tests/           # Unit tests, system tests, and shared test fixtures
  CMakeLists.txt   # Build and packaging configuration
  pyproject.toml   # Package metadata, dependencies, and pytest configuration
```

See the [Developer Guide](docs/developer_guide_en.md) for the test framework, coverage, performance tests, and debugging.
