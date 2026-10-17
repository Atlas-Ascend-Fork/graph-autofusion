# Multistream

## 原理介绍

Multistream 是基于 MiniDAG 的纯 Python 多流规划模块。调用方将算子及其依赖关系描述为有向无环图，模块通过路径覆盖生成逻辑流，再将逻辑流合并为有限数量的物理流，返回每个节点的流 ID 和所需流数。调用方负责根据规划结果创建流、安排任务及跨流同步。

模块支持 `LOAD_BALANCE`（负载均衡）和 `MAIN_STREAM`（主流优先）两种基础合并策略。如果性能采集数据（profiling）中有参与普通路径合并的节点记录，模块会自动使用加权负载均衡，根据执行时间和核占用情况分配流。具体能提升多少性能，需要在目标设备上测试。

## 环境准备与安装

`pyproject.toml` 中声明的 Python 版本要求为 3.7 及以上，不依赖第三方 Python 运行时包，也不包含自身的 C/C++ 扩展。在仓库根目录执行：

```bash
python3 -m pip install ./multistream
```

导入模块和创建图不需要加载 CANN Runtime。调用 `plan()` 前，调用程序需要配置好 CANN Runtime 环境，确保系统能找到 `libruntime.so`，并完成运行时初始化或离线目标 SoC 配置。规划通过 `rtGetSocSpec` 查询当前目标的核数；模块不会自行设置目标，也不会在查询失败时使用默认核数。

## 基础用法

以下示例在上述 Runtime 环境就绪后运行：

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

`add_node()` 要求节点名称在图内唯一，重复名称返回 `None`；`find_node()` 未找到节点时也返回 `None`。`add_edge(src, src_port, dst, dst_port)` 的两个节点必须属于当前图，端口接受非负整数或 `-1`。`plan()` 会检查图是否有环。

## 接口说明

模块提供以下接口： `DAGGraph`、`DAGNode`、`NodeCost`、`CoreType`、`StreamPlan`、`MergeStrategy` 和 `plan`。节点应通过 `DAGGraph.add_node()` 创建，结果通过 `StreamPlan` 的查询方法读取。

### 规划参数与结果

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `graph` | 必填 | `DAGGraph` 对象 |
| `max_stream_num` | `None`，按 8 处理 | 普通节点的物理流数上限，取值为 1～64 |
| `base_stream_id` | `0` | 非负整数，分配流 ID 的起点 |
| `merge_strategy` | `MergeStrategy.LOAD_BALANCE` | 基础合并策略，另支持 `MergeStrategy.MAIN_STREAM` |
| `profiling_path` | `None` | profiling CSV 文件路径，空字符串也表示不使用 profiling |

除 `graph` 外的参数均通过关键字传递。

- `get_stream_id(node)`：查询原图节点的流 ID；被排除的普通节点返回 `ms.INVALID_STREAM_ID`（`-1`）。
- `get_required_stream_count()`：返回普通流与额外串行流的总数，不包含 `base_stream_id` 偏移。
- `get_core_num(node, "AIC")` / `get_core_num(node, "AIV")`：查询建议核数。当前版本始终返回 `0`，表示调用方保留原配置；查询时不能传入 `MIX_AIC` 或 `MIX_AIV`。

规划不修改输入图及其原始成本；后续规划不会覆盖已有结果。查询结果时应传入本次规划原图中的节点。

### 节点耗时、资源占用与串行标签

通过 `node.set_node_cost(ms.NodeCost(...))` 设置节点耗时和资源占用信息。`NodeCost` 仅接受关键字参数，创建后不可修改；需要调整时，创建新的 `NodeCost` 对象并重新设置。

| 字段 | 默认值 | 含义 |
|------|--------|------|
| `execution_time` | `-1.0` | 未知耗时；与 profiling 一起使用时按微秒填写，存储时转换为 float32 |
| `memory_usage`、`bandwidth_usage` | `0` | 内存、带宽用量 |
| `cube_block_num`、`vec_block_num` | `0` | 算子实际占用的 Cube、Vector 核数 |
| `cube_core_num_limit`、`vec_core_num_limit` | `0` | 用户配置的核数上限，`0` 表示不限制 |

耗时必须是有限数值，其他字段要求非负整数。普通节点用于本次规划的耗时绝对值不超过 `1e-6` 时，不为其分配流；默认耗时 `-1.0` 仍参与分流。手动设置的 `0.0` 耗时不会被 profiling 覆盖。

`node.set_core_type()` 接受 `CoreType.AIC`、`MIX_AIC`、`AIV`、`MIX_AIV` 或 `UNKNOWN`，描述节点的执行引擎类型，默认是 `UNKNOWN`。

`node.set_stream_label("serial-group")` 将节点加入串行组。同一非空标签的节点分配到同一条额外流，不同标签按首次出现顺序追加在普通流之后。带标签节点即使耗时为零也会分配流，因此总流数可能超过 `max_stream_num`，调用方应以 `get_required_stream_count()` 为准分配资源。

### 使用 profiling

`profiling_path` 接受 UTF-8 编码的 CSV 文件，例如：

```csv
Op Name,Task Type,Task Duration(us),Block Num,Mix Block Num
left,AI_CORE,10.0,8,0
right,AI_VECTOR_CORE,6.0,8,0
```

在基础用法中的 `ms.plan(...)` 调用增加 `profiling_path="op_summary.csv"` 即可使用。五个表头均为必需项，`Task Duration(us)` 必须为有限正数，两个 Block Num 字段必须在 uint32 范围内。

| Task Type | Cube 核数来源 | Vector 核数来源 |
|-----------|---------------|-----------------|
| `AI_CORE` | `Block Num` | 0 |
| `AI_VECTOR_CORE` | 0 | `Block Num` |
| `MIX_AIC` | `Block Num` | `Mix Block Num` |
| `MIX_AIV` | `Mix Block Num` | `Block Num` |

匹配优先使用节点全名，再以节点名最后一个 `/` 之后的部分查找 CSV 记录；同名有效记录以最后一条为准。无效行会被跳过并记录告警；如果文件无法读取或缺少必需表头，规划会失败。

profiling 中的耗时和实际核占用数只用于本次规划，不修改节点原有信息，也不修改用户设置的核数上限。如果有效记录匹配到参与普通路径合并的节点，即使指定了 `MAIN_STREAM`，也会改用加权负载均衡。仅设置 `NodeCost`，或者 profiling 只匹配到串行节点、不参与分流的节点时，仍使用指定的基础策略。

## 目录结构

```text
multistream/
  algorithm/       # 路径覆盖、流合并、profiling 解析与规划入口
  common/          # 日志、内部错误和 Runtime 核数查询
  minidag/         # 图、节点、成本类型和规划结果
  docs/            # 开发者资料
  scripts/         # UT/ST 覆盖率配置
  tests/           # 单元测试、系统测试及公共测试函数
  CMakeLists.txt   # 构建和打包配置
  pyproject.toml   # 包信息、依赖和 pytest 配置
```

测试框架、覆盖率、性能测试及调试方法见[开发者指南](docs/developer_guide.md)。
