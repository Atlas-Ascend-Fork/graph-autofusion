# fusion_precision_analyzer

Fusion operator precision localization tool. When the network precision is normal with auto-fusion disabled but degrades with auto-fusion enabled, this tool compares dump data from both scenarios to identify which fusion operator caused the precision degradation.

## Features

The tool supports two running modes:

- **Mode 1 (default)**: Batch-compares the inputs and outputs of each fusion operator against the corresponding original operator outputs based on dump graph JSON and datadump NPY directories, outputting a console table
- **Mode 2**: Directly compares two NPY files, outputting cosine similarity, max absolute error, and max relative error

### Precision Metrics

| Metric | Formula |
|--------|---------|
| Cosine similarity | `dot(a, b) / (norm(a) * norm(b) + 1e-8)` |
| Max absolute error | `max(|a - b|)` |
| Max relative error | `max(|a - b| / (max(|a|, |b|) + 1e-8))` |

### Data Difference Handling

| Scenario | Handling | Status |
|----------|----------|--------|
| NPY file not found | Skip | `FILE_NOT_FOUND` |
| NPY load failure | Skip | `NPY_LOAD_ERROR` |
| Array shape does not match the declared layout | Skip current pair | `FORMAT_CONVERSION_ERROR` |
| Format mismatch, conversion supported | NC1HWC0 to NHWC/ND, NDC1HWC0 to NDHWC/ND | `FORMAT_CONVERTED` |
| Format mismatch, conversion unsupported | Skip | `FORMAT_UNSUPPORTED` |
| dtype mismatch | Promote lower precision to higher | `DTYPE_CAST` |
| Shape mismatch, same element count | Flatten and compare | `SHAPE_FLATTENED` |
| Shape mismatch, different element count | Skip | `SHAPE_MISMATCH` |
| Input source is Constant/Data | Skip | `SKIPPED_CONST_DATA` |
| Missing mapping attribute | Skip | `NO_MAPPING` |
| Metric computation failure | Skip | `COMPUTE_ERROR` |

## Dependencies

```bash
pip install numpy
```

## Usage

### Mode 1: Batch Compare Fusion Operator Inputs and Outputs

```bash
python3 fusion_precision_analyzer.py \
  --af-open-graph <af-open dump graph JSON> \
  --af-close-graph <af-close dump graph JSON> \
  --af-open-data <af-open datadump NPY directory> \
  --af-close-data <af-close datadump NPY directory> \
  [--compare-input]
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--mode` | No | Running mode, 1 or 2, default 1 |
| `--af-open-graph` | Yes | Dump graph JSON file path with auto-fusion enabled |
| `--af-close-graph` | Yes | Dump graph JSON file path with auto-fusion disabled |
| `--af-open-data` | Yes | Datadump NPY directory with auto-fusion enabled |
| `--af-close-data` | Yes | Datadump NPY directory with auto-fusion disabled |
| `--compare-input` | No | Whether to compare fusion operator inputs, default off |

### Mode 2: Directly Compare Two NPY Files

```bash
python3 fusion_precision_analyzer.py \
  --mode 2 \
  --npy-a <NPY file A> \
  --npy-b <NPY file B>
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--mode` | No | Running mode, set to 2 |
| `--npy-a` | Yes | First NPY file path |
| `--npy-b` | Yes | Second NPY file path |

## Examples

### Example 1: Mode 1, Compare Outputs Only

```bash
python3 fusion_precision_analyzer.py \
  --af-open-graph /home/user/dumpgraph_af_open/Build.json \
  --af-close-graph /home/user/dumpgraph_af_close/Build.json \
  --af-open-data /home/user/datadump_af_open \
  --af-close-data /home/user/datadump_af_close
```

Sample output:

```
解析开启融合 dump 图: /home/user/dumpgraph_af_open/Build.json
解析关闭融合 dump 图: /home/user/dumpgraph_af_close/Build.json
找到 8 个融合算子输出映射

类型 | 融合算子名                            | 索引 | 原算子名                  | 原索引 | 余弦相似度    | 绝对误差最大值 | 相对误差最大值 | 状态
-----+--------------------------------------+----+-------------------------+-----+------------+------------+------------+------
输出 | autofuse_28_Pow_RealDiv_Square       | 0  | ArithmeticOptimizer_... | 0   | 1.00000000 | 0.0000e+00 | 0.0000e+00 | OK
输出 | autofuse_38_Pow_Minimum_Reshape_...  | 0  | truediv_9               | 0   | 0.99999832 | 1.2000e-05 | 3.4000e-06 | OK
输出 | autofuse_35_Pow_Maximum_Square_...   | 0  | Sum_6                   | 0   | -          | -          | -          | FILE_NOT_FOUND
输出 | autofuse_33_Square_RealDiv           | 0  | truediv_16              | 0   | 1.00000000 | 0.0000e+00 | 0.0000e+00 | OK
```

### Example 2: Mode 1, Compare Both Inputs and Outputs

```bash
python3 fusion_precision_analyzer.py \
  --af-open-graph /home/user/dumpgraph_af_open/Build.json \
  --af-close-graph /home/user/dumpgraph_af_close/Build.json \
  --af-open-data /home/user/datadump_af_open \
  --af-close-data /home/user/datadump_af_close \
  --compare-input
```

Sample output:

```
解析开启融合 dump 图: /home/user/dumpgraph_af_open/Build.json
解析关闭融合 dump 图: /home/user/dumpgraph_af_close/Build.json
找到 8 个融合算子输出映射, 5 个融合算子输入映射

类型 | 融合算子名                            | 索引 | 原算子名                  | 原索引 | 余弦相似度    | 绝对误差最大值 | 相对误差最大值 | 状态
-----+--------------------------------------+----+-------------------------+-----+------------+------------+------------+------------------
输出 | autofuse_28_Pow_RealDiv_Square       | 0  | ArithmeticOptimizer_... | 0   | 1.00000000 | 0.0000e+00 | 0.0000e+00 | OK
输出 | autofuse_33_Square_RealDiv           | 0  | truediv_16              | 0   | 1.00000000 | 0.0000e+00 | 0.0000e+00 | OK
输入 | autofuse_28_Pow_RealDiv_Square       | 0  | dynamic_const_2309685_16| 0   | -          | -          | -          | SKIPPED_CONST_DATA
输入 | autofuse_33_Square_RealDiv           | 0  | truediv_3               | 0   | 1.00000000 | 0.0000e+00 | 0.0000e+00 | SHAPE_FLATTENED
```

### Example 3: Mode 2, Directly Compare Two NPY Files

```bash
python3 fusion_precision_analyzer.py \
  --mode 2 \
  --npy-a /home/user/datadump_af_open/AscBackend.autofuse_28_Pow_RealDiv_Square.3.44.1783906987170021.output.0.npy \
  --npy-b /home/user/datadump_af_close/Square.ArithmeticOptimizer_ReplaceMulWithSquare_Mul_7.9.44.1783906974676130.output.0.npy
```

Sample output:

```
文件A: /home/user/datadump_af_open/AscBackend.autofuse_28_Pow_RealDiv_Square.3.44.1783906987170021.output.0.npy
文件B: /home/user/datadump_af_close/Square.ArithmeticOptimizer_ReplaceMulWithSquare_Mul_7.9.44.1783906974676130.output.0.npy
状态: OK
余弦相似度: 1.0000000000
绝对误差最大值: 0.000000e+00
相对误差最大值: 0.000000e+00
```

## How It Works

### Mode 1

1. Parse the auto-fusion-enabled dump graph JSON to identify fusion operators (`type` is `AscBackend` or `FusedAscBackend`)
2. Extract `_datadump_origin_name` and `_datadump_origin_output_index` attributes from the fusion operators' `output_desc` to build the mapping from fusion operator outputs to original operator outputs
3. Parse the auto-fusion-disabled dump graph JSON to build a lookup table from original operator names to output formats
4. Parse `"node:idx"` references from the fusion operators' `input` field; if the input source is a fusion operator, further resolve it to the corresponding original operator
5. Match NPY files in both datadump directories by op name (replacing `/` with `_`)
6. Obtain the fusion-side format from the af_open graph and the original-side format from the af_close graph; for each matched pair, handle format/dtype/shape differences and compute precision metrics
7. Output a console table, ordered by the fusion operator traversal order in the graph

### Mode 2

1. Load two NPY files
2. Handle dtype/shape differences and compute precision metrics
3. Output results

## Prerequisites

Users need to obtain the following data through the CANN precision debugging toolchain:

- **Dump graph JSON**: GE dump graph from the auto-fusion-enabled scenario, generated via environment variable `DUMP_GE_GRAPH=2` or similar
- **Datadump NPY files**: Binary data collected via the CANN datadump mechanism, then converted to NPY format using the `msaccucmp.py` tool
