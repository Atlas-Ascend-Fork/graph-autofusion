# fusion_precision_analyzer

融合算子精度定位工具。当关闭自动融合网络精度正常、开启自动融合精度劣化时，通过比较两个场景的 dump 数据，定位是哪个融合算子导致了整网精度劣化。

## 功能

工具支持两种运行模式：

- **模式1（默认）**：基于 dump 图 JSON 和 datadump NPY 目录，批量比较每个融合算子的输入输出与对应原算子输出的精度差异，输出控制台表格
- **模式2**：直接比较两个 NPY 文件，输出余弦相似度、绝对误差最大值、相对误差最大值

### 精度指标

| 指标 | 计算方式 |
|------|----------|
| 余弦相似度 | `dot(a, b) / (norm(a) * norm(b) + 1e-8)` |
| 绝对误差最大值 | `max(|a - b|)` |
| 相对误差最大值 | `max(|a - b| / (max(|a|, |b|) + 1e-8))` |

### 数据差异处理

| 场景 | 处理方式 | 状态标记 |
|------|----------|----------|
| NPY 文件未找到 | 跳过 | `FILE_NOT_FOUND` |
| NPY 加载失败 | 跳过 | `NPY_LOAD_ERROR` |
| 数组形状与声明布局不匹配，转换失败 | 跳过当前 pair | `FORMAT_CONVERSION_ERROR` |
| format 不一致，支持转换 | NC1HWC0→NHWC/ND、NDC1HWC0→NDHWC/ND | `FORMAT_CONVERTED` |
| format 不一致，不支持转换 | 跳过 | `FORMAT_UNSUPPORTED` |
| dtype 不一致 | 低精度提升为高精度 | `DTYPE_CAST` |
| shape 不一致但元素数相同 | 展平后比较 | `SHAPE_FLATTENED` |
| shape 不一致且元素数不同 | 跳过 | `SHAPE_MISMATCH` |
| 输入源为 Constant/Data | 跳过 | `SKIPPED_CONST_DATA` |
| 缺少映射属性 | 跳过 | `NO_MAPPING` |
| 指标计算失败 | 跳过 | `COMPUTE_ERROR` |

## 依赖

```bash
pip install numpy
```

## 使用方式

### 模式1：批量比较融合算子输入输出

```bash
python3 fusion_precision_analyzer.py \
  --af-open-graph <开启融合dump图JSON> \
  --af-open-data <开启融合datadump NPY目录> \
  --af-close-data <关闭融合datadump NPY目录> \
  [--compare-input]
```

| 参数 | 必选 | 说明 |
|------|------|------|
| `--mode` | 否 | 运行模式，1或2，默认1 |
| `--af-open-graph` | 是 | 开启自动融合的 dump 图 JSON 文件路径 |
| `--af-close-graph` | 是 | 关闭自动融合的 dump 图 JSON 文件路径 |
| `--af-open-data` | 是 | 开启自动融合的 datadump NPY 目录 |
| `--af-close-data` | 是 | 关闭自动融合的 datadump NPY 目录 |
| `--compare-input` | 否 | 是否比较融合算子输入，默认不比较 |

### 模式2：直接比较两个 NPY 文件

```bash
python3 fusion_precision_analyzer.py \
  --mode 2 \
  --npy-a <NPY文件A> \
  --npy-b <NPY文件B>
```

| 参数 | 必选 | 说明 |
|------|------|------|
| `--mode` | 否 | 运行模式，设为2 |
| `--npy-a` | 是 | 第一个 NPY 文件路径 |
| `--npy-b` | 是 | 第二个 NPY 文件路径 |

## 使用样例

### 样例1：模式1，仅比较输出

```bash
python3 fusion_precision_analyzer.py \
  --af-open-graph /home/user/dumpgraph_af_open/Build.json \
  --af-close-graph /home/user/dumpgraph_af_close/Build.json \
  --af-open-data /home/user/datadump_af_open \
  --af-close-data /home/user/datadump_af_close
```

输出示例：

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

### 样例2：模式1，同时比较输入和输出

```bash
python3 fusion_precision_analyzer.py \
  --af-open-graph /home/user/dumpgraph_af_open/Build.json \
  --af-close-graph /home/user/dumpgraph_af_close/Build.json \
  --af-open-data /home/user/datadump_af_open \
  --af-close-data /home/user/datadump_af_close \
  --compare-input
```

输出示例：

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

### 样例3：模式2，直接比较两个 NPY 文件

```bash
python3 fusion_precision_analyzer.py \
  --mode 2 \
  --npy-a /home/user/datadump_af_open/AscBackend.autofuse_28_Pow_RealDiv_Square.3.44.1783906987170021.output.0.npy \
  --npy-b /home/user/datadump_af_close/Square.ArithmeticOptimizer_ReplaceMulWithSquare_Mul_7.9.44.1783906974676130.output.0.npy
```

输出示例：

```
文件A: /home/user/datadump_af_open/AscBackend.autofuse_28_Pow_RealDiv_Square.3.44.1783906987170021.output.0.npy
文件B: /home/user/datadump_af_close/Square.ArithmeticOptimizer_ReplaceMulWithSquare_Mul_7.9.44.1783906974676130.output.0.npy
状态: OK
余弦相似度: 1.0000000000
绝对误差最大值: 0.000000e+00
相对误差最大值: 0.000000e+00
```

## 工作原理

### 模式1

1. 解析开启融合的 dump 图 JSON，识别融合算子（`type` 为 `AscBackend` 或 `FusedAscBackend`）
2. 从融合算子的 `output_desc` 中提取 `_datadump_origin_name` 和 `_datadump_origin_output_index` 属性，建立融合算子输出到原算子输出的映射
3. 解析关闭融合的 dump 图 JSON，构建原算子节点名到输出 format 的查找表
4. 从融合算子的 `input` 字段解析 `"node:idx"` 引用，若输入源为融合算子则进一步解析为对应原算子
5. 根据 op name（`/` 替换为 `_`）在两个 datadump 目录中匹配 NPY 文件
6. 融合侧 format 从 af_open 图获取，原算子侧 format 从 af_close 图获取，对每对匹配的数据处理 format/dtype/shape 差异后计算精度指标
7. 输出控制台表格，按图中融合算子遍历顺序排列

### 模式2

1. 加载两个 NPY 文件
2. 处理 dtype/shape 差异后计算精度指标
3. 输出结果

## 前置条件

用户需通过 CANN 精度调试工具链获取以下数据：

- **dump 图 JSON**：开启自动融合场景的 GE dump 图，通过环境变量 `DUMP_GE_GRAPH=2` 等方式生成
- **datadump NPY 文件**：通过 CANN datadump 机制采集二进制数据，再通过 `msaccucmp.py` 工具转换为 NPY 格式
