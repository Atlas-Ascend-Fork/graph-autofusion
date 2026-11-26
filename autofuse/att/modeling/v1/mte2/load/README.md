# A3 MTE2 Load（GM2UB）模型

[English](README_en.md)

A3 采集证据对应 GM → UB `LoadApi`，按模型家族归档到 `v1/mte2/load/`。[parameters.json](parameters.json) 顶层的 `parameters`、`formula` 和 `source.implementation` 对应当前 V1 实现，原归档的 v35 参数快照、公式及实现来源独立保存在 `v35_snapshot` 中。当前 V1 实现见 [perf_param_v1.cpp](../../../../gen_model_info/api_perf_register/v1/perf_param_v1.cpp)。快照公式见 [模型设计](design.md#模型公式)，计量口径见 [统一口径](../../../v35/mte2/load/README.md#统一口径)。

## 文档

| 文件 | 内容 |
| --- | --- |
| [design.md](design.md) | 实现公式、分支和数据覆盖 |
| [cases.csv](cases.csv) | A3 采集阶段与证据索引 |
| [parameters.json](parameters.json) | 实现参数和适用范围 |

## 原归档参数快照（v35）

```text
Load:                 t = 11.8292 + 6.6155 / block_dim; h = 160; hl = 0
LoadSmallBlk:         t = 13.1355 + 6.4088 / block_dim; h = 160; hl = 0
LoadSingleCore:       t = 23.0520; h = 160; hl = 0
LoadStride:           0.005 * block_count * min(gm_stride * data_type_size, 4096.0)
pipe head:            1174.3 for blocklen >= 256B, otherwise 775.0
```

`block_dim==1` 使用单核参数表；其余使用多核参数表。LoadPerf 按 blocklen 选择大块或小块表达式，并将 stride 成本加入对应分支。

## 数据覆盖

A3 的连续、小尺度、headcost 和 srcStride 证据索引见 [cases.csv](cases.csv)。这些文件用于记录采集覆盖和复现入口，当前模型计算以 V1 实现参数表为准。
