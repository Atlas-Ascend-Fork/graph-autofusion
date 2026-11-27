# ATT 性能模型归档

用于归档模型设计、用例和验证结果，后续由各模型负责人补充。

## 目录

```text
modeling/v35/
├── README.md
├── common/                 # 公共开销
│   └── op_head_cost/
├── mte2/                   # 搬入模型
│   ├── README.md
│   ├── load/
│   └── nddma/
├── mte3/                   # 写出模型
│   ├── README.md
│   └── store/
├── vector/                 # Vector API 模型
│   ├── README.md
│   ├── broadcast/
│   ├── concat/
│   ├── elewise/
│   │   └── cast/
│   ├── gather/
│   ├── reduce/
│   ├── transpose/
│   └── vf/
└── simt/                   # 按实际执行路径增加子模型
    └── README.md
```

各分类 README 用于补充本类模型的建模方法。共享归档约定放在本文件中。

采集与拟合代码统一放在 [Autofuse 建模工具](../../../tools/modeling/README.md)。公共开销见 [GetOpHeadCost](common/op_head_cost/design.md)。

A5 Store 的原始标定及小块连续 Store 增补见 [MTE3 Store 归档](mte3/store/README.md)；A3 归档位于 [V1 Store](../v1/mte3/store/README.md)。

## 模型归档

每个模型按需补充以下文件：

```text
<model>/
├── design.md               # 建模思路、公式、参数来源和适用范围
├── cases.csv               # 用例、覆盖条件和执行入口
└── results/<run_id>/
    ├── manifest.json       # 环境、版本、命令及原始证据索引
    ├── samples.csv         # 模型预测与实际测量
    └── report.md           # 误差分析、结论及限制
```

设计文档突出建模依据，用例应可执行，结果须来自实际验证。
记录模型与环境版本，保持预测和实测口径一致；有拟合过程时区分拟合数据与独立验证数据。
原始日志、profiling 和二进制不入库，通过证据索引关联。
空目录使用 `.gitkeep` 保留，有实际归档内容后移除。

[v35 ATT 建模增补与验证入口](design.md)
