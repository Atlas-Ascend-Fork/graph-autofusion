# ATT 性能模型归档

按模型家族归档设计、描述符和精简验证证据：[V1](v1/README.md) 包含 A3 Load 和 Store 归档；[v35](v35/README.md) 包含 A5 Store 归档、已有标定模型及[增补设计](v35/design.md)。`V2` 是 v35 注册键，不是第三个平台家族。

```text
modeling/
├── README.md / README_en.md
├── v1/
│   ├── README.md / README_en.md
│   ├── mte2/load/           # A3 采集证据与原归档参数快照
│   └── mte3/store/          # A3 采集证据与 v35 历史参数快照
└── v35/
    ├── design.md / design_en.md
    ├── common/op_head_cost/
    ├── mte2/{load,nddma}/
    ├── mte3/store/          # A5
    └── vector/
```

Load 采集资料按 A3 → [v1/mte2/load](v1/mte2/load/README.md)、A5 → [v35/mte2/load](v35/mte2/load/README.md) 归档。原始采集仓中的 A3/A5 路径保留作为证据索引。

每个模型包含 `design.md`、`cases.csv` 可执行描述符，以及 `results/<run_id>/` 下的版本/命令索引、samples 和报告。既有没有 run ID 的摘要保留在 `results/` 根，不再重复 `results/v35` 平台层。`direct-store-v1` 等 run ID 表示实验修订，不代表平台 V1；占位分类不代表已支持相应模型。

保留计量单位、样本 ID、训练/留出角色、参数及历史有效性标记，明确模型与 API 版本。原始 profiling、二进制和业务 tiling 回归数据通过固定版本证据索引关联，不复制入库。manifest 的 `case_inventory`、`reproduction` 相对自身目录解析；外部 source root 保持原声明语义。采集与拟合代码见[建模工具](../../tools/modeling/README.md)。
