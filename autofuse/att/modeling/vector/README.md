# Vector 性能模型

分类沿用 `autofuse/codegen/api_call/`：`broadcast/`、`concat/`、`elewise/`、`gather/`、`reduce/`、`transpose/`。
`cast` 归入 `elewise/cast/`，其他逐元素 API 按需在 `elewise/` 下归档。
`datacopy` 的 load/store 分别归档于 mte2/mte3；`utils` 是公共工具，不作为模型分类。

## 待补充

- 建模方法：v35 基于 AscendC API 实现逻辑生成建模表达式，说明底层成本来源。
- 用例设计及执行方式。
- 验证结果、误差分析和适用范围。

模型文件结构见 [归档约定](../README.md)。
