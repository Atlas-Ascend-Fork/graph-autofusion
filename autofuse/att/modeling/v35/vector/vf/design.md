# v35 VF 指令估值增补

FP16/FP32 `TrueDiv` 使用既有 Div 指令的 latency 和 issue cost。`Rsqrt` 根据 MicroAPI 实际降低顺序建模。Duplicate 与 Sqrt 是 Div 的独立前驱，关键路径为 `max(latency(Duplicate), latency(Sqrt)) + latency(Div)`，issue cost 为三条指令之和。参数复用已有表，未对 VF 指令另行拟合。对应断言为 `UTestAscirPerfV2.VectorFunctionCountsTrueDivAndRsqrtLowering`。

公共启动项单独归档在 [GetOpHeadCost](../../common/op_head_cost/design.md)。
