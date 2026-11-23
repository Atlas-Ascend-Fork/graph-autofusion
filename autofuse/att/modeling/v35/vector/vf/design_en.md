# v35 VF Instruction Estimate Additions

FP16/FP32 `TrueDiv` uses the latency and issue cost of the existing Div instruction. `Rsqrt` is modeled from the
actual MicroAPI lowering order. Duplicate and Sqrt are independent predecessors of Div, so the critical path is
`max(latency(Duplicate), latency(Sqrt)) + latency(Div)`, and the issue cost is the sum of the three instructions.
The parameters reuse the existing table; no separate VF-instruction fitting is performed. The corresponding
assertion is `UTestAscirPerfV2.VectorFunctionCountsTrueDivAndRsqrtLowering`.

The shared startup term is archived separately in [GetOpHeadCost](../../common/op_head_cost/design_en.md).
