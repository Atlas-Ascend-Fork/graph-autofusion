# v35 Load 模型增补

## 范围和公式

v35 的 FP32 Load 根据 Codegen 的实际搬运描述符选择估值。以下分支复用 `LoadStoreFunc`，单位为 cycle。设 B 为单核单次搬运的有效字节数，C 为活跃核数，估值为 `h + B / (a + b/C)`。表中宽度、间距和步长均以 FP32 元素计。普通 Load 在分支外继续使用原 payload、stride 和 cacheline 路径。

| 分支 | 描述符条件 | h | a | b |
| --- | --- | ---: | ---: | ---: |
| SmallPayload | 1–2 维，GM/UB 无行间隙，行宽按 32 B 对齐，512–8192 B，8–48 核 | 292.936412 | 20.724828 | 873.958753 |
| SmallPayloadCompact | 2–18 行，行宽 32–256 且 32 B 对齐，GM 行间隙 64–4096，UB 无行间隙，512–8192 B，8–48 核 | 282.480993 | 20.736478 | 779.354414 |
| AlignedSingleRow | 2 维描述符的单行，行宽 512–30680 且 32 B 对齐，24–56 核，单行不推进 stride | 253.290566 | 0 | 1310.182034 |
| AlignedDensePayload | 删除静态单位轴后为 1–2 维，GM/UB 无行间隙，行宽按 32 B 对齐，1024–4096 B，16–56 核 | 394.947854 | 41.831205 | 1134.576300 |
| AlignedWideGap | 16 或 32 行，行宽 384–768 且 32 B 对齐，GM pitch 131072，UB 无行间隙，40–56 核 | 0 | 0 | 987.983765 |
| WidePitch128B | 256–512 行，行宽 32，GM pitch 为 4096 的倍数且在 4096–16384 内，UB 无行间隙 | 160 | 21.813945 | 6.4088 |
| AlignedLoopPayload | 一层硬件 LoopMode，shape `[M,N,W]`，GM stride `[W,M*W,1]`，UB stride `[(N+p)*W,W,1]`，M=24–56、N=2–6、W=32–64 且按平台缓存行对齐，p=0 或 1，16–56 核 | 294.009388 | 32.296047 | 252.871196 |
| NarrowLoopPayload | 同一 LoopMode 物理描述符，M=24–56、N=2–8、W=16（64 B），p=0 或 1，16–56 个活跃核；仅 FP32 | 531.6421388441569 | 18.86785800658518 | 0.7583841815917162 |

同一描述符可能命中多个条件。实际优先顺序以 `LoadPerf` 和 `LoadApi` 的表达式覆盖顺序为准。LoopMode 分支从原始三轴描述符取值，保留外轴地址推进。估值公式沿用现有 tiling 表达式接口，Codegen API、模板语义和设备同步均保持原有实现。

64 B 行宽没有命中原缓存行对齐分支，会回退 stride 估值。新增分支使用独立训练和留出系数，复用相同公式及描述符检查；不把宽行参数外推到窄行，也不修改 NDDMA 补偿 Load。其他 dtype 和域外保持原路径。核数范围是实测适用域，不按设备型号或设备总核数选择模型。

## 用例和验证

[用例清单](cases.csv)保留原始点 ID、数据角色、设备核数、shape 和 stride。[实测归档](results/v35/pr2921-d68/report.md)记录拟合来源、采集口径和复现入口。宽 pitch 的禁用 L2 结果属于冷读校准，其他模式的结果分别记录。LoopMode 训练与留出独立，稠密 Pad Load 同时核验展平和多行等价。模型单位为每核单次 API cycle，整图和 kernel 的收益另外验收。

UT 入口为 `UTestAscirPerfV2.TestSmallDmaCalibratedDomains`、`TestAlignedSingleRowLoadEstimate`、`TestAlignedLoopLoadDescriptorDomain`、`TestAlignedDenseLoadDomainAndDescriptorEquivalence`、`TestAlignedWideGapLoadEstimate` 和 `TestWidePitch128BLoadEstimate`。

窄行 UT 为 `UTestAscirPerfV2.TestNarrowLoopLoadDescriptorDomain`。独立采样、冻结参数、域外尾块控制及复现索引见 [窄行报告](results/v35/pr2921-d68/report.md)。

上板采集与复现见 [DMA 建模采集工程](../../../../tools/modeling/dma/README.md)。
