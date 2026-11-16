# 诊断矩阵

| 问题 | 查看位置 | 证据与边界 |
| --- | --- | --- |
| 原图与融合规模 | Graph 顶部统计及 scope | 分别计算原始节点、融合成员、Kernel 成员、SK 和 scope，不以净减少节点替代融合数 |
| 更新后变化 | Graph Update | 真实 after 快照；Log 仅有修改记录时标为部分还原 |
| scope 内排布 | Graph scope 浮动面板 | 最终 scope 的原始 stream 和 task 排布 |
| SK 内部排布 | Graph SK 下钻 | AIC/AIV task 队列和同步信息，无真实时间比例 |
| hang / coredump | DFX 运行诊断 | 结论、定位对象、counter/PC/op_trace 和折叠证据；证据不足时不给正常结论 |
| 解析失败或两侧不一致 | DFX 分析数据质量 | 来源、共同字段差异、单侧信息、完整度、回退及下一步 |
| 日志旧格式或新 JSON 不支持 | DFX 分析数据质量 | 按能力标为 partial/unavailable，Both 使用有效来源，不静默拼图 |

真实耗时分析不属于当前报告能力。
