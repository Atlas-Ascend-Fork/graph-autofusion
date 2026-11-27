# V1 ATT 模型归档

已归档 [A3 MTE2 Load 采集资料](mte2/load/README.md)和 [A3 MTE3 Store 设计与覆盖索引](mte3/store/README.md)，按 A3 对应 V1 的目录规则存放。各归档引用的 v35 历史参数快照不代表当前 V1 参数或已完成 V1 实现的参数标定及实测验证；Load 的 V1 实现参数对照见其归档。

V1 实现入口为 [api_perf_register/v1](../../gen_model_info/api_perf_register/v1)。V1 参数表沿用空字符串默认注册键；v35 使用 `PerfParamTableV2` 和注册键 `V2`。注册键不作为平台目录名。
