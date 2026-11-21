# v35 ATT 建模工具

本目录仅包含 v35 专用工具；架构参数可配置不代表已支持 V1。

采集和拟合代码集中在这里，设计、用例和精简实测结果归档在 [ATT 模型目录](../../../att/modeling/README.md)。

| 工具 | 用途 |
| --- | --- |
| [NDDMA](nddma/README.md) | Round1–5 采集、拟合、预测，以及单配置实测 |
| [普通 Load / Store](dma/README.md) | v35 FP32 DMA 上板采集、精度校验和复现 |
| [MTE2](mte2/README.md) | Load / NDDMA 已归档数据的误差核对与重拟合 |
| [Gather](vector/gather/README.md) | bank 与 mask 组件上板采集 |
| [IsInf](vector/isinf/README.md) | 单 API 的上板采集与拟合 |
| [GetOpHeadCost](common/op_head_cost/README.md) | 核数相关公共开销的上板采集、证据核对与离线复算 |

以上工具按需手动运行。模板、tiling 和日志定位工具仍在 [ATT Analyze](../../att_analyze/README.md)。


## 归档覆盖与路径

| 本次建模项 | 采集、拟合或验证入口 | 覆盖边界 |
| --- | --- | --- |
| FP32 Load／Store 九分支 | [DMA](dma/README.md)、[CSV 复核](mte2/README.md) | 部分早期采集器仅找到同系列源码，精确重放限制见 DMA README |
| NDDMA 物理 1D、packed NG2、多轴模型 | [Round1–5](nddma/README.md) | 通用描述符采集和拟合；PR 校准域另见模型设计 |
| padded NG2、packed rank3、sparse rank3/4、small INT64 | [CSV 复核](mte2/README.md) | padded NG2 支持重新拟合；其他分支保留冻结预测与原始证据索引，专用历史采集脚本尚未全部归档 |
| Gather bank／mask | [Gather](vector/gather/README.md) | 组件采集源码已归档，完整产品公式由定向 UT 验证 |
| TrueDiv／Rsqrt VF | [VF 设计](../../../att/modeling/v35/vector/vf/design.md) | 复用已有指令参数，无新增实测系数，验证入口为设计页指定的 UT |
| 公共核数开销 | [GetOpHeadCost](common/op_head_cost/README.md) | 两个采集程序与原始 profiling 转换、离线拟合 |
| IsInf | [IsInf](vector/isinf/README.md) | 沿用既有独立工程 |

命令中的仓内路径均从仓库根执行，Python／CMake 的内部默认路径从脚本所在目录解析。
CANN、设备架构、输入数据和实验输出通过参数或环境变量提供。

manifest 的 `case_inventory`、`reproduction` 相对其自身目录解析。
`source_root=external_evidence` 表示 source 相对使用者提供的外部证据根目录，
`source_root=external_runs` 表示相对历史实验 runs 根目录。
源文件 checksum 用于核对历史证据，路径迁移后继续使用同一 checksum。
外部证据位置仅用于溯源，离线误差复核直接读取仓内 cases 和 samples。
