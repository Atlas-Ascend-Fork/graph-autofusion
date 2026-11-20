# ATT 建模工具

按模型家族组织工具。[V1](v1/README.md) 尚无经过验证的工具；[v35](v35/README.md) 包含 DMA、NDDMA、MTE2 复算、GetOpHeadCost、Gather 和 IsInf。设计、用例和精简结果见[模型归档](../../att/modeling/README.md)。

原六组工具整体移入 `v35/`，其中 `op_head_cost` 位于 `v35/common/op_head_cost`。各工程保持完整，NDDMA 预测器使用的模型 JSON 一并保留。CLI 参数、target 和数据格式不变，调用者需更新脚本路径；不保留重复工程或新增平台分派。使用私有构建/输出目录和既有共用设备锁，不复制 CANN、不保存张量转储。

命令中的路径默认相对仓库根，CMake/Python 内部默认路径相对源码位置解析。实际设备架构、API 头文件版本和模型家族分别记录，不能相互替代。模板、tiling 和日志定位仍使用 [ATT Analyze](../att_analyze/README.md)。
