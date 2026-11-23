# Gather bank 与掩码组件采集

本工程归档历史 `selectprefuse-gather-heldout-rows-v1` 的组件采集源码。
输入为 CSV，列为 `id,role,width,pitch,axis,rows`。归档八个留出点可从
[现有用例](../../../../../att/modeling/v35/vector/transpose/cases-pr2921.csv)生成，dims 为 rows×width，
ub_stride_elements 对应 pitch，axis 沿用原实验的独立索引分支选择。

从仓库根执行以下命令，先按实际安装设置 CANN 环境和 NPU_ARCH，work 指向新实验目录。

```bash
tool=autofuse/tools/modeling/v35/vector/gather
work=$(mktemp -d)
cmake -S "$tool" -B "$work/build" -DASC_DIR="$ASCEND_HOME_PATH/lib64/cmake" -DNPU_ARCH="$NPU_ARCH"
cmake --build "$work/build" -j 4
# factors.csv 由实际描述符提供，输出必须使用新文件。
"$work/build/att_gather" "$work/factors.csv" "$work/verify.jsonl" verify
msprof --application="$work/build/att_gather $work/factors.csv $work/profile.jsonl profile" \
  --output="$work/profile" --aic-mode=task-based --aic-metrics=PipeUtilization
```

历史协议使用设备 0、55 个活跃核，三个预热和九个有效样本。
每点轮换 2／5／8 次组件调用，按每核 VEC cycle 的调用次数斜率拟合。
GM 输入复用以隔离 Gather，不能用于标定冷读 MTE2。
先确认设备支持该实验核数并持有实验共用的设备锁。
该核数是采集协议的一部分，产品 Gather 规则按描述符判断。

输入数据来自既有用例时，应从原始证据核对 axis，现有精简 CSV 没有记录该字段。
不得根据编号猜测 axis。模型绝对误差可以直接通过
[MTE2 的通用 CSV 复核入口](../../mte2/README.md)核验。
完整 Transpose 的产品估值仍由仓内 `UTestAscirPerfV2` 定向测试验证。
