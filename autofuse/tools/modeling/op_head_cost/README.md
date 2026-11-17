# GetOpHeadCost 采集与复算

[建模方法](../../../att/modeling/common/op_head_cost/design.md)说明两个独立实验和用例划分。
已有实测可以离线复算，仅依赖 Python 3.9+：

```bash
# 从仓库根目录执行。
tool=autofuse/tools/modeling/op_head_cost
data=autofuse/att/modeling/common/op_head_cost/results/v35/pr2921-d68
python3 "$tool/calibrate.py" fit "$data/samples.csv"
```

重新采集需可用的 CANN 编译器（bisheng）、msprof 和至少有 50 个可用 AIV 核的 v35 设备。下列命令使用设备 0；
采集前确保该设备未运行其他任务。用例核数是实验采样点。

```bash
source /path/to/cann/set_env.sh
export NPU_ARCH=dav-3510  # 按实际设备设置
work=$(mktemp -d)
cmake -S "$tool" -B "$work/build" -DASC_DIR="$ASCEND_HOME_PATH/lib64/cmake" -DNPU_ARCH="$NPU_ARCH"
cmake --build "$work/build" -j 8
bash "$tool/collect.sh" "$work/build" "$work/run"
python3 "$tool/calibrate.py" fit "$work/run/samples.csv"
```

采集目录必须不存在；失败时保留现场，换新目录重试。按正、反、反、正顺序采集两个 bench，
检查 launch 与 profiling 的 kernel、核数、顺序和输出回执，丢弃预热记录后生成小体积 `samples.csv`。
离线证据需要保留时，保存该 CSV、工具/环境版本及原始 op_summary 的索引和 SHA-256。
完整 profiling、程序二进制和运行日志留在实验目录。
