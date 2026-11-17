<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# MTE2 建模复验

`calibrate.py` 按 case ID 关联 ATT 归档的用例和实测 CSV，按模型、采集口径和数据角色输出误差。没有冻结预测的点只计入样本数。运行时提供两份 CSV 的路径。

```bash
mte2=autofuse/att/modeling/mte2
for model in load nddma; do
  python3 autofuse/tools/modeling/mte2/calibrate.py \
    "$mte2/$model/cases.csv" "$mte2/$model/results/v35/pr2921-d68/samples.csv"
done
```

可离线重拟合 FP32 小搬运 Load、Compact Load 和 padded NG2。安装 NumPy、SciPy 后使用 `--fit` 和 `--policy`。只有 `train` 点参与拟合，留出和已知候选分别报告。Compact 重新拟合时允许估计非负 stride 项，产品分支的冻结系数应结合留出和候选排序验收。

```bash
python3 autofuse/tools/modeling/mte2/calibrate.py \
  "$mte2/load/cases.csv" "$mte2/load/results/v35/pr2921-d68/samples.csv" \
  --fit LoadSmallPayload --policy N

python3 autofuse/tools/modeling/mte2/calibrate.py \
  "$mte2/nddma/cases.csv" "$mte2/nddma/results/v35/pr2921-d68/samples.csv" \
  --fit 'NG2 padded FP32' --policy forward/reverse
```

这里输入的是采集后得到的每核、每次 API cycles。NDDMA 单配置上板采集、描述符预测和 Round1–5 拟合见 [NDDMA 工具](../nddma/README.md)。新采集的 Load CSV 需按 [Load 用例](../../../att/modeling/mte2/load/cases.csv)和[实测数据](../../../att/modeling/mte2/load/results/v35/pr2921-d68/samples.csv)列对齐，并记录独立的缓存和地址推进口径。

FP32 窄连续源长轴 NDDMA 沿用启动、线性搬运和核间竞争形式。
以 strided position 数代替 UB payload，避免单位轴与极窄连续源的请求成本被低估。
以下命令只使用 train 点重拟合，独立留出、已知点和冻结验证分别输出。

~~~bash
archive=autofuse/att/modeling/mte2/nddma
python3 autofuse/tools/modeling/mte2/calibrate.py \
  "$archive/cases-narrow-source.csv" "$archive/results/v35/narrow-source-v1/samples.csv" \
  --fit 'NG2 narrow source FP32' --policy N_fresh_normal
~~~
