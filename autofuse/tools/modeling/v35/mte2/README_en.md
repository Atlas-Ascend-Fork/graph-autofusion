<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# MTE2 Modeling Revalidation

`calibrate.py` joins ATT-archived cases and measured CSV by case ID and reports the error by model, collection
policy, and data role. Points without a frozen prediction only count toward the sample count. Provide both CSV paths
at runtime.

```bash
mte2=autofuse/att/modeling/v35/mte2
for model in load nddma; do
  python3 autofuse/tools/modeling/v35/mte2/calibrate.py \
    "$mte2/$model/cases.csv" "$mte2/$model/results/pr2921-d68/samples.csv"
done
```

Offline refitting is available for FP32 small-transfer Load, Compact Load, and padded NG2. After installing NumPy and
SciPy, use `--fit` and `--policy`. Only `train` points participate in fitting; holdout and known candidates are
reported separately. Compact refitting may estimate a non-negative stride term; frozen product coefficients should
be accepted together with holdout and candidate-ordering results.

```bash
python3 autofuse/tools/modeling/v35/mte2/calibrate.py \
  "$mte2/load/cases.csv" "$mte2/load/results/pr2921-d68/samples.csv" \
  --fit LoadSmallPayload --policy N

python3 autofuse/tools/modeling/v35/mte2/calibrate.py \
  "$mte2/nddma/cases.csv" "$mte2/nddma/results/pr2921-d68/samples.csv" \
  --fit 'NG2 padded FP32' --policy forward/reverse
```

The inputs are per-core, per-API cycles obtained from the collection. For single-configuration NDDMA device
collection, descriptor prediction, and Round1-5 fitting, see the [NDDMA tool](../nddma/README_en.md). New Load CSV
data must align with the columns of the [Load cases](../../../../att/modeling/v35/mte2/load/cases.csv) and
[measured data](../../../../att/modeling/v35/mte2/load/results/pr2921-d68/samples.csv), and must record its own cache
and address-progression semantics.

The FP32 narrow contiguous source long-axis NDDMA keeps the startup, linear-transfer, and inter-core-contention form.
It uses the number of strided positions instead of the UB payload to avoid underestimating the request cost of unit
axes and extremely narrow contiguous sources. The following command refits only with training points; independent
holdout, known points, and frozen validation are output separately.

~~~bash
archive=autofuse/att/modeling/v35/mte2/nddma
python3 autofuse/tools/modeling/v35/mte2/calibrate.py \
  "$archive/cases-narrow-source.csv" "$archive/results/narrow-source-v1/samples.csv" \
  --fit 'NG2 narrow source FP32' --policy N_fresh_normal
~~~

## Dense narrow-row ordinary Load

`--fit LoadAlignedDenseBulkPayload --policy C_fresh_disable` retains the LoadStore form and fits training rows only. Five independent holdouts and three post-freeze points remain separate. See [cases](../../../../att/modeling/v35/mte2/load/results/dense-bulk-v1/cases.csv), [samples](../../../../att/modeling/v35/mte2/load/results/dense-bulk-v1/samples.csv), and the [report](../../../../att/modeling/v35/mte2/load/results/dense-bulk-v1/report_en.md). Collection reuses the DMA loop binary.
