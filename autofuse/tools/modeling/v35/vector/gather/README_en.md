# Gather Bank and Mask Component Collection

This project archives the component collection source of the historical
`selectprefuse-gather-heldout-rows-v1`. The input is a CSV with columns `id,role,width,pitch,axis,rows`. The eight
archived holdout points can be generated from the
[existing cases](../../../../../att/modeling/v35/vector/transpose/cases-pr2921.csv); dims are rows x width,
`ub_stride_elements` corresponds to pitch, and axis follows the independent index-branch selection of the original
experiment.

Run the following from the repository root. First set the CANN environment and `NPU_ARCH` according to the actual
installation, and point `work` to a new experiment directory.

```bash
tool=autofuse/tools/modeling/v35/vector/gather
work=$(mktemp -d)
cmake -S "$tool" -B "$work/build" -DASC_DIR="$ASCEND_HOME_PATH/lib64/cmake" -DNPU_ARCH="$NPU_ARCH"
cmake --build "$work/build" -j 4
# factors.csv is provided by the actual descriptor; the output must use a new file.
"$work/build/att_gather" "$work/factors.csv" "$work/verify.jsonl" verify
msprof --application="$work/build/att_gather $work/factors.csv $work/profile.jsonl profile" \
  --output="$work/profile" --aic-mode=task-based --aic-metrics=PipeUtilization
```

The historical protocol uses device 0 and 55 active cores, with three warmups and nine valid samples.
Each point rotates 2/5/8 component calls, and the fit uses the slope of per-core VEC cycles versus call count.
The GM input is reused to isolate Gather and must not be used to calibrate cold-read MTE2.
First confirm that the device supports the experiment core count and hold the shared device lock.
The core count is part of the collection protocol; the product Gather rule is determined by the descriptor.

When the input data comes from existing cases, check axis against the raw evidence; the current compact CSV does not
record this field. Do not guess axis from the ID. Model absolute errors can be checked directly through the
[generic MTE2 CSV revalidation entry](../../mte2/README_en.md). The full Transpose product estimate is still
validated by the targeted `UTestAscirPerfV2` tests in the repository.
