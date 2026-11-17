# GetOpHeadCost Collection and Revalidation

The [modeling method](../../../att/modeling/common/op_head_cost/design_en.md) describes the two independent
experiments and the case partition. Existing measurements can be recomputed offline with Python 3.9+ only:

```bash
# Run from the repository root.
tool=autofuse/tools/modeling/op_head_cost
data=autofuse/att/modeling/common/op_head_cost/results/v35/pr2921-d68
python3 "$tool/calibrate.py" fit "$data/samples.csv"
```

Re-collection requires a working CANN compiler (bisheng), msprof, and a v35 device with at least 50 available AIV
cores. The commands below use device 0; make sure it is not running other tasks before collection. The case core
counts are experiment sample points.

```bash
source /path/to/cann/set_env.sh
export NPU_ARCH=dav-3510  # Set according to the actual device
work=$(mktemp -d)
cmake -S "$tool" -B "$work/build" -DASC_DIR="$ASCEND_HOME_PATH/lib64/cmake" -DNPU_ARCH="$NPU_ARCH"
cmake --build "$work/build" -j 8
bash "$tool/collect.sh" "$work/build" "$work/run"
python3 "$tool/calibrate.py" fit "$work/run/samples.csv"
```

The output directory must not already exist; keep the artifacts after a failure and retry in a new directory.
Collect the two benches in forward, reverse, reverse, forward order, check the kernel, core count, order, and output
receipts in launch and profiling, discard warmup records, and generate the compact `samples.csv`. When offline
evidence must be kept, store this CSV, the tool/environment versions, and the index and SHA-256 of the raw
op_summary. Full profiles, program binaries, and run logs stay in the experiment directory.
