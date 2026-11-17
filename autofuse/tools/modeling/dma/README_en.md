# v35 FP32 Load / Store Device Collection

This project uses input descriptors archived by the existing ATT models and supports ordinary Load, single-layer
hardware LoopMode Load, and small contiguous Store. The original model formulas, sampling roles, and measured data
are in the [Load archive](../../../att/modeling/mte2/load/results/v35/pr2921-d68/report_en.md) and
[Store archive](../../../att/modeling/mte3/store/results/v35/pr2921-d68/report_en.md). Training, holdout, known
candidates, and out-of-domain controls follow the archived labels; write re-collection results to a new directory.

## Build

Use a Linux environment that supports v35 and has at least 56 available AIV cores on device 0, with CANN 9.3.0,
bisheng, msprof, CMake 3.20+, Python 3.9+, and flock installed. Device 0 should be idle, and all collection tasks
must use the same device lock. The following example paths are executed from the repository root:

```bash
source /path/to/cann/set_env.sh
tool=autofuse/tools/modeling/dma
archive=autofuse/att/modeling
work=/path/to/dma-calibration
lock=/path/to/device0.lock
mkdir -p "$work"
cmake -S "$tool" -B "$work/build" -DASC_DIR="$ASCEND_HOME_PATH/lib64/cmake" -DNPU_ARCH=dav-3510
cmake --build "$work/build" -j 4
```

The repository RegBase headers are used by default. For reproduction, pass
`-DAUTOFUSE_ROOT=/path/to/frozen-source` to point to the source frozen at measurement time, and record the SHA-256
of `datacopy.h`, the binary, and the input manifest. One binary can execute all descriptors in the selected batch.

## Cases and Collection Protocol

| Model | CSV policy | Target program | Evidence and collection version |
| --- | --- | --- | --- |
| LoadSmallPayload | `N`, `C` | `steady` | S01 original short-sequence results reuse the existing archive. The project uses the same-series long-call validation version |
| LoadSmallPayloadCompact | `N`, `C` | `steady` | Same as S01. The input uses Compact Load and preserves the GM row gap through the original descriptor |
| LoadAlignedSingleRow | `N` | `steady` | S04 OneRec single row. Uses the same-series long-call version; original training/holdout errors reference the archive |
| LoadAlignedWideGap | `fresh` | `steady` | S05 0238 independent holdout. Re-collects the large-pitch shape with `C`; the sampling sequence differs from the historical cold-read batch |
| LoadWidePitch128B | `C` | `steady` | S06 0254 narrow-row large pitch. Keeps the existing cold-read report; the project can re-collect the descriptor |
| LoadAlignedDensePayload | `N-forward`, `C-forward`, `N-reverse` | `stability` | S03 original collection version, 32 warmups + 64 valid samples. Flattened input is converted to an equivalent one-row DMA |
| LoadAlignedLoopPayload | Same as above | `stability` | S02 original holdout collection version, 32 warmups + 64 valid samples |
| LoadNarrowLoopPayload | Same as above | `loop` | S07 original collection version, 16 warmups + 16 valid samples |
| StoreSmallPayload | `normal`, `bypass` | `store` | S01, re-collected with the same-series Store source, 3 warmups + 10 valid samples for each of 4/16 calls |

`N` rotates GM addresses across cores, calls, and launches and uses the NORMAL L2 hint; `C` additionally disables the
L2 hint. LoopMode uses the raw three-axis descriptor. Store uses `DataCopyPadExtend<Normal>` with a single row and
zero gap, rotates the GM address by core, repetition, and sample, and validates the written data and protection
region. The Store list includes 7-core and >8 KiB archived control points; the product applicable domain is in the
Store design document.

The LoopMode short and long windows come from the historical `matrix17-int64-contiguous-slope-v1` and
`matrix17-fp32-stability-v3`; this project shares one source and selects the window through a compile definition.
Ordinary Load comes from `small-fp32-load-steady-domain-v1`, with 16/32/64/128 calls per group. Store comes from the
same-series `transpose0139-wide-store-mte3-v1` collector; re-collection is limited to small contiguous layouts and
allocates a separate aligned region for each written sample. Exact source snapshots of the original S01/S04/S05/S06
ordinary Load and S01 Store were not found in the existing evidence directory. The substitute collections are used
only to re-collect the same descriptors; historical fitting values continue to reference the archived original results.

## Examples

Generate collection inputs from the existing lists. `--role train`, `--role holdout`, and similar filters are
supported and can be repeated. Add `--limit 1` to check only the entry. There is no limit for full collection.
Load profiles are kept separately for different call counts. The following uses narrow-row LoopMode and contiguous
Store as examples:

```bash
python3 "$tool/prepare_cases.py" "$archive/mte2/load/cases.csv" \
  LoadNarrowLoopPayload N-forward "$work/narrow-N.csv"
ATT_HOST_THREADS=4 ATT_FIXED_POOL_GIB=4 bash "$tool/collect.sh" \
  loop "$work/build" "$work/narrow-N.csv" N "$lock" "$work/narrow-N"

python3 "$tool/prepare_cases.py" "$archive/mte3/store/cases.csv" \
  StoreSmallPayload normal "$work/store-normal.csv"
bash "$tool/collect.sh" store "$work/build" "$work/store-normal.csv" normal "$lock" "$work/store-normal"
```

Switch the model, policy, program, and cache mode in the table with the same commands. To reproduce `N-reverse`, keep
the archived order of that policy in the file and set the collection mode to `N`. Large pitch consumes more HBM, so
`steady` can raise `ATT_FIXED_POOL_GIB` according to the measured protocol; for example, S06 uses a 24 GiB input
pool. The collector checks available device memory and descriptors, and uses the corresponding budget only when the
next group has enough space.

Use a new directory for each collection and keep the accuracy receipt, launch JSONL, and msprof op_summary. After
checking the launch ID, core count, sample index, warmups, cache hint, and actual kernel, compute the per-core
pipeline cycles as `aiv_total_cycles * aiv_mte2_ratio / Block Num` or
`aiv_total_cycles * aiv_mte3_ratio / Block Num`. The main S07 Load result uses the K=1 median; old short sequences
and long-call validation use their own reported call windows. Store uses the difference between the 16-call and
4-call cycle medians divided by 12. The [existing fitting revalidation tool](../mte2/README_en.md) can directly
check the archived Load data.

## FP32 Narrow Contiguous Source NDDMA

Raw descriptors, training, and independent validation are in the
[cases](../../../att/modeling/mte2/nddma/cases-narrow-source.csv) and
[results](../../../att/modeling/mte2/nddma/results/v35/narrow-source-v1/report_en.md).
Use the same queues, address isolation, L2 hint, synchronization, and protection-region checks, and fit the MTE2
slope by call count. The same program reads the list in batch and selects roles as needed.

Run the following from the repository root, continuing the `tool`, `work`, and `lock` settings above.

~~~bash
archive=autofuse/att/modeling/mte2/nddma
python3 "$tool/prepare_cases.py" "$archive/cases-narrow-source.csv" \
  'NG2 narrow source FP32' N_fresh_normal "$work/nddma.csv" --role train --role holdout
ATT_HOST_THREADS=4 ATT_FIXED_POOL_GIB=49 bash "$tool/collect.sh" \
  nddma "$work/build" "$work/nddma.csv" N "$lock" "$work/nddma-run"
~~~

nddma uses 1/2/4/10 calls, with 3 warmups and 10 valid samples per group.
The frozen long-axis validation uses `--role frozen_fit_validation` to generate another list, selects the
`nddma_long` program with 1/2/3/4 calls, and uses 1 warmup and 4 valid samples per group.
The two measurement windows are counted separately and are not mixed in fitting.
The 49 GiB input pool covers large-span fresh GM re-collection; the tool also checks actual free HBM and the check
output budget. Smaller descriptors can reduce this pool parameter.
