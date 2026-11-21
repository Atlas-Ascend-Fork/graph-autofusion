# IsInf On-Device Performance Validation Tool

Single-API on-device validation code, used as the first step of IsInf performance modeling: measuring execution cycles of each dtype branch under different `count` values, providing measured data for fitting `cycles = a * repeat_time + b`.

## Source and Branch Basis

- API declaration: `asc-devkit/include/adv_api/math/is_inf.h` (official API, v35 target arch is `__NPU_ARCH__ == 3510`).
- API implementation: `asc-devkit/impl/adv_api/detail/math/isinf/is_inf_common_impl.h` `IsInfImplVF`.
- Codegen invocation: `autofuse/v35/codegen/reg_api_call/unary_bitwidth_change_api_call_v2.cpp`, the output always goes through `ReinterpretCast<bool>`.
- Device: dav-3510 (Ascend 950PR/950DT series); other architectures such as 910B are not in the support list of `is_inf.h` and cannot run.

The 4 covered source dtype branches:

| combo | DstT | SrcT | repeat_elm | Source difference |
| --- | --- | --- | --- | --- |
| `bool_float` | bool | float | 64 | MaskPack×2 + tail MaskPack×2, CompareScalar(uint32)×2 |
| `bool_half` | bool | half | 128 | MaskPack×1 + tail MaskPack×1, CompareScalar(uint16)×2 |
| `float_float` | float | float | 64 | No MaskPack, Select outputs directly |
| `half_half` | half | half | 128 | No MaskPack, Select outputs directly |

Among them, `bool_float` / `bool_half` are the actual v35 codegen paths.

## Measurement Criteria

- Single block execution; the kernel uses `TPipe` + `TQue` (VECIN/VECOUT) to allocate UB, consistent with the codegen-generated kernel structure; the `IsInf` form without tmpBuffer internally relies on `PopStackBuffer` which requires a TPipe context and cannot be replaced with `LocalMemAllocator`.
- **The bench does no internal timing**: performance data is collected entirely by msprof op_summary in npu mode (see batch collection instructions).
- **No warmup**: msprof op_summary collects the AIV cycles of the entire kernel; the total number of API invocations inside the kernel must be exactly equal to `loop_count` (extra invocations would inflate the parsed value by a multiple); cold-start overhead (icache miss, etc.) is counted into the fitting intercept b.
- Inside the API, `repeat_time = ceil(count / repeat_elm)`, `repeat_elm = 256 / sizeof(SrcT)`.
- Inputs contain +Inf/-Inf/normal values; each run also validates the correctness of the first `count` outputs.

## Build

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..
make -j8
```

CPU simulation for correctness validation (no cycle output):

```bash
cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..
make -j8
```

## Run

### Single-Point Manual Test

```bash
./isinf_perf_bench <combo> <count> [loop_count]
# Example:
./isinf_perf_bench bool_float 1024 1000
```

Output (CSV):

```text
VERIFY,PASS,bool_float,1024
RESULT,bool_float,1024,64,16,1000
```

`RESULT` column meaning: `combo, count, repeat_elm, repeat_time, loop_count` (parameter echo, no performance field; performance data is collected by msprof op_summary).

### Batch Collection (run_bench.py, cases read from cases.csv)

Test cases are defined in `cases.csv` in this directory, with columns `combo,count,repeat,loop_count` (`repeat` is repeat_time, `ceil(count/repeat_elm)`, the tool validates consistency). Current case matrix: 8 aligned points per branch (`repeat_time ∈ {1,2,4,8,16,32,64,128}`, `count = rt * repeat_elm`), 32 cases in total; `loop_count` is fixed to 1 (a single API invocation inside the timing loop, directly measuring the full single-call cycle).
Each point is repeatedly executed 5 times and the median is taken.

```bash
# CPU simulation for the full pipeline (locally testable, correctness only, loop_count auto-converges)
python3 run_bench.py --run-mode cpu --cases mini_cases.csv

# Full case measurement on the target device (dav-3510); npu mode automatically enables profiling:
# each case is launched via msprof --parse=on --application="{exe} {combo} {count} {loop_count}",
# profiling data is written isolated by batch:
#   {profile-dir}/{run-id}/{combo}_{count}/PROF_{timestamp}_*/
# run-id identifies one execution batch (--run-id, default run_YYYYmmdd_HHMMSS); multiple PROF
# files from multiple runs all land in the current batch directory, and parsing only takes
# the data of this run-id
source ${ASCEND_HOME_PATH}/set_env.sh
python3 run_bench.py --output samples.csv

# Extra measurements such as tail-block extrapolation validation (use a separate csv,
# do not modify the formal case set)
python3 run_bench.py --skip-build --cases extra_cases.csv
```

In npu mode, after each case finishes testing, `op_summary_*.csv` is searched recursively in its profiling subdirectory, rows whose `OP Type` value contains the tested API name (in this tool `API_NAME = "isinf"`, case-insensitive) are taken, the three columns `Block Num`, `aiv_total_cycles`, and `aiv_vec_ratio` are read, and
`aiv_total_cycles / Block Num * aiv_vec_ratio` is computed as the measured value of this case (median over multiple runs), refreshing the `measured_cycles` of samples.csv (in npu mode the measured value is authoritative from msprof parsing; the internal clock of the bench is no longer used as the measured source).

In npu mode, the device environment is self-checked via `npu-smi` at startup: if the command is available, device info is printed (to confirm the target form and archive environment evidence); if the command does not exist or fails, it does not block, and execution continues with `--arch` (dav-3510) by default.

`profiling/` is not committed to git but **must be preserved after testing** (raw measured evidence, indexed and associated by `results/<platform>/<run_id>/manifest.json` during archiving); no validation or cleanup operation may delete it.

The output `samples.csv` contains all cases in cases.csv (line-by-line consistent, failed points also occupy rows), with two appended columns:

| Column | When filled | Description |
| --- | --- | --- |
| `measured_cycles` | Filled when measured by this tool | Median of the **total cycles** of the timing loop; divided by loop_count during fitting to get the single-call cycle. Left empty for failed/invalid points |
| `model_predict_cycles` | Backfilled after fitting a, b and completing the perf code | Computed as `a * repeat + b`, used for model-vs-measurement alignment archiving |

In npu mode, a measured cycles value of 0 is judged INVALID; the exit code is non-zero if FAILED/INVALID points exist.

### Fitting (fit_curve.py)

Reads the `loop_count,repeat,measured_cycles` columns of samples.csv and fits grouped by combo (source branch): first normalize `y = measured_cycles / loop_count` (single-call cycle), then least-squares fit
`y = a * repeat + b`; each branch independently gets a set of a, b with R2 and per-point residuals, written to `report.md` in this directory.

```bash
python3 fit_curve.py                        # reads tools/samples.csv, writes tools/report.md
python3 fit_curve.py --result r.csv --report report.md
```

`report.md` is always generated and kept: branches with insufficient data are annotated with the reason in the report (other branches are not interrupted), and the exit code is non-zero when unfitted branches exist. Rows with empty or non-positive `measured_cycles` are skipped (0-value placeholder data does not participate in fitting). The current report.md is based on CPU simulation placeholder data (all 0), and all branches are in the "unfitted" state; after on-device measurement, re-run to generate the formal fitting results.

## Fitting Usage

For each combo, scan multiple `count` groups (recommended to cover repeat_time 1~256 and tail-block points), collect the `RESULT` rows, then perform least-squares fitting on
`cycles_per_call = a * repeat_time + b` to obtain the `a` and `b` of each branch.

## Verified Status

- dav-3510 NPU mode compilation passed; the full pipeline of `run_bench.py` in both npu/cpu modes (build, matrix, run, collection, CSV) verified.
- Under CPU simulation (`CMAKE_ASC_RUN_MODE=cpu`), all 4 combos and tail-block scenarios (count=1/63/65/129, etc.) returned `VERIFY,PASS`.
