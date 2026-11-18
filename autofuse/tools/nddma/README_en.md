# NDDMA Performance Modeling

This directory contains the NDDMA performance modeling workflow, including
data collection scripts, fitted model parameters, diagnostic plots, and ad-hoc
prediction tools.

## Functionality

The model is built in five rounds:

| Round | Purpose | Output model |
| --- | --- | --- |
| Round1 | 1D contiguous single/multi-core baseline | `Modeling/Ana/round1/round1_1d_single_multi_core_model.json` |
| Round2 | 1D non-contiguous GM/UB correction and multi-core `rho` | `Modeling/Ana/round2/round2_1d_noncontiguous_model.json` |
| Round3 | 2D/3D/4D/5D unified multidimensional expansion | `Modeling/Ana/round3/round3_multidim_model.json` |
| Round4 | 2D transpose / Round5 `N_G2` UB-contiguous specialization | `Modeling/Ana/round4/round4_2d_ub_contiguous_ng2_model.json` |
| Round5 | 3D specialization: freeze Base, refit `N_G/N_GU/rho` | `Modeling/Ana/round5/round5_3d_specialized_model.json` |

The high-level modeling notes are in:

```text
DOC/nddma_performance_modeling.md
```

## Code Structure

```text
DOC/
  *.md                         Model documents and workflow notes

Modeling/Ana/round1/ ... round4/
  *.json                       Model parameters; also the default fitting outputs

Modeling/Data_collection/
  common/executables/          Standalone NDDMA profiling harness
  common/tools/                Ad-hoc run and prediction tools
  round1/scripts/              Round1 collect/fit/draw/e2e scripts
  round2/scripts/              Round2 collect/fit/draw/e2e scripts
  round3/scripts/              Round3 collect/fit/draw/e2e scripts
  round4/scripts/              Round4 collect/fit/draw/e2e scripts
  round5/scripts/              Round5 3D collect/fit/draw/e2e/predict scripts
```

Generated analysis outputs are written under `Modeling/Ana/round*/` by default.

## Tools

Predict one arbitrary-dimensional config with the unified Round3 model:

```bash
python3 Modeling/Data_collection/common/tools/predict_nddma_ad_hoc.py \
  12x32x32 512x65536x2097152 1024x32x1 72 int32_t \
  --kernel-repeat 72
```

Predict one 2D transpose / Round4 `N_G2` config:

```bash
python3 Modeling/Data_collection/common/tools/predict_nddma_2d_transpose_ad_hoc.py \
  63x2 1x2 2x1 8 int16_t
```

Run one config through the standalone harness and profiling flow:

```bash
python3 Modeling/Data_collection/common/tools/run_nddma_ad_hoc.py \
  63x2 1x2 2x1 8 int16_t \
  --msprof-bin "$(which msprof)"
```

The prediction tools read parameters from `Modeling/Ana/round*/*.json`; they do not require
refitting.

## Refit Workflow

Round1/2 now match the HW_GE_ATT contiguous/non-contiguous fitting workflows.
Round3 aligns the multidimensional evaluation grids and uses the unscaled formula
without d_0 (48368 configurations). Round4 aligns the HW Round5 D/F/G N_G2
training grid (13374 configurations) and ordinary least-squares fitting.
Round5 uses the same 3D formula without d_0, keeps Round2 Base parameters fixed,
and refits the ten N_G/N_GU/rho coefficients using 3D measurements.
Its 178661 collection configurations cover 1/2/4/8/16/32/56 cores;
see `DOC/round5_3d_specialized_model.md` for fitting stages and requirements.
Round2 fitting requires NumPy (`python3 -m pip install numpy`). Its aligned grid
contains 7146 configurations; see `DOC/round2_1d_noncontiguous_model.md`.
The checked-in `Modeling/Ana/round*/*.json` files remain historical measurement snapshots and
need regeneration with the aligned measurements before predictions reflect these changes.

Run the commands below from `autofuse/tools/nddma/`. Full collection requires a working Ascend
profiling environment and `msprof`.

Round1:

```bash
python3 Modeling/Data_collection/round1/scripts/e2e.py all \
  --msprof-bin "$(which msprof)"
```

Round2:

```bash
python3 Modeling/Data_collection/round2/scripts/e2e.py all \
  --msprof-bin "$(which msprof)"
```

Round3:

```bash
python3 Modeling/Data_collection/round3/scripts/e2e.py all \
  --msprof-bin "$(which msprof)"
```

Round4:

```bash
python3 Modeling/Data_collection/round4/scripts/e2e.py all \
  --msprof-bin "$(which msprof)"
```

Round5 (3D specialization):

```bash
python3 Modeling/Data_collection/round5/scripts/e2e.py all \
  --msprof-bin "$(command -v msprof)"
```

Each round can also be run step by step:

```bash
python3 Modeling/Data_collection/roundX/scripts/e2e.py collect --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/roundX/scripts/e2e.py fit
python3 Modeling/Data_collection/roundX/scripts/e2e.py draw
```

Dependency chain:

```text
Round2 fit inherits Round1 model parameters.
Round3 fit inherits Round2 model parameters.
Round4 fit inherits Round2 model parameters and only fits N_G2.
Round5 fit freezes Round2 Base and refits N_G/N_GU/rho using only 3D data.
```

To refit from existing measurement CSVs without recollecting, run the rounds
in dependency order from this directory. Replace the CSV paths with actual
measurements; the first two commands generate the inherited models used below.
The four model JSONs are checked in under `Modeling/Ana/round1/` through `round4/`.
Refitting overwrites these files; ad-hoc tools read the updated parameters on the next run.
Other generated outputs remain ignored.

```bash
python3 Modeling/Data_collection/round1/scripts/e2e.py fit --measurement-csv /path/to/round1.csv
python3 Modeling/Data_collection/round2/scripts/e2e.py fit --measurement-csv /path/to/round2.csv --round1-model Modeling/Ana/round1/round1_1d_single_multi_core_model.json
python3 Modeling/Data_collection/round3/scripts/e2e.py fit --measurement-csv /path/to/round3.csv --round2-model Modeling/Ana/round2/round2_1d_noncontiguous_model.json
python3 Modeling/Data_collection/round4/scripts/e2e.py fit --measurement-csv /path/to/round4.csv --round2-model Modeling/Ana/round2/round2_1d_noncontiguous_model.json
python3 Modeling/Data_collection/round5/scripts/e2e.py fit --measurement-csv /path/to/3d_measurements.csv --round2-model Modeling/Ana/round2/round2_1d_noncontiguous_model.json
```

For an isolated fit in a clean checkout, explicitly select a checked-in snapshot:

```bash
python3 Modeling/Data_collection/round2/scripts/e2e.py fit --measurement-csv /path/to/round2.csv --round1-model Modeling/Ana/round1/round1_1d_single_multi_core_model.json
python3 Modeling/Data_collection/round4/scripts/e2e.py fit --measurement-csv /path/to/round4.csv --round2-model Modeling/Ana/round2/round2_1d_noncontiguous_model.json
```

Round3 and Round5 accept the same `--round2-model` option. Snapshot-based fits
inherit historical coefficients; they are not a replacement for refitting the
upstream rounds from matching measurements.

All checked-in JSON snapshots declare `snapshot.metrics_valid=false`. Their
stored counts/errors describe historical datasets (for example Round2: 7221,
Round4: 8858), not the current grids (7146 and 13374). Prediction tools warn when
loading these snapshots. The stored metrics must not be reported as validation
of the current formulas or aligned parameters.

Use `--dry-run` on any `e2e.py` command to print the invoked subcommands without
executing collection or fitting.

## Dependencies and reproducibility

The modeling sources were imported from NDDMA commit `4232efff8533a44ed6f168a4577b0681f05879a0`.
Python fitting, prediction and plotting do not require GE sources. Install NumPy
for the fitting workflows and tests (`python3 -m pip install numpy`).
The collection harness requires a compatible CANN Toolkit (ASC compiler, ACL and
runtime libraries), an Ascend device and `msprof`; source the Toolkit environment
before collecting data.

By default the harness uses this repository's
`autofuse/v35/ascendc/api_regbase/datacopy_nddma.h`. No ge-develop checkout is
required. To reproduce measurements with an external GE checkout, use a separate
build directory and override the existing CMake option:

```bash
cmake -S Modeling/Data_collection/common/executables/standalone_nddma \
  -B build/external-ge \
  -DREGBASE_INCLUDE_DIR=/path/to/ge-develop/compiler/graph/optimize/autofuse/v35/ascendc/api_regbase
cmake --build build/external-ge --target demo_nddma -j 8
```

The collection scripts (`round1` through `round5` `collect.py`) and
`common/tools/run_nddma_ad_hoc.py` also accept
`--cmake-extra-arg=-DREGBASE_INCLUDE_DIR=/path/to/api_regbase`.
Pass this option directly to these scripts; the `e2e.py` wrappers do not forward it.
Use separate output directories for different API implementations.

The imported GE header called `AscendC::NdDmaDci()` before `AscendC::DataCopy`;
the current graph-autofusion header does not. Historical model JSON files are
preserved as snapshots, not validation of the current implementation. Recollect
and refit before using them as calibrated parameters for this repository.
Record the graph-autofusion/GE revision, selected header path and hash, CANN
version, device, collection command and fitting inputs with each result, following
[the modeling archive convention](../../att/modeling/README.md).
Generated measurements, profiling logs and binaries are not source artifacts.

Run the portable model tests from this directory:

```bash
python3 -m unittest discover -s Modeling/tests -v
```

Reference parity tests additionally require `HW_NDDMA_MODELING` pointing to the
external reference `Modeling` directory; otherwise those tests are skipped.

## Collection scope and platform selection

The harness targets the v35 NDDMA API and performs MTE2 loads only. The MTE3
store path, output GM allocation and `--enable-store` option are removed. New
factor CSVs omit `enable_store`; extra columns in old CSVs do not enable stores.

Set `NPU_ARCH` in the environment before device collection (for example,
`export NPU_ARCH=dav-3510` for a matching device/Toolkit), or pass
`--npu-arch` to `e2e.py`, `collect.py` or the ad-hoc runner.
The explicit option overrides the environment; e2e forwards it only for `all`
and `collect`. Missing architecture stops actual collection before file writes
or compilation and prints setup instructions. Collection/ad-hoc dry-runs warn
and continue; fitting and plotting do not require an architecture. There is no hardcoded
architecture default. CPU debug requires an installed v35 tikicpulib target via
`--cmake-extra-arg=-DNDDMA_CPU_TARGET=<target>`.
All collection/ad-hoc `--dry-run` commands leave the filesystem unchanged,
including when combined with `--generate-only`.
