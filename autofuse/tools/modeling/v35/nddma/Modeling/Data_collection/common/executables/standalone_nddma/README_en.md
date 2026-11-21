<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# standalone_nddma

[简体中文](README.md)

This executable is the round1 NDDMA sensitivity-scan harness.

This harness measures v35 NDDMA MTE2 loads only; it does not store UB data back to GM.
Set `NPU_ARCH` for the actual platform before building, or pass `-DNPU_ARCH=...`.
For CPU debug, also specify the installed `-DNDDMA_CPU_TARGET=tikicpulib::<target>`.

Build one binary per sensitivity group:

```bash
cmake -S . -B build/bytes_shape -DNDDMA_STAGE=2
cmake --build build/bytes_shape --target demo_nddma -j 8
```

Run with a generated factor csv:

```bash
./build/bytes_shape/demo_nddma --factor-csv=/path/to/v1_round1_bytes_shape.csv
```

`NDDMA_STAGE` is only a compile-time label for the sensitivity group:

- `1`: dtype/dim
- `2`: bytes/shape
- `3`: input stride
- `4`: output stride
- `5`: alignment
- `6`: input stride zero/broadcast

The harness reads `dtype`, `dim`, `output_dims`, `output_stride`, `input_stride`, offsets, and repeat count from the factor csv. It dispatches `DataCopyNddma<T, dim>` through template instances for `dim=1..5`. There is no empty stage and no outer-loop modeling path; every measured case repeatedly executes the NDDMA API.

The default API header comes from this repository at
`autofuse/v35/ascendc/api_regbase/datacopy_nddma.h`. Override it with
`-DREGBASE_INCLUDE_DIR=/path/to/api_regbase` to use an external checkout.
See the [model README](../../../../../README_en.md#dependencies-and-reproducibility)
for CANN requirements and the historical `NdDmaDci` measurement difference.
