# SuperKernel Sample Guide

## Overview

This directory contains two groups of SuperKernel Python samples. See the `README_en.md` in each sample directory for its supported products and execution instructions.

- `jit/`: builds and runs fused kernels through the SuperKernel JIT interface.
- `aot/`: enables SuperKernel optimization during TorchAir `npugraph_ex` static compilation.

## Directory Structure

```text
examples/
├── run_example.sh                                   # Engineering test batch script
├── jit/
│   ├── example01_super_kernel_base/                 # Basic SuperKernel usage
│   ├── example02_super_kernel_profiling/            # Profiling comparison
│   └── example03_super_kernel_runtime_ascendc_only/ # Minimal AscendC and Runtime sample
└── aot/
    ├── scripts/                                     # Shared Bash functions for AOT samples
    ├── example01_dual_stream/                       # Two streams with NPU events
    ├── example02_sk_options/                        # SuperKernel options
    └── example03_kernel_pybind/                     # Pybind custom operator fusion
```

## Prerequisites

The samples support the following product models:

- Ascend 950PR/Ascend 950DT
- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products

Follow the [source build guide](../../docs/en/build.md) and install the officially released matching
PyTorch and TorchNPU versions. See the
[Ascend Extension for PyTorch user guide](https://www.hiascend.com/document/redirect/pytorchuserguide).
Finally, install the Python dependencies of these samples:

```bash
pip install -r super_kernel/examples/requirements.txt
```

## Sample Summary

Use the table below to route quickly to a sample. Refer to the README in each sample directory for supported products and execution instructions.

| Type | Sample | Documentation and instructions |
| --- | --- | --- |
| JIT | [example01_super_kernel_base](./jit/example01_super_kernel_base/README_en.md) | [README.md](./jit/example01_super_kernel_base/README.md) / [README_en.md](./jit/example01_super_kernel_base/README_en.md) |
| JIT | [example02_super_kernel_profiling](./jit/example02_super_kernel_profiling/README_en.md) | [README.md](./jit/example02_super_kernel_profiling/README.md) / [README_en.md](./jit/example02_super_kernel_profiling/README_en.md) |
| JIT | [example03_super_kernel_runtime_ascendc_only](./jit/example03_super_kernel_runtime_ascendc_only/README_en.md) | [README.md](./jit/example03_super_kernel_runtime_ascendc_only/README.md) / [README_en.md](./jit/example03_super_kernel_runtime_ascendc_only/README_en.md) |
| AOT | [example01_dual_stream](./aot/example01_dual_stream/README_en.md) | [README.md](./aot/example01_dual_stream/README.md) / [README_en.md](./aot/example01_dual_stream/README_en.md) |
| AOT | [example02_sk_options](./aot/example02_sk_options/README_en.md) | [README.md](./aot/example02_sk_options/README.md) / [README_en.md](./aot/example02_sk_options/README_en.md) |
| AOT | [example03_kernel_pybind](./aot/example03_kernel_pybind/README_en.md) | [README.md](./aot/example03_kernel_pybind/README.md) / [README_en.md](./aot/example03_kernel_pybind/README_en.md) |
