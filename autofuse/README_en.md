# Autofuse

English | [简体中文](./README.md)

## Introduction

AutoFuse is an automatic fusion framework based on Ascend C. It supports automatic fusion scope identification, automatic operator code generation, Auto Tiling optimization, dynamic shape, mixed precision, and other features. In algorithm networks, a large number of Vector computations may cause substantial memory transfers between Vector computations, resulting in Memory Bound issues. AutoFuse automatically fuses multiple operators into a single operator, reducing the number of operators and memory transfers in the network. This alleviates Memory Bound issues, unleashes Ascend computing power, and improves model execution performance.

For details, refer to [AutoFuse Automatic Fusion](https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/autofuse).

## Autofuse Directory Structure

```text
autofuse/
├── ascendc                # Ascend C API definitions
├── ascir                  # AscIR operator registration
├── att                    # Automatic tiling generation module
├── cmake                  # CMake script files
├── codegen                # Kernel code generation module
├── common                 # Common utility methods
├── compiler               # External API interfaces
├── examples               # Example scripts demonstrating typical usage
├── graph_metadef          # Basic graph interfaces
├── inc                    # Interfaces provided for GE
├── optimize               # Scheduling and partitioning module
├── scripts                # Script directory
├── tests                  # Test cases and test framework
├── tools                  # Debugging and analysis tools
├── v35                    # Ascend 950 chip-related optimizations
├── CMakeLists.txt         # CMake configuration file
├── blacklist.txt          # Project configuration file
├── README.md              # Chinese documentation
└── README_en.md           # English documentation
```

## Build and Installation

Refer to [Build Instructions](../docs/en/build.md).

## On-Device Verification Guide

Users who want to experience the functionality and performance of AutoFuse on Ascend devices can first refer to [Quick Installation](../docs/en/quick_install.md) to prepare the environment. Both developers without Ascend devices and developers who already have Ascend devices can quickly set up the environment. On this basis, follow the previous [Build and Installation](../docs/en/build.md) instructions to incrementally install the CANN package compiled from the graph-autofusion repository.

AutoFuse currently provides sample use cases for both PyTorch and TensorFlow, with support for additional frameworks planned in the future. Refer to the corresponding documentation based on your actual use case to set up the environment and run the samples:

- [PyTorch Scenario Use Cases](./examples/pytorch/README_en.md)
- [TensorFlow Scenario Use Cases](./examples/tensorflow/README_en.md)

The following sections use the PyTorch samples to explain how to set up the environment, run the samples, and use profiling data to examine the resulting kernel performance.

### Install Dependencies

#### Install torch_npu

```bash
pip3 install numpy
pip3 install pyyaml
pip3 install setuptools
```

To use a `torch_npu Daily` environment that matches the AutoFuse AscendC backend, install `torch_npu` with the repository's [PyTorch environment setup script](../scripts/env_install/pytorch/setup_torch_npu_daily.sh). Installing `torch_npu` directly from PyPI is not recommended.

Run the following command from the Graph-AutoFusion repository root:

```bash
bash scripts/env_install/pytorch/setup_torch_npu_daily.sh
```

#### Sample Runtime Dependencies

When you run the [PyTorch samples](./examples/pytorch/README_en.md), fused operator code generation invokes generated scripts such as `asc_tilling.py`. These scripts require Python packages including `decorator`, `scipy`, and `attr` (provided by `attrs`), which are not included in the `numpy`, `pyyaml`, and `setuptools` installation above.

You may encounter an error such as:

```text
RuntimeError: Failed to build ascend kernel. trigger bug by following command: '.../asc_tilling.py'
```

This error reports a command for reproducing the failure rather than exposing the missing dependency directly. Run the `.py` file shown in the error manually, for example, `python3 .../asc_tilling.py`, to see the underlying error, such as `ModuleNotFoundError: No module named 'decorator'`. Then install the missing package. Alternatively, install all dependencies at once from the repository root:

```bash
pip3 install -r super_kernel/requirements-dev.txt
```

#### Other Environment Requirements

```text
CMake >= 3.16.0
GCC >= 7.3.0
```

To switch to GCC 15 or GCC 16, explicitly set `CC` and `CXX` before building. For example:

```bash
export CC=gcc-15
export CXX=g++-15
```

You can also set `GCC_VERSION=15` or `GCC_VERSION=16` to let the scripts select the corresponding compiler commands. This variable applies only to test scripts, such as `scripts/test/run_autofuse_test.sh`, and CI builds. The main `--pkg` build does not read it; explicitly set `CC` and `CXX` to change the compiler for that build. Do not use `update-alternatives` to change the system's default GCC.

After switching compilers, clean `build/` first so that CMake does not keep using the cached compiler configuration.

On openEuler, install the requirements with:

```bash
sudo yum install cmake gcc
```

On Ubuntu, install them with:

```bash
sudo apt-get install cmake gcc
```

### Set Environment Variables

Before running a sample, set up the environment and select the NPU device. Adjust the driver and CANN paths to match your installation:

```bash
# Path to your driver installation.
source /usr/local/Ascend/driver/bin/setenv.bash
# Path to your CANN installation.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Use device 0; keep this consistent with the sample script.
export ASCEND_DEVICE_ID=0
```

### Run a Sample

For a sample named `test.py`, run:

```bash
python3 test.py
```

### Additional Debugging Environment Variables

#### TORCH_COMPILE_DEBUG

This native PyTorch environment variable enables detailed debugging logs and saves intermediate compilation artifacts.

```bash
export TORCH_COMPILE_DEBUG=1
```

Repeated runs of the same script may skip compilation because of cached artifacts. Combine this variable with `TORCHINDUCTOR_FORCE_DISABLE_CACHES` to force recompilation on every run.

#### TORCHINDUCTOR_FORCE_DISABLE_CACHES

This native PyTorch environment variable disables the Inductor cache, forcing recompilation on every run.

```bash
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```

It significantly increases graph startup time. Do not use it in production deployments.

#### Optional: ASCEND_LAUNCH_BLOCKING

This native `torch_npu` environment variable enables synchronous Ascend kernel execution. Each kernel launch waits for completion, making it easier to identify the first failing kernel.

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

It significantly reduces launch performance. Do not use it in production deployments.

#### Optional: AUTOFUSE_DFX_FLAGS

This AutoFuse diagnostic environment variable dumps the internal fusion graph for each automatically fused operator. You can open the `.pbtxt` files with netron.app to inspect them.

```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/path-to-dump/"
```

The AutoFuse backend writes each fused operator's graph to the specified dump directory.

Compilation performance diagnostics are controlled by `codegen_compile_debug=true`. For example:

```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true"
```

When enabled, this option:

- Reports the time spent in each LLVM pass (`-ftime-report=per-pass`).
- Generates compiler timeline JSON files, saved to `~/.cache/autofuse_compile_trace` by default. The terminal prints `[CompileTrace] <file path>` for each trace.

Host compilation reuses an existing precompiled header (PCH) and attempts to create one on a cache miss. The PCH cache directory is `~/.cache/autofuse_pch_cache`. If caching or PCH creation fails, compilation automatically falls back to ordinary host compilation.

### Analyze Results and Debugging Output

With `TORCH_COMPILE_DEBUG` enabled, debugging information is written to a `torch_compile_debug` subdirectory of the current working directory. Directories prefixed with `autofused_` contain fused operator artifacts generated by the `torch_npu` AscendC backend; the other directories contain native PyTorch Inductor artifacts. Each `autofused_` directory exposes the structure of one fused operator and can be used to inspect its fusion scope and generated code. If no such directory is generated, no fused operator was produced during that compilation. Check terminal messages such as `Fallback aten.xxxx $reason: <reason>` to understand why fusion did not occur.

You can also use profiling to measure the operator performance benefit of automatic fusion. For the samples above, comment out the entire `torch.compile(...)` block to run the model without compilation as the baseline without automatic fusion:

```python
# model = torch.compile(
#     model,
#     dynamic=False,
#     fullgraph=True,
#     options={"npu_backend": "ascendc"},
# )
```

Collect profiling data with automatic fusion disabled and enabled, then compare the total execution time of all relevant operators within the same computation scope.

For details on the profiling tools, see the [Profiling Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolProfiling).

Not all operators in a model can be fused. Operators that are not lowered at the Inductor layer remain standalone operators. The relative execution-time reduction is `(total operator time before fusion - total operator time after fusion) / total operator time before fusion`. You can also compare `aiv_mte2_time` (input transfer time) and `aiv_mte3_time` (output transfer time) between the fused operator and the standalone operators.

For numerical accuracy analysis, see the [Accuracy Debugging Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolAccucacy).

### Enable AutoFuse in Larger Networks

You do not need to import `inductor_npu_ext` separately to enable AutoFuse in a network. Specify the AscendC backend in `torch.compile`:

```python
model = torch.compile(
    model,
    dynamic=False,
    fullgraph=True,
    options={"npu_backend": "ascendc"},
)
```
