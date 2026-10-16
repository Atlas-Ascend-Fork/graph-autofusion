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

The following uses a PyTorch scenario as an example to demonstrate how to set up the PyTorch environment, run the sample, and evaluate the performance of the resulting kernels using profiling data.

### Install Dependencies

#### Install torch_npu

```bash
pip3 install numpy
pip3 install pyyaml
pip3 install setuptools
```

To ensure the `torch_npu Daily` environment is compatible with the AutoFuse AscendC backend, install `torch_npu` using the [PyTorch Environment Installation Script](../scripts/env_install/pytorch/setup_torch_npu_daily.sh) provided in the repository. Installing `torch_npu` directly through PyPI is not recommended.

Run the following command in the root directory of the Graph-AutoFusion repository:

```bash
bash scripts/env_install/pytorch/setup_torch_npu_daily.sh
```

#### Sample Runtime Dependencies

When running the [PyTorch scenario examples](./examples/pytorch/README_en.md), the fused operator code generation phase invokes generated scripts such as `asc_tilling.py`. These scripts depend on Python packages such as `decorator`, `scipy`, and `attr` (`attrs`), which are not included in the `numpy`, `pyyaml`, and `setuptools` installation above.

If the following error occurs:

```text
RuntimeError: Failed to build ascend kernel. trigger bug by following command: '.../asc_tilling.py'
```

This error wraps the real cause as a "trigger bug" command and does not directly reveal the missing dependency. In this case, manually run the `.py` file shown in the error (for example, `python3 .../asc_tilling.py`) to see the actual error (such as `ModuleNotFoundError: No module named 'decorator'`), and then install the missing package accordingly. Alternatively, you can install all dependencies at once from the repository root:

```bash
pip3 install -r super_kernel/requirements-dev.txt
```

#### Other Environment Dependencies

```bash
CMake >= 3.16.0
GCC >= 7.3.0
```

To switch to gcc15/gcc16, explicitly set `CC/CXX` before building, for example:

```bash
export CC=gcc-15
export CXX=g++-15
```

You can also set `GCC_VERSION=15` or `GCC_VERSION=16` and let the scripts generate the matching compiler commands. This variable only takes effect for test scripts (e.g. `scripts/test/run_autofuse_test.sh`) and CI builds. The `--pkg` main build does not read this variable; set `CC/CXX` explicitly to switch compilers for the main build. Do not use `update-alternatives` to change the system default gcc.

After switching compilers, clean `build/` before reconfiguring so CMake does not reuse the old compiler cache.

On openEuler systems, run the following command:

```bash
sudo yum install cmake gcc
```

On Ubuntu systems, run the following command:

```bash
sudo apt-get install cmake gcc
```

### Set Environment Variables

Before executing the use cases, set the following environment variables to configure the NPU device:

```bash
# Installation path of your driver package
source /usr/local/Ascend/driver/bin/setenv.bash
# Installation path of your CANN package
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Assume that the script runs on device 0, consistent with the device configured in the script
export ASCEND_DEVICE_ID=0
```

### Execute Use Cases

Assume that the use case is named `test.py`. Run it directly:

```bash
python3 test.py
```

### More Debugging-Related Environment Variables

#### TORCH_COMPILE_DEBUG

Purpose: A native torch environment variable that enables detailed debugging logs and saves intermediate compilation artifacts.

Usage:

```bash
export TORCH_COMPILE_DEBUG=1
```

Note: Repeatedly executing the same script may skip compilation because of cached data. You can use `TORCHINDUCTOR_FORCE_DISABLE_CACHES` together with this variable to force recompilation during each execution.

#### TORCHINDUCTOR_FORCE_DISABLE_CACHES

Purpose: A native torch environment variable that disables the Inductor cache and forces recompilation during each execution.

Usage:

```bash
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```

Note: This significantly increases graph startup time. Do not use this environment variable in actual deployment.

#### Optional: ASCEND_LAUNCH_BLOCKING

Purpose: A native torch_npu environment variable that enables synchronous execution of Ascend kernels. Each Kernel launch waits for completion, making it easier to identify the first Kernel that reports an error.

Usage:

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

Note: This significantly reduces launch performance. Do not use this environment variable in actual deployment.

#### Optional: AUTOFUSE_DFX_FLAGS

Purpose: An AutoFuse DFX environment variable that saves the internal fusion graph structure corresponding to each automatically fused operator. The generated `.pbtxt` files can be opened and viewed using netron.app.

Usage:

```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/path-to-dump/"
```

Note: The AutoFuse backend generates a dump graph for each fused operator in the specified dump path.

Enable compiler performance diagnostics with `codegen_compile_debug=true`. For example:

```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true"
```

When enabled, AutoFuse:

- Prints elapsed time for every LLVM pass with `-ftime-report=per-pass`.
- Writes compiler timeline JSON files to `~/.cache/autofuse_compile_trace` by default and prints `[CompileTrace] <file path>`.

Trace file names include a unique identifier to prevent overwrites from concurrent or repeated compilations. Host compilation reuses an existing PCH and attempts to create one on a cache miss. PCH is cached in `~/.cache/autofuse_pch_cache`; Host compilation automatically falls back without PCH if its cache or creation is unavailable.

### Result Analysis & Debug Output Analysis

After `TORCH_COMPILE_DEBUG` is enabled, debugging information is output to the `torch_compile_debug` subdirectory under the current execution directory. Directories prefixed with `autofused_` contain fused operator artifacts generated by the `torch_npu` AscendC backend, while the remaining directories contain native artifacts generated by PyTorch Inductor. Each directory prefixed with `autofused_` corresponds to the white-box structure of a fused operator and can be used to view the fusion scope and code generation results. If no directory prefixed with `autofused_` is generated, no fused operator was produced during the current compilation process. In this case, analyze the reason why fusion did not occur based on information such as `Fallback aten.xxxx $reason: xx reason` in the terminal output.

Users can also use Profiling configurations to observe the operator performance gains after automatic fusion is enabled. For the preceding Sample use cases, comment out the entire `torch.compile(...)` code block so that the model runs in non-compiled mode, which can be used as a comparison scenario without automatic fusion enabled.

```python
# model = torch.compile(
#     model,
#     dynamic=False,
#     fullgraph=True,
#     options={"npu_backend": "ascendc"},
# )
```

Collect profiling data for both scenarios—with auto-fusion disabled and enabled—and compare the total execution time of all relevant operators within the same computation scope.

For details about how to use the Profiling performance analysis tool, refer to the [Profiling Performance Analysis Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolProfiling).

Note that not all operators in a model can be fused. Operators that are not lowered at the Inductor layer remain as standalone operators.The fusion performance improvement is calculated as follows:

`(Total execution time of all operators before fusion - Total execution time of all operators after fusion) / Total execution time of all operators before fusion`

For further analysis, you can compare the fused operator with the corresponding standalone operators in terms of `aiv_mte2_time` (input data transfer time) and `aiv_mte3_time` (output data transfer time) to evaluate the reduction in data transfer overhead.

For precision analysis, refer to the [Precision Debugging Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolAccucacy).

### Enabling AutoFuse in Complex Networks

To enable AutoFuse in a network, users do not need to import `inductor_npu_ext` separately. Specify the AscendC backend in `torch.compile`:

```python
model = torch.compile(
    model,
    dynamic=False,
    fullgraph=True,
    options={"npu_backend": "ascendc"},
)
```
