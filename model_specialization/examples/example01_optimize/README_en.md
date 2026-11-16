# Model Specialization Example

## Example Functionality

This example captures two custom Add calls and specializes the entire model using `aclmdlRISpecOptimize`.

## Directory Structure

```text
examples/
├── common/
│   └── sample_resources.h  # Shared Runtime, Tensor, and executor resource management
└── example01_optimize/
    ├── CMakeLists.txt      # Host build configuration
    ├── main.cpp            # Input preparation, capture, optimization, execution, and verification
    ├── README.md
    └── README_en.md
```

## Example Description

The shared custom Add operator computes elementwise addition for two float32 tensors of shape `[1024]`. Inputs are `x[i] = i` and `y[i] = 1`.

This example captures `t = x + y` followed by `z = t + y`, optimizes the entire model, and checks `t[i] = i + 1` and `z[i] = i + 2` on the Host.

### Implementation Flow

1. Initialize ACL, Context, and Stream; prepare inputs and float32 Tensors.
2. Call `aclnnAddCustomGetWorkspaceSize`, allocate workspace, and make the executors repeatable.
3. Capture both Add calls.
4. End capture and call `aclmdlRISpecOptimize`.
5. Execute the model, synchronize the Stream, and copy outputs back for numerical verification.
6. Destroy the model before releasing executors, Tensors, Device memory, and Runtime resources.

## Prerequisites

| Product | Supported |
| --- | --- |
| Ascend 950PR/Ascend 950DT series | √ |
| Atlas A3 series | √ |
| Atlas A2 series | √ |

Prepare the environment using the [source build guide](../../../docs/en/build.md). CMake 3.16 or later, a C++17 compiler, and a compatible CANN environment are required. Install or build graph-autofusion to provide `libmodel_specialization.so`.

The CANN environment must provide ASC operator-package build support and Kernel Spec compilation. Both examples share `../custom_op`; build and install its package once.

## Commands

Run these commands from this example directory.

1. Load the CANN environment:

   ```bash
   source /usr/local/Ascend/cann/set_env.sh
   ```

   For another installation path, load its corresponding `set_env.sh`.

2. Build the shared custom operator package:

   ```bash
   cmake -S ../custom_op -B ../build/operator -DASCEND_COMPUTE_UNIT=Ascendxxx
   cmake --build ../build/operator --target binary package -j8
   ```

   Replace `Ascendxxx` with the actual AI processor series. The project enables Kernel Spec through `KERNEL_SPEC_MODE Normal`; the package includes the Kernel binary, compilation resources, and source. For operator project build configuration, see the asc-devkit operator development documentation.

3. Install the package into the example build directory:

   ```bash
   mkdir -p ../build/opp
   bash ../build/operator/custom_opp_*.run --quiet --install-path="$(cd ../build/opp && pwd)"
   source ../build/opp/vendors/customize/bin/set_env.bash
   export ASCEND_CUSTOM_OPP_PATH="$(cd ../build/opp/vendors/customize && pwd)"
   ```

   Both examples share this package. After it has been built and installed, skip step 2 and the first two commands of step 3. In a new terminal, load the CANN environment, source the vendor environment script, and export `ASCEND_CUSTOM_OPP_PATH` again.

4. Build the Host program:

   ```bash
   cmake -S . -B build/host
   cmake --build build/host -j8
   ```

   For a non-default model library location, append `-DMODEL_SPECIALIZATION_LIBRARY=/path/to/libmodel_specialization.so` to the configure command. For another operator installation, set `-DOPP_VENDOR_PATH=/path/to/vendors/customize`.

5. Run on the NPU:

   ```bash
   export ASCEND_GLOBAL_LOG_LEVEL=1
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ./build/host/demo 0 > build/host/specialization.log 2>&1
   ```

   The first two commands enable INFO logging and console output. Both logs and program output are saved to `build/host/specialization.log` under this example directory. Inspect the saved log as described in Expected Result without rerunning the example.

   The argument is the device ID and defaults to 0.

   > [!WARNING]
   > **Notice**: This example supports only NPU execution.

## Expected Result

A successful run prints:

```text
Numerical check passed: 1024 float elements
Numerical check passed: 1024 float elements
Example passed
```

`Example passed` indicates that model execution and output verification passed. The original Kernel can also produce correct results, so this message alone does not confirm specialization.

Check the compilation and task update results in `build/host/specialization.log` to confirm that the target Kernel was successfully compiled and the corresponding model task was updated.
