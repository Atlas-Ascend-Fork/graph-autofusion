# TensorFlow Scenario Examples

## Description

Use AutoFuse to perform operator fusion in TensorFlow networks. The AutoFuse fusion pass in GE (Graph Engine) automatically identifies operators that can be fused and completes the fusion.

## Directory Structure

```text
├── README.md
├── README_en.md
├── common/
│   ├── __init__.py
│   ├── config.py                           # Common configuration
│   ├── profiling_utils.py                  # Common Profiling utilities
│   └── tf_runner.py                        # Common TF1/TF2 runtime framework
├── af_tf_eleandele/
│   ├── README.md
│   ├── README_en.md
│   └── test_abs_relu_exp.py                # abs + relu + exp fusion example
├── af_tf_eleandbroadcast/
│   ├── README.md
│   ├── README_en.md
│   └── test_abs_add_relu.py                # abs + add + relu fusion example
└── af_tf_eleandreduce/
    ├── README.md
    ├── README_en.md
    └── test_abs_reducesum.py               # abs + reduce_sum fusion example
```

## Prerequisites

Before running this example, complete the following steps in sequence:

> Run all the following commands from the graph-autofusion repository root.

1. Follow the [Installation Guide](../../../docs/en/quick_install.md) to correctly install the Toolkit and Ops packages and configure the environment variables.
2. Follow [Environment Build and Deployment](../../../docs/env_install/tensorflow/env_tf.md) to set up the TensorFlow environment. On x86_64, TensorFlow can be installed directly using pip. On aarch64, TensorFlow must be built from source.
3. Alternatively, use the one-click configuration script to automatically set up the environment. This script is available only for the **x86_64 architecture**:

   ```bash
   bash scripts/env_install/tensorflow/setup_tf_env.sh
   ```

   After the script is complete, activate the environment:

   ```bash
   source scripts/env_install/tensorflow/env/activate_tf1.sh    # TensorFlow 1.15
   # Or
   source scripts/env_install/tensorflow/env/activate_tf2.sh    # TensorFlow 2.6.5
   ```

   > **This script does not support the aarch64 architecture.** On aarch64, follow [Building TensorFlow from Source on aarch64](../../../docs/env_install/tensorflow/build_tf_aarch64.md) to perform the build manually.

## Set Environment Variables

```bash
# Define the CANN package installation path based on the actual installation location.
export CANN_INSTALL_PATH=/usr/local/Ascend

# Load the driver-related environment variables from the CANN package.
source $CANN_INSTALL_PATH/driver/bin/setenv.bash

# Load the Toolkit-related environment variables from the CANN package.
source $CANN_INSTALL_PATH/ascend-toolkit/set_env.sh

# Assume that the example runs on device 0.
export ASCEND_DEVICE_ID=0

# Enable automatic fusion.
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

## Run the Examples

The following three examples are available. Refer to the corresponding document based on the scenario you want to run:

- [Elementwise + Elementwise](./af_tf_eleandele/README_en.md): `abs + relu + exp`
- [Elementwise + Broadcast](./af_tf_eleandbroadcast/README_en.md): `abs + add + relu`
- [Elementwise + Reduce](./af_tf_eleandreduce/README_en.md): `abs + reduce_sum`

## Expected Result

Each example performs 100 inference steps. If no errors occur, the example has executed successfully. Use graph dump files or Profiling data to confirm whether fusion takes effect.

## References

- [AutoFuse Introduction](../../README_en.md)
- [Environment Build and Deployment](../../../docs/env_install/tensorflow/env_tf.md)
- [Building TensorFlow from Source on aarch64](../../../docs/env_install/tensorflow/build_tf_aarch64.md)
- [Precision Debugging Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)
- [Profiling Performance Analysis Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolProfiling)
