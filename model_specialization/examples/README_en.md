# Model specialization examples

## Overview

These examples show how to specialize a captured model or only the operators inside a Scope. Use only one optimization API per model.

## Directory Structure

```text
examples/
├── common/                     # Shared resource management
├── custom_op/                  # Shared AddCustom operator package
├── example01_optimize/          # Whole-model specialization
└── example02_optimize_by_flag/  # Scope-based specialization
```

## Sample Summary

| Example | Scenario | APIs | Documentation |
| --- | --- | --- | --- |
| [example01_optimize](example01_optimize/README_en.md) | Specialize a model containing two AddCustom calls in the whole model | aclmdlRISpecOptimize | [中文 README](./example01_optimize/README.md) / [English README](./example01_optimize/README_en.md) |
| [example02_optimize_by_flag](example02_optimize_by_flag/README_en.md) | Specialize only the AddCustom inside a Scope | aclmdlRISpecScopeBegin, aclmdlRISpecScopeEnd, aclmdlRISpecOptimizeByFlag | [中文 README](./example02_optimize_by_flag/README.md) / [English README](./example02_optimize_by_flag/README_en.md) |

## Shared Operator

`custom_op` provides an elementwise Add for two float32 tensors of shape `[1024]`. The standard operator-package build enables `KERNEL_SPEC_MODE Normal`, producing the binary, compilation resources, and source package together. No separate resource preparation script is required.

Follow the complete build, install, and execution commands in either example README. Both examples share `build/operator` and `build/opp`; build and install the operator package only once.
