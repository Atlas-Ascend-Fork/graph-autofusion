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
