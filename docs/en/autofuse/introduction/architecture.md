# Architecture Introduction of AutoFuse

## Overview of System Architecture

AutoFuse is an automatic operator fusion component for Ascend chips in the CANN ecosystem. It receives fused subgraphs and unified IR produced by graph compilation components such as GE and Inductor after graph transformation, Lowering, and fusion-range determination. Within the determined fusion range, AutoFuse performs scheduling optimization, Tiling solving, and code generation, and finally outputs high-performance AscendC fused operators. By fusing multiple operators that would otherwise execute independently into a single Kernel, AutoFuse effectively reduces intermediate-result GM reads and writes, Kernel launches, and Host-Device scheduling overhead, thereby significantly improving model execution performance on Ascend NPU.

The overall logic structure is shown in the figure below:

![image.png](https://raw.gitcode.com/user-images/assets/9919791/49b11a34-91e6-4496-9ff6-f5137ea8b1b0/image.png 'image.png')

As shown in the figure, based on the unified AscendLoopIR (IR modeled for the AscendC programming language) and supporting Schedule and code generation capabilities, the automatic fusion solution constructs two fusion implementation paths:

- **GE Path**: Based on the self-developed GE framework of Ascend, focusing on Ascend NPU affinity. GE completes symbolic shape derivation, Lowering and fusion range determination, and AutoFuse is responsible for backend scheduling, tiling and code generation.
- **Inductor Path**: Interfaces with PyTorch Inductor, focusing on ecosystem adaptation, reuses Inductor's fusion range identification capability, and backend processing is still completed by AutoFuse.

## What Problems Does It Solve

### Challenges

When a large number of operators are executed independently on AI chips, the following challenges arise:

- **Memory Bound constrains performance**: The computing power of AI chips is usually higher than the memory bandwidth. When a model is composed of many small operators, each intermediate result needs to be written to and reread from global memory. The memory handling overhead may exceed the computation itself, causing computing units to wait for data.
- **High cost of hand-written fused operators**: For recommendation-like models, whose structures change frequently and contain many operators, manually writing fused Kernel/Tiling code requires substantial effort and a long development cycle, making it difficult to keep up with model iteration.

### Fusion Scheme Selection

For the above problems, the industry has three mainstream operator fusion schemes, compared as follows:

| Scheme                       | Advantages                                             | Disadvantages                                                                                                                   |
| -------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **Hand-written Fused Operator**     | Highest performance upper bound                                     | Only suitable for high-value fixed patterns, highly customized; heavy development workload, high maintenance cost when model structure changes                                                |
| **Pattern Matching-based Fusion** | Stable and controllable, easy to implement engineering, good performance                 | Limited fusion scope, can only handle known patterns, high failure probability, difficult to replicate effects across models; it will generate a large number of pattern recognition passes and specific kernel implementations |
| **JIT Automatic Fusion**     | Strong generalization capability, can cover a large number of small operator chains, suitable for rapid model iteration | Relies on unified IR representation, and performance modeling is difficult                                                                                    |

AutoFuse chooses the **JIT automatic fusion** scheme: it prioritizes **generalization capability**, covers a large number of operator combinations through unified IR and compile-time automatic decision-making, adapts to rapid model iteration, and improves post-fusion execution performance through scheduling, tiling and code generation.

## Key Technical Solutions

AutoFuse classifies network operators into two categories based on their computational characteristics. The first category consists of basic computation types, including Elemwise, Broadcast and View operators (Transpose, Slice and Split). The second category consists of computation types such as Reduce, Concat and MatMul, which extend the fusion capabilities based on the basic computation types.
This means that each extended fusion capability needs to support fusion with the basic computation types.

### Supported Operator Types

The following table lists the main supported operator types and their corresponding computing units:

| Operator Type             | Description                                           | Computing Unit   | Typical Operator Examples                    |
| :------------------- | :--------------------------------------------- | :--------- | :------------------------------ |
| **Elemwise**    | Element-wise computation, each output element corresponds to one input element one-to-one | Vector     | Add, Mul, Abs, Exp, Relu, Cast  |
| **Broadcast**  | Broadcast computation, extend small shape data along the broadcast axis to target shape for element-wise computation | Vector     | BroadcastTo, BiasAdd            |
| **View**       | View transformation that changes the logical shape, axis order or partitioning method of data | MTE/Vector | Transpose, Slice, Split         |
| **Reduce**     | Reduction computation, aggregate multiple elements along specified axis | Vector     | ReduceSum, ReduceMax, ReduceMin |
| **Generic Norm** | Normalization computation formed by combining same-axis Reduce, Broadcast and Elemwise operations; not a single operator | Vector | LayerNorm, RMSNorm              |
| **Concat**     | Concatenation, concatenate multiple Tensors into one along the specified axis         | MTE/Vector | Concat                          |
| **MatMul**     | Matrix computation, including convolution and matrix multiplication                       | Cube       | MatMul, Conv2D                  |

### Supported Fusion Capabilities

AutoFuse currently supports two main fusion categories: VV and CV.

| Fusion Type | Fusible Operator Types |
| :---------- | :--------------------- |
| **VV fusion** (Vector + Vector) | Supports fusion among Vector operators, including Elemwise, Broadcast, View (including Transpose, Slice, and Split), Reduce, and Concat. |
| **CV fusion** (Cube + Vector) | Supports fusion between Cube and Vector operators. |

## Frontend Adaptation

Frontend adaptation is responsible for converting the model graph from mainstream deep learning frameworks such as PyTorch and TensorFlow into graph IR that AutoFuse can process. Currently there are two main access paths:

- **GE Path**: In online scenarios, TorchAir or TensorFlow Adapter converts graph representations such as AtenIR and GraphDef into AscendIR, which then enters the GE graph compilation process. In offline scenarios, the ATC built-in Parser parses models in formats such as TensorFlow and ONNX into AscendIR. For related information, see the [GE Project](https://gitcode.com/cann/ge).
- **Inductor Path**: Convert PyTorch AtenIR to InductorIR and use the PyTorch Inductor path. For the `torch-npu` module, see the [PyTorch Project](https://gitcode.com/Ascend/pytorch); for the `inductor-npu-ext` module, see the [TorchAir Project](https://gitcode.com/Ascend/torchair/tree/master/experimental/_inductor_npu_ext).

## Graph Engine

As the frontend of AutoFuse, Graph Engine is responsible for symbolic shape derivation, Lowering, and CanFuse fusion condition judgment on the AscendIR graph, and determines the fusible operator range and the computational expression after fusion.

| Step                          | Responsibility                                                                                                               | Related Materials                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **Symbolization**              | Express operator Shape with symbols, enhance dynamic Shape processing capability, provide simplification, derivation and Guard functions, and provide key information for subsequent loop axis merging and memory optimization | [Official Document](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/graphdevg/autofuse_1_0006.html) |
| **Lowering**            | Convert high-level AscendIR to low-level AscendLoopIR, express computing logic close to AscendC semantics, determine fusion structure and data dependency               | [Official Document](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/graphdevg/autofuse_1_0012.html) |
| **CanFuse (Fusion Strategy)** | Determine the fusion boundary from three levels: semantic correctness, expressibility and resource budget                                                               | [Official Document](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/graphdevg/autofuse_1_0018.html) |

## AutoFuse

As the backend of upper-layer GE or Inductor, AutoFuse is the core of automatic fusion compilation. After receiving the determined fusion range, AutoFuse completes fused kernel scheduling, code generation and tiling strategy solving through three core modules: Schedule, Codegen and Auto Tiling, and provides operator interface support with the help of AscendC API.

| Module                  | Responsibility                                                                                          | Related Materials                                                |
| --------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **Schedule**    | Scheduling strategy generation: compute rearrangement, loop merging, parallel optimization, memory optimization and multi-template generation                              | [Feature Description](../design/features/schedule.md)                |
| **Codegen**     | Code generation: parse the scheduled graph and generate Host-side and Device-side code                                            | [Feature Description](../design/features/codegen.md)                 |
| **Auto Tiling** | Tiling solving: solve Tile size and core partitioning strategy under UB constraints, evaluate the performance of the tiling scheme, and select the appropriate template and tiling strategy | [Feature Description](../design/features/auto_tiling.md)             |
| **AscendC API** | Provide APIs for Vector computation, Cube computation, data movement, type conversion, etc.                                         | [Source Reference](../../../../autofuse/v35/ascendc/api_regbase) |

## Compilation and Execution

The Host and Device source code generated by AutoFuse is further compiled into Host-side shared library and Device-side Kernel binary by the **BiSheng compiler**. At runtime, the upper framework prepares Tiling parameters according to the actual input and starts the fused kernel. **CANN Runtime** is responsible for device resource management, kernel launching and execution. For the operator chain originally completed by multiple kernels, fusion can usually reduce the number of kernel launches and global memory reads/writes of intermediate tensors, thereby reducing data movement and scheduling overhead, and improving the hardware resource utilization and model execution performance of **Ascend NPU**.

## Enable AutoFuse by Access Path

Enable AutoFuse according to the two fusion implementation paths:

- **GE Path**: [Enable AutoFuse on the GE Path (TensorFlow)](./enable_autofuse_ge_tensorflow.md). This document uses TensorFlow as an example to describe dependency versions, `AUTOFUSE_FLAGS` configuration, environment variables and usage examples. For accessing the GE path from PyTorch via TorchAir, see [Enable AutoFuse on the GE Path (PyTorch)](./enable_autofuse_ge_pytorch.md), which also demonstrates a dynamic-shape scenario.
- **Inductor Path**: [Enable AutoFuse on the Inductor Path](./enable_autofuse_inductor.md). This document uses PyTorch as an example to describe dependency versions, `torch.compile` configuration, environment variables and usage examples.

## Project Structure

```text
graph-autofusion/
├── autofuse                     # AutoFuse core code
│   ├── ascir                    # AscIR operator metadata, built-in operators and registration capability
│   ├── graph_metadef            # Basic definitions of graph, node, tensor, attribute and symbolic expression
│   ├── inc                      # External public header files and fusion-related interfaces
│   ├── optimize                 # Graph optimization, task partitioning, automatic scheduling and memory planning
│   ├── att                      # Auto Tiling, candidate solution solving and Host Tiling generation
│   ├── codegen                  # Kernel, Tiling Data, Host Tiling and InferShape code generation
│   ├── ascendc                  # AscendC API definitions and extensions used for code generation
│   ├── compiler                 # Python/C++ interfaces, Host/Device compilation and artifact publishing
│   ├── common                   # Logging, configuration, platform context and common utilities
│   ├── v35                      # Optimizations for Ascend 950 chip
│   ├── examples                 | Access and operation examples for PyTorch, TensorFlow, etc.
│   ├── tests                    # Unit tests, system tests and end-to-end tests
│   └── tools                    # Auxiliary tool scripts
├── super_kernel                 # Independent SuperKernel fusion component
├── docs                         # Graph-autofusion project documents
├── cmake                        # Public CMake scripts and dependency configuration
└── scripts                      # Environment check, test, packaging and installation scripts
```

## Supported Products

The automatic fusion feature is currently only supported on the following product models:

| Series       | Product Models                                      |
| ---------- | --------------------------------------------- |
| Ascend 950 | Ascend 950PR / Ascend 950DT                   |
| Atlas A3   | Atlas A3 Training Series Products / Atlas A3 Inference Series Products |
| Atlas A2   | Atlas A2 Training Series Products / Atlas A2 Inference Series Products |

> The above support list may expand with CANN version iteration. For the latest information, please refer to the product support information released on the [Ascend Official Website](https://www.hiascend.com/).

## Related Materials

- [AutoFuse Overview (official documentation)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/graphdevg/docs/zh/user_guides/graph_dev/autofuse/overview.md)
- [Autofuse Component Description (with multi-type operator fusion examples)](../../../../autofuse/README_en.md)
