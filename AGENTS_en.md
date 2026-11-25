# Graph-autofusion Agent Guide

See [AGENTS.md](AGENTS.md) for the full repository guidance.

## Overview

Graph-autofusion is a collection of fusion acceleration components for Ascend chips. It currently includes major components such as SuperKernel, Autofuse, Multistream (multi-stream planning), and model_specialization (model specialization). The project reduces task scheduling and data-copy overhead through capabilities such as SuperKernel, autofusion, and ASCIR/AscendC codegen.

### Key Directories

| Directory | Purpose |
|------|------|
| `super_kernel/` | SuperKernel component source code, Python/C++ tests, and device-side verification cases |
| `autofuse/` | Autofuse source code, optimization passes, codegen, and test framework |
| `multistream/` | Multistream pure-Python multi-stream planning component |
| `model_specialization/` | Model specialization component (specialized kernels, source code, and tests) |
| `cmake/` | Common CMake scripts, dependencies, and packaging logic |
| `scripts/` | Build, test, and OAT check scripts |
| `docs/` | Build, design, guideline, and component documentation |
| `.claude/skills/` | Project-specific agent skills |

## Sequence Diagram Colors

When feature designs, issue analyses, or PR descriptions use sequence diagrams to explain a workflow change, show the original and updated workflows and identify the key steps. Include a legend beside the diagram and use both color and text labels inside it.

- **Light red · Original workflow** shows the previous call order, key branches, and confirmed failure points.
- **Light green · Current implementation** shows the updated call order and identifies the steps that were added or changed.
- **Light blue · Proposed workflow** shows the target workflow or follow-up changes still under design.

In Mermaid `sequenceDiagram`, use `rect rgba(...)` to highlight the relevant steps and `Note over` to label the phase and its key changes. Shared steps can use the default styling for easy comparison.
