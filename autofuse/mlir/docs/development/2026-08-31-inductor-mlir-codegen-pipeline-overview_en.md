# Inductor MLIR Codegen Pipeline Overview

This is the English companion of [2026-08-31-inductor-mlir-codegen-pipeline-overview.md](2026-08-31-inductor-mlir-codegen-pipeline-overview.md).
The Chinese document remains the detailed source of truth for this development
record.

## Current pipeline

The current flow uses `mlir_codegen_config=none|mlir|fallback` and excludes
unsupported Cube fusion before entering MLIR code generation. `BuildPipelineSession`
derives one `TargetProfile`, and `PreparePlanGeneration` enumerates the complete
candidate set, builds the schema and generates one complete ATT program for both
static and dynamic modes.

Static and dynamic modes now retain the original plan and template identities.
Static code generation lowers the complete selected plan after runtime ATT solving;
dynamic code generation lowers all candidates and leaves selection to runtime ATT.
Host tiling, device code and wrapper compilation can be performed in parallel by
Inductor. The generated `AutofuseTiling` call computes the selected workspace.

Failures are handled by the outer strict/fallback policy. The pipeline does not
silently drop candidates, renumber plans, or deliver partial kernel and tiling
artifacts. The resource budget is derived once and shared by ATT, AFIR and the host
ABI; the 256-byte UB reservation is deducted once.

The detailed record includes the updated candidate policy, failure semantics,
resource contract and validation results. This companion does not claim that the
remaining compile-time, memory and artifact-size measurements are complete.
