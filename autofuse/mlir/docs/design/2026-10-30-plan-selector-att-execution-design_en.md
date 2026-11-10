# PlanSelector and ATT Execution Design

This is the English companion of [2026-10-30-plan-selector-att-execution-design.md](2026-10-30-plan-selector-att-execution-design.md).
The Chinese document remains the authoritative implementation record; this page
summarizes its current contracts and validation status.

## Purpose and scope

The design makes candidate enumeration, ATT solving, resource budgeting, template
identity and MLIR code generation obey one explicit contract. It covers static and
dynamic scheduling, ATT host execution, tiling delivery, failure classification and
regression coverage. It does not change the ATT scoring algorithm, add template
families, restore capability pre-filtering, or change public Python, kernel or
tiling ABIs.

## Current contract

- Candidates are enumerated once in schedule order. Original subgraph, plan, stage,
  template and tiling keys are preserved; filtered positions must not be used as
  replacement identities.
- Static mode executes the real ATT solver once, validates the complete winning
  plan and lowers only that plan's required stages. A missing winner, invalid key,
  toolchain failure or incomplete result stops the MLIR path. Strict versus legacy
  fallback behavior is selected by the outer MLIR configuration.
- Dynamic mode keeps all candidates and the ATT program. Runtime dispatch keys must
  map to generated implementations; candidates are not silently removed after a
  lowering failure.
- ATT preparation, execution and result decoding are separate responsibilities.
  The executor receives an explicit host toolchain and resource budget and returns
  either validated solver data or a stage-specific diagnostic.
- ATT and AFIR consume one `TargetProfile`-derived budget. Available UB is
  `physical_ub - reserved_ub`, with the reservation deducted exactly once; GM
  workspace is accounted for separately.

## Identity and resource rules

Explicit and default template keys are validated through the shared schema helpers.
Mixed or duplicate keys in one stage, unknown mappings and ambiguous mappings are
errors. Reuse does not replace logical identity. Static results carry selections,
solved tiling and the target budget as one indivisible result; incompatible shape,
resource or configuration identities must not reuse a frozen result.

## Reliability and compatibility

The implementation preserves DFX behavior, installation layout, Python interfaces,
ATT entry points and public ABI. Temporary host compilation uses direct argument
vectors, bounded subprocess lifetime and cleanup on success, failure, cancellation
or timeout. A failure never commits partial selections or tiling data, and toolchain
errors are not relabeled as capability failures.

## Validation status

The implementation record reports 201/201 selector, protocol and bridge unit tests,
271/271 lit tests, 40/40 Inductor adapter tests, CPU E2E coverage for RCore,
Sum+Max and broadcast cases, and a documented NPU reduce matrix run. The record
also identifies remaining validation for multi-subgraph/RCore identity combinations,
explicit keys, reuse semantics, compile-time cost, peak memory and artifact size;
those items are not claimed as complete by this companion.
