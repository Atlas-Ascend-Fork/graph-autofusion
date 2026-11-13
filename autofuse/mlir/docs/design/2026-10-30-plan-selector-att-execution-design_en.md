# Introduction

## Purpose

Clarify the MLIR bridge PlanSelector and the contracts between template selection, ATT solving, resource budgets, and code generation. This document is intended for bridge, ATT, AFIR, and test maintainers reviewing and implementing the design.

Status: **Historical design. Static winner freezing was not integrated into the current production flow, and its implementation has been removed.** The static execution, frozen results, toolchain, and acceptance descriptions below are retained as a historical proposal; they do not describe current functionality or test coverage.

Implementation as of 2026-11-09:

- `PreparePlanGeneration` enumerates all candidates and returns a `TilingCodeBundle`. For both static and dynamic shapes, the generated host tiling program selects the plan/template and solves tile sizes.
- The standalone static solver result parser, frozen snapshots and emitter, `afplan.static_selection`, and dedicated tests have been removed. Requests no longer retain the `mode` / `shape` parameters that had no behavioral semantics.
- The current MLIR emitter returns `false` from `AutofuseIsStaticShape()` and does not enable Python const tiling baking. This cleanup does not change that behavior or remove the general baking capability on the Python side.
- Candidate enumeration, resource budgets, field contracts, runtime tiling source generation, and original template identity tests remain. These tests do not demonstrate advance selection of a static winner.
- This cleanup only changes internal MLIR C++ interfaces and IR attributes. It does not change SuperKernel, public Python interfaces, installation or packaging, runtime calls, or the production tiling ABI. It adds no runtime solving, logging, or data copies. Historical hand-written IR carrying `afplan.static_selection` is no longer supported.

## Scope

The scope includes candidate enumeration, static/dynamic orchestration, the ATT host execution boundary, unified target resources, template identity and tiling delivery, error classification, and regression coverage.

This design does not change ATT scoring or solving algorithms, restore capability prefiltering, introduce template families, or change operator lowering, numerical tolerances, SuperKernel interfaces, or NPU execution tools. Slow LLD linking is investigated separately.

# Overview

## Software Overview

### Project Introduction

Schedule generates candidates, ATT determines the execution solution, and MLIR implements that solution. PlanSelector determines the scope of code generation and ensures consistency with ATT template identities and tiling parameters.

Mixed responsibilities before implementation (`196052cf`):

| Location | Existing behavior | Target |
|---|---|---|
| `PlanSelector.cpp` | The static entry enumerates candidates, then the solver enumerates them again; the dynamic entry only enumerates | Enumerate once and distinguish compilation modes explicitly |
| `AttStaticTiling.cpp` | Platform queries, ATT generation, compilation/execution, and decoding are mixed | Separate ATT preparation from execution; the executor does not define selection policy |
| `CodegenOrchestration.cpp` | Static ATT preparation is internal, dynamic preparation is external; a complete schedule is still named `narrowed_result` | Unify preparation and preserve original schedule identities |
| `TilingCodeBundle` | Source code coexists with optional static results | Separate ATT source artifacts from final static/dynamic deliverables |
| Resource sources | ATT reads PlatformContext; AFIR reads TargetProfile | Resolve resources once at the entry and pass them explicitly |

The historical simplified `AttCostEvaluator` has been removed from the current branch and will not be reintroduced as an optional fallback.

### Product Environment

Static solving depends on CANN host tiling code, platform information, and an available host C++ compiler. The implementation generates source, compiles a temporary executable, runs it, and reads its output. Dynamic artifacts carry the runtime ATT tiling entry.

Internal C++ orchestration interfaces may change. Python calling interfaces, public kernel/tiling ABIs, and installation directories retain their existing contracts. Cross-machine builds must distinguish the host toolchain from target resources; NPU budgets cannot be inferred from the host CPU architecture.

## Software Functions

- Enumerate all candidates and validate their structure.
- Execute real ATT for static inputs and generate all stages of only the winning solution.
- Generate all candidates for dynamic inputs and select through ATT at runtime.
- Deliver selection identities, parameters, and resource context atomically.
- Report structural errors, ATT failure to find a solution, toolchain failures, and insufficient lowering capabilities distinctly.

## Design Constraints

1. Do not rescore candidates, select a second-best template independently, or switch to all candidates on failure.
2. Do not renumber the original schedule after pruning. Schema, fields, and dispatch must use verifiable identity mappings.
3. Do not replace lowering capability checks with operator or broadcast allowlists.
4. Reuse the existing template semantic vocabulary, DFX selection mechanisms, and error policies. Do not introduce another classification scheme or plan-pinning switch.
5. ATT generation and execution must not depend on network access. Temporary resources must be cleaned up on failure. Limit build parallelism to `-j 8`.
6. Do not introduce caches, a JIT framework, or a performance scoring system as part of this cleanup.

## Assumptions and Dependencies

- The static entry assumes that schedule shapes have been concretized. An empty `ConcreteShape` does not imply missing input. Symbolic shapes marked as static must fail rather than default to zero.
- ATT plan keys and template keys map uniquely to original identities. An explicit template key need not equal its array index.
- Real ATT calls must preserve existing DFX and scenario semantics. If PGO, TopN, or another configuration that can change the winner is incompatible with static freezing, reject freezing explicitly rather than silently ignoring the configuration.
- The current MLIR host tiling ABI and `afir.target` verifier have been confirmed to reserve 256 bytes. This design consolidates that existing convention in `resource_budget.h`; it does not extend it to other backends. ATT currently integrates through source generation, and temporary host program execution is retained.
- The WIP encountered differing UB values of `245760` and `196608`. This observation is input to budget validation, not justification for choosing either value.

# Requirements Analysis and Design

## Overall Design

```text
Complete schedule + mode + scenario/ATT options + unified target resources
                                  |
                    Validate structure; enumerate once
                                  |
                Build complete schema; generate ATT once
                                  |
                     +------------+------------+
                     |                         |
                   Static                    Dynamic
                     |                         |
          Execute ATT; validate results   Keep all candidates and ATT program
                     |                         |
          Winning identities + parameters   Solve with ATT at runtime
                     +------------+------------+
                                  |
                 Build AFIR / artifacts using original identities
```

"Narrowing" refers only to code generation scope; it does not change schedule container indices. Dynamic runtime solving and static compile-time solving share the ATT contract but occur at different times.

## Functional Requirements

### Functional Requirement 1: Candidate Enumeration and Identity

1. Introduction: Check only structure and coordinates, not resource feasibility or implementation capability.
2. Input: A complete `FusedScheduledResult`.
3. Processing: Enumerate subgraphs, plans, stages, and templates in input order. Reject empty structures, invalid indices, and duplicate or ambiguous keys. Enumerate once and reuse the result. Remove the enumeration interface's always-false `unsupported` output.
4. Output: Reuse `PlanSelection`, preserving original coordinates and existing key mappings. A plan is a complete stage combination; RCore phases 1 and 2 are not selected independently. Resolve noncontiguous explicit keys through mappings rather than assuming array indices.

#### ATT and Codegen Numbering Contract

Distinguish coordinates from dispatch keys. The contract aligns how both sides interpret an execution solution; it does not require different layers to use the same integer.

| Identifier | Scope and meaning | Contract |
|---|---|---|
| subgraph_id | Subgraph coordinate in the complete schedule | Preserved through ATT, AFIR, and artifacts; static filtering does not renumber it |
| plan_id | Original ScheduledResult coordinate within a subgraph | The graph-level ATT selector must map uniquely to this plan; never substitute its filtered position |
| stage_id | Original schedule group coordinate within a plan | Preserve stage order and dependencies; both RCore stages carry their own identities |
| template_id | Original impl_graph coordinate within a stage | Locates the graph to lower; it is not the tiling_key |
| tiling_key | Template dispatch value returned by ATT for this stage | Maps uniquely to template_id within the full (subgraph, plan, stage) scope |

ATT `CheckKeyValid` has been confirmed to require either all explicit keys or all default keys within a stage. Reject mixed settings instead of assigning missing keys independently. Under this condition, default keys equal original template indices; explicit keys must be unique. Bridge enumeration, result decoding, and AFIR use the shared `template_keys.h` helpers for validation and resolution without changing ATT encoding. Missing or ambiguous mappings are contract errors; do not choose the first match.

For example, in original subgraph 0 / plan 2, stage 0 template 3 maps to key 17, and stage 1 template 1 maps to key 9. When ATT returns that combination, static code generation emits only those two stage templates. They retain plan 2 and the original template coordinates; a single remaining plan must not become plan 0. Dynamic generation retains the existing dispatch implementation for every combination, and runtime keys 17/9 still map to the same templates. Even when a flat schedule omits the graph selector, template key meanings remain unchanged.

Schema field paths, workspace ownership, AFIR attributes, dispatch, and diagnostics use these identities. Physical reuse does not replace logical identity. The winner, per-stage keys, and tiling parameters are delivered together from one solve. This contract does not change ATT's external key encoding; it removes independent inference in bridge/codegen.

### Functional Requirement 2: Static Selection and Freezing

1. Introduction: Freeze the execution solution for the concrete invocation using real ATT results.
2. Input: All candidates, schema, ATT program, concrete shape, resource budget, and scenario configuration.
3. Processing: Invoke the executor, decode the winning identities and all required fields, and verify exactly one winning plan per subgraph and an unambiguous template per stage. All stage and workspace information must come from the same solve. Lower only the complete ATT-selected solution: one winning plan per subgraph and one selected template per stage, including every required stage of that plan. Only a single-subgraph, single-stage case generates just one template; RCore must generate both phase 1 and phase 2 of the winning plan. Unselected candidates do not enter AFIR lowering, but ATT still receives the complete candidate set before solving. Do not solve again to fill missing parameters.
4. Output: An indivisible static result containing selections, concrete tiling, and solving resource context. Failure must not return a partially usable result.

**If static ATT does not select a complete solution, PlanSelector returns failure and stops subsequent lowering on the current MLIR path.** It must not generate all candidates, select a second-best solution, or use approximate scoring. The outer MLIR mode policy, controlled by environment variables, determines whether to report an error or fall back to legacy for the whole graph. PlanSelector must not independently classify these failures as nonrecoverable. If ATT selects a solution but lowering lacks capability, the outer layer also handles that failure without changing the selection.

Failure origins and propagation responsibilities:

- ATT determines whether a solution exists under the current input, resource, and configuration constraints and provides its existing diagnostics on failure. Failure under DFX pinning only means failure under those constraints; it does not prove that every candidate is infeasible.
- The executor preserves ATT failures and diagnostics. The bridge preserves the failure category and propagates it to the outer MLIR mode policy for an error or legacy fallback. It must not log and continue or mislabel solving failure as insufficient lowering capability.
- Missing valid output, toolchain failure, invalid keys, or incomplete stage combinations are not necessarily errors raised by the ATT solver itself. The executor and bridge detect these errors, terminate the current MLIR compilation path, and preserve their actual causes rather than reporting every failure as "ATT has no solution."

The WIP generated call is `if (!optiling::GetTiling(tiling, -1, nullptr)) return 3;`: ATT's return value determines failure, and the wrapper converts it to a process exit code. This Boolean alone cannot distinguish all internal causes. Implementation must preserve ATT diagnostics and distinguish solver failure from compilation failure and abnormal exit. The required propagation must not be assumed to be already implemented.

When mapping a complete-schema result to the final static ABI, use existing field identities and types; do not copy raw structure memory between different layouts. Validate field widths, types, scopes, missing/duplicate fields, and bounds. Only fields known to belong to ungenerated candidates may be removed. Preserve block_dim, workspace sizes/offsets, and all execution-required fields.

Frozen parameters are valid only under the same input and resource contract. Artifact reuse must include shape, target resources, and relevant configuration identity. Explicitly reject reuse with incompatible resources rather than invoking a runtime solver that could select an ungenerated template.

### Functional Requirement 3: Retain All Templates for Dynamic Inputs

1. Introduction: Do not choose a winner at compile time; ATT selects using actual inputs at runtime.
2. Input: All candidates, symbolic schema, the same target resource contract, and the ATT program.
3. Processing: Generate AFIR and dispatch for every candidate and verify that every possible runtime key has corresponding code. If any candidate fails lowering, follow the existing whole-path strict/fallback policy; do not silently delete it.
4. Output: All selections and the runtime tiling program, without a fabricated static parameter bundle.

### Functional Requirement 4: ATT Execution Boundary

1. Introduction: The executor runs a prepared program; it does not enumerate candidates, generate schema, or query the platform.
2. Input: ATT program, invocation parameters, resource budget, and explicit host toolchain configuration.
3. Processing: Prefer existing execution capabilities. If none is suitable, retain temporary source compilation/execution and encapsulate files, subprocesses, diagnostics, and cleanup in the execution layer. Pass argv directly rather than constructing shell commands, and preserve failure stages and exit codes. Validate result encoding/decoding at the ATT adapter boundary; selection policy does not handle file protocols.
4. Output: Validated solving data or a distinct generation, compilation, execution, or result error. Timeout and cancellation must terminate this invocation's subprocesses and reclaim resources without terminating other tasks.

### Functional Requirement 5: Unified Resource Budget

1. Introduction: Solving and lowering use consistent target resource facts rather than independently querying and interpreting them.
2. Input: Physical resources, execution-scenario reservation rules, and resource provenance obtained at the entry through existing platform capability interfaces.
3. Processing: Form one budget and pass it to ATT, AFIR target, and memory validation. Reuse TargetProfile/platform objects first and add fields only for missing semantics.
4. Output: Physical UB capacity, reserved UB, available UB, available vector core count, and compatibility identity.

Budget contract: `available_ub = physical_ub - reserved_ub`; validate ranges before subtraction. Deduct each reservation once and identify ownership of tiling and runtime reservations. If ATT requires available capacity, pass available capacity. If AFIR stores physical capacity, derive the same available budget through the unified rule rather than mixing differently defined values under one name.

Distinguish available core count from ATT-selected block_dim. The selected value comes from the solver and must not exceed available resources. GM workspace is not deducted from UB. Workspace configuration comes from the winning solution and is cross-checked against launch and allocation interfaces.

## Nonfunctional Requirements

### Maintainability

PlanSelection retains coordinates and structural validation; PlanSelector expresses static/dynamic policy; AttSchemaAdapter generates programs and maps fields and keys; the ATT executor handles the toolchain; CodegenOrchestration connects them and applies the external failure policy. Do not create another axis, template, or resource classification table.

### Testability

Policy UT injects ATT results or execution stubs. Executor tests cover actual process success/failure. Real ATT integration tests verify their connection rather than relying only on fabricated results. Reuse PlanSelectorPolicyTest, AttSchemaAdapterTest, bridge UT, and lit.

### Portability

Support the existing host toolchains. Compiler and CANN paths enter through configuration boundaries. Preserve separate capability sources for CPU simulation and NPU execution without hardcoded chip or framework branches. Static execution with an unconfigured toolchain must fail explicitly rather than use approximate scoring.

### Reliability

Commit results only after complete validation; failures must not retain old selections or tiling. Input schedules are read-only, and temporary directories are isolated per invocation. Retain the existing 120-second process timeout and distinguish compilation from execution failure. Toolchain errors must not masquerade as capability limitations. Do not use globally mutable "last solving result" state.

### Cross-Feature Impact

| Scenario | Applicable | Conclusion |
|---|---|---|
| SuperKernel Python | No | No interface or wheel changes |
| SuperKernel C++/AOT | No | No libascendsk or AOT path changes |
| Autofuse graph optimization | Yes | No graph changes; preserve Schedule identities, group roles, and copy semantics |
| Autofuse Codegen/Backend | Yes | Static generation scope, dynamic dispatch, tiling, and workspace are core validation areas |
| AscendC/Runtime | Yes | No new device API; validate launch core count, parameters, and workspace lifetime |
| Python/C++ bindings | Yes | Public signatures stay unchanged; validate compilation failure propagation and actual module imports |
| Build and packaging | Yes | Static execution depends on the host compiler and CANN; validate installed headers/libraries without new network dependencies |
| Tests and coverage | Yes | Layered UT, lit, CPU, and NPU validation |
| Performance and logging | Yes | Measure ATT generation, compilation, execution, and total compilation time separately |
| Compatibility | Yes | Preserve DFX, public ABI, field and template identity contracts; do not reuse old caches across incompatible budgets |

## Performance

### Compilation Time

Each compilation request enumerates at most once and generates ATT once. Static compilation solves once; dynamic compilation does not solve at compile time. Reuse artifacts rather than solving again for logging. Record ATT generation, host compilation, execution, lowering, and total time separately. Use the original WIP timings as a baseline and retain existing performance gates. Establish baselines where thresholds do not exist rather than claiming no regression.

### Execution Performance

Static artifacts use frozen parameters without rerunning the solver. Dynamic artifacts retain the existing solving path. Under identical resources/configuration, templates and execution parameters must match the real ATT baseline. Do not relax numerical tolerances or performance thresholds for acceptance. Measure with observation enabled and disabled separately.

### Memory and Artifact Size

Avoid copying the entire schedule to narrow generation scope; hold read-only references within existing lifetimes. Static artifacts contain only winning code and required parameters, while dynamic artifacts retain all code. Record temporary host artifact sizes, final code size, and peak memory. Do not add a persistent solver cache.

## Interface Design

### New or Modified Interfaces

The logical internal contracts are shown below. Names are implementation suggestions, not a requirement for a general plugin framework:

```text
PreparePlanGeneration(schedule, mode, scene, targetBudget, attOptions)
  -> StaticPlanResult | DynamicPlanResult | existing diagnostic

StaticPlanResult = selections + solvedTiling + targetBudget
DynamicPlanResult = allSelections + attProgram/schema + targetBudget

ExecuteAtt(attProgram, invocation, hostToolchain)
  -> raw solver result | execution diagnostic
```

The orchestration entry owns the final result. ATT programs and schemas remain valid throughout preparation/execution. Borrowed schedule references must not escape the invocation lifetime. Static solvedTiling is mandatory; remove the optional output pointer. Use the existing error system for stages and causes instead of adding external status codes.

### Interface Checklist

| Item | Subitem | Applicable | Description |
|---|---|---|---|
| Documentation | Interface review | Yes | Internal policy/execution boundaries and budget contract |
| Documentation | Documentation updates | Yes | This design and the implemented call chain |
| Compatibility | Behavior | Yes | Preserve static winner selection and dynamic retention of all candidates |
| Compatibility | ABI/API | Yes | Adjust internal signatures; preserve public ABI and inspect actual export boundaries |
| Constraints | Clear diagnostics | Yes | Distinguish no solution, toolchain failure, invalid keys, and budget mismatch |
| Testing | Dedicated cases | Yes | Separate policy, execution, decoding, and integration tests |

## Software Design

### Key Data Structures

Reuse `PlanSelection`, `TilingFieldSchema`, and field identities. `TilingCodeBundle` returns to representing ATT source artifacts. Separate concrete static parameters from the source bundle through mutually exclusive final result types. The budget holds explicit results from one query rather than introducing another platform database.

### Key Techniques and Algorithms

Enumerate candidates linearly in schedule order. Map winning keys through existing identity helpers. Build any required index within the current request, without a persistent template inventory. PlanSelector does not evaluate cost expressions or infer global identity from a local group_id.

Follow the [Codegen semantic naming rules](../../../../docs/zh/autofuse/design/features/codegen.md#template-semantic-naming) and [Schedule semantic propagation rules](../../../../docs/zh/autofuse/design/features/schedule.md#template-semantic-propagation). Do not add implementation categories. Propagate strategies/roles from Schedule and collect API features at final API selection points. Carry complete graph/result/group/key/subcase/template and reuse logical identities in the winning snapshot. Do not solve again to obtain names or mask missing information with parent names.

### Process Design

Resolve target and mode, then prepare candidates/schema/ATT uniformly. Execute, validate, and freeze static results; deliver the program directly for dynamic inputs. Lower selections and assemble artifacts. Incomplete static ATT solutions and toolchain/result contract errors terminate the current MLIR path, preserve error categories, and defer to the outer mode policy. Do not deliver partial artifacts. Invalid dynamic runtime keys must report errors rather than defaulting to plan 0.

### Submodule Changes

| Module | Required change |
|---|---|
| PlanSelector/PlanSelection | Remove duplicate enumeration and ineffective unsupported output; deliver policy results together |
| AttSchemaAdapter | Separate ATT preparation and execution; make scenario and resource inputs explicit |
| CodegenOrchestration | Unify preparation, remove misleading narrowed_result naming and unnecessary copies, and unify target resolution |
| target / memory budget consumers | Align physical, reserved, and available capacities; prevent duplicate deductions |
| Static parameter / artifact assembly consumers | Require static results and validate field mappings, workspace, and dispatch identities |
| UT / lit / E2E | Add contract boundary cases using existing fixtures |

Implementation order: first clarify responsibilities and compare existing behavior, then complete the budget and atomic result contracts. Both parts must be complete before acceptance. Completing the first step alone does not establish that budget inconsistencies are fixed.

### Error Handling

#### System Errors

File errors, missing compilers, linking failures, process exits, timeouts, and read failures report their actual stage with controlled diagnostics. Missing solver output does not mean ATT found no solution. Cleanup covers failure, cancellation, and success.

#### Interface Errors

Malformed structures, missing fields, duplicate/unknown keys, invalid byte lengths, and inconsistent target budgets are contract errors. Record ATT rejection, infeasible solving, and insufficient lowering capability separately. PlanSelector does not decide whether legacy fallback is allowed; the outer MLIR environment-variable policy handles each category. Infrastructure failures and corrupted data retain their true category rather than being disguised as unsupported behavior.

## Safety Checks

The design has been checked against the coding red lines: validate indices, keys, and field lengths; guard budget subtraction and workspace sizes against overflow; define ownership of temporary directories and processes; avoid shell command construction and hardcoded sensitive information/chips; preserve graph dependencies and public ABI; and avoid cross-SO static-destructor cleanup. Implementation requires another review of actual code; this document does not replace it.

## Compatibility Checks

Preserve scripts, installation layout, Python APIs, ATT DFX, and kernel entries. Internal result types do not escape the boundary. If the static ABI removes unselected fields, assemble it through existing schema mappings and verify artifact cache versions. Incompatible old artifacts must not be reused. CPU and device execution retain their verified resource sources; do not replace them with one fictitious capacity merely to pass tests.

## Test Design

### Test Boundaries

Policy gtest may stub the executor. Decoding tests use real schemas and manually constructed invalid data. Executor tests cover temporarily compiled programs. Integration uses real ATT without stubbing winners. lit validates AFIR template identity and static parameter propagation; CPU/NPU tests cover compilation through numerical results. Preserve all existing numerical tolerances.

### Test Cases

| Category | Key checks | Method | Type |
|---|---|---|---|
| Functional | Nonzero static winning plan/template; noncontiguous explicit keys | Compare winning identities and parameters against real ATT with identical inputs | gtest/ST |
| Functional | Multiple subgraphs/stages; RCore AR/RA | Preserve complete stage combinations; inspect phase 1 workspace and phase 2 consumption | ST/CPU/NPU E2E |
| Functional | Different templates selected for dynamic shapes | Check all candidate code, actual dispatch keys, and numerical results | ST/NPU E2E |
| Functional | Single candidate still requires solving | Assert concrete parameters are obtained; candidate count alone cannot bypass ATT | UT/ST |
| Functional | Sum + Max shared scratch | Compare TBuf/queue, tiling, workspace, and results under identical budgets | CPU/NPU E2E |
| Identity | Reject mixed explicit/default keys; noncontiguous keys; equal keys across stages; duplicate keys within a stage; flat schedule | Compare ATT numbering, original static identities, and dynamic dispatch; reject same-stage ambiguity | UT/ST |
| Identity | Reuse, multiple groups, semantic name snapshots | Use shared helpers and verify physical primary does not overwrite logical identity | UT/ST |
| Budget | Physical/reserved/available capacity and core-count bounds | Test zero, excessive reservation, exact boundaries, and unsigned underflow; ATT and AFIR agree | gtest |
| Failure | Static ATT failure in strict and fallback-enabled modes | Stop current MLIR lowering in both modes; strict reports an error, fallback follows outer legacy policy; retain ATT diagnostics | bridge UT/ST |
| Failure | No solution, invalid keys, missing/duplicate fields, invalid widths, truncated output | No partial results, plan 0 default, or approximate scoring fallback | gtest |
| Failure | Missing host compiler, compilation/execution failure, timeout | Check classification, diagnostics, and process/directory cleanup | UT/ST |
| Failure | Unsupported static winner or dynamic candidate | Verify whole-path policy without second-best selection or silent candidate deletion | bridge UT/ST |
| Compatibility | DFX-pinned plan/case, scenario options, reuse after resource changes | Preserve pinned identity and reject incompatible artifacts explicitly | UT/E2E |
| Delivery | Installed CANN headers/libraries and Python extension | Compile the solver and import modules in a real environment without accidental source-tree dependencies | ST/NPU |
| Performance | Enumeration/generation/execution counts and per-stage timings | One static solve, zero dynamic compile-time solves; record total time, memory, and artifact size | UT/Benchmark |

Numerical correctness alone does not prove correct selection. Every selection test checks actual execution identity. Dynamic tests cover at least two actual winners. At least one RCore shape satisfies `nA < corenum`, avoiding false coverage from single-core execution or inactive R-axis core splitting.

### Test Commands

In a build tree with MLIR tests configured:

```bash
cmake --build build --target check-ut check-lit check-e2e-cpu -j 8
```

On Mac, use the repository development container:

```bash
AF_CANN_VOLUME=cann-volume bash autofuse/mlir/scripts/enter_dev_container.sh -- \
  bash scripts/test/run_autofuse_test.sh -u -s -m mlir -j 8
```

On the device, configure the environment through the existing unified entry and run:

```bash
python3 autofuse/mlir/test/e2e/npu/run.py --list
python3 autofuse/mlir/test/e2e/npu/run.py
```

Integrate real ATT static/dynamic fixtures into existing test targets rather than creating a parallel runner. Device validation records commit, toolchain, resource budget, module provenance, and actual keys. Report fallback separately; it does not count as an MLIR pass. Record implementation results below and leave unfinished items pending acceptance.

### Implementation and Validation Record (2026-10-31)

The unified `PreparePlanGeneration` entry was implemented, with separate variant branches for static results and dynamic ATT source. A single TargetProfile supplies ATT, AFIR, and the host ABI; reserved UB is deducted once. Field snapshots include each stage's own resource fields rather than substituting same-named fusion-scope values. Static aggregation preserves the original winning plan ID, while dynamic generation validates complete candidate numbering.

The standalone ATT program reuses `TilingLib::GenGetWorkspaceSizeFunc` for real workspace queries, satisfying ATT host dependencies. One invocation captures workspace size and offsets; the static ABI returns the frozen size without reevaluating symbolic formulas. No placeholder that always returns zero is provided. Field mapping failures identify the scope/field. The outer fallback policy remains unchanged, and failures deliver no partial artifacts.

Device validation found ATT split markers incorrectly included in frozen single-file tiling, causing Inductor to report `split host source has no TilingHead`. Static artifacts were changed to the existing single-file compilation protocol, with bridge UT asserting the absence of split markers. Dynamic ATT's multi-file protocol and Python-side validation remain unchanged.

- Selector, protocol, and bridge UT: 201/201 passed, including static nonzero-plan aggregation, rejection of dynamic numbering gaps, and negative resource/field protocol regressions.
- Full lit: 271/271 passed. Existing Inductor Python adapter single-file/split protocol tests: 40/40 passed.
- CPU E2E: Three numerical tests each passed for RCore AR/RA. One numerical test each passed for Sum + Max, static broadcast, and dynamic broadcast. RCore logs verified actual `ATT_SELECTED plan=1`.
- NPU: DevEnvC_1AtLk / CANN 9.2.0, workspace snapshot `311622910d3cdca05ddb03055bd51d1cea363217`. Run directory: `/mnt/workspace/af-npu/runs/311622910d3c-45daa5776fe0`. All 14 Reduce entries and 62 matrix cells passed, with zero fallbacks, defects, or expected deviations. This was not a full regression of the entire NPU inventory.
- Additional dynamic validation reused `load_sum_store.py --dynamic`: default selection and DFX-pinned plan 1 each ran three shapes; both entries returned zero without changing tolerances or PASS expectations. Results are in `dynamic-v2-cases` and `dynamic-plan1-v2-cases`. The first supplementary driver used case IDs inconsistent with expected names, producing reconciliation false positives. After correction, it was rerun in new directories.
- Artifact identity audit: Each of 76 static host units contained one original plan, covering plans 0/1/2. Each of two dynamic scenarios had one compilation unit retaining plans 0/1/2. The run directory contains `artifact-identity-audit.json`.
- Dynamic native ABI: Direct calls to the dynamic artifacts' `kernel.so` read actual results for `ks0/ks1=(16,32), (16,37), (17,32)`. Default selection returned plan 0, block_dim=2, workspace=0 for all shapes. DFX pinning returned plan 1, block_dim=2, workspace=512. Results are in `abi-probe/*/result.json`; identity was not inferred only from environment variables.
- Device RCore frozen artifacts retained original plan 1. The input budget was 40 cores and 196608 bytes of physical UB, with 196352 bytes available after one reservation deduction.
- Local CANN LLD stalled in linking, so validation used system LLD 14. Device CANN LLD also took abnormally long; device host builds temporarily used GNU ld.bfd 2.42, recorded in `linker-override.json`. Project linker configuration was not changed. These results do not establish successful builds with the default CANN LLD.

## Acceptance Criteria

- [x] Both modes share complete candidate/schema/ATT preparation; static compilation performs one real solve.
- [x] Static lowering processes only the complete ATT-winning solution, including each subgraph's winning plan and all required stages; unselected templates do not enter lowering.
- [x] Static winning identities and parameters are inseparable; dynamic candidate coverage agrees with runtime keys.
- [x] Incomplete static ATT selection stops current MLIR lowering. The outer environment-variable policy selects error reporting or legacy fallback, and ATT causes remain distinguishable from executor/bridge errors.
- [ ] Behavioral validation covers complete multi-subgraph/RCore stages, explicit keys, reuse, and semantic name identities.
- [x] ATT and AFIR use the same resource budget, reservations are deducted once, and workspace is accounted for separately from UB.
- [x] Templates, tiles, block_dim, and workspace agree with real ATT; numerical and performance gates are not relaxed.
- [x] No solution, toolchain errors, protocol errors, and capability limitations are distinct; failures yield no partial artifacts or implicit alternative selections.
- [x] UT, lit, CPU E2E, and relevant NPU validation are complete; unexecuted and fallback items are reported explicitly.
- [ ] Record compilation time, peak memory, and artifact size; update call-chain documentation and check public compatibility boundaries.

Design document checks: Cross-feature checks, coding red lines, and template semantic rules have been read and covered. UT/ST/E2E strategies and performance measurements are specified. ATT execution retains the existing temporary-program approach. Explicit runtime PGO rejects static freezing, with the outer configuration deciding between an error and legacy fallback. Other runtime tuning combinations still require dedicated validation; passing the default configuration does not establish their coverage.

### Code Review Checklist

- [x] Coding red lines: Checked `docs/guidelines/编码红线.md`, including real ATT selection, absence of partial artifacts on failure, budget deductions, and temporary-directory lifetimes.
- [x] Cross-feature impact: Checked Autofuse bridge/codegen, host ABI, Python compilation entry, and device execution against `cross_feature_check.md`. SuperKernel is unchanged. The static single-file artifact protocol issue with Inductor was reproduced and fixed on hardware.
- [x] Contribution guidelines: Checked `CONTRIBUTING.md`. This change included only PlanSelector-related code, tests, and documentation, excluding other tasks. An independent Git snapshot supported traceable device validation; no PR was created in that round.
- [x] Code format: Relevant C++ files were formatted using the repository `.clang-format`; `git diff --check` passed.
- [x] Pre-commit: Confirmed clang-format/OAT requirements in `.pre-commit-config.yaml`. Full-repository OAT and packaging regression were not run in that round and are not counted as passed.
