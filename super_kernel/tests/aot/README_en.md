# SuperKernel AOT Host Tests

This directory contains SuperKernel AOT unit tests (UT) and host-side system tests (ST).
ST enters the complete AOT host flow through public APIs and uses shared Runtime/ACL stubs
to supply external model and task state. ST links the test-only `libascendsk_st.so`, which
uses the same AOT source list as production `ascendsk`, including `sk_dump_json.cpp`,
and shares its compilation, visibility, ABI, and linker hardening settings.
The test library is not installed and does not link real device kernels or Runtime libraries.

## Layout and Boundaries

| Directory | Responsibility |
| --- | --- |
| `ut/` | Unit tests for internal classes and functions using gtest/mockcpp, including seven shared model fixture tests |
| `st/` | Cross-module tests through four public APIs, including only public API and test dependency headers |
| `depends/` | The `super_kernel_aot_stub` shared library, Runtime/ACL stubs, and external RI model fixtures |
| `cmake/` | Shared test options, execution helpers, and coverage scripts |

The ST executable and `ascendsk_st` link the same `super_kernel_aot_stub` shared library,
so fault injection, task state, allocation records, and destruction callbacks have a single owner.
`intf_llt_options` supplies ABI, coverage, and sanitizer options. The UT-specific `intf_llt_ut`
also introduces gtest/mockcpp; ST and the shared stub do not depend on mockcpp.

| Public API | Verification boundary |
| --- | --- |
| `aclskOptimize` | External RI model queries, scope processing, fused task replacement, cross-stream synchronization, model updates, and resource cleanup |
| `aclskScopeBegin` | Begin marker names, streams, dispatch behavior, and invalid inputs |
| `aclskScopeEnd` | End marker names, streams, dispatch order, and invalid inputs |
| `aclskScopeVerify` | Public graph validation and splitting, including output nodes referencing the original inputs |

ST covers the default whole-model scope without markers, single-scope fusion, preservation
of cross-stream events outside scopes, external wait/reset rewriting after fused record tasks,
synchronization memory cleanup, unmatched scopes, model update failures, begin/end markers,
and verification. Extended scenarios cover cube/MIX 1:1/MIX 1:2 entry selection, per-operator
debugging, merging same-name scopes across streams, debug JSON options and task ordering,
Runtime query/entry resolution/synchronization memory initialization failures, and Verify output
capacity, dynamic core limits, cross-stream deadlocks, and invalid inputs. Further scenarios
cover SIMT dynamic UBUF, multi-stream scopes and memory wait/write, option boundaries,
compiler capabilities and entry binding, mixed pipelines, DFX exception callbacks, and
profiling start/stop, export on exit, and failure cleanup.

Fixture UT covers query capacity and state preservation, model isolation and parameter snapshot
ownership, metadata isolation for identical names with different kernel types, deep copying of
parameters, non-kernel parameter copying, the compiler metadata byte protocol, and compatibility
with opaque device addresses used by existing UT.

Profiling and SIMT scenarios run in separate processes to isolate production singletons.
Child processes exit normally to verify recorder cleanup and write coverage data. A 15-second
timeout fails the test and reaps the process. Children do not inherit GTest sharding settings,
preventing exact-filtered tests from being omitted by a second round of sharding.
DFX only copies real fused entry arguments and provides external exception registers and
standard ELF symbols; it does not construct or parse private SK arguments. Profiling currently
checks empty-event export and does not simulate device-generated event records.

Test execution does not require an NPU. Top-level CMake configuration still locates CANN packages
and configures the ASC compiler, so a usable CANN Toolkit and environment are required.
These tests verify host flows and stub contracts, not real Runtime ABI compatibility, device
execution, or numerical correctness. Those properties require device tests.

## Running Tests

Configure the CANN Toolkit for your environment, then run from the repository root:

```bash
source /path/to/cann/set_env.sh
bash build.sh -s --module=superkernel --impl=cpp --no-autofuse -j 8
```

The dependency cache, test filter, and coverage options are optional. This example combines them:

```bash
bash build.sh -s --module=superkernel --impl=cpp --no-autofuse -j 8 \
    --cann_3rd_lib_path=/path/to/third_party \
    --test_case='AotSystemTest.Optimize*' -c
```

ST coverage is written to `super_kernel/coverage/cpp_st/`, with the HTML entry at `html/index.html`.
ST scans coverage data only in its own build directory. Reports retain only production files
under `super_kernel/src/aot`, excluding stubs, test cases, and third-party headers.
UT continues to use `super_kernel/coverage/cpp_ut/`. Filtered runs measure only the selected
scenarios, not the complete test suite. The `-c` route cleans build targets before running.

For an existing build directory, use the separate CMake switches and targets:

```bash
cmake -S . -B build -DBUILD_AUTOFUSE=OFF \
    -DENABLE_CPP_UTEST=OFF -DENABLE_CPP_STEST=ON -DENABLE_GCOV=OFF \
    -DGTEST_FILTER=
cmake --build build --target super_kernel_aot_stest -j 8
cmake --build build --target run_super_kernel_aot_stest -j 8
```

On first configuration, also provide `ASCEND_INSTALL_PATH` and `CANN_3RD_LIB_PATH` for your environment.
`ENABLE_CPP_UTEST` and `ENABLE_CPP_STEST` can be enabled separately or together. Set both explicitly
when switching to avoid inheriting cached values. `build.sh` explicitly sets the selected switches.
UT targets are `super_kernel_aot_utest` and `run_super_kernel_aot_utest`.
With `ENABLE_GCOV=ON`, the ST coverage target is `collect_coverage_data_cpp_st`; UT retains
`collect_coverage_data`. To configure a filter directly, pass
`-DGTEST_FILTER=--gtest_filter=AotSystemTest.Optimize*` and quote the entire argument to prevent shell expansion.

## Extending Fixtures and Tests

Add ST cases as `st/test_*.cpp`; CMake discovers them automatically. Reuse `AotSystemTest`
from `st_fixture.h`. Construct external models with `sk::test::Model`, using `AddStream`,
`AddKernel`, `AddEvent`, and `AddTask` for inputs. After calling public APIs, observe results
through `Tasks`, `Snapshot`, `Launches`, and update counts. Do not include private production
headers, construct internal `SkGraph` objects, or call internal optimizers instead of public APIs.

Express new Runtime behavior as external API contracts in `depends/`, add fixture UT, and then
verify production flows with ST. Fixtures manage handle validity, parameter ownership, and
observable state; they must not duplicate fusion algorithms to calculate expected results.
`SetParams` saves copies of host parameters, configuration attributes, and opInfo. Function
metadata remains stable within the process so production binary-cache references stay valid.
`KernelSpec` describes kernel types, ratios, and compiler capabilities. `SetKernelBindings`
overrides external binary bindings and must be called before Optimize first consumes that binary.
`AddTask` accepts only non-kernel tasks; callers must keep referenced external addresses valid.
The model fixture supports AIC/AIV/MIX metadata, record/wait/reset events, and memory wait/write.
Tests run serially; fixtures do not provide general device simulation or concurrent Runtime state.

Resource lifecycle ordering is destroy first, reset second: let `Model` destruct or explicitly
call `Model::Destroy()` to execute registered production cleanup callbacks before resetting
global stub state. To check resource release, call `Destroy()` before asserting allocation and
callback counts. Resetting first would erase state and hide leaks. Local models in tests are
destroyed before fixture `TearDown()`, preserving this ordering.

## Failure Diagnosis

The ST fixture prints buffered production logs on failure and clears them on success.
Execution targets print test output and errors and propagate failing exit codes.
A filter that selects no tests fails, as do skipped tests. The runner removes temporary logs
on exit. To retain a log, run:

```bash
set -o pipefail
bash build.sh -s --module=superkernel --impl=cpp --no-autofuse -j 8 \
    --test_case='AotSystemTest.*' 2>&1 | tee /tmp/super-kernel-aot-st.log
```

Distinguish CANN/dependency configuration failures, compilation/link failures, and assertion failures.
For undefined `testing::*` symbols, check that the gtest core library is linked. For resource
assertions, check model destruction versus stub reset ordering. Task types, disabled states,
parameter update counts, and synchronization addresses in assertions help identify rewriting differences.

Two production problem reproductions are retained: when a mixed pipeline's final task declares
both wait/set capabilities, early-start may create synchronization without an associated node
and fail; when a single kernel exceeds the device core limit, scope splitting may not terminate.
The latter test, `DISABLED_RuntimeCoreLimitKeepsOversizedKernelOutsideFusion`, is disabled by default
and must not be counted as passing. Do not enable it in regular runs before fixing production code.
