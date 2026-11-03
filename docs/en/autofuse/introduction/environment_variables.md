# AutoFuse Environment Variable Reference

This document summarizes commonly used environment variables and control options for AutoFuse during operation and debugging on the GE path and Inductor path.

## Contents

- [Shared AutoFuse Environment Variables](#shared-autofuse-environment-variables)
	- [`AUTOFUSE_FLAGS` Options](#autofuse_flags-options)
	- [`AUTOFUSE_DFX_FLAGS` Options](#autofuse_dfx_flags-options)
- [Path-specific Environment Variables](#path-specific-environment-variables)
	- [Inductor Path](#inductor-path)
	- [GE Path](#ge-path)
- [Notes](#notes)

## Shared AutoFuse Environment Variables

The following variables control core AutoFuse functions and apply to both the GE path and the Inductor path.

| Environment Variable | Applicable Paths | Purpose, values, and usage constraints |
| :--- | :--- | :--- |
| `AUTOFUSE_FLAGS` | GE path, Inductor path | Controls AutoFuse functions. Multiple options are separated by semicolons. |
| `AUTOFUSE_DFX_FLAGS` | GE path, Inductor path | Controls fusion-graph dumps, code-generation debugging, and Auto Tiling debugging. Multiple options are separated by semicolons. |

## `AUTOFUSE_FLAGS` Options

`AUTOFUSE_FLAGS` controls AutoFuse functions.

**Enable only the basic AutoFuse fusion function (minimal configuration):**

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

For the GE path, the above configuration enables the basic AutoFuse fusion function.

For the Inductor path, AutoFuse is enabled by configuring the `ascendc` backend through `torch.compile`. `AUTOFUSE_FLAGS` is mainly used to configure extended functions.

| Option | Applicable Paths | Purpose, values, and usage constraints |
| :--- | :--- | :--- |
| `--enable_autofuse` | GE path | Controls whether automatic fusion is enabled globally. Accepts `true` or `false`; `false` is the default. Other AutoFuse control options have no effect when it is disabled. |
| `--autofuse_enable_pass` | GE path | Enables specified extended fusion capabilities. Currently, `reduce` and `concat` are supported. Multiple values are separated by commas; the default is empty and extended fusion is disabled. The same value must not be configured with `--autofuse_disable_pass`. |
| `--autofuse_disable_pass` | GE path | Disables specified extended fusion capabilities. Supports `reduce` and `concat`; multiple capabilities can be disabled by separating their values with commas. The default is empty. The same value must not be configured with `--autofuse_enable_pass`. |
| `--autofuse_enhance_precision_blacklist` | GE path | Controls whether specified AscIR operator types skip precision enhancement. Accepts comma-separated AscIR operator type strings or `all`; default: empty. `Sum`, `Mean`, and `Prod` still require precision enhancement. When `Div`, `Reciprocal`, `Rsqrt`, `Sqrt`, `Pow`, or `all` is configured, `Div`, `Reciprocal`, `Rsqrt`, `Sqrt`, and `Pow` operators are generated with the default precision mode instead of the high-precision mode, which may affect accuracy. |
| `--recomputation_threshold` | GE path | Sets the automatic-fusion recomputation threshold. Accepts an integer from `0` to `255`; default: `1`. none. |
| `--max_fusion_size` | GE path | Sets the maximum number of nodes in a fused operator. Accepts `0` to the maximum `uint64_t` value; `0` disables fusion; the default is implementation-defined. none. |
| `--autofuse_enable_pgo` | GE path, Inductor path | Enables PGO tuning by selecting better-performing Tiling through pre-run sampling. Accepts `true` or `false`; default: `false`. static graphs only, `mspti` is required, and the first configuration cannot be used with other Profiling features. |
| `--experimental_enable_jit_executor_v2` | GE path | Enables split-graph compilation. Accepts `true` or `false`; default: `false`. dynamic bucketing, resource operators, V1 control-flow operators, data-preprocessing sinking, and some AOE tuning scenarios are unsupported. |

Example:

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce,concat"
```

## `AUTOFUSE_DFX_FLAGS` Options

`AUTOFUSE_DFX_FLAGS` is used to debug AutoFuse compilation, Auto Tiling, and fusion results.

| Option | Applicable Paths | Purpose, values, and usage constraints |
| :--- | :--- | :--- |
| `--codegen_compile_debug` | GE path, Inductor path | Controls whether intermediate files and debug artifacts are retained. Accepts `true` or `false`; default: `false`. When enabled, Kernel, Tiling, CMake projects, compilation results, and fusion-graph dumps are retained. none. |
| `--debug_dir` | GE path, Inductor path | Specifies the directory for AscGraph dump files. Accepts a valid directory path; defaults to the current execution directory. requires `--codegen_compile_debug=true` and read, write, and execute permissions on the path. |
| `--autofuse_att_algorithm` | GE path, Inductor path | Selects the Auto Tiling algorithm. Accepts `AxesReorder` (default) or experimental `HighPerf`. `HighPerf` does not guarantee better performance; invalid values fall back to the default. |
| `--att_accuracy_level` | GE path, Inductor path | Controls Auto Tiling solving accuracy. Accepts `1` for high-accuracy solving or `0` for low-accuracy solving; default: `1`. high-accuracy solving may find better Tiling but takes longer; invalid values fall back to the default. |
| `--att_enable_multicore_ub_tradeoff` | GE path, Inductor path | Controls whether the multicore-utilization and UB-utilization trade-off is enabled. Accepts `true` or `false`; default: `false`. invalid values fall back to the default. |
| `--att_ub_threshold` | GE path, Inductor path | Sets the UB-utilization threshold for Auto Tiling. Accepts an integer from `0` to `100`; default: `20`. requires `--att_enable_multicore_ub_tradeoff=true`; invalid values fall back to the default. |
| `--att_corenum_threshold` | GE path, Inductor path | Sets the multicore-utilization threshold for Auto Tiling. Accepts an integer from `0` to `100`; default: `40`. requires `--att_enable_multicore_ub_tradeoff=true`; invalid values fall back to the default. |
| `--att_profiling` | GE path, Inductor path | Controls Auto Tiling Profiling. Accepts `true` or `false`; default: `false`. use only to locate Auto Tiling execution-time issues; invalid values fall back to the default. |
| `--disable_lifting` | GE path | Controls whether Lifting is disabled. Accepts `true` to disable or `false` to enable; default: `false`. use only to locate AscBackend rollback issues; enabling it may cause accuracy issues for `ApplyAdamD`. |
| `--autofuse_pgo_algo` | GE path | Selects the PGO algorithm. Accepts `core_select` (default) or `pruning`. requires `--autofuse_enable_pgo=true`; invalid values fall back to the default. |
| `--autofuse_pgo_step_max` | GE path | Sets the PGO pruning step. Accepts a power of 2 from `2` to `1024`; default: `16`. effective only when `--autofuse_pgo_algo=pruning`; invalid values fall back to the default. |
| `--autofuse_pgo_topn` | GE path | Sets the number of candidates for static PGO tuning. Accepts `0` or a positive integer; default: `5`; `0` selects all candidates. requires `--autofuse_enable_pgo=true`; invalid values fall back to the default. |
| `--skip_node_names_cfg` | GE path | Specifies operator names or types to skip during fusion. Accepts a valid `.ini` configuration path. entries must be placed one per line under `[ByNodeName]` or `[ByNodeType]`; invalid content makes this option ineffective. |

Example:

```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/path/to/dump"
```

## Path-specific Environment Variables

### Inductor Path

These variables are used for Inductor path compilation or runtime debugging and do not apply to the GE path:

| Environment Variable | Description | Usage |
| :--- | :--- | :--- |
| `TORCH_COMPILE_DEBUG` | Enables PyTorch compilation debugging and saves intermediate artifacts under `torch_compile_debug`. Directories prefixed with `autofused_` usually indicate fused operators generated by the AscendC backend. | `export TORCH_COMPILE_DEBUG=1` |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES` | Disables Inductor caches and forces recompilation on every execution. This increases compilation and graph-launch overhead and is intended for debugging. | `export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` |
| `ASCEND_LAUNCH_BLOCKING` | Makes Ascend Kernels execute synchronously to locate the first failing Kernel. This reduces performance and is recommended only for troubleshooting. | `export ASCEND_LAUNCH_BLOCKING=1` |

### GE Path

The GE path currently has no path-exclusive environment variables; all GE path-related options are included in the shared AutoFuse variables.

## Notes

- Debugging variables add compilation or runtime overhead. Unset them after troubleshooting.
- For details, see the [AUTOFUSE_FLAGS reference](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/graphdevg/autofuse_1_0061.html) and [AUTOFUSE_DFX_FLAGS reference](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/graphdevg/autofuse_1_0062.html).
