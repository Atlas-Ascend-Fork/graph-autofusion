# AutoFuse 相关环境变量参考

本文汇总 GE 路径和 Inductor 路径下，AutoFuse 自动融合功能运行和调测过程中常用的环境变量及控制项。

## 目录

- [AutoFuse 共享环境变量](#autofuse-共享环境变量)
  - [`AUTOFUSE_FLAGS` 功能控制](#autofuse_flags-控制项)
  - [`AUTOFUSE_DFX_FLAGS` 调测控制](#autofuse_dfx_flags-控制项)
- [路径专属环境变量](#路径专属环境变量)
  - [Inductor 路径专属](#inductor-路径专属)
  - [GE 路径专属](#ge-路径专属)
- [使用注意事项](#使用注意事项)

---

## AutoFuse 共享环境变量

以下环境变量为 AutoFuse 核心功能控制项，同时适用于 GE 路径和 Inductor 路径。

| 环境变量 | 适用场景 | 作用、取值与使用约束 |
| :------- | :------- | :------------------- |
| `AUTOFUSE_FLAGS`     | GE 路径、Inductor 路径 | AutoFuse 功能控制。多个控制项使用英文分号分隔。                                                   |
| `AUTOFUSE_DFX_FLAGS` | GE 路径、Inductor 路径 | AutoFuse 调测控制，用于融合图 Dump、代码生成调测和 Auto Tiling 调测；多个控制项使用英文分号分隔。 |

## `AUTOFUSE_FLAGS` 控制项

`AUTOFUSE_FLAGS` 用于控制 AutoFuse 功能。

**仅开启基础 AutoFuse 融合功能（最简配置）：**

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

对于 GE 路径，以上配置用于开启基础 AutoFuse 融合功能。

对于 Inductor 路径，AutoFuse 通过 `torch.compile` 配置 `ascendc` 后端来开启，`AUTOFUSE_FLAGS` 主要用于配置扩展功能。

下表列出所有可选控制项，可根据需要组合使用：

| 控制项 | 适用场景 | 作用、取值与使用约束 |
| :----- | :------- | :------------------- |
| `--enable_autofuse` | GE 路径 | 控制整体自动融合功能是否开启。取值为 `true` 或 `false`，`false` 为默认值；未开启时，其他 AutoFuse 控制项均不生效。 |
| `--autofuse_enable_pass` | GE 路径 | 控制指定的扩展融合能力是否开启。目前支持 `reduce` 和 `concat`；多个取值使用英文逗号分隔，默认不配置，扩展融合默认关闭。不能与 `--autofuse_disable_pass` 配置相同取值。 |
| `--autofuse_disable_pass` | GE 路径 | 控制指定的扩展融合能力是否关闭。支持配置 `reduce`、`concat`，也可以使用英文逗号分隔，同时关闭多个扩展融合能力。默认不配置；不能与 `--autofuse_enable_pass` 配置相同取值。 |
| `--autofuse_enhance_precision_blacklist` | GE 路径 | 控制指定 AscIR 算子类型是否跳过精度提升。取值为 AscIR 算子类型字符串，多个类型使用英文逗号分隔，也可配置为 `all`；默认值为空。`Sum`、`Mean`、`Prod` 不支持低精度类型，即使加入黑名单也仍会提升精度。配置 `Div`、`Reciprocal`、`Rsqrt`、`Sqrt`、`Pow` 或 `all` 后，`Div`、`Reciprocal`、`Rsqrt`、`Sqrt`、`Pow` 算子在代码生成时关闭高精度模式，按默认精度生成，可能影响计算精度。 |
| `--recomputation_threshold` | GE 路径 | 设置自动融合重计算阈值。取值为 `0`～`255` 的整数，默认值为 `1`。 |
| `--max_fusion_size` | GE 路径 | 设置单个融合算子最多包含的节点数量。取值为 `0`～`uint64_t` 最大值，配置为 `0` 表示不融合，默认值由实现决定。 |
| `--autofuse_enable_pgo` | GE 路径、Inductor 路径 | 开启 PGO 调优，通过预先上板采样选择性能更优的 Tiling。取值为 `true` 或 `false`，默认值为 `false`。仅支持静态图调优，需要准备对应版本的 `mspti`；首次配置时不能与其他 Profiling 功能同时开启。 |
| `--experimental_enable_jit_executor_v2` | GE 路径 | 开启切图编译。取值为 `true` 或 `false`，默认值为 `false`。动态分档、资源类算子、V1 控制流算子、数据预处理下沉和部分 AOE 调优场景不支持该功能。 |

示例：

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce,concat"
```

## `AUTOFUSE_DFX_FLAGS` 控制项

`AUTOFUSE_DFX_FLAGS` 用于 AutoFuse 编译、Auto Tiling 和融合结果调测。

| 控制项 | 适用场景 | 作用、取值与使用约束 |
| :----- | :------- | :------------------- |
| `--codegen_compile_debug` | GE 路径、Inductor 路径 | 控制是否保留融合算子生成过程中的中间文件。取值为 `true` 或 `false`，默认值为 `false`；开启后保留 Kernel、Tiling、CMake 工程、编译结果并生成融合图 Dump。 |
| `--debug_dir` | GE 路径、Inductor 路径 | 指定融合过程中 AscGraph Dump 图的保存路径。取值为有效目录路径，未指定时保存到当前执行目录。需要先开启 `--codegen_compile_debug=true`，且执行用户需要具备目标路径的读、写和执行权限。 |
| `--autofuse_att_algorithm` | GE 路径、Inductor 路径 | 选择 Auto Tiling 求解算法。取值为 `AxesReorder`（默认）或试验性的 `HighPerf`。`HighPerf` 不保证一定获得更好的执行性能；非法值恢复为默认值。 |
| `--att_accuracy_level` | GE 路径、Inductor 路径 | 控制 Auto Tiling 算法的求解精度。取值为 `1`（高精度求解）或 `0`（低精度求解），默认值为 `1`。高精度求解可能得到更优的 Tiling，但会增加 Tiling 执行时间；非法值恢复为默认值。 |
| `--att_enable_multicore_ub_tradeoff` | GE 路径、Inductor 路径 | 控制多核利用率与 UB 利用率权衡策略是否开启。取值为 `true` 或 `false`，默认值为 `false`；非法值恢复为默认值。 |
| `--att_ub_threshold` | GE 路径、Inductor 路径 | 设置 Auto Tiling 的 UB 利用率阈值。取值为 `0`～`100` 的整数，默认值为 `20`。需要配合 `--att_enable_multicore_ub_tradeoff=true` 使用；非法值恢复为默认值。 |
| `--att_corenum_threshold` | GE 路径、Inductor 路径 | 设置 Auto Tiling 的多核利用率阈值。取值为 `0`～`100` 的整数，默认值为 `40`。需要配合 `--att_enable_multicore_ub_tradeoff=true` 使用；非法值恢复为默认值。 |
| `--att_profiling` | GE 路径、Inductor 路径 | 控制 Auto Tiling Profiling 是否开启。取值为 `true` 或 `false`，默认值为 `false`；仅用于定位 Auto Tiling 模块的执行时间问题，非法值恢复为默认值。 |
| `--disable_lifting` | GE 路径 | 控制是否关闭 Lifting。取值为 `true`（关闭）或 `false`（开启），默认值为 `false`。仅用于定位 AscBackend 回滚问题，开启后可能导致 `ApplyAdamD` 算子精度异常。 |
| `--autofuse_pgo_algo` | GE 路径 | 选择 PGO 调优算法。取值为 `core_select`（默认）或 `pruning`。需要配合 `--autofuse_enable_pgo=true` 使用；非法值恢复为默认值。 |
| `--autofuse_pgo_step_max` | GE 路径 | 设置 PGO 剪枝算法步长。取值为 `2`～`1024` 范围内的 2 的幂，默认值为 `16`。仅在 `--autofuse_pgo_algo=pruning` 生效；非法值恢复为默认值。 |
| `--autofuse_pgo_topn` | GE 路径 | 设置参与 PGO 静态调优的候选解数量。取值为 `0` 或任意正整数，默认值为 `5`，`0` 表示选择全部候选解。需要配合 `--autofuse_enable_pgo=true` 使用；非法值恢复为默认值。 |
| `--skip_node_names_cfg` | GE 路径 | 设置需要跳过融合的算子名称或算子类型。取值为有效 `.ini` 配置文件路径。配置文件中的名称或类型分别写在 `[ByNodeName]` 或 `[ByNodeType]` 段中，每个条目单独占一行；非法内容会导致该配置项不生效。 |

示例：

```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/path/to/dump"
```

---

## 路径专属环境变量

以下环境变量为对应路径专属，仅用于特定路径的调测：

### Inductor 路径专属

这些环境变量用于 Inductor 路径的编译或运行调测，不用于 GE 路径：

| 环境变量                               | 说明                                                                                                                                                             | 使用方式                                        |
| :------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------- |
| `TORCH_COMPILE_DEBUG`                | 开启 PyTorch 编译调试信息，并将编译中间产物保存到当前目录的`torch_compile_debug` 目录。以 `autofused_` 为前缀的目录通常表示 AscendC 后端生成的融合算子产物。 | `export TORCH_COMPILE_DEBUG=1`                |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES` | 禁用 Inductor 缓存，强制每次执行都重新编译。该配置会增加编译和图启动耗时，仅用于调试。                                                                           | `export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` |
| `ASCEND_LAUNCH_BLOCKING`             | 使 Ascend Kernel 同步执行，便于定位首个报错的 Kernel。该配置会降低执行性能，仅建议在问题定位时使用。                                                             | `export ASCEND_LAUNCH_BLOCKING=1`             |

### GE 路径专属

当前没有 GE 路径专属环境变量，所有 GE 路径相关控制项均已包含在 AutoFuse 共享环境变量中。

---

## 使用注意事项

- 调测环境变量会增加编译或运行开销，完成问题定位后请及时取消设置。
- 详细控制点说明请参见 [AUTOFUSE_FLAGS 环境变量控制点](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/graphdevg/autofuse_1_0061.html) 和 [AUTOFUSE_DFX_FLAGS 环境变量控制点](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/graphdevg/autofuse_1_0062.html)。
