# SuperKernel 迁移设计

## 概览&目的

本文分析 SuperKernel 与 AscendC 解耦方案，将 SuperKernel 实现从当下的 AscendC 仓、包迁移到 Graph-autofusion 。

希望通过解耦，SuperKernel 可做到独立演进，SuperKernel 新增特性、Bugfix 不依赖升级 AscendC。

如果不做特别说明，本文提及的 AscendC 包含 AscendC 工程与 AscendC API。所谓 AscendC 工程，是指以 compile_op 为入口的 AscendC 编译工程基础设施；所谓 AscendC API 是 AscendC 开发者使用的 AscendC API 类库。

## Proposed Implementation

要将 SuperKernel 拆出，主要有如下内容需要考虑：

1. **SuperKernel 工程编译入口拆出**：`ascendc_super_kernel.ascendc_super_kernel_plus`为入口的，专为 SuperKernel 服务的编译工程基础设施，拆分到 `Graph-autofusion` 仓
2. **gentask 逻辑解耦**：CANN 组件图模式适配
3. **单个算子 AscendC 工程中对 SuperKernel 的适配解耦**：`compile_op.compile_op`为入口的，为 SuperKernel 做判断的分支，做特性化处理，或其他消减措施，目标为 `compile_op.compile_op` 不感知 SuperKernel 场景
4. **AscendC API 对 SuperKernel 的解耦**：AscendC API 与 SuperKernel 的耦合主要体现在，在 API 实现中判断 SuperKernel 编译宏，并作特别处理，解耦后，目标为所有 AscendC API 不应感知 SuperKernel 场景

由前期的分析认为：

* 内容 1（SuperKernel 工程编译入口拆出）、2与内容3、4相对独立，可以分阶段进行，降低风险
* 内容 3、4 由 17 个耦合点组成，可以并行分析、设计、落地


### 1 SuperKernel 工程编译拆分

#### 拆分文件和函数概览

以`ascendc_super_kernel.ascendc_super_kernel_plus`为入口，如下文件为 SuperKernel 特性专有，所以拆分到`Graph-autofusion`：

| 文件名 | 功能 | 新文件 |
| ----- | --- | ----|
| ascendc_super_kernel.py | SuperKernel 编译入口，包含参数解析、codegen 壳代码、编译 | super_kernel.py (入口由ascendc_super_kernel_plus变为compile) |
| ascendc_super_operator_infos.py | SuperKernel 顶层数据结构 | super_kernel_op_infos.py |
| ascendc_sub_super_kernel.py | 子kernel信息保存 | super_kernel_sub_op_infos.py |

如下函数为 SuperKernel 专有，普通算子编译流程不需使用，所以拆分到`Graph-autofusion`

| 函数名 | 功能 | 新文件 |
| ----- | --- | --- |
| CommonUtility.is_support_super_kernel | 判断当前芯片类型类型是否支持 SuperKernel | super_kernel_utils.py |
| ascendc_constants.SubOperatorType | 子算子动、静类型，其中STATIC_OP_WITH_SEND\STATIC_OP_WITH_RECEIVE可删除 | super_kernel_types.py |
| ascendc_constants.STR_TO_SUPER_TASK_TYPE | string类型映射ascendc_constants.SubOperatorType |super_kernel_types.py |
| ascendc_constants.SuperKernelPreLoadMode | 子kernel指令预取模式选择 |super_kernel_types.py |
| ascendc_constants.SuperKernelLinkMode | 子kernel链接模式选择 |super_kernel_types.py |
| ascendc_constants.SuperKernelEarlyStartMode | 子kernel间early-start模式选择 |super_kernel_types.py |
| ascendc_constants.SuperKernelStreamFusionMode | 是否使能双流 |super_kernel_types.py |
| ascendc_constants.SuperKernelDebugDcciAllMode | 是否在子算子末尾增加entry cache的dcci |super_kernel_types.py |
| ascendc_constants.SuperKernelDebugSyncAllMode | 是否在子算子末尾增加全核同步 |super_kernel_types.py |
| ascendc_constants.SuperKernelFeedSyncAllMode | 是否支持算子不满核情况下使用全核硬同步 |super_kernel_types.py |
| ascendc_constants.SuperKernelDataCacheMode | 子kernel数据预取模式 |super_kernel_types.py |
| ascendc_constants.SuperKernelProfilingMode | 支持SuperKernel profiling数据 |super_kernel_types.py |
| ascendc_compile_gen_code.gen_super_dump_code | 生产SuperKernel dump的初始化代码 |super_kernel_types.py |
| ascendc_compile_base.gen_symbol_rename_file | 拆分子kernel后，重命名 |super_kernel_compile_base.py |
|ascendc_common_utility.gen_func_align_attribute| 生成对齐attr | super_kernel_compile_base.py
| ascendc_compile_base.split_dynamic_o_in_super_kernel | 在SuperKernel中拆分子kernel为多个二进制 |super_kernel_op_infos.py |
| ascendc_common_utility.parse_super_kernel_options | SuperKernel 参数解析及校验 |super_kernel_options.py |
| OptionParser | 同上 |super_kernel_options.py |
| ParserFactory | 同上 |super_kernel_options.py |
| CodeTextAlignParser | 同上 |super_kernel_options.py |
| EnumParser | 同上 |super_kernel_options.py |
| BinaryParser | 同上 |super_kernel_options.py |
| NumberParser | 同上 |super_kernel_options.py |
| NonEmptyParser | 同上 |super_kernel_options.py |
| setup_super_kernel_option_parsers | 同上 |super_kernel_options.py |

如下函数为 SuperKernel 与常规算子共用，或者紧耦合，依赖（可能）不稳定，先行拆解，暂不迁移，考虑后续阶段在特性解耦中分析（不再使用、API标准化、或其他措施）

| 函数名 | 功能 | 新文件 |
| ----- | --- | --- |
| ascendc_compile_base.super_kernel_gen_entry | 顶层SuperKernel入口拆四分代码 | super_kernel_gen_entry_code.py |
| ascendc_compile_base.gen_spk_kernel_call | 同上 |super_kernel_gen_entry_code.py|
| ascendc_compile_base.split_spk_kernel_objs | 同上 |super_kernel_gen_entry_code.py |
| ascendc_compile_base.super_kernel_gen_entry | 同上 |super_kernel_gen_entry_code.py|
| ascendc_compile_base.gen_file_header | 同上 ps:生成双页表全局变量（代码实现仅支持v220，需考虑不同芯片版本的宏隔离，可参考ascendc_compile_gen_code.gen_global_isolation_macro）且只支持split_mode为1，双叶表才生效 |super_kernel_gen_entry_code.py|
| ascendc_compile_base.gen_system_run_cfg | 同上 |super_kernel_gen_entry_code.py|
| ascendc_compile_base.localize_symbol_of_sk | 链接SuperKernel所有二进制 |super_kernel_compile.py|
| ascendc_compile_base.gen_super_kernel_compile_info | 生成SuperKernel的compile info |super_kernel_compile.py|
| ascendc_compile_base.gen_sub_super_kernel_early_start_compile_options | 处理子kernel early-start特性的编译选项 |super_kernel_compile.py  |
| ascendc_compile_base.gen_sub_super_kernel_compile_options | 处理子kernel编译选项 |super_kernel_compile.py |
| ascendc_compile_base.split_sub_kernel_objs | 子kernel拆分成四个object文件 |super_kernel_compile.py |
| ascendc_compile_base.split_kernel_arch_str | 同上 |super_kernel_compile.py |
| ascendc_compile_base.split_kernel | 同上 |super_kernel_compile.py |
| ascendc_compile_base.gen_super_kernel_link_obj_sequence | 获取所有子kernel的二进制 |super_kernel_compile.py |
| ascendc_compile_base.localization_sub_op_func_sym | 拆分子kernel二进制后，修改子kernel global符号为local |super_kernel_compile.py |
| ascendc_compile_base.localize_symbol_of_sk | 修改SuperKernel的global符号为local |super_kernel_compile.py |
| compile_op.compile_super_kernel | SuperKernel编译入口 |super_kernel_compile.py |

如下接口，ascendc_sub_super_kernel.py、ascendc_super_kernel.py、ascendc_super_operator_infos.py文件有使用，需在super_kernel_utils.py新增接口封装，避免ascendc开源整改引入问题，影响SuperKernel功能：

| 函数名 | 功能 | 新接口 |
| ----- | --- | ------|
|tbe.common.platform.platform_info.get_soc_spec| 获取当前芯片类型 |get_soc_version |
|tbe.tvm.error_mgr.raise_tbe_python_err| 异常抛出 | super_kernel_raise_err |
|tbe.tvm.error_mgr.TBE_DEFAULT_PYTHON_ERROR_CODE| 异常码 | "EB0500" |
|tbe.common.buildcfg.get_current_build_config| 获取编译上下文信息 | get_compile_config_info |
|tbe.common.buildcfg.buildcfg.op_debug_config| 获取日志级别 | 集成进get_compile_config_info接口中 |
| log_utils及相关接口 | 日志模板 | 直接使用 |
| ascendc_common_utility及相关接口 | 工具模板 | 当前阶段一先直接使用 |
| global_storage.global_var_storage | 全局变量设置 | 分析是否可以删除 |

若有未尽函数，按照上述原则划分。

#### Graph-autofusion 目录规划

在 `src` 目录下，为 `SuperKernel` 创建独立目录：

```bash
├── doc
├── README.md
└── src
    └── superkernel
        ├── sub_operator_infos.py  --> 原 ascendc_sub_super_kernel.py 中的函数迁移到该文件中
        ├── super_kernel.py --> 原 ascendc_super_kernel.py 中的函数迁移到该文件中
        ├── super_operator_infos.py --> 原 ascendc_super_operator_infos.py 中的函数迁移到该文件中
        ├── super_kernel_compile_base.py  --> 原 ascendc_compile_base.py 中的函数迁移到该文件中
        └── super_kernel_constants.py   --> 原 ascendc_constants.py 中的函数迁移到该文件中
```

原则为：

* 文件迁移的：整文件迁移到`src/superkernel`目录下
* 函数/类等迁移的：文件前缀由`ascendc_`修改为`super_kernel_`，新建文件于`src/superkernel`目录下

#### 安装目录

安装在 `site-packages.tbe.superkernel`

### 2 GE 适配

todo GE/FE 按照 Graph-autofusion 中 SuperKernel 接口界面做重构适配

### 3 特性解耦

todo
