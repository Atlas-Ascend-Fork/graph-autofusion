# SuperKernel Migration Design

## Overview & Purpose

This document analyzes the SuperKernel and AscendC decoupling solution, migrating SuperKernel implementation from the current AscendC repository and package to Graph-autofusion.

Through decoupling, SuperKernel can evolve independently. SuperKernel new features and bugfixes do not depend on upgrading AscendC.

Unless otherwise specified, this document mentions AscendC includes both AscendC project and AscendC API. AscendC project refers to AscendC compilation project infrastructure with compile_op as entry point; AscendC API refers to AscendC API class library used by AscendC developers.

## Proposed Implementation

To extract SuperKernel, the following content needs consideration:

1. **SuperKernel project compilation entry extraction**: `ascendc_super_kernel.ascendc_super_kernel_plus` as entry point, compilation project infrastructure specifically for SuperKernel, extracted to `Graph-autofusion` repository
2. **gentask logic decoupling**: CANN component graph mode adaptation
3. **Single operator AscendC project SuperKernel adaptation decoupling**: `compile_op.compile_op` as entry point, branches judging for SuperKernel, feature-specific processing, or other reduction measures. Target is `compile_op.compile_op` does not sense SuperKernel scenarios
4. **AscendC API SuperKernel decoupling**: AscendC API and SuperKernel coupling mainly manifests in API implementation judging SuperKernel compilation macros and special processing. After decoupling, target is all AscendC APIs should not sense SuperKernel scenarios

From preliminary analysis:

* Content 1 (SuperKernel project compilation entry extraction), 2 and content 3, 4 are relatively independent, can proceed in phases, reducing risk
* Content 3, 4 consists of 17 coupling points, can parallel analyze, design, implement


### 1 SuperKernel Project Compilation Extraction

#### Extracted Files and Functions Overview

With `ascendc_super_kernel.ascendc_super_kernel_plus` as entry point, the following files are SuperKernel feature-specific, therefore extracted to `Graph-autofusion`:

| File Name | Function | New File |
| ----- | --- | ----|
| ascendc_super_kernel.py | SuperKernel compilation entry, includes parameter parsing, codegen wrapper code, compilation | super_kernel.py (entry changed from ascendc_super_kernel_plus to compile) |
| ascendc_super_operator_infos.py | SuperKernel top-level data structure | super_kernel_op_infos.py |
| ascendc_sub_super_kernel.py | Sub-kernel information storage | super_kernel_sub_op_infos.py |

The following functions are SuperKernel-specific, regular operator compilation flow does not use, therefore extracted to `Graph-autofusion`:

| Function Name | Function | New File |
| ----- | --- | --- |
| CommonUtility.is_support_super_kernel | Determine if current chip type supports SuperKernel | super_kernel_utils.py |
| ascendc_constants.SubOperatorType | Sub-operator dynamic/static types, where STATIC_OP_WITH_SEND\STATIC_OP_WITH_RECEIVE can be deleted | super_kernel_types.py |
| ascendc_constants.STR_TO_SUPER_TASK_TYPE | string type mapping ascendc_constants.SubOperatorType |super_kernel_types.py |
| ascendc_constants.SuperKernelPreLoadMode | Sub-kernel instruction prefetch mode selection |super_kernel_types.py |
| ascendc_constants.SuperKernelLinkMode | Sub-kernel link mode selection |super_kernel_types.py |
| ascendc_constants.SuperKernelEarlyStartMode | Sub-kernel early-start mode selection |super_kernel_types.py |
| ascendc_constants.SuperKernelStreamFusionMode | Whether to enable dual stream |super_kernel_types.py |
| ascendc_constants.SuperKernelDebugDcciAllMode | Whether to add entry cache dcci at sub-operator end |super_kernel_types.py |
| ascendc_constants.SuperKernelDebugSyncAllMode | Whether to add full-core synchronization at sub-operator end |super_kernel_types.py |
| ascendc_constants.SuperKernelFeedSyncAllMode | Whether to support full-core hard synchronization when operator not full-core |super_kernel_types.py |
| ascendc_constants.SuperKernelDataCacheMode | Sub-kernel data prefetch mode |super_kernel_types.py |
| ascendc_constants.SuperKernelProfilingMode | Support SuperKernel profiling data |super_kernel_types.py |
| ascendc_compile_gen_code.gen_super_dump_code | Generate SuperKernel dump initialization code |super_kernel_types.py |
| ascendc_compile_base.gen_symbol_rename_file | After splitting sub-kernel, rename |super_kernel_compile_base.py |
|ascendc_common_utility.gen_func_align_attribute| Generate alignment attr | super_kernel_compile_base.py
| ascendc_compile_base.split_dynamic_o_in_super_kernel | Split sub-kernel into multiple binaries in SuperKernel |super_kernel_op_infos.py |
| ascendc_common_utility.parse_super_kernel_options | SuperKernel parameter parsing and validation |super_kernel_options.py |
| OptionParser | Same as above |super_kernel_options.py |
| ParserFactory | Same as above |super_kernel_options.py |
| CodeTextAlignParser | Same as above |super_kernel_options.py |
| EnumParser | Same as above |super_kernel_options.py |
| BinaryParser | Same as above |super_kernel_options.py |
| NumberParser | Same as above |super_kernel_options.py |
| NonEmptyParser | Same as above |super_kernel_options.py |
| setup_super_kernel_option_parsers | Same as above |super_kernel_options.py |

The following functions are shared by SuperKernel and regular operators, or tightly coupled. Their dependencies (possibly) are unstable, first extracted, temporarily not migrated, consider analyzing in subsequent phase feature decoupling (no longer used, API standardization, or other measures)

| Function Name | Function | New File |
| ----- | --- | --- |
| ascendc_compile_base.super_kernel_gen_entry | Top-level SuperKernel entry generates quartered code | super_kernel_gen_entry_code.py |
| ascendc_compile_base.gen_spk_kernel_call | Same as above |super_kernel_gen_entry_code.py|
| ascendc_compile_base.split_spk_kernel_objs | Same as above |super_kernel_gen_entry_code.py |
| ascendc_compile_base.super_kernel_gen_entry | Same as above |super_kernel_gen_entry_code.py|
| ascendc_compile_base.gen_file_header | Same as above ps: generates dual page table global variables (implementation only supports v220, need consider macro isolation for different chip versions, can reference ascendc_compile_gen_code.gen_global_isolation_macro) and only supports split_mode=1, dual page table takes effect |super_kernel_gen_entry_code.py|
| ascendc_compile_base.gen_system_run_cfg | Same as above |super_kernel_gen_entry_code.py|
| ascendc_compile_base.localize_symbol_of_sk | Link all SuperKernel binaries |super_kernel_compile.py|
| ascendc_compile_base.gen_super_kernel_compile_info | Generate SuperKernel compile info |super_kernel_compile.py|
| ascendc_compile_base.gen_sub_super_kernel_early_start_compile_options | Process sub-kernel early-start feature compilation options |super_kernel_compile.py  |
| ascendc_compile_base.gen_sub_super_kernel_compile_options | Process sub-kernel compilation options |super_kernel_compile.py |
| ascendc_compile_base.split_sub_kernel_objs | Sub-kernel splits into four object files |super_kernel_compile.py |
| ascendc_compile_base.split_kernel_arch_str | Same as above |super_kernel_compile.py |
| ascendc_compile_base.split_kernel | Same as above |super_kernel_compile.py |
| ascendc_compile_base.gen_super_kernel_link_obj_sequence | Get all sub-kernel binaries |super_kernel_compile.py |
| ascendc_compile_base.localization_sub_op_func_sym | After splitting sub-kernel binary, modify sub-kernel global symbols to local |super_kernel_compile.py |
| ascendc_compile_base.localize_symbol_of_sk | Modify SuperKernel global symbols to local |super_kernel_compile.py |
| compile_op.compile_super_kernel | SuperKernel compilation entry |super_kernel_compile.py |

The following interfaces are used in ascendc_sub_super_kernel.py, ascendc_super_kernel.py, ascendc_super_operator_infos.py files, need to add interface encapsulation in super_kernel_utils.py, avoid AscendC open source rectification introducing issues affecting SuperKernel functionality:

| Function Name | Function | New Interface |
| ----- | --- | ------|
|tbe.common.platform.platform_info.get_soc_spec| Get current chip type |get_soc_version |
|tbe.tvm.error_mgr.raise_tbe_python_err| Exception throwing | super_kernel_raise_err |
|tbe.tvm.error_mgr.TBE_DEFAULT_PYTHON_ERROR_CODE| Exception code | "EB0500" |
|tbe.common.buildcfg.get_current_build_config| Get compilation context information | get_compile_config_info |
|tbe.common.buildcfg.buildcfg.op_debug_config| Get log level | Integrate into get_compile_config_info interface |
| log_utils and related interfaces | Log templates | Use directly |
| ascendc_common_utility and related interfaces | Utility templates | Directly use in current phase |
| global_storage.global_var_storage | Global variable setting | Analyze if can delete |

If any functions are not covered, divide according to above principles.

#### Graph-autofusion Directory Planning

Create independent directory for `SuperKernel` under `src` directory:

```bash
├── doc
├── README.md
└── src
    └── superkernel
        ├── sub_operator_infos.py  --> functions from original ascendc_sub_super_kernel.py migrate to this file
        ├── super_kernel.py --> functions from original ascendc_super_kernel.py migrate to this file
        ├── super_operator_infos.py --> functions from original ascendc_super_operator_infos.py migrate to this file
        ├── super_kernel_compile_base.py  --> functions from original ascendc_compile_base.py migrate to this file
        └── super_kernel_constants.py   --> functions from original ascendc_constants.py migrate to this file
```

Principles:

* File migration: whole file migration to `src/superkernel` directory
* Function/class migration: file prefix changes from `ascendc_` to `super_kernel_`, new file created in `src/superkernel` directory

#### Installation Directory

Install in `site-packages.tbe.superkernel`

### 2 GE Adaptation

todo GE/FE refactoring adaptation according to SuperKernel interface boundary in Graph-autofusion

### 3 Feature Decoupling

todo
