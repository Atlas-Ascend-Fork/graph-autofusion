#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

"""Minimal smoke test ensuring the ST harness executes."""

import pytest
from utils import validate_codegen_output, validate_compile_options
from utils.sk_compile import compile_and_capture
from asc_op_compile_base.asc_op_compiler.global_storage import global_var_storage
from asc_op_compile_base.common.platform.platform_info import (
    set_current_compile_soc_info,
)
from asc_op_compile_base.common.ccec import current_build_config
from asc_op_compile_base.common.buildcfg.buildcfg_mapping import (
    kernel_meta_parent_dir,
    op_debug_config,
    tbe_debug_level,
)
from asc_op_compile_base.common.context.op_context import OpContext
from superkernel.super_kernel_feature_manager import get_features


@pytest.mark.parametrize(
    # 测试参数：
    #   1. 子内核1 fixture
    #   2. 子内核2 fixture
    #   3. compile_options: 编译前配置的编译选项
    #   4. golden_codegen_path: 期望编译生成的代码结果路径，路径内部按照${dir}/expect_sk_code.cc组织
    #   5. golden_options: 期望编译结果的编译选项
    "subkernel_inf, subkernel_pows, compile_options, \
    golden_codegen_path, golden_options",
    [
        # 默认配置
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:",
            "test_sk_1_stream_2_ops_default_aic_only",
            [
                "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
                "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
                "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 新配置1：early-start=0，默认1
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:early-start=0:",
            "test_sk_1_stream_2_ops_early_start_0_aic_only",
            [],
        ),
        # 新配置2：func-align=256, 默认是512
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:func-align=256:",
            "test_sk_1_stream_2_ops_func_align_256_aic_only",
            [
                "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
                "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
                "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 新配置3：preload-code=max, 支持[max, pre-func, none], 默认pre-func
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:preload-code=max:",
            "test_sk_1_stream_2_ops_preload_code_max_aic_only",
            [
                "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
                "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
                "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 新配置4：debug-dcci-all=1, 支持0跟1，默认0
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:debug-dcci-all=1:",
            "test_sk_1_stream_2_ops_debug_dcci_all_1_aic_only",
            [
                "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
                "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
                "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 新配置5：debug-sync-all=1, 支持0跟1，默认0
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:debug-sync-all=1:",
            "test_sk_1_stream_2_ops_debug_sync_all_1_aic_only",
            [
                "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
                "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
                "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 新配置6：feed-sync-all=1, 支持0跟1，默认0
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:feed-sync-all=1:",
            "test_sk_1_stream_2_ops_feed_sync_all_1_aic_only",
            [
                "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
                "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
                "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
                "-D__ASCENDC_SUPERKERNEL_AUTO_SYNC_ALL__",
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 新配置7：split-mode=1
        (
            "subkernel_is_inf_default",
            "subkernel_is_finite_split_mode1",
            "compile-options=-g:split-mode=1:",
            "test_sk_1_stream_2_ops_split_mode_1_aic_only",
            [
                "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
                "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
                "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 新配置8：json_split_none
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:split-mode=4:",
            "test_sk_1_stream_2_ops_json_split_none_aic_only",
            [
                "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
                "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
                "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 新配置9：profiling=1, 支持0跟1，默认0
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:profiling=1:",
            "test_sk_1_stream_2_ops_profiling_1_aic_only",
            [
                "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
                "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
                "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
                "-DASCENDC_DUMP=0",
            ],
        ),
    ],
    # 通过 indirect 实现 fixture 动态传入
    indirect=["subkernel_inf", "subkernel_pows"],
)
def test_sk_1_stream_2_ops(
    soc_version,
    tmp_dir,
    data_dir,
    json_dir,
    subkernel_inf,
    subkernel_pows,
    compile_options,
    golden_codegen_path,
    golden_options,
):
    kernel_meta_dir = tmp_dir / golden_codegen_path
    json_meta_dir = json_dir / golden_codegen_path
    global_var_storage.set_variable("ascendc_compile_debug_config", True)
    set_current_compile_soc_info(soc_version)
    current_build_config()[kernel_meta_parent_dir] = str(kernel_meta_dir)
    current_build_config()[op_debug_config] = ["dump_cce"]
    current_build_config()[tbe_debug_level] = 0

    kernel_info = {
        "super_kernel_options": compile_options,
        "op_list": [
            {
                "stream_id": 1,
                "bin_path": str(subkernel_inf.o()),
                "json_path": str(json_meta_dir / subkernel_inf.json()),
            },
            {
                "stream_id": 1,
                "bin_path": str(subkernel_pows.o()),
                "json_path": str(json_meta_dir / subkernel_pows.json()),
            },
        ],
    }
    kernel_name = "te_superkernel_1_stream_2_ops"

    with OpContext("super_kernel"):
        compile_info = compile_and_capture(kernel_info, kernel_name)

    scenario_dir = data_dir / golden_codegen_path

    validate_codegen_output(
        kernel_meta_dir,
        kernel_name,
        scenario_dir / "expect_sk_code.cc",
    )

    validate_compile_options(compile_info, golden_options)


@pytest.mark.parametrize(
    # 测试参数：
    #   1. 子内核1 fixture
    #   2. 子内核2 fixture
    #   3. compile_options: 编译前配置的编译选项
    #   4. send_event_list: 发送事件列表
    #   5. recv_event_list: 接收事件列表
    #   6. task_type: 可选dynamic或normal
    #   7. golden_codegen_path: 期望编译生成的代码结果路径，路径内部按照${dir}/expect_sk_code.cc组织
    #   8. golden_options: 期望编译结果的编译选项
    "subkernel_inf, subkernel_pows, compile_options, \
    send_event_list, recv_event_list, task_type, golden_codegen_path, golden_options",
    [
        # 双流场景AIC_ONLY默认配置: stream-fusion=1触发双流 + send_event_list空 + recv_event_list空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:",
            [],
            [],
            "normal",
            "test_sk_2_stream_2_ops_default_aic_only",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 双流场景AIV_1_0默认配置: stream-fusion=1触发双流 + send_event_list空 + recv_event_list空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:",
            [],
            [],
            "normal",
            "test_sk_2_stream_2_ops_default_aiv_1_0",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 双流场景MIX_AIC_1_0默认配置: stream-fusion=1触发双流 + send_event_list空 + recv_event_list空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:",
            [],
            [],
            "normal",
            "test_sk_2_stream_2_ops_default_aic_1_0",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 双流场景MIX_AIC_1_1默认配置: stream-fusion=1触发双流 + send_event_list空 + recv_event_list空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:",
            [],
            [],
            "normal",
            "test_sk_2_stream_2_ops_default_aic_1_1",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 双流场景MIX_AIC_1_2默认配置: stream-fusion=1触发双流 + send_event_list空 + recv_event_list空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:",
            [],
            [],
            "normal",
            "test_sk_2_stream_2_ops_default_aic_1_2",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 配置1：双流场景AIC_ONLY + feed-sync-all=1 + send_event_list空 + recv_event_list空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:feed-sync-all=1:",
            [],
            [],
            "normal",
            "test_sk_2_stream_2_ops_feed_sync_all_1_aic_only",
            [
                "-D__ASCENDC_SUPERKERNEL_AUTO_SYNC_ALL__",
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 配置2：双流场景AIC_ONLY + profiling=1 + send_event_list空 + recv_event_list空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:profiling=1:",
            [],
            [],
            "normal",
            "test_sk_2_stream_2_ops_profiling_1_aic_only",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 配置3：双流场景AIC_ONLY + debug-sync-all=1 + send_event_list空 + recv_event_list空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:debug-sync-all=1:",
            [],
            [],
            "normal",
            "test_sk_2_stream_2_ops_debug_sync_all_1_aic_only",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 配置4：双流场景AIV_ONLY + debug-sync-all=1 + send_event_list空 + recv_event_list空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:debug-sync-all=1:",
            [],
            [],
            "normal",
            "test_sk_2_stream_2_ops_debug_sync_all_1_aiv_only",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 配置5：双流场景AIC_ONLY + preload-code=max + send_event_list空 + recv_event_list空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:preload-code=max:",
            [],
            [],
            "normal",
            "test_sk_2_stream_2_ops_preload_code_max_aic_only",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 配置6：双流场景AIC_ONLY + send_event_list非空 + recv_event_list非空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:",
            [100],
            [101],
            "normal",
            "test_sk_2_stream_2_ops_default_send_recv_aic_only",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 配置7：双流场景AIV_ONLY + send_event_list非空 + recv_event_list非空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:",
            [100],
            [101],
            "normal",
            "test_sk_2_stream_2_ops_default_send_recv_aiv_only",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 配置8：双流场景AIC_ONLY + send_event_list空 + recv_event_list非空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:",
            [],
            [101],
            "normal",
            "test_sk_2_stream_2_ops_default_recv_aic_only",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 配置9：双流场景AIC_ONLY + send_event_list非空 + recv_event_list空
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:",
            [100],
            [],
            "normal",
            "test_sk_2_stream_2_ops_default_send_aic_only",
            [
                "-DASCENDC_DUMP=0",
            ],
        ),
        # 配置10：双流场景AIC_ONLY + send_event_list和recv_event_list无交集 + 第一个算子dynamic + timestamp_option=True
        (
            "subkernel_is_inf_default",
            "subkernel_is_finite_default",
            "compile-options=-g:stream-fusion=1:profiling=1:",
            [100],
            [101],
            "dynamic",
            "test_sk_2_stream_2_ops_dynamic_send_recv_default_aic_only",
            [
                "-D__SUPER_KERNEL_DYNAMIC_BLOCK_NUM__",
                "-DASCENDC_DUMP",
            ],
        ),
        # 配置11：双流场景AIV_ONLY + send_event_list和recv_event_list无交集 + 第一个算子dynamic + timestamp_option=True
        (
            "subkernel_is_inf_default",
            "subkernel_is_finite_default",
            "compile-options=-g:stream-fusion=1:profiling=1:",
            [100],
            [101],
            "dynamic",
            "test_sk_2_stream_2_ops_dynamic_send_recv_default_aiv_only",
            [
                "-D__SUPER_KERNEL_DYNAMIC_BLOCK_NUM__",
                "-DASCENDC_DUMP",
            ],
        ),
        # 配置12：双流场景AIC_ONLY + send_event_list和recv_event_list无交集 + with_synal_all=1 + 第一个算子dynamic
        (
            "subkernel_is_inf_default",
            "subkernel_is_finite_default",
            "compile-options=-g:stream-fusion=1:profiling=1:feed-sync-all=1:",
            [100],
            [101],
            "dynamic",
            "test_sk_2_stream_2_ops_dynamic_send_recv_with_synal_all_1_aic_only",
            [
                "-D__SUPER_KERNEL_DYNAMIC_BLOCK_NUM__",
                "-DASCENDC_DUMP",
            ],
        ),
        # 配置13：双流场景AIC_ONLY + send_event_list和recv_event_list有交集 + 第一个算子dynamic
        (
            "subkernel_is_inf_default",
            "subkernel_is_finite_default",
            "compile-options=-g:stream-fusion=1:profiling=1:",
            [100, 101],
            [100, 101],
            "dynamic",
            "test_sk_2_stream_2_ops_dynamic_send_recv_intersect_aic_only",
            [
                "-D__SUPER_KERNEL_DYNAMIC_BLOCK_NUM__",
                "-DASCENDC_DUMP",
            ],
        ),
    ],
    indirect=["subkernel_inf", "subkernel_pows"],
)
def test_sk_2_stream_2_ops(
    soc_version,
    tmp_dir,
    data_dir,
    json_dir,
    subkernel_inf,
    subkernel_pows,
    compile_options,
    send_event_list,
    recv_event_list,
    task_type,
    golden_codegen_path,
    golden_options,
):
    kernel_meta_dir = tmp_dir / golden_codegen_path
    json_meta_dir = json_dir / golden_codegen_path

    global_var_storage.set_variable("ascendc_compile_debug_config", True)
    set_current_compile_soc_info(soc_version)
    current_build_config()[kernel_meta_parent_dir] = str(kernel_meta_dir)
    current_build_config()[op_debug_config] = ["dump_cce"]
    current_build_config()[tbe_debug_level] = 0

    kernel_info = {
        "super_kernel_options": compile_options,
        "op_list": [
            {
                "stream_id": 0,
                "bin_path": str(subkernel_inf.o()),
                "json_path": str(json_meta_dir / subkernel_inf.json()),
                "send_event_list": send_event_list,
                "task_type": task_type,
            },
            {
                "stream_id": 1,  # 不同的stream_id触发双流模式
                "bin_path": str(subkernel_pows.o()),
                "json_path": str(json_meta_dir / subkernel_pows.json()),
                "recv_event_list": recv_event_list,
            },
        ],
    }
    kernel_name = "te_superkernel_2_stream_2_ops"
    if task_type == "dynamic":
        with OpContext("dynamic"):
            compile_info = compile_and_capture(kernel_info, kernel_name)
    else:
        with OpContext("super_kernel"):
            compile_info = compile_and_capture(kernel_info, kernel_name)

    scenario_dir = data_dir / golden_codegen_path

    validate_codegen_output(
        kernel_meta_dir,
        kernel_name,
        scenario_dir / "expect_sk_code.cc",
    )
    validate_compile_options(compile_info, golden_options)


@pytest.mark.parametrize(
    # 测试参数：
    #   1. 子内核1 fixture
    #   2. 子内核2 fixture
    #   3. compile_options: 编译前配置的编译选项
    #   4. golden_codegen_path: 期望编译生成的代码结果路径，路径内部按照${dir}/expect_sk_code.cc组织
    #   5. golden_options: 期望编译结果的编译选项
    "subkernel_inf, subkernel_pows, compile_options, \
    golden_codegen_path, golden_options",
    [
        # 默认配置
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:",
            "test_sk_1_stream_3_ops_default_aic_only",
            [
                "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
                "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
                "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
                "-DASCENDC_DUMP=0",
            ],
        ),
    ],
    # 通过 indirect 实现 fixture 动态传入
    indirect=["subkernel_inf", "subkernel_pows"],
)
def test_sk_1_stream_3_ops(
    soc_version,
    tmp_dir,
    data_dir,
    json_dir,
    subkernel_inf,
    subkernel_pows,
    compile_options,
    golden_codegen_path,
    golden_options,
):
    tmp = get_features()
    assert tmp == {}
    kernel_meta_dir = tmp_dir / golden_codegen_path
    json_meta_dir = json_dir / golden_codegen_path
    global_var_storage.set_variable("ascendc_compile_debug_config", True)
    set_current_compile_soc_info(soc_version)
    current_build_config()[kernel_meta_parent_dir] = str(kernel_meta_dir)
    current_build_config()[op_debug_config] = ["dump_cce"]
    current_build_config()[tbe_debug_level] = 0

    kernel_info = {
        "super_kernel_options": compile_options,
        "op_list": [
            {
                "stream_id": 1,
                "bin_path": str(subkernel_inf.o()),
                "json_path": str(json_meta_dir / subkernel_inf.json()),
            },
            {
                "stream_id": 1,
                "bin_path": str(subkernel_pows.o()),
                "json_path": str(json_meta_dir / subkernel_pows.json()),
            },
            {
                "stream_id": 1,
                "bin_path": str(subkernel_inf.o()),
                "json_path": str(json_meta_dir / subkernel_inf.json()),
            },
        ],
    }
    kernel_name = "te_superkernel_1_stream_3_ops"
    with OpContext("super_kernel"):
        compile_info = compile_and_capture(kernel_info, kernel_name)

    scenario_dir = data_dir / golden_codegen_path

    validate_codegen_output(
        kernel_meta_dir,
        kernel_name,
        scenario_dir / "expect_sk_code.cc",
    )

    validate_compile_options(compile_info, golden_options)


@pytest.mark.parametrize(
    # 测试参数：
    #   1. 子内核1 fixture
    #   2. 子内核2 fixture
    #   3. compile_options: 编译前配置的编译选项
    #   4. stream_ids: 各个算子stream_id
    #   5. send_event_list: 发送事件列表
    #   6. recv_event_list: 接收事件列表
    #   7. golden_codegen_path: 期望编译生成的代码结果路径，路径内部按照${dir}/expect_sk_code.cc组织
    #   8. golden_options: 期望编译结果的编译选项
    "subkernel_inf, subkernel_pows, compile_options, stream_ids,\
    send_event_lists, recv_event_lists, golden_codegen_path, golden_options",
    [
        # 测试用例1: 触发 cub->vec 交叉线移除
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:",
            [0, 1, 0, 1],
            [[100], [], [101], []],  # send_event_lists
            [[], [101], [], [100]],  # recv_event_lists
            "test_sk_2_stream_4_ops_remove_crossed_cub_to_vec",
            ["-DASCENDC_DUMP=0"],
        ),
        # 测试用例2: 触发 vec->cub 交叉线移除
        (
            "subkernel_is_inf_default",
            "subkernel_pows_default",
            "compile-options=-g:stream-fusion=1:",
            [0, 1, 0, 1],
            [[200], [], [201], []],  # send_event_lists
            [[], [201], [], [200]],  # recv_event_lists
            "test_sk_2_stream_4_ops_remove_crossed_vec_to_cub",
            ["-DASCENDC_DUMP=0"],
        ),
    ],
    indirect=["subkernel_inf", "subkernel_pows"],
)
def test_sk_2_stream_4_ops(
    soc_version,
    tmp_dir,
    data_dir,
    json_dir,
    subkernel_inf,
    subkernel_pows,
    compile_options,
    stream_ids,
    send_event_lists,
    recv_event_lists,
    golden_codegen_path,
    golden_options,
):
    kernel_meta_dir = tmp_dir / golden_codegen_path
    global_var_storage.set_variable("ascendc_compile_debug_config", True)
    set_current_compile_soc_info(soc_version)
    current_build_config()[kernel_meta_parent_dir] = str(kernel_meta_dir)
    current_build_config()[op_debug_config] = ["dump_cce"]
    current_build_config()[tbe_debug_level] = 0

    json_meta_dir = json_dir / golden_codegen_path
    kernel_info = {
        "super_kernel_options": compile_options,
        "op_list": [
            {
                "stream_id": stream_ids[0],
                "bin_path": str(subkernel_inf.o()),
                "json_path": str(json_meta_dir / subkernel_inf.json()),
                "send_event_list": send_event_lists[0],
                "recv_event_list": recv_event_lists[0],
            },
            {
                "stream_id": stream_ids[1],
                "bin_path": str(subkernel_pows.o()),
                "json_path": str(json_meta_dir / subkernel_pows.json()),
                "send_event_list": send_event_lists[1],
                "recv_event_list": recv_event_lists[1],
            },
            {
                "stream_id": stream_ids[2],
                "bin_path": str(subkernel_inf.o()),
                "json_path": str(json_meta_dir / subkernel_inf.json()),
                "send_event_list": send_event_lists[2],
                "recv_event_list": recv_event_lists[2],
            },
            {
                "stream_id": stream_ids[3],
                "bin_path": str(subkernel_pows.o()),
                "json_path": str(json_meta_dir / subkernel_pows.json()),
                "send_event_list": send_event_lists[3],
                "recv_event_list": recv_event_lists[3],
            },
        ],
    }
    kernel_name = "te_superkernel_2_stream_4_ops"
    with OpContext("super_kernel"):
        compile_info = compile_and_capture(kernel_info, kernel_name)
    scenario_dir = data_dir / golden_codegen_path
    validate_codegen_output(
        kernel_meta_dir, kernel_name, scenario_dir / "expect_sk_code.cc"
    )
    validate_compile_options(compile_info, golden_options)
