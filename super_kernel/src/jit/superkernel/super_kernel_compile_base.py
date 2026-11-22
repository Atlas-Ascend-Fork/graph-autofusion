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
"""
super kernel compile base
"""

from asc_op_compile_base.asc_op_compiler.super_kernel_utility import CommonUtility

from .super_kernel_constants import ERR_CODE, SuperKernelKernelType


def check_func_align(s):
    try:
        num = int(s)
    except Exception as err:
        CommonUtility().ascendc_raise_python_err(
            ERR_CODE, "Invalid func-align option, reason is: ", err
        )
    if num < 0:
        CommonUtility().ascendc_raise_python_err(
            ERR_CODE,
            f"Invalid func-align option, func-align should be a positive integer, but got {num}",
        )
    elif num > 0 and (num & (num - 1)) != 0:
        CommonUtility().ascendc_raise_python_err(
            ERR_CODE,
            f"Invalid func-align option, func-align should be a power of two, but got {num}",
        )
    return


def gen_func_align_attribute(align_size):
    check_func_align(align_size)
    if int(align_size) == 0:
        return ""
    else:
        return f"__attribute__((aligned({align_size})))"


def gen_system_run_cfg(kernel_type):
    file_header = ""
    if (
        kernel_type == SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
        or kernel_type == SuperKernelKernelType.KERNEL_TYPE_MIX_AIV_1_0
    ):
        file_header += "#if (defined(__DAV_VEC__) && __NPU_ARCH__ == 2201)\n"
    else:
        file_header += "#if (defined(__DAV_CUBE__) && __NPU_ARCH__ == 2201)\n"

    file_header += f"    __gm__ struct OpSystemRunCfg g_opSystemRunCfg = {{{0}}};\n"
    file_header += "#else\n"
    file_header += "    extern __gm__ struct OpSystemRunCfg g_opSystemRunCfg;\n"
    file_header += "#endif\n\n"
    return file_header


def gen_file_header(kernel_type, split_mode):
    file_header = """
#if 1
#include "kernel_operator.h"
"""
    if split_mode <= 1:
        file_header += gen_system_run_cfg(kernel_type)
    return file_header


def gen_super_dump_code(is_mix: bool, dump_size: int, offset: int):
    source = ""
    source += "    #if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON\n"
    source += "    constexpr uint32_t ASCENDC_DUMP_SIZE = 0;\n"
    if is_mix:
        source += (
            f"    AscendC::InitDump(true, workspace + {offset}, ASCENDC_DUMP_SIZE);\n"
        )
    else:
        source += (
            f"    AscendC::InitDump(false, workspace + {offset}, ASCENDC_DUMP_SIZE);\n"
        )
    source += "    #endif\n"
    return source
