#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import subprocess
import shutil
import tempfile
import platform
import json
import time
from collections import namedtuple

from tbe.common.buildcfg import get_current_build_config
from tbe.tikcpp.compile_op import CommonUtility, AscendCLogLevel
from asc_op_compile_base.common.platform.platform_info import get_soc_spec
import tbe.common.utils.log as logger

# Python3 lib pyautofuse.so
from .pyautofuse import Schedule, CodeGen, ascir
from .ascbc_kernel_compile import ascbc_kernel_compile, camel_to_snake
from .compile_adapter import TILING_HEADER_FILES, get_pgo_env_flag, get_pgo_topn
from tbe.tikcpp.get_op_tiling import (
    TilingInfo,
    _change_param_name_to_name,
    gen_static_shape_v2,
)
from tbe.common.utils.op_tiling import do_op_tiling
from tbe.common.context import get_context


def _log_warning(message, *args):
    warning = getattr(logger, "warning", None)
    if warning is None:
        warning = getattr(logger, "warn")
    warning(message, *args)


PYF_PATH = os.path.dirname(os.path.realpath(__file__))
ASCEND_PATH = os.path.join(PYF_PATH, "..", "..", "..")
timestamp_list = []
HOST_TILING_COMPILE_JOBS = 32
_HOST_SOURCE_EXTENSIONS = (".cpp", ".h", ".hpp")
TILING_FIXED_CPP_FILES = {
    "BCubeKernelTilingWrapperCpp": "cube_kernel_tiling_wrapper.cpp",
    "CubeKernelTilingWrapperCpp": "cube_kernel_tiling_wrapper.cpp",
}
_SUPPORTED_CROSS_COMPILE_PREFIXES = {
    ("linux", "aarch64"): "aarch64-linux-gnu-",
    ("linux", "x86_64"): "x86_64-linux-gnu-",
}

StaticCompileContext = namedtuple(
    "StaticCompileContext",
    [
        "kernel_name",
        "temp_dir",
        "graph_name",
        "vector_core_num",
        "is_cross_compile_flag",
        "cross_compiler_prefix",
        "target_machine",
    ],
)

CodegenCompileContext = namedtuple(
    "CodegenCompileContext",
    [
        "extra_params",
        "vector_core_num",
        "ai_core_num_limited",
        "device_id",
        "static_compile_vector_core_num",
    ],
)

CrossCompileInfo = namedtuple(
    "CrossCompileInfo",
    ["is_cross_compile_flag", "cross_compiler_prefix", "target_machine"],
)

HostCompileContext = namedtuple(
    "HostCompileContext",
    [
        "graph_name",
        "kernel_name",
        "host_build_dir",
        "is_last_compile",
        "is_static_shape",
        "is_cube",
    ],
)

ShapeCompileContext = namedtuple(
    "ShapeCompileContext", ["kernel_name", "temp_dir", "graph_name"]
)

ConvArgsConfig = namedtuple(
    "ConvArgsConfig",
    [
        "input_num",
        "data_format",
        "has_bias",
        "has_scale0",
        "is_extend_conv2d",
    ],
    defaults=[False, False, False],
)


def get_target_machine(params):
    host_env_cpu = params.get("host_env_cpu", "")
    if host_env_cpu:
        return host_env_cpu
    return platform.machine()


def is_cross_compile(params):
    host_env_cpu = params.get("host_env_cpu", "")
    native_machine = platform.machine()
    result = host_env_cpu != "" and host_env_cpu != native_machine
    return result


def get_cross_compiler_prefix(params):
    host_env_os = params.get("host_env_os", "")
    host_env_cpu = params.get("host_env_cpu", "")
    prefix = _SUPPORTED_CROSS_COMPILE_PREFIXES.get((host_env_os, host_env_cpu))
    if prefix is None:
        raise ValueError(
            f"Unsupported cross-compile configuration: host_env_os='{host_env_os}', "
            f"host_env_cpu='{host_env_cpu}'. "
            f"Supported: {list(_SUPPORTED_CROSS_COMPILE_PREFIXES.keys())}"
        )
    return prefix


def get_native_host_so_path(host_build_dir, kernel_name):
    return os.path.join(host_build_dir + "_native", f"lib{kernel_name}.so")


def _copy_host_source_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        if os.path.isfile(src_path):
            if item.endswith(_HOST_SOURCE_EXTENSIONS):
                shutil.copy2(src_path, os.path.join(dst_dir, item))
        elif os.path.isdir(src_path):
            _copy_host_source_files(src_path, os.path.join(dst_dir, item))


def compile_native_host_so(host_ctx):
    (
        graph_name,
        kernel_name,
        host_build_dir,
        is_last_compile,
        is_static_shape,
        is_cube,
    ) = host_ctx
    native_build_dir = host_build_dir + "_native"
    if os.path.exists(native_build_dir):
        shutil.rmtree(native_build_dir)
    _copy_host_source_files(host_build_dir, native_build_dir)
    ascbc_host_compile(host_ctx._replace(host_build_dir=native_build_dir))


CV_COMMON_MIX_WHITE_LIST = [
    4097,
    4113,
    4161,
    4177,
    8193,
    8209,
    8257,
    8273,
    12289,
    12305,
    12353,
    12369,
    16449,
    28673,
    28689,
    28737,
    28753,
    1048577,
    1048593,
    1048641,
    1048657,
    1179649,
    1179665,
    1179713,
    1179729,
    2097153,
    2097169,
    2097217,
    2097233,
    2101249,
    2101265,
    2101313,
    2101329,
    2125825,
    2125841,
    2125889,
    2125905,
    2228225,
    2228241,
    2228289,
    2228305,
]

CV_COMMON_BMM_MIX_WHITE_LIST = [
    513,
    4097,
    4113,
    4161,
    4177,
    8193,
    2097409,
    2097425,
    2097473,
    2097489,
    2101249,
    2101265,
    2101313,
    2101329,
]


def generate_cmake_lists(host_ctx, cross_info=None):
    (
        graph_name,
        kernel_name,
        host_build_dir,
        is_last_compile,
        is_static_shape,
        is_cube,
    ) = host_ctx
    is_cross_compile_flag = cross_info.is_cross_compile_flag if cross_info else False
    target_machine = cross_info.target_machine if cross_info else None
    source = f"################ {kernel_name}.so ################\n"

    machine = platform.machine()
    source += "cmake_minimum_required(VERSION 3.16.0)\n"
    source += "project(asc_codegen)\n"
    source += "set(CMAKE_CXX_STANDARD 17)\n"
    if is_cross_compile_flag and target_machine:
        source += f"link_directories({ASCEND_PATH}/devlib/{target_machine}/\n"
        source += f"                  {ASCEND_PATH}/devlib/device/\n"
    else:
        source += f"link_directories({ASCEND_PATH}/{machine}-linux/lib64/\n"
    source += ")\n\n"

    source += "file(GLOB ALL_CPP_SRCS\n"
    source += "    *tiling_func*.cpp\n"
    source += "    *infershape*.cpp\n"
    if not is_static_shape and is_cube:
        source += "    cube_kernel_tiling_wrapper.cpp\n"
    if is_last_compile:
        source += "    *get_kernel*.cpp\n"
    source += ")\n\n"
    source += f"add_library({kernel_name} SHARED ${{ALL_CPP_SRCS}})\n\n"

    opt_level = "-O0" if is_static_shape else "-O2"
    source += (
        f"target_compile_options({kernel_name} PRIVATE\n"
        f"    {opt_level} -fno-common -Werror -Wextra -Wfloat-equal -fvisibility=default -DLOG_CPP"
        f" -ffile-prefix-map=${{CMAKE_CURRENT_SOURCE_DIR}}/=\n)\n\n"
    )
    source += f'message(STATUS "Using environment variable ASCEND_INSTALL_PATH: {ASCEND_PATH}")\n'
    # 交叉编译场景仅有libgraph.so的桩文件，libgraph桩已包含graph_base的符号，故link graph；
    # 非交叉编译场景保持link graph_base不变
    graph_lib = "graph" if is_cross_compile_flag else "graph_base"
    source += (
        f"target_link_libraries({kernel_name} PRIVATE c_sec ascendalog platform error_manager"
        f" tiling_api {graph_lib} register)\n"
    )
    source += f"target_include_directories({kernel_name} PRIVATE\n"
    source += f"    {ASCEND_PATH}/include\n"
    source += f"    {ASCEND_PATH}/include/base\n"
    source += f"    {ASCEND_PATH}/include/experiment\n"
    source += f"    {ASCEND_PATH}/{machine}-linux/include\n"
    source += f"    {ASCEND_PATH}/{machine}-linux/pkg_inc/base\n"
    source += f"    {ASCEND_PATH}/{machine}-linux/ascendc/include/highlevel_api/tiling/platform\n"
    source += f"    {ASCEND_PATH}/{machine}-linux/ascendc/include/highlevel_api\n"
    if is_cube:
        source += f"    {ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/mat_mul_v3\n"
        source += f"    {ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/mat_mul_v3\n"
        source += f"    {ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/conv2d_v2\n"
        source += f"    {ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/conv2d_v2\n"
        source += f"    {ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common\n"
    source += "    )\n\n"

    with open(host_build_dir + "/CMakeLists.txt", "w") as cmake_file:
        cmake_file.write(source)


def generate_file(dst_dir, file_name, text, append=False):
    os.makedirs(dst_dir, exist_ok=True)
    file_path = os.path.join(dst_dir, file_name)
    mode = "a" if append else "w"
    with open(file_path, mode) as file:
        file.write(text)


def ascbc_host_compile(host_ctx, cross_info=None):
    (
        graph_name,
        kernel_name,
        host_build_dir,
        is_last_compile,
        is_static_shape,
        is_cube,
    ) = host_ctx
    cross_compiler_prefix = cross_info.cross_compiler_prefix if cross_info else None
    generate_cmake_lists(host_ctx, cross_info=cross_info)
    ori_directory = os.getcwd()
    # 切换到临时工作目录
    os.chdir(host_build_dir)

    if cross_compiler_prefix:
        cmake_command = [
            "cmake",
            "-S",
            ".",
            "-B",
            "./",
            f"-DCMAKE_C_COMPILER={cross_compiler_prefix}gcc",
            f"-DCMAKE_CXX_COMPILER={cross_compiler_prefix}g++",
        ]
    else:
        cmake_command = [
            "cmake",
            "-S",
            ".",
            "-B",
            "./",
            "-DCMAKE_C_COMPILER=gcc",
            "-DCMAKE_CXX_COMPILER=g++",
        ]

    # 运行CMake
    cmake_ret = subprocess.run(cmake_command, capture_output=True, text=True)
    if cmake_ret.returncode != 0:
        os.chdir(ori_directory)
        CommonUtility.print_compile_log(
            "", f"cmake fail: {cmake_ret.stderr}", AscendCLogLevel.LOG_ERROR
        )
        error_msg = f"execute cmake failed with return code {cmake_ret.returncode}\n"
        error_msg += f"Standard Output:\n{cmake_ret.stdout}\n"
        error_msg += f"Standard Error:\n{cmake_ret.stderr}\n"
        raise Exception(error_msg)

    # 定义 CMake 编译命令
    make_command = ["make", "-C", "./", "-j", str(HOST_TILING_COMPILE_JOBS)]
    # 编译生成so
    make_ret = subprocess.run(make_command, capture_output=True, text=True)
    if make_ret.returncode != 0:
        os.chdir(ori_directory)
        CommonUtility.print_compile_log(
            "", f"make fail: {make_ret.stderr}", AscendCLogLevel.LOG_ERROR
        )
        error_msg = f"execute cmake failed with return code {make_ret.returncode}\n"
        error_msg += f"Standard Output:\n{make_ret.stdout}\n"
        error_msg += f"Standard Error:\n{make_ret.stderr}\n"
        raise Exception(error_msg)
    # 回退到原始工作目录
    os.chdir(ori_directory)


def gen_valid_name(t_name):
    result = []
    last_was_underscore = False

    for c in t_name:
        if c.isalnum():
            result.append(c)
            last_was_underscore = False
        else:
            if not last_was_underscore:
                result.append("_")
                last_was_underscore = True

    ret_name = "".join(result)

    # 删除开头的下划线
    if ret_name and ret_name[0] == "_":
        ret_name = ret_name[1:]

    # 如果以数字开头，添加前缀
    if ret_name and ret_name[0].isdigit():
        ret_name = "t_" + ret_name

    return ret_name


def is_valid_path(path):
    """
    判断路径是否合法。

    :param path: 路径字符串
    :return: 如果路径合法返回 True，否则返回 False
    """
    try:
        # 尝试获取路径的绝对路径
        os.path.abspath(path)
        return True
    except (TypeError, ValueError):
        # 如果路径无效，会抛出异常
        return False


def check_keys_in_dict(d, keys):
    """
    检查字典中是否缺少指定的键。如果所有指定的键都存在，返回True,

    :param d: 字典
    :param keys: 需要检查的键列表
    """
    # 检查每个键是否在字典中
    for key in keys:
        if key not in d:
            return False  # 如果任何一个键不存在，直接返回false

    # 如果所有键都存在
    return True


def modify_json_file(json_file, host_so, is_cross_compile_flag=False):
    """
    修改json文件中的binFileName/binFileSuffix
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    kernenl_file_name = data["binFileName"]
    lib_file_name = "lib" + kernenl_file_name
    lib_file_suffix = ".so"
    data["binFileName"] = lib_file_name
    data["binFileSuffix"] = lib_file_suffix

    ctypes_so = host_so
    if is_cross_compile_flag:
        host_dir = os.path.dirname(host_so)
        so_name = os.path.basename(host_so)
        ctypes_so = os.path.join(host_dir + "_native", so_name)

    import ctypes

    # 加载共享库
    lib = ctypes.CDLL(ctypes_so)

    # 定义函数参数和返回类型
    lib.GetTilingDataSize.argtypes = []
    lib.GetTilingDataSize.restype = ctypes.c_size_t

    # 调用函数
    data["opParaSize"] = int(lib.GetTilingDataSize())
    CommonUtility.print_compile_log(
        "",
        f"{kernenl_file_name} tiling size: {data['opParaSize']} bytes",
        AscendCLogLevel.LOG_INFO,
    )
    # 自动融合 workspace 默认只有一个
    data["workspace"]["num"] = int(1)
    data["workspace"]["size"] = list([-1])
    data["workspace"]["type"] = list([0])
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)


def timestamp_set(is_start_stamp, graph_name, stage, need_report=False):
    """
    设置时间戳列表
    """
    timestamp = time.time_ns()
    if is_start_stamp:
        timestamp_list.append(timestamp)
    else:
        start = timestamp_list[-1]
        cost_time = timestamp - start
        timestamp_list[-1] = cost_time
        ascir.utils.duration_record([stage, graph_name], int(start), int(cost_time))
    if need_report:
        ascir.utils.report_durations()


def ascendc_clean(temp_dir):
    src_directory = os.getcwd()
    keep_dirs = {"host", "device"}
    # 遍历目录中的所有条目
    for entry in os.listdir(temp_dir):
        entry_path = os.path.join(temp_dir, entry)
        # 检查是否是文件或目录
        if os.path.isfile(entry_path):
            # 如果是文件，直接删除
            os.remove(entry_path)
            CommonUtility.print_compile_log(
                "", f"delete file: {entry_path}", AscendCLogLevel.LOG_INFO
            )
        else:
            # 如果是目录，检查是否需要保留
            if entry not in keep_dirs:
                shutil.rmtree(entry_path)
                CommonUtility.print_compile_log(
                    "", f"delete dir: {entry_path}", AscendCLogLevel.LOG_INFO
                )
    os.chdir(src_directory)


def static_shape_kernel_proc(kernel_src, temp_dir, kernel_type, use_cv_common=None):
    import re

    base_device_files = os.path.basename(kernel_src)
    if use_cv_common and use_cv_common[0]:
        kernel_file = os.path.join(temp_dir, "device", "cv_common", base_device_files)
    else:
        kernel_file = os.path.join(temp_dir, "device", base_device_files)

    with open(kernel_file, "r", encoding="utf-8") as file:
        code = file.read()

    # 使用正则表达式进行替换
    code = re.sub(
        r"GET_TILING_DATA\(t, gm_tiling_data\);", "const AutofuseTilingData t;", code
    )
    code = re.sub(
        r"GM_ADDR workspace, AutofuseTilingData& t",
        "GM_ADDR workspace, const AutofuseTilingData& t",
        code,
    )
    if kernel_type is not None:
        code = re.sub(
            r"KERNEL_TASK_TYPE_DEFAULT\(.*\)",
            f"KERNEL_TASK_TYPE_DEFAULT({kernel_type})",
            code,
        )

    # 将修改后的内容写回文件
    with open(kernel_file, "w", encoding="utf-8") as file:
        file.write(code)


def get_static_compile_vector_core_num(vector_core_num=None):
    soc_vector_core_cnt = int(get_soc_spec("vector_core_cnt"))
    if vector_core_num is None or str(vector_core_num) in ("", "0"):
        return soc_vector_core_cnt
    return min(int(vector_core_num), soc_vector_core_cnt)


def get_codegen_compile_context():
    import tbe.common.context.op_context as op_context

    context = op_context.get_context()
    op_info = context.get_op_info()
    vector_core_num = context.get_addition("_op_vectorcore_num")
    return CodegenCompileContext(
        op_info[0].extra_params,
        vector_core_num,
        context.get_addition("_op_aicore_num"),
        context.get_addition("device_id"),
        get_static_compile_vector_core_num(vector_core_num),
    )


def log_core_limit_info(codegen_context):
    CommonUtility.print_compile_log(
        "",
        f"[AUTOFUSE_DEBUG] vector_core_num={codegen_context.static_compile_vector_core_num}, "
        f"vector_core_num_limited={codegen_context.vector_core_num}",
        AscendCLogLevel.LOG_INFO,
    )
    CommonUtility.print_compile_log(
        "",
        f"[AUTOFUSE_DEBUG] ai_core_num_limited={codegen_context.ai_core_num_limited}",
        AscendCLogLevel.LOG_INFO,
    )


def static_shape_compile(
    kernel_name,
    temp_dir,
    graph_name,
    tiling_key_list=None,
    kernel_type_list=None,
    use_cv_common=None,
    is_cube=False,
    vector_core_num=None,
    is_cross_compile_flag=False,
    cross_compiler_prefix=None,
    target_machine=None,
):
    if use_cv_common and use_cv_common[0]:
        host_build_dir = os.path.join(temp_dir, "host", "cv_common")
    else:
        host_build_dir = os.path.join(temp_dir, "host")
    ascbc_host_compile(
        HostCompileContext(
            graph_name, kernel_name, host_build_dir, False, True, is_cube
        ),
        cross_info=CrossCompileInfo(
            is_cross_compile_flag, cross_compiler_prefix, target_machine
        ),
    )
    if is_cross_compile_flag:
        compile_native_host_so(
            HostCompileContext(
                graph_name, kernel_name, host_build_dir, False, True, is_cube
            )
        )

    kernel_src = graph_name + "_op_kernel.cpp"
    host_so = os.path.join(host_build_dir, f"lib{kernel_name}.so")
    if is_cross_compile_flag:
        host_so = get_native_host_so_path(host_build_dir, kernel_name)
    pgo_config_path = os.path.abspath(
        os.path.join(temp_dir, "..", "..", "pgo", f"{graph_name}_config.txt")
    )

    import ctypes

    lib = ctypes.CDLL(host_so)

    CommonUtility.print_compile_log(
        "", "static shape compile", AscendCLogLevel.LOG_INFO
    )
    ascendc_clean(temp_dir)
    result = ""
    if hasattr(lib, "GenConstTilingData"):
        lib.GenConstTilingData.argtypes = []
        lib.GenConstTilingData.restype = ctypes.c_char_p
        result = lib.GenConstTilingData(
            ctypes.c_char_p(pgo_config_path.encode("utf-8")),
            ctypes.c_int(get_static_compile_vector_core_num(vector_core_num)),
            ctypes.c_int(int(get_soc_spec("ub_size"))),
        )

    if not result:
        raise RuntimeError("GenConstTilingData returned empty const tiling data")
    const_tiling_data = result.decode("utf-8")
    if hasattr(lib, "GetCVUBFusionStageSizeName"):
        stage_size_name = get_cv_ub_fusion_stage_size_name(
            kernel_name=kernel_name,
            temp_dir=temp_dir,
            graph_name=graph_name,
            host_so=host_so,
        )
        const_tiling_data = (
            const_tiling_data + "\n#define STAGE_SIZE_NAME " + stage_size_name + "\n"
        )

    if use_cv_common and use_cv_common[0]:
        tiling_data = os.path.join(
            temp_dir, "device", "cv_common", "autofuse_tiling_data.h"
        )
        tiling_data_bak = os.path.join(
            temp_dir, "device", "cv_common", "autofuse_tiling_data_bak.h"
        )
    else:
        tiling_data = os.path.join(temp_dir, "device", "autofuse_tiling_data.h")
        tiling_data_bak = os.path.join(temp_dir, "device", "autofuse_tiling_data_bak.h")

    # 备份原tilingdata文件
    shutil.copy(tiling_data, tiling_data_bak)
    with open(tiling_data, "w") as file:
        file.write(const_tiling_data)

    # 静态shape可以在编译期间获取tiling_key
    tiling_key = None
    if hasattr(lib, "GetTilingKeyForStatic"):
        lib.GetTilingKeyForStatic.argtypes = []
        lib.GetTilingKeyForStatic.restype = ctypes.c_int64  # int64_t对应c_int64
        tiling_key = lib.GetTilingKeyForStatic()
    if tiling_key_list is not None and tiling_key is not None:
        tiling_key_list[0] = tiling_key
    # 静态shape可以在编译期间获取kernel_type
    kernel_type = None
    if hasattr(lib, "GetTilingKeyKernelTypeForStatic"):
        lib.GetTilingKeyKernelTypeForStatic.argtypes = []
        lib.GetTilingKeyKernelTypeForStatic.restype = ctypes.c_char_p
        kernel_type_result = lib.GetTilingKeyKernelTypeForStatic()
        kernel_type = kernel_type_result.decode("utf-8")
    if kernel_type_list is not None and kernel_type is not None:
        kernel_type_list[0] = kernel_type
    # 修改kernel文件
    static_shape_kernel_proc(
        kernel_src, temp_dir, kernel_type, use_cv_common=use_cv_common
    )


def dynamic_shape_compile(
    compile_ctx, use_cv_common=None, is_cube=False, cross_info=None
):
    kernel_name, temp_dir, graph_name = compile_ctx
    is_cross_compile_flag = cross_info.is_cross_compile_flag if cross_info else False
    if use_cv_common and use_cv_common[0]:
        host_build_dir = os.path.join(temp_dir, "host", "cv_common")
    else:
        host_build_dir = os.path.join(temp_dir, "host")
    ascbc_host_compile(
        HostCompileContext(
            graph_name, kernel_name, host_build_dir, False, True, is_cube
        ),
        cross_info=cross_info,
    )
    if is_cross_compile_flag:
        compile_native_host_so(
            HostCompileContext(
                graph_name, kernel_name, host_build_dir, False, True, is_cube
            )
        )

    host_so = os.path.join(host_build_dir, f"lib{kernel_name}.so")
    if is_cross_compile_flag:
        host_so = get_native_host_so_path(host_build_dir, kernel_name)

    import ctypes

    lib = ctypes.CDLL(host_so)

    CommonUtility.print_compile_log(
        "", "dynamic shape compile", AscendCLogLevel.LOG_INFO
    )
    ascendc_clean(temp_dir)

    const_tiling_data = ""
    if hasattr(lib, "GetCVUBFusionStageSizeName"):
        stage_size_name = get_cv_ub_fusion_stage_size_name(
            kernel_name=kernel_name,
            temp_dir=temp_dir,
            graph_name=graph_name,
            host_so=host_so,
        )
        const_tiling_data = (
            const_tiling_data + "\n#define STAGE_SIZE_NAME " + stage_size_name + "\n"
        )

    if use_cv_common and use_cv_common[0]:
        tiling_data = os.path.join(
            temp_dir, "device", "cv_common", "autofuse_tiling_data.h"
        )
        tiling_data_bak = os.path.join(
            temp_dir, "device", "cv_common", "autofuse_tiling_data_bak.h"
        )
    else:
        tiling_data = os.path.join(temp_dir, "device", "autofuse_tiling_data.h")
        tiling_data_bak = os.path.join(temp_dir, "device", "autofuse_tiling_data_bak.h")

    # 备份原tilingdata文件
    shutil.copy(tiling_data, tiling_data_bak)
    with open(tiling_data, "w") as file:
        file.write(const_tiling_data)


def _compile_and_load_cv_host_so(
    kernel_name, graph_name, host_build_dir, temp_dir, cross_info=None
):
    is_cross_compile_flag = cross_info.is_cross_compile_flag if cross_info else False
    ascbc_host_compile(
        HostCompileContext(graph_name, kernel_name, host_build_dir, False, True, True),
        cross_info=cross_info,
    )
    if is_cross_compile_flag:
        compile_native_host_so(
            HostCompileContext(
                graph_name, kernel_name, host_build_dir, False, True, True
            )
        )
    host_so = os.path.join(host_build_dir, f"lib{kernel_name}.so")
    if is_cross_compile_flag:
        host_so = get_native_host_so_path(host_build_dir, kernel_name)
    pgo_config_path = os.path.abspath(
        os.path.join(temp_dir, "..", "..", "pgo", f"{graph_name}_config.txt")
    )

    import ctypes

    lib = ctypes.CDLL(host_so)
    CommonUtility.print_compile_log(
        "", "static shape cv compile", AscendCLogLevel.LOG_INFO
    )
    ascendc_clean(temp_dir)
    return lib, pgo_config_path


def static_shape_cv_compile(
    kernel_name, temp_dir, graph_name, vector_core_num=None, cross_info=None
):
    host_build_dir = os.path.join(temp_dir, "host")
    lib, pgo_config_path = _compile_and_load_cv_host_so(
        kernel_name, graph_name, host_build_dir, temp_dir, cross_info=cross_info
    )
    result = -1
    if hasattr(lib, "GenCVFusionTilingKey"):
        import ctypes

        result = lib.GenCVFusionTilingKey(
            ctypes.c_char_p(pgo_config_path.encode("utf-8")),
            ctypes.c_int(get_static_compile_vector_core_num(vector_core_num)),
            ctypes.c_int(int(get_soc_spec("ub_size"))),
        )
    return result


def get_cv_ub_fusion_stage_size_name(kernel_name, temp_dir, graph_name, host_so):
    if host_so is None:
        host_build_dir = os.path.join(temp_dir, "host")
        host_so = os.path.join(host_build_dir, f"lib{kernel_name}.so")

    import ctypes

    lib = ctypes.CDLL(host_so)
    CommonUtility.print_compile_log(
        "", "get cv ub fusion stage size", AscendCLogLevel.LOG_INFO
    )
    ascendc_clean(temp_dir)
    lib.GetCVUBFusionStageSizeName.argtypes = []
    lib.GetCVUBFusionStageSizeName.restype = ctypes.c_char_p
    result = lib.GetCVUBFusionStageSizeName()
    stage_size_name = result.decode("utf-8")
    return stage_size_name


def static_shape_cv_common_compile(
    kernel_name, temp_dir, graph_name, vector_core_num=None, cross_info=None
):
    host_build_dir = os.path.join(temp_dir, "host", "cv_common")
    lib, pgo_config_path = _compile_and_load_cv_host_so(
        kernel_name, graph_name, host_build_dir, temp_dir, cross_info=cross_info
    )
    vec_block_dim = -1
    wss = -1
    if hasattr(lib, "GenTilingDataValueBlockDimAndWss"):
        import ctypes

        # 创建变量接收返回值
        workspace_size = ctypes.c_uint64()
        block_dim = ctypes.c_uint32()
        ret = lib.GenTilingDataValueBlockDimAndWss(
            ctypes.c_char_p(pgo_config_path.encode("utf-8")),
            ctypes.c_uint32(get_static_compile_vector_core_num(vector_core_num)),
            ctypes.c_uint32(int(get_soc_spec("ub_size"))),
            ctypes.byref(workspace_size),
            ctypes.byref(block_dim),
        )

        if ret != -1:
            vec_block_dim = block_dim.value
            wss = workspace_size.value
    return vec_block_dim, wss


def is_static_compile(params, tiling_func_srcs):
    import re

    force_unknown = params.get("force_unknown")
    if force_unknown:
        return False

    static_shape = False
    pattern = r"AutofuseIsStaticShape\s*\(\s*\)\s*\{[^{}]*return\s+(true|false)\s*;"
    for tiling_func_src in tiling_func_srcs.values():
        match = re.search(pattern, tiling_func_src, re.IGNORECASE)
        if match:
            return_value = match.group(1).lower()
            static_shape = return_value == "true"
    return static_shape


def get_pgo_dir(temp_dir):
    # 获取pgo缓存路径
    pgo_dir = os.path.abspath(os.path.join(temp_dir, "..", "..", "pgo"))
    return pgo_dir


# 处理tiling数据字典
def _process_tiling_data_dict(tiling_dict, kernel_build_dir, host_build_dir):
    for key, tiling_data_src in tiling_dict.items():
        if key == "common":
            cv_common_dir = os.path.join(kernel_build_dir, "cv_common")
            generate_file(cv_common_dir, "autofuse_tiling_data.h", tiling_data_src)
            host_cv_common_dir = os.path.join(host_build_dir, "cv_common")
            generate_file(host_cv_common_dir, "autofuse_tiling_data.h", tiling_data_src)
        else:
            generate_file(kernel_build_dir, "autofuse_tiling_data.h", tiling_data_src)
            generate_file(host_build_dir, "autofuse_tiling_data.h", tiling_data_src)


# 处理kernel字典
def _process_kernel_dict(kernel_dict, graph_name, kernel_build_dir):
    ret_op_kernel_src = None
    for key, op_kernel_src in kernel_dict.items():
        if key == "common":
            cv_common_dir = os.path.join(kernel_build_dir, "cv_common")
            generate_file(cv_common_dir, f"{graph_name}_op_kernel.cpp", op_kernel_src)
        else:
            generate_file(
                kernel_build_dir, f"{graph_name}_op_kernel.cpp", op_kernel_src
            )
            ret_op_kernel_src = op_kernel_src
    return ret_op_kernel_src


# 处理tiling函数和infershape文件
def _process_tiling_funcs_and_infershape(
    tiling_func_srcs, graph_name, host_build_dir, infershape_src
):
    ret_tiling_func_srcs = None
    common_template_processed = False

    for template_key, template_dict in tiling_func_srcs.items():
        if template_key == "common":
            template_dir = os.path.join(host_build_dir, "cv_common")
            # 生成cv_common目录下的infershape文件
            generate_file(template_dir, graph_name + "_infershape.cpp", infershape_src)
            common_template_processed = True
            ret_tiling_func_srcs = template_dict
        else:
            template_dir = host_build_dir
            ret_tiling_func_srcs = template_dict

        has_atomic_headers = "TilingStateHeader" in template_dict
        for key, value in template_dict.items():
            if key == "TilingHead" and has_atomic_headers:
                continue
            if key in TILING_HEADER_FILES:
                generate_file(template_dir, TILING_HEADER_FILES[key], value)
            elif key in TILING_FIXED_CPP_FILES:
                generate_file(template_dir, TILING_FIXED_CPP_FILES[key], value)
            elif "TilingData" not in key:
                generate_file(
                    template_dir, graph_name + "_tiling_func_" + key + ".cpp", value
                )

    # 确保始终生成host_build_dir目录下的infershape文件
    if not common_template_processed or ret_tiling_func_srcs is not None:
        generate_file(host_build_dir, graph_name + "_infershape.cpp", infershape_src)

    return ret_tiling_func_srcs


def generate_device_and_host_code(graph_name, temp_dir, params, code_gen):
    # 生成device代码
    timestamp_set(True, graph_name, "GenDevice")
    schedule_results = params["schedule_results"]
    vector_core_num = (
        "0" if params["vector_core_num"] is None else str(params["vector_core_num"])
    )
    tiling_dict, kernel_dict = code_gen.device_code_generator(schedule_results)
    kernel_build_dir = os.path.join(temp_dir, "device")
    pgo_dir = get_pgo_dir(temp_dir)
    host_build_dir = os.path.join(temp_dir, "host")

    # 处理tiling数据字典
    _process_tiling_data_dict(tiling_dict, kernel_build_dir, host_build_dir)

    # 处理kernel字典，获取返回的op_kernel_src
    ret_op_kernel_src = _process_kernel_dict(kernel_dict, graph_name, kernel_build_dir)
    timestamp_set(False, graph_name, "GenDevice")

    # 生成host代码
    if not check_keys_in_dict(params, ["output_symbol_shape"]):
        CommonUtility.print_compile_log(
            "", "output_symbol_shape does not exist", AscendCLogLevel.LOG_ERROR
        )
        raise Exception("An error occurred autofuse compile for check extra_params")
    CommonUtility.print_compile_log(
        "",
        f"params output_symbol_shape : {params['output_symbol_shape']}",
        AscendCLogLevel.LOG_INFO,
    )
    timestamp_set(True, graph_name, "GenHost")
    shape_info = params.get("symbol_source_info")
    output_symbol = json.loads(params.get("output_symbol_shape"))
    tiling_func_srcs, infershape_src = code_gen.host_code_generator(
        schedule_results, shape_info, output_symbol, pgo_dir, vector_core_num
    )

    # 处理tiling函数和infershape文件
    ret_tiling_func_srcs = _process_tiling_funcs_and_infershape(
        tiling_func_srcs, graph_name, host_build_dir, infershape_src
    )

    timestamp_set(False, graph_name, "GenHost")

    return ret_op_kernel_src, ret_tiling_func_srcs


def pgo_kernel_compile(*args, temp_dir, params, op_kernel_src, code_gen):
    """PGO依赖的动态so编译"""
    schedule_results = params["schedule_results"]
    kernel_name = args[-1]
    input_num = schedule_results.get_input_num()
    output_num = schedule_results.get_output_num()
    kernel_build_dir = os.path.join(temp_dir, "device")
    host_build_dir = os.path.join(temp_dir, "host")
    use_list_tensor_desc = op_kernel_src.find("kernel_operator_list_tensor_intf.h") > 0
    enable_parallel_compile = op_kernel_src.rfind("void fake_tiling_ids()") > 0
    pgo_dir = get_pgo_dir(temp_dir)
    graph_name = gen_valid_name(schedule_results.get_name())
    graph_name = camel_to_snake(graph_name)

    kernel_file_path = None
    json_file_path = None
    target_lib_path = None

    try:
        kernel_file, json_file = ascbc_kernel_compile(
            args,
            graph_name=graph_name,
            kernel_name=kernel_name,
            input_num=input_num,
            output_num=output_num,
            temp_build_dir=kernel_build_dir,
            impl_mode=params.get("impl_mode"),
            use_list_tensor_desc=use_list_tensor_desc,
            enable_parallel_compile=enable_parallel_compile,
        )
        kernel_file_path = os.path.join(pgo_dir, os.path.basename(kernel_file))
        json_file_path = os.path.join(pgo_dir, os.path.basename(json_file))
        shutil.copy(kernel_file, kernel_file_path)
        shutil.copy(json_file, json_file_path)
    except Exception as e:
        msg = f"pgo kernel compile for {kernel_name} failed: {str(e)}. skip pgo tune."
        CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_ERROR)
        return False, []
    get_kernel_src = code_gen.get_kernel_and_json_generator(kernel_file, json_file)
    generate_file(host_build_dir, graph_name + "_get_kernel.cpp", get_kernel_src)
    ascbc_host_compile(
        HostCompileContext(graph_name, kernel_name, host_build_dir, True, False, False)
    )
    source_lib_path = os.path.join(host_build_dir, f"lib{kernel_name}.so")
    target_lib_path = os.path.join(pgo_dir, f"lib{graph_name}.so")
    shutil.copy(source_lib_path, target_lib_path)
    return True, [kernel_file_path, json_file_path, target_lib_path]


def pgo_cleanup_kernel_and_json(pgo_temp_files, config_path=None):
    """清理PGO流程中生成的其他中间文件"""
    if config_path is not None:
        if os.path.exists(config_path):
            os.remove(config_path)

    from autofuse.compile_adapter import get_debug_flag

    compile_debug = get_debug_flag()
    if compile_debug:
        return
    if not pgo_temp_files:
        return
    for item in pgo_temp_files:
        if not item:
            continue
        try:
            if os.path.exists(item):
                os.remove(item)
                logger.info("[PGO] cleanup file: %s", item)
        except Exception as e:
            _log_warning("[PGO] cleanup file failed: %s, err: %s", item, str(e))


def check_dir_permissions(path):
    import stat

    # 检查是否为软链接
    if path.is_symlink():
        raise Exception(
            f"Warning: {path} is a symbolic link. Please configure --replace_kernel to a real directory."
        )

    # 检查目录是否存在
    if not path.exists():
        raise FileNotFoundError(
            f"Directory {path} does not exist. "
            f"Please configure --replace_kernel to an existing directory via AUTOFUSE_DFX_FLAGS."
        )

    # 检查读写执行权限
    mode = path.stat().st_mode
    if not (mode & stat.S_IRUSR):
        raise PermissionError(
            f"No read permission for {path}. "
            f"Please check the directory permissions configured by --replace_kernel in AUTOFUSE_DFX_FLAGS."
        )
    if not (mode & stat.S_IWUSR):
        raise PermissionError(
            f"No write permission for {path}. "
            f"Please check the directory permissions configured by --replace_kernel in AUTOFUSE_DFX_FLAGS."
        )
    if not (mode & stat.S_IXUSR):
        raise PermissionError(
            f"No execute permission for {path}. "
            f"Please check the directory permissions configured by --replace_kernel in AUTOFUSE_DFX_FLAGS."
        )


def get_replace_kernel_root():
    import re
    from pathlib import Path

    autofuse_dfx_flags_env = os.getenv("AUTOFUSE_DFX_FLAGS", "")
    if not autofuse_dfx_flags_env or "replace_kernel=" not in autofuse_dfx_flags_env:
        return None
    pattern = r'replace_kernel=([^";]+)'
    match = re.search(pattern, autofuse_dfx_flags_env)
    if not match:
        _log_warning(
            "match env replace_kernel failed. AUTOFUSE_DFX_FLAGS is %s: ",
            autofuse_dfx_flags_env,
        )
        return None
    replace_kernel_path = match.group(1)
    kernel_path = Path(replace_kernel_path).expanduser().absolute()
    check_dir_permissions(kernel_path)
    return kernel_path


def replace_device_kernel(replace_root, kernel_build_dir, graph_name):
    src_kernel_path = os.path.join(str(replace_root), f"{graph_name}_op_kernel.cpp")
    if not os.path.exists(src_kernel_path):
        return
    dst_kernel_path = os.path.join(kernel_build_dir, f"{graph_name}_op_kernel.cpp")
    shutil.copy(src_kernel_path, dst_kernel_path)
    logger.info("replace kernel: %s to %s", src_kernel_path, dst_kernel_path)


def find_host_replace_source_dir(replace_root, graph_name):
    from pathlib import Path

    matched_dirs = sorted(
        {
            candidate.parent
            for candidate in Path(replace_root).rglob(f"{graph_name}*_tiling_func*.cpp")
            if candidate.is_file()
        }
    )
    if len(matched_dirs) > 1:
        raise RuntimeError(
            f"Multiple host replace source dirs matched for {graph_name}: "
            f"{[str(path) for path in matched_dirs]}"
        )
    return matched_dirs[0] if matched_dirs else None


def cleanup_host_tiling_files(host_build_dir, graph_name):
    from pathlib import Path

    removed_files = []
    for old_file in Path(host_build_dir).glob(f"{graph_name}*_tiling_func*.cpp"):
        old_file.unlink()
        removed_files.append(old_file.name)
    return removed_files


def replace_host_files(replace_root, host_build_dir, graph_name):
    from pathlib import Path

    source_dir = find_host_replace_source_dir(replace_root, graph_name)
    if source_dir is None:
        return
    logger.info("replace host source dir: %s", source_dir)
    logger.info("replace host target dir: %s", host_build_dir)
    host_build_path = Path(host_build_dir)
    if source_dir.resolve() == host_build_path.resolve():
        logger.info("skip host replacement because source dir equals target dir")
        return
    removed_files = cleanup_host_tiling_files(host_build_dir, graph_name)
    logger.info("cleanup stale host tiling files: %s", removed_files)
    copied_files = []
    for src in source_dir.glob(f"{graph_name}*_tiling_func*.cpp"):
        dst = host_build_path / src.name
        if src.resolve() == dst.resolve():
            continue
        shutil.copy(str(src), str(dst))
        copied_files.append(src.name)
    for header_name in [
        "autofuse_tiling_data.h",
        "autofuse_tiling_func_common.h",
        "autofuse_tiling_func_state.h",
        "autofuse_tiling_func_log.h",
        "autofuse_tiling_func_pgo.h",
        "autofuse_tiling_func_base.h",
        "autofuse_tiling_func_solver.h",
        "autofuse_tiling_func_api.h",
        "autofuse_tiling_func_entry.h",
        "autofuse_tiling_func_tail.h",
        "autofuse_cube_tiling_data.h",
        "cube_kernel_tiling_wrapper.h",
    ]:
        header_path = source_dir / header_name
        if header_path.exists():
            dst = host_build_path / header_name
            if header_path.resolve() == dst.resolve():
                continue
            shutil.copy(str(header_path), str(dst))
            copied_files.append(header_name)
    logger.info("copied host files: %s", copied_files)


def replace_kernel(kernel_build_dir, graph_name):
    replace_root = get_replace_kernel_root()
    if replace_root is None:
        return
    replace_device_kernel(replace_root, kernel_build_dir, graph_name)


def _clean_cmake_cache(host_build_dir):
    for cache_file in ["CMakeCache.txt", "cmake_install.cmake"]:
        cache_path = os.path.join(host_build_dir, cache_file)
        if os.path.exists(cache_path):
            os.remove(cache_path)
    cmake_files_dir = os.path.join(host_build_dir, "CMakeFiles")
    if os.path.exists(cmake_files_dir):
        shutil.rmtree(cmake_files_dir)


def _run_pgo_and_clean_cache(
    *args, temp_dir, params, op_kernel_src, cross_compile_flag, host_build_dir
):
    pgo_env = get_pgo_env_flag()
    use_list_tensor_desc = op_kernel_src.find("kernel_operator_list_tensor_intf.h") > 0
    enable_parallel_compile = op_kernel_src.rfind("void fake_tiling_ids()") > 0
    if not pgo_env or enable_parallel_compile or use_list_tensor_desc:
        return
    asc_pgo_exec(
        *args,
        temp_dir=temp_dir,
        params=params,
        op_kernel_src=op_kernel_src,
        code_gen=CodeGen(),
    )
    if cross_compile_flag:
        _clean_cmake_cache(host_build_dir)
        _clean_cmake_cache(os.path.join(host_build_dir, "cv_common"))


def get_host_build_dir(temp_dir, use_cv_common):
    if use_cv_common and use_cv_common[0]:
        return os.path.join(temp_dir, "host", "cv_common")
    return os.path.join(temp_dir, "host")


def get_device_build_dir(temp_dir, use_cv_common):
    if use_cv_common and use_cv_common[0]:
        return os.path.join(temp_dir, "device", "cv_common")
    return os.path.join(temp_dir, "device")


def replace_host_files_if_needed(replace_root, temp_dir, use_cv_common, graph_name):
    if replace_root is None:
        return
    host_build_dir = get_host_build_dir(temp_dir, use_cv_common)
    replace_host_files(replace_root, host_build_dir, graph_name)


def generate_pgo_code(params, code_gen, pgo_dir, host_build_dir):
    """生成PGO代码"""
    schedule_results = params["schedule_results"]
    graph_name = gen_valid_name(schedule_results.get_name())
    graph_name = camel_to_snake(graph_name)

    pgo_src = code_gen.pgo_code_generator(schedule_results, pgo_dir)
    generate_file(host_build_dir, graph_name + "_pgo.cpp", pgo_src)


def pgo_get_mspti_config():
    """获取PGO依赖的mspti相关配置"""
    mspti_dir = os.path.join(ASCEND_PATH, "tools", "mspti")
    mspti_lib64_dir = os.path.join(mspti_dir, "lib64")
    mspti_so = os.path.join(mspti_lib64_dir, "libmspti.so")
    libprof_common_so = os.path.join(mspti_lib64_dir, "libprof_common.so")

    if not os.path.exists(mspti_so):
        return None

    has_prof_common = os.path.exists(libprof_common_so)
    preload_so_paths = [libprof_common_so, mspti_so] if has_prof_common else [mspti_so]

    link_flags = [f"-L{mspti_lib64_dir}", "-lmspti"]
    if has_prof_common:
        link_flags.append("-lprof_common")

    return mspti_dir, preload_so_paths, link_flags


def build_pgo_compile_command(source_file, output_file, mspti_dir, mspti_link_flags):
    """构建PGO编译命令"""
    machine = platform.machine()
    base_cmd = ["g++", "-std=c++17", "-O2", "-fPIC"]
    includes = [
        f"-I{ASCEND_PATH}/include/",
        f"-I{ASCEND_PATH}/include/base",
        f"-I{ASCEND_PATH}/include/experiment/runtime",
        f"-I{ASCEND_PATH}/include/experiment/msprof",
        f"-I{ASCEND_PATH}/{machine}-linux/include/toolchain",
        f"-I{ASCEND_PATH}/{machine}-linux/pkg_inc/base",
        f"-I{mspti_dir}/include",
    ]
    libs = [
        f"-L{ASCEND_PATH}/lib64",
        "-lascendcl",
        "-lruntime",
        "-lunified_dlog",
        "-lc_sec",
        *mspti_link_flags,
        "-ldl",
    ]
    cmd = base_cmd + [source_file, "-o", output_file] + includes + libs
    return cmd


def pgo_compile(graph_name, host_build_dir, mspti_dir, mspti_link_flags):
    """编译PGO代码"""
    pgo_source_file = os.path.join(host_build_dir, f"{graph_name}_pgo.cpp")
    output_file = os.path.join(host_build_dir, "pgo")

    cmd = build_pgo_compile_command(
        pgo_source_file, output_file, mspti_dir, mspti_link_flags
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = "cmd = " + " ".join(cmd) + "\n"
        msg += f"stdout: {result.stdout}\n"
        msg += f"stderr: {result.stderr}\n"
        CommonUtility.print_compile_log(
            "", f"pgo compile fail: {msg}", AscendCLogLevel.LOG_ERROR
        )
        return None
    return output_file


def pgo_program_exec(temp_dir, exec_type=0, *, pgo_exec_params, mspti_so_list):
    """执行PGO, 0表示首次动态调优，1表示二次调优"""
    host_build_dir = os.path.join(temp_dir, "host")
    pgo_file = os.path.join(host_build_dir, "pgo")
    device_id = pgo_exec_params.get("device_id")
    aiv_num = pgo_exec_params.get("aiv_num")
    ub_size = pgo_exec_params.get("ub_size")
    kernel_name = pgo_exec_params.get("kernel_name")
    try:
        graph_so = f"{ASCEND_PATH}/lib64/libgraph_af.so"
        ld_preload_items = [graph_so] + mspti_so_list
        existing_ld_preload = os.environ.get("LD_PRELOAD")
        if existing_ld_preload:
            ld_preload_items.append(existing_ld_preload)
        ld_preload = ":".join([p for p in ld_preload_items if p])
        result = subprocess.run(
            [
                pgo_file,
                str(exec_type),
                str(device_id),
                str(aiv_num),
                str(ub_size),
                str(kernel_name),
            ],
            text=True,
            env={
                **dict(os.environ),
                "LD_PRELOAD": ld_preload,
            },
            timeout=1800,  # 暂定，设置超时时间为1800秒，防止大解集case提前退出
        )
        if result.returncode != 0:
            msg = f"pgo exec fail, return code: {result.returncode}\n"
            # 执行失败时，回退原始解，不打断模型正常执行流程
            CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_ERROR)
            return False
        return True
    except Exception as e:
        CommonUtility.print_compile_log(
            "", f"pgo exec fail: {str(e)}", AscendCLogLevel.LOG_ERROR
        )
        return False


def get_pgo_exec_params(*args, params):
    """提取PGO执行所需的基础参数。

    Returns:
        dict: {"device_id": str, "aiv_num": str, "ub_size": str, "kernel_name": str}
    """
    device_id = params.get("device_id")
    device_id = (
        "0" if device_id is None or not str(device_id).isdigit() else str(device_id)
    )

    soc_vector_core_cnt = int(get_soc_spec("vector_core_cnt"))
    param_vector_core_num = (
        soc_vector_core_cnt
        if params["vector_core_num"] is None
        else int(params["vector_core_num"])
    )
    aiv_num = str(min(param_vector_core_num, soc_vector_core_cnt))
    ub_size = str(get_soc_spec("ub_size"))
    kernel_name = args[-1]

    return {
        "device_id": device_id,
        "aiv_num": aiv_num,
        "ub_size": ub_size,
        "kernel_name": kernel_name,
    }


def pgo_second_optimization(*args, temp_dir, params, op_kernel_src, mspti_so_list):
    """静态tiling的二次调优"""
    from autofuse.compile_adapter import (
        pgo_get_top_result,
        pgo_write_config,
        pgo_generate_config,
    )

    pgo_exec_params = get_pgo_exec_params(*args, params=params)
    kernel_name = pgo_exec_params["kernel_name"]
    schedule_results = params["schedule_results"]
    graph_name = camel_to_snake(gen_valid_name(schedule_results.get_name()))
    input_num = schedule_results.get_input_num()
    output_num = schedule_results.get_output_num()
    use_list_tensor_desc = op_kernel_src.find("kernel_operator_list_tensor_intf.h") > 0
    enable_parallel_compile = op_kernel_src.rfind("void fake_tiling_ids()") > 0
    kernel_build_dir = os.path.join(temp_dir, "device")
    pgo_dir = get_pgo_dir(temp_dir)
    config_path = os.path.join(pgo_dir, f"{graph_name}_config.txt")
    search_path = os.path.join(pgo_dir, f"{graph_name}_search.txt")
    top_n = get_pgo_topn()

    top_lines, origin_line, top_n = pgo_get_top_result(search_path, top_n)
    if top_lines is None or origin_line is None or top_n is None:
        return False

    def comile_and_exec(config_line):
        tiling_key_list, kernel_type_list = [-1], ["KERNEL_TYPE_AIV_ONLY"]
        pgo_write_config(config_path, config_line)
        static_shape_compile(
            kernel_name=kernel_name,
            temp_dir=temp_dir,
            graph_name=graph_name,
            tiling_key_list=tiling_key_list,
            kernel_type_list=kernel_type_list,
            vector_core_num=params.get("vector_core_num"),
        )
        try:
            kernel_file, json_file = ascbc_kernel_compile(
                args,
                graph_name=graph_name,
                kernel_name=kernel_name,
                input_num=input_num,
                output_num=output_num,
                temp_build_dir=kernel_build_dir,
                impl_mode=params.get("impl_mode"),
                use_list_tensor_desc=use_list_tensor_desc,
                enable_parallel_compile=enable_parallel_compile,
                tiling_key=tiling_key_list[0],
                kernel_type=kernel_type_list[0],
                is_cube=False,
            )
            shutil.copy(
                kernel_file, os.path.join(pgo_dir, os.path.basename(kernel_file))
            )
            shutil.copy(json_file, os.path.join(pgo_dir, os.path.basename(json_file)))
        except Exception as e:
            logger.error(
                "[pgo] kernel compile for %s failed: %s. revert pgo tune.",
                kernel_name,
                str(e),
            )
            return False

        result = pgo_program_exec(
            temp_dir=temp_dir,
            exec_type=1,
            mspti_so_list=mspti_so_list,
            pgo_exec_params=pgo_exec_params,
        )
        return result

    for line in top_lines:
        result = comile_and_exec(line)
        if result is False:
            return False

    result = comile_and_exec(origin_line)
    if result is False:
        return False

    pgo_generate_config(search_path, config_path, top_n)
    return True


def asc_pgo_exec(*args, temp_dir, params, op_kernel_src, code_gen):
    """执行PGO"""
    schedule_results = params["schedule_results"]
    graph_name = camel_to_snake(gen_valid_name(schedule_results.get_name()))
    logger.info(f"[PGO] Start PGO tuning for graph: {graph_name}")
    mspti_cfg = pgo_get_mspti_config()
    if mspti_cfg is None:
        _log_warning("[PGO] libmspti.so not installed, skip pgo tuning")
        return
    mspti_dir, mspti_so_list, mspti_link_flags = mspti_cfg

    pgo_dir = get_pgo_dir(temp_dir)
    os.makedirs(pgo_dir, exist_ok=True)
    host_build_dir = os.path.join(temp_dir, "host")

    pgo_temp_files = []

    config_path = os.path.join(pgo_dir, f"{graph_name}_config.txt")
    if os.path.exists(config_path):
        logger.info(f"[PGO] {config_path} exists, skip pgo tuning")
        return

    timestamp_set(True, graph_name, "CompileKernelForPGO")
    result, pgo_temp_files = pgo_kernel_compile(
        *args,
        temp_dir=temp_dir,
        params=params,
        op_kernel_src=op_kernel_src,
        code_gen=code_gen,
    )
    timestamp_set(False, graph_name, "CompileKernelForPGO")
    if result is False:
        pgo_cleanup_kernel_and_json(pgo_temp_files)
        return

    timestamp_set(True, graph_name, "GenerateForPGO")
    generate_pgo_code(
        params=params, code_gen=code_gen, pgo_dir=pgo_dir, host_build_dir=host_build_dir
    )
    timestamp_set(False, graph_name, "GenerateForPGO")

    timestamp_set(True, graph_name, "CompilePGO")
    output = pgo_compile(
        graph_name=graph_name,
        host_build_dir=host_build_dir,
        mspti_dir=mspti_dir,
        mspti_link_flags=mspti_link_flags,
    )
    timestamp_set(False, graph_name, "CompilePGO")
    if output is None:
        pgo_cleanup_kernel_and_json(pgo_temp_files)
        return

    timestamp_set(True, graph_name, "RunForPGO")
    pgo_exec_params = get_pgo_exec_params(*args, params=params)
    result = pgo_program_exec(
        temp_dir=temp_dir,
        exec_type=0,
        mspti_so_list=mspti_so_list,
        pgo_exec_params=pgo_exec_params,
    )
    timestamp_set(False, graph_name, "RunForPGO")
    if result is False:
        pgo_cleanup_kernel_and_json(pgo_temp_files, config_path=config_path)
        return

    timestamp_set(True, graph_name, "SecondOptimizeForPGO")
    result = pgo_second_optimization(
        *args,
        temp_dir=temp_dir,
        params=params,
        op_kernel_src=op_kernel_src,
        mspti_so_list=mspti_so_list,
    )
    timestamp_set(False, graph_name, "SecondOptimizeForPGO")
    if result is False:
        pgo_cleanup_kernel_and_json(pgo_temp_files, config_path=config_path)
        return

    pgo_cleanup_kernel_and_json(pgo_temp_files)


def _build_mm_args(args_list, input_num, mm_attr1, mm_attr2):
    _inputs_ = []
    _origin_inputs_ = []
    _origin_outputs_ = []
    _m = 0
    _n = 0
    logger.info("CV fusion op, matmul input(%s)", input_num)
    # 遍历args中的每个元素
    for i, arg in enumerate(args_list[:input_num]):
        if i >= input_num:
            continue
        msg = "Processing input " + str(i) + ":" + str(arg)
        logger.info("CV fusion op, matmul input info: %s", msg)
        CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
        _origin_inputs_.append(arg)
        if arg is not None:
            if isinstance(arg, (list, tuple)) and len(arg) != 0:
                _inputs_.append(arg[0])
            else:
                _inputs_.append(arg)
        else:
            _inputs_.append(arg)
        _inputs_[-1]["param_name"] = "input" + str(i)
        shape = _inputs_[-1]["shape"]
        _inputs_[-1]["shape"] = shape
        _inputs_[-1]["ori_shape"] = shape
        if i == 0:
            write_shape = list(shape)
            if mm_attr1["value"]:  # a矩阵是否有transpose
                _m = shape[-1]
            else:
                _m = shape[-2]
        if i == 1:
            if mm_attr2["value"]:  # b矩阵是否有transpose
                _n = shape[-2]
            else:
                _n = shape[-1]

    msg = "Processing output " + ":" + str(args_list[-2])
    logger.info("CV fusion op, matmul output info: %s", msg)
    CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
    _origin_outputs_.append(args_list[-2])
    _origin_outputs_[-1]["param_name"] = "output0"
    write_shape[-1] = _n
    write_shape[-2] = _m
    _origin_outputs_[-1]["shape"] = tuple(write_shape)
    _origin_outputs_[-1]["dtype"] = _inputs_[-1]["dtype"]
    _origin_outputs_[-1]["ori_shape"] = tuple(write_shape)
    logger.info(
        "CV fusion op, m=%s, n=%s new matmul output info: %s",
        _m,
        _n,
        str(_origin_outputs_[-1]),
    )
    return _origin_inputs_, _origin_outputs_, _inputs_


def fill_conv_origin_input(_origin_inputs_, input_arg, logical_index):
    op_msg = "Processing input " + str(logical_index) + ":" + str(input_arg)
    logger.info("CV fusion op, conv input info: %s", op_msg)
    CommonUtility.print_compile_log("", op_msg, AscendCLogLevel.LOG_INFO)
    _origin_inputs_.append(input_arg)
    # 与 CreateConvSubgraphAttr 对齐：x=NCHW，filter=FRACTAL_Z，其余可选输入按 ND。
    _origin_inputs_[-1]["ori_shape"] = _origin_inputs_[-1]["shape"]
    if logical_index == 0:
        _origin_inputs_[-1]["ori_format"] = "NCHW"
        _origin_inputs_[-1]["format"] = "NCHW"
    elif logical_index == 1:
        _origin_inputs_[-1]["ori_format"] = "NCHW"
        _origin_inputs_[-1]["format"] = "FRACTAL_Z"
    else:
        _origin_inputs_[-1]["ori_format"] = "ND"
        _origin_inputs_[-1]["format"] = "ND"


def build_conv_args(args_list, config):
    """构造 Conv2DV2/ExtendConv2D 二次 tiling 入参。

    ExtendConv2D 逻辑槽位固定 10 个：0 x、1 filter、2 bias(has_bias)、3 offset_w 恒空、
    4 scale0(has_scale0)、5-9 恒空。Conv2DV2 仍按已连接输入紧凑排布。
    """
    _origin_inputs_ = []
    _origin_outputs_ = []
    input_num = config.input_num
    logger.info("CV fusion op, conv input(%s)", input_num)

    input_args = args_list[:input_num]  # input_num是实际输入个数
    input_index = 0

    def take_connected(logical_index):
        nonlocal input_index
        if input_index >= len(input_args):
            logger.error(
                "CV fusion op, conv input %s missing, input_num=%s, has_bias=%s, has_scale0=%s",
                logical_index,
                input_num,
                config.has_bias,
                config.has_scale0,
            )
            _origin_inputs_.append(None)
            return
        fill_conv_origin_input(_origin_inputs_, input_args[input_index], logical_index)
        input_index += 1

    def take_empty(logical_index):
        _origin_inputs_.append(None)
        logger.info("CV fusion op, conv input %s is nullptr", logical_index)

    if config.is_extend_conv2d:
        extend_conv2d_logical_input_num = 10
        for i in range(extend_conv2d_logical_input_num):
            if i in (0, 1):
                take_connected(i)
            elif i == 2:
                take_connected(i) if config.has_bias else take_empty(i)
            elif i == 4:
                take_connected(i) if config.has_scale0 else take_empty(i)
            else:
                take_empty(i)
    else:
        for i in range(input_num):
            take_connected(i)

    op_msg = "Processing output " + ":" + str(args_list[-2])
    logger.info("CV fusion op, conv output info: %s", op_msg)
    CommonUtility.print_compile_log("", op_msg, AscendCLogLevel.LOG_INFO)
    _origin_outputs_.append(args_list[-2])
    _origin_outputs_[-1]["param_name"] = "output0"
    _origin_outputs_[-1]["shape"] = args_list[-2]["shape"]
    _origin_outputs_[-1]["dtype"] = args_list[-2]["dtype"]
    _origin_outputs_[-1]["ori_shape"] = args_list[-2]["shape"]
    _origin_outputs_[-1]["ori_format"] = config.data_format
    _origin_outputs_[-1]["format"] = config.data_format
    if config.is_extend_conv2d:
        # ExtendConv2D proto 含可选第二输出；tiling 侧按双输出占位，避免输出个数校验失败。
        _origin_outputs_.append(_origin_outputs_[-1])
    logger.info("CV fusion op, new conv output info: %s", str(_origin_outputs_[-1]))
    return _origin_inputs_, _origin_outputs_


def _process_tiling_info(
    is_batch, compile_info, origin_inputs, origin_outputs, attrs, tiling_key_list
):
    tiling_info = TilingInfo()
    op_type = "BatchMatMulV3" if is_batch else "MatMulV3"
    tiling_data_type = f"{op_type}TilingData"

    run_info = do_op_tiling(
        op_type, compile_info, origin_inputs, origin_outputs, None, None, attrs
    )
    tiling_info.tiling_data = run_info["tiling_data"]
    tiling_info.tiling_key = run_info["tiling_key"]
    tiling_key_list[0] = tiling_info.tiling_key
    cube_block_dim = run_info["block_dim"]
    tiling_info.file_content = gen_static_shape_v2(
        op_type, tiling_data_type, run_info["tiling_data"]
    )

    return tiling_info, cube_block_dim


def is_matmul_relu_fixpip(tiling_info, cube_info):
    has_relu = cube_info[4]
    tiling_key_transpose_and_full_load_mask = (
        0xF10F0  # transpose/全载模式/基础或者切k模板
    )
    cube_tiling_key_fixpip = (
        tiling_info.tiling_key & ~tiling_key_transpose_and_full_load_mask
    )
    return has_relu and (cube_tiling_key_fixpip == 1)


def _is_fp32_large_k(a_trans, a_shape, cube_output_type_size, enable_hf32):
    """fp32类型且K轴>=2048且未启用hf32时, CV融合不能选择UB模板(精度问题), 需走common兜底"""
    if cube_output_type_size != 4:  # 非fp32
        return False
    if enable_hf32:  # 启用hf32时不走common兜底
        return False
    if not a_shape or len(a_shape) < 2:
        return False
    k_value = a_shape[-2] if a_trans else a_shape[-1]
    logger.info(
        "CV fusion op, fp32 large K check, a_trans=%s, k_value=%s", a_trans, k_value
    )
    return k_value >= 2048


def is_fp32_large_k_matmul(tiling_info, cube_info, enable_hf32=False):
    """静态场景: 从 tiling_key bit4-5 解码 A_TRANS"""
    if tiling_info is None or len(cube_info) <= 5:
        return False
    origin_inputs = cube_info[5]
    if not origin_inputs or not isinstance(origin_inputs[0], dict):
        return False
    a_trans = (tiling_info.tiling_key >> 4) & 0x3
    return _is_fp32_large_k(
        a_trans, origin_inputs[0].get("shape"), cube_info[0], enable_hf32
    )


def is_fp32_large_k_matmul_dynamic(cube_info, a_trans, enable_hf32=False):
    """动态场景: 直接传入 a_trans, origin_inputs 在 cube_info[4]"""
    if len(cube_info) <= 4:
        return False
    origin_inputs = cube_info[4]
    if not origin_inputs or not isinstance(origin_inputs[0], dict):
        return False
    return _is_fp32_large_k(
        a_trans, origin_inputs[0].get("shape"), cube_info[0], enable_hf32
    )


def _apply_fixpip_fusion(tiling_info, cube_block_dim):
    """fixpip融合模式: relu + fixpip场景, 走UB无db模板"""
    content = f"#define CUBE_BLOCK_DIM {cube_block_dim}\n"
    logger.info("CV fusion op, entering fixpip fusion mode.")
    tiling_info.file_content += "\n#define CV_UB_NO_DB 1\n"  # 防止编译问题
    return content


SafetyFusionCtx = namedtuple(
    "SafetyFusionCtx",
    [
        "compile_kwargs",
        "cross_info",
        "tiling_info",
        "is_batch",
        "cube_block_dim",
        "tiling_key",
        "use_cv_common",
    ],
)


def _should_use_safety_fusion(
    tiling_key, cube_tiling_key_ub, tiling_info, cube_info, enable_hf32
):
    """判断是否需要走safety兜底模式"""
    if tiling_key == -1:
        return True
    if cube_tiling_key_ub != 1:
        return True
    return is_fp32_large_k_matmul(tiling_info, cube_info, enable_hf32)


def _apply_safety_fusion(ctx):
    """safety兜底融合模式: 走common模板"""
    tiling_info = ctx.tiling_info
    is_in_mix_white_list = (
        ctx.is_batch and tiling_info.tiling_key in CV_COMMON_BMM_MIX_WHITE_LIST
    ) or (not ctx.is_batch and tiling_info.tiling_key in CV_COMMON_MIX_WHITE_LIST)
    if is_in_mix_white_list:
        tiling_info.file_content += "\n#define CV_SAFETY_FUSION_MIX_MODE 1\n"
    logger.info(
        "CV fusion op, entering safety fusion mode. vector_tiling_key=%s, is_batch=%s, cube_tiling_key=%s",
        ctx.tiling_key,
        ctx.is_batch,
        tiling_info.tiling_key,
    )
    tiling_info.file_content += "\n#define CV_SAFETY_FUSION 1\n"
    ctx.use_cv_common[0] = True
    vec_block_dim, wss = static_shape_cv_common_compile(
        **ctx.compile_kwargs, cross_info=ctx.cross_info
    )
    logger.info(
        "CV fusion op, CV_AIC_NUM=[%s] CV_AIV_NUM=[%s] CV_VEC_WSS=[%s]",
        str(ctx.cube_block_dim),
        str(vec_block_dim),
        str(wss),
    )
    for name, value in [
        ("CV_AIC_NUM", ctx.cube_block_dim),
        ("CV_AIV_NUM", vec_block_dim),
        ("CV_VEC_WSS", wss),
    ]:
        if value >= 0:
            tiling_info.file_content += f"\n#define {name} {value}\n"
    new_block_dim = (
        (vec_block_dim + 1) // 2
        if ctx.cube_block_dim * 2 < vec_block_dim
        else ctx.cube_block_dim
    )
    return f"#define CUBE_BLOCK_DIM {new_block_dim}\n"


def _apply_ub_fusion(tiling_info, tiling_key, cube_block_dim):
    """UB融合模式: 走UB复用模板"""
    content = f"#define CUBE_BLOCK_DIM {cube_block_dim}\n"
    logger.info(
        "CV fusion op, entering UB fusion mode, cube_tiling_key=%s, vector tilingkey=%s.",
        tiling_info.tiling_key,
        tiling_key,
    )
    tiling_info.file_content += "\n#define CV_UB_FUSION 1\n"
    if tiling_key == 0:
        logger.info("CV fusion op, entering UB fusion mode with no db.")
        tiling_info.file_content += "\n#define CV_UB_NO_DB 1\n"
    else:  # tiling_key 为1表示UB复用循环模板(非全载模板)
        logger.info("CV fusion op, entering UB fusion mode with db.")
        tiling_info.file_content += "\n#define CV_UB_DB 1\n"
    return content


def template_decider(
    compile_context, tiling_info, cube_info, is_conv=False, enable_hf32=False
):
    _, is_batch, cube_block_dim, use_cv_common, has_relu = cube_info[:5]
    compile_kwargs = {
        "kernel_name": compile_context.kernel_name,
        "temp_dir": compile_context.temp_dir,
        "graph_name": compile_context.graph_name,
        "vector_core_num": compile_context.vector_core_num,
    }
    cross_info = CrossCompileInfo(
        is_cross_compile_flag=compile_context.is_cross_compile_flag,
        cross_compiler_prefix=compile_context.cross_compiler_prefix,
        target_machine=compile_context.target_machine,
    )
    tiling_key = static_shape_cv_compile(**compile_kwargs, cross_info=cross_info)
    host_tiling_content = f"#define CUBE_TILING_KEY {tiling_key}\n"
    logger.info("CV fusion op, get vector tilingkey(%s)", tiling_key)
    use_cv_common = use_cv_common or [False]
    cube_tiling_key_ub = tiling_info.tiling_key & ~0xF0

    # 分支1: fixpip融合模式
    if is_matmul_relu_fixpip(tiling_info, cube_info) and not is_conv:
        host_tiling_content += _apply_fixpip_fusion(tiling_info, cube_block_dim)
    # 分支2: safety兜底模式(非UB key/无UB模板/fp32大K)
    elif not is_conv and _should_use_safety_fusion(
        tiling_key, cube_tiling_key_ub, tiling_info, cube_info, enable_hf32
    ):
        host_tiling_content += _apply_safety_fusion(
            SafetyFusionCtx(
                compile_kwargs=compile_kwargs,
                cross_info=cross_info,
                tiling_info=tiling_info,
                is_batch=is_batch,
                cube_block_dim=cube_block_dim,
                tiling_key=tiling_key,
                use_cv_common=use_cv_common,
            )
        )
    # 分支3: UB融合模式
    else:
        host_tiling_content += _apply_ub_fusion(tiling_info, tiling_key, cube_block_dim)
    return host_tiling_content


def map_dtype_to_string(dtype):
    dtype_map = {
        "bfloat16": "bfloat16_t",
        "float16": "half",
        "float32": "float",
        "int8": "int8_t",
        "int16": "int16_t",
        "int32": "int32_t",
        "int64": "int64_t",
        "uint8": "uint8_t",
        "uint16": "uint16_t",
        "uint32": "uint32_t",
        "uint64": "uint64_t",
    }

    return dtype_map.get(dtype.lower(), dtype)


def create_matmul_tiling_undef_header():
    return """
#ifndef __UNDEF_MATMULV3_HEADER__
#define __UNDEF_MATMULV3_HEADER__
#undef GET_TILING_DATA_PTR_WITH_STRUCT
#undef COPY_TILING_WITH_STRUCT
#undef COPY_TILING_WITH_ARRAY
#undef GET_TILING_DATA
#undef GET_TILING_DATA_MEMBER
#undef GET_TILING_DATA_WITH_STRUCT
#undef __tiling_data_ptr__
#endif
"""


def _write_matmul_host_tiling_files(
    compile_context, tiling_info, cube_info, enable_hf32, host_tiling_content_old
):
    """写入matmul静态场景host端tiling文件(含cv_common兜底目录)"""
    host_tiling_content_old += template_decider(
        compile_context, tiling_info, cube_info, enable_hf32=enable_hf32
    )
    for sub in ("", "cv_common"):
        generate_file(
            os.path.join(compile_context.temp_dir, "host", sub),
            "autofuse_cube_tiling_data.h",
            host_tiling_content_old,
        )


def _write_matmul_device_tiling_files(compile_context, tiling_info, cube_info):
    """写入matmul静态场景device端tiling文件(含cv_common兜底目录)"""
    _, _, _, _, has_relu, origin_inputs, origin_outputs = cube_info[:7]
    device_tiling_data = f"""\n#include "arch35/mat_mul_tiling_data.h"
#define IS_ENABLE_RELU {str(has_relu).lower()}
#define OP_TYPE_RELU_VALUE {5 if has_relu else 0}UL // 自动融合新增
#define DTYPE_X1 {map_dtype_to_string(origin_inputs[-1]["dtype"])}
#define DTYPE_X2 {map_dtype_to_string(origin_inputs[-1]["dtype"])}
#define DTYPE_Y {map_dtype_to_string(origin_outputs[-1]["dtype"])}
#define DTYPE_BIAS {map_dtype_to_string(origin_outputs[-1]["dtype"])}
"""
    tiling_info_undef = create_matmul_tiling_undef_header()
    tiling_info.file_content += device_tiling_data
    device_tiling_content = tiling_info_undef + tiling_info.file_content
    for sub in ("", "cv_common"):
        generate_file(
            os.path.join(compile_context.temp_dir, "device", sub),
            "autofuse_cube_tiling_data.h",
            device_tiling_content,
        )


def create_matmul_tiling_data(
    compile_context, tiling_info, cube_info, enable_hf32=False
):
    cube_output_type_size = cube_info[0]
    is_batch = cube_info[1]

    # 根据is_batch设置结构体名称和数据访问路径
    struct_name = (
        "BatchMatMulV3BasicTilingData" if is_batch else "MatMulV3BasicTilingData"
    )
    data_prefix = "tmpTilingData->matMulTilingData" if is_batch else "(*tmpTilingData)"

    # 生成host端tiling数据
    host_tiling_data = f"""
#include "arch35/mat_mul_tiling_data.h"
const int32_t cube_output_type_size = {cube_output_type_size};
GET_TILING_DATA_PTR_WITH_STRUCT({struct_name}, tmpTilingData, tmpTilingGM);
"""

    # 构建class_body，使用data_prefix处理不同的数据访问路径
    # ub_align_value固定32元素: 与kernel侧de445b95的上确界行宽对齐(basen_align
    # 为所有dtype的32B对齐行宽上确界), 保证host侧basen_basem_align与kernel一致
    class_body = "const int32_t ub_align_value = 32;\n"
    class_body += (
        f"const int32_t basen_align = ({data_prefix}.baseN + ub_align_value - 1) "
        f"/ ub_align_value * ub_align_value;\n"
    )
    class_body += f"const int32_t basen_basem_align = ({data_prefix}.baseM * basen_align) / 2 + basen_align;\n"
    if is_matmul_relu_fixpip(tiling_info, cube_info):
        class_body += "#define CV_RELU_FIXPIP_MODE 1\n"

    # 写入host端文件(空tiling_key/cube_block_dim占位)
    host_tiling_content = tiling_info.file_content + host_tiling_data + class_body
    host_tiling_content_old = host_tiling_content
    host_tiling_content += "#define CUBE_TILING_KEY 0\n"
    host_tiling_content += "#define CUBE_BLOCK_DIM 0\n"
    for sub in ("", "cv_common"):
        generate_file(
            os.path.join(compile_context.temp_dir, "host", sub),
            "autofuse_cube_tiling_data.h",
            host_tiling_content,
        )
    # 写入host端文件(带vector模板选择宏)
    _write_matmul_host_tiling_files(
        compile_context, tiling_info, cube_info, enable_hf32, host_tiling_content_old
    )
    # 写入device端文件
    _write_matmul_device_tiling_files(compile_context, tiling_info, cube_info)


def create_matmul_dynamic_tiling_data(temp_dir, cube_info):
    _, is_batch, use_cv_common, has_relu, origin_inputs, origin_outputs = cube_info[:7]
    use_cv_common_flag = (
        use_cv_common[0] if isinstance(use_cv_common, list) else bool(use_cv_common)
    )
    # 生成host端tiling数据
    struct_name = (
        "BatchMatMulV3BasicTilingData" if is_batch else "MatMulV3BasicTilingData"
    )
    head_tiling_data = f"""
#ifndef DYNAMIC_MM_TILING_DATA
#define DYNAMIC_MM_TILING_DATA
#include "arch35/mat_mul_tiling_data.h"
#include "autofuse_tiling_data.h"
struct CVTilingData {{
    uint32_t fusion_mode; // 0:ub; 1:safety
    uint32_t ub_mode; // 0:no db; 1:db
    uint32_t cv_aic_num;
    uint32_t cv_aiv_num;
    uint32_t cv_vec_wss;
    uint32_t mix_mode;
}};
struct CVAutofuseTilingData {{
    AutofuseTilingData tiling_data;
    CVTilingData  cv_tiling_data;
    {struct_name} matmul_tiling_data;
}};
"""

    # 写入host端文件
    host_tiling_content = head_tiling_data
    host_tiling_content += "#endif"
    generate_file(
        os.path.join(temp_dir, "host"),
        "autofuse_cube_tiling_data.h",
        host_tiling_content,
    )
    generate_file(
        os.path.join(temp_dir, "host", "cv_common"),
        "autofuse_cube_tiling_data.h",
        host_tiling_content,
    )

    # 写入device端文件
    if use_cv_common_flag:
        fusion_mode_define = "#define CV_SAFETY_FUSION 1\n"
    else:
        fusion_mode_define = "#define CV_UB_FUSION 1\n#define CV_UB_NO_DB 1\n"
    device_tiling_data = head_tiling_data
    device_tiling_data += f"""
{fusion_mode_define}
#define IS_ENABLE_RELU {str(has_relu).lower()}
#define OP_TYPE_RELU_VALUE {5 if has_relu else 0}UL // 自动融合新增
#define DTYPE_X1 {map_dtype_to_string(origin_inputs[-1]["dtype"])}
#define DTYPE_X2 {map_dtype_to_string(origin_inputs[-1]["dtype"])}
#define DTYPE_Y {map_dtype_to_string(origin_outputs[-1]["dtype"])}
#define DTYPE_BIAS {map_dtype_to_string(origin_outputs[-1]["dtype"])}
#endif
"""

    device_tiling_content = device_tiling_data
    generate_file(
        os.path.join(temp_dir, "device"),
        "autofuse_cube_tiling_data.h",
        device_tiling_content,
    )
    generate_file(
        os.path.join(temp_dir, "device", "cv_common"),
        "autofuse_cube_tiling_data.h",
        device_tiling_content,
    )


def create_conv_tiling_undef_header():
    return """
#ifndef __UNDEF_CONV2D_HEADER__
#define __UNDEF_CONV2D_HEADER__
#undef GET_TILING_DATA_PTR_WITH_STRUCT
#undef COPY_TILING_WITH_STRUCT
#undef COPY_TILING_WITH_ARRAY
#undef GET_TILING_DATA
#undef GET_TILING_DATA_MEMBER
#undef GET_TILING_DATA_WITH_STRUCT
#undef __tiling_data_ptr__
#endif
"""


def create_conv_tiling_data(compile_context, tiling_info, cube_info, cube_attributes):
    (
        cube_output_type_size,
        _,
        _,
        _,
        _,
        has_bias,
        has_offset_w,
        has_scale0,
        origin_inputs,
        origin_outputs,
    ) = cube_info[:10]
    struct_name = "Conv2DTilingData"
    data_prefix = "(*tmpTilingData)"

    host_tiling_data = f"""
// conv2d
#include "arch35/conv2d_v2_tiling_def.h"
#include <algorithm>
const int32_t cube_output_type_size = {cube_output_type_size};
GET_TILING_DATA_PTR_WITH_STRUCT({struct_name}, tmpTilingData, tmpTilingGM);
"""

    class_body = "const int32_t ub_align_value = 32 / cube_output_type_size;\n"
    class_body += (
        f"const int32_t basen_align = ({data_prefix}.hoL0 + ub_align_value - 1) "
        f"/ ub_align_value * ub_align_value;\n"
    )
    npu_arch = cube_attributes.get("npu_arch", "5102")
    vec_num = 1
    if npu_arch != "5102":
        vec_num = 2
    class_body += f"const int32_t basen_basem_align = std::max({data_prefix}.nL0 / {vec_num}, 16u) * basen_align;\n"

    host_tiling_content = tiling_info.file_content + host_tiling_data + class_body
    host_tiling_content_old = host_tiling_content
    host_tiling_content += "#define CUBE_TILING_KEY 0\n"
    host_tiling_content += "#define CUBE_BLOCK_DIM 0\n"
    generate_file(
        os.path.join(compile_context.temp_dir, "host"),
        "autofuse_cube_tiling_data.h",
        host_tiling_content,
    )
    generate_file(
        os.path.join(compile_context.temp_dir, "host", "cv_common"),
        "autofuse_cube_tiling_data.h",
        host_tiling_content,
    )
    # 生成决定走哪一个vector模板的宏
    host_tiling_content_old += template_decider(
        compile_context, tiling_info, cube_info, True
    )
    generate_file(
        os.path.join(compile_context.temp_dir, "host"),
        "autofuse_cube_tiling_data.h",
        host_tiling_content_old,
    )
    generate_file(
        os.path.join(compile_context.temp_dir, "host", "cv_common"),
        "autofuse_cube_tiling_data.h",
        host_tiling_content_old,
    )

    device_tiling_data = f"""\n#include "arch35/conv2d_v2_tiling_def.h"
#define IS_ENABLE_BIAS {str(has_bias).lower()}
#define IS_ENABLE_OFFSET_W {str(has_offset_w).lower()}
#define IS_ENABLE_SCALE0 {str(has_scale0).lower()}
#define DTYPE_X1 {map_dtype_to_string(origin_inputs[0]["dtype"])}
#define DTYPE_X2 {map_dtype_to_string(origin_inputs[1]["dtype"])}
#define DTYPE_Y {map_dtype_to_string(origin_outputs[0]["dtype"])}
"""
    device_tiling_data += (
        f"""#define DTYPE_BIAS {map_dtype_to_string(origin_inputs[2]["dtype"])}\n"""
        if has_bias
        else "#define DTYPE_BIAS int32_t\n"
    )
    device_tiling_data += (
        f"""#define DTYPE_OFFSET_W {map_dtype_to_string(origin_inputs[3]["dtype"])}\n"""
        if has_offset_w
        else "#define DTYPE_OFFSET_W uint64_t\n"
    )
    device_tiling_data += (
        f"""#define DTYPE_SCALE0 {map_dtype_to_string(origin_inputs[4]["dtype"])}\n"""
        if has_scale0
        else "#define DTYPE_SCALE0 uint64_t\n"
    )
    tiling_data_undef = create_conv_tiling_undef_header()
    tiling_info.file_content += device_tiling_data
    device_tiling_content = tiling_data_undef
    device_tiling_content += tiling_info.file_content
    generate_file(
        os.path.join(compile_context.temp_dir, "device"),
        "autofuse_cube_tiling_data.h",
        device_tiling_content,
    )
    generate_file(
        os.path.join(compile_context.temp_dir, "device", "cv_common"),
        "autofuse_cube_tiling_data.h",
        device_tiling_content,
    )


def ascbc_conv_kernel_tiling_pro(
    *args,
    temp_dir,
    graph_name,
    kernel_name,
    input_num,
    output_num,
    use_list_tensor_desc,
    cube_attrs,
    tiling_key_list,
    use_cv_common=None,
    vector_core_num=None,
    is_cross_compile_flag=False,
    cross_compiler_prefix=None,
    target_machine=None,
):
    graph_name = camel_to_snake(graph_name)
    args_list = args[0]
    if use_list_tensor_desc:
        inputs = args_list[:input_num]
        outputs = args_list[input_num : input_num + output_num]
        args_list = [inputs, outputs]
        input_num = 1
        output_num = 1

    if cube_attrs is None:
        logger.error("kernel_name=[%s] can't find cube attrs", kernel_name)
        return
    cube_attributes = cube_attrs.get("cube_attributes", {})
    if not cube_attributes:
        logger.error("kernel_name=[%s] can't find cube attributes", kernel_name)
        return

    has_bias = cube_attributes.get("has_bias", False)
    has_offset_w = cube_attributes.get("has_offset_w", False)
    has_scale0 = cube_attributes.get("has_scale0", False)
    is_extend_conv2d = cube_attributes.get("is_extend_conv2d", False)

    strides = cube_attributes.get("strides", [1, 1])
    pads = cube_attributes.get("pads", [0, 0, 0, 0])
    dilations = cube_attributes.get("dilations", [1, 1])
    groups = cube_attributes.get("groups", 1)
    data_format = cube_attributes.get("data_format", "NCHW")
    offset_x = cube_attributes.get("offset_x", 0)
    round_mode = cube_attributes.get("round_mode", "rint")
    pad_mode = cube_attributes.get("pad_mode", "SPECIFIC")
    enable_hf32 = cube_attributes.get("enable_hf32", False)
    fixed_shift_value = cube_attributes.get("fixed_shift_value", 0)
    enable_relu0 = cube_attributes.get("enable_relu0", False)
    enable_relu1 = cube_attributes.get("enable_relu1", False)
    dual_output = cube_attributes.get("dual_output", False)
    dtype0 = cube_attributes.get("dtype0", -1)
    dtype1 = cube_attributes.get("dtype1", -1)

    op_type = "ExtendConv2D" if is_extend_conv2d else "Conv2DV2"

    # attr 顺序需与对应 op proto 一致：ExtendConv2D 在 pad_mode 前插入 round_mode。
    conv_attr = []
    conv_attr.append({"name": "strides", "dtype": "list_int", "value": strides})
    conv_attr.append({"name": "pads", "dtype": "list_int", "value": pads})
    conv_attr.append({"name": "dilations", "dtype": "list_int", "value": dilations})
    conv_attr.append({"name": "groups", "dtype": "int", "value": groups})
    conv_attr.append({"name": "data_format", "dtype": "str", "value": data_format})
    conv_attr.append({"name": "offset_x", "dtype": "int", "value": offset_x})
    if is_extend_conv2d:
        conv_attr.append({"name": "round_mode", "dtype": "str", "value": round_mode})
    conv_attr.append({"name": "pad_mode", "dtype": "str", "value": pad_mode})
    conv_attr.append({"name": "enable_hf32", "dtype": "bool", "value": enable_hf32})
    if is_extend_conv2d:
        conv_attr.append(
            {"name": "enable_relu0", "dtype": "bool", "value": enable_relu0}
        )
        conv_attr.append(
            {"name": "enable_relu1", "dtype": "bool", "value": enable_relu1}
        )
        conv_attr.append({"name": "dual_output", "dtype": "bool", "value": dual_output})
        conv_attr.append({"name": "dtype0", "dtype": "int", "value": dtype0})
        conv_attr.append({"name": "dtype1", "dtype": "int", "value": dtype1})

    conv_input_num = cube_attributes.get("input_num", 0)
    conv_args_config = ConvArgsConfig(
        input_num=conv_input_num,
        data_format=data_format,
        has_bias=has_bias,
        has_scale0=has_scale0,
        is_extend_conv2d=is_extend_conv2d,
    )
    _origin_inputs_, _origin_outputs_ = build_conv_args(args_list, conv_args_config)

    tiling_info = TilingInfo()
    context = get_context()
    non_none_origin_inputs = [item for item in _origin_inputs_ if item is not None]
    _change_param_name_to_name(non_none_origin_inputs)
    compile_info = context.get_compile_info()
    tiling_config = {
        "name": "ascendc_op_para_size",
        "dtype": "int",
        "value": 2 * 1024 * 1024,
    }
    conv_attr.append(tiling_config)
    # fixed_shift_value 为 private attr，需显式追加，保证与 ascendc_op_para_size 的 IR 序一致。
    conv_attr.append(
        {"name": "fixed_shift_value", "dtype": "int", "value": fixed_shift_value}
    )

    tiling_data_type = "Conv2DTilingData"

    run_info = do_op_tiling(
        op_type, compile_info, _origin_inputs_, _origin_outputs_, None, None, conv_attr
    )
    tiling_info.tiling_data = run_info["tiling_data"]
    tiling_info.tiling_key = run_info["tiling_key"]
    tiling_key_list[0] = tiling_info.tiling_key
    cube_block_dim = run_info["block_dim"]
    tiling_info.file_content = gen_static_shape_v2(
        op_type, tiling_data_type, run_info["tiling_data"]
    )

    logger.info(
        "kernel_name=[%s], cube tiling_key[%s], tiling_data=[%s], tiling_file_context=[%s]",
        kernel_name,
        str(tiling_info.tiling_key),
        str(tiling_info.tiling_data),
        str(tiling_info.file_content),
    )
    cube_output_type_size = cube_attributes.get("type_size", 4)
    cube_info = [
        cube_output_type_size,
        False,
        cube_block_dim,
        use_cv_common,
        False,
        has_bias,
        has_offset_w,
        has_scale0,
        _origin_inputs_,
        _origin_outputs_,
    ]
    compile_context = StaticCompileContext(
        kernel_name,
        temp_dir,
        graph_name,
        vector_core_num,
        is_cross_compile_flag,
        cross_compiler_prefix,
        target_machine,
    )
    create_conv_tiling_data(compile_context, tiling_info, cube_info, cube_attributes)


def ascbc_matmul_kernel_tiling_pro(
    *args,
    temp_dir,
    graph_name,
    kernel_name,
    input_num,
    output_num,
    use_list_tensor_desc,
    cube_attrs,
    tiling_key_list,
    use_cv_common=None,
    vector_core_num=None,
    is_cross_compile_flag=False,
    cross_compiler_prefix=None,
    target_machine=None,
):
    graph_name = camel_to_snake(graph_name)
    args_list = args[0]
    if use_list_tensor_desc:
        inputs = args_list[:input_num]
        outputs = args_list[input_num : input_num + output_num]
        args_list = [inputs, outputs]
        input_num = 1
        output_num = 1

    if cube_attrs is None:
        logger.error("kernel_name=[%s] can't find cube attrs", kernel_name)
        return
    cube_attributes = cube_attrs.get("cube_attributes", {})
    if not cube_attributes:
        logger.error("kernel_name=[%s] can't find cube attributes", kernel_name)
        return

    is_batch = cube_attributes.get("is_batch", False)
    has_relu = cube_attributes.get("has_relu", False)

    mm_attr1 = {
        "name": "transpose_x1",
        "dtype": "bool",
        "value": cube_attributes.get("transpose_x1", False),
    }
    mm_attr2 = {
        "name": "transpose_x2",
        "dtype": "bool",
        "value": cube_attributes.get("transpose_x2", False),
    }
    mm_attr3 = {"name": "offset_x", "dtype": "int", "value": 0x80}
    if is_batch:
        mm_attr4 = {
            "name": "enable_hf32",
            "dtype": "bool",
            "value": cube_attributes.get("enable_hf32", False),
        }
    else:
        mm_attr4 = {
            "name": "opImplMode",
            "dtype": "int",
            "value": cube_attributes.get("enable_hf32", 0x1),
        }

    mm_input_num = cube_attributes.get("input_num", 0)
    _origin_inputs_, _origin_outputs_, _inputs_ = _build_mm_args(
        args_list, input_num=mm_input_num, mm_attr1=mm_attr1, mm_attr2=mm_attr2
    )

    attrs = [mm_attr1, mm_attr2, mm_attr3, mm_attr4]
    tiling_info = TilingInfo()
    context = get_context()
    _change_param_name_to_name(_inputs_)
    _change_param_name_to_name(_origin_inputs_)
    compile_info = context.get_compile_info()
    tiling_config = {
        "name": "ascendc_op_para_size",
        "dtype": "int",
        "value": 2 * 1024 * 1024,
    }
    attrs.append(tiling_config)
    is_batch = cube_attributes.get("is_batch", False)
    tiling_info, cube_block_dim = _process_tiling_info(
        is_batch,
        compile_info,
        _origin_inputs_,
        _origin_outputs_,
        attrs,
        tiling_key_list,
    )

    logger.info(
        "kernel_name=[%s], cube tiling_key[%s], tiling_data=[%s], tiling_file_context=[%s]",
        kernel_name,
        str(tiling_info.tiling_key),
        str(tiling_info.tiling_data),
        str(tiling_info.file_content),
    )
    cube_output_type_size = cube_attributes.get("type_size", 4)
    cube_info = [
        cube_output_type_size,
        is_batch,
        cube_block_dim,
        use_cv_common,
        has_relu,
        _origin_inputs_,
        _origin_outputs_,
    ]
    compile_context = StaticCompileContext(
        kernel_name,
        temp_dir,
        graph_name,
        vector_core_num,
        is_cross_compile_flag,
        cross_compiler_prefix,
        target_machine,
    )
    create_matmul_tiling_data(
        compile_context,
        tiling_info,
        cube_info,
        enable_hf32=cube_attributes.get("enable_hf32", False),
    )


def ascbc_matmul_kernel_dynamic_tiling_pro(
    *args,
    temp_dir,
    graph_name,
    kernel_name,
    input_num,
    output_num,
    use_list_tensor_desc,
    cube_attrs,
    use_cv_common=None,
):
    graph_name = camel_to_snake(graph_name)
    args_list = args[0]
    if use_list_tensor_desc:
        inputs = args_list[:input_num]
        outputs = args_list[input_num : input_num + output_num]
        args_list = [inputs, outputs]
        input_num = 1
        output_num = 1

    if cube_attrs is None:
        logger.error("kernel_name=[%s] can't find cube attrs", kernel_name)
        return
    cube_attributes = cube_attrs.get("cube_attributes", {})
    if not cube_attributes:
        logger.error("kernel_name=[%s] can't find cube attributes", kernel_name)
        return

    is_batch = cube_attributes.get("is_batch", False)
    has_relu = cube_attributes.get("has_relu", False)

    mm_attr1 = {
        "name": "transpose_x1",
        "dtype": "bool",
        "value": cube_attributes.get("transpose_x1", False),
    }
    mm_attr2 = {
        "name": "transpose_x2",
        "dtype": "bool",
        "value": cube_attributes.get("transpose_x2", False),
    }

    mm_input_num = cube_attributes.get("input_num", 0)
    _origin_inputs_, _origin_outputs_, _ = _build_mm_args(
        args_list, input_num=mm_input_num, mm_attr1=mm_attr1, mm_attr2=mm_attr2
    )

    cube_output_type_size = cube_attributes.get("type_size", 4)
    use_cv_common = use_cv_common or [False]
    cube_info = [
        cube_output_type_size,
        is_batch,
        use_cv_common,
        has_relu,
        _origin_inputs_,
        _origin_outputs_,
    ]
    a_trans = 1 if cube_attributes.get("transpose_x1", False) else 0
    enable_hf32 = cube_attributes.get("enable_hf32", False)
    if is_fp32_large_k_matmul_dynamic(cube_info, a_trans, enable_hf32):
        use_cv_common[0] = True
        logger.info("CV fusion op, fp32 large K dynamic, force use_cv_common")
    create_matmul_dynamic_tiling_data(temp_dir, cube_info)


def asc_graph_compile_post(
    host_build_dir,
    code_gen,
    graph_name,
    kernel_file,
    json_file,
    kernel_name,
    static_compile_flag,
    is_cube=False,
    is_cross_compile_flag=False,
    cross_compiler_prefix=None,
    target_machine=None,
):
    # 生成get_kernel.cpp代码
    timestamp_set(True, graph_name, "GenGetKernel")
    get_kernel_src = code_gen.get_kernel_and_json_generator(kernel_file, json_file)
    generate_file(host_build_dir, graph_name + "_get_kernel.cpp", get_kernel_src)
    timestamp_set(False, graph_name, "GenGetKernel")

    # 重新编译host so
    timestamp_set(True, graph_name, "CompileSecondHost")
    ascbc_host_compile(
        HostCompileContext(
            graph_name, kernel_name, host_build_dir, True, static_compile_flag, is_cube
        ),
        cross_info=CrossCompileInfo(
            is_cross_compile_flag, cross_compiler_prefix, target_machine
        ),
    )
    if is_cross_compile_flag:
        compile_native_host_so(
            HostCompileContext(
                graph_name,
                kernel_name,
                host_build_dir,
                True,
                static_compile_flag,
                is_cube,
            )
        )
    timestamp_set(False, graph_name, "CompileSecondHost", True)

    # 拷贝so和json
    host_so = os.path.join(host_build_dir, f"lib{kernel_name}.so")
    kernel_meta_dir = get_current_build_config("kernel_meta_parent_dir")
    dest = os.path.join(kernel_meta_dir, "kernel_meta")
    shutil.copy(host_so, dest)
    if is_cross_compile_flag:
        native_host_so = os.path.join(
            host_build_dir + "_native", f"lib{kernel_name}.so"
        )
        native_so_name = f"lib{kernel_name}_native.so"
        native_dest_path = os.path.join(dest, native_so_name)
        shutil.copy(native_host_so, native_dest_path)
    modify_json_file(json_file, host_so, is_cross_compile_flag=is_cross_compile_flag)
    if os.path.exists(kernel_file):
        os.remove(kernel_file)


def get_graph_basic_info(params, args) -> tuple:
    schedule_results = params["schedule_results"]
    graph_name = camel_to_snake(gen_valid_name(schedule_results.get_name()))
    input_num = schedule_results.get_input_num()
    output_num = schedule_results.get_output_num()
    is_cube = schedule_results.is_cube_type()
    cube_attrs = schedule_results.get_cube_attributes()
    is_conv = schedule_results.is_conv_type()
    kernel_name = args[-1]
    vector_core_num = params["vector_core_num"]
    msg = (
        f"graph_name:{graph_name}, input_num:{input_num}, output_num:{output_num},"
        f"kernel_name:{kernel_name}, vector_core_num:{vector_core_num}, is_cube:{is_cube}"
    )
    CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_DEBUG)
    return graph_name, input_num, output_num, is_cube, is_conv, cube_attrs


def create_compile_dirs(temp_dir) -> tuple:
    """创建device/host/pgo目录"""
    kernel_build_dir = os.path.join(temp_dir, "device")
    host_build_dir = os.path.join(temp_dir, "host")
    for dir_path in [kernel_build_dir, host_build_dir]:
        os.makedirs(dir_path, exist_ok=True)
    return kernel_build_dir, host_build_dir


def copy_so_and_modify_json(host_build_dir, kernel_name, json_file):
    host_so = os.path.join(host_build_dir, f"lib{kernel_name}.so")
    kernel_meta_dir = get_current_build_config("kernel_meta_parent_dir")
    shutil.copy(host_so, os.path.join(kernel_meta_dir, "kernel_meta"))
    modify_json_file(json_file, host_so)


def asc_graph_compile(*args, temp_dir, params):
    """入口为asc graph场景"""
    graph_name, input_num, output_num, is_cube, is_conv, cube_attrs = (
        get_graph_basic_info(params, args)
    )
    tiling_key_list, kernel_type_list = [-1], ["KERNEL_TYPE_AIV_ONLY"]
    _, host_build_dir = create_compile_dirs(temp_dir)
    kernel_name = args[-1]
    code_gen = CodeGen()

    cross_compile_flag = is_cross_compile(params)
    cross_compiler_prefix = (
        get_cross_compiler_prefix(params) if cross_compile_flag else None
    )
    target_machine = get_target_machine(params) if cross_compile_flag else None

    op_kernel_src, tiling_func_srcs = generate_device_and_host_code(
        graph_name=graph_name, temp_dir=temp_dir, params=params, code_gen=code_gen
    )
    static_compile_flag = is_static_compile(params, tiling_func_srcs)
    use_list_tensor_desc = op_kernel_src.find("kernel_operator_list_tensor_intf.h") > 0
    enable_parallel_compile = op_kernel_src.rfind("void fake_tiling_ids()") > 0
    use_cv_common = [False]
    vector_core_num = params.get("vector_core_num")
    replace_root = get_replace_kernel_root()

    if is_cube:
        if static_compile_flag:
            if is_conv:
                ascbc_conv_kernel_tiling_pro(
                    args,
                    temp_dir=temp_dir,
                    graph_name=graph_name,
                    kernel_name=kernel_name,
                    input_num=input_num,
                    output_num=output_num,
                    use_list_tensor_desc=use_list_tensor_desc,
                    cube_attrs=cube_attrs,
                    tiling_key_list=tiling_key_list,
                    use_cv_common=use_cv_common,
                    vector_core_num=vector_core_num,
                    is_cross_compile_flag=cross_compile_flag,
                    cross_compiler_prefix=cross_compiler_prefix,
                    target_machine=target_machine,
                )
            else:
                ascbc_matmul_kernel_tiling_pro(
                    args,
                    temp_dir=temp_dir,
                    graph_name=graph_name,
                    kernel_name=kernel_name,
                    input_num=input_num,
                    output_num=output_num,
                    use_list_tensor_desc=use_list_tensor_desc,
                    cube_attrs=cube_attrs,
                    tiling_key_list=tiling_key_list,
                    use_cv_common=use_cv_common,
                    vector_core_num=vector_core_num,
                    cross_compiler_prefix=cross_compiler_prefix,
                    is_cross_compile_flag=cross_compile_flag,
                    target_machine=target_machine,
                )
            replace_host_files_if_needed(
                replace_root, temp_dir, use_cv_common, graph_name
            )
            static_shape_compile(
                kernel_name=kernel_name,
                temp_dir=temp_dir,
                graph_name=graph_name,
                tiling_key_list=tiling_key_list,
                kernel_type_list=kernel_type_list,
                use_cv_common=use_cv_common,
                is_cube=is_cube,
                vector_core_num=vector_core_num,
                is_cross_compile_flag=cross_compile_flag,
                cross_compiler_prefix=cross_compiler_prefix,
                target_machine=target_machine,
            )
        else:
            if not is_conv:
                replace_host_files_if_needed(
                    replace_root, temp_dir, use_cv_common, graph_name
                )
                ascbc_matmul_kernel_dynamic_tiling_pro(
                    args,
                    temp_dir=temp_dir,
                    graph_name=graph_name,
                    kernel_name=kernel_name,
                    input_num=input_num,
                    output_num=output_num,
                    use_list_tensor_desc=use_list_tensor_desc,
                    cube_attrs=cube_attrs,
                    use_cv_common=use_cv_common,
                )
                dynamic_shape_compile(
                    ShapeCompileContext(kernel_name, temp_dir, graph_name),
                    use_cv_common=use_cv_common,
                    is_cube=is_cube,
                    cross_info=CrossCompileInfo(
                        cross_compile_flag, cross_compiler_prefix, target_machine
                    ),
                )

    elif static_compile_flag:
        _run_pgo_and_clean_cache(
            *args,
            temp_dir=temp_dir,
            params=params,
            op_kernel_src=op_kernel_src,
            cross_compile_flag=cross_compile_flag,
            host_build_dir=host_build_dir,
        )
        replace_host_files_if_needed(replace_root, temp_dir, use_cv_common, graph_name)
        timestamp_set(True, graph_name, "CompileHost")
        static_shape_compile(
            kernel_name=kernel_name,
            temp_dir=temp_dir,
            graph_name=graph_name,
            tiling_key_list=tiling_key_list,
            kernel_type_list=kernel_type_list,
            vector_core_num=vector_core_num,
            is_cross_compile_flag=cross_compile_flag,
            cross_compiler_prefix=cross_compiler_prefix,
            target_machine=target_machine,
        )
        timestamp_set(False, graph_name, "CompileHost")

    kernel_build_dir = get_device_build_dir(temp_dir, use_cv_common)
    replace_kernel(kernel_build_dir=kernel_build_dir, graph_name=graph_name)
    timestamp_set(True, graph_name, "CompileDevice")
    kernel_file, json_file = ascbc_kernel_compile(
        args,
        graph_name=graph_name,
        kernel_name=kernel_name,
        input_num=input_num,
        output_num=output_num,
        temp_build_dir=kernel_build_dir,
        impl_mode=params.get("impl_mode"),
        use_list_tensor_desc=use_list_tensor_desc,
        enable_parallel_compile=enable_parallel_compile,
        tiling_key=tiling_key_list[0],
        kernel_type=kernel_type_list[0],
        is_cube=is_cube,
        is_conv=is_conv,
    )
    timestamp_set(False, graph_name, "CompileDevice")
    host_build_dir = get_host_build_dir(temp_dir, use_cv_common)
    asc_graph_compile_post(
        host_build_dir,
        code_gen,
        graph_name,
        kernel_file,
        json_file,
        kernel_name,
        static_compile_flag,
        is_cube,
        is_cross_compile_flag=cross_compile_flag,
        cross_compiler_prefix=cross_compiler_prefix,
        target_machine=target_machine,
    )


def compute_graph_compile(*args, temp_dir, params, vector_core_num, device_id):
    """入口为compute graph场景"""
    timestamp_set(True, "start", "Schedule")
    compute_graph = ascir.utils.deserialize("compute_graph", params["compute_graph"])
    logger.info(
        "[AUTOFUSE_BACKEND]compile start graph[%s], build dir[%s], kernel_name[%s], "
        "symbol_source_info[%s], output_symbol_shape[%s]",
        compute_graph.get_name(),
        temp_dir,
        args[-1],
        params.get("symbol_source_info", "None"),
        params.get("output_symbol_shape", "[]"),
    )
    value = params.get("symbol_source_info")
    CommonUtility.print_compile_log(
        "", f"symbol_source_info is {type(value)}, {value}", AscendCLogLevel.LOG_DEBUG
    )
    if value is None or value == "null":
        CommonUtility.print_compile_log(
            "", "maybe a static shape", AscendCLogLevel.LOG_WARNING
        )
        symbol_source_info = None
    else:
        symbol_source_info = ascir.utils.deserialize("symbol_source_info", value)
    # Do schedule
    sched = Schedule()
    schedule_results = sched.scheduleV2(compute_graph)
    timestamp_set(False, gen_valid_name(compute_graph.get_name()), "Schedule")
    # 构造ascgraph参数
    asc_param = {}
    asc_param["schedule_results"] = schedule_results
    asc_param["symbol_source_info"] = symbol_source_info
    asc_param["output_symbol_shape"] = params.get("output_symbol_shape", "[]")
    asc_param["vector_core_num"] = vector_core_num
    asc_param["device_id"] = device_id
    asc_param["host_env_os"] = params.get("host_env_os", "")
    asc_param["host_env_cpu"] = params.get("host_env_cpu", "")
    asc_graph_compile(*args, temp_dir=temp_dir, params=asc_param)


def asc_codegen_compile(*args, **kwargs):
    """自动融合算子注册到tefusion的统一编译入口，编译完成后，将编译后的`so`和`json`写入`kernel meta`目录
    **args**
    * inputs：包含输入shape、dtype
    * outputs：包含输出shape、dtype
    * attrs：对于自动融合节点，不存在原型属性，因此该字段不存在
    * kernel_name: kernel name用于生成so名称 确保在最后一个参数中

    **op_info.extra_params**
    * fused_graph: compute graph序列化后的字符串 <string>
    * symbol_source_info: shape的符号化信息，存储了如`s0`是第0个输入的第2个维度信息 <dict>
    * output_symbol_shape: infer shape所需符号化表达 <list<str>>
    """

    def asc_codegen_compile_with_tmpdir(*args, temp_dir, **kwargs):
        codegen_context = get_codegen_compile_context()
        extra_params = codegen_context.extra_params
        log_core_limit_info(codegen_context)

        # 设置 platform 到 context
        npu_arch = get_soc_spec("NpuArch")
        if npu_arch:
            ascir.utils.set_platform(
                npu_arch,
                codegen_context.static_compile_vector_core_num,
                int(get_soc_spec("ub_size")),
            )
            CommonUtility.print_compile_log(
                "",
                f"Set platform from get_soc_spec: {npu_arch}",
                AscendCLogLevel.LOG_DEBUG,
            )

        CommonUtility.print_compile_log(
            "",
            f"params type: {type(extra_params)}, params: {extra_params}",
            AscendCLogLevel.LOG_DEBUG,
        )
        # 反序列化graph和symbol_source_info
        if not check_keys_in_dict(
            extra_params, ["compute_graph", "symbol_source_info"]
        ):
            CommonUtility.print_compile_log(
                "",
                "compute_graph and symbol_source_info do not exist",
                AscendCLogLevel.LOG_ERROR,
            )
            raise Exception("An error occurred autofuse compile for check extra_params")
        try:
            compute_graph_compile(
                *args,
                temp_dir=temp_dir,
                params=extra_params,
                vector_core_num=codegen_context.vector_core_num,
                device_id=codegen_context.device_id,
            )
        except Exception:
            dfx_flags = os.environ.get("AUTOFUSE_DFX_FLAGS", "")
            if "codegen_compile_debug=true" not in dfx_flags:
                try:
                    CommonUtility.print_compile_log(
                        "",
                        "generate debug info when catch exception",
                        AscendCLogLevel.LOG_INFO,
                    )
                    os.environ["AUTOFUSE_DFX_FLAGS"] = (
                        "codegen_compile_debug=true;" + dfx_flags
                    )
                    compute_graph_compile(
                        *args,
                        temp_dir=temp_dir,
                        params=extra_params,
                        vector_core_num=codegen_context.vector_core_num,
                        device_id=codegen_context.device_id,
                    )
                except Exception:
                    CommonUtility.print_compile_log(
                        "",
                        "catch exception when generate debug info",
                        AscendCLogLevel.LOG_INFO,
                    )
            raise

    kernel_meta_dir = get_current_build_config("kernel_meta_parent_dir")
    if not os.path.isabs(kernel_meta_dir):
        kernel_meta_dir = os.path.abspath(kernel_meta_dir)

    if not is_valid_path(kernel_meta_dir):
        CommonUtility.print_compile_log(
            "",
            f"invalid kernel meta path: `{kernel_meta_dir}'",
            AscendCLogLevel.LOG_ERROR,
        )
        raise Exception("An error occurred autofuse compile for check kernel_meta path")

    if not os.path.exists(kernel_meta_dir):
        CommonUtility.print_compile_log(
            "",
            f"kernel meta parent dir does not exist: `{kernel_meta_dir}'",
            AscendCLogLevel.LOG_ERROR,
        )
        raise Exception("An error occurred autofuse compile for check kernel_meta path")

    kernel_name = args[-1]
    from autofuse.compile_adapter import get_debug_flag

    compile_debug = get_debug_flag()
    if not compile_debug:
        with tempfile.TemporaryDirectory(
            prefix=kernel_name, dir=kernel_meta_dir
        ) as temp_dir:
            msg = f"kernel name:{kernel_name}, kernel_meta_dir:{kernel_meta_dir}, tmp:{temp_dir}"
            CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
            asc_codegen_compile_with_tmpdir(*args, temp_dir=temp_dir, **kwargs)
    else:
        os.environ["ASCEND_OP_COMPILE_SAVE_KERNEL_META"] = "1"
        temp_dir = tempfile.mkdtemp(prefix=kernel_name, dir=kernel_meta_dir)
        msg = f"kernel name:{kernel_name}, kernel_meta_dir:{kernel_meta_dir}, tmp:{temp_dir}"
        CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
        asc_codegen_compile_with_tmpdir(*args, temp_dir=temp_dir, **kwargs)
        os.environ.pop("ASCEND_OP_COMPILE_SAVE_KERNEL_META")

    CommonUtility.print_compile_log(
        "", "asc_codegen_compile finish", AscendCLogLevel.LOG_INFO
    )
