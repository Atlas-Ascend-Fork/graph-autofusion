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
import shutil
import re
from asc_op_compile_base.common.platform.platform_info import get_soc_spec
from asc_op_compile_base.asc_op_compiler import (
    compile_op,
    get_code_channel,
    OpInfo,
    compile_op_with_customized_config,
    KernelMetaType,
    TilingKeyConfig,
    CustomizedConfig,
)
from asc_op_compile_base.asc_op_compiler.compile_op import (
    CommonUtility,
    AscendCLogLevel,
)
from asc_op_compile_base.common.buildcfg import get_current_build_config


def camel_to_snake(camel_str):
    # 使用正则表达式匹配大写字母
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    # 使用正则表达式匹配小写字母后跟大写字母的情况
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def get_shortsoc_compile_option(compile_option_list: list, shortsoc: str):
    compile_options = []
    if shortsoc in compile_option_list:
        compile_options.extend(compile_option_list[shortsoc])
    if "__ALLSOC__" in compile_option_list:
        compile_options.extend(compile_option_list["__ALLSOC__"])
    return compile_options


def _build_args(args_list, input_num, output_num):
    _inputs_ = []
    _outputs_ = []
    _origin_inputs_ = []
    _origin_outputs_ = []
    # 遍历args中的每个元素
    for i, arg in enumerate(args_list[: (input_num + output_num)]):
        if i < input_num:
            msg = "Processing input " + str(i) + ":" + str(arg)
            CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
            _origin_inputs_.append(arg)
            if arg is not None:
                if isinstance(arg, (list, tuple)):
                    if len(arg) == 0:
                        continue
                    _inputs_.append(arg[0])
                else:
                    _inputs_.append(arg)
            else:
                _inputs_.append(arg)
            _inputs_[-1]["param_name"] = "input" + str(i)
            shape = _inputs_[-1]["shape"]
            ori_shape = _inputs_[-1]["ori_shape"]
            _inputs_[-1]["shape"] = shape + (-1,)
            _inputs_[-1]["ori_shape"] = ori_shape + (-1,)
        else:
            msg = "Processing output " + str(i - input_num) + ":" + str(arg)
            CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
            _origin_outputs_.append(arg)
            if arg is not None:
                if isinstance(arg, (list, tuple)):
                    if len(arg) == 0:
                        continue
                    _outputs_.append(arg[0])
                else:
                    _outputs_.append(arg)
            else:
                _outputs_.append(arg)
            _outputs_[-1]["param_name"] = "output" + str(i - input_num)
            shape = _outputs_[-1]["shape"]
            ori_shape = _outputs_[-1]["ori_shape"]
            _outputs_[-1]["shape"] = shape + (-1,)
            _outputs_[-1]["ori_shape"] = ori_shape + (-1,)
    return _origin_inputs_, _origin_outputs_, _inputs_, _outputs_


def _join_path(options, asc_path, is_cube, is_conv):
    options.append("-I" + os.path.join(asc_path, "impl", "adv_api"))
    options.append("-I" + os.path.join(asc_path, "impl", "basic_api"))
    options.append("-I" + os.path.join(asc_path, "impl", "utils"))
    options.append("-I" + os.path.join(asc_path, "include"))
    options.append("-I" + os.path.join(asc_path, "include", "adv_api"))
    options.append("-I" + os.path.join(asc_path, "include", "basic_api"))
    options.append("-I" + os.path.join(asc_path, "include", "aicpu_api"))
    options.append("-I" + os.path.join(asc_path, "include", "utils"))
    options.append("-I" + os.path.join(asc_path, "..", "..", "include"))
    options.append("-I" + os.path.join(asc_path, "..", "..", "include", "ascendc"))
    options.append("-I" + os.path.join(asc_path, "..", "ascendc", "act"))
    options.append("-I" + os.path.join(asc_path, "..", "tikcpp"))
    options.append("-I" + os.path.join(asc_path, "..", "tikcpp", "tikcfw"))
    options.append("-I" + os.path.join(asc_path, "..", "tikcpp", "tikcfw", "impl"))
    options.append("-I" + os.path.join(asc_path, "..", "tikcpp", "tikcfw", "interface"))
    options.append("-I" + os.path.join(asc_path, "..", "ascendc"))
    options.append("-I" + os.path.join(asc_path, "..", "ascendc", "include"))
    options.append("-I" + os.path.join(asc_path, "..", "asc", "include", "tiling"))
    if is_cube:
        options.append(
            "-I"
            + os.path.join(
                asc_path,
                "..",
                "..",
                "opp",
                "built-in",
                "op_impl",
                "ai_core",
                "tbe",
                "impl",
                "ascendc",
                "common",
            )
        )
        options.append(
            "-I"
            + os.path.join(
                asc_path,
                "..",
                "..",
                "opp",
                "built-in",
                "op_impl",
                "ai_core",
                "tbe",
                "impl",
                "ops_nn",
                "ascendc",
                "common",
            )
        )
        if is_conv:
            # conv2d kernel 编译统一使用 act_conv 自定义算子路径（模板库/tilingkey 单一来源）
            options.append(
                "-I"
                + os.path.join(
                    asc_path,
                    "..",
                    "..",
                    "opp",
                    "built-in",
                    "op_impl",
                    "ai_core",
                    "tbe",
                    "impl",
                    "ops_nn",
                    "ascendc",
                    "act_conv",
                )
            )
            options.append(
                "-I"
                + os.path.join(
                    asc_path,
                    "..",
                    "..",
                    "opp",
                    "built-in",
                    "op_impl",
                    "ai_core",
                    "tbe",
                    "impl",
                    "ops_nn",
                    "ascendc",
                    "act_conv",
                    "conv2d_v2",
                )
            )
            options.append(
                "-I"
                + os.path.join(
                    asc_path,
                    "..",
                    "..",
                    "opp",
                    "built-in",
                    "op_impl",
                    "ai_core",
                    "tbe",
                    "impl",
                    "ops_nn",
                    "ascendc",
                    "act_conv",
                    "common",
                )
            )
        else:
            options.append(
                "-I"
                + os.path.join(
                    asc_path,
                    "..",
                    "..",
                    "opp",
                    "built-in",
                    "op_impl",
                    "ai_core",
                    "tbe",
                    "impl",
                    "ascendc",
                    "mat_mul_v3",
                )
            )
            options.append(
                "-I"
                + os.path.join(
                    asc_path,
                    "..",
                    "..",
                    "opp",
                    "built-in",
                    "op_impl",
                    "ai_core",
                    "tbe",
                    "impl",
                    "ascendc",
                    "batch_mat_mul_v3",
                )
            )
            options.append(
                "-I"
                + os.path.join(
                    asc_path,
                    "..",
                    "..",
                    "opp",
                    "built-in",
                    "op_impl",
                    "ai_core",
                    "tbe",
                    "impl",
                    "ops_nn",
                    "ascendc",
                    "mat_mul_v3",
                )
            )
            options.append(
                "-I"
                + os.path.join(
                    asc_path,
                    "..",
                    "..",
                    "opp",
                    "built-in",
                    "op_impl",
                    "ai_core",
                    "tbe",
                    "impl",
                    "ops_nn",
                    "ascendc",
                    "batch_mat_mul_v3",
                )
            )
            options.append(
                "-I"
                + os.path.join(
                    asc_path,
                    "..",
                    "..",
                    "opp",
                    "built-in",
                    "op_impl",
                    "ai_core",
                    "tbe",
                    "impl",
                    "ascendc",
                    "common",
                    "cmct",
                )
            )
            options.append(
                "-I"
                + os.path.join(
                    asc_path,
                    "..",
                    "..",
                    "opp",
                    "built-in",
                    "op_impl",
                    "ai_core",
                    "tbe",
                    "impl",
                    "ops_nn",
                    "ascendc",
                    "common",
                    "cmct",
                )
            )


def _build_options(temp_build_dir, impl_mode, is_cube, is_conv):
    options = ["-x", "cce"]
    ascend_home_path = os.environ.get("ASCEND_HOME_PATH")
    import platform

    archlinux = platform.machine()
    if ascend_home_path is None or ascend_home_path == "":
        asc_opc_path = shutil.which("asc_opc")
        if asc_opc_path is not None:
            asc_opc_path_link = os.path.dirname(asc_opc_path)
            asc_opc_real_path = os.path.realpath(asc_opc_path_link)
            ascend_home_path = os.path.realpath(
                os.path.join(asc_opc_real_path, "..", "..")
            )
        else:
            ascend_home_path = "/usr/local/Ascend/cann"

    if "x86" in archlinux:
        asc_path = os.path.realpath(
            os.path.join(ascend_home_path, "x86_64-linux", "asc")
        )
    else:
        asc_path = os.path.realpath(
            os.path.join(ascend_home_path, "aarch64-linux", "asc")
        )
    if asc_path is None:
        asc_path = os.path.realpath(os.path.join(ascend_home_path, "compiler", "asc"))
    _join_path(options, asc_path, is_cube, is_conv)
    options.append("-I" + os.path.join(temp_build_dir))
    if impl_mode == "high_performance":
        options.append("-DHIGH_PERFORMANCE=1")
    elif impl_mode == "high_precision":
        options.append("-DHIGH_PRECISION=1")
    if get_current_build_config("enable_deterministic_mode") == 1:
        options.append("-DDETERMINISTIC_MODE=1")
    else:
        options.append("-DDETERMINISTIC_MODE=0")

    options.append("-DAUTO_FUSE_DEVICE=1")

    custom_compile_options = (
        {
            "_ALLSOC_": [
                "--cce-auto-sync=off",
                "-Wno-deprecated-declarations",
                "-Werror",
            ]
        },
    )
    custom_all_compile_options = ({"ascend950": ["--cce-simd-vf-fusion=true"]},)
    soc_short = get_soc_spec("SHORT_SOC_VERSION").lower()
    custom_compile_options_soc = get_shortsoc_compile_option(
        custom_compile_options[0], soc_short
    )
    custom_all_compile_options_soc = get_shortsoc_compile_option(
        custom_all_compile_options[0], soc_short
    )
    options += custom_all_compile_options_soc
    options += custom_compile_options_soc
    return options


def get_customized_tiling_config(tiling_key, kernel_type) -> CustomizedConfig:
    kernel_type_enum = (
        KernelMetaType.KERNEL_TYPE_AIV_ONLY
        if kernel_type == "KERNEL_TYPE_AIV_ONLY"
        else KernelMetaType.KERNEL_TYPE_MIX_AIV_1_0
    )
    tilingkey_config = TilingKeyConfig(
        kernel_type=kernel_type_enum,
        tiling_struct_name="AutofuseTilingData",
    )
    return CustomizedConfig(
        default_kernel_type=KernelMetaType.KERNEL_TYPE_AIV_ONLY,
        default_tiling_struct_name="AutofuseTilingData",
        tiling_key_infos={str(tiling_key): tilingkey_config},
    )


def ascbc_kernel_compile(
    *args,
    graph_name,
    kernel_name,
    input_num,
    output_num,
    temp_build_dir,
    impl_mode,
    use_list_tensor_desc,
    enable_parallel_compile,
    tiling_key=-1,
    kernel_type="KERNEL_TYPE_AIV_ONLY",
    is_cube=False,
    is_conv=False,
):
    graph_name = camel_to_snake(graph_name)
    args_list = args[0]
    if use_list_tensor_desc:
        inputs = args_list[:input_num]
        outputs = args_list[input_num : input_num + output_num]
        args_list = [inputs, outputs]
        input_num = 1
        output_num = 1
        param_type = "dynamic"
    else:
        param_type = "required"

    _origin_inputs_, _origin_outputs_, _inputs_, _outputs_ = _build_args(
        args_list, input_num=input_num, output_num=output_num
    )
    _options_ = _build_options(temp_build_dir, impl_mode, is_cube, is_conv)
    if enable_parallel_compile:
        _options_ += ["--cce-long-call=true"]
        os.environ["ASCENDC_PAR_COMPILE_JOB"] = "1"
        # 防止compile_op组件编译时make信息打屏
        if "MAKEFLAGS" not in os.environ:
            os.environ["MAKEFLAGS"] = "-s"

    origin_func_name = graph_name
    ascendc_src_file = graph_name + "_op_kernel.cpp"
    src = os.path.join(temp_build_dir, ascendc_src_file)
    msg = "start compile Acend C Operator Ascbc, kernel name is " + kernel_name
    CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
    code_channel = get_code_channel(src, kernel_name, "AscBc", _options_)

    msg = f"op info inputs num:{len(_inputs_)}, origin inputs num:{len(_origin_inputs_)}, "
    msg += f"output num:{len(_outputs_)}, origin outputs num:{len(_origin_outputs_)}"
    CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
    op_info = OpInfo(
        kernel_name=kernel_name,
        op_type="AscBc",
        inputs=_inputs_,
        outputs=_outputs_,
        attrs=[],
        impl_mode=impl_mode,
        origin_inputs=_origin_inputs_,
        origin_outputs=_origin_outputs_,
        param_type_dynamic=use_list_tensor_desc,
        mc2_ctx=[],
        param_type_list=[param_type] * (input_num + output_num),
        init_value_list=[],
        output_shape_depend_on_compute=[],
    )
    extend_option = {}
    extend_option["enable_no_tiling_func_compile"] = "enable"

    if tiling_key > -1:
        if is_cube:
            extend_option["customized_tiling_key_list"] = [str(tiling_key)]
            compile_op(
                src,
                origin_func_name,
                op_info,
                _options_,
                code_channel,
                "{}",
                extend_option,
            )
        else:
            compile_op_with_customized_config(
                src,
                origin_func_name,
                op_info,
                _options_,
                code_channel,
                "{}",
                extend_option,
                get_customized_tiling_config(tiling_key, kernel_type),
            )
    else:
        compile_op(
            src, origin_func_name, op_info, _options_, code_channel, "{}", extend_option
        )
    kernel_meta_dir = os.path.join(
        get_current_build_config("kernel_meta_parent_dir"), "kernel_meta"
    )
    _kernel_bin_file_ = os.path.join(kernel_meta_dir, f"{kernel_name}.o")
    _kernel_json_file_ = os.path.join(kernel_meta_dir, f"{kernel_name}.json")
    return _kernel_bin_file_, _kernel_json_file_
