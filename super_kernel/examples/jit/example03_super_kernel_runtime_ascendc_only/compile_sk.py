#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------
# fmt: off

import os
import json
from pathlib import Path
from typing import List
import ctypes
import dataclasses
from contextlib import nullcontext

from superkernel import super_kernel
from utils import SkCompileContext

# TODO: AscendC parts to be refactored, which should be decoupled from tbe
from asc_op_compile_base.common.context.op_context import OpContext
from asc_op_compile_base.common.context import op_info
from asc_op_compile_base.common.context import get_context
from asc_op_compile_base.common.buildcfg.buildcfg_mapping import kernel_meta_parent_dir, \
    op_debug_config, tbe_debug_level
from asc_op_compile_base.common.buildcfg.buildcfg import build_config
from asc_op_compile_base.common.ccec import current_build_config
from asc_op_compile_base.common.platform.platform_info import set_current_compile_soc_info
from asc_op_compile_base.asc_op_compiler.global_storage import global_var_storage

MAGIC_MAPPING = {
    "RT_DEV_BINARY_MAGIC_PLAIN": 0xabceed50,
    "RT_DEV_BINARY_MAGIC_PLAIN_AICPU": 0xabceed51,
    "RT_DEV_BINARY_MAGIC_PLAIN_AIVEC": 0xabceed52,
    "RT_DEV_BINARY_MAGIC_ELF": 0x43554245,
    "RT_DEV_BINARY_MAGIC_ELF_AICPU": 0x41415243,
    "RT_DEV_BINARY_MAGIC_ELF_AIVEC": 0x41415246,
    "RT_DEV_BINARY_MAGIC_ELF_AICUBE": 0x41494343
}

DEFAULT_SOC_VERSION = "Ascend910_9391"
SOC_VERSION_ENV_KEYS = ("SUPERKERNEL_COMPILE_SOC_VERSION", "ASCEND_COMPILE_SOC_VERSION", "ASCEND_SOC_VERSION")


@dataclasses.dataclass
class BinData:
    bin_data: ctypes.c_void_p
    bin_size: int


class KernelResult:
    def __init__(self, path, name):
        self.root = path
        self.name = name
        self._json_data = self._load_json()
        self._bin = self._load_bin()



    @property
    def bin_data(self):
        return self._bin.bin_data

    @property
    def bin_size(self):
        return self._bin.bin_size

    @staticmethod
    def _get_magic_number(magic_str: str):
        return MAGIC_MAPPING.get(magic_str)

    def json_path(self):
        return self.root / "kernel_meta" / (self.name + ".json")

    def bin_path(self):
        return self.root / "kernel_meta" / (self.name + ".o")

    def block_dim(self):
        return self._json_data["blockDim"]

    def magic_number(self):
        return self._get_magic_number(self._json_data["magic"])

    def op_para_size(self):
        return self._json_data["opParaSize"]

    def kernel_name(self):
        return self._json_data["kernelName"]

    def bin_file_name(self):
        return self._json_data["binFileName"]

    def _load_json(self):
        try:
            # Check whether the file exists.
            if not os.path.exists(self.json_path()):
                raise FileNotFoundError(f"File does not exist: {self.json_path()}")
            # Read and parse the JSON file.
            with open(self.json_path(), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            return json_data
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON parsing error: {e}")
        except Exception as e:
            raise Exception(f"An error occurred while reading the file: {e}")

    def _load_bin(self):
        file_path = self.root / "kernel_meta" / (self.name + ".o")
        with open(file_path, 'rb') as f:
            data = f.read()

        # Get the file size
        file_size = len(data)

        # Create the ctypes buffer
        buffer = (ctypes.c_ubyte * file_size)()

        # Copy the data into the buffer
        ctypes.memmove(buffer, data, file_size)
        c_pointer = ctypes.cast(buffer, ctypes.c_void_p)

        return BinData(c_pointer, file_size)


class SubkernelResult(KernelResult):
    def __init__(self, path, name):
        super().__init__(path, name)
        self._input_addr: List[ctypes.c_void_p] = []
        self._output_addr: List[ctypes.c_void_p] = []
        self._workspace_addr: List[ctypes.c_void_p] = []
        self._input: List[str] = []
        self._output: List[str] = []

    @property
    def input(self) -> List[str]:
        return self._input

    @property
    def output(self) -> List[str]:
        return self._output

    @property
    def workspaces_addr(self) -> List[ctypes.c_void_p]:
        """I'm the 'workspaces_addr' property."""
        return self._workspace_addr

    @workspaces_addr.setter
    def workspaces_addr(self, value: List[ctypes.c_void_p]):
        self._workspace_addr = value

    @property
    def input_addr(self) -> List[ctypes.c_void_p]:
        """I'm the 'input_addr' property."""
        return self._input_addr

    @input_addr.setter
    def input_addr(self, value: List[ctypes.c_void_p]):
        self._input_addr = value

    @property
    def output_addr(self) -> List[ctypes.c_void_p]:
        """I'm the 'output_addr' property."""
        return self._output_addr

    @output_addr.setter
    def output_addr(self, value: List[ctypes.c_void_p]):
        self._output_addr = value

    def workspaces_size(self) -> List[int]:
            return self._json_data["workspace"]["size"]

    # Set the input and output names through set_input and set_output so that the reuse relation
    # can be obtained for SuperKernel. For example, for A->B the output of A is the input of B,
    # so the output name of A must be set to the input name of B.
    # For workspace reuse the input and output names are meaningless: the workspaces of different
    # operators on the same stream can be reused by fitting a smaller workspace into a larger one.
    def set_input(self, input_name: List[str]):
        self._input = input_name

    def set_output(self, output_name: List[str]):
        self._output = output_name


class SuperkernelResult(KernelResult):
    def __init__(self, path, name):
        super().__init__(path, name)
        self._output = List[str]

    @property
    def output(self) -> List[str]:
        return self._output

    @output.setter
    def output(self, value: List[str]):
        self._output = value


def get_compile_soc_version():
    for env_key in SOC_VERSION_ENV_KEYS:
        soc_version = os.getenv(env_key)
        if soc_version:
            return soc_version

    try:
        ascendcl = ctypes.CDLL("libascendcl.so")
        ascendcl.aclrtGetSocName.argtypes = []
        ascendcl.aclrtGetSocName.restype = ctypes.c_char_p
        soc_name = ascendcl.aclrtGetSocName()
        if soc_name:
            return soc_name.decode("utf-8")
    except (AttributeError, OSError, UnicodeDecodeError):
        pass

    return DEFAULT_SOC_VERSION


def set_example_compile_soc_info():
    set_current_compile_soc_info(get_compile_soc_version())


def _compile_sub_kernel(kernel_meta_dir, op_name, op_type, func, extend_op_info: dict = None):
    current_build_config()[kernel_meta_parent_dir] = kernel_meta_dir
    current_build_config()[tbe_debug_level] = 0
    set_example_compile_soc_info()

    # compile_op resets global_var_storage at the beginning, so the following configuration has no effect:
    # global_var_storage.set_variable("ascendc_compile_debug_config", True)
    # Only this configuration takes effect
    current_build_config()[op_debug_config] = ["dump_cce", ]

    # enable_deterministic_mode must be configured; otherwise, when the C++ tiling function is
    # called, deterministic in extra_params_c is set to null and the C++ side core dumps
    current_build_config()['enable_deterministic_mode'] = 0

    current_build_config()[kernel_meta_parent_dir] = kernel_meta_dir

    current_build_config()['enable_super_kernel'] = 1
    sp_info = {}
    sp_info['super_kernel_sub_loc'] = 'middle'
    sp_info['super_kernel_options'] = 'early-start=0'
    sp_info['super_kernel_count'] = 0
    sp_info['super_kernel_sub_id'] = 0
    if extend_op_info:
        sp_info.update(extend_op_info)

    with OpContext('static'):
        opinfo = op_info.OpInfo(op_name, op_type)
        get_context().set_graph_op_info(opinfo)
        get_context().add_addition('super_kernel_sub_info', sp_info)

        func()


def compile_subkernel(ctx: SkCompileContext):
    def make_subkernel(
            impl_module_name,  # Implementation module name
            func_name,  # Function name
            op_name,  # Operator name
            op_type,  # Operator type
            input_count=1,  # Number of input parameters
            output_count=1,  # Number of output parameters
            extend_op_info=None  # Extended configuration
    ):
        with nullcontext(ctx.tmp_dir) as tmp_dir:
            # 1. Define the kernel metadata directory
            kernel_meta_dir = Path(tmp_dir) / f"subkernel_{op_name}"

            # 2. Dynamically import the implementation module and function
            module = __import__(f"impl.ops_math.dynamic.{impl_module_name}", fromlist=[func_name])
            func = getattr(module, func_name)

            # 3. Dynamically create the input and output parameters
            tensor_template = {
                "shape": [256],
                "ori_shape": [256],
                "format": "ND",
                "ori_format": "ND",
                "dtype": "float32"
            }

            # Create the input tensor list
            inputs = [tensor_template.copy() for _ in range(input_count)]
            # Create the output tensor list
            outputs = [tensor_template.copy() for _ in range(output_count)]

            # 4. Compile the sub-kernel
            with build_config():
                _compile_sub_kernel(
                    str(kernel_meta_dir),
                    op_name,
                    op_type,
                    extend_op_info=extend_op_info,
                    func=lambda: func(*inputs, *outputs)  # Dynamically call the function
                )

            # 5. Return the path management object
            return SubkernelResult(kernel_meta_dir, impl_module_name)

    # Use the unified make_subkernel function to create operators with different configurations
    is_inf_op = make_subkernel(
        impl_module_name="is_inf",
        func_name="is_inf",
        op_name="IsInf_Default_1",
        op_type="IsInf",
        input_count=1,
        output_count=1
    )

    pows_op = make_subkernel(
        impl_module_name="pows",
        func_name="pows",
        op_name="Pows_Default_2",
        op_type="Pows",
        input_count=2,
        output_count=1
    )

    return [is_inf_op, pows_op]


def compile_superkernel(ctx: SkCompileContext, sub_kernels: list[KernelResult]):
    with nullcontext(ctx.tmp_dir) as tmp_dir:
        kernel_meta_dir = tmp_dir / "superkernel_1"

        global_var_storage.set_variable("ascendc_compile_debug_config", True)
        set_example_compile_soc_info()

        current_build_config()[kernel_meta_parent_dir] = str(kernel_meta_dir)
        current_build_config()[op_debug_config] = ["dump_cce"]
        current_build_config()[tbe_debug_level] = 0

        compile_options = "compile-options=-g:"
        kernel_info = {
            "super_kernel_options": compile_options,
            "op_list": [
                {
                    "stream_id": 1,
                    "bin_path": str(sub_kernels[0].bin_path()),
                    "json_path": str(sub_kernels[0].json_path()),
                },
                {
                    "stream_id": 1,
                    "bin_path": str(sub_kernels[1].bin_path()),
                    "json_path": str(sub_kernels[1].json_path()),
                },
            ],
        }
        kernel_name = "te_superkernel_1"
        with OpContext('super_kernel'):
            super_kernel.compile(kernel_info, kernel_name)

        return SuperkernelResult(kernel_meta_dir, kernel_name)
# fmt: on
