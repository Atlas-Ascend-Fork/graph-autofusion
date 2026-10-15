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
import sys
import ctypes
from utils import print_float_array_ptr, write_data_to_host_memory, \
    allocat_memory_with_reuse, free_all_memorys, SkCompileContext, assert_true
from compile_sk import compile_superkernel, compile_subkernel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from third_party.acl.acl_wrapper import acl

DEVICE_ID = 0
"""
ScopeScope:
    in1 = input1()
    in2 = input2()
    out1 = pow(in1, 2)
    out2 = isinf(out1)
"""

if __name__ == '__main__':
    # 1. Initialize ACL
    acl.acl_init()
    acl.aclrt_set_device(DEVICE_ID)

    stream = ctypes.c_void_p()
    acl.aclrt_create_stream(ctypes.byref(stream))

    # Compile
    # 2.0 Compile the sub-kernel
    with SkCompileContext(need_clean=True) as ctx:
        sub_kernels = compile_subkernel(ctx)
        # 2.1 Set the input/output memory reuse relation
        assert_true(len(sub_kernels) == 2, "sub_kernels should contain 2 subkernels")
        sub_kernels[0].set_input(["arg_in1", "arg_in2"])
        sub_kernels[0].set_output(["out1"])
        sub_kernels[1].set_input(["out1"])
        sub_kernels[1].set_output(["out2"])
        # 2.2 Compile the SuperKernel
        super_kernel_result = compile_superkernel(ctx, sub_kernels)
        # 2.3 Set the SuperKernel output to the output of subkernel[1]
        super_kernel_result.output = sub_kernels[1].output

    # 3. Prepare the input data and allocate the input/output memory of the sub-kernels
    size = 1024 * 4
    host_ptr = ctypes.c_void_p()
    acl.aclrt_malloc_host(ctypes.byref(host_ptr), size)
    float_array_ptr = write_data_to_host_memory(host_ptr, size)

    for sub_kernel in sub_kernels:
        for inp in sub_kernel.input:
            addr = allocat_memory_with_reuse(size, inp)
            if inp.startswith("arg_"):  # Input parameters must be copied from host to device
                acl.aclrt_memcpy_async(addr, size, host_ptr, size, acl.aclrt_memcpy_kind.ACL_MEMCPY_HOST_TO_DEVICE, stream)
            sub_kernel.input_addr.append(addr)

        for output in sub_kernel.output:
            addr = allocat_memory_with_reuse(size, output)
            sub_kernel.output_addr.append(addr)

        for ws_size in sub_kernel.workspaces_size():
            dev_ptr = ctypes.c_void_p()
            acl.aclrt_malloc_align32(ctypes.byref(dev_ptr), ws_size,
                                     acl.aclrt_mem_malloc_policy.ACL_MEM_MALLOC_HUGE_FIRST)
            sub_kernel.workspaces_addr.append(dev_ptr.value)

    # 4. Load the operator
    magic_value = super_kernel_result.magic_number()
    dev_op_binary = acl.aclrt_dev_binary(magic_value, 1, super_kernel_result.bin_data, super_kernel_result.bin_size)

    # Create the handle pointer (double pointer)
    hdl = ctypes.c_void_p()  # First-level pointer
    hdl_ptr = ctypes.pointer(hdl)  # Second-level pointer
    acl.aclrt_dev_binary_register(ctypes.byref(dev_op_binary), hdl_ptr)

    stub_func = super_kernel_result.bin_file_name()
    stub_func_buf = ctypes.create_string_buffer(stub_func.encode('utf-8'))
    stub_func_ptr = ctypes.cast(stub_func_buf, ctypes.POINTER(ctypes.c_void_p))

    stub_name = super_kernel_result.bin_file_name()
    stub_name_buf = ctypes.create_string_buffer(stub_name.encode('utf-8'))
    stub_name_ptr = ctypes.cast(stub_name_buf, ctypes.POINTER(ctypes.c_char))

    kernel_info_ext = super_kernel_result.kernel_name()
    kernel_info_ext_buf = ctypes.create_string_buffer(kernel_info_ext.encode('utf-8'))
    kernel_info_ext_ptr = ctypes.cast(kernel_info_ext_buf, ctypes.POINTER(ctypes.c_char))

    acl.aclrt_function_register(hdl, stub_func_buf, stub_name_ptr, kernel_info_ext_ptr,
                                acl.aclrt_func_mode_type.FUNC_MODE_NORMAL)

    stub_func = ctypes.c_void_p()  # First-level pointer
    stub_func_ptr = ctypes.pointer(stub_func)  # Second-level pointer
    acl.aclrt_get_function_by_name(stub_name_ptr, stub_func_ptr)

    # 5. Launch the operator
    block_dim = super_kernel_result.block_dim()

    c2c_ctrl_addr = ctypes.c_uint64()  # Maps to uint64_t*
    c2c_ctrl_len = ctypes.c_uint32()  # Maps to uint32_t*
    acl.aclrt_get_c2c_ctrl_addr(ctypes.byref(c2c_ctrl_addr), ctypes.byref(c2c_ctrl_len))

    # FFTS placeholder
    void_args = ctypes.c_void_p(c2c_ctrl_addr.value)
    args_list = [void_args]
    for sub_kernel in sub_kernels:
        args_list.extend(sub_kernel.input_addr)
        args_list.extend(sub_kernel.output_addr)
        args_list.extend(sub_kernel.workspaces_addr)
    args_array = (ctypes.c_void_p * len(args_list))(*args_list)

    args = ctypes.cast(args_array, ctypes.c_void_p)
    sm_desc = acl.aclrt_sm_desc()

    acl.aclrt_kernel_launch(stub_func, block_dim, args, ctypes.sizeof(args_array), ctypes.byref(sm_desc), stream)

    # 6. Get the operator result
    acl.aclrt_synchronize_stream(stream)

    host_out_ptr = ctypes.c_void_p()
    acl.aclrt_malloc_host(ctypes.byref(host_out_ptr), size)

    # Get the SuperKernel output pointer
    sk_output = super_kernel_result.output
    assert_true(len(sk_output) == 1, f"superkernel should have 1 output, but got {len(sk_output)}")
    device_ptr_out = allocat_memory_with_reuse(size, sk_output[0])

    acl.aclrt_memcpy(host_out_ptr, size, device_ptr_out, size, acl.aclrt_memcpy_kind.ACL_MEMCPY_DEVICE_TO_HOST)

    print_float_array_ptr(ctypes.cast(host_out_ptr, ctypes.POINTER(ctypes.c_float * (size // 4))))

    # 7. Release resources
    free_all_memorys()
    acl.aclrt_free(host_out_ptr)
    acl.aclrt_free(host_ptr)
    acl.aclrt_dev_binary_unregister(hdl)
    acl.aclrt_destroy_stream(stream)
    acl.aclrt_reset_device(DEVICE_ID)
    acl.acl_finalize()
    print("execute sample success")
# fmt: on
