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

import ctypes
import os
import sys
from pathlib import Path
from datetime import datetime
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from third_party.acl.acl_wrapper import acl


def print_float_array_ptr(float_array_ptr, count=20):
    """
    Print the data pointed to by a float array pointer.

    Args:
        float_array_ptr: ctypes float array pointer
        count: number of elements to print; print all of them if None
    """
    if float_array_ptr is None:
        print("The pointer is None")
        return

    # Get the array contents
    array_content = float_array_ptr.contents

    # Determine how many elements to print
    if count is None:
        count = len(array_content)
    else:
        count = min(count, len(array_content))

    print(f"Float array (first {count} elements):")
    for i in range(count):
        print(f"  [{i}] = {array_content[i]}")


def write_data_to_host_memory(host_ptr, size):
    float_size = size // 4
    float_array_ptr = ctypes.cast(host_ptr, ctypes.POINTER(ctypes.c_float * float_size))

    if hasattr(float_array_ptr.contents, '__getitem__'):
        for i in range(float_size):
            float_array_ptr.contents[i] = i % 256

    print_float_array_ptr(float_array_ptr)
    return float_array_ptr


GLOBAL_MEMORY = dict()


def allocat_memory_with_reuse(mem_size, mem_reuse_key: str):
    global GLOBAL_MEMORY
    if mem_reuse_key in GLOBAL_MEMORY:
        return GLOBAL_MEMORY[mem_reuse_key]

    mem_ptr = ctypes.c_void_p()
    acl.aclrt_malloc_align32(ctypes.byref(mem_ptr), mem_size, acl.aclrt_mem_malloc_policy.ACL_MEM_MALLOC_HUGE_FIRST)
    GLOBAL_MEMORY[mem_reuse_key] = mem_ptr
    return mem_ptr


def free_all_memorys():
    global GLOBAL_MEMORY
    for mem_reuse_key in GLOBAL_MEMORY:
        acl.aclrt_free(GLOBAL_MEMORY[mem_reuse_key])
    GLOBAL_MEMORY.clear()


class SkCompileContext:
    def __init__(self, need_clean: bool = True):
        example_root = Path(os.path.dirname(os.path.abspath(__file__)))
        base_dir = example_root / "generated"
        base_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        pid = os.getpid()
        tmp_path = base_dir / f"{timestamp}_{pid}"

        if tmp_path.exists():
            raise FileExistsError(
                f"Temporary directory already exists before test run starts: {tmp_path}"
            )

        tmp_path.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = tmp_path
        self.cleanup = need_clean

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)


def assert_true(condition, msg: str = ""):
    if not condition:
        raise RuntimeError(f"condition failed, msg: {msg}")
# fmt: on
