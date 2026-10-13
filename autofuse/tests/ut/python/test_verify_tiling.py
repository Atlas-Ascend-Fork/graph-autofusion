# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import os
import types

import pytest

from compile_test_utils import load_compile_module


MODULE_NAME = "commands.verify_tiling"
MODULE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "..",
    "..",
    "tools",
    "att_analyze",
    "src",
    "commands",
    "verify_tiling.py",
)


@pytest.fixture()
def verify_tiling_module():
    core_module = types.ModuleType("core")
    file_utils_module = types.ModuleType("core.file_utils")
    file_utils_module.ensure_output_dir = lambda path: None
    core_module.file_utils = file_utils_module
    with load_compile_module(
        MODULE_NAME,
        MODULE_PATH,
        extra_modules={"core": core_module, "core.file_utils": file_utils_module},
    ) as loaded_module:
        yield loaded_module


def test_execute_tiling_uses_uint64_workspace(verify_tiling_module, monkeypatch):
    calls = []

    def fake_tiling(*args):
        calls.append(args)
        args[2]._obj.value = 0x1B0000000
        args[3]._obj.value = 7
        return 0

    fake_lib = types.SimpleNamespace(
        GetTilingDataSize=lambda: 16, AutofuseTiling=fake_tiling
    )
    monkeypatch.setattr(verify_tiling_module.ctypes, "CDLL", lambda path: fake_lib)

    result = verify_tiling_module.execute_tiling(
        "fake.so",
        {
            "dynamic_dims": [1024],
            "aiv_num": 48,
            "ub_size": 196608,
            "abi": {"kind": "inductor", "shape_dims": 1, "block_dim_width": 64},
        },
        "inductor",
    )

    assert result == {"block_dim": 7, "workspace_size": 0x1B0000000}
    assert isinstance(calls[0][2]._obj, ctypes.c_uint64)
    assert fake_lib.AutofuseTiling.argtypes[2] == ctypes.POINTER(ctypes.c_uint64)
