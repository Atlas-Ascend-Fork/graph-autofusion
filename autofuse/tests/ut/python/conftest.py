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

import os
import types

import pytest

from compile_test_utils import (
    PYTHON_DIR,
    load_compile_module,
    make_asc_log_stub_modules,
)

MODULE_NAME = "autofuse.compile_adapter"
MODULE_PATH = os.path.join(PYTHON_DIR, "compile_adapter.py")


@pytest.fixture()
def compile_adapter_module():
    ascendc_compile_module = types.ModuleType("autofuse.ascendc_compile")

    class CompileError(Exception):
        pass

    def _noop_main(args):
        return None

    ascendc_compile_module.CompileError = CompileError
    ascendc_compile_module.main = _noop_main
    with load_compile_module(
        MODULE_NAME,
        MODULE_PATH,
        extra_autofuse_attrs={"ascendc_compile": ascendc_compile_module},
        extra_modules={
            "autofuse.ascendc_compile": ascendc_compile_module,
            # Stub asc log so the fixture never triggers the real (slow, and
            # process-global-state polluting) ASC package init.
            **make_asc_log_stub_modules(),
        },
    ) as loaded_module:
        yield loaded_module
