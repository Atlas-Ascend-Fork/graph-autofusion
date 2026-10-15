#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

"""Run the SuperKernel compiler with the device compile step stubbed out."""

from unittest import mock

from superkernel import super_kernel


def compile_and_capture(kernel_info, kernel_name):
    """Generate the SuperKernel source and return its compile info.

    The system tests assert on generated source and on the options the compiler
    is invoked with. Turning those options into a device binary requires real
    Ascend objects and a device toolchain, which is out of scope here, so
    ``compile_super_kernel`` is replaced by a stub capturing its arguments.
    """
    with mock.patch.object(super_kernel, "compile_super_kernel") as compile_backend:
        super_kernel.compile(kernel_info, kernel_name)

    if not compile_backend.called:
        raise AssertionError(
            f"SuperKernel compile did not reach the compile backend: {kernel_name}"
        )
    return compile_backend.call_args[0][0]
