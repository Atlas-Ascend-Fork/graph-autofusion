#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

import sys

from support.npu_runner import NpuRunner

# 纯 permute+contiguous 会被 torch 2.10 inductor 降级为 extern aten.clone（不生成
# kernel，autofuse 无从介入）。转置作为计算 kernel 的输入前缀才是 autofuse
# load_transpose 的目标形态，故用 permute + add 复合图。
INPUT_SHAPE_CASES = (
    ((2, 37, 23), (2, 23, 37)),
    ((4, 16, 32), (4, 32, 16)),
)


def build_module(torch):
    class TransposeAdd(torch.nn.Module):
        def forward(self, x, y):
            return (x.permute(0, 2, 1) + y).contiguous()

    return TransposeAdd()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_TRANSPOSE_STORE", build_module, input_shape_cases=INPUT_SHAPE_CASES
        )
    )
