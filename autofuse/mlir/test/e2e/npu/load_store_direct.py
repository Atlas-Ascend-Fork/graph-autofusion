#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

# load 直连 store 的纯搬运（MTE2 -> MTE3），物化为双端 TQueBind 队列。
# 数值基准是恒等映射（输出逐元素等于输入）。

import sys

from support.npu_runner import NpuRunner

INPUT_SHAPE_CASES = (
    ((128, 23, 5),),
    ((128, 128, 128),),
    ((256, 23, 5),),
)


def build_module(torch):
    # 使用 clone 保留 Load 直连 Store 的搬运 kernel；直接恒等映射会被 Inductor 消除。
    class Copy(torch.nn.Module):
        def forward(self, x):
            return x.clone()

    return Copy()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_STORE_DIRECT", build_module, input_shape_cases=INPUT_SHAPE_CASES
        )
    )
