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

# 同一输入同时喂矢量计算（x + 1）与直连搬运（x.clone()）。load 的消费者既有矢量算子
# 也有 store，同步走「队列 + 显式事件对」的混合路径（V 侧队列 + MTE3 侧
# AllocEventID/SetFlag/WaitFlag/ReleaseEventID）。数值基准：第一路 = x + 1，第二路 = x。

import sys

from support.npu_runner import NpuRunner

INPUT_SHAPE_CASES = (
    ((128, 23, 5),),
    ((128, 128, 128),),
    ((256, 23, 5),),
)


def build_module(torch):
    class MixedConsumption(torch.nn.Module):
        def forward(self, x):
            return x + 1, x.clone()

    return MixedConsumption()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "MIXED_CONSUMPTION", build_module, input_shape_cases=INPUT_SHAPE_CASES
        )
    )
