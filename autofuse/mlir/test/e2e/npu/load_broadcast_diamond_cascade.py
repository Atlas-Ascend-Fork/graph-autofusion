#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and is made available under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details and restrictions.
# ----------------------------------------------------------------------------------------------------------------------

import sys

from support.npu_runner import NpuRunner

# One broadcast-fed intermediate result is consumed by two downstream nodes:
# add(broadcast) -> {abs, neg} -> add.  This combines diamond topology with a
# broadcast operand and mixed multi-axis shapes.
INPUT_SHAPE_CASES = (
    ((1, 5), (33, 5), (33, 5), (33, 5)),
    ((8, 1), (8, 128), (8, 128), (8, 128)),
    ((1, 67, 1, 209), (4, 67, 4, 209), (4, 67, 4, 209), (4, 67, 4, 209)),
)


def build_module(torch):
    class BroadcastDiamondCascade(torch.nn.Module):
        def forward(self, source, full, left, right):
            broadcasted = torch.abs(source) + full
            return torch.abs(broadcasted) + torch.neg(broadcasted) * left + right

    return BroadcastDiamondCascade()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_BROADCAST_DIAMOND_CASCADE",
            build_module,
            input_shape_cases=INPUT_SHAPE_CASES,
        )
    )
