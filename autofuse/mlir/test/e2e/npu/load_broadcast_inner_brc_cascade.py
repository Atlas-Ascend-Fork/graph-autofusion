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

# Form-B: the innermost axis is the broadcast axis.  The large case forces a
# multi-tile inner row and distinguishes a correct broadcast result from a
# repeat-one only implementation.  Graph: abs -> add(broadcast) -> mul -> relu.
INPUT_SHAPE_CASES = (
    ((8, 1), (8, 128), (8, 128)),
    ((128, 1), (128, 128), (128, 128)),
)


def build_module(torch):
    class InnerBroadcastCascade(torch.nn.Module):
        def forward(self, source, full, bias):
            return torch.relu((torch.abs(source) + full) * bias)

    return InnerBroadcastCascade()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_BROADCAST_INNER_BRC_CASCADE",
            build_module,
            input_shape_cases=INPUT_SHAPE_CASES,
        )
    )
