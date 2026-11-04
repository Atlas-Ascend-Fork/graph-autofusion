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

# Form-A pitch: the broadcast source is the innermost non-broadcast segment.
# Tail bytes are 20B / 12B / 4B for the three cases, so the destination pitch
# has to be 32B after padding.  Each graph is abs -> add(broadcast) -> mul -> relu.
INPUT_SHAPE_CASES = (
    ((1, 5), (33, 5), (33, 5)),
    ((1, 1, 3), (4, 4, 3), (4, 4, 3)),
    ((1, 1, 209), (8, 8, 209), (8, 8, 209)),
)


def build_module(torch):
    class TailPitchCascade(torch.nn.Module):
        def forward(self, source, full, bias):
            return torch.relu((torch.abs(source) + full) * bias)

    return TailPitchCascade()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_BROADCAST_TAIL_PITCH_CASCADE",
            build_module,
            input_shape_cases=INPUT_SHAPE_CASES,
        )
    )
