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

# Direct probe-T3 equivalent, no unary before the broadcast operand.
INPUT_SHAPE_CASES = (
    ((1, 5), (33, 5)),
    ((8, 1), (8, 128)),
    ((1, 8, 1, 8), (4, 8, 4, 8)),
)


def build_module(torch):
    class BroadcastAddDirect(torch.nn.Module):
        def forward(self, source, full):
            return source + full

    return BroadcastAddDirect()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_BROADCAST_ADD_DIRECT",
            build_module,
            input_shape_cases=INPUT_SHAPE_CASES,
        )
    )
