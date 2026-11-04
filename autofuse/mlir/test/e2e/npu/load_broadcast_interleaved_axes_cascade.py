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

# Broadcast axes surround a normal axis ([brc][normal][brc][normal]).  The
# last case also has a non-32B-aligned inner row.  Graph:
# abs -> add(broadcast) -> mul -> relu, so the broadcast result feeds later nodes.
INPUT_SHAPE_CASES = (
    ((1, 8, 1, 8), (4, 8, 4, 8), (4, 8, 4, 8)),
    ((1, 67, 1, 209), (4, 67, 4, 209), (4, 67, 4, 209)),
    ((1, 33, 1, 5), (8, 33, 4, 5), (8, 33, 4, 5)),
)


def build_module(torch):
    class InterleavedBroadcastCascade(torch.nn.Module):
        def forward(self, source, full, bias):
            return torch.relu((torch.abs(source) + full) * bias)

    return InterleavedBroadcastCascade()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_BROADCAST_INTERLEAVED_AXES_CASCADE",
            build_module,
            input_shape_cases=INPUT_SHAPE_CASES,
        )
    )
