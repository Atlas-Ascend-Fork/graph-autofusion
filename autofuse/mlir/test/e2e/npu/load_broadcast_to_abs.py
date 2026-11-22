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

# Minimal real-Inductor equivalent of probe T2: brc -> abs -> store.
INPUT_SHAPE_CASES = (
    ((1, 5),),
    ((8, 1),),
    ((1, 8, 1, 8),),
)


def build_module(torch):
    class BroadcastToAbs(torch.nn.Module):
        def forward(self, source):
            if source.dim() == 2 and source.shape[-1] == 5:
                broadcasted = torch.broadcast_to(source, (33, 5))
            elif source.dim() == 2 and source.shape[-1] == 1:
                broadcasted = torch.broadcast_to(source, (8, 128))
            else:
                broadcasted = torch.broadcast_to(source, (4, 8, 4, 8))
            return torch.abs(broadcasted)

    return BroadcastToAbs()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_BROADCAST_TO_ABS", build_module, input_shape_cases=INPUT_SHAPE_CASES
        )
    )
