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

# Isolate brc -> unary compute without mixing a full-shape queue operand.
# Torch emits aten.expand, which becomes afir.broadcast on the Autofuse path.
INPUT_SHAPE_CASES = (
    ((1, 5),),
    ((8, 1),),
    ((1, 8, 1, 8),),
)


def build_module(torch):
    class BroadcastToCascade(torch.nn.Module):
        def forward(self, source):
            if source.dim() == 2 and source.shape[-1] == 5:
                broadcasted = torch.broadcast_to(source, (33, 5))
            elif source.dim() == 2 and source.shape[-1] == 1:
                broadcasted = torch.broadcast_to(source, (8, 128))
            else:
                broadcasted = torch.broadcast_to(source, (4, 8, 4, 8))
            return torch.relu(torch.abs(broadcasted) * 0.25)

    return BroadcastToCascade()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_BROADCAST_TO_CASCADE",
            build_module,
            input_shape_cases=INPUT_SHAPE_CASES,
        )
    )
