#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and is made available under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details and restrictions.
# ----------------------------------------------------------------------------------------------------------------------

import sys

import torch

from support.npu_runner import NpuRunner

# expand_0006 / expand_0009 map to 60 / 120 half elements per contiguous tensor
# (leading dims are 1, so the whole tensor is one 120B / 240B contiguous run).
# Such non-32B-aligned runs expose truncation by an unpadded UB-to-GM DataCopy;
# the third case (64 elements = 128B) is the aligned control. Random signed
# inputs also exercise Abs.
INPUT_SHAPE_CASES = (
    ((1, 3, 4, 5),),
    ((1, 3, 4, 5, 2),),
    ((1, 4, 4, 4),),
)


def build_module(torch):
    class ExpandAbs(torch.nn.Module):
        def forward(self, source):
            return torch.abs(source.expand(2, *source.shape[1:]))

    return ExpandAbs()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_EXPAND_ABS_FP16",
            build_module,
            input_shape_cases=INPUT_SHAPE_CASES,
            dtype=torch.float16,
        )
    )
