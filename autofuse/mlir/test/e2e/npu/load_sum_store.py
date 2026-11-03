#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""Run the MLIR ReduceSum E2E case on a real Ascend NPU through Inductor."""

import sys

from support.npu_runner import NpuRunner


# Keep these shapes aligned with the CPU Reduce E2E case:
#   16x32: reduction extent is 32B aligned;
#   16x37: reduction extent requires ReduceInit padding;
#   17x32: the kept axis has a tail tile.
INPUT_SHAPE_CASES = (
    ((16, 32),),
    ((16, 37),),
    ((17, 32),),
)


def build_module(torch):
    class ReduceSum(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=-1, keepdim=True)

    return ReduceSum()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_SUM_STORE",
            build_module,
            input_shape_cases=INPUT_SHAPE_CASES,
            description="Run ReduceSum AR shapes through the MLIR NPU E2E path.",
            rtol=1e-5,
            atol=1e-5,
        )
    )
