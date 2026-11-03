#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""尾维 1 消除后 ARA 变成 AR；AllLoad 对应 plan 2，由 cases.py 指定。"""

import sys

from support.npu_runner import NpuRunner

INPUT_SHAPE_CASES = (((4, 15, 1),),)


def build_module(torch):
    class ReduceMin(torch.nn.Module):
        def forward(self, x):
            return torch.amin(x, dim=1, keepdim=True)

    return ReduceMin()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_MIN_STORE_ALLLOAD_TAIL1",
            build_module,
            input_shape_cases=INPUT_SHAPE_CASES,
            description="Verify AR AllLoad after eliminating a trailing unit dimension.",
            rtol=0,
            atol=0,
        )
    )
