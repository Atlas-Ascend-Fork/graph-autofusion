#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""全轴归约（all-reduce）的现状探针。

遗留清单 A3 记的是「`A=1` 全归约在 LoopGenerate 报 tensor SSA value crosses sibling/outer
schedule scopes」。那条结论是在计划选择还会整体失败的时候记的，现在候选集变了、下降路径也变了，
需要重新实测而不是沿用。

逐例隔离、fd 2 捕获与判级都走 `matrix_report.run_matrix`，和三张矩阵表同一套词汇：
`REFUSED` 是已记录的支持边界（A3 那条就在这一级），只有 `MISMATCH` / `ERROR` 让退出码非零。
这与本脚本早先「任何一格不过就红」的规则不同——那条规则下 A3 边界每次跑都是红的，
红就不再是信号；现在它由 `cases.py` 的预期等级钉住，变好变坏都会被记账。
"""

import sys

from support.matrix_report import run_matrix
from support.npu_runner import NpuRunner

# 秩 1/2/3 各两档，keepdim 两种。含对齐与非对齐。
CASES = (
    ((1024,), False),
    ((999,), True),
    ((16, 32), False),
    ((37, 257), True),
    ((8, 16, 32), False),
    ((7, 33, 65), True),
)


def make_module(torch, rank, keepdim):
    # `torch.sum(x)` 不收 keepdim，全轴归约要把轴显式列全。两种写法在 inductor 侧降下来的
    # 是同一个 `aten.sum.dim_IntList`，但只有显式列轴这条能表达 keepdim。
    dims = tuple(range(rank))

    class ReduceAll(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=dims, keepdim=keepdim)

    return ReduceAll()


def main():
    runner = NpuRunner.from_args(NpuRunner.parse_args("all reduce probe")).setup()

    def make(name, shape, keepdim):
        def invoke():
            runner.run_case(
                name,
                make_module(runner.torch, len(shape), keepdim),
                input_shapes=(shape,),
                rtol=1e-4,
                atol=1e-4,
            )

        return invoke

    cells = [
        (
            f"REDUCE_ALL_{index}",
            f"rank{len(shape)}",
            make(f"REDUCE_ALL_{index}", shape, keepdim),
            (shape,),
            f"keepdim={keepdim}",
        )
        for index, (shape, keepdim) in enumerate(CASES)
    ]
    return run_matrix("REDUCE_ALL", cells)


if __name__ == "__main__":
    sys.exit(main())
