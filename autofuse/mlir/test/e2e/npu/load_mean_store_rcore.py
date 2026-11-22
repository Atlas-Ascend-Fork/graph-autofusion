#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""Run `torch.mean` on RCore two-stage shapes through the MLIR NPU E2E path.

**它不覆盖 `afir.mean` 的下降。** 2026-10-15 实测：torch Inductor 在后端看到图之前就把
`aten.mean.dim` 拆成了「求和 + 除常数」，落到 ASCIR 里是 `ascir.ops.Sum` + `ascir.ops.TrueDiv`
（四个 shape 的 `asc_graph.py` 里都**没有** Mean 节点），`TrueDiv` 还是另一个 op 里的
elementwise。所以这条用例实际验的是 Sum 的 RCore 两阶段 + 一次 elementwise 除法。

留着它有两个用处：
  1. 它是上述结论的可复现证据——`asc_graph.py` 里的 op 清单就是判据；
  2. 它是**跳线**：哪天前端不再拆（或换个前端送进真的 Mean 节点），这条用例就会自动开始走
     `afir.mean`，那时它的绿才说明 Mean 的下降在设备上对。

要在设备上验 `afir.mean` 本身，得绕开 torch，用手搭的 ASCIR 图（`ascir.ops.Mean`，与 CPU E2E
的 `AscGraphBuilder.Reduce<ascir_op::Mean>` 同一个节点）走
`device_validation_runner`；见 `docs/design/2026-10-14-reduce-mean-correction-design.md`
的「真机验收：前端不产出 Mean 节点」。
"""

import sys

from support.npu_runner import NpuRunner


# 与 `load_sum_store_rcore.py` 同一组 shape，只把算子换成 `mean`。同形是刻意的：
# RCore 的切分、核数、workspace 组织都由 shape 决定，两份对照着跑才能把「Mean 多出来的那
# 一次缩放」从求和那一半里分离出来。
#
# 为什么必须是这组 shape（理由逐字同那一份）：RCore 的目标区间是「A 轴小、R 轴大」，
# phase 1 的 `block_dim = Ceiling(A/a0t) x Ceiling(Ceiling(R/a1t)/a1Tt)`，只有 `a1Tt` 是
# 多核变量。`nA = Ceiling(A/a0t) < corenum` 时核数必须从 R 方向补，跨核归约才真正启用；
# `nA == corenum` 时 R 根本不跨核切，phase 2 只合并一份 partial。前三格属前者，第四格是
# 后者的边界，成对用来区分「管道通」与「合并对」。
#
# Mean 在这条路上特有的两件事——也就是本用例相对 Sum 的全部增量：
#
#   1. 每个核各自除同一个 `N`，`N` 是完整逻辑归约域的元素数（这里就是 R），不是本核
#      那一段的长度。各核分到的段长不等（`Ceiling` 的尾块），按本核段长除就是「均值的均值」，
#      错得没有固定倍数。
#   2. phase 2 只求和（调度侧把第二阶段的 reducer 改写成 `Sum`），所以 `Σ_c S_c/N = S/N`。
#      若第二阶段也除一次，结果会再小 `Rm` 倍——那是跨核才暴露的错，CPU E2E 的 `block_dim`
#      是 1，第二阶段只合并一份 partial，测不到。
#
# A=1（纯全归约）同样不在列：它在 `LoopGenerate` 就报
# `tensor SSA value crosses sibling/outer schedule scopes`，与 Mean 无关，是另一条下降缺口；
# 全轴 Mean 作为后续项单列，见
# `docs/design/2026-10-14-reduce-mean-correction-design.md` 的「后续项：全轴 Mean 的最小完整环」。
INPUT_SHAPE_CASES = (
    ((8, 131072),),
    ((16, 65536),),
    ((20, 131072),),
    ((40, 32768),),
)


def build_module(torch):
    class ReduceMean(torch.nn.Module):
        def forward(self, x):
            return torch.mean(x, dim=-1, keepdim=True)

    return ReduceMean()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_MEAN_STORE_RCORE",
            build_module,
            input_shape_cases=INPUT_SHAPE_CASES,
            # 浮点归约的累加顺序因实现而异，逐位相等不可达；取 float32 的常规容差。
            rtol=1e-5,
            # Mean 的绝对误差比 Sum 小 R 倍：求和那一半的舍入误差约 n*eps*|x|，除以 N 之后
            # 回到 eps*|x| 量级。所以这里不沿用 Sum 那条 1e-2——那会把 100 倍的错分母也放过。
            atol=1e-5,
            description="Run Mean small-A/large-R shapes through the MLIR NPU E2E path.",
        )
    )
