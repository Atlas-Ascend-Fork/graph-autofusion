#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""Run the MLIR ReduceSum E2E case on shapes that actually exercise the RCore two-stage template."""

import sys

from support.npu_runner import NpuRunner


# RCore（两段式 reduce）的目标区间是「A 轴小、R 轴大」：A 轴单独喂不满核，只能靠切 R 轴
# 分核，phase 1 每个核算自己那一段的 partial sum 写进 workspace，phase 2 再跨核合并。
#
# phase 1 的 block_dim = Ceiling(A/a0t) × Ceiling(Ceiling(R/a1t)/a1Tt)，其中只有 a1Tt 是
# 多核变量，A 方向那一项由 UB tile 决定、没有旋钮。所以：
#
#   - nA = Ceiling(A/a0t) > corenum  =>  ATT 的求解器解不出 tiling（`load_sum_store_large`
#     那批大 A 的 shape 就落在这里，那是 RCore 不该被选用的区间，不在本用例范围内）；
#   - nA == corenum                  =>  A 轴单独就喂满核，Ceiling(nR/a1Tt) 恒为 1，
#     R 根本没有跨核切分，phase 2 只合并一份 partial sum——两段式的管道走通了，
#     合并逻辑却一次都没被触发；
#   - nA < corenum                   =>  核数必须从 R 方向补，跨核归约才真正启用。
#
# 最后一档是本用例存在的理由。`load_sum_store.py` 的 16x32 / 17x32 在这里 block_dim 为 1，
# 连多核都没进，对 RCore 零区分度——2026-09-28 那个「phase 1 写 workspace 的地址漏掉 R 块号、
# 所有核互相覆盖」的缺陷就是因此一直没被测到（IR 合法、产出合法，只是算错）。
#
# 前三个 shape 的 nA 是 8 / 16 / 20，都落在 nA < 40 一档；第四个是 nA == corenum 的边界，
# 与前三个成对，用来区分「管道通」和「合并对」。
#
# A=1（纯全归约）暂不在列：它在 `LoopGenerate` 就报
# `tensor SSA value crosses sibling/outer schedule scopes`，是另一条待修的下降缺口。
INPUT_SHAPE_CASES = (
    ((8, 131072),),
    ((16, 65536),),
    ((20, 131072),),
    ((40, 32768),),
)


def build_module(torch):
    class ReduceSum(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=-1, keepdim=True)

    return ReduceSum()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "LOAD_SUM_STORE_RCORE",
            build_module,
            input_shape_cases=INPUT_SHAPE_CASES,
            # 浮点归约的累加顺序因实现而异，逐位相等不可达；取 float32 的常规容差。
            rtol=1e-5,
            # 十万量级 float32 求和的绝对舍入误差约 n*eps*|x|，1e-5 过紧。
            atol=1e-2,
            description="Run ReduceSum small-A/large-R shapes through the MLIR NPU E2E path.",
        )
    )
