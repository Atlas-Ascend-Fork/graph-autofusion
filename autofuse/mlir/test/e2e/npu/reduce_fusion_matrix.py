#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""reduce 与逐元素节点**融合在同一张图**里时的形态矩阵。

基本门槛保留 12 格，不展开拓扑与形状的全组合：

- 10 种形态各跑 `(16, 37)`：归约前融合、归约后一元/二元消费者、前后融合、
  fork 多输出、同图双归约；同时覆盖非对齐输入。
- `fork_mul` 和 `multi_two_sums` 额外跑 `(8192, 128)`，验证较大输入下的分支多输出和双归约正确性。
  实际多核、跨轮或跨核归约行为由调度决定，这里不对 tiling 作断言。

CPU 的 `load_sum_mul_store` / `load_sum_relu_store` / `load_sum0_mul_store` 已通过
AscGraphBuilder 和真实优化调度覆盖部分融合形态。本用例补充 PyTorch / Inductor 到
真机的集成验证，重点保留 fork 与双归约组合；普通归约模板由 reduce_template_matrix 覆盖。
归约后的一元消费者可能留在同段，带额外输入的消费者可能触发切图与 workspace 传递。
本用例验证数值和回退诊断，不断言固定切图段数。keepdim 一律 False，不扩展广播组合。

归约一律 1e-4 容差——加法次序两侧不同，零容差会在相消的格子上假报缺陷。

用法：

    python3 autofuse/mlir/test/e2e/npu/reduce_fusion_matrix.py --steps 0 --work-dir /tmp/rf
    python3 autofuse/mlir/test/e2e/npu/reduce_fusion_matrix.py --form fork --steps 0
"""

import argparse
import os
import sys

from support.matrix_report import run_matrix
from support.npu_runner import NpuRunner

SMALL_TAIL = (16, 37)
LARGE = (8192, 128)


def build_prod_mul_sum(torch):
    """sum(x * y, -1)：逐元素喂归约，末维归约。"""

    class ProdMulSum(torch.nn.Module):
        def forward(self, x, y):
            return torch.sum(torch.mul(x, y), dim=-1)

    return ProdMulSum()


def build_prod_mul_sum0(torch):
    """sum(x * y, 0)：逐元素喂归约，首维归约（RA 形态）。"""

    class ProdMulSum0(torch.nn.Module):
        def forward(self, x, y):
            return torch.sum(torch.mul(x, y), dim=0)

    return ProdMulSum0()


def build_cons_sum_relu(torch):
    """relu(sum(x, -1))：归约喂一元消费者。"""

    class ConsSumRelu(torch.nn.Module):
        def forward(self, x):
            return torch.relu(torch.sum(x, dim=-1))

    return ConsSumRelu()


def build_cons_sum_mul(torch):
    """sum(x, -1) * y：归约喂二元消费者，第二个输入是降维后的形状。"""

    class ConsSumMul(torch.nn.Module):
        def forward(self, x, y):
            return torch.mul(torch.sum(x, dim=-1), y)

    return ConsSumMul()


def build_cons_sum0_mul(torch):
    """sum(x, 0) * y：首维归约后接消费者。CPU 侧 load_sum0_mul_store 的对应形态。"""

    class ConsSum0Mul(torch.nn.Module):
        def forward(self, x, y):
            return torch.mul(torch.sum(x, dim=0), y)

    return ConsSum0Mul()


def build_both_mul_sum_relu(torch):
    """relu(sum(x * y, -1))：归约前后都有邻居。"""

    class BothMulSumRelu(torch.nn.Module):
        def forward(self, x, y):
            return torch.relu(torch.sum(torch.mul(x, y), dim=-1))

    return BothMulSumRelu()


def build_fork_mul(torch):
    """t = x * y; return sum(t, -1), t —— 一个节点同时喂归约侧与输出侧。

    这是 InvalidWorkspace「推不出形状」那条的图形状，两个输出形状不同（降维 / 全量）。
    """

    class ForkMul(torch.nn.Module):
        def forward(self, x, y):
            value = torch.mul(x, y)
            return torch.sum(value, dim=-1), value

    return ForkMul()


def build_fork_relu(torch):
    """t = relu(x); return sum(t, -1), t —— 同上但 fork 点是一元节点。"""

    class ForkRelu(torch.nn.Module):
        def forward(self, x):
            value = torch.relu(x)
            return torch.sum(value, dim=-1), value

    return ForkRelu()


def build_multi_two_sums(torch):
    """sum(x, -1) + sum(x * x, -1)：一张图两个独立归约。"""

    class MultiTwoSums(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=-1) + torch.sum(torch.mul(x, x), dim=-1)

    return MultiTwoSums()


def build_multi_sum_max(torch):
    """sum(x, -1) 与 max(x, -1) 两种归约共存，验的是不同归约 op 在同图里各自切段。"""

    class MultiSumMax(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=-1) + torch.amax(x, dim=-1)

    return MultiSumMax()


# (格子名, 形态族, 构图函数, 输入 shape 生成器, 容差)
# shape 生成器收一个基础 shape，返回输入 shape 元组——因为 cons_* 的第二个输入是
# 降维后的形状，不能直接用基础 shape。
FORMS = (
    ("prod_mul_sum", "prod", build_prod_mul_sum, lambda s: (s, s), 1e-4),
    ("prod_mul_sum0", "prod", build_prod_mul_sum0, lambda s: (s, s), 1e-4),
    ("cons_sum_relu", "cons", build_cons_sum_relu, lambda s: (s,), 1e-4),
    ("cons_sum_mul", "cons", build_cons_sum_mul, lambda s: (s, (s[0],)), 1e-4),
    ("cons_sum0_mul", "cons", build_cons_sum0_mul, lambda s: (s, (s[1],)), 1e-4),
    ("both_mul_sum_relu", "cons", build_both_mul_sum_relu, lambda s: (s, s), 1e-4),
    ("fork_mul", "fork", build_fork_mul, lambda s: (s, s), 1e-4),
    ("fork_relu", "fork", build_fork_relu, lambda s: (s,), 1e-4),
    ("multi_two_sums", "multi", build_multi_two_sums, lambda s: (s,), 1e-4),
    ("multi_sum_max", "multi", build_multi_sum_max, lambda s: (s,), 1e-4),
)

SHAPES = (
    ("tail", SMALL_TAIL),
    ("large", LARGE),
)

# 仅这两种复杂拓扑保留大规模执行门槛。
LARGE_FORMS = {"fork_mul", "multi_two_sums"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="reduce-with-neighbours fusion matrix on 910B"
    )
    NpuRunner.add_common_arguments(parser)
    parser.add_argument(
        "--form",
        default=os.environ.get("AF_E2E_RF_FORM", "prod,cons,fork,multi"),
        help="逗号分隔的形态族：prod / cons / fork / multi。",
    )
    parser.add_argument(
        "--shapes",
        default=os.environ.get("AF_E2E_RF_SHAPES", "tail,large"),
        help="逗号分隔：tail / large；large 仅含 fork_mul 和 multi_two_sums。",
    )
    parser.add_argument(
        "--only",
        default=os.environ.get("AF_E2E_RF_ONLY", ""),
        help="逗号分隔的格子名，只跑这些格子。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    forms = {item.strip() for item in args.form.split(",") if item.strip()}
    shape_names = {item.strip() for item in args.shapes.split(",") if item.strip()}
    only = {item.strip() for item in args.only.split(",") if item.strip()}

    grid = []
    for shape_name, shape in SHAPES:
        if shape_name not in shape_names:
            continue
        for name, form, factory, shaper, tol in FORMS:
            if shape_name == "large" and name not in LARGE_FORMS:
                continue
            if form not in forms:
                continue
            cell = f"{name}_{shape_name}"
            if only and cell not in only:
                continue
            grid.append((cell, form, factory, shaper(shape), tol))

    if not grid:
        print("REDUCE_FUSION 没有选中任何格子", file=sys.stderr)
        return 2

    runner = NpuRunner.from_args(args).setup()

    def make(cell_name, factory, shapes, tol):
        def invoke():
            runner.run_case(
                f"REDUCE_FUSION_{cell_name}",
                factory(runner.torch),
                input_shapes=shapes,
                rtol=tol,
                atol=tol,
            )

        return invoke

    cells = [
        (cell, form, make(cell, factory, shapes, tol), shapes)
        for cell, form, factory, shapes, tol in grid
    ]

    return run_matrix(
        "REDUCE_FUSION",
        cells,
        plan_note=f"forms={sorted(forms)} shapes={sorted(shape_names)}",
    )


if __name__ == "__main__":
    sys.exit(main())
