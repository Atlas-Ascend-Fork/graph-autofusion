#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RARR 最小复现：保留轴不在末位的多轴归约。

`(8, 8, 128, 128)` 归约 `(0, 2, 3)`：保留的 A 轴是 axis 1，归约轴里含 axis 0。现有 RCore
用例全是 `(A, R)` 两维、A 在首位，这个轴序没被覆盖过。

逐例隔离、fd 2 捕获与判级走 `matrix_report.run_matrix`，与三张矩阵表同一套词汇。
"""

import sys

from support.matrix_report import run_matrix
from support.npu_runner import NpuRunner

CASES = (
    ((8, 8, 128, 128), (0, 2, 3), True),  # 失败样本
    ((8, 8, 128, 128), (0,), True),  # 只归约首轴
    ((8, 8, 128, 128), (2, 3), True),  # 只归约末两轴，A 在前
    ((8, 8, 128, 128), (0, 1), True),  # 归约前两轴，A 在后
)


def make_module(torch, dims, keepdim):
    class ReduceSum(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=dims, keepdim=keepdim)

    return ReduceSum()


def main():
    runner = NpuRunner.from_args(NpuRunner.parse_args("RARR repro")).setup()

    def make(name, shape, dims, keepdim):
        def invoke():
            runner.run_case(
                name,
                make_module(runner.torch, dims, keepdim),
                input_shapes=(shape,),
                rtol=1e-4,
                atol=1e-4,
            )

        return invoke

    cells = [
        (
            f"RARR_{index}",
            "rarr",
            make(f"RARR_{index}", shape, dims, keepdim),
            (shape,),
            f"dims={dims}",
        )
        for index, (shape, dims, keepdim) in enumerate(CASES)
    ]
    return run_matrix("RARR", cells)


if __name__ == "__main__":
    sys.exit(main())
