#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""reduce 的「调度模板 × 归约形态」矩阵在 910B 实机上的数值验收。

`2026-09-20-two-dim-transfer-addressing-design.md` 的验收项要求 **RCommon / RCore / AllLoad
× AR / RA** 六个组合在 910B 上数值正确。这个文件提供其中可达格子的语料；**模板由外部钉**
（`force_schedule_result`），本文件只负责给形态、建图和对数。

为什么模板不在文件里选：plan 下标是候选列表里的位置，而候选是调度器按
「General -> RCore -> AllLoad」顺序追加出来的（`reduce_schedule_case_generator.cpp` 的
`GeneratorTask`），同一个模板在不同图里下标不同（二维图和三维 ARA 图的 AllLoad 都是 2：
当前候选顺序为 General、RCore、AllLoad）。把下标写进文件等于把调度器的候选构成抄一遍，调度器一改就
静默指到别的模板上——用例照样过，测的东西变了。所以下标留给调用方，本文件把它**打印出来**
入账。

形态怎么选（`FORMS`）：

- `ar`  —— 二维末维归约，tile 布局 `[A, R]`。
- `ra`  —— 二维首维归约，tile 布局 `[R, A]`。
- `ara` —— 三维中间维归约。它存在的唯一理由是 **AllLoad 下只有它能产出 RA 形态**：
  二维 RA 根本不生成 AllLoad 候选（`IsLegacyFullLoadReduce` 要 `in[0] == out[0]`，
  `IsTailReduceFullLoad` 要轴数 ≥ 3 且归约末轴，二维 RA 两条都不满足），所以矩阵里
  「AllLoad × RA」这一格只能由 ARA 顶上。详见
  `docs/design/2026-10-01-afir-reduce-allload-template-design.md` 2.2。
- `ara_agap` —— 同为 ARA，但内层 A 取 5（`align(5) = 8`，destination 里 A 槽位跨步），
  且 A_outer 大到一个 tile 装得下多根。配 plan 0（RCommon）跑，验的是跨 tile 累加的 count
  按**槽位跨度**而不是元素数算，见 `docs/design/2026-09-21-afir-reduce-sum-ra-lowering-design.md`
  3.4。`ara` 的 `(4, 32, 8)` 测不到：`align(8) == 8`，跨步与连续恰好等价。
- `rcore_ar` / `rcore_ra` —— RCore 两格。形状必须**大**：RCore 的 block 轴是
  `MergeAxis({A_outer, R_block})`，核数在 nA 喂满时全部来自 A 方向，`Ceiling(nR/a1Tt) = 1`，
  phase 2 只合并一份 partial，**跨核归约根本不触发**。上面那些 `ar` / `ra` 的小形状
  block_dim = 1，对 RCore 零区分度。这里取 nA = 8/16/20 < corenum(40)，核数必须从 R 方向来，
  跨核合并才真的走到。可解条件是 `Ceiling(A/a0t) ≤ corenum`（a0t 是贪心喂饱 a1t 后剩给 A 的
  UB 余量），所以 A 小 R 大正是 RCore 擅长的区间，不是在压它的边界。

二维 `ar` / `ra` 的形状取和 CPU E2E 同一组，便于两侧对账：对齐 / R 尾块 / A 尾块各一条。

用法（`AF_E2E_REDUCE_FORM` 与 `--form` 等价，前者是给不转发额外参数的 runner 用的）。
使用共用 ATT DFX 机制指定运行期候选，例如：
    AUTOFUSE_DFX_FLAGS="--force_schedule_result=0;--force_tiling_case=0"
不再裁剪 codegen 候选集合。编号以最终生成的 ATT schedule/tiling key 为准。
下表中的 N 对应 --force_schedule_result=N：

    AUTOFUSE_DFX_FLAGS="--force_schedule_result=0" AF_E2E_REDUCE_FORM=ar   ...  # RCommon x AR
    AUTOFUSE_DFX_FLAGS="--force_schedule_result=0" AF_E2E_REDUCE_FORM=ra   ...  # RCommon x RA
    AUTOFUSE_DFX_FLAGS="--force_schedule_result=2" AF_E2E_REDUCE_FORM=ar   ...  # AllLoad  x AR
    AUTOFUSE_DFX_FLAGS="--force_schedule_result=2" AF_E2E_REDUCE_FORM=ara  ...  # AllLoad  x RA
    AUTOFUSE_DFX_FLAGS="--force_schedule_result=1" AF_E2E_REDUCE_FORM=rcore_ar ... # RCore    x AR
    AUTOFUSE_DFX_FLAGS="--force_schedule_result=1" AF_E2E_REDUCE_FORM=rcore_ra ... # RCore    x RA
    AUTOFUSE_DFX_FLAGS="--force_schedule_result=0" AF_E2E_REDUCE_FORM=ara_agap ... # 3.4 的槽位跨度回归
"""

import argparse
import math
import os
import sys

from support.matrix_report import run_matrix
from support.npu_runner import NpuRunner
from support.att_selection import selected_plan

# form -> (归约维, ((形状, 说明), ...))
FORMS = {
    "ar": (
        -1,
        (
            ((16, 32), "R 为 32B 对齐"),
            ((16, 37), "R 带尾块，内维要 align 到 40"),
            ((17, 32), "A 带尾块"),
        ),
    ),
    "ra": (
        0,
        (
            ((37, 16), "R 带尾块"),
            ((32, 17), "A 带尾块"),
            ((37, 17), "A / R 都带尾块"),
        ),
    ),
    "ara": (
        1,
        (((4, 32, 8), "A 组两根轴，srcShape 内维是连乘"),),
    ),
    "ara_agap": (
        1,
        (
            ((64, 8, 5), "每 tile 5 根 A_outer，A 槽位跨步"),
            ((64, 32, 5), "每 tile 3 根 A_outer，A 槽位跨步"),
        ),
    ),
    "rcore_ar": (
        -1,
        (
            ((8, 131072), "nA=8 < corenum，跨核归约真的启用"),
            ((16, 65536), "nA=16"),
            ((20, 131072), "nA=20"),
        ),
    ),
    "rcore_ra": (
        0,
        (
            ((131072, 8), "nA=8，与 rcore_ar 镜像"),
            ((65536, 16), "nA=16"),
            ((131072, 20), "nA=20"),
        ),
    ),
}


# fp32 的机器 epsilon。判据的绝对项按它乘以累加长度的平方根给，理由见 `tolerances`。
_FP32_EPS = 1.19e-7
# 经验安全系数：实测 `(131072, 20)` 的最大偏差 4.58e-5，公式给 3.4e-5，比值 1.35。留到 8 倍。
_TOLERANCE_MARGIN = 8.0


def tolerances(reduced_length):
    """按归约长度给判据。

    写死 `atol=1e-5` 对长归约是错的判据，而且会**假报缺陷**。131072 个 `randn` 求和的典型
    量级是 `sqrt(131072) ≈ 362`，但个别输出列会发生严重相消落到 1 附近；此时 fp32 的累加
    舍入噪声（约 `sqrt(n) × eps × E|x|`）相对该元素就是 1e-4 量级，相对判据完全失效。实测
    `(131072, 20)` 就是这样：20 个元素只错 1 个，偏差 4.58e-5，反推真值 ≈ 0.77，比典型值小
    470 倍——是相消，不是算错。

    所以绝对项取 `margin × sqrt(n) × eps`，相对项仍留 1e-5：
    地址错位 / 丢项 / 跨核合并漏块这类真缺陷会让整行或整片错、且偏差量级与真值相当
    （见 `rcore` 形态注释引的 910B 实测），放宽到这个量级不会把它们放过。
    """
    atol = max(1e-5, _TOLERANCE_MARGIN * math.sqrt(float(reduced_length)) * _FP32_EPS)
    return 1e-5, atol


def make_module(torch, dim):
    class ReduceSum(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=dim, keepdim=True)

    return ReduceSum()


def parse_args():
    parser = argparse.ArgumentParser(
        description="reduce 模板 x 形态矩阵的 910B 数值验收"
    )
    NpuRunner.add_common_arguments(parser)
    parser.add_argument(
        "--form",
        choices=sorted(FORMS),
        default=os.environ.get("AF_E2E_REDUCE_FORM", "ar"),
        help="归约形态；也可用 AF_E2E_REDUCE_FORM 指定（runner 不转发额外参数时用它）。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    form = args.form
    dim, shape_cases = FORMS[form]
    # 钉的是哪个 plan 要入账：没有它，日志只能证明「某个模板算对了」，证明不了是哪个。
    pinned = selected_plan()
    print(f"MATRIX_CONFIG form={form} dim={dim} pinned_plan={pinned}")

    runner = NpuRunner.from_args(args).setup()

    def make(name, shape, rtol, atol):
        def invoke():
            runner.run_case(
                name,
                make_module(runner.torch, dim),
                input_shapes=(shape,),
                rtol=rtol,
                atol=atol,
            )

        return invoke

    cells = []
    for index, (shape, note) in enumerate(shape_cases):
        name = f"MATRIX_{form.upper()}_{index}"
        rtol, atol = tolerances(shape[dim])
        # 容差是按归约长度算的，不是写死的——写死 atol 会在相消的格子上假报缺陷。
        # 它跟着形状走，所以必须逐格打出来，否则事后无法复核某一格用的是什么判据。
        print(f"MATRIX_TOL {name} reduced={shape[dim]} rtol={rtol:g} atol={atol:g}")
        cells.append((name, form, make(name, shape, rtol, atol), (shape,), note))

    return run_matrix("MATRIX", cells, plan_note=f"form={form} pinned_plan={pinned}")


if __name__ == "__main__":
    sys.exit(main())
