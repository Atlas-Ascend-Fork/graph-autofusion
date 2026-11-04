#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""上板用例清单：跑哪些、怎么跑、预期是什么。

为什么要有这份清单：此前「有哪些上板用例」的唯一答案是 `ls *.py`。那说不出三件事——
哪些脚本需要额外参数才算跑到了点上（`reduce_template_matrix` 不钉 plan 就只测到一个形态）、
哪些脚本必须配特定环境变量（钉 plan 走 ATT 的 `AUTOFUSE_DFX_FLAGS`）、以及**每一格现在
应该是什么结果**。前两件导致「跑了但没测到东西」，第三件导致漂移测不出来。

一条清单项是 `(脚本, argv, env)` 三元组而不只是脚本名：同一个脚本配不同的
`--force_schedule_result` 测的是不同的调度模板，它们是**不同的用例**，不是一个用例的参数。

## 预期等级（`expect`）

`expect` 是 `{格子名: 等级}`，选填，判据见 `support/case_status.py`：

- **给了就断言**——实测比它差就红，比它好就提示「可以收紧了」。
- **没给就不断言**——沿用既有规则，只有 `MISMATCH` / `ERROR` 让退出码非零。

必填是错的，理由和 CPU 侧 `AddCpuCase.cmake` 里 `COVERS` 那条一样：必填只会逼着新用例
填一个没推导过的值，而「声明与实际不符还能通过」正是这套机制要防的事。

预期来自 2026-10-22 的 DevEnvC_1AtLk 实机校准（CANN 9.2.0，scratch 修改及
solver 头文件修复后的构建），按本清单默认 argv/env 运行；broadcast_matrix 和 elewise_matrix 现已撤下。
AllLoad 后续复测确认 `(4,15,1)` 尾维消除后是 AR，应钉 plan 2；已单独注册固定回归。
ARA plan 2 排除两端 A 维为 1 的形状，补足 12 格并实测全部 PASS；相关预期已回填。

PASS_FALLBACK 只表示数值通过且出现已知拒收诊断，不代表 MLIR 全链路通过；当前广播
能力不足会转 legacy。撤下矩阵前的 8 个广播清单项（22 格）修改前后等级及标签完全一致。
后续运行 `python3 run.py --calibrate` 可生成新的预期，回填前应检查异常退出和缺失格子。

"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

from support import case_status


def dfx(plan: int, tiling_case: int = None) -> Dict[str, str]:
    """钉调度候选用的 ATT DFX 环境变量。

    用 `AUTOFUSE_DFX_FLAGS` 而不是 `AF_MLIR_PINNED_PLAN_ID`：后者是 MLIR 专用的旧开关，
    已经删了（`docs/design/2026-10-03-mlir-plan-pinning-switch-design.md` 标注「已移除」），
    编译器已不再读取它（设备 worker 仅清理同名环境变量）。设置它不会钉住 plan——跑出来的是未钉 plan 的结果，
    而日志看着像钉上了。
    """
    flags = ["--force_schedule_result={}".format(plan)]
    if tiling_case is not None:
        flags.append("--force_tiling_case={}".format(tiling_case))
    return {"AUTOFUSE_DFX_FLAGS": ";".join(flags)}


@dataclass(frozen=True)
class Case:
    """清单里的一项。`id` 是它在统一入口里的名字，必须唯一。"""

    id: str
    script: str
    argv: Tuple[str, ...] = ()
    env: Dict[str, str] = field(default_factory=dict)
    groups: Tuple[str, ...] = ()
    expect: Dict[str, str] = field(default_factory=dict)
    note: str = ""


# `--steps 0` 关 profiling：这些都是**数值与支持面**的用例，不是性能用例。开着 profiling
# 每格多跑 100 轮，整轮时间翻几倍而结论一个字不变。要测性能的走 device_validation_runner，
# 它和这里不是一套判据（见 docs 里上板对照那篇）。
NO_PROFILING = ("--steps", "0")

CASES = (
    # ------------------------------------------------------------------ 逐元素与直通
    Case(
        id="load_store_direct",
        script="load_store_direct.py",
        argv=NO_PROFILING,
        groups=("smoke", "elementwise"),
        note="最短路径：load → store，环境没搭对的话它最先挂",
        expect={
            "load_store_direct": case_status.PASS,
        },
    ),
    Case(
        id="load_add_store",
        script="load_add_store.py",
        argv=NO_PROFILING,
        groups=("smoke", "elementwise"),
        expect={
            "load_add_store": case_status.PASS,
        },
    ),
    Case(
        id="mixed_consumption",
        script="mixed_consumption.py",
        argv=NO_PROFILING,
        groups=("elementwise",),
        expect={
            "mixed_consumption": case_status.PASS,
        },
    ),
    # ------------------------------------------------------------------ 广播
    Case(
        id="load_broadcast_add_direct",
        script="load_broadcast_add_direct.py",
        argv=NO_PROFILING,
        groups=("broadcast",),
        expect={
            "load_broadcast_add_direct": case_status.PASS_FALLBACK,
        },
    ),
    Case(
        id="load_broadcast_to_abs",
        script="load_broadcast_to_abs.py",
        argv=NO_PROFILING,
        groups=("broadcast",),
        expect={
            "load_broadcast_to_abs": case_status.PASS,
        },
    ),
    Case(
        id="load_broadcast_to_cascade",
        script="load_broadcast_to_cascade.py",
        argv=NO_PROFILING,
        groups=("broadcast", "cascade"),
        expect={
            "load_broadcast_to_cascade": case_status.PASS_FALLBACK,
        },
    ),
    Case(
        id="load_broadcast_diamond_cascade",
        script="load_broadcast_diamond_cascade.py",
        argv=NO_PROFILING,
        groups=("broadcast", "cascade"),
        expect={
            "load_broadcast_diamond_cascade": case_status.PASS_FALLBACK,
        },
    ),
    Case(
        id="load_broadcast_inner_brc_cascade",
        script="load_broadcast_inner_brc_cascade.py",
        argv=NO_PROFILING,
        groups=("broadcast", "cascade"),
        expect={
            "load_broadcast_inner_brc_cascade": case_status.PASS,
        },
    ),
    Case(
        id="load_broadcast_interleaved_axes_cascade",
        script="load_broadcast_interleaved_axes_cascade.py",
        argv=NO_PROFILING,
        groups=("broadcast", "cascade"),
        expect={
            "load_broadcast_interleaved_axes_cascade": case_status.PASS_FALLBACK,
        },
    ),
    Case(
        id="load_broadcast_tail_pitch_cascade",
        script="load_broadcast_tail_pitch_cascade.py",
        argv=NO_PROFILING,
        groups=("broadcast", "cascade"),
        expect={
            "load_broadcast_tail_pitch_cascade": case_status.PASS_FALLBACK,
        },
    ),
    # ------------------------------------------------------------------ 归约：单用例
    Case(
        id="load_sum_store",
        script="load_sum_store.py",
        argv=NO_PROFILING,
        groups=("reduce",),
        expect={
            "load_sum_store": case_status.PASS,
        },
    ),
    Case(
        id="load_sum_store_rcore",
        script="load_sum_store_rcore.py",
        argv=NO_PROFILING,
        groups=("reduce", "rcore"),
        expect={
            "load_sum_store_rcore": case_status.PASS,
        },
    ),
    Case(
        id="load_mean_store_rcore",
        script="load_mean_store_rcore.py",
        argv=NO_PROFILING,
        groups=("reduce", "rcore"),
        expect={
            "load_mean_store_rcore": case_status.PASS,
        },
    ),
    Case(
        id="reduce_all",
        script="reduce_all.py",
        argv=NO_PROFILING,
        groups=("reduce",),
        note="全轴归约的现状探针；A3 那条边界要重新实测，所以不预设 expect",
        expect={
            "REDUCE_ALL_0": case_status.PASS,
            "REDUCE_ALL_1": case_status.PASS,
            "REDUCE_ALL_2": case_status.PASS,
            "REDUCE_ALL_3": case_status.PASS,
            "REDUCE_ALL_4": case_status.PASS,
            "REDUCE_ALL_5": case_status.PASS,
        },
    ),
    Case(
        id="reduce_rarr",
        script="reduce_rarr.py",
        argv=NO_PROFILING,
        groups=("reduce",),
        note="保留轴不在末位的多轴归约",
        expect={
            "RARR_0": case_status.PASS,
            "RARR_1": case_status.PASS,
            "RARR_2": case_status.PASS,
            "RARR_3": case_status.PASS,
        },
    ),
    Case(
        id="reduce_fusion_matrix",
        script="reduce_fusion_matrix.py",
        argv=NO_PROFILING,
        groups=("matrix", "reduce"),
        note="归约与邻居的融合形态：prod / cons / fork / multi",
        expect={
            "prod_mul_sum_small": case_status.PASS,
            "prod_mul_sum0_small": case_status.PASS,
            "cons_sum_relu_small": case_status.PASS,
            "cons_sum_mul_small": case_status.PASS,
            "cons_sum0_mul_small": case_status.PASS,
            "both_mul_sum_relu_small": case_status.PASS,
            "fork_mul_small": case_status.PASS,
            "fork_relu_small": case_status.PASS,
            "multi_two_sums_small": case_status.PASS,
            "multi_sum_max_small": case_status.PASS,
            "prod_mul_sum_tail": case_status.PASS,
            "prod_mul_sum0_tail": case_status.PASS,
            "cons_sum_relu_tail": case_status.PASS,
            "cons_sum_mul_tail": case_status.PASS,
            "cons_sum0_mul_tail": case_status.PASS,
            "both_mul_sum_relu_tail": case_status.PASS,
            "fork_mul_tail": case_status.PASS,
            "fork_relu_tail": case_status.PASS,
            "multi_two_sums_tail": case_status.PASS,
            "multi_sum_max_tail": case_status.PASS,
            "prod_mul_sum_large": case_status.PASS,
            "prod_mul_sum0_large": case_status.PASS,
            "cons_sum_relu_large": case_status.PASS,
            "cons_sum_mul_large": case_status.PASS,
            "cons_sum0_mul_large": case_status.PASS,
            "both_mul_sum_relu_large": case_status.PASS,
            "fork_mul_large": case_status.PASS,
            "fork_relu_large": case_status.PASS,
            "multi_two_sums_large": case_status.PASS,
            "multi_sum_max_large": case_status.PASS,
        },
    ),
    # ------------------------------------------------- 归约：模板 × 形态（钉 plan）
    #
    # 形态与 `--force_schedule_result` 的对应抄自 `reduce_template_matrix.py` 的用法说明，
    # 编号以最终生成的 ATT schedule key 为准。**同一个 form 配不同的 plan 是两个用例**：
    # `(ar, 0)` 测 RCommon，`(ar, 2)` 测 AllLoad，走的是不同的模板。
    Case(
        id="template_rcommon_ar",
        script="reduce_template_matrix.py",
        argv=NO_PROFILING + ("--form", "ar"),
        env=dfx(0),
        groups=("matrix", "reduce", "plan-pinned"),
        note="RCommon × AR",
        expect={
            "MATRIX_AR_0": case_status.PASS,
            "MATRIX_AR_1": case_status.PASS,
            "MATRIX_AR_2": case_status.PASS,
        },
    ),
    Case(
        id="template_rcommon_ra",
        script="reduce_template_matrix.py",
        argv=NO_PROFILING + ("--form", "ra"),
        env=dfx(0),
        groups=("matrix", "reduce", "plan-pinned"),
        note="RCommon × RA",
        expect={
            "MATRIX_RA_0": case_status.PASS,
            "MATRIX_RA_1": case_status.PASS,
            "MATRIX_RA_2": case_status.PASS,
        },
    ),
    Case(
        id="template_allload_ar",
        script="reduce_template_matrix.py",
        argv=NO_PROFILING + ("--form", "ar"),
        env=dfx(2),
        groups=("matrix", "reduce", "plan-pinned"),
        note="AllLoad × AR",
        expect={
            "MATRIX_AR_0": case_status.PASS,
            "MATRIX_AR_1": case_status.PASS,
            "MATRIX_AR_2": case_status.PASS,
        },
    ),
    Case(
        id="template_allload_ra",
        script="reduce_template_matrix.py",
        argv=NO_PROFILING + ("--form", "ara"),
        env=dfx(2),
        groups=("matrix", "reduce", "plan-pinned"),
        note="AllLoad × RA；只有 ARA 能产出这一格",
        expect={
            "MATRIX_ARA_0": case_status.PASS,
        },
    ),
    Case(
        id="template_rcore_ar",
        script="reduce_template_matrix.py",
        argv=NO_PROFILING + ("--form", "rcore_ar"),
        env=dfx(1),
        groups=("matrix", "reduce", "plan-pinned", "rcore"),
        note="RCore × AR；nA < corenum，跨核归约真的启用",
        expect={
            "MATRIX_RCORE_AR_0": case_status.PASS,
            "MATRIX_RCORE_AR_1": case_status.PASS,
            "MATRIX_RCORE_AR_2": case_status.PASS,
        },
    ),
    Case(
        id="template_rcore_ra",
        script="reduce_template_matrix.py",
        argv=NO_PROFILING + ("--form", "rcore_ra"),
        env=dfx(1),
        groups=("matrix", "reduce", "plan-pinned", "rcore"),
        note="RCore × RA；与 rcore_ar 镜像",
        expect={
            "MATRIX_RCORE_RA_0": case_status.PASS,
            "MATRIX_RCORE_RA_1": case_status.PASS,
            "MATRIX_RCORE_RA_2": case_status.PASS,
        },
    ),
    Case(
        id="template_ara_agap",
        script="reduce_template_matrix.py",
        argv=NO_PROFILING + ("--form", "ara_agap"),
        env=dfx(0),
        groups=("matrix", "reduce", "plan-pinned"),
        note="AGENTS.md §3.4 槽位跨度的回归：A 组多轴时每个 tile 尾部几行",
        expect={
            "MATRIX_ARA_AGAP_0": case_status.PASS,
            "MATRIX_ARA_AGAP_1": case_status.PASS,
        },
    ),
    # 固定的退化 AR AllLoad 回归不属于 fuzz，继续保留。
    Case(
        id="allload_ar_tail1",
        script="load_min_store_allload_tail1.py",
        argv=NO_PROFILING,
        env=dfx(2),
        groups=("reduce", "plan-pinned"),
        expect={"allload_ar_tail1": case_status.PASS},
        note="amin(4,15,1), dim=1：尾维消除后为 AR，AllLoad 是 plan 2",
    ),
)


def by_id() -> Dict[str, Case]:
    index = {}
    for case in CASES:
        if case.id in index:
            raise ValueError("清单里 id 重复：{}".format(case.id))
        index[case.id] = case
    return index


def all_groups() -> Tuple[str, ...]:
    groups = set()
    for case in CASES:
        groups.update(case.groups)
    return tuple(sorted(groups))


def select(ids=(), groups=()) -> Tuple[Case, ...]:
    """按 id 与分组挑选。两者都空则全选；都给则取并集。"""
    if not ids and not groups:
        return CASES
    index = by_id()
    unknown = [name for name in ids if name not in index]
    if unknown:
        raise ValueError(
            "清单里没有这些 id：{}\n    可用：{}".format(
                ", ".join(unknown), ", ".join(sorted(index))
            )
        )
    wanted = set(ids)
    chosen = [
        case
        for case in CASES
        if case.id in wanted or (groups and set(case.groups) & set(groups))
    ]
    return tuple(chosen)
