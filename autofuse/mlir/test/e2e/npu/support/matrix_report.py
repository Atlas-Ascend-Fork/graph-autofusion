#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""矩阵表（`elewise_matrix` / `broadcast_matrix` / `reduce_fusion_matrix`）共用的跑表循环。

为什么单独一个模块：这三张表的成败词汇必须完全一致，否则跨表对账就没有意义；而判级规则
是从实测里长出来的，还会继续改（2026-10-16 第一轮就改了一次）。三份拷贝必然跑偏。

词汇表、严重度次序、判级规则和机读上报格式都在 `case_status` 里，本模块只负责「逐格隔离地
跑一张表并打印结果」这一件事。单用例脚本走的是 `NpuRunner.main`，判级用的是同一份规则。
"""

import traceback
from typing import Any, Callable, Dict, List, Sequence, Tuple

from support import case_status
from support.npu_runner import capture_fd_stderr

# 判级与拒收族的定义已下沉到 `case_status`：统一入口 `run.py` 的解析端和 `npu_runner` 的
# 单用例上报端都要用它，留在这里会和 `npu_runner` 形成循环导入。这里再导出，是为了让
# 既有的 `from support.matrix_report import ...` 继续成立。
from support.case_status import REFUSAL_MARKERS, classify, refusal_tags  # noqa: F401


def run_matrix(
    prefix: str,
    cells: Sequence[Tuple[str, str, Callable[[], None], Any]],
    plan_note: str = "",
) -> int:
    """跑一张矩阵表并打印统一格式的结果。

    `cells` 的每项是 `(格子名, 分组名, 跑这格的闭包, 入账用的 shape)`，可选再跟一个
    说明字符串。闭包自己负责建图和调 `runner.run_case`——这样这个模块不需要知道任何
    算子或形状的事。

    返回进程退出码：只有 MISMATCH / ERROR 让它非零。REFUSED 和 PASS_FALLBACK 都是
    已记录的现状，让它们变红的话每次跑都是红的，红就不再是信号。
    """
    print(f"{prefix}_PLAN cases={len(cells)} {plan_note}")
    rows: List[Tuple[str, str, str, str, Any, str, str]] = []

    for cell in cells:
        name, group, invoke, shapes = cell[0], cell[1], cell[2], cell[3]
        note = cell[4] if len(cell) > 4 else ""
        error = None
        with capture_fd_stderr() as diagnostics:
            try:
                invoke()
            except Exception as caught:  # noqa: BLE001 - 逐例隔离本来就是这轮的目的
                error = caught
                traceback.print_exc()
        status, tag = classify(error, diagnostics["text"])
        reason = ""
        if error is not None:
            text = str(error).strip()
            reason = text.splitlines()[0] if text else ""
        rows.append((name, group, status, tag, shapes, reason, note))
        label = f"{status}({tag})" if tag else status
        suffix = f" :: {reason}" if reason else ""
        print(f"{prefix}_RESULT {name} {label} shapes={shapes}{suffix}")
        # 机读行，给 `run.py` 跨脚本对账用。人读那行一个字都没改。
        print(
            case_status.emit_case(prefix, name, status, tag, shapes, reason, group),
            flush=True,
        )

    tally: Dict[str, int] = {}
    for row in rows:
        tally[row[2]] = tally.get(row[2], 0) + 1
    summary = " ".join(f"{key}={value}" for key, value in sorted(tally.items()))
    print(f"{prefix}_SUMMARY total={len(rows)} {summary}")
    print(case_status.emit_suite(prefix, tally))
    # 分组小计。各表本来就有分组这个维度（算子族 / dtype / 形态），此前只有
    # 所有矩阵共用分组小计，避免各脚本重复实现。
    for group_name in sorted({row[1] for row in rows}):
        group_rows = [row for row in rows if row[1] == group_name]
        clean = sum(1 for row in group_rows if row[2] == case_status.PASS)
        print(f"{prefix}_SUMMARY_GROUP {group_name} PASS={clean}/{len(group_rows)}")
    for name, group, status, tag, shapes, reason, note in rows:
        label = f"{status}({tag})" if tag else status
        tail = f" // {note}" if note else ""
        print(
            f"{prefix}_SUMMARY_ROW {name} group={group} {label} shapes={shapes}{tail} {reason}"
        )

    bad = sum(1 for row in rows if case_status.is_defect(row[2]))
    return 1 if bad else 0
