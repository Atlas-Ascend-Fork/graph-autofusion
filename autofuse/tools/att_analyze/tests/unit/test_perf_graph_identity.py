#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
from types import SimpleNamespace
from commands.perf_formula import run
from core.tiling_func_reader import parse_perf_lines


PERF = """[Op] [PERF][graph0_result0_g0_R0] Load_0[Load]: AIV_MTE2 = base_cost=1
[Op] [PERF][graph1_result0_g0_R0] Load_0[Load]: AIV_MTE2 = base_cost=100
"""


def test_different_graphs_remain_separate():
    cases = parse_perf_lines(PERF, "Op")
    assert len(cases) == 2
    assert [(case.graph_id, case.nodes[0].total) for case in cases] == [
        (0, 1),
        (1, 100),
    ]
    assert all(len(case.nodes) == 1 for case in cases)


def test_legacy_untagged_perf_uses_requested_graph():
    cases = parse_perf_lines(
        "[Op] [PERF] Load_0[Load]: AIV_MTE2 = base_cost=1", "Op", graph_id=2
    )
    assert cases[0].graph_id == 2


def test_report_uses_selected_graph_only(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    log = tmp_path / "run.log"
    log.write_text(
        PERF
        + "[Op] [PROF]Among all schedule results, graph1_result0 is the best choice\n"
        "[Op] [PROF]Among the templates, tiling case 0 of graph1_result0_g0 is the best choice\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    args = SimpleNamespace(
        source_dir=str(source), log_path=str(log), case=None, output=str(output)
    )
    assert run(args) == 0
    svg = (output / "perf_formula.svg").read_text(encoding="utf-8")
    assert "Selected: graph1_result0" in svg
    assert "AIV_MTE2 = 100.00" in svg
    assert "AIV_MTE2 = 1.00" not in svg
