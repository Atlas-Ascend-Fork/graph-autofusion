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
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from commands.perf_formula import render_svg
from core.tiling_func_reader import CasePerfInfo, NodePerfInfo, SubItem


def node(name, pipe, **metrics):
    return NodePerfInfo(
        name,
        name.split("_")[0],
        pipe,
        [SubItem(key, value, 0.5) for key, value in metrics.items()],
        total=sum(metrics.values()),
        is_bottleneck=False,
    )


def case(identifier, nodes):
    return CasePerfInfo(
        result_id=1,
        group_id=0,
        case_id=identifier,
        nodes=nodes,
        bottleneck_pipe="AIV_MTE2",
    )


def comparison_text(cases):
    fragment, _ = render_svg("Add", cases, selected_case=(1, 0, cases[0].case_id))
    texts = [element.text for element in ET.fromstring(fragment).iter("text")]
    index = next(i for i, text in enumerate(texts) if text.startswith("敏感参数:"))
    return texts[index:]


def test_distinct_nodes_do_not_create_artificial_cross_case_variance():
    cases = [
        case(
            0,
            [
                node("Load_0", "AIV_MTE2", head_cost=10, actual_change=20),
                node("Store_0", "AIV_MTE3", head_cost=100),
            ],
        ),
        case(
            1,
            [
                node("Load_0", "AIV_MTE2", head_cost=12, actual_change=50),
                node("Store_0", "AIV_MTE3", head_cost=100),
            ],
        ),
    ]
    texts = comparison_text(cases)
    assert "敏感参数: actual_change（Load_0/AIV_MTE2；" in texts[0]
    assert texts[1:] == ["case0", "20.00", "case1", "50.00"]


def test_missing_metric_keeps_its_own_case_label():
    cases = [
        case(3, []),
        case(7, [node("Load_0", "AIV_MTE2", head_cost=10)]),
        case(9, [node("Load_0", "AIV_MTE2", head_cost=30)]),
    ]
    assert comparison_text(cases)[1:] == ["case7", "10.00", "case9", "30.00"]


def test_same_node_name_on_different_pipes_is_not_combined():
    cases = [
        case(
            0,
            [
                node("Copy_0", "AIV_MTE2", head_cost=10),
                node("Copy_0", "AIV_MTE3", head_cost=100),
            ],
        ),
        case(
            1,
            [
                node("Copy_0", "AIV_MTE2", head_cost=20),
                node("Copy_0", "AIV_MTE3", head_cost=100),
            ],
        ),
    ]
    texts = comparison_text(cases)
    assert "敏感参数: head_cost（Copy_0/AIV_MTE2；" in texts[0]
    assert texts[1:] == ["case0", "10.00", "case1", "20.00"]
