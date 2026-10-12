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
import re
import xml.etree.ElementTree as ET
import pytest
from commands.perf_formula import render_svg
from core.tiling_func_reader import CasePerfInfo, NodePerfInfo


@pytest.mark.parametrize("count", [1, 3, 4, 5, 6])
def test_group_rows_have_distinct_coordinates(count):
    cases = [
        CasePerfInfo(0, i, 0, [NodePerfInfo(f"node{i}", "Load", "AIV_MTE2", total=1)])
        for i in range(count)
    ]
    svg, height = render_svg("Op", cases, None)
    texts = ET.fromstring(svg).findall(".//text")
    groups = [
        (int(text.get("x")), int(text.get("y")))
        for text in texts
        if re.fullmatch(r"group\d+", text.text or "")
    ]
    assert len(groups) == len(set(groups)) == count
    assert max(y for _, y in groups) < height
    if count > 3:
        assert groups[2][1] > groups[0][1]
