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
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from compare_csv import CSVComparator, ComparisonReporter


def comparison(tmp_path, first, second):
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    for path, value in zip(paths, [first, second]):
        path.write_text(
            f"Operator,Graph,Result,Group,Case,Result Perf\nAdd,0,0,0,1,{value}\n",
            encoding="utf-8",
        )
    return CSVComparator().compare_csv_files(*(str(path) for path in paths))


def test_zero_baseline_performance_matches_detailed_field_difference(tmp_path, capsys):
    result = comparison(tmp_path, 0, 10)
    perf = result.performance_differences[0]
    field = next(
        item
        for item in result.operator_differences["Add|0|0|0"]
        if item.field_name == "Result Perf"
    )
    assert perf.percentage_diff == field.percentage_diff == float("inf")
    assert perf.absolute_diff == 10
    assert not perf.improvement
    ComparisonReporter(result).print_console_report()
    report = capsys.readouterr().out
    assert "+0.00%" not in report
    assert "+10.000000" in report


@pytest.mark.parametrize(
    ("first", "second", "percentage", "improvement"),
    [
        (10, 20, 100.0, False),
        (10, 0, -100.0, True),
        (10, 5, -50.0, True),
    ],
)
def test_nonzero_baseline_formula_stays_unchanged(
    tmp_path, first, second, percentage, improvement
):
    perf = comparison(tmp_path, first, second).performance_differences[0]
    assert perf.percentage_diff == percentage
    assert perf.improvement is improvement


def test_equal_zeros_do_not_create_a_performance_difference(tmp_path):
    result = comparison(tmp_path, 0, 0)
    assert not result.performance_differences
    assert not result.operator_differences
