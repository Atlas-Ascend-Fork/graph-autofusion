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
import importlib.util
import os
import subprocess
import sys

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
ENTRYPOINT = os.path.join(SRC, "att.py")
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
EXAMPLES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../examples/examples.py")
)
sys.path.insert(0, SRC)
from core.log_parser import LogParser  # noqa: E402


def test_help_lists_all_commands():
    result = subprocess.run(
        [sys.executable, ENTRYPOINT, "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert all(
        x in result.stdout
        for x in (
            "summary",
            "compare",
            "verify-tiling",
            "split-slog",
            "perf-formula",
            "topn-templates",
        )
    )


def test_compare_missing_csv_returns_failure(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            ENTRYPOINT,
            "compare",
            str(tmp_path / "missing-a.csv"),
            str(tmp_path / "missing-b.csv"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "not found" in result.stderr
    assert result.stdout == ""


def test_summary_missing_log_returns_failure(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            ENTRYPOINT,
            "summary",
            str(tmp_path / "missing.log"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "not found" in result.stdout


def test_examples_start_with_cli_usage(capsys):
    spec = importlib.util.spec_from_file_location("att_examples", EXAMPLES)
    examples = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(examples)

    examples.example_1_cli_usage()
    output = capsys.readouterr().out

    assert output.startswith("Example 1: Command-line usage")
    for command in (
        "summary",
        "compare",
        "evidence",
        "split-slog",
        "perf-formula",
        "verify-tiling",
    ):
        assert f"python3 autofuse/tools/att_analyze/src/att.py {command}" in output


def test_verify_tiling_accepts_aiv_num_override():
    from att import build_parser

    args = build_parser().parse_args(
        ["verify-tiling", "source", "--preset", "B", "--aiv-num", "40"]
    )
    assert args.aiv_num == 40


def test_verify_tiling_rejects_conflicting_input_sources():
    from att import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["verify-tiling", "source", "--preset", "B", "--input-json", "input.json"]
        )


def test_operator_order_is_first_seen_and_stable():
    content = (
        "[Z] [PROF]Among all schedule results, graph0_result0 is the best choice\n"
        "[A] [PROF]Among all schedule results, graph0_result0 is the best choice\n"
        "[Z] [PROF]Among all schedule results, graph0_result0 is the best choice\n"
    )
    assert LogParser().extract_operator_names(content) == ["Z", "A"]


@pytest.mark.parametrize(
    "name,status",
    [
        ("test_missing_graph.log", "inferred_graph_result"),
        ("test_missing_group.log", "missing_group_case"),
        ("test_missing_result_perf.log", "missing_result_performance"),
    ],
)
def test_missing_parts_are_explicit(name, status):
    summaries = LogParser().parse_log_file(os.path.join(DATA, name))
    assert summaries and status in summaries[0].parse_status
