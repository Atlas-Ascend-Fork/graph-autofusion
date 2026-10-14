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

from commands.case_filter import parse_case_arg
from commands.split_slog import SlogSplitter, run


def compiler_log(case_ids):
    lines = [
        "[DFX]Begin to gen model info for asc graph 0, schedule result 1, schedule group 0\n"
    ]
    for case in case_ids:
        lines += [
            f"[DFX]Begin to generate model info for graph Add of tiling case id {case}\n",
            f"payload-{case}\n",
            f"[DFX]End to generate model info for graph Add of tiling case id {case}\n",
        ]
    lines.append("[DFX]End to gen model info for graph0_result1_g0\n")
    return "".join(lines)


def test_noncontiguous_compiler_cases_keep_names(tmp_path):
    splitter = SlogSplitter()
    groups = splitter.parse_compiler_model_info(compiler_log([2, 7]))
    splitter.write_compiler_output(groups, "Add", str(tmp_path))
    directory = tmp_path / "Add/compiler/graph0_result1/g0"
    assert sorted(path.name for path in directory.iterdir()) == [
        "case2.log",
        "case7.log",
    ]
    assert "payload-7" in (directory / "case7.log").read_text()


def test_case_filter_uses_recorded_id(tmp_path):
    log = tmp_path / "source.log"
    log.write_text(compiler_log([2, 7]), encoding="utf-8")
    output = tmp_path / "split"
    run(SimpleNamespace(log_path=str(log), op="Add", output=str(output), case="c=7"))
    directory = output / "Add/compiler/graph0_result1/g0"
    assert sorted(path.name for path in directory.iterdir()) == ["case7.log"]


def test_reversed_case_order_keeps_source_identity(tmp_path):
    splitter = SlogSplitter()
    groups = splitter.parse_compiler_model_info(compiler_log([7, 2]))
    selector = parse_case_arg("c=2")
    cases = groups[(0, 1, 0)]
    kept = [
        text
        for index, text in enumerate(cases)
        if selector.match(1, 0, splitter._case_id(text, index))
    ]
    assert len(kept) == 1 and "payload-2" in kept[0]


def test_runtime_and_legacy_fallbacks_keep_identity():
    assert SlogSplitter._case_id("[Add] [PROF] graph0_result1_g0_R42", 0) == 42
    assert SlogSplitter._case_id("body without case labels", 9) == 9
