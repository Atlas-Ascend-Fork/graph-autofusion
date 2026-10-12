#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def analyzer(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "fusion_precision_analyzer.py"
    name = "nwa_precision_cpu_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reverse", [False, True])
def test_bad_layout_rank_is_reported_per_pair(analyzer, tmp_path, reverse):
    path = tmp_path / "bad.npy"
    np.save(path, np.ones((1, 2, 2, 16)))
    source = analyzer.NpySource(str(path), "NC1HWC0", "bad-layout")
    other = analyzer.NpySource(str(path), "NHWC", "reference")
    args = (other, source) if reverse else (source, other)
    assert analyzer.compare_data(*args) == (None, None, None, "FORMAT_CONVERSION_ERROR")


def test_valid_conversion_is_preserved(analyzer, tmp_path):
    source, reference = tmp_path / "blocked.npy", tmp_path / "plain.npy"
    np.save(source, np.ones((1, 1, 2, 2, 16)))
    np.save(reference, np.ones((1, 2, 2, 16)))
    result = analyzer.compare_data(
        analyzer.NpySource(str(source), "NC1HWC0", "source"),
        analyzer.NpySource(str(reference), "NHWC", "reference"),
    )
    assert result[1:3] == (0, 0)
    assert result[3] == "FORMAT_CONVERTED"


def test_unsupported_conversion_retains_its_status(analyzer, tmp_path):
    path = tmp_path / "plain.npy"
    np.save(path, np.ones((2, 2)))
    result = analyzer.compare_data(
        analyzer.NpySource(str(path), "ND", "a"),
        analyzer.NpySource(str(path), "UNSUPPORTED", "b"),
    )
    assert result == (None, None, None, "FORMAT_UNSUPPORTED")


def test_mode1_reports_bad_pair_and_continues_to_valid_output(
    analyzer, tmp_path, capsys
):
    opened, closed = tmp_path / "open", tmp_path / "closed"
    opened.mkdir()
    closed.mkdir()
    fused_ops, origin_ops = [], []
    for name, shape in [("Bad", (1, 2, 2, 16)), ("Good", (1, 1, 2, 2, 16))]:
        fused_name, origin_name = "Fused" + name, "Origin" + name
        fused_ops.append(
            {
                "name": fused_name,
                "type": "AscBackend",
                "output_desc": [
                    {
                        "layout": "NC1HWC0",
                        "attr": [
                            {
                                "key": "_datadump_origin_name",
                                "value": {"s": origin_name},
                            },
                            {"key": "_datadump_origin_output_index", "value": {"i": 0}},
                        ],
                    }
                ],
            }
        )
        origin_ops.append(
            {
                "name": origin_name,
                "type": "Identity",
                "output_desc": [{"layout": "NHWC"}],
            }
        )
        np.save(opened / f"prefix.{fused_name}.time.output.0.npy", np.ones(shape))
        np.save(
            closed / f"prefix.{origin_name}.time.output.0.npy", np.ones((1, 2, 2, 16))
        )
    open_graph, close_graph = tmp_path / "open.json", tmp_path / "closed.json"
    open_graph.write_text(json.dumps({"graph": [{"op": fused_ops}]}), encoding="utf-8")
    close_graph.write_text(
        json.dumps({"graph": [{"op": origin_ops}]}), encoding="utf-8"
    )
    args = SimpleNamespace(
        af_open_graph=str(open_graph),
        af_close_graph=str(close_graph),
        af_open_data=str(opened),
        af_close_data=str(closed),
        compare_input=False,
    )
    analyzer.run_mode1(args)
    output = capsys.readouterr().out
    assert "FORMAT_CONVERSION_ERROR" in output
    assert "FusedGood" in output
    assert "FORMAT_CONVERTED" in output
