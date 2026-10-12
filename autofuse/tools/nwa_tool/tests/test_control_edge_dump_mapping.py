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
import sys
from pathlib import Path

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


def test_ge_serialized_control_dependency_has_no_tensor_mapping(analyzer):
    op = {
        "name": "Fused",
        "type": "AscBackend",
        "input": ["input:0", "barrier:-1"],
        "input_desc": [{"layout": "ND"}],
    }
    _, mappings = analyzer.extract_fusion_mappings({"graph": [{"op": [op]}]})
    assert len(mappings) == 1
    assert mappings[0]["source_op_name"] == "input"
    assert mappings[0]["source_output_index"] == 0
    assert mappings[0]["fused_input_index"] == 0
    assert mappings[0]["fused_format"] == "ND"


def test_control_only_operator_has_no_data_inputs(analyzer):
    assert analyzer.extract_input_mappings({"input": ["barrier:-1"]}, "Fused") == []


def test_data_port_and_colon_in_node_name_are_preserved(analyzer):
    mappings = analyzer.extract_input_mappings(
        {"input": ["scope:source:3"], "input_desc": [{"layout": "ND"}]}, "Fused"
    )
    assert len(mappings) == 1
    assert mappings[0]["source_op_name"] == "scope:source"
    assert mappings[0]["source_output_index"] == 3


def test_existing_optional_missing_input_diagnostic_is_preserved(analyzer):
    mappings = analyzer.extract_input_mappings({"input": [""]}, "Fused")
    assert mappings[0]["status"] == "NO_MAPPING"
