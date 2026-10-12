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


@pytest.mark.parametrize(
    ("folder", "operator"),
    [
        ("capture[0]", "Add"),
        ("capture", "Add[0]"),
        ("capture[0]", "scope/Add[0]"),
        ("capture", "Add"),
    ],
)
def test_real_dump_paths_are_matched_literally(analyzer, tmp_path, folder, operator):
    directory = tmp_path / folder
    directory.mkdir()
    target = directory / f"prefix.{operator.replace('/', '_')}.timestamp.output.0.npy"
    np.save(target, np.array([1, 2], dtype=np.int32))
    found = analyzer.find_npy(str(directory), operator, "output", 0)
    assert found == str(target)
    np.testing.assert_array_equal(analyzer.load_npy(found), [1, 2])


def test_missing_port_remains_missing(analyzer, tmp_path):
    np.save(tmp_path / "prefix.Add.time.output.0.npy", np.array([1]))
    assert analyzer.find_npy(str(tmp_path), "Add", "output", 1) is None
