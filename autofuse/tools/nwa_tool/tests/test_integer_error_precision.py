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
    ("dtype", "left", "right"),
    [
        ("int64", 2**53 + 1, 2**53),
        ("int64", 2**63 - 1, 2**63 - 2),
        ("uint64", 2**64 - 1, 2**64 - 2),
        ("int64", -(2**63), -(2**63) + 1),
    ],
)
def test_adjacent_large_integers_keep_their_error(analyzer, dtype, left, right):
    _, absolute, relative = analyzer.compute_metrics(
        np.array([left], dtype=dtype), np.array([right], dtype=dtype)
    )
    assert absolute == 1.0
    assert relative > 0


def test_real_mixed_signed_unsigned_files_do_not_lose_difference(analyzer, tmp_path):
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(a, np.array([2**53 + 1], dtype=np.int64))
    np.save(b, np.array([2**53], dtype=np.uint64))
    _, absolute, relative, status = analyzer.compare_data(
        analyzer.NpySource(str(a), "ND", "a"), analyzer.NpySource(str(b), "ND", "b")
    )
    assert absolute == 1.0
    assert relative > 0
    assert status == "DTYPE_CAST"


@pytest.mark.parametrize("dtype", ["int8", "float32", "float64"])
def test_small_value_metrics_are_unchanged(analyzer, dtype):
    _, absolute, relative = analyzer.compute_metrics(
        np.array([1, 3], dtype=dtype), np.array([1, 2], dtype=dtype)
    )
    assert absolute == 1.0
    assert relative == pytest.approx(1 / (3 + 1e-8))
