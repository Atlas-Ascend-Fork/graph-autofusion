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


@pytest.mark.parametrize("sign", [1, -1])
def test_finite_large_vectors_have_finite_cosine(analyzer, sign):
    left = np.array([1e200, 2e200], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        cosine, _, _ = analyzer.compute_metrics(left, sign * left)
    assert np.isfinite(cosine)
    assert cosine == pytest.approx(sign)


def test_large_identical_real_files_keep_valid_similarity(analyzer, tmp_path):
    path = tmp_path / "large.npy"
    np.save(path, np.array([1e200, 2e200]))
    source = analyzer.NpySource(str(path), "ND", "large")
    with np.errstate(over="ignore", invalid="ignore"):
        cosine, absolute, relative, status = analyzer.compare_data(source, source)
    assert cosine == pytest.approx(1)
    assert absolute == relative == 0
    assert status == "OK"


@pytest.mark.parametrize("magnitude", [1.0, 1e-8])
def test_existing_epsilon_formula_is_preserved(analyzer, magnitude):
    left = np.array([magnitude, 2 * magnitude])
    expected = np.dot(left, left) / (np.linalg.norm(left) ** 2 + 1e-8)
    assert analyzer.compute_metrics(left, left)[0] == pytest.approx(expected)


def test_zero_vector_convention_is_preserved(analyzer):
    assert analyzer.compute_metrics(np.zeros(2), np.zeros(2)) == (0, 0, 0)
