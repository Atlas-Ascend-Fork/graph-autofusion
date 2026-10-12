#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


TOOL = Path(__file__).resolve().parents[1] / "fusion_precision_analyzer.py"
SPEC = importlib.util.spec_from_file_location("nwa_complex_inputs", TOOL)
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def save_pair(tmp_path, a, b):
    lhs, rhs = tmp_path / "lhs.npy", tmp_path / "rhs.npy"
    np.save(lhs, a)
    np.save(rhs, b)
    return lhs, rhs


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("side", ["both", "left", "right"])
def test_complex_npy_comparison_reports_unavailable_metrics(
    tmp_path, capsys, dtype, side
):
    real = np.array([1, 3], dtype=np.float32)
    a = np.array([1 + 2j, 3 + 4j], dtype=dtype) if side != "right" else real
    b = np.array([1 - 2j, 3 - 4j], dtype=dtype) if side != "left" else real
    assert float(np.max(np.abs(a - b))) > 0
    lhs, rhs = save_pair(tmp_path, a, b)
    result = analyzer.compare_data(
        analyzer.NpySource(str(lhs), "ND", "lhs"),
        analyzer.NpySource(str(rhs), "ND", "rhs"),
    )
    assert result == (None, None, None, "COMPUTE_ERROR")
    assert "Complex-valued NPY data" in capsys.readouterr().out


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_mode2_does_not_print_zero_error_for_different_complex_data(tmp_path, dtype):
    lhs, rhs = save_pair(
        tmp_path,
        np.array([1 + 2j, 3 + 4j], dtype=dtype),
        np.array([1 - 2j, 3 - 4j], dtype=dtype),
    )
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--mode",
            "2",
            "--npy-a",
            str(lhs),
            "--npy-b",
            str(rhs),
        ],
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "状态: COMPUTE_ERROR" in result.stdout
    assert "Complex-valued NPY data" in result.stdout
    assert "绝对误差最大值:" not in result.stdout
    assert "ComplexWarning" not in result.stderr


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_real_npy_comparison_preserves_metrics(tmp_path, dtype):
    lhs, rhs = save_pair(
        tmp_path, np.array([1, 3], dtype=dtype), np.array([1, 5], dtype=dtype)
    )
    cosine, max_abs, max_rel, status = analyzer.compare_data(
        analyzer.NpySource(str(lhs), "ND", "lhs"),
        analyzer.NpySource(str(rhs), "ND", "rhs"),
    )
    assert status == "OK"
    assert np.isfinite(cosine) and 0 < cosine < 1
    assert max_abs == 2
    assert max_rel == pytest.approx(2 / (5 + 1e-8))


def test_real_mode2_preserves_success_output(tmp_path):
    lhs, rhs = save_pair(
        tmp_path, np.array([1, 3], dtype=np.float32), np.array([1, 5], dtype=np.float32)
    )
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--mode",
            "2",
            "--npy-a",
            str(lhs),
            "--npy-b",
            str(rhs),
        ],
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "状态: OK" in result.stdout
    assert "绝对误差最大值: 2.000000e+00" in result.stdout
