# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Portable TorchInductor NPU CV validation runner.

This script is intentionally self-contained so a fresh environment without
historical temp scripts can still produce reproducible env probes, dry-run
plans, and matrix case artifacts.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
_LOGGER = logging.getLogger(__name__)


STAGES = {
    "env",
    "build",
    "deploy",
    "fusion_codegen",
    "ascendc_compile",
    "kernel_load",
    "launch_sync",
    "numeric",
    "summary",
}


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    kind: str
    dynamic: bool
    dtype: str
    shape: dict[str, Any]
    variants: tuple[dict[str, int], ...]
    fusion_mode: int | None
    ub_mode: int | None
    mix_mode: int | None
    atol: float
    rtol: float
    timeout: int
    op: str = ""
    family: str = "core"
    mode_variant: str = "cv"
    expect_cv: bool = False
    expected_template: str | None = None
    expected_tiling_key: int | None = None
    bias: str = "row"
    a_trans: bool = False
    b_trans: bool = False


def _base_matrix() -> tuple[Case, ...]:
    core = [
        Case(
            "vv_static_add_mul_mul",
            "vv",
            False,
            "float32",
            {"x": [32, 64]},
            ({"m": 32, "n": 64},),
            None,
            None,
            None,
            1e-3,
            1e-3,
            180,
            mode_variant="vv",
            family="baseline",
        ),
        Case(
            "vv_dynamic_add_mul_mul",
            "vv",
            True,
            "float32",
            {"x": [32, 64]},
            ({"m": 32, "n": 64}, {"m": 40, "n": 64}),
            None,
            None,
            None,
            1e-3,
            1e-3,
            240,
            mode_variant="vv",
            family="baseline",
        ),
        Case(
            "non_fused_static_mm_only",
            "non_fused",
            False,
            "float32",
            {"m": 16, "n": 18, "k": 4},
            ({"m": 16, "n": 18, "k": 4},),
            None,
            None,
            None,
            1e-2,
            1e-2,
            240,
            mode_variant="vv",
            family="baseline",
        ),
        Case(
            "cv_static_mm_small_ub",
            "mm",
            False,
            "float32",
            {"m": 16, "n": 18, "k": 4},
            ({"m": 16, "n": 18, "k": 4},),
            0,
            0,
            None,
            1e-2,
            1e-2,
            300,
            expect_cv=True,
        ),
        Case(
            "cv_dynamic_mm_small_ub",
            "mm",
            True,
            "float32",
            {"m": 16, "n": 18, "k": 4},
            ({"m": 16, "n": 18, "k": 4}, {"m": 32, "n": 18, "k": 4}),
            0,
            0,
            None,
            1e-2,
            1e-2,
            360,
            expect_cv=True,
        ),
        Case(
            "cv_static_mm_512_non_ub",
            "mm",
            False,
            "float32",
            {"m": 512, "n": 512, "k": 512},
            ({"m": 512, "n": 512, "k": 512},),
            0,
            0,
            None,
            2e-2,
            2e-2,
            600,
            expect_cv=True,
        ),
        Case(
            "cv_dynamic_mm_512_non_ub",
            "mm",
            True,
            "float32",
            {"m": 512, "n": 512, "k": 512},
            ({"m": 512, "n": 512, "k": 512}, {"m": 384, "n": 512, "k": 512}),
            0,
            0,
            None,
            2e-2,
            2e-2,
            720,
            expect_cv=True,
        ),
        Case(
            "cv_static_mm_relu_fixpipe",
            "mm_fixpipe",
            False,
            "float32",
            {"m": 128, "n": 128, "k": 128},
            ({"m": 128, "n": 128, "k": 128},),
            0,
            None,
            None,
            1e-2,
            1e-2,
            360,
            expect_cv=True,
        ),
        Case(
            "cv_dynamic_bmm_key1",
            "bmm",
            True,
            "float32",
            {"b": 4, "m": 16, "k": 64, "n": 16},
            ({"b": 4, "m": 16, "k": 64, "n": 16}, {"b": 6, "m": 16, "k": 64, "n": 16}),
            0,
            0,
            None,
            2e-2,
            2e-2,
            420,
            expect_cv=True,
        ),
        Case(
            "cv_static_bmm_to_mul_k1_mix2",
            "bmm_k1_mix",
            False,
            "float32",
            {"b": 128, "m": 8, "k": 1, "n": 16},
            ({"b": 128, "m": 8, "k": 1, "n": 16},),
            1,
            None,
            1,
            1e-2,
            1e-2,
            360,
            expect_cv=True,
        ),
        Case(
            "cv_dynamic_bmm_to_mul_k1_mix2",
            "bmm_k1_mix",
            True,
            "float32",
            {"b": 128, "m": 8, "k": 1, "n": 16},
            ({"b": 128, "m": 8, "k": 1, "n": 16}, {"b": 96, "m": 8, "k": 1, "n": 16}),
            1,
            None,
            1,
            1e-2,
            1e-2,
            420,
            expect_cv=True,
        ),
        Case(
            "fragment08_mm_add_exp",
            "fragment08",
            False,
            "float32",
            {"x": [16384, 26], "w": [26, 256], "bias": [1, 256]},
            ({"m": 16384, "k": 26, "n": 256},),
            0,
            0,
            None,
            2e-2,
            2e-2,
            420,
            expect_cv=True,
        ),
    ]
    return tuple(
        core + _template_coverage_cases() + _mixed_dtype_cases() + _extended_cases()
    )


def _template_coverage_cases() -> list[Case]:
    specs = [
        ("mm_ub_basic", "mm", (1, 16, 64, 16), "ub", None, False, False, "mm_mul_mul"),
        (
            "mm_fallback_stream_k",
            "mm",
            (1, 64, 4096, 64),
            "stream_k",
            None,
            False,
            False,
            "mm_mul_mul",
        ),
        ("bmm_ub_basic", "bmm", (4, 16, 64, 16), "ub", 1, False, False, "bmm_mul_mul"),
        (
            "bmm_fallback_stream_k",
            "bmm",
            (2, 64, 4096, 64),
            "stream_k",
            4097,
            False,
            False,
            "bmm_mul_mul",
        ),
        (
            "bmm_k0_clear_output",
            "bmm",
            (128, 8, 0, 16),
            "k_equal_zero",
            8192,
            False,
            False,
            "bmm_mul_mul",
        ),
        (
            "bmm_to_mul_k1",
            "bmm",
            (128, 8, 1, 16),
            "batch_matmul_to_mul",
            513,
            False,
            False,
            "bmm_mul_mul",
        ),
        (
            "bmm_a_trans_medium",
            "bmm",
            (4, 16, 64, 16),
            "basic",
            17,
            True,
            False,
            "bmm_mul_mul",
        ),
        (
            "bmm_b_trans_medium",
            "bmm",
            (4, 16, 64, 16),
            "basic",
            65,
            False,
            True,
            "bmm_mul_mul",
        ),
        (
            "bmm_ab_trans_medium",
            "bmm",
            (4, 16, 64, 16),
            "basic",
            81,
            True,
            True,
            "bmm_mul_mul",
        ),
        (
            "bmm_small_n_full_load",
            "bmm",
            (64, 32, 64, 2),
            "iter_batch",
            257,
            False,
            False,
            "bmm_mul_mul",
        ),
        (
            "bmm_iter_batch_1v2_fixpipe",
            "bmm",
            (40, 47, 8, 129),
            "iter_batch_1v2_fixpipe",
            2097473,
            False,
            True,
            "bmm_mul_mul",
        ),
        (
            "bmm_high_level_iter_batch",
            "bmm",
            (100, 32, 32, 32),
            "high_level",
            256,
            False,
            False,
            "bmm_mul_mul",
        ),
        (
            "bmm_many_batch_iter",
            "bmm",
            (512, 4, 64, 16),
            "iter_batch",
            None,
            False,
            False,
            "bmm_mul_mul",
        ),
        (
            "bmm_many_tiny_batch_merge",
            "bmm",
            (1024, 1, 32, 8),
            "merge_batch",
            None,
            False,
            False,
            "bmm_mul_mul",
        ),
    ]
    cases: list[Case] = []
    for stem, kind, dims, template, tiling_key, a_trans, b_trans, op in specs:
        keys = ("b", "m", "k", "n") if kind == "bmm" else ("_b", "m", "k", "n")
        variant = {key: value for key, value in zip(keys, dims) if key != "_b"}
        shape = {key: value for key, value in variant.items()}
        for dynamic in (False, True):
            name = f"cv_{'dynamic' if dynamic else 'static'}_{stem}"
            cases.append(
                Case(
                    name,
                    kind,
                    dynamic,
                    "float16",
                    shape,
                    (variant,),
                    None,
                    None,
                    None,
                    2e-2,
                    2e-2,
                    900,
                    op=op,
                    family="template_coverage",
                    expect_cv=True,
                    expected_template=template,
                    expected_tiling_key=tiling_key,
                    a_trans=a_trans,
                    b_trans=b_trans,
                )
            )
    return cases


def _mixed_dtype_cases() -> list[Case]:
    cases = [
        Case(
            "cv_static_mixed_fp32_mm_cast_fp16_mul_add",
            "mixed_dtype",
            False,
            "float32",
            {
                "shapes": [[32, 64, 7], [32, 64, 48], [64, 64, 7]],
                "cast_dtype": "float16",
            },
            (
                {"m": 32, "k": 64, "n": 7},
                {"m": 32, "k": 64, "n": 48},
                {"m": 64, "k": 64, "n": 7},
            ),
            0,
            None,
            None,
            2e-2,
            2e-2,
            180,
            op="fp32_mm_cast_fp16_mul_add",
            family="mixed_dtype",
            expect_cv=True,
        ),
        Case(
            "cv_static_fp16_mm_mul_add_alignment",
            "mixed_dtype",
            False,
            "float16",
            {"shapes": [[32, 64, 7], [32, 64, 48], [64, 64, 7]]},
            (
                {"m": 32, "k": 64, "n": 7},
                {"m": 32, "k": 64, "n": 48},
                {"m": 64, "k": 64, "n": 7},
            ),
            0,
            None,
            None,
            2e-2,
            2e-2,
            600,
            op="fp16_mm_mul_add",
            family="mixed_dtype",
            expect_cv=True,
        ),
    ]
    for op_name, out_dtype in (("compare", "bool"), ("where", "float32")):
        for rhs_dtype in ("float16", "bfloat16"):
            for rhs_layout, rhs_shape in (("brc", [1, 4]), ("nobrc", [16, 4])):
                cases.append(
                    Case(
                        f"cv_static_mm_{op_name}_fp32_{rhs_dtype.replace('float', 'f')}_{rhs_layout}",
                        "mixed_dtype",
                        False,
                        out_dtype,
                        {
                            "m": 16,
                            "k": 18,
                            "n": 4,
                            "rhs_dtype": rhs_dtype,
                            "rhs_shape": rhs_shape,
                        },
                        ({"m": 16, "k": 18, "n": 4},),
                        0,
                        None,
                        None,
                        0.0 if op_name == "compare" else 1e-3,
                        0.0 if op_name == "compare" else 1e-3,
                        420,
                        op=f"mm_{op_name}_mixed",
                        family="mixed_dtype",
                        expect_cv=True,
                    )
                )
    return cases


def _extended_cases() -> list[Case]:
    return [
        Case(
            "cv_static_vv_large_relu_mul",
            "vv",
            False,
            "float32",
            {"x": [1025, 1024]},
            ({"m": 1025, "n": 1024},),
            None,
            None,
            None,
            1e-4,
            1e-4,
            240,
            op="vv_relu_mul",
            family="extended",
            mode_variant="vv",
        ),
        Case(
            "cv_dynamic_vv_tail_where",
            "vv",
            True,
            "float32",
            {"x": [31, 33]},
            ({"m": 31, "n": 33}, {"m": 1023, "n": 1025}),
            None,
            None,
            None,
            1e-4,
            1e-4,
            300,
            op="vv_where",
            family="extended",
            mode_variant="vv",
        ),
        Case(
            "cv_dynamic_vv_large_cast_chain",
            "vv",
            True,
            "float32",
            {"x": [1001, 257]},
            ({"m": 1001, "n": 257}, {"m": 1025, "n": 129}),
            None,
            None,
            None,
            1e-2,
            1e-2,
            300,
            op="vv_cast_chain",
            family="extended",
            mode_variant="vv",
        ),
        Case(
            "cv_dynamic_vv_boundary_arith",
            "vv",
            True,
            "float32",
            {"x": [32, 32]},
            ({"m": 32, "n": 32}, {"m": 1024, "n": 1025}),
            None,
            None,
            None,
            1e-4,
            1e-4,
            300,
            op="vv_arith",
            family="extended",
            mode_variant="vv",
        ),
        Case(
            "cv_static_mm_add_row_large_n1001",
            "mm_ext",
            False,
            "float32",
            {"m": 1024, "k": 64, "n": 1001, "bias": "row"},
            ({"m": 1024, "k": 64, "n": 1001},),
            0,
            None,
            None,
            1e-3,
            1e-3,
            600,
            op="mm_add_bias",
            family="extended",
            expect_cv=True,
            bias="row",
        ),
        Case(
            "cv_static_mm_add_col_large_m1025",
            "mm_ext",
            False,
            "float32",
            {"m": 1025, "k": 64, "n": 257, "bias": "col"},
            ({"m": 1025, "k": 64, "n": 257},),
            0,
            None,
            None,
            1e-3,
            1e-3,
            600,
            op="mm_add_bias",
            family="extended",
            expect_cv=True,
            bias="col",
        ),
        Case(
            "cv_dynamic_mm_add_row_large_tail",
            "mm_ext",
            True,
            "float32",
            {"m": 1001, "k": 64, "n": 257, "bias": "row"},
            ({"m": 1001, "k": 64, "n": 257}, {"m": 1025, "k": 64, "n": 513}),
            0,
            None,
            None,
            1e-3,
            1e-3,
            720,
            op="mm_add_bias",
            family="extended",
            expect_cv=True,
            bias="row",
        ),
        Case(
            "cv_static_mm_compare_col_m1025",
            "mm_ext",
            False,
            "bool",
            {"m": 1025, "k": 64, "n": 33, "bias": "col"},
            ({"m": 1025, "k": 64, "n": 33},),
            0,
            None,
            None,
            0.0,
            0.0,
            420,
            op="mm_compare_bias",
            family="extended",
            expect_cv=True,
            bias="col",
        ),
        Case(
            "cv_static_mm_where_full_n1025",
            "mm_ext",
            False,
            "float32",
            {"m": 65, "k": 64, "n": 1025, "bias": "full"},
            ({"m": 65, "k": 64, "n": 1025},),
            0,
            None,
            None,
            1e-3,
            1e-3,
            600,
            op="mm_where_full",
            family="extended",
            expect_cv=True,
            bias="full",
        ),
        Case(
            "cv_dynamic_mm_cast_fp16_large",
            "mm_ext",
            True,
            "float32",
            {"m": 1001, "k": 64, "n": 128},
            ({"m": 1001, "k": 64, "n": 128}, {"m": 1025, "k": 64, "n": 129}),
            0,
            None,
            None,
            1e-2,
            1e-2,
            600,
            op="mm_cast_fp16",
            family="extended",
            expect_cv=True,
        ),
        Case(
            "cv_static_mm_exp_n1025",
            "mm_ext",
            False,
            "float32",
            {"m": 129, "k": 64, "n": 1025},
            ({"m": 129, "k": 64, "n": 1025},),
            0,
            None,
            None,
            1e-3,
            1e-3,
            600,
            op="mm_exp",
            family="extended",
            expect_cv=True,
        ),
        Case(
            "cv_boundary_n1_m1025",
            "mm_ext",
            False,
            "float32",
            {"m": 1025, "k": 32, "n": 1, "bias": "row"},
            ({"m": 1025, "k": 32, "n": 1},),
            0,
            None,
            None,
            1e-3,
            1e-3,
            420,
            op="mm_add_bias",
            family="extended",
            expect_cv=True,
            bias="row",
        ),
        Case(
            "cv_boundary_n31_compare",
            "mm_ext",
            False,
            "bool",
            {"m": 1024, "k": 33, "n": 31, "bias": "row"},
            ({"m": 1024, "k": 33, "n": 31},),
            0,
            None,
            None,
            0.0,
            0.0,
            420,
            op="mm_compare_bias",
            family="extended",
            expect_cv=True,
            bias="row",
        ),
        Case(
            "cv_boundary_n33_where",
            "mm_ext",
            False,
            "float32",
            {"m": 33, "k": 1024, "n": 33, "bias": "full"},
            ({"m": 33, "k": 1024, "n": 33},),
            0,
            None,
            None,
            1e-3,
            1e-3,
            420,
            op="mm_where_full",
            family="extended",
            expect_cv=True,
            bias="full",
        ),
        Case(
            "cv_bmm_k1_static_boundary",
            "bmm_k1_mix",
            False,
            "float32",
            {"b": 128, "m": 8, "k": 1, "n": 16},
            ({"b": 128, "m": 8, "k": 1, "n": 16},),
            1,
            None,
            2,
            1e-3,
            1e-3,
            420,
            op="bmm_mul_bias",
            family="extended",
            expect_cv=True,
        ),
        Case(
            "cv_bmm_k1_dynamic_boundary",
            "bmm_k1_mix",
            True,
            "float32",
            {"b": 128, "m": 8, "k": 1, "n": 16},
            ({"b": 128, "m": 8, "k": 1, "n": 16}, {"b": 129, "m": 8, "k": 1, "n": 17}),
            1,
            None,
            2,
            1e-3,
            1e-3,
            420,
            op="bmm_mul_bias",
            family="extended",
            expect_cv=True,
        ),
    ]


def _with_vv_only_variants(cases: tuple[Case, ...]) -> tuple[Case, ...]:
    vv_only = [
        dataclasses.replace(
            case,
            name=f"vv_only_{case.name}",
            mode_variant="vv_only",
            expect_cv=False,
            fusion_mode=None,
            ub_mode=None,
            mix_mode=None,
        )
        for case in cases
        if case.expect_cv
    ]
    return cases + tuple(vv_only)


MATRIX: tuple[Case, ...] = _with_vv_only_variants(_base_matrix())

CV_KINDS = {"mm", "mm_fixpipe", "bmm", "bmm_k1_mix", "fragment08"}
REQUIRED_MATMUL_V3_MACROS = (
    "MAT_MUL_SLICE",
    "MAT_MUL_BASIC_SPLIT_K",
    "MAT_MUL_SK_SPLIT_K",
)
RUNTIME_ENV_KEYS = (
    "ASCEND_HOME_PATH",
    "ASCEND_AICPU_PATH",
    "ASCEND_OPP_PATH",
    "TOOLCHAIN_HOME",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "LD_PRELOAD",
    "ENABLE_TILING_SHIM",
    "TORCHINDUCTOR_NPU_BACKEND",
    "TORCHINDUCTOR_NPU_EXT_DEBUG",
)


def run_cmd(
    args: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def timeout_stream_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def runtime_env_snapshot() -> dict[str, str | None]:
    return {key: os.getenv(key) for key in RUNTIME_ENV_KEYS}


def load_npu_backend_package() -> tuple[str, Any]:
    errors: dict[str, str] = {}
    prefer_builtin = os.getenv("TORCHINDUCTOR_NPU_BACKEND") == "ascendc"
    if prefer_builtin:
        try:
            import torch_npu._inductor.ascendc as backend  # type: ignore

            return "torch_npu._inductor.ascendc", backend
        except Exception as exc:  # noqa: BLE001
            errors["torch_npu._inductor.ascendc"] = repr(exc)
    try:
        import inductor_npu_ext as backend  # type: ignore

        return "inductor_npu_ext", backend
    except Exception as exc:  # noqa: BLE001
        errors["inductor_npu_ext"] = repr(exc)
    if not prefer_builtin:
        try:
            import torch_npu._inductor.ascendc as backend  # type: ignore

            return "torch_npu._inductor.ascendc", backend
        except Exception as exc:  # noqa: BLE001
            errors["torch_npu._inductor.ascendc"] = repr(exc)
    raise ImportError(f"cannot load NPU Inductor backend package: {errors}")


def matmul_v3_header_probe(env: dict[str, str], cann: Path | None) -> dict[str, Any]:
    opp = (
        Path(env["ASCEND_OPP_PATH"]).expanduser()
        if env.get("ASCEND_OPP_PATH")
        else (cann / "opp" if cann else None)
    )
    header = (
        opp
        / "built-in"
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "impl"
        / "ops_nn"
        / "ascendc"
        / "mat_mul_v3"
        / "arch35"
        / "mat_mul_v3_tiling_key_public.h"
        if opp is not None
        else None
    )
    result: dict[str, Any] = {
        "path": str(header) if header is not None else None,
        "exists": bool(header is not None and header.is_file()),
        "required_macros": list(REQUIRED_MATMUL_V3_MACROS),
        "missing_macros": list(REQUIRED_MATMUL_V3_MACROS),
        "ok": False,
    }
    if header is None or not header.is_file():
        return result
    text = header.read_text(encoding="utf-8", errors="ignore")
    missing = [
        macro
        for macro in REQUIRED_MATMUL_V3_MACROS
        if re.search(rf"\b{re.escape(macro)}\b", text) is None
    ]
    result["missing_macros"] = missing
    result["ok"] = not missing
    return result


def resolve_repo_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    proc = run_cmd(["git", "rev-parse", "--show-toplevel"])
    if proc.returncode != 0:
        raise RuntimeError("cannot discover repo root; pass --repo-root")
    return Path(proc.stdout.strip()).resolve()


def normalize_torchair_experimental(
    value: str | None, repo_root: Path
) -> tuple[Path | None, Path | None, str]:
    candidates: list[tuple[Path, str]] = []
    if value:
        candidates.append((Path(value).expanduser(), "arg"))
    if os.getenv("TORCHAIR_EXPERIMENTAL"):
        candidates.append(
            (
                Path(os.environ["TORCHAIR_EXPERIMENTAL"]).expanduser(),
                "env:TORCHAIR_EXPERIMENTAL",
            )
        )
    sibling = repo_root.parent / "torchair" / "experimental"
    candidates.append((sibling, "repo-sibling"))
    for raw, source in candidates:
        base = raw.resolve()
        package_dir = base
        if base.name == "experimental":
            package_dir = base / "_inductor_npu_ext" / "python"
        elif base.name == "_inductor_npu_ext":
            package_dir = base / "python"
        if (package_dir / "inductor_npu_ext" / "__init__.py").is_file():
            return base, package_dir, source
    return None, None, "not-found"


def valid_cann_root(path: Path) -> bool:
    return (path / "set_env.sh").is_file() and (
        (path / "lib64").is_dir() or (path / "compiler").exists()
    )


def cann_candidates(explicit: str | None) -> list[dict[str, str]]:
    raw: list[tuple[Path, str]] = []
    if explicit:
        raw.append((Path(explicit).expanduser(), "arg"))
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "TOOLCHAIN_HOME"):
        val = os.getenv(key)
        if val:
            raw.append((Path(val).expanduser(), f"env:{key}"))
    for key in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH"):
        for part in os.getenv(key, "").split(os.pathsep):
            if part and ("Ascend" in part or "cann" in part.lower()):
                p = Path(part).expanduser()
                for parent in [p, *p.parents]:
                    raw.append((parent, f"env:{key}"))
    for default_path in (
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/cann",
        "/usr/local/Ascend/ascend-toolkit",
    ):
        raw.append((Path(default_path), "default"))
    seen: set[Path] = set()
    result: list[dict[str, str]] = []
    for path, source in raw:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(
            {
                "path": str(resolved),
                "source": source,
                "exists": str(resolved.exists()),
                "valid": str(valid_cann_root(resolved)),
            }
        )
    return result


def source_cann_env(cann: Path, base_env: dict[str, str]) -> dict[str, str]:
    set_env = cann / "set_env.sh"
    if not set_env.is_file():
        raise RuntimeError(f"CANN set_env.sh not found: {set_env}")
    script = f"source {str(set_env)!r} >/dev/null 2>&1 && env -0"
    bash_path = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run(
        [bash_path, "-lc", script],
        env=base_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace"))
    env: dict[str, str] = {}
    for item in proc.stdout.split(b"\0"):
        if not item:
            continue
        key, _, value = item.partition(b"=")
        env[key.decode()] = value.decode("utf-8", "replace")
    return env


def add_cann_host_include_env(env: dict[str, str], cann: Path | None) -> None:
    if cann is None:
        return
    include_paths = [
        cann / "x86_64-linux" / "asc" / "include" / "utils" / "tiling" / "platform",
        cann / "x86_64-linux" / "pkg_inc" / "base",
        cann / "x86_64-linux" / "include" / "base",
        cann / "x86_64-linux" / "include",
        cann
        / "opp"
        / "built-in"
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "impl"
        / "ops_nn"
        / "ascendc"
        / "common",
        cann
        / "opp"
        / "built-in"
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "impl"
        / "ops_nn"
        / "ascendc"
        / "mat_mul_v3",
        cann
        / "opp"
        / "built-in"
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "impl"
        / "ops_transformer"
        / "ascendc"
        / "3rd"
        / "mat_mul_v3"
        / "op_kernel",
    ]
    values = [str(path) for path in include_paths if path.is_dir()]
    for key in ("CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
        merged = list(values)
        if env.get(key):
            merged.append(env[key])
        if merged:
            env[key] = os.pathsep.join(merged)


def is_relative_to(path: Path, base: Path) -> bool:
    with contextlib.suppress(ValueError):
        path.resolve().relative_to(base.resolve())
        return True
    return False


def _is_foreign_cann_path(resolved: str) -> bool:
    lowered = resolved.lower()
    return "Ascend" in resolved or "cann" in lowered or "simulator" in lowered


def filter_path_list(
    value: str,
    repo_root: Path,
    cann: Path | None,
    autofuse_prefix: Path | None,
    allow_driver: bool = False,
) -> str:
    allowed_roots = [repo_root]
    if cann is not None:
        allowed_roots.append(cann)
    if autofuse_prefix is not None:
        allowed_roots.append(autofuse_prefix)
    kept: list[str] = []
    seen: set[str] = set()
    for item in value.split(os.pathsep):
        if not item:
            continue
        path = Path(item).expanduser()
        resolved = str(path.resolve())
        is_allowed = any(is_relative_to(path, root) for root in allowed_roots)
        if allow_driver and "/Ascend/driver" in resolved:
            is_allowed = True
        if not is_allowed and _is_foreign_cann_path(resolved):
            continue
        if resolved not in seen:
            kept.append(resolved)
            seen.add(resolved)
    return os.pathsep.join(kept)


def with_debug_option(value: str | None, option: str) -> str:
    options = [item for item in (value or "").split("+") if item]
    if option not in options:
        options.append(option)
    return "+".join(sorted(options))


def without_debug_option(value: str | None, option: str) -> str:
    return "+".join(
        item for item in (value or "").split("+") if item and item != option
    )


def clean_env(
    repo_root: Path,
    cann: Path | None,
    torchair_package: Path | None,
    run_dir: Path,
    autofuse_prefix: Path | None,
) -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "LD_PRELOAD",
        "ENABLE_TILING_SHIM",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCHINDUCTOR_NPU_EXT_CACHE_DIR",
        "TORCHINDUCTOR_DEBUG_DIR",
        "TORCH_COMPILE_DEBUG_DIR",
        "TRITON_CACHE_DIR",
    ):
        env.pop(key, None)
    if cann is not None:
        env = source_cann_env(cann, env)
        env.pop("LD_PRELOAD", None)
        env.pop("ENABLE_TILING_SHIM", None)
    add_cann_host_include_env(env, cann)
    if env.get("PATH"):
        env["PATH"] = filter_path_list(
            env["PATH"], repo_root, cann, autofuse_prefix, allow_driver=True
        )
    staging_site = (
        repo_root
        / "build"
        / "_CPack_Packages"
        / "makeself_staging"
        / "python"
        / "site-packages"
    )
    py_paths = []
    if torchair_package is not None:
        py_paths.append(str(torchair_package))
    if autofuse_prefix is not None:
        prefix_site = autofuse_prefix / "python" / "site-packages"
        if prefix_site.is_dir():
            py_paths.append(str(prefix_site))
    if staging_site.is_dir():
        py_paths.append(str(staging_site))
    py_paths.append(str(repo_root))
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    if py_paths:
        env["PYTHONPATH"] = filter_path_list(
            os.pathsep.join(py_paths), repo_root, cann, autofuse_prefix
        )
    ld_paths = []
    for path in (
        autofuse_prefix / "lib64" if autofuse_prefix is not None else None,
        repo_root / "build" / "autofuse",
        repo_root
        / "build"
        / "_CPack_Packages"
        / "makeself_staging"
        / "x86_64-linux"
        / "lib64",
    ):
        if path is not None and path.is_dir():
            ld_paths.append(str(path))
    if env.get("LD_LIBRARY_PATH"):
        ld_paths.append(env["LD_LIBRARY_PATH"])
    if ld_paths:
        env["LD_LIBRARY_PATH"] = filter_path_list(
            os.pathsep.join(ld_paths),
            repo_root,
            cann,
            autofuse_prefix,
            allow_driver=True,
        )
    env["RUN_DIR"] = str(run_dir)
    env.setdefault("TORCHINDUCTOR_NPU_BACKEND", "ascendc")
    env["TORCHINDUCTOR_NPU_EXT_DEBUG"] = with_debug_option(
        env.get("TORCHINDUCTOR_NPU_EXT_DEBUG"), "matmul"
    )
    env.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
    return env


def prepare_case_env(
    base_env: dict[str, str], case: Case, run_dir: Path
) -> dict[str, str]:
    case_dir = run_dir / "cases" / case.name
    env = dict(base_env)
    env["RUN_DIR"] = str(run_dir)
    env["TORCH_COMPILE_DEBUG"] = "1"
    env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    env["TORCHINDUCTOR_CACHE_DIR"] = str(case_dir / "cache" / "inductor")
    env["TORCHINDUCTOR_NPU_EXT_CACHE_DIR"] = str(case_dir / "cache")
    env["TORCHINDUCTOR_DEBUG_DIR"] = str(case_dir / "torch_compile_debug")
    env["TORCH_COMPILE_DEBUG_DIR"] = str(case_dir / "torch_compile_debug")
    env["TRITON_CACHE_DIR"] = str(case_dir / "cache" / "triton")
    env["AUTOFUSE_DFX_FLAGS"] = (
        f"--codegen_compile_debug=true;--debug_dir={case_dir / 'autofuse_dump'}"
    )
    env["TORCHINDUCTOR_NPU_EXT_TILING_DEBUG"] = "1"
    env["TORCHINDUCTOR_NPU_EXT_DISABLE_TOPN_TILING"] = "1"
    env["ASCEND_LAUNCH_BLOCKING"] = "1"
    if case.mode_variant == "vv_only":
        env["TORCHINDUCTOR_NPU_EXT_DEBUG"] = without_debug_option(
            env.get("TORCHINDUCTOR_NPU_EXT_DEBUG"), "matmul"
        )
    elif case.expect_cv:
        env["TORCHINDUCTOR_NPU_EXT_DEBUG"] = with_debug_option(
            env.get("TORCHINDUCTOR_NPU_EXT_DEBUG"), "matmul"
        )
    return env


@dataclasses.dataclass
class EnvReportContext:
    repo_root: Path
    cann: Path | None
    torchair_base: Path | None
    torchair_package: Path | None
    run_dir: Path
    explicit_cann: str | None
    autofuse_prefix: Path | None


def collect_env_report(ctx: EnvReportContext) -> dict[str, Any]:
    repo_root = ctx.repo_root
    cann = ctx.cann
    torchair_base = ctx.torchair_base
    torchair_package = ctx.torchair_package
    run_dir = ctx.run_dir
    explicit_cann = ctx.explicit_cann
    autofuse_prefix = ctx.autofuse_prefix
    git_head = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_status = run_cmd(
        ["git", "status", "--short", "--untracked-files=all"], cwd=repo_root
    )
    git_diff = run_cmd(["git", "diff", "--binary"], cwd=repo_root)
    candidates = cann_candidates(explicit_cann)
    selected = str(cann.resolve()) if cann else None
    build_artifacts = {
        "libaihac_codegen": repo_root / "build" / "autofuse" / "libaihac_codegen.so",
        "pyautofuse": repo_root
        / "build"
        / "autofuse"
        / "compiler"
        / "py_module"
        / "pyautofuse.so",
        "staging_site_packages": (
            repo_root
            / "build"
            / "_CPack_Packages"
            / "makeself_staging"
            / "python"
            / "site-packages"
            / "autofuse"
            / "pyautofuse.so"
        ),
        "staging_libaihac_codegen": (
            repo_root
            / "build"
            / "_CPack_Packages"
            / "makeself_staging"
            / "x86_64-linux"
            / "lib64"
            / "libaihac_codegen.so"
        ),
        "staging_bisheng": (
            repo_root
            / "build"
            / "_CPack_Packages"
            / "makeself_staging"
            / "tools"
            / "bisheng_compiler"
            / "bin"
            / "bisheng"
        ),
    }
    if autofuse_prefix is not None:
        build_artifacts.update(
            {
                "prefix_pyautofuse": autofuse_prefix
                / "python"
                / "site-packages"
                / "autofuse"
                / "pyautofuse.so",
                "prefix_libaihac_codegen": autofuse_prefix
                / "lib64"
                / "libaihac_codegen.so",
                "prefix_bisheng": autofuse_prefix
                / "tools"
                / "bisheng_compiler"
                / "bin"
                / "bisheng",
            }
        )
    report = {
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "git_head": git_head.stdout.strip() if git_head.returncode == 0 else None,
        "git_status": git_status.stdout.splitlines(),
        "git_diff_sha256": hashlib.sha256(git_diff.stdout.encode()).hexdigest()
        if git_diff.returncode == 0
        else None,
        "cann_selected": selected,
        "cann_candidates": candidates,
        "torchair_experimental": str(torchair_base) if torchair_base else None,
        "torchair_package": str(torchair_package) if torchair_package else None,
        "autofuse_prefix": str(autofuse_prefix) if autofuse_prefix else None,
        "build_artifacts": {
            name: {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path),
            }
            for name, path in build_artifacts.items()
        },
        "env_keys": {k: os.getenv(k) for k in RUNTIME_ENV_KEYS},
    }
    return report


def missing_build_artifacts(repo_root: Path, autofuse_prefix: Path | None) -> list[str]:
    required = [
        repo_root / "build" / "autofuse" / "libaihac_codegen.so",
        repo_root / "build" / "autofuse" / "compiler" / "py_module" / "pyautofuse.so",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if autofuse_prefix is not None:
        runtime_required = [
            autofuse_prefix / "python" / "site-packages" / "autofuse" / "pyautofuse.so",
            autofuse_prefix / "lib64" / "libaihac_codegen.so",
            autofuse_prefix / "tools" / "bisheng_compiler" / "bin" / "bisheng",
        ]
        missing.extend(str(path) for path in runtime_required if not path.exists())
        return missing
    staging_bisheng = (
        repo_root
        / "build"
        / "_CPack_Packages"
        / "makeself_staging"
        / "tools"
        / "bisheng_compiler"
        / "bin"
        / "bisheng"
    )
    if not staging_bisheng.exists():
        missing.append(
            str(staging_bisheng)
            + " (run package is built but not installed; install/overlay it and pass --autofuse-prefix)"
        )
    return missing


def import_probe(probe_env: dict[str, str]) -> dict[str, Any]:
    code = r"""
import importlib
import json
import os
import sys
result = {}
def load_npu_backend_package():
    errors = {}
    prefer_builtin = os.getenv("TORCHINDUCTOR_NPU_BACKEND") == "ascendc"
    if prefer_builtin:
        try:
            import torch_npu._inductor.ascendc as backend
            return "torch_npu._inductor.ascendc", backend
        except Exception as exc:
            errors["torch_npu._inductor.ascendc"] = repr(exc)
    try:
        import inductor_npu_ext as backend
        return "inductor_npu_ext", backend
    except Exception as exc:
        errors["inductor_npu_ext"] = repr(exc)
    if not prefer_builtin:
        try:
            import torch_npu._inductor.ascendc as backend
            return "torch_npu._inductor.ascendc", backend
        except Exception as exc:
            errors["torch_npu._inductor.ascendc"] = repr(exc)
    raise ImportError(f"cannot load NPU Inductor backend package: {errors}")
try:
    import torch
    result["torch"] = {
        "ok": True,
        "version": getattr(torch, "__version__", None),
        "file": getattr(torch, "__file__", None),
    }
    try:
        import torch_npu
        result["torch_npu"] = {
            "ok": True,
            "file": getattr(torch_npu, "__file__", None),
            "npu_available": bool(torch.npu.is_available()) if hasattr(torch, "npu") else False,
        }
    except Exception as exc:
        result["torch_npu"] = {"ok": False, "error": repr(exc)}
    try:
        import autofuse
        result["autofuse"] = {"ok": True, "file": getattr(autofuse, "__file__", None)}
    except Exception as exc:
        result["autofuse"] = {"ok": False, "error": repr(exc)}
    try:
        import autofuse.compile_adapter as compile_adapter
        result["autofuse_compile_adapter"] = {"ok": True, "file": getattr(compile_adapter, "__file__", None)}
    except Exception as exc:
        result["autofuse_compile_adapter"] = {"ok": False, "error": repr(exc)}
    try:
        backend_kind, backend = load_npu_backend_package()
        result["npu_backend_package"] = {"ok": True, "kind": backend_kind, "file": getattr(backend, "__file__", None)}
        try:
            config = importlib.import_module(backend_kind + ".config")
            result["npu_backend_package"].update({
                "soc": config.get_cached_soc_version(),
                "enable_matmul_fuse": config.enable_matmul_fuse,
                "disable_canfuse": config.disable_canfuse,
            })
        except Exception as exc:
            result["npu_backend_package"]["config_error"] = repr(exc)
    except Exception as exc:
        result["npu_backend_package"] = {"ok": False, "error": repr(exc)}
except Exception as exc:
    result["torch"] = {"ok": False, "error": repr(exc)}
sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=probe_env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    try:
        data = (
            json.loads(proc.stdout.strip().splitlines()[-1])
            if proc.stdout.strip()
            else {}
        )
    except Exception:  # noqa: BLE001
        data = {}
    data["probe_exit_code"] = proc.returncode
    if proc.stderr:
        data["probe_stderr"] = proc.stderr[-4000:]
    return data


def dtype_from_name(torch: Any, name: str) -> Any:
    return getattr(torch, name)


def make_inputs(
    torch: Any,
    case: Case,
    variant: dict[str, int],
    device: str,
    seed: int,
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    def rand_dtype(dtype_name: str, *shape: int) -> Any:
        return torch.randn(
            *shape, generator=gen, dtype=dtype_from_name(torch, dtype_name)
        ).to(device)

    def rand(*shape: int) -> Any:
        return rand_dtype(case.dtype if case.dtype != "bool" else "float32", *shape)

    def apply_transpose(tensor: Any, transposed: bool) -> Any:
        return tensor.transpose(-1, -2) if transposed else tensor

    def bias_for(m: int, n: int, bias_kind: str, dtype_name: str | None = None) -> Any:
        dtype_name = dtype_name or (case.dtype if case.dtype != "bool" else "float32")
        if bias_kind == "col":
            return rand_dtype(dtype_name, m, 1)
        if bias_kind == "full":
            return rand_dtype(dtype_name, m, n)
        return rand_dtype(dtype_name, 1, n)

    if case.kind == "vv":
        m, n = variant["m"], variant["n"]

        def fn(x: Any, y: Any, z: Any) -> Any:
            if case.op == "vv_relu_mul":
                return torch.relu(x + y) * z
            if case.op == "vv_where":
                return torch.where(x > y, x - y, x + y)
            if case.op == "vv_cast_chain":
                return (x + y).to(torch.float16).to(torch.float32) * 0.5
            if case.op == "vv_arith":
                return (x * 1.25 + y * 0.75) / 2.0
            return (x + y) * z

        return fn, (rand(m, n), rand(m, n), rand(m, n))
    if case.kind == "non_fused":
        m, n, k = variant["m"], variant["n"], variant["k"]

        def fn(a: Any, b: Any) -> Any:
            return torch.mm(a, b)

        return fn, (rand(m, k), rand(k, n))
    if case.kind in {"mm", "mm_fixpipe", "mm_ext"}:
        m, n, k = variant["m"], variant["n"], variant["k"]

        def fn(a: Any, b: Any, bias: Any) -> Any:
            out = torch.mm(a, b)
            if case.kind == "mm_fixpipe":
                return torch.relu(out) * bias
            if case.op == "mm_add_bias":
                return out + bias
            if case.op == "mm_mul_bias":
                return out * bias
            if case.op == "mm_compare_bias":
                return out > bias
            if case.op == "mm_where_full":
                return torch.where(out > bias, out, bias)
            if case.op == "mm_cast_fp16":
                return out.to(torch.float16)
            if case.op == "mm_exp":
                return torch.exp(out)
            return (out + bias) * 0.5

        a = apply_transpose(rand(k, m) if case.a_trans else rand(m, k), case.a_trans)
        b = apply_transpose(rand(n, k) if case.b_trans else rand(k, n), case.b_trans)
        return fn, (a, b, bias_for(m, n, case.bias))
    if case.kind in {"bmm", "bmm_k1_mix"}:
        bsz, m, k, n = variant["b"], variant["m"], variant["k"], variant["n"]

        def fn(a: Any, b: Any, bias: Any) -> Any:
            out = torch.bmm(a, b) + bias
            return (
                out * 0.5
                if case.kind == "bmm_k1_mix" or case.op == "bmm_mul_bias"
                else out
            )

        a = apply_transpose(
            rand(bsz, k, m) if case.a_trans else rand(bsz, m, k), case.a_trans
        )
        b = apply_transpose(
            rand(bsz, n, k) if case.b_trans else rand(bsz, k, n), case.b_trans
        )
        return fn, (a, b, rand(bsz, m, n))
    if case.kind == "mixed_dtype":
        m, k, n = variant["m"], variant["k"], variant["n"]

        def fn(x: Any, w: Any, scale: Any, bias: Any) -> Any:
            out = torch.mm(x, w)
            if case.op == "fp32_mm_cast_fp16_mul_add":
                return out.to(torch.float16) * scale + bias
            if case.op == "fp16_mm_mul_add":
                return out * scale + bias
            if case.op == "mm_compare_mixed":
                return out > bias
            if case.op == "mm_where_mixed":
                return torch.where(out > bias, out, bias)
            raise ValueError(case.op)

        if case.op in {"fp32_mm_cast_fp16_mul_add", "fp16_mm_mul_add"}:
            x_dtype = "float32" if case.op.startswith("fp32") else "float16"
            return fn, (
                rand_dtype(x_dtype, m, k),
                rand_dtype(x_dtype, k, n),
                rand_dtype("float16", 1, n),
                rand_dtype("float16", 1, n),
            )
        rhs_dtype = str(case.shape.get("rhs_dtype", "float16"))
        rhs_shape = case.shape.get("rhs_shape", [1, n])
        return fn, (
            rand_dtype("float32", m, k),
            rand_dtype("float32", k, n),
            rand_dtype(rhs_dtype, *rhs_shape),
            rand_dtype(rhs_dtype, *rhs_shape),
        )
    if case.kind == "fragment08":
        m, k, n = variant["m"], variant["k"], variant["n"]

        def fn(x: Any, w: Any, bias: Any) -> Any:
            return torch.exp(torch.matmul(x, w) + bias)

        return fn, (rand(m, k), rand(k, n), rand(1, n))
    raise ValueError(case.kind)


def tensor_metrics(
    torch: Any, actual: Any, expected: Any, atol: float, rtol: float
) -> dict[str, Any]:
    if isinstance(actual, (tuple, list)):
        actual = actual[0]
        expected = expected[0]
    a = actual.detach().cpu()
    e = expected.detach().cpu()
    if (
        getattr(a, "dtype", None) == torch.bool
        or getattr(e, "dtype", None) == torch.bool
    ):
        mismatch = a != e
        fail_count = int(mismatch.sum().item()) if mismatch.numel() else 0
        max_idx = (
            int(mismatch.reshape(-1).to(torch.int64).argmax().item())
            if mismatch.numel()
            else 0
        )
        return {
            "max_abs_diff": 1.0 if fail_count else 0.0,
            "mean_abs_diff": float(fail_count / mismatch.numel())
            if mismatch.numel()
            else 0.0,
            "max_rel_diff": 1.0 if fail_count else 0.0,
            "fail_count": fail_count,
            "argmax_abs_diff_flat": max_idx,
            "actual_at_argmax": bool(a.reshape(-1)[max_idx].item())
            if a.numel()
            else False,
            "expected_at_argmax": bool(e.reshape(-1)[max_idx].item())
            if e.numel()
            else False,
            "ok": fail_count == 0,
        }
    diff = (a - e).abs()
    rel = diff / e.abs().clamp_min(1e-12)
    mismatch = diff > (atol + rtol * e.abs())
    max_idx = int(diff.argmax().item()) if diff.numel() else 0
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel_diff": float(rel.max().item()) if rel.numel() else 0.0,
        "fail_count": int(mismatch.sum().item()) if mismatch.numel() else 0,
        "argmax_abs_diff_flat": max_idx,
        "actual_at_argmax": float(a.reshape(-1)[max_idx].item()) if a.numel() else 0.0,
        "expected_at_argmax": float(e.reshape(-1)[max_idx].item())
        if e.numel()
        else 0.0,
        "ok": bool(not mismatch.any().item()) if mismatch.numel() else True,
    }


def scan_artifacts(case_dir: Path) -> dict[str, list[str]]:
    patterns = {
        "asc_kernel": "**/asc_kernel.py",
        "wrapper": "**/inductor_wrapper.cpp",
        "kernel_so": "**/kernel.so",
        "logs": "*.log",
        "dump_dirs": "autofuse_dump",
        "debug_dirs": "torch_compile_debug",
    }
    found: dict[str, list[str]] = {}
    for key, pattern in patterns.items():
        found[key] = [str(p) for p in case_dir.glob(pattern)]
    return found


def scan_cv_artifacts(case_dir: Path) -> dict[str, list[str]]:
    wrappers: list[str] = []
    for wrapper in case_dir.glob("**/inductor_wrapper.cpp"):
        with contextlib.suppress(Exception):
            if "INDUCTOR_CV_FUSION" in wrapper.read_text(errors="ignore")[:200000]:
                wrappers.append(str(wrapper))
    return {"wrapper": wrappers}


def extract_modes(case_dir: Path) -> dict[str, int | None]:
    modes = extract_modes_from_kernel(case_dir)
    if any(value is not None for value in modes.values()):
        return modes
    text = ""
    for pattern in ("**/inductor_wrapper.cpp", "**/*.log", "**/*.json", "**/*.py"):
        for path in case_dir.glob(pattern):
            with contextlib.suppress(Exception):
                text += "\n" + path.read_text(errors="ignore")[:200000]
    modes: dict[str, int | None] = {
        "fusion_mode": None,
        "ub_mode": None,
        "mix_mode": None,
    }
    for key in modes:
        matches = re.findall(rf"cv_tiling_data\.{key}\s*=\s*(-?\d+)\s*;", text)
        unique_values = {int(value) for value in matches}
        if len(unique_values) == 1:
            modes[key] = unique_values.pop()
    return modes


def extract_modes_from_kernel(case_dir: Path) -> dict[str, int | None]:
    modes: dict[str, int | None] = {
        "fusion_mode": None,
        "ub_mode": None,
        "mix_mode": None,
    }
    cv_wrapper = False
    for wrapper in case_dir.glob("**/inductor_wrapper.cpp"):
        with contextlib.suppress(Exception):
            if "INDUCTOR_CV_FUSION" in wrapper.read_text(errors="ignore")[:200000]:
                cv_wrapper = True
                break
    if not cv_wrapper:
        return modes
    try:
        import ctypes

        # 布局必须与 codegen 生成的 autofuse_tiling_data.h 保持一致：
        # block_dim/tiling_key 已收窄为 uint16_t，workspace* 为 uint64_t，
        # 且 corenum/hbm_size 已删除。
        class AutofuseTilingData(ctypes.Structure):
            _fields_ = [
                ("block_dim", ctypes.c_uint16),
                ("tiling_key", ctypes.c_uint16),
                ("ub_size", ctypes.c_uint32),
                ("workspace0", ctypes.c_uint64),
                ("a0a1t_size", ctypes.c_uint32),
                ("a0a1Tb_size", ctypes.c_uint32),
                ("q0_size", ctypes.c_uint32),
                ("q1_size", ctypes.c_uint32),
            ]

        class CVTilingData(ctypes.Structure):
            _fields_ = [
                ("fusion_mode", ctypes.c_uint8),
                ("ub_mode", ctypes.c_uint8),
                ("cv_aic_num", ctypes.c_uint8),
                ("cv_aiv_num", ctypes.c_uint8),
                # 生成侧为 uint64_t，mix_mode 因此位于 offset 16。
                ("cv_vec_wss", ctypes.c_uint64),
                ("mix_mode", ctypes.c_uint8),
            ]

        # 生成侧前缀：union（tiling_data/cube_tiling_key/cv_tiling_data）后依次为
        # stage_size_name、cube_ub_stage_size、cube_m_stage_size，
        # matmul_tiling_data 以 alignas(8) 起始，故前缀固定 80 字节。
        class CVTilingDataPrefix(ctypes.Structure):
            _fields_ = [
                ("tiling_data", AutofuseTilingData),
                ("cube_tiling_key", ctypes.c_uint64),
                ("cv_tiling_data", CVTilingData),
                ("stage_size_name", ctypes.c_uint32),
                ("cube_ub_stage_size", ctypes.c_uint32),
                ("cube_m_stage_size", ctypes.c_uint32),
            ]

        prefix_size = ctypes.sizeof(CVTilingDataPrefix)

        def make_tiling_struct(total_size: int) -> type[ctypes.Structure] | None:
            # matmul_tiling_data 长度取决于生成侧 MATMUL_TILING_DATA_STORAGE_SIZE，
            # 随 CANN 版本变化；按 GetTilingDataSize() 的真实总长动态补齐，
            # 并用 uint64 元素保证 8 字节对齐，避免越界写。
            matmul_bytes = total_size - prefix_size
            if matmul_bytes <= 0 or matmul_bytes % 8 != 0:
                return None

            class CVAutofuseTilingData(ctypes.Structure):
                _fields_ = CVTilingDataPrefix._fields_ + [
                    ("matmul_tiling_data", ctypes.c_uint64 * (matmul_bytes // 8)),
                ]

            return CVAutofuseTilingData

        for kernel in case_dir.glob("**/kernel.so"):
            with contextlib.suppress(Exception):
                lib = ctypes.CDLL(str(kernel))
                get_tiling_size = getattr(lib, "GetTilingDataSize", None)
                if callable(get_tiling_size):
                    get_tiling_size.restype = ctypes.c_size_t
                    tiling_size = int(get_tiling_size())
                else:
                    tiling_size = 0
                tiling_struct = make_tiling_struct(tiling_size)
                if tiling_struct is None:
                    # 生成头布局与脚本静态布局不一致时跳过，避免按错误偏移读取
                    continue
                fn = lib.AutofuseTiling
                fn.argtypes = [
                    ctypes.POINTER(tiling_struct),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_void_p,
                ]
                fn.restype = ctypes.c_int64
                tiling_data = tiling_struct()
                workspace_size = ctypes.c_uint32()
                block_dim = ctypes.c_uint32()
                if (
                    fn(
                        ctypes.byref(tiling_data),
                        ctypes.byref(workspace_size),
                        ctypes.byref(block_dim),
                        None,
                    )
                    == 0
                ):
                    fusion_mode = int(tiling_data.cv_tiling_data.fusion_mode)
                    ub_mode = int(tiling_data.cv_tiling_data.ub_mode)
                    if fusion_mode not in (0, 1) or ub_mode not in (0, 1):
                        # 偏移不一致时读到非法值，跳过而不是返回错误模式
                        continue
                    return {
                        "fusion_mode": fusion_mode,
                        "ub_mode": ub_mode,
                        "mix_mode": int(tiling_data.cv_tiling_data.mix_mode),
                    }
    except Exception:  # noqa: BLE001
        return modes
    return modes


def validate_case_result(case: Case, result: dict[str, Any]) -> None:
    if not result.get("ok"):
        return
    errors: list[str] = []
    artifacts = result.get("artifact_paths") or {}
    cv_artifacts = result.get("cv_artifacts") or {}
    if case.expect_cv:
        missing = [
            key
            for key in ("asc_kernel", "wrapper", "kernel_so", "dump_dirs")
            if not artifacts.get(key)
        ]
        if missing:
            errors.append("missing artifacts: " + ", ".join(missing))
        if not any(cv_artifacts.get(key) for key in cv_artifacts):
            errors.append("missing CV artifact")
    elif case.mode_variant == "vv_only":
        if any(cv_artifacts.get(key) for key in cv_artifacts):
            errors.append("unexpected CV artifact in vv_only case")
        unexpected_modes = [
            key
            for key in ("fusion_mode", "ub_mode", "mix_mode")
            if result.get(key) is not None
        ]
        if unexpected_modes:
            errors.append(
                "unexpected CV mode in vv_only case: " + ", ".join(unexpected_modes)
            )
    for key, expected in (
        ("fusion_mode", case.fusion_mode),
        ("ub_mode", case.ub_mode),
        ("mix_mode", case.mix_mode),
    ):
        if expected is None:
            continue
        actual = result.get(key)
        if actual is None:
            errors.append(f"missing {key}: expected {expected}")
        elif actual != expected:
            errors.append(f"unexpected {key}: expected {expected}, got {actual}")
    if errors:
        result["ok"] = False
        result["structure_ok"] = False
        result["stage"] = "summary"
        result["error"] = "; ".join(errors)
        result["exit_code"] = 1
    else:
        result["structure_ok"] = True


def should_stop_after_case(result: dict[str, Any], fail_fast: bool) -> bool:
    return fail_fast and not result.get("ok")


def case_progress_line(
    status: str, index: int, total: int, case: Case, timeout: int
) -> str:
    return f"[{index}/{total}] {status} {case.name} timeout={timeout}s"


def child_case(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    case = next(c for c in MATRIX if c.name == args.child_case)
    case_dir = run_dir / "cases" / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    result: dict[str, Any] = {
        "name": case.name,
        "kind": case.kind,
        "family": case.family,
        "mode_variant": case.mode_variant,
        "op": case.op,
        "dynamic": case.dynamic,
        "dtype": case.dtype,
        "shape": case.shape,
        "expect_cv": case.expect_cv,
        "expected_template": case.expected_template,
        "expected_tiling_key": case.expected_tiling_key,
        "seed": args.seed,
        "fusion_mode": case.fusion_mode,
        "ub_mode": case.ub_mode,
        "mix_mode": case.mix_mode,
        "atol": case.atol,
        "rtol": case.rtol,
        "stage": "env",
        "case_dir": str(case_dir),
        "command": " ".join(sys.argv),
        "start_time": start,
        "exit_code": 1,
        "artifact_paths": {},
        "cv_artifacts": {},
        "numeric_ok": False,
        "structure_ok": False,
        "runtime_env": runtime_env_snapshot(),
    }
    try:
        import torch  # type: ignore
        import torch_npu  # noqa: F401  # type: ignore

        backend_kind, backend = load_npu_backend_package()
        result["npu_backend_package"] = {
            "kind": backend_kind,
            "file": getattr(backend, "__file__", None),
        }

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("torch.npu is not available")
        device = "npu"
        dynamo = getattr(torch, "_dynamo")
        dynamo.reset()
        compiled = None
        all_ok = True
        metrics: list[dict[str, Any]] = []
        strides: list[Any] = []
        for idx, variant in enumerate(case.variants):
            fn, inputs = make_inputs(torch, case, variant, device, args.seed + idx)
            strides.append([tuple(x.stride()) for x in inputs if hasattr(x, "stride")])
            if compiled is None:
                result["stage"] = "fusion_codegen"
                compiled = torch.compile(fn, dynamic=case.dynamic, fullgraph=True)
            with torch.no_grad():
                expected = fn(*inputs)
                actual = compiled(*inputs)
                result["stage"] = "launch_sync"
                torch.npu.synchronize()
            m = tensor_metrics(torch, actual, expected, case.atol, case.rtol)
            m["variant"] = variant
            all_ok = all_ok and bool(m["ok"])
            metrics.append(m)
        artifacts = scan_artifacts(case_dir)
        modes = extract_modes(case_dir)
        result.update(modes)
        result["stride"] = strides
        result["metrics"] = metrics
        result["max_abs_diff"] = max((m["max_abs_diff"] for m in metrics), default=0.0)
        result["mean_abs_diff"] = max(
            (m["mean_abs_diff"] for m in metrics), default=0.0
        )
        result["max_rel_diff"] = max((m["max_rel_diff"] for m in metrics), default=0.0)
        result["fail_count"] = sum(int(m["fail_count"]) for m in metrics)
        result["artifact_paths"] = artifacts
        result["cv_artifacts"] = scan_cv_artifacts(case_dir)
        result["ok"] = all_ok
        result["numeric_ok"] = all_ok
        result["stage"] = "summary" if all_ok else "numeric"
        result["exit_code"] = 0 if all_ok else 1
        validate_case_result(case, result)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        error_text = repr(exc) + "\n" + tb
        result["error"] = repr(exc)
        result["traceback"] = tb
        if (
            "asc_kernel.py" in error_text
            or "CompileError" in error_text
            or "build_ascend_lib" in error_text
        ):
            result["stage"] = "ascendc_compile"
        elif "Kernel load failed" in error_text or "undefined symbol" in error_text:
            result["stage"] = "kernel_load"
        elif "torch.npu.synchronize" in error_text or "Launch" in error_text:
            result["stage"] = "launch_sync"
        result["artifact_paths"] = scan_artifacts(case_dir)
        result["cv_artifacts"] = scan_cv_artifacts(case_dir)
        result["ok"] = False
        result["numeric_ok"] = False
        result["exit_code"] = 1
    finally:
        result["end_time"] = time.time()
        json_dump(case_dir / "result.json", result)
        _LOGGER.info("CASE_RESULT " + json.dumps(result, sort_keys=True))
    return int(result["exit_code"])


def selected_cases(case_filter: str | None) -> list[Case]:
    if not case_filter:
        return list(MATRIX)
    wanted = {item.strip() for item in case_filter.split(",") if item.strip()}
    result: list[Case] = []
    for case in MATRIX:
        case_keys = {case.name, case.kind, case.family, case.mode_variant}
        if case_keys & wanted:
            result.append(case)
    return result


def write_summary(
    run_dir: Path,
    env_report: dict[str, Any],
    case_results: list[dict[str, Any]],
    final: bool,
) -> None:
    summary = {"env": env_report, "cases": case_results, "final_matrix": final}
    json_dump(run_dir / "logs" / "summary.json", summary)
    lines = [
        "### Inductor CV Validation Result",
        "",
        f"- CANN Toolkit: {env_report.get('cann_selected')}",
        f"- Run dir: {run_dir}",
        f"- TorchAir package: {env_report.get('torchair_package')}",
        "",
        "### Matrix",
        "",
        "| Case | Dynamic | Kind | Result | Stage | Evidence |",
        "|------|---------|------|--------|-------|----------|",
    ]
    by_name = {r.get("name"): r for r in case_results}
    for case in MATRIX:
        r = by_name.get(case.name)
        if r is None:
            lines.append(
                f"| {case.name} | {case.dynamic} | {case.kind} | NOT_RUN | summary | missing |"
            )
            continue
        result = "PASS" if r.get("ok") else "FAIL"
        lines.append(
            f"| {case.name} | {case.dynamic} | {case.kind} | {result} | {r.get('stage')} | {r.get('case_dir')} |"
        )
    if final and (
        len(case_results) != len(MATRIX) or any(not r.get("ok") for r in case_results)
    ):
        conclusion = "未完成验证: full matrix has missing or failed cases."
    elif final:
        conclusion = "Inductor CV fusion 验证通过."
    else:
        conclusion = "未完成验证: subset/dry-run only."
    lines.extend(["", "### Conclusion", "", f"- {conclusion}", ""])
    (run_dir / "logs" / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def copy_self(run_dir: Path) -> None:
    dst = run_dir / "script" / "run_inductor_cv_matrix.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run portable TorchInductor NPU CV validation matrix"
    )
    parser.add_argument("--repo-root")
    parser.add_argument("--cann-toolkit")
    parser.add_argument("--torchair-experimental")
    parser.add_argument(
        "--autofuse-prefix",
        help="Installed graph_autofusion/overlay root containing python/site-packages, lib64, and tools",
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--case-filter")
    parser.add_argument(
        "--mode", choices=("env-probe", "dry-run", "run"), default="env-probe"
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--allow-multiple-cann",
        action="store_true",
        help="Only for env investigation; never use for final validation",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed selected case; for debugging only",
    )
    parser.add_argument("--child-case", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child_case:
        return child_case(args)

    repo_root = resolve_repo_root(args.repo_root)
    autofuse_prefix = (
        Path(args.autofuse_prefix).expanduser().resolve()
        if args.autofuse_prefix
        else None
    )
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else (repo_root / "temp" / "inductor_cv_validation" / timestamp).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    copy_self(run_dir)

    torchair_base, torchair_package, torchair_source = normalize_torchair_experimental(
        args.torchair_experimental,
        repo_root,
    )
    candidates = cann_candidates(args.cann_toolkit)
    valid = [c for c in candidates if c.get("valid") == "True"]
    selected_cann: Path | None = None
    env_errors: list[str] = []
    if args.cann_toolkit:
        selected_cann = Path(args.cann_toolkit).expanduser().resolve()
        if not valid_cann_root(selected_cann):
            env_errors.append(f"explicit CANN is invalid: {selected_cann}")
    elif len(valid) == 1:
        selected_cann = Path(valid[0].get("path", "")).resolve()
    elif len(valid) > 1:
        env_errors.append("multiple valid CANN candidates; pass --cann-toolkit")
    else:
        env_errors.append("no valid CANN candidate; pass --cann-toolkit")
    if torchair_package is None:
        env_errors.append(
            "TorchAir experimental package not found; pass --torchair-experimental"
        )

    env_report = collect_env_report(
        EnvReportContext(
            repo_root=repo_root,
            cann=selected_cann,
            torchair_base=torchair_base,
            torchair_package=torchair_package,
            run_dir=run_dir,
            explicit_cann=args.cann_toolkit,
            autofuse_prefix=autofuse_prefix,
        )
    )
    env_report["argv"] = sys.argv
    env_report["args"] = vars(args)
    env_report["torchair_source"] = torchair_source
    env_report["env_errors"] = env_errors
    probe_env = dict(os.environ)
    for key in ("LD_PRELOAD", "ENABLE_TILING_SHIM"):
        probe_env.pop(key, None)
    if selected_cann is not None and valid_cann_root(selected_cann):
        with contextlib.suppress(Exception):
            probe_env = source_cann_env(selected_cann, probe_env)
            probe_env.pop("LD_PRELOAD", None)
            probe_env.pop("ENABLE_TILING_SHIM", None)
    add_cann_host_include_env(probe_env, selected_cann)
    if torchair_package is not None:
        py_paths = [str(torchair_package)]
        if (
            autofuse_prefix is not None
            and (autofuse_prefix / "python" / "site-packages").is_dir()
        ):
            py_paths.append(str(autofuse_prefix / "python" / "site-packages"))
        staging_site = (
            repo_root
            / "build"
            / "_CPack_Packages"
            / "makeself_staging"
            / "python"
            / "site-packages"
        )
        if staging_site.is_dir():
            py_paths.append(str(staging_site))
        if probe_env.get("PYTHONPATH"):
            py_paths.append(probe_env["PYTHONPATH"])
        probe_env["PYTHONPATH"] = filter_path_list(
            os.pathsep.join(py_paths), repo_root, selected_cann, autofuse_prefix
        )
    if probe_env.get("PATH"):
        probe_env["PATH"] = filter_path_list(
            probe_env["PATH"],
            repo_root,
            selected_cann,
            autofuse_prefix,
            allow_driver=True,
        )
    probe_ld_paths = []
    for path in (
        autofuse_prefix / "lib64" if autofuse_prefix is not None else None,
        repo_root / "build" / "autofuse",
        repo_root
        / "build"
        / "_CPack_Packages"
        / "makeself_staging"
        / "x86_64-linux"
        / "lib64",
    ):
        if path is not None and path.is_dir():
            probe_ld_paths.append(str(path))
    if probe_env.get("LD_LIBRARY_PATH"):
        probe_ld_paths.append(probe_env["LD_LIBRARY_PATH"])
    if probe_ld_paths:
        probe_env["LD_LIBRARY_PATH"] = filter_path_list(
            os.pathsep.join(probe_ld_paths),
            repo_root,
            selected_cann,
            autofuse_prefix,
            allow_driver=True,
        )
    probe_env.setdefault("TORCHINDUCTOR_NPU_BACKEND", "ascendc")
    probe_env["TORCHINDUCTOR_NPU_EXT_DEBUG"] = with_debug_option(
        probe_env.get("TORCHINDUCTOR_NPU_EXT_DEBUG"), "matmul"
    )
    env_report["probe_env_keys"] = {k: probe_env.get(k) for k in RUNTIME_ENV_KEYS}
    env_report["matmul_v3_header_probe"] = matmul_v3_header_probe(
        probe_env, selected_cann
    )
    selected = selected_cases(args.case_filter)
    if (
        args.mode == "run"
        and any(case.expect_cv for case in selected)
        and not env_report["matmul_v3_header_probe"].get("ok")
    ):
        missing = ", ".join(
            env_report["matmul_v3_header_probe"].get("missing_macros") or []
        )
        path = env_report["matmul_v3_header_probe"].get("path")
        env_errors.append(
            f"MatMulV3 header missing required macros: {missing} at {path}"
        )
    env_report["env_errors"] = env_errors
    env_report["import_probe"] = import_probe(probe_env)
    env_report["probe_env_removed"] = {
        "LD_PRELOAD": os.getenv("LD_PRELOAD"),
        "ENABLE_TILING_SHIM": os.getenv("ENABLE_TILING_SHIM"),
    }
    json_dump(run_dir / "logs" / "env.log", env_report)

    if args.mode in {"env-probe", "dry-run"}:
        dry_cases = [dataclasses.asdict(c) for c in selected]
        json_dump(run_dir / "logs" / "matrix_plan.json", dry_cases)
        write_summary(run_dir, env_report, [], final=False)
        _LOGGER.info(
            json.dumps(
                {
                    "mode": args.mode,
                    "run_dir": str(run_dir),
                    "env_errors": env_errors,
                    "case_count": len(dry_cases),
                },
                sort_keys=True,
            )
        )
        return 2 if env_errors and args.mode == "env-probe" else 0

    if env_errors:
        write_summary(run_dir, env_report, [], final=True)
        _LOGGER.info(
            json.dumps(
                {
                    "mode": args.mode,
                    "run_dir": str(run_dir),
                    "env_errors": env_errors,
                },
                sort_keys=True,
            )
        )
        return 2

    missing_artifacts = missing_build_artifacts(repo_root, autofuse_prefix)
    if missing_artifacts:
        env_report["env_errors"] = [
            *env_errors,
            "missing build artifacts: " + ", ".join(missing_artifacts),
        ]
        json_dump(run_dir / "logs" / "env.log", env_report)
        write_summary(run_dir, env_report, [], final=True)
        _LOGGER.info(
            json.dumps(
                {
                    "mode": args.mode,
                    "run_dir": str(run_dir),
                    "env_errors": env_report["env_errors"],
                },
                sort_keys=True,
            )
        )
        return 2

    clean = clean_env(
        repo_root, selected_cann, torchair_package, run_dir, autofuse_prefix
    )
    case_results: list[dict[str, Any]] = []
    cases = selected
    final = args.case_filter is None
    total_cases = len(cases)
    for index, case in enumerate(cases, start=1):
        case_dir = run_dir / "cases" / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        env = prepare_case_env(clean, case, run_dir)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-case",
            case.name,
            "--run-dir",
            str(run_dir),
            "--seed",
            str(args.seed),
        ]
        start = time.time()
        case_timeout = min(args.timeout, case.timeout)
        _LOGGER.info(
            case_progress_line("START", index, total_cases, case, case_timeout)
        )
        try:
            proc = subprocess.run(
                command,
                env=env,
                text=True,
                capture_output=True,
                timeout=case_timeout,
                check=False,
            )
            (case_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
            (case_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
            match = re.findall(r"CASE_RESULT (\{.*\})", proc.stdout)
            if match:
                result = json.loads(match[-1])
            else:
                result = {
                    "name": case.name,
                    "ok": False,
                    "stage": "summary",
                    "case_dir": str(case_dir),
                    "error": "missing CASE_RESULT",
                }
            result["exit_code"] = proc.returncode
        except subprocess.TimeoutExpired as exc:
            result = {
                "name": case.name,
                "ok": False,
                "stage": "launch_sync",
                "case_dir": str(case_dir),
                "error": "timeout",
                "timeout": case_timeout,
                "command": " ".join(command),
            }
            (case_dir / "stdout.log").write_text(
                timeout_stream_to_text(exc.stdout), encoding="utf-8"
            )
            (case_dir / "stderr.log").write_text(
                timeout_stream_to_text(exc.stderr), encoding="utf-8"
            )
        result.setdefault("start_time", start)
        result.setdefault("end_time", time.time())
        result.setdefault("artifact_paths", scan_artifacts(case_dir))
        case_results.append(result)
        elapsed = time.time() - start
        status = "PASS" if result.get("ok") else "FAIL"
        _LOGGER.info(
            f"{case_progress_line(status, index, total_cases, case, case_timeout)}"
            f" elapsed={elapsed:.1f}s stage={result.get('stage')}"
        )
        if should_stop_after_case(result, args.fail_fast):
            break
    write_summary(run_dir, env_report, case_results, final=final)
    _LOGGER.info(
        json.dumps(
            {
                "mode": args.mode,
                "run_dir": str(run_dir),
                "passed": sum(1 for r in case_results if r.get("ok")),
                "total_run": len(case_results),
                "final_matrix": final,
            },
            sort_keys=True,
        )
    )
    return (
        0
        if final
        and len(case_results) == len(MATRIX)
        and all(r.get("ok") for r in case_results)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
