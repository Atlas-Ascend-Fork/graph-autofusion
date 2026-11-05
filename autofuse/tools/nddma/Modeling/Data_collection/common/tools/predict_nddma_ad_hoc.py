#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Predict one ad-hoc NDDMA config with the documented unified model."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
MODEL_PATH = REPO_ROOT / "Modeling" / "Ana" / "round3" / "round3_multidim_model.json"
MODEL_SOURCE = str(MODEL_PATH.relative_to(REPO_ROOT))
BASE_MODEL_PATH = (
    REPO_ROOT / "Modeling" / "Ana" / "round1" / "round1_1d_single_multi_core_model.json"
)
BASE_MODEL_SOURCE = str(BASE_MODEL_PATH.relative_to(REPO_ROOT))


DTYPE_ALIASES = {
    "int8_t": "int8_t",
    "int8": "int8_t",
    "i8": "int8_t",
    "b8": "int8_t",
    "int16_t": "int16_t",
    "int16": "int16_t",
    "i16": "int16_t",
    "b16": "int16_t",
    "int32_t": "int32_t",
    "int32": "int32_t",
    "i32": "int32_t",
    "b32": "int32_t",
    "int64_t": "int64_t",
    "int64": "int64_t",
    "i64": "int64_t",
    "b64": "int64_t",
}
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "results_analysis" / "ad_hoc_formula"
DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}


def warn_snapshot(model: dict[str, object], source: Path) -> None:
    if model.get("snapshot", {}).get("metrics_valid") is False:
        warnings.warn(
            f"{source.name}: historical snapshot; metrics_valid=false. "
            "Stored metrics do not validate current predictions; refit before reporting accuracy.",
            UserWarning,
            stacklevel=2,
        )


def load_model() -> dict[str, object]:
    with MODEL_PATH.open(encoding="utf-8") as file_obj:
        model = json.load(file_obj)
    warn_snapshot(model, MODEL_PATH)
    return model


MODEL = load_model()
with BASE_MODEL_PATH.open(encoding="utf-8") as file_obj:
    BASE_MODEL = json.load(file_obj)
warn_snapshot(BASE_MODEL, BASE_MODEL_PATH)
ONE_DIMENSIONAL_PARAMETERS = BASE_MODEL["parameters"]
CORRECTION_PARAMETERS = MODEL["parameters"]


def parse_vector(text: str) -> list[int]:
    values = [
        int(part)
        for part in text.replace(",", "x").replace(";", "x").split("x")
        if part
    ]
    if not values:
        raise argparse.ArgumentTypeError("vector must not be empty")
    return values


def parse_positive_vector(text: str) -> list[int]:
    values = parse_vector(text)
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all vector entries must be positive")
    return values


def parse_non_negative_vector(text: str) -> list[int]:
    values = parse_vector(text)
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("all vector entries must be non-negative")
    return values


def parse_dtype(text: str) -> str:
    key = text.strip().lower()
    if key in DTYPE_ALIASES:
        return DTYPE_ALIASES[key]
    supported = ", ".join(sorted(DTYPE_ALIASES))
    raise argparse.ArgumentTypeError(
        f"unsupported dtype {text!r}; supported: {supported}"
    )


def join_vector(values: Sequence[int]) -> str:
    return "x".join(str(value) for value in values)


def product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def one_dimensional_base_and_t1(
    dtype: str, byte_value: float, input_stride: int, output_stride: int, block_dim: int
) -> tuple[float, float]:
    """Return ``(N_base, N_1')`` from the documented one-dimensional model."""

    parameters = ONE_DIMENSIONAL_PARAMETERS[dtype]
    dtype_size = float(DTYPE_SIZES[dtype])
    byte_value = float(byte_value)
    stride = min(float(input_stride) * dtype_size, 128.0)
    output_gap_indicator = min(1.0, max(0.0, float(output_stride) - 1.0))
    base_params = parameters.get("base", parameters)
    correction_params = CORRECTION_PARAMETERS[dtype]

    suffix = "1" if block_dim <= 2 else "2"
    base = byte_value / float(base_params[f"T_{suffix}"]) + float(
        base_params[f"H_{suffix}"]
    )

    n_g_params = correction_params["N_G"]
    n_gu_params = correction_params["N_GU"]
    n_g = (float(n_g_params["a1"]) + float(n_g_params["a2"]) * byte_value) * stride
    n_gu = (
        (float(n_gu_params["b1"]) + float(n_gu_params["b2"]) * stride)
        + (float(n_gu_params["b3"]) + float(n_gu_params["b4"]) * stride) * byte_value
    ) * output_gap_indicator
    correction = n_g + n_gu
    if block_dim > 2:
        rho_params = correction_params["rho"]
        rho = (
            float(rho_params["c1"])
            + float(rho_params["c2"]) * stride
            + output_gap_indicator
            * (float(rho_params["c3"]) + float(rho_params["c4"]) * stride)
        )
        correction *= rho
    return float(base), float(correction)


def predict(
    output_dims: Sequence[int],
    input_stride: Sequence[int],
    output_stride: Sequence[int],
    block_dim: int,
    dtype: str,
    *,
    kernel_repeat: int,
    formula_source: str = MODEL_SOURCE,
) -> dict[str, object]:
    dim = len(output_dims)
    if dim < 1 or dim > 5:
        raise ValueError("unified model supports dim in [1, 5]")
    if len(input_stride) != dim or len(output_stride) != dim:
        raise ValueError(
            "output_dims, input_stride, and output_stride must have the same length"
        )
    if any(int(value) <= 0 for value in output_dims):
        raise ValueError("output_dims values must be positive")
    if any(int(value) < 0 for value in input_stride):
        raise ValueError("input_stride values must be non-negative")
    if any(int(value) < 0 for value in output_stride):
        raise ValueError("output_stride values must be non-negative")
    if dtype not in DTYPE_SIZES:
        raise ValueError(f"unsupported dtype: {dtype!r}")
    if block_dim <= 0:
        raise ValueError("block_dim must be positive")
    if kernel_repeat <= 0:
        raise ValueError("kernel_repeat must be positive")

    dtype_size = DTYPE_SIZES[dtype]
    loop_sizes = tuple(reversed(tuple(int(value) for value in output_dims)))
    input_stride_by_loop_axis = tuple(
        reversed(tuple(int(value) for value in input_stride))
    )
    output_stride_by_loop_axis = tuple(
        reversed(tuple(int(value) for value in output_stride))
    )
    total_elems = product(loop_sizes)
    data_volume_bytes = total_elems * dtype_size
    n_base, _axis0_t1 = one_dimensional_base_and_t1(
        dtype,
        data_volume_bytes,
        input_stride_by_loop_axis[0],
        output_stride_by_loop_axis[0],
        block_dim,
    )

    terms = []
    inner_product = 1
    for axis, loop_size in enumerate(loop_sizes):
        term_elems = total_elems // inner_product
        term_bytes = term_elems * dtype_size
        if axis == 0:
            input_delta = input_stride_by_loop_axis[0]
            output_delta = output_stride_by_loop_axis[0]
            expected_input_stride = 0
            expected_output_stride = 0
        else:
            expected_input_stride = sum(
                int(loop_sizes[lower]) * int(input_stride_by_loop_axis[lower])
                for lower in range(axis)
            )
            expected_output_stride = sum(
                int(loop_sizes[lower]) * int(output_stride_by_loop_axis[lower])
                for lower in range(axis)
            )
            input_delta = (
                abs(int(input_stride_by_loop_axis[axis]) - expected_input_stride) + 1
            )
            output_delta = (
                abs(int(output_stride_by_loop_axis[axis]) - expected_output_stride) + 1
            )
        _term_base, t1 = one_dimensional_base_and_t1(
            dtype, term_bytes, input_delta, output_delta, block_dim
        )
        terms.append(
            {
                "axis": axis,
                "loop_size": int(loop_size),
                "term_elems": int(term_elems),
                "data_volume_bytes": int(term_bytes),
                "input_stride": int(input_stride_by_loop_axis[axis]),
                "output_stride": int(output_stride_by_loop_axis[axis]),
                "expected_contiguous_input_stride": int(expected_input_stride),
                "expected_contiguous_output_stride": int(expected_output_stride),
                "input_stride_delta": int(input_delta),
                "output_stride_delta": int(output_delta),
                "s_bytes": float(min(input_delta * dtype_size, 128)),
                "output_gap_indicator": float(min(1, int(output_delta) - 1)),
                "n1_cycles_per_kernel_repeat": float(t1),
                "n1_cycles": float(t1 * kernel_repeat),
            }
        )
        inner_product *= int(loop_size)

    t1_total = sum(float(term["n1_cycles"]) for term in terms)
    t1_per_kernel_repeat = sum(
        float(term["n1_cycles_per_kernel_repeat"]) for term in terms
    )
    predicted_per_kernel_repeat = float(n_base + t1_per_kernel_repeat)
    predicted = float(predicted_per_kernel_repeat * kernel_repeat)
    return {
        "formula": (
            "N=N_base(B)+sum_k N_1'(B/prod_{j<k}ls_j,"
            "|is_k-sum_{j<k}ls_j*is_j|+1,"
            "|os_k-sum_{j<k}ls_j*os_j|+1)"
        ),
        "formula_source": formula_source,
        "one_dimensional_formula_source": BASE_MODEL_SOURCE,
        "dtype": dtype,
        "dtype_size": dtype_size,
        "dim": dim,
        "block_dim": int(block_dim),
        "kernel_repeat": int(kernel_repeat),
        "api_order": {
            "output_dims": join_vector(output_dims),
            "input_stride": join_vector(input_stride),
            "output_stride": join_vector(output_stride),
        },
        "loop_axis_order": {
            "loop_sizes": join_vector(loop_sizes),
            "input_stride": join_vector(input_stride_by_loop_axis),
            "output_stride": join_vector(output_stride_by_loop_axis),
        },
        "total_elems": int(total_elems),
        "data_volume_bytes": int(data_volume_bytes),
        "n_base_cycles_per_kernel_repeat": float(n_base),
        "n_base_cycles": float(n_base * kernel_repeat),
        "n1_total_cycles_per_kernel_repeat": float(t1_per_kernel_repeat),
        "n1_total_cycles": float(t1_total),
        "predicted_cycles_per_kernel_repeat": predicted_per_kernel_repeat,
        "predicted_cycles": predicted,
        "terms": terms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict one NDDMA config with the documented unified "
            "N_base+sum(N_1') formula. "
            "Vectors are in NDDMA API order, for example: 63x2 2x1 3x1 72 b64."
        )
    )
    parser.add_argument("output_dims", type=parse_positive_vector)
    parser.add_argument("input_stride", type=parse_non_negative_vector)
    parser.add_argument("output_stride", type=parse_non_negative_vector)
    parser.add_argument("block_dim", type=int)
    parser.add_argument(
        "dtype_positional",
        nargs="?",
        type=parse_dtype,
        help="Optional dtype positional argument: b8/b16/b32/b64 or int*_t.",
    )
    parser.add_argument(
        "--dtype",
        type=parse_dtype,
        default=None,
        help="Dtype used when the optional positional dtype is omitted.",
    )
    parser.add_argument(
        "--actual-cycles",
        type=float,
        default=None,
        help="Optional measured cycles; prints residual diagnostics.",
    )
    parser.add_argument(
        "--kernel-repeat",
        type=int,
        default=200,
        help=(
            "Kernel inner-loop repeat count. Prediction is multiplied by this "
            "value, matching run_nddma_ad_hoc.py total-cycle output."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for prediction.json.",
    )
    parser.add_argument(
        "--no-write-json",
        action="store_true",
        help="Only print results; do not write prediction.json.",
    )
    args = parser.parse_args()
    if args.block_dim <= 0:
        parser.error("block_dim must be positive")
    if args.kernel_repeat <= 0:
        parser.error("--kernel-repeat must be positive")
    if args.dtype_positional and args.dtype and args.dtype_positional != args.dtype:
        parser.error("positional dtype and --dtype disagree")
    args.dtype = args.dtype_positional or args.dtype or "int64_t"
    if args.actual_cycles is not None and not math.isfinite(args.actual_cycles):
        parser.error("--actual-cycles must be finite")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = predict(
            args.output_dims,
            args.input_stride,
            args.output_stride,
            args.block_dim,
            args.dtype,
            kernel_repeat=args.kernel_repeat,
            formula_source=MODEL_SOURCE,
        )
    except (TypeError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 2

    if args.actual_cycles is not None:
        predicted = float(result["predicted_cycles"])
        actual = float(args.actual_cycles)
        result["actual_cycles"] = actual
        result["error_actual_minus_predicted"] = actual - predicted
        result["relative_error_actual_minus_predicted_over_actual"] = (
            (actual - predicted) / actual if actual else None
        )

    if not args.no_write_json:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "prediction.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[INFO] wrote prediction JSON: {args.output_dir / 'prediction.json'}")

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    print(f"[RESULT] predicted_cycles={float(result['predicted_cycles']):.12g}")
    print(
        "[RESULT] predicted_cycles_per_kernel_repeat="
        f"{float(result['predicted_cycles_per_kernel_repeat']):.12g}"
    )
    print(f"[RESULT] n_base_cycles={float(result['n_base_cycles']):.12g}")
    if "n1_total_cycles" in result:
        print(f"[RESULT] n1_total_cycles={float(result['n1_total_cycles']):.12g}")
    if args.actual_cycles is not None:
        print(
            "[RESULT] error_actual_minus_predicted="
            f"{float(result['error_actual_minus_predicted']):.12g}"
        )
        relative = result["relative_error_actual_minus_predicted_over_actual"]
        print(
            f"[RESULT] relative_error={relative:.12g}"
            if relative is not None
            else "[RESULT] relative_error=N/A"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
