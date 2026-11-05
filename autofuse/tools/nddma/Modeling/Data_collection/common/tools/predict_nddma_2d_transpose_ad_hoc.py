#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Predict one ad-hoc NDDMA 2D config with the documented Round5 N_G2 model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import predict_nddma_ad_hoc as unified_ad_hoc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "results_analysis" / "ad_hoc_round5_ng2"
MODEL_PATH = (
    REPO_ROOT / "Modeling" / "Ana" / "round4" / "round4_2d_ub_contiguous_ng2_model.json"
)
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
DTYPE_SIZES = unified_ad_hoc.DTYPE_SIZES


def load_model() -> dict[str, object]:
    with MODEL_PATH.open(encoding="utf-8") as file_obj:
        model = json.load(file_obj)
    unified_ad_hoc.warn_snapshot(model, MODEL_PATH)
    return model


MODEL = load_model()
ROUND4_NG2_PARAMETERS = MODEL["parameters"]["N_G2"]
ROUND4_CORRECTION_PARAMETERS = MODEL["parameters"]["one_dimensional"]
with BASE_MODEL_PATH.open(encoding="utf-8") as file_obj:
    BASE_MODEL = json.load(file_obj)
ROUND4_ONE_DIMENSIONAL_PARAMETERS = BASE_MODEL["parameters"]


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


def one_d_base(dtype: str, bytes_value: float, block_dim: int) -> float:
    params = ROUND4_ONE_DIMENSIONAL_PARAMETERS[dtype]
    params = params.get("base", params)
    suffix = "1" if block_dim <= 2 else "2"
    return float(bytes_value) / float(params[f"T_{suffix}"]) + float(
        params[f"H_{suffix}"]
    )


def one_d_g1(
    dtype: str, bytes_value: float, input_stride: int, block_dim: int
) -> float:
    params = ROUND4_CORRECTION_PARAMETERS[dtype]
    s = min(float(input_stride) * float(DTYPE_SIZES[dtype]), 128.0)
    ng = (
        float(params["N_G"]["a1"]) + float(params["N_G"]["a2"]) * float(bytes_value)
    ) * s
    if block_dim <= 2:
        return ng
    rho = params["rho"]
    multiplier = float(rho["c1"]) + float(rho["c2"]) * s
    return multiplier * ng


def predict_ng2(dtype: str, m: int, is2: int) -> float:
    params = ROUND4_NG2_PARAMETERS[dtype]
    return (
        (float(params["g10"]) + float(params["g11_M"]) * float(m)) * float(is2)
        + float(params["g00"])
        + float(params["g01_M"]) * float(m)
    )


def predict(
    output_dims: Sequence[int],
    input_stride: Sequence[int],
    output_stride: Sequence[int],
    block_dim: int,
    dtype: str,
    *,
    kernel_repeat: int,
    formula_source: str = MODEL_SOURCE,
    one_dimensional_formula_source: str = MODEL_SOURCE,
) -> dict[str, object]:
    dim = len(output_dims)
    if dim != 2:
        raise ValueError("round5 ng2 ad hoc model supports dim=2 only")
    if len(input_stride) != 2 or len(output_stride) != 2:
        raise ValueError(
            "output_dims, input_stride, and output_stride must have length 2"
        )
    if any(int(value) <= 0 for value in output_dims):
        raise ValueError("output_dims values must be positive")
    if any(int(value) < 0 for value in input_stride):
        raise ValueError("input_stride values must be non-negative")
    if any(int(value) < 0 for value in output_stride):
        raise ValueError("output_stride values must be non-negative")
    if block_dim <= 0:
        raise ValueError("block_dim must be positive")
    if kernel_repeat <= 0:
        raise ValueError("kernel_repeat must be positive")
    if dtype not in DTYPE_SIZES:
        raise ValueError(f"unsupported dtype: {dtype!r}")

    m, n = (int(output_dims[0]), int(output_dims[1]))
    is2, is1 = (int(input_stride[0]), int(input_stride[1]))
    os2, os1 = (int(output_stride[0]), int(output_stride[1]))
    if is2 >= is1:
        raise ValueError("expected input_stride=[is2,is1] with is2 < is1")
    if os2 != n or os1 != 1:
        raise ValueError("expected output_stride=[N,1] for the Round5 N_G2 model")

    dtype_size = DTYPE_SIZES[dtype]
    model_values = dict(ROUND4_NG2_PARAMETERS[dtype])
    data_volume_bytes = m * n * dtype_size
    b1 = n * dtype_size
    s_bytes = min(is1 * dtype_size, 128)
    n_base = one_d_base(dtype, data_volume_bytes, block_dim)
    n_g1 = one_d_g1(dtype, b1, is1, block_dim)
    n_g2 = predict_ng2(dtype, m, is2)
    predicted_per_repeat = n_base + n_g1 * n_g2
    predicted = predicted_per_repeat * kernel_repeat
    token = (
        f"r5ng2_adhoc_{dtype}_k{block_dim}"
        f"_shape{join_vector(output_dims)}_in{join_vector(input_stride)}"
        f"_out{join_vector(output_stride)}"
    )
    return {
        "formula": (
            "N2_hat=N_base(B,k)+N_1'(B1,is1,1,k)*N_G2(M,is2); "
            "N_G2=(g10+g11_M*M)*is2+g00+g01_M*M"
        ),
        "formula_source": formula_source,
        "one_dimensional_formula_source": one_dimensional_formula_source,
        "model_scope": (
            "[M,N]/[is2,is1]/[N,1]; N_base uses the Round1 d/T+H "
            "parameters, while N_G1 and N_G2 use the Round4 parameters from Ana/round4"
        ),
        "token": token,
        "config_id": token,
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
        "m": m,
        "n": n,
        "is2": is2,
        "is1": is1,
        "data_volume_bytes": int(data_volume_bytes),
        "b1_bytes": int(b1),
        "s_bytes": int(s_bytes),
        "n_base_cycles_per_kernel_repeat": float(n_base),
        "n_g1_cycles_per_kernel_repeat": float(n_g1),
        "n1_prime_cycles_per_kernel_repeat": float(n_g1),
        "n_gu1_cycles_per_kernel_repeat": 0.0,
        "n_gw1_cycles_per_kernel_repeat": 0.0,
        "n_g2": float(n_g2),
        "n_g2_multiplier": float(n_g2),
        "predicted_cycles_per_kernel_repeat": float(predicted_per_repeat),
        "predicted_cycles": float(predicted),
        "ng2_parameters": {
            key: float(model_values[key]) for key in ("g10", "g11_M", "g00", "g01_M")
        },
        "terms": [
            {"name": "N_base", "value": float(n_base)},
            {
                "name": "N_1_prime",
                "value": float(n_g1),
                "B1": int(b1),
                "s_bytes": int(s_bytes),
            },
            {"name": "N_G2", "value": float(n_g2)},
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict one ad-hoc NDDMA config with the documented Round5 N_G2 model."
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
    parser.add_argument("--kernel-repeat", type=int, default=200)
    parser.add_argument(
        "--actual-cycles",
        type=float,
        default=None,
        help="Optional measured cycles; prints residual diagnostics.",
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
            one_dimensional_formula_source=BASE_MODEL_SOURCE,
        )
    except (KeyError, TypeError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 2

    if args.actual_cycles is not None:
        actual = float(args.actual_cycles)
        predicted = float(result["predicted_cycles"])
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
    print(f"[RESULT] n_g2={float(result['n_g2']):.12g}")
    print(
        "[RESULT] predicted_cycles_per_kernel_repeat="
        f"{float(result['predicted_cycles_per_kernel_repeat']):.12g}"
    )
    print(
        f"[RESULT] n_base_cycles={float(result['n_base_cycles_per_kernel_repeat']):.12g}"
    )
    print(f"[RESULT] n_g1_cycles={float(result['n_g1_cycles_per_kernel_repeat']):.12g}")
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
