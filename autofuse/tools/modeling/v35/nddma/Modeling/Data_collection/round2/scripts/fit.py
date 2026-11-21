#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round2"
DEFAULT_DATA_DIR = DEFAULT_ANA_DIR / "collection"
DEFAULT_ROUND1_MODEL = (
    MODELING_DIR / "Ana" / "round1" / "round1_1d_single_multi_core_model.json"
)
MODEL_FILENAME = "round2_1d_noncontiguous_model.json"
PREDICTIONS_FILENAME = "round2_1d_noncontiguous_predictions.csv"
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
METRIC_FIELDS = (
    "actual_y",
    "nddma_mte2_cycles_per_block",
    "mte2_cycles_per_block",
    "mte2_cycles",
    "nddma_mte2_cycles",
)
LOWER_BRANCH = "le2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the NDDMA2 Round2 1D non-contiguous staged model."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--measurement-csv", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_ANA_DIR))
    parser.add_argument("--round1-model", default=str(DEFAULT_ROUND1_MODEL))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def actual_value(row: dict[str, str]) -> float:
    for field in METRIC_FIELDS:
        text = str(row.get(field, "")).strip()
        if not text:
            continue
        value = float(text)
        if field != "actual_y":
            value /= max(1.0, float(row.get("repeat") or 1.0))
        if math.isfinite(value) and value > 0:
            return value
    raise ValueError(f"no positive measurement found for {row.get('token', '')}")


def find_files(data_dir: Path) -> list[Path]:
    direct = data_dir / "measurements.csv"
    if direct.exists():
        return [direct]
    files = sorted(data_dir.rglob("profiling_with_params_mean.csv"))
    return files or sorted(data_dir.rglob("profiling_with_params.csv"))


def aggregate(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[dict[str, str], float]]] = defaultdict(list)
    for row in rows:
        if row.get("dim") != "1":
            continue
        token = row.get("config_id") or row.get("token") or ""
        grouped[token].append((row, actual_value(row)))
    result = []
    for values in grouped.values():
        row = values[0][0]
        measurements = sorted(value for _, value in values)
        middle = len(measurements) // 2
        actual = (
            measurements[middle]
            if len(measurements) % 2
            else (measurements[middle - 1] + measurements[middle]) / 2.0
        )
        result.append(
            {
                "token": row.get("token", ""),
                "group": row.get("data_split") or row.get("group_id", ""),
                "dtype": row["dtype"],
                "block_dim": int(row["block_dim"]),
                "bytes_per_core": float(row["bytes_per_core"]),
                "logical_total_bytes": float(row["logical_total_bytes"]),
                "input_stride": float(row["input_stride"]),
                "output_stride": float(row["output_stride"]),
                "actual": actual,
            }
        )
    return result


def branch(block_dim: int) -> str:
    return "le2" if block_dim <= 2 else "gt2"


def solve(
    matrix: list[list[float]], target: list[float], *, scaled: bool = True
) -> list[float]:
    """Match HW single-core scaled OLS and multicore unscaled OLS."""
    design = np.asarray(matrix, dtype=float)
    values = np.asarray(target, dtype=float)
    if design.ndim != 2 or len(design) < design.shape[1]:
        raise ValueError("not enough samples for linear fit")
    scale = np.max(np.abs(design), axis=0) if scaled else np.ones(design.shape[1])
    scale = np.where(scale > 1e-15, scale, 1.0)
    coeff, _, rank, _ = np.linalg.lstsq(
        design / scale, values, rcond=1e-12 if scaled else None
    )
    if rank < design.shape[1]:
        raise ValueError("rank-deficient linear fit")
    return (coeff / scale).tolist()


def fit_single_core(
    dtype_rows: list[dict[str, object]], dtype: str, base: dict[str, object]
) -> tuple[list[float], list[float]]:
    """HW Round4 canonical N_G/N_GU stages, with frozen continuous Base."""
    rows = [
        r
        for r in dtype_rows
        if r["block_dim"] == 1 and r["logical_total_bytes"] < 50000
    ]
    size = DTYPE_SIZES[dtype]
    by_bytes = defaultdict(list)
    for r in rows:
        if r["group"] == "A" and r["input_stride"] * size < 128:
            by_bytes[r["bytes_per_core"]].append(r)
    slope_points = []
    for b, group in sorted(by_bytes.items()):
        strides = np.asarray([r["input_stride"] * size for r in group])
        residual = np.asarray(
            [r["actual"] - predict_base(base, dtype, 1, b) for r in group]
        )
        slope_points.append((b, float((strides @ residual) / (strides @ strides))))
    ng = solve([[1.0, b] for b, _ in slope_points], [v for _, v in slope_points])

    coefficient_points = []
    for stride in (2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        groups = defaultdict(list)
        for r in rows:
            if (
                r["group"] == "C"
                and r["input_stride"] == stride
                and r["output_stride"] == 2
            ):
                b = r["bytes_per_core"]
                residual = (
                    r["actual"]
                    - predict_base(base, dtype, 1, b)
                    - (ng[0] + ng[1] * b) * min(stride * size, 128)
                )
                groups[b].append(residual)
        points = [(b, median(values)) for b, values in sorted(groups.items())]
        q0, q1 = solve([[1.0, b] for b, _ in points], [v for _, v in points])
        if stride * size < 128:
            coefficient_points.append((stride * size, q0, q1))
    design = [[1.0, stride] for stride, _, _ in coefficient_points]
    b1, b2 = solve(design, [q0 for _, q0, _ in coefficient_points])
    b3, b4 = solve(design, [q1 for _, _, q1 in coefficient_points])
    return ng, [b1, b2, b3, b4]


def predict_base(params: dict[str, object], dtype: str, k: int, b: float) -> float:
    values = params[dtype] if dtype in params else params
    if k <= 2:
        suffix = "1"
    else:
        suffix = "2"
    if f"h_{suffix}" in values:
        return b * (
            k / float(values[f"T_{suffix}"]) + float(values[f"h_{suffix}"])
        ) + float(values[f"H_{suffix}"])
    # Accept legacy Round1 models that used H as the per-core intercept.
    return float(values[f"H_{suffix}"]) + b / float(values[f"T_{suffix}"])


def s_value(dtype: str, input_stride: float) -> float:
    return min(128.0, input_stride * DTYPE_SIZES[dtype])


def predict_correction(
    params: dict[str, object],
    dtype: str,
    b: float,
    input_stride: float,
    output_stride: float,
) -> tuple[float, float, float]:
    s = s_value(dtype, input_stride)
    gate = min(1.0, max(0.0, output_stride - 1.0))
    ng = (float(params[dtype]["N_G"]["a1"]) + float(params[dtype]["N_G"]["a2"]) * b) * s
    ngu = (
        (float(params[dtype]["N_GU"]["b1"]) + float(params[dtype]["N_GU"]["b2"]) * s)
        + (float(params[dtype]["N_GU"]["b3"]) + float(params[dtype]["N_GU"]["b4"]) * s)
        * b
    ) * gate
    return ng, ngu, s


def rho(params: dict[str, object], dtype: str, s: float, gate: float) -> float:
    values = params[dtype]["rho"]
    return (float(values["c1"]) + float(values["c2"]) * s) + gate * (
        float(values["c3"]) + float(values["c4"]) * s
    )


def calc_metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        return {
            "count": 0,
            "rmse_cycles": 0.0,
            "mae_cycles": 0.0,
            "mape_percent": 0.0,
            "max_absolute_error_cycles": 0.0,
        }
    errors = [float(row["predicted"]) - float(row["actual"]) for row in rows]
    absolute = [abs(value) for value in errors]
    ape = [
        abs(error) / max(abs(float(row["actual"])), 1e-12)
        for error, row in zip(errors, rows)
    ]
    return {
        "count": len(rows),
        "rmse_cycles": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "mae_cycles": sum(absolute) / len(absolute),
        "mape_percent": 100.0 * sum(ape) / len(ape),
        "max_absolute_error_cycles": max(absolute),
    }


def fit_model(
    rows: list[dict[str, object]],
    round1_model: dict[str, object],
) -> dict[str, object]:
    round1_parameters = round1_model["parameters"]
    round1_formula = round1_model.get("formula", {})
    params: dict[str, dict[str, object]] = {}
    for dtype in DTYPES:
        dtype_rows = [row for row in rows if row["dtype"] == dtype]
        base = round1_parameters[dtype]

        ng_coefficients, ngu_coefficients = fit_single_core(dtype_rows, dtype, base)
        params[dtype] = {
            "base": base,
            "N_G": {"a1": ng_coefficients[0], "a2": ng_coefficients[1]},
            "N_GU": {
                "b1": ngu_coefficients[0],
                "b2": ngu_coefficients[1],
                "b3": ngu_coefficients[2],
                "b4": ngu_coefficients[3],
            },
        }

        multi_rows = [
            row
            for row in dtype_rows
            if row["group"] in {"G", "H"}
            and row["block_dim"] > 2
            and row["input_stride"] * DTYPE_SIZES[dtype] <= 128
        ]
        matrix = []
        target = []
        for row in multi_rows:
            ng, ngu, s = predict_correction(
                params,
                dtype,
                row["bytes_per_core"],
                row["input_stride"],
                row["output_stride"],
            )
            correction = ng + ngu
            gate = min(1.0, row["output_stride"] - 1.0)
            matrix.append(
                [correction, s * correction, gate * correction, gate * s * correction]
            )
            target.append(
                row["actual"]
                - predict_base(base, dtype, row["block_dim"], row["bytes_per_core"])
            )
        rho_coefficients = solve(matrix, target, scaled=False)
        params[dtype]["rho"] = {
            "c1": rho_coefficients[0],
            "c2": rho_coefficients[1],
            "c3": rho_coefficients[2],
            "c4": rho_coefficients[3],
        }

    output_rows = []
    for row in rows:
        dtype = str(row["dtype"])
        k = int(row["block_dim"])
        b = float(row["bytes_per_core"])
        base_prediction = predict_base({dtype: params[dtype]["base"]}, dtype, k, b)
        ng, ngu, s = predict_correction(
            params,
            dtype,
            b,
            float(row["input_stride"]),
            float(row["output_stride"]),
        )
        gate = min(1.0, max(0.0, float(row["output_stride"]) - 1.0))
        multiplier = 1.0 if k <= 2 else rho(params, dtype, s, gate)
        predicted = base_prediction + (ng + ngu) * multiplier
        output_rows.append(
            {
                **row,
                "branch": branch(k),
                "base_cycles": base_prediction,
                "N_G": ng,
                "N_GU": ngu,
                "rho": multiplier,
                "predicted": predicted,
                "error": predicted - float(row["actual"]),
            }
        )

    return {
        "model": "NDDMA_ROUND2_1D_NONCONTIGUOUS_NG_NGU_MULTICORE",
        "formula": {
            "base": round1_formula,
            "N_G": "N_G=(a1+a2*B)*s",
            "N_GU": "N_GU=((b1+b2*s)+(b3+b4*s)*B)*min(1,os-1)",
            "rho": "rho=(c1+c2*s)+min(1,os-1)*(c3+c4*s)",
            "final": "N_1=N_base+N_G+N_GU (k<=2); N_base+rho*(N_G+N_GU) (k>2)",
            "definitions": {
                "B": "bytes_per_core = logical_total_bytes / block_dim",
                "s": "min(input_stride*dtype_size,128)",
                "os_gate": "min(1,output_stride-1)",
            },
            "fit_order": ["inherit_round1_base", "N_G", "N_GU", "multicore_rho"],
            "base_source": "round1_1d_single_multi_core_model.json",
            "base_parameters": "parameters.<dtype>.base is copied from round1.parameters.<dtype>",
        },
        "parameters": params,
        "fit_scope": {
            "reference": "HW_GE_ATT Round4 canonical 1D model",
            "single_core": "B<50000; N_G: A, byte stride<128, per-B slope then B regression; N_GU: C/os=2, per-stride B regression then byte stride<128 coefficient regression",
            "rho": "G/H, k>2, input_stride*dtype_size<=128; unscaled least squares",
            "validation": "B, F, I, J and out-of-fit-range samples; no Base refit",
        },
        "metrics": {
            "all": calc_metrics(output_rows),
            "by_dtype": {
                dtype: calc_metrics(
                    [row for row in output_rows if row["dtype"] == dtype]
                )
                for dtype in DTYPES
            },
            "by_group": {
                group: calc_metrics(
                    [row for row in output_rows if row["group"] == group]
                )
                for group in sorted({str(row["group"]) for row in output_rows})
            },
        },
        "predictions": output_rows,
    }


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "token",
        "group",
        "dtype",
        "block_dim",
        "bytes_per_core",
        "logical_total_bytes",
        "input_stride",
        "output_stride",
        "branch",
        "base_cycles",
        "N_G",
        "N_GU",
        "rho",
        "actual_cycles",
        "predicted_cycles",
        "error_cycles",
    )
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "token": row["token"],
                    "group": row["group"],
                    "dtype": row["dtype"],
                    "block_dim": row["block_dim"],
                    "bytes_per_core": row["bytes_per_core"],
                    "logical_total_bytes": row["logical_total_bytes"],
                    "input_stride": row["input_stride"],
                    "output_stride": row["output_stride"],
                    "branch": row["branch"],
                    "base_cycles": row["base_cycles"],
                    "N_G": row["N_G"],
                    "N_GU": row["N_GU"],
                    "rho": row["rho"],
                    "actual_cycles": row["actual"],
                    "predicted_cycles": row["predicted"],
                    "error_cycles": row["error"],
                }
            )


def main() -> int:
    args = parse_args()
    files = (
        [Path(args.measurement_csv).resolve()]
        if args.measurement_csv
        else find_files(Path(args.data_dir).resolve())
    )
    if not files:
        raise SystemExit(
            f"[ERROR] no profiling measurement csv found under {args.data_dir}"
        )
    rows = aggregate(row for path in files for row in read_rows(path))
    if not rows:
        raise SystemExit("[ERROR] no Round2 1D measurements found")
    round1_model_path = Path(args.round1_model).resolve()
    if not round1_model_path.is_file():
        raise SystemExit(f"[ERROR] round1 model JSON not found: {round1_model_path}")
    try:
        round1_model = json.loads(round1_model_path.read_text(encoding="utf-8"))
        round1_parameters = round1_model["parameters"]
        if not isinstance(round1_model["formula"], dict):
            raise ValueError("round1 formula must be an object")
        for dtype in DTYPES:
            values = round1_parameters[dtype]
            for key in ("T_1", "H_1", "T_2", "H_2"):
                float(values[key])
            if "h_1" in values or "h_2" in values:
                for key in ("h_1", "h_2"):
                    float(values[key])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"[ERROR] invalid round1 model JSON: {round1_model_path}: {error}"
        ) from error
    model = fit_model(rows, round1_model)
    model["formula"]["base_source"] = str(round1_model_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    model_without_rows = {
        key: value for key, value in model.items() if key != "predictions"
    }
    model_path.write_text(
        json.dumps(model_without_rows, indent=2) + "\n", encoding="utf-8"
    )
    write_predictions(output_dir / PREDICTIONS_FILENAME, model["predictions"])
    print(f"[INFO] fitted {len(rows)} points")
    print(f"[INFO] wrote model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
