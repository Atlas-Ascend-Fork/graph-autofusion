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
from typing import Iterable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round4"
DEFAULT_DATA_DIR = DEFAULT_ANA_DIR / "collection"
DEFAULT_ROUND2_MODEL = (
    MODELING_DIR / "Ana" / "round2" / "round2_1d_noncontiguous_model.json"
)
MODEL_FILENAME = "round4_2d_ub_contiguous_ng2_model.json"
PREDICTIONS_FILENAME = "round4_2d_ub_contiguous_ng2_predictions.csv"
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
LOW_BYTE_STRIDE_MAX = 128.0
METRIC_FIELDS = (
    "actual_y",
    "nddma_mte2_cycles_per_block",
    "mte2_cycles_per_block",
    "mte2_cycles",
    "nddma_mte2_cycles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit NDDMA2 Round4 Round5 N_G2 2D UB-contiguous model."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--measurement-csv", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_ANA_DIR))
    parser.add_argument("--round2-model", default=str(DEFAULT_ROUND2_MODEL))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def find_files(data_dir: Path) -> list[Path]:
    direct = data_dir / "measurements.csv"
    if direct.exists():
        return [direct]
    files = sorted(data_dir.rglob("profiling_with_params_mean.csv"))
    return files or sorted(data_dir.rglob("profiling_with_params.csv"))


def actual_value(row: Mapping[str, str]) -> float:
    for field in METRIC_FIELDS:
        text = str(row.get(field, "")).strip()
        if not text or text.upper() == "N/A":
            continue
        value = float(text)
        if field != "actual_y":
            value /= max(1.0, float(row.get("repeat") or 1.0))
        if math.isfinite(value) and value > 0:
            return value
    raise ValueError(f"no positive measurement found for {row.get('token', '')}")


def parse_dims(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.replace(";", "x").split("x") if part)


def aggregate(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (row.get("data_split") or row.get("group_id")) not in {"D", "F", "G"}:
            continue
        if int(row.get("block_dim") or 1) != 1:
            continue
        if row.get("dim") != "2":
            continue
        dims = parse_dims(row.get("output_dims", ""))
        input_stride = parse_dims(row.get("input_stride", ""))
        output_stride = parse_dims(row.get("output_stride", ""))
        if len(dims) != 2 or len(input_stride) != 2 or len(output_stride) != 2:
            continue
        if not (input_stride[0] < input_stride[1]):
            continue
        if output_stride != (dims[1], 1):
            continue
        token = row.get("config_id") or row.get("token") or ""
        grouped[token].append(row)
    result = []
    for token, values in grouped.items():
        row = dict(values[0])
        row["config_id"] = token
        row["actual_y"] = str(median(actual_value(value) for value in values))
        result.append(row)
    return result


def round2_parameters(round2: Mapping[str, object]) -> Mapping[str, object]:
    return round2.get("parameters", round2.get("dtype_models", {}))


def base_params(
    round2: Mapping[str, object], dtype: str, block_dim: int
) -> Mapping[str, object]:
    base = round2_parameters(round2)[dtype]["base"]
    if "T_1" in base:
        return base
    return base["le2" if block_dim <= 2 else "gt2"]


def one_d_base(
    round2: Mapping[str, object], dtype: str, bytes_value: float, block_dim: int
) -> float:
    params = base_params(round2, dtype, block_dim)
    if "T_1" in params:
        suffix = "1" if block_dim <= 2 else "2"
        return bytes_value / float(params[f"T_{suffix}"]) + float(params[f"H_{suffix}"])
    return float(params["alpha"]) + bytes_value / float(params["T_bytes_per_cycle"])


def one_d_gm_correction(
    round2: Mapping[str, object],
    dtype: str,
    bytes_value: float,
    input_stride: int,
    block_dim: int,
) -> float:
    params = round2_parameters(round2)[dtype]
    s = min(float(input_stride) * DTYPE_SIZES[dtype], LOW_BYTE_STRIDE_MAX)
    if "N_G" in params:
        ng = (float(params["N_G"]["a1"]) + float(params["N_G"]["a2"]) * bytes_value) * s
        if block_dim <= 2:
            return ng
        rho = params["rho"]
        multiplier = float(rho["c1"]) + float(rho["c2"]) * s
        return multiplier * ng

    ng = (float(params["a_1"]) + float(params["a_2"]) * bytes_value) * s
    if block_dim <= 2:
        return ng
    multiplier = float(params["c_1"]) + float(params["c_2"]) * s
    return multiplier * ng


def predict_ng2(params: Mapping[str, object], m: int, is2: int) -> float:
    return (
        (float(params["g10"]) + float(params["g11_M"]) * float(m)) * float(is2)
        + float(params["g00"])
        + float(params["g01_M"]) * float(m)
    )


def solve(matrix: list[list[float]], target: list[float]) -> list[float]:
    """HW Round5 N_G2: unscaled, unregularized four-parameter OLS."""
    design = np.asarray(matrix, dtype=float)
    if design.ndim != 2 or len(design) < design.shape[1]:
        raise ValueError("not enough samples for N_G2 fit")
    coeff, _, rank, _ = np.linalg.lstsq(
        design, np.asarray(target, dtype=float), rcond=None
    )
    if rank < design.shape[1]:
        raise ValueError("rank-deficient N_G2 fit")
    return coeff.tolist()


def prepare(
    rows: Sequence[Mapping[str, str]], round2: Mapping[str, object]
) -> list[dict[str, object]]:
    points = []
    for row in rows:
        dtype = row["dtype"]
        m, n = parse_dims(row["output_dims"])
        is2, is1 = parse_dims(row["input_stride"])
        block_dim = int(row["block_dim"])
        dtype_size = DTYPE_SIZES[dtype]
        total_bytes = float(m * n * dtype_size)
        inner_bytes = float(n * dtype_size)
        n_base = one_d_base(round2, dtype, total_bytes, block_dim)
        n_g1 = one_d_gm_correction(round2, dtype, inner_bytes, is1, block_dim)
        actual = actual_value(row)
        if abs(n_g1) <= 1e-12:
            raise ValueError(f"{row.get('token', '')}: inherited N_G1 is zero")
        points.append(
            {
                "token": row.get("config_id") or row.get("token"),
                "dtype": dtype,
                "block_dim": block_dim,
                "m": m,
                "n": n,
                "is2": is2,
                "is1": is1,
                "bytes_per_core": float(row.get("bytes_per_core") or total_bytes),
                "logical_total_bytes": float(
                    row.get("logical_total_bytes") or total_bytes
                ),
                "n_base_cycles": n_base,
                "n_g1_cycles": n_g1,
                "actual_cycles": actual,
                "observed_n_g2": (actual - n_base) / n_g1,
            }
        )
    return points


def calc_metrics(points: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    if not points:
        return {
            "count": 0,
            "rmse_cycles": 0.0,
            "mae_cycles": 0.0,
            "mape_percent": 0.0,
            "max_absolute_error_cycles": 0.0,
        }
    errors = [
        float(point["predicted_cycles"]) - float(point["actual_cycles"])
        for point in points
    ]
    absolute = [abs(value) for value in errors]
    ape = [
        abs(error) / max(abs(float(point["actual_cycles"])), 1e-12)
        for error, point in zip(errors, points)
    ]
    return {
        "count": len(points),
        "rmse_cycles": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "mae_cycles": sum(absolute) / len(absolute),
        "mape_percent": 100.0 * sum(ape) / len(ape),
        "max_absolute_error_cycles": max(absolute),
    }


def inherited_one_dimensional_parameters(
    round2: Mapping[str, object],
) -> dict[str, object]:
    parameters = round2_parameters(round2)
    result = {}
    for dtype in DTYPES:
        values = parameters[dtype]
        result[dtype] = {
            "base": values["base"],
            "N_G": values["N_G"],
            "rho": values.get("rho", {}),
        }
    return result


def fit_model(
    rows: list[dict[str, str]], round2: Mapping[str, object], round2_model_source: str
) -> dict[str, object]:
    rows = aggregate(rows)
    for dtype in DTYPES:
        groups = {
            row.get("data_split") or row.get("group_id")
            for row in rows
            if row["dtype"] == dtype
        }
        if groups != {"D", "F", "G"}:
            raise ValueError(
                f"{dtype}: aligned N_G2 fit requires D/F/G measurements; found {sorted(groups)}"
            )
    points = prepare(rows, round2)
    parameters: dict[str, dict[str, float]] = {}
    for dtype in DTYPES:
        selected = [point for point in points if point["dtype"] == dtype]
        if len(selected) < 4:
            raise ValueError(f"{dtype}: need at least four Round5 N_G2 samples")
        matrix = [
            [
                float(point["is2"]),
                float(point["m"]) * float(point["is2"]),
                1.0,
                float(point["m"]),
            ]
            for point in selected
        ]
        target = [float(point["observed_n_g2"]) for point in selected]
        g10, g11_m, g00, g01_m = solve(matrix, target)
        parameters[dtype] = {
            "g10": g10,
            "g11_M": g11_m,
            "g00": g00,
            "g01_M": g01_m,
        }

    for point in points:
        dtype = str(point["dtype"])
        n_g2 = predict_ng2(parameters[dtype], int(point["m"]), int(point["is2"]))
        predicted = float(point["n_base_cycles"]) + float(point["n_g1_cycles"]) * n_g2
        point["predicted_n_g2"] = n_g2
        point["predicted_cycles"] = predicted
        point["error_cycles"] = predicted - float(point["actual_cycles"])

    return {
        "model": "NDDMA_ROUND4_ROUND5_NG2_2D_UB_CONTIGUOUS",
        "formula": {
            "scope": "[M,N]/[is2,is1]/[N,1], is2<is1",
            "N_base": "N_base=round2.base(dtype,B,k), B=M*N*dtype_size",
            "N_G1": "N_G1=round2.N_G(dtype,B1,is1,k), B1=N*dtype_size, output_stride=1",
            "N_G2": "N_G2=(g10+g11_M*M)*is2+g00+g01_M*M",
            "prediction": "N_2=N_base+N_G1*N_G2",
            "round4_fitted_parameters": "N_G2 only",
        },
        "fit_scope": {
            "groups": ["D", "F", "G"],
            "block_dim": 1,
            "method": "OLS on observed N_G2 with [is2, M*is2, 1, M]; no d_0",
        },
        "round2_model_source": round2_model_source,
        "parameters": {
            "N_G2": parameters,
            "one_dimensional": inherited_one_dimensional_parameters(round2),
        },
        "metrics": {
            "all": calc_metrics(points),
            "by_dtype": {
                dtype: calc_metrics(
                    [point for point in points if point["dtype"] == dtype]
                )
                for dtype in DTYPES
            },
            "by_block_dim": {
                str(block_dim): calc_metrics(
                    [point for point in points if int(point["block_dim"]) == block_dim]
                )
                for block_dim in sorted({int(point["block_dim"]) for point in points})
            },
        },
        "predictions": points,
    }


def write_predictions(path: Path, points: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "token",
        "dtype",
        "block_dim",
        "m",
        "n",
        "is2",
        "is1",
        "bytes_per_core",
        "logical_total_bytes",
        "n_base_cycles",
        "n_g1_cycles",
        "observed_n_g2",
        "predicted_n_g2",
        "actual_cycles",
        "predicted_cycles",
        "error_cycles",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for point in points:
            writer.writerow({field: point.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    round2_model_path = Path(args.round2_model).resolve()
    if not round2_model_path.exists():
        raise SystemExit(f"[ERROR] round2 model JSON not found: {round2_model_path}")
    try:
        round2 = json.loads(round2_model_path.read_text(encoding="utf-8"))
        parameters = round2_parameters(round2)
        for dtype in DTYPES:
            values = parameters[dtype]
            base = values["base"]
            for key in ("T_1", "H_1", "T_2", "H_2"):
                float(base[key])
            for key in ("h_1", "h_2"):
                if key in base:
                    float(base[key])
            for key in ("a1", "a2"):
                float(values["N_G"][key])
            for key in ("c1", "c2"):
                float(values["rho"][key])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"[ERROR] invalid round2 model JSON: {round2_model_path}: {error}"
        ) from error
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
        raise SystemExit("[ERROR] no Round5 N_G2 2D UB-contiguous measurements found")
    model = fit_model(rows, round2, str(round2_model_path))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    model_without_predictions = {
        key: value for key, value in model.items() if key != "predictions"
    }
    model_path.write_text(
        json.dumps(model_without_predictions, indent=2) + "\n",
        encoding="utf-8",
    )
    write_predictions(output_dir / PREDICTIONS_FILENAME, model["predictions"])
    print(f"[INFO] fitted {len(rows)} Round5 N_G2 points")
    print(f"[INFO] wrote model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
