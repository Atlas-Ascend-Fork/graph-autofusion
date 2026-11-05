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
from statistics import mean
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round1"
DEFAULT_DATA_DIR = DEFAULT_ANA_DIR / "collection"
DEFAULT_OUTPUT_DIR = DEFAULT_ANA_DIR
MODEL_FILENAME = "round1_1d_single_multi_core_model.json"
PREDICTIONS_FILENAME = "round1_1d_single_core_predictions.csv"
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
BLOCK_DIMS = tuple(range(1, 57))
SPLIT_BLOCK_DIM = 2
PER_BLOCK_METRIC_FIELDS = (
    "nddma_mte2_cycles_per_block",
    "mte2_cycles_per_block",
)
TOTAL_METRIC_FIELDS = ("nddma_mte2_cycles", "mte2_cycles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the NDDMA2 Round1 1D single/multi-core contiguous model."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--measurement-csv", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def measurement_value(row: dict[str, str]) -> float:
    """Return single-core/per-block cycles."""
    repeat = max(1.0, float(row.get("repeat") or "1"))
    block_dim = max(1.0, float(row.get("block_dim") or "1"))
    for field in PER_BLOCK_METRIC_FIELDS:
        raw = str(row.get(field, "")).strip()
        if not raw:
            continue
        value = float(raw) / repeat
        if math.isfinite(value) and value > 0:
            return value
    for field in TOTAL_METRIC_FIELDS:
        raw = str(row.get(field, "")).strip()
        if not raw:
            continue
        value = float(raw) / repeat / block_dim
        if math.isfinite(value) and value > 0:
            return value
    raise ValueError(f"no positive measurement found for token={row.get('token', '')}")


def find_measurement_files(data_dir: Path) -> list[Path]:
    direct = data_dir / "measurements.csv"
    if direct.exists():
        return [direct]
    files = sorted(data_dir.rglob("profiling_with_params_mean.csv"))
    if not files:
        files = sorted(data_dir.rglob("profiling_with_params.csv"))
    return files


def aggregate_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, float], list[tuple[dict[str, str], float]]] = (
        defaultdict(list)
    )
    valid_block_dims = {str(value) for value in BLOCK_DIMS}
    for row in rows:
        if row.get("fit_role") != "fit":
            continue
        if row.get("sensitivity_key") not in (
            "group_a_stage1_linear_fit",
            "group_c_t_alpha_block_model",
        ):
            continue
        if row.get("dim") != "1":
            continue
        if row.get("block_dim") not in valid_block_dims:
            continue
        if row.get("input_stride") != "1" or row.get("output_stride") != "1":
            continue
        if row.get("input_stride_pattern", "") not in ("", "contiguous"):
            continue
        if row.get("output_stride_pattern", "") not in ("", "contiguous"):
            continue
        block_dim = int(row["block_dim"])
        byte_value = float(
            row.get("model_x_bytes_per_core")
            or (
                float(row["logical_total_bytes"]) / block_dim
                if row.get("logical_total_bytes")
                else row.get("bytes_per_core") or row["total_bytes"]
            )
        )
        key = (row["dtype"], block_dim, byte_value)
        grouped[key].append((row, measurement_value(row)))

    result: list[dict[str, object]] = []
    for (_, block_dim, byte_value), values in sorted(grouped.items()):
        row = values[0][0]
        actual = mean(value for _, value in values)
        result.append(
            {
                "token": row.get("token", ""),
                "dtype": row["dtype"],
                "block_dim": int(row["block_dim"]),
                "bytes_per_core": byte_value,
                "logical_total_bytes": byte_value * block_dim,
                "group_id": row.get("group_id", ""),
                "sensitivity_key": row.get("sensitivity_key", ""),
                "actual": actual,
            }
        )
    return result


def branch_for_block_dim(block_dim: int) -> str:
    return "le2" if block_dim <= SPLIT_BLOCK_DIM else "gt2"


def predict(params: dict[str, object], block_dim: int, bytes_per_core: float) -> float:
    suffix = "1" if block_dim <= SPLIT_BLOCK_DIM else "2"
    return bytes_per_core / float(params[f"T_{suffix}"]) + float(params[f"H_{suffix}"])


def summarize_metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
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


def fit_branch(points: list[dict[str, object]], branch_name: str) -> dict[str, object]:
    if not points:
        raise ValueError(f"{branch_name} has no fitting samples")
    by_core: dict[int, list[dict[str, object]]] = defaultdict(list)
    for point in points:
        by_core[int(point["block_dim"])].append(point)
    local_fits = []
    for core, core_points in sorted(by_core.items()):
        by_bytes: dict[float, list[float]] = defaultdict(list)
        for point in core_points:
            by_bytes[float(point["bytes_per_core"])].append(float(point["actual"]))
        xy = [(b, mean(values)) for b, values in sorted(by_bytes.items())]
        if len(xy) < 2:
            raise ValueError(
                f"{branch_name} core {core}: need two distinct byte values"
            )
        x_mean = mean(x for x, _ in xy)
        y_mean = mean(y for _, y in xy)
        slope = sum((x - x_mean) * (y - y_mean) for x, y in xy) / sum(
            (x - x_mean) ** 2 for x, _ in xy
        )
        if slope <= 0:
            raise ValueError(f"{branch_name} core {core}: non-positive fitted slope")
        local_fits.append(
            {
                "block_dim": core,
                "T": 1.0 / slope,
                "H": y_mean - slope * x_mean,
                "point_count": len(xy),
            }
        )
    return {
        "T": mean(p["T"] for p in local_fits),
        "H": mean(p["H"] for p in local_fits),
        "per_core": local_fits,
    }


def write_predictions(
    path: Path,
    rows: list[dict[str, object]],
    parameters: dict[str, dict[str, float]],
) -> None:
    output_rows = []
    for point in rows:
        dtype = str(point["dtype"])
        block_dim = int(point["block_dim"])
        branch_name = branch_for_block_dim(block_dim)
        predicted = predict(
            parameters[dtype], block_dim, float(point["bytes_per_core"])
        )
        output_rows.append(
            {
                "token": point["token"],
                "dtype": dtype,
                "group_id": point["group_id"],
                "sensitivity_key": point["sensitivity_key"],
                "block_dim": point["block_dim"],
                "bytes_per_core": point["bytes_per_core"],
                "logical_total_bytes": point["logical_total_bytes"],
                "branch": branch_name,
                "actual_cycles": point["actual"],
                "predicted_cycles": predicted,
                "error_cycles": predicted - float(point["actual"]),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        fieldnames = list(output_rows[0])
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def fit_model(rows: list[dict[str, object]]) -> dict[str, object]:
    parameters: dict[str, dict[str, float]] = {}
    per_core_fits = {}
    for dtype in DTYPES:
        dtype_rows = [row for row in rows if row["dtype"] == dtype]
        if not dtype_rows:
            raise ValueError(f"no fitting rows found for {dtype}")
        branch_points = {
            branch_name: [
                row
                for row in dtype_rows
                if branch_for_block_dim(int(row["block_dim"])) == branch_name
            ]
            for branch_name in ("le2", "gt2")
        }
        branches = {
            branch_name: fit_branch(points, branch_name)
            for branch_name, points in branch_points.items()
        }
        per_core_fits[dtype] = {
            name: branch["per_core"] for name, branch in branches.items()
        }
        parameters[dtype] = {
            "T_1": float(branches["le2"]["T"]),
            "H_1": float(branches["le2"]["H"]),
            "T_2": float(branches["gt2"]["T"]),
            "H_2": float(branches["gt2"]["H"]),
        }

    metric_rows = []
    for row in rows:
        dtype = str(row["dtype"])
        block_dim = int(row["block_dim"])
        branch_name = branch_for_block_dim(block_dim)
        predicted = predict(parameters[dtype], block_dim, float(row["bytes_per_core"]))
        metric_rows.append({**row, "branch": branch_name, "predicted": predicted})

    return {
        "model": "NDDMA_ROUND1_1D_PIECEWISE_D_OVER_T_PLUS_H",
        "formula": {
            "name": "arbitrary_dim_multicore_1d_contiguous_base",
            "split_block_dim": SPLIT_BLOCK_DIM,
            "le2": "cycles = bytes_per_core / T_1(dtype) + H_1(dtype)",
            "gt2": "cycles = bytes_per_core / T_2(dtype) + H_2(dtype)",
            "metric": "cycles is single-core/per-block cycles",
            "bytes_definition": "bytes_per_core = logical_total_bytes / block_dim",
            "fitting_dataset": (
                "HW_GE_ATT round2 group A stage1 representative points plus "
                "group C all-block-dim points"
            ),
            "fitting_method": (
                "mean cycles at each dtype/core/byte value; per-core ordinary least "
                "squares; arithmetic mean of per-core T and H within each branch"
            ),
        },
        "parameters": parameters,
        "per_core_fits": per_core_fits,
        "metrics": {
            "all": summarize_metrics(metric_rows),
            "by_dtype": {
                dtype: summarize_metrics(
                    [row for row in metric_rows if row["dtype"] == dtype]
                )
                for dtype in DTYPES
            },
            "by_branch": {
                branch_name: summarize_metrics(
                    [row for row in metric_rows if row["branch"] == branch_name]
                )
                for branch_name in ("le2", "gt2")
            },
        },
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    files = (
        [Path(args.measurement_csv).resolve()]
        if args.measurement_csv
        else find_measurement_files(Path(args.data_dir).resolve())
    )
    if not files:
        raise SystemExit(
            f"[ERROR] no profiling measurement csv found under {args.data_dir}"
        )

    raw_rows = [row for path in files for row in read_rows(path)]
    rows = aggregate_rows(raw_rows)
    if not rows:
        raise SystemExit(
            "[ERROR] no 1D contiguous measurements with block_dim in 1..56 found"
        )
    model = fit_model(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    model_path.write_text(
        json.dumps(model, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_predictions(
        output_dir / PREDICTIONS_FILENAME,
        rows,
        model["parameters"],
    )
    print(f"[INFO] fitted {len(rows)} points")
    print(f"[INFO] wrote model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
