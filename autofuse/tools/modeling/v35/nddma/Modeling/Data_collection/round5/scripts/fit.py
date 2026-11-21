#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Refit 3D N_G/N_GU/rho with the inherited continuous Base frozen."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from statistics import median

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round5"
MODEL_FILENAME = "round5_3d_specialized_model.json"
PREDICTIONS_FILENAME = "round5_3d_specialized_predictions.csv"

spec = importlib.util.spec_from_file_location(
    "round5_formula_base", SCRIPT_DIR.parents[1] / "round3/scripts/fit.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_ANA_DIR / "collection"))
    parser.add_argument(
        "--measurement-csv",
        action="append",
        default=[],
        help="May be repeated to combine 3D measurement CSVs.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_ANA_DIR))
    parser.add_argument("--round2-model", default=str(common.DEFAULT_ROUND2_MODEL))
    return parser.parse_args()


def aggregate(rows):
    """Median repeats by physical configuration, including dtype and core count."""
    grouped = defaultdict(list)
    for row in rows:
        if str(row.get("dim", "")).strip() != "3":
            continue
        dtype = row["dtype"]
        if dtype not in common.DTYPE_SIZES:
            raise ValueError(f"unsupported dtype: {dtype}")
        core = int(row.get("block_dim") or 1)
        descriptors = tuple(
            common.parse_tuple(row[field], 3)
            for field in ("output_dims", "input_stride", "output_stride")
        )
        if core < 1 or any(value <= 0 for value in descriptors[0]):
            raise ValueError("3D dimensions and block_dim must be positive")
        if any(value < 0 for stride in descriptors[1:] for value in stride):
            raise ValueError("3D strides must be nonnegative")
        grouped[(dtype, core, descriptors)].append(row)
    result = []
    for (dtype, core, _), values in grouped.items():
        row = dict(values[0], dtype=dtype, block_dim=str(core), dim="3")
        row["actual_y"] = str(median(common.actual_value(value) for value in values))
        result.append(row)
    return result


def solve(design, target, label):
    """Column scaling improves conditioning without changing the OLS objective."""
    matrix = np.asarray(design, dtype=float)
    values = np.asarray(target, dtype=float)
    if matrix.ndim != 2 or len(matrix) < matrix.shape[1]:
        raise ValueError(f"{label}: not enough samples")
    if not np.isfinite(matrix).all() or not np.isfinite(values).all():
        raise ValueError(f"{label}: nonfinite fitting values")
    scale = np.max(np.abs(matrix), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    coeff, _, rank, _ = np.linalg.lstsq(matrix / scale, values, rcond=1e-12)
    if rank != matrix.shape[1]:
        raise ValueError(
            f"{label}: rank-deficient design ({rank}/{matrix.shape[1]}); "
            "collect more varied 3D shapes/strides"
        )
    return (coeff / scale).tolist()


def correction_features(terms, dtype):
    # Coefficient order: a1, a2, b1, b2, b3, b4.
    features = np.zeros(6)
    for term in terms:
        b = float(term["data_volume_bytes"])
        s = min(float(term["input_stride_delta"]) * common.DTYPE_SIZES[dtype], 128.0)
        g = min(1.0, max(0.0, float(term["output_stride_delta"]) - 1.0))
        features += np.array([s, s * b, g, g * s, g * b, g * s * b])
    return features


def rho_features(terms, dtype):
    # The six correction parameters have already been fitted and are now fixed.
    features = np.zeros(4)
    for term in terms:
        s = min(float(term["input_stride_delta"]) * common.DTYPE_SIZES[dtype], 128.0)
        g = min(1.0, max(0.0, float(term["output_stride_delta"]) - 1.0))
        features += float(term["t1_cycles"]) * np.array([1.0, s, g, g * s])
    return features


def fit_model(rows, round2, round2_model_source):
    rows = aggregate(rows)
    if not rows:
        raise ValueError("no 3D measurements found")
    specialized = {"parameters": deepcopy(common.round2_parameters(round2))}
    diagnostics = {}
    for dtype in common.DTYPE_SIZES:
        selected = [row for row in rows if row["dtype"] == dtype]
        lower = [row for row in selected if int(row["block_dim"]) <= 2]
        upper = [row for row in selected if int(row["block_dim"]) > 2]
        if not lower or not upper:
            raise ValueError(
                f"{dtype}: need both block_dim<=2 and block_dim>2 3D data "
                "to refit all correction parameters"
            )
        design, target = [], []
        for row in lower:
            base, terms = common.inherited_terms(round2, row, use_core_rho=False)
            design.append(correction_features(terms, dtype))
            target.append(common.actual_value(row) - base)
        coeff = solve(design, target, f"{dtype} N_G/N_GU")
        params = specialized["parameters"][dtype]
        params["N_G"] = dict(zip(("a1", "a2"), coeff[:2]))
        params["N_GU"] = dict(zip(("b1", "b2", "b3", "b4"), coeff[2:]))
        design, target = [], []
        for row in upper:
            base, terms = common.inherited_terms(specialized, row, use_core_rho=False)
            design.append(rho_features(terms, dtype))
            target.append(common.actual_value(row) - base)
        coeff = solve(design, target, f"{dtype} rho")
        params["rho"] = dict(zip(("c1", "c2", "c3", "c4"), coeff))
        diagnostics[dtype] = {
            "correction_samples": len(lower),
            "rho_samples": len(upper),
            "correction_rank": 6,
            "rho_rank": 4,
        }

    model = common.fit_model(rows, specialized, round2_model_source)
    inherited = common.fit_model(rows, round2, round2_model_source)
    for row, point, old in zip(rows, model["predictions"], inherited["predictions"]):
        point["block_dim"] = int(row["block_dim"])
        point["baseline_predicted_cycles"] = old["predicted_cycles"]
        point["baseline_error_cycles"] = old["error_cycles"]
        point["baseline_t1_cycles"] = old["t1_cycles"]
        point["baseline_n1_terms_json"] = old["n1_terms_json"]
    model["model"] = "NDDMA2_ROUND5_3D_SPECIALIZED"
    model["formula"].pop("round3_fitted_parameters")
    model["formula"].update(
        dimensions=[3], round5_fitted_parameters="N_G/N_GU/rho; frozen N_base; no d_0"
    )
    model["baseline"] = {
        "name": "inherited_round2",
        "description": "Round3 formula with unchanged Round2 parameters.",
        "metrics": inherited["metrics"],
    }
    model["baseline_metrics"] = inherited["metrics"]
    model["fit_scope"].update(
        {
            "method": "staged column-scaled OLS, rcond=1e-12; no ridge or clipping",
            "base_parameters": "copied unchanged from Round2",
            "correction_fit": "block_dim<=2; solve a1,a2,b1,b2,b3,b4 jointly against actual-N_base",
            "rho_fit": "block_dim>2; freeze fitted a/b, solve c1,c2,c3,c4 against actual-N_base",
            "diagnostics": diagnostics,
            "metrics_scope": "in-sample; no held-out validation split",
        }
    )
    model["metrics_by_core"] = {
        dtype: {
            str(core): common.metrics(
                [
                    p
                    for p in model["predictions"]
                    if p["dtype"] == dtype and p["block_dim"] == core
                ]
            )
            for core in sorted(
                {int(r["block_dim"]) for r in rows if r["dtype"] == dtype}
            )
        }
        for dtype in common.DTYPE_SIZES
    }
    return model


def write_predictions(path, points):
    # Extend the shared prediction schema with the core count used in fitting.
    fields = list(points[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(points)


def main() -> int:
    args = parse_args()
    source = Path(args.round2_model).resolve()
    try:
        round2 = json.loads(source.read_text(encoding="utf-8"))
        common.validate_round2_model(round2, source)
        files = [
            Path(path).resolve() for path in args.measurement_csv
        ] or common.find_files(Path(args.data_dir).resolve())
        if not files:
            raise ValueError(f"no measurement CSV found under {args.data_dir}")
        model = fit_model(
            [row for path in files for row in common.read_rows(path)],
            round2,
            str(source),
        )
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"[ERROR] {error}") from error
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / MODEL_FILENAME).write_text(
        json.dumps(
            {key: value for key, value in model.items() if key != "predictions"},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_predictions(output / PREDICTIONS_FILENAME, model["predictions"])
    print(
        f"[INFO] fitted {len(model['predictions'])} 3D points; wrote {output / MODEL_FILENAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
