#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check archived estimates or refit the small Load and padded NG2 models."""

import argparse
import csv
import json
import math
from collections import defaultdict
from functools import reduce
from operator import mul
from pathlib import Path

SMALL_LOAD = ("LoadSmallPayload", "LoadSmallPayloadCompact")
DENSE_LOAD = "LoadAlignedDenseBulkPayload"
PADDED_NG2 = "NG2 padded FP32"
NARROW_NG2 = "NG2 narrow source FP32"


def read_by_id(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            case_id = row["case_id"]
            if not case_id or case_id in rows:
                raise ValueError(f"{path}: empty or duplicate case_id {case_id!r}")
            rows[case_id] = row
        return rows


def cycles(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid cycle count: {text!r}")
    return value


def read_points(cases_path, samples_path):
    cases = read_by_id(cases_path)
    samples = read_by_id(samples_path)
    if cases.keys() != samples.keys():
        raise ValueError("cases and samples must have identical case IDs")
    for case_id, case in cases.items():
        sample = samples[case_id]
        prediction = sample["model_predict_cycles"]
        yield {
            **case,
            "measured": cycles(sample["measured_cycles"]),
            "predicted": cycles(prediction) if prediction else None,
        }


def summarize(points):
    groups = defaultdict(list)
    for point in points:
        groups[point["role"]].append(point)
    result = {}
    for role, group in sorted(groups.items()):
        errors = [
            abs(point["predicted"] / point["measured"] - 1)
            for point in group
            if point["predicted"] is not None
        ]
        result[role] = {
            "points": len(group),
            "predicted_points": len(errors),
            "max_relative_error": max(errors, default=None),
        }
    return result


def minimax(features, measured, np, linprog):
    normalized = features / measured[:, None]
    count = features.shape[1]
    result = linprog(
        [0.0] * count + [1.0],
        A_ub=np.column_stack(
            (np.vstack((normalized, -normalized)), -np.ones(2 * len(measured)))
        ),
        b_ub=np.r_[np.ones(len(measured)), -np.ones(len(measured))],
        bounds=[(0, None)] * (count + 1),
        method="highs",
    )
    if not result.success:
        raise ValueError(f"minimax fit failed: {result.message}")
    return result.x[:-1], result.fun


def fit(points, model):
    import numpy as np
    from scipy.optimize import linprog

    if not points or any(point["dtype"] != "float32" for point in points):
        raise ValueError("the fit requires FP32 data for one model and policy")
    train = np.array([point["role"] == "train" for point in points])
    if not train.any() or not any(point["role"] == "holdout" for point in points):
        raise ValueError("the fit requires separate training and holdout points")
    measured = np.array([point["measured"] for point in points])
    payload = np.array(
        [4 * reduce(mul, map(int, point["dims"].split("x")), 1) for point in points]
    )
    cores = np.array([int(point["blocks"]) for point in points])
    if (payload <= 0).any() or (cores <= 0).any():
        raise ValueError("payload and active core count must be positive")
    intercept = np.ones(len(points))
    if model == PADDED_NG2:
        scaled = payload / 65536
        features = np.column_stack(
            (intercept, scaled, scaled * np.maximum(cores - 32, 0) / 24)
        )
        params, _ = minimax(features[train], measured[train], np, linprog)
        names = ("startup", "bandwidth_cost", "contention_cost")
        predicted = features @ params
    elif model == NARROW_NG2:
        positions = np.array([int(point["dims"].split("x")[-1]) for point in points])
        features = np.column_stack(
            (intercept, positions, positions * np.maximum(cores - 32, 0) / 24)
        )
        params, _ = minimax(features[train], measured[train], np, linprog)
        names = ("startup", "position_cost", "contention_cost")
        predicted = features @ params
    else:
        stride_features = []
        if model == "LoadSmallPayloadCompact":
            gaps = []
            for point in points:
                rows, width = map(int, point["dims"].split("x"))
                gm_pitch, _ = map(int, point["gm_stride_elements"].split("x"))
                gaps.append(min(max(4 * (gm_pitch - width), 0), 4096) * rows)
            stride_features.append(np.array(gaps))
        best = None
        for ratio in np.r_[0.0, np.geomspace(0.001, 1000.0, 241)]:
            features = np.column_stack(
                (intercept, payload / (1 + ratio / cores), *stride_features)
            )
            params, error = minimax(features[train], measured[train], np, linprog)
            if best is None or error < best[0]:
                best = (error, ratio, params)
        _, ratio, (head, slope, *stride_params) = best
        if slope <= 0:
            raise ValueError("training points produced a zero payload slope")
        names = ("h", "a", "b")
        params = (head, 1 / slope, ratio / slope)
        predicted = head + payload / (params[1] + params[2] / cores)
        if stride_features:
            names += ("k",)
            params += (stride_params[0],)
            predicted += stride_params[0] * stride_features[0]
    fitted = [
        {
            **point,
            "predicted": None
            if model == NARROW_NG2 and point["role"] == "domain_control"
            else float(value),
        }
        for point, value in zip(points, predicted)
    ]
    return {
        "parameters": dict(zip(names, map(float, params))),
        "errors_by_role": summarize(fitted),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument(
        "--fit",
        choices=(*SMALL_LOAD, DENSE_LOAD, PADDED_NG2, NARROW_NG2),
        help="refit using train rows only",
    )
    parser.add_argument(
        "--policy", help="select one cache/address protocol for fitting"
    )
    args = parser.parse_args()
    try:
        groups = defaultdict(list)
        for point in read_points(args.cases, args.samples):
            groups[point["model"], point["policy"]].append(point)
        if args.fit:
            if args.policy is None:
                parser.error("--fit requires --policy")
            output = fit(groups.get((args.fit, args.policy), []), args.fit)
        else:
            if args.policy is not None:
                parser.error("--policy requires --fit")
            output = [
                {"model": model, "policy": policy, "errors_by_role": summarize(points)}
                for (model, policy), points in sorted(groups.items())
            ]
    except (OSError, KeyError, ValueError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
