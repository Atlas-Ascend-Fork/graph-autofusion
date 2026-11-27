#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

"""Fit the legacy stride residual for physical ND layouts."""

import argparse
import csv
import json
from collections import defaultdict
from math import prod
from pathlib import Path
from statistics import median


LEGACY_PARAMS = {
    "float16": (
        2,
        13.259,
        399.909,
        (
            1.9720487,
            0.0073653238,
            -65.754578,
            -1.7626747,
            3.4565828,
            -0.0073719789,
            1.6501023,
            0.016002982,
            0.33029872,
            -0.015981633,
        ),
    ),
    "float32": (
        4,
        29.4096,
        453.859,
        (
            1.2140303,
            0.0037227686,
            31.174096,
            -0.83730451,
            0.21970194,
            -0.0032801415,
            1.9774169,
            0.0056400644,
            -0.61376163,
            0.021569482,
        ),
    ),
}


def legacy_cost(row):
    dims, gm, ub = (
        list(reversed([int(value) for value in row[key].split("x")]))
        for key in ("dims", "gm_or_input_strides", "ub_or_output_strides")
    )
    dtype_size, bandwidth, startup, residual_params = LEGACY_PARAMS[row["dtype"]]
    a1, a2, b1, b2, b3, b4, c1, c2, c3, c4 = residual_params
    base = prod(dims) * dtype_size / bandwidth + startup
    residual = 0.0
    input_end = output_end = 0
    for i, extent in enumerate(dims):
        if extent <= 1:
            continue
        read = min(
            128, dtype_size * (abs(gm[i] - input_end) + 1 if input_end else gm[i])
        )
        write = abs(ub[i] - output_end) + 1 if output_end else ub[i]
        gap = min(1, write - 1)
        size = prod(dims[i:]) * dtype_size
        request_cost = (a1 + a2 * size) * read
        ub_cost = b1 + b2 * read + (b3 + b4 * read) * size
        contention = c1 + c2 * read + gap * (c3 + c4 * read)
        residual += (request_cost + ub_cost * gap) * contention
        input_end += extent * gm[i]
        output_end += extent * ub[i]
    return base, residual


def read_points(case_path, sample_path, window):
    samples = defaultdict(list)
    with sample_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["policy"] == "N" and row.get("warmup", "false").lower() != "true":
                samples[row["factor_id"], int(row["calls"])].append(
                    float(row["mte2_cycles_per_core"])
                )

    points = []
    with case_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["role"] == "control":
                continue
            key = row["factor_id"]
            short_calls, long_calls, samples_per_call = window
            short, long = samples[key, short_calls], samples[key, long_calls]
            if len(short) != samples_per_call or len(long) != samples_per_call:
                raise ValueError(
                    f"{key} requires {samples_per_call} samples for both call counts"
                )
            measured = (median(long) - median(short)) / (long_calls - short_calls)
            if measured <= 0:
                raise ValueError(f"{key} has no positive call slope")
            points.append((key, row["role"], *legacy_cost(row), measured))
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument(
        "--also",
        nargs=2,
        type=Path,
        metavar=("CASES", "SAMPLES"),
        action="append",
        default=[],
    )
    parser.add_argument("--short-window", action="store_true")
    parser.add_argument(
        "--window", nargs=3, type=int, metavar=("SHORT", "LONG", "SAMPLES")
    )
    parser.add_argument(
        "--objective", choices=("relative", "absolute"), default="relative"
    )
    args = parser.parse_args()
    window = args.window or ((1, 4, 4) if args.short_window else (2, 10, 10))
    if window[0] < 1 or window[1] <= window[0] or window[2] < 1:
        parser.error("window requires 0 < SHORT < LONG and SAMPLES > 0")
    points = [
        point
        for cases, samples in [(args.cases, args.samples), *args.also]
        for point in read_points(cases, samples, window)
    ]
    train = [point for point in points if point[1] == "train"]
    if not train:
        parser.error("no training points")
    if args.objective == "absolute":
        scale = sum(r * (y - b) for _, _, b, r, y in train) / sum(
            r * r for _, _, _, r, _ in train
        )
    else:
        scale = sum((r / y) * (1 - b / y) for _, _, b, r, y in train) / sum(
            (r / y) ** 2 for _, _, _, r, y in train
        )
    outcomes = []
    for key, role, base, residual, measured in points:
        after = base + scale * residual
        outcomes.append(
            dict(
                id=key,
                role=role,
                measured=measured,
                before=base + residual,
                after=after,
                error=abs(after / measured - 1),
            )
        )
    max_error = max(row["error"] for row in outcomes if row["role"] == "holdout")
    print(
        json.dumps(
            dict(scale=scale, max_holdout_error=max_error, points=outcomes), indent=2
        )
    )
    return int(max_error > 0.5)


if __name__ == "__main__":
    raise SystemExit(main())
