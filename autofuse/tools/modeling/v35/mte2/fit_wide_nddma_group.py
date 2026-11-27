#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fit the same-loop wide NDDMA read-group model using archived train rows."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fit(cases_path, samples_path):
    samples = defaultdict(list)
    for row in read_csv(samples_path):
        cycles = float(row["mte2_cycles_per_core"])
        if not math.isfinite(cycles) or cycles <= 0:
            raise ValueError("cycles must be finite and positive")
        samples[row["case_id"], int(row["wide_mask"])].append(cycles)
    points = []
    for case in read_csv(cases_path):
        name, role = case["case_id"], case["role"]
        rows, cols, outer, cores, pitch = (
            int(case[key])
            for key in (
                "tile_rows",
                "tile_cols",
                "outer_rows",
                "active_cores",
                "fp32_pitch_bytes",
            )
        )
        if min(rows, cols, outer, cores) <= 0 or pitch != rows * cols * outer * 16:
            raise ValueError(f"{name}: invalid geometry or GM pitch")
        row_calls = 4 * cols * outer * rows / cores
        values = {}
        for mask in (0, 1, 2, 7):
            lane = samples[name, mask]
            if len(lane) != 4:
                raise ValueError(
                    f"{name}: expected four active samples for mask {mask}"
                )
            values[mask] = median(lane)
        points.append(
            {
                "case_id": name,
                "role": role,
                "pitch": pitch,
                "fp32_increment": [
                    (values[mask] - values[0]) / row_calls for mask in (1, 2)
                ],
                "full_cycles_per_row": values[7] / row_calls,
            }
        )
    train = [point for point in points if point["role"] == "train"]
    pitches = sorted({point["pitch"] for point in train})
    if len(pitches) < 2 or not any(point["role"] == "holdout" for point in points):
        raise ValueError("need distinct training pitches and independent holdouts")
    if any(
        point["role"] in ("holdout", "boundary_holdout") and point["pitch"] in pitches
        for point in points
    ):
        raise ValueError("holdout pitch overlaps training")
    if any(min(point["fp32_increment"]) <= 0 for point in train):
        raise ValueError("training FP32 increments must be positive")

    def evaluate_cut(cut):
        partitions = defaultdict(list)
        for point in train:
            high = cut is not None and point["pitch"] > cut
            partitions[high].append(mean(point["fp32_increment"]))
        fitted = {high: median(values) for high, values in partitions.items()}
        return max(
            abs(fitted[cut is not None and point["pitch"] > cut] / measured - 1)
            for point in train
            for measured in point["fp32_increment"]
        )

    cuts = [(left + right) // 2 for left, right in zip(pitches, pitches[1:])]
    cut = min(cuts, key=evaluate_cut)
    # Use a pitch split only when it reduces FP32 training error by over five percentage points.
    if evaluate_cut(None) <= evaluate_cut(cut) + 0.05:
        cut = None
    bins = defaultdict(list)
    for point in train:
        bins[cut is not None and point["pitch"] > cut].append(
            point["full_cycles_per_row"]
        )
    parameters = {high: median(values) for high, values in bins.items()}
    summary = defaultdict(list)
    for point in points:
        estimated = parameters[cut is not None and point["pitch"] > cut]
        summary[point["role"]].append(abs(estimated / point["full_cycles_per_row"] - 1))
    return {
        "cut_bytes": cut,
        "lower_pitch_cycles_per_row": parameters[False],
        "upper_pitch_cycles_per_row": parameters.get(True),
        "validation": {
            role: {"cases": len(errors), "max_relative_error": max(errors)}
            for role, errors in sorted(summary.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("samples", type=Path)
    args = parser.parse_args()
    try:
        result = fit(args.cases, args.samples)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
