#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

"""Fit the FP16 TransposeExtend<1, 3> masked row-loop cost."""

import argparse
import csv
import json
from collections import defaultdict
from math import gcd
from statistics import mean, median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases")
    parser.add_argument("samples")
    args = parser.parse_args()
    indexed = defaultdict(list)
    with open(args.samples) as stream:
        for row in csv.DictReader(stream):
            indexed[row["id"], int(row["calls"])].append(float(row["vec_cycles"]))

    data = []
    with open(args.cases) as stream:
        for row in csv.DictReader(stream):
            key = row["id"]
            dimensions = ("rows", "middle", "lanes", "pitch")
            rows, middle, lanes, pitch = [int(row[field]) for field in dimensions]
            short = indexed[key, 4]
            long = indexed[key, 16]
            if len(short) != 10 or len(long) != 10:
                raise ValueError(f"Expected ten samples in both windows for {key}")
            cycles = (median(long) - median(short)) / 12 / (rows * middle)
            if cycles <= 0:
                raise ValueError(f"Invalid VEC cycle slope for {key}")
            groups = lanes * gcd(pitch * middle // 16, 4) / 4
            data.append((key, row["role"], groups, cycles))

    train = [(groups, cycles) for _, role, groups, cycles in data if role == "train"]
    fits = []
    for pivot in sorted({g for g, _ in train})[:-1]:
        low = [cycles for groups, cycles in train if groups <= pivot]
        high = [(groups, cycles) for groups, cycles in train if groups > pivot]
        floor = mean(low)
        serial = mean(cycles - groups for groups, cycles in high)
        if pivot + serial <= floor <= min(groups for groups, _ in high) + serial:
            error = sum(
                (max(floor, groups + serial) - cycles) ** 2 for groups, cycles in train
            )
            fits.append((error, floor, serial))
    _, floor, serial = min(fits)
    points = []
    for key, role, groups, cycles in data:
        estimate = max(floor, groups + serial)
        points.append(
            {
                "case": key,
                "role": role,
                "measured_cycles_per_row": cycles,
                "estimated_cycles_per_row": estimate,
                "relative_error": abs(estimate / cycles - 1),
            }
        )
    max_error = max(
        point["relative_error"] for point in points if point["role"] == "holdout"
    )
    print(
        json.dumps(
            {
                "row_floor": floor,
                "serial_base": serial,
                "holdout_max": max_error,
                "points": points,
            },
            indent=2,
        )
    )
    return int(max_error > 0.5)


if __name__ == "__main__":
    raise SystemExit(main())
