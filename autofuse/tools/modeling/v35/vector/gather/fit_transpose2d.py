#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

"""Fit FP32 TransposeExtend<1,2> movement cost with active bank requests."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("windows", type=Path)
    args = parser.parse_args()
    grouped = defaultdict(list)
    with args.windows.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["warmup"].lower() == "false":
                grouped[row["id"], int(row["calls"])].append(float(row["cycles"]))
    points = []
    with args.cases.open(newline="") as stream:
        for row in csv.DictReader(stream):
            m, n, p = (int(row[k]) for k in ("rows", "columns", "pitch"))
            if row["dtype"] != "float32" or row["api"] != "TransposeExtend<1;2>":
                raise ValueError("unexpected API or dtype")
            low, high = (grouped[row["id"], k] for k in (1, 16))
            if len(low) != 10 or len(high) != 10:
                raise ValueError("expected ten active samples in each timing window")
            measured = (median(high) - median(low)) / 15
            if measured <= 0:
                raise ValueError("nonpositive API slope")
            gcd = math.gcd(p, 32)
            gather = m * (2 * gcd * (n // 64) + math.ceil((n % 64) * gcd / 32))
            points.append(
                dict(
                    id=row["id"],
                    role=row["role"],
                    measured=measured,
                    repeats=m * math.ceil(n / 64),
                    gather=gather,
                )
            )
    train = [p for p in points if p["role"] == "train"]
    if len(train) < 2:
        raise ValueError("independent training descriptors required")
    # Preserve the API's 20-cycle VF head and 69-cycle two-axis startup.
    slope = sum(p["repeats"] * (p["measured"] - 89 - p["gather"]) for p in train) / sum(
        p["repeats"] ** 2 for p in train
    )
    if slope <= 0:
        raise ValueError("movement must have positive cost")
    for p in points:
        p["predicted"] = 89 + slope * p["repeats"] + p["gather"]
        p["error"] = abs(p["predicted"] / p["measured"] - 1)
    held = [p["error"] for p in points if p["role"] == "holdout"]
    if not held:
        raise ValueError("independent holdout required")
    print(
        json.dumps(
            dict(
                startup=89,
                movement_cycles=slope,
                max_holdout_error=max(held),
                points=points,
            ),
            indent=2,
        )
    )
    if max(held) > 0.1:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
