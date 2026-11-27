#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

"""Refit physical multirow Load by dtype using the existing shared-bandwidth form."""

import argparse
import csv
import json
from collections import defaultdict
from statistics import median
from pathlib import Path


def fit(points):
    # Weighted least squares for h + inverse_bandwidth * bytes * active_cores.
    # Project a negative intercept to its nonnegative boundary and refit the slope.
    train = [p for p in points if p["role"] == "train"]
    if len(train) < 2:
        raise ValueError("at least two independent training points are required")
    s0 = sx = sxx = sy = sxy = 0.0
    for p in train:
        x, y = p["bytes"] * p["cores"], p["measured"]
        weight = 1.0 / (y * y)
        s0 += weight
        sx += weight * x
        sxx += weight * x * x
        sy += weight * y
        sxy += weight * x * y
    determinant = s0 * sxx - sx * sx
    if determinant <= 0:
        raise ValueError("training payload/core combinations do not identify the fit")
    head = (sy * sxx - sxy * sx) / determinant
    inverse = (s0 * sxy - sx * sy) / determinant
    if head < 0:
        head, inverse = 0.0, sxy / sxx
    if inverse <= 0:
        raise ValueError("fit has no positive shared bandwidth")
    return head, 1.0 / inverse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--policy", choices=("C", "N"), default="C")
    parser.add_argument("--dtype", choices=("float32", "uint8"), default="float32")
    parser.add_argument(
        "--burst-bytes",
        type=int,
        default=0,
        help="Measured GM payload quantum; zero uses actual bytes",
    )
    args = parser.parse_args()
    if args.burst_bytes < 0:
        parser.error("burst bytes must be nonnegative")
    samples = defaultdict(list)
    with args.samples.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["policy"] == args.policy and row["warmup"].lower() == "false":
                samples[row["factor"], int(row["calls"])].append(float(row["cycles"]))
    points = []
    with args.cases.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["dtype"] != args.dtype:
                continue
            windows = [samples[row["factor_id"], k] for k in (1, 2, 3, 4)]
            if any(len(values) != 4 for values in windows):
                raise ValueError(
                    f"{row['factor_id']} requires four active samples per window"
                )
            measured = (median(windows[3]) - median(windows[0])) / 3
            if measured <= 0:
                raise ValueError(f"{row['factor_id']} has no positive call slope")
            rows, width = map(int, row["dims"].split("x"))
            cores = int(row["block_dim"])
            dtype_size = 4 if args.dtype == "float32" else 1
            if int(row["dtype_size"]) != dtype_size:
                raise ValueError("dtype and payload size disagree")
            points.append(
                dict(
                    id=row["factor_id"],
                    role=row["role"],
                    measured=measured,
                    bytes=rows
                    * (
                        (
                            (width * dtype_size + args.burst_bytes - 1)
                            // args.burst_bytes
                        )
                        * args.burst_bytes
                        if args.burst_bytes
                        else width * dtype_size
                    ),
                    cores=cores,
                )
            )
    head, bandwidth = fit(points)
    for point in points:
        point["predicted"] = head + point["bytes"] * point["cores"] / bandwidth
        point["relative_error"] = abs(point["predicted"] / point["measured"] - 1)
    held = [p["relative_error"] for p in points if p["role"] == "holdout"]
    if not held:
        raise ValueError("independent holdout is required")
    print(
        json.dumps(
            dict(
                formula="h + bytes/(a+b/cores)",
                h=head,
                a=0,
                b=bandwidth,
                max_holdout_error=max(held),
                points=points,
            ),
            indent=2,
        )
    )
    return int(max(held) > 0.5)


if __name__ == "__main__":
    raise SystemExit(main())
