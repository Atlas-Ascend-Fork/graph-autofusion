#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

"""Refit the bandwidth of wide-pitch FP16 Compact Load."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("samples", type=Path)
    args = parser.parse_args()
    samples = defaultdict(list)
    with args.samples.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["policy"] == "N":
                samples[row["factor_id"], int(row["calls"])].append(
                    float(row["mte2_cycles_per_core"])
                )

    points = []
    with args.cases.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["factor_id"]
            short, long = samples[key, 2], samples[key, 10]
            if len(short) != 10 or len(long) != 10:
                raise ValueError(f"{key} requires ten samples at two and ten calls")
            measured = (median(long) - median(short)) / 8
            if measured <= 0:
                raise ValueError(f"{key} has no positive call slope")
            rows, width = map(int, row["dims"].split("x"))
            step, _ = map(int, row["gm_or_input_strides"].split("x"))
            cores = int(row["block_dim"])
            base = 2 * rows * width / (13.1355 + 6.4088 / cores)
            stride = 0.005 * rows * min(4096, 2 * (step - width))
            points.append((key, row["role"], base, stride, measured))

    train = [point for point in points if point[1] == "train"]
    scale = sum(x / y * (1 - 160 / y) for _, _, x, _, y in train) / sum(
        (x / y) ** 2 for _, _, x, _, y in train
    )
    results = []
    for key, role, base, stride, measured in points:
        after = 160 + scale * base
        results.append(
            dict(
                id=key,
                role=role,
                measured=measured,
                before=160 + base + stride,
                after=after,
                error=abs(after / measured - 1),
            )
        )
    max_error = max(x["error"] for x in results if x["role"] == "holdout")
    print(
        json.dumps(
            dict(
                payload_scale=scale,
                a=13.1355 / scale,
                b=6.4088 / scale,
                max_holdout_error=max_error,
                points=results,
            ),
            indent=2,
        )
    )
    return int(max_error > 0.5)


if __name__ == "__main__":
    raise SystemExit(main())
