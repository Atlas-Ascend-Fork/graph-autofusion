#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

"""Fit the FP32 short-row Compact Load stride and contention terms."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def read_batch(case_file, sample_file, hardware_loop):
    samples = defaultdict(list)
    with sample_file.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["policy"] == "N":
                samples[row["factor_id"], int(row["calls"])].append(
                    float(row["mte2_cycles_per_core"])
                )
    with case_file.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["role"] == "control" or (
                hardware_loop and row["api"] != "LOAD_COMPACT_LOOP1"
            ):
                continue
            key = row["factor_id"]
            first, last = samples[key, 1], samples[key, 4]
            if len(first) != 4 or len(last) != 4:
                raise ValueError(f"{key} requires four samples at one and four calls")
            measured = (median(last) - median(first)) / 3
            dims = list(map(int, row["dims"].split("x")))
            strides = list(map(int, row["gm_or_input_strides"].split("x")))
            rows, width = dims[-2:]
            pitch = strides[-2]
            if hardware_loop:
                rows *= dims[0]
            cores = int(row["block_dim"])
            if measured <= 0 or pitch < width:
                raise ValueError(f"{key} has an invalid measurement or stride")
            a, b = (11.8292, 6.6155) if hardware_loop else (13.1355, 6.4088)
            base = 160 + 4 * rows * width / (a + b / cores)
            stride = rows * min(4096, 4 * (pitch - width))
            yield key, row["role"], base, stride, cores / 56, measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", nargs=2, type=Path, metavar=("CASES", "SAMPLES"))
    parser.add_argument("--also", nargs=2, type=Path, metavar=("CASES", "SAMPLES"))
    parser.add_argument("--hardware-loop", action="store_true")
    args = parser.parse_args()
    if not args.hardware_loop and args.also is None:
        parser.error("short-row fitting requires --also")
    batches = [args.batch] + ([args.also] if args.also is not None else [])
    points = [
        point
        for batch in batches
        for point in read_batch(*batch, hardware_loop=args.hardware_loop)
    ]
    aa, ab, bb, ay, by = (0.0,) * 5
    for _, role, base, stride, cores, measured in points:
        if role != "train":
            continue
        a, b, y = stride / measured, stride * cores / measured, 1 - base / measured
        aa += a * a
        ab += a * b
        bb += b * b
        ay += a * y
        by += b * y
    det = aa * bb - ab * ab
    if args.hardware_loop:
        if aa <= 0:
            raise ValueError("Training must identify the stride coefficient")
        scale, contention = ay / aa, 0.0
    else:
        if det <= 0:
            raise ValueError("Training must identify both stride coefficients")
        scale = (ay * bb - by * ab) / det
        contention = (aa * by - ab * ay) / det
    results = []
    for key, role, base, stride, cores, measured in points:
        after = base + stride * (scale + contention * cores)
        results.append(
            dict(
                id=key,
                role=role,
                measured=measured,
                before=base + 0.005 * stride,
                after=after,
                error=abs(after / measured - 1),
            )
        )
    max_error = max(row["error"] for row in results if row["role"] == "holdout")
    print(
        json.dumps(
            dict(
                stride_base=scale,
                stride_per_core=contention / 56,
                max_holdout_error=max_error,
                points=results,
            ),
            indent=2,
        )
    )
    return int(min(scale, contention) < 0 or max_error > 0.5)


if __name__ == "__main__":
    raise SystemExit(main())
