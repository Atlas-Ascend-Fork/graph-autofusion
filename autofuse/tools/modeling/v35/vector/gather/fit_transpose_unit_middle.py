#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

"""Fit the FP32 TransposeExtend<3,3> tail cost from task-based PMU samples."""

import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import median


def read_samples(profile, launches):
    with profile.open(newline="") as stream:
        events = sorted(
            (
                row
                for row in csv.DictReader(stream)
                if "transpose_ub2d" in row["Op Name"]
            ),
            key=lambda row: Decimal(row["Task Start Time(us)"].strip()),
        )
    samples = list(map(json.loads, launches.read_text().splitlines()))
    if len(events) != len(samples):
        raise ValueError("profiler task count differs from the collector manifest")
    for event, sample in zip(events, samples):
        ratio = float(event["aiv_vec_ratio"])
        if int(event["Block Num"]) != 1 or not 0 < ratio <= 1:
            raise ValueError(
                "expected one active vector core and valid VEC utilization"
            )
        yield dict(
            point=sample["case"].rsplit("_", 1)[0],
            calls=sample["calls"],
            sample=sample["sample"],
            descriptor=(sample["rows"], sample["columns"]),
            aiv_total_cycles=float(event["aiv_total_cycles"]),
            aiv_vec_ratio=ratio,
        )


def read_archive(path):
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            yield dict(
                point=row["point"],
                calls=int(row["calls"]),
                sample=int(row["sample"]),
                aiv_total_cycles=float(row["aiv_total_cycles"]),
                aiv_vec_ratio=float(row["aiv_vec_ratio"]),
            )


def fit(cases, samples):
    grouped = defaultdict(dict)
    descriptors = {}
    for row in samples:
        key = (row["point"], row["calls"])
        if "descriptor" in row:
            descriptor = descriptors.setdefault(row["point"], row["descriptor"])
            if descriptor != row["descriptor"]:
                raise ValueError(f"mismatched launch descriptors for {row['point']}")
        index = row["sample"]
        if index in grouped[key]:
            raise ValueError(f"duplicate sample {key}/{index}")
        grouped[key][index] = row["aiv_total_cycles"] * row["aiv_vec_ratio"]
    points = []
    with cases.open(newline="") as stream:
        for row in csv.DictReader(stream):
            ident = row["point"]
            medians = {}
            for calls in (1, 4, 8, 16):
                window = grouped.pop((ident, calls))
                if set(window) != set(range(5)):
                    raise ValueError(
                        f"expected one warmup and four active samples for {ident}/{calls}"
                    )
                medians[calls] = median(window[index] for index in range(1, 5))
            measured = (medians[16] - medians[1]) / 15
            if measured <= 0:
                raise ValueError(f"nonpositive VEC slope for {ident}")
            rows, width = int(row["rows"]), int(row["width"])
            if ident in descriptors and descriptors[ident] != (rows, width):
                raise ValueError(f"{ident} launch does not match the frozen descriptor")
            repeats = (rows * width + 63) // 64
            points.append(
                dict(
                    point=ident,
                    role=row["role"],
                    measured=measured,
                    legacy=408 + 18 * repeats,
                    tail=repeats - 1,
                )
            )
    train = [point for point in points if point["role"] == "train"]
    holdout = [point for point in points if point["role"] == "holdout"]
    if grouped or len(train) < 2 or not holdout:
        raise ValueError(
            "sample and descriptor lists must match and include training and holdout"
        )
    slope = sum(p["tail"] * (p["measured"] - p["legacy"]) for p in train) / sum(
        p["tail"] ** 2 for p in train
    )
    for point in points:
        point["corrected"] = point["legacy"] + slope * point["tail"]
    print(json.dumps(dict(coefficient=slope, rows=points), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument(
        "samples", type=Path, help="Archived CSV or msprof op_summary with --launches"
    )
    parser.add_argument(
        "--launches",
        type=Path,
        help="Collector launch JSONL for matching a fresh op_summary",
    )
    args = parser.parse_args()
    samples = (
        read_samples(args.samples, args.launches)
        if args.launches
        else read_archive(args.samples)
    )
    fit(args.cases, samples)


if __name__ == "__main__":
    main()
