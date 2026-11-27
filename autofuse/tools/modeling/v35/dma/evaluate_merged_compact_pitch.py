#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Evaluate the unchanged FP32 Compact Load formula with and without GM pitch."""

import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import median


def read_profile_runs(root):
    for policy in ("N", "C"):
        run = root / policy
        profiles = list(
            (run / "profile").glob("PROF*/mindstudio_profiler_output/op_summary*.csv")
        )
        if len(profiles) != 1:
            raise ValueError(f"{policy} requires exactly one exported op_summary")
        with profiles[0].open(newline="") as stream:
            tasks = sorted(
                (
                    row
                    for row in csv.DictReader(stream)
                    if row["Op Name"] == "att_mte2_fixed_pool_kernel"
                ),
                key=lambda row: Decimal(row["Task Start Time(us)"].strip()),
            )
        launches = [
            json.loads(line)
            for line in (run / "launches.jsonl").read_text().splitlines()
        ]
        if len(launches) != len(tasks):
            raise ValueError(f"{policy} launch and profile rows disagree")
        expected_policy = {"N": "N_fresh_normal", "C": "C_unique_disable"}[policy]
        for index, (launch, task) in enumerate(zip(launches, tasks)):
            cores = launch["cores"]
            if (
                launch["launch"] != index
                or launch["protocol"] != expected_policy
                or launch["sync_mode"] != "transpose_store"
                or launch["address_policy"] != "production_tile_grid"
                or cores <= 0
                or int(task["Block Num"]) != cores
            ):
                raise ValueError(f"{policy} launch metadata disagree at {index}")
            yield {
                "policy": policy,
                "factor": launch["factor"],
                "calls": launch["calls"],
                "warmup": str(launch["warmup"]),
                "mte2_cycles_per_core": (
                    float(task["aiv_total_cycles"])
                    / cores
                    * float(task["aiv_mte2_ratio"])
                ),
            }


def validate(cases, samples):
    windows = defaultdict(list)
    for sample in samples:
        if sample["warmup"].lower() == "false":
            key = (sample["policy"], sample["factor"], int(sample["calls"]))
            windows[key].append(float(sample["mte2_cycles_per_core"]))

    shapes = defaultdict(set)
    points = []
    for row in cases:
        rows, width = map(int, row["dims"].split("x"))
        pitch = int(row["gm_or_input_strides"].split("x")[0])
        cores = int(row["block_dim"])
        role = row["role"]
        if role not in {"train", "holdout", "control"} or row["dtype"] != "float32":
            raise ValueError("unsupported role or data type")
        if not (rows > 1 and pitch > width > 0 and cores > 0):
            raise ValueError(f"invalid descriptor for {row['factor_id']}")
        shapes[role].add((rows, width, pitch))
        payload_only = 160 + rows * width * 4 / (11.8292 + 6.6155 / cores)
        with_pitch = payload_only + 0.005 * rows * min((pitch - width) * 4, 4096)
        for policy in ("N", "C"):
            api = row["factor_id"]
            call_counts = (16, 32, 48, 64)
            points_by_window = [windows[policy, api, calls] for calls in call_counts]
            if any(len(values) != 4 for values in points_by_window):
                raise ValueError(
                    f"need four active samples per window for {policy}/{api}"
                )
            observed = (median(points_by_window[-1]) - median(points_by_window[0])) / 48
            if observed <= 0:
                raise ValueError(f"nonpositive call slope for {policy}/{api}")
            points.append(
                {
                    "policy": policy,
                    "factor": api,
                    "role": role,
                    "measured": observed,
                    "payload_only": payload_only,
                    "with_pitch": with_pitch,
                }
            )
    if (
        not shapes["train"]
        or not shapes["holdout"]
        or shapes["train"] & shapes["holdout"]
    ):
        raise ValueError("training and holdout need independent descriptors")
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument(
        "samples",
        type=Path,
        help="Archived CSV or directory containing N and C profiles",
    )
    args = parser.parse_args()
    with args.cases.open(newline="") as cases:
        if args.samples.is_dir():
            points = validate(csv.DictReader(cases), read_profile_runs(args.samples))
        else:
            with args.samples.open(newline="") as samples:
                points = validate(csv.DictReader(cases), csv.DictReader(samples))
    summary = {}
    for policy in ("N", "C"):
        holdout = [x for x in points if (x["policy"], x["role"]) == (policy, "holdout")]
        for model in ("payload_only", "with_pitch"):
            errors = [abs(x[model] / x["measured"] - 1) for x in holdout]
            summary[f"{policy}/{model}"] = {
                "count": len(errors),
                "median": median(errors),
                "max": max(errors),
            }
    print(json.dumps({"holdout_relative_error": summary, "points": points}, indent=2))


if __name__ == "__main__":
    main()
