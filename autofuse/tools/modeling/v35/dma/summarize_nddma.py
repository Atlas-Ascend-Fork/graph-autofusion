# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate launch/profile pairing and summarize the short-window MTE2 slope."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factors", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples-per-k", type=int, default=10)
    args = parser.parse_args()
    if args.samples_per_k < 1:
        parser.error("--samples-per-k must be positive")
    factors = list(csv.DictReader(args.factors.open()))
    factor_apis = {row["factor_id"]: row["api"] for row in factors}
    launches = [
        json.loads(line)
        for line in (args.run / "launches.jsonl").read_text().splitlines()
    ]
    profiles = list((args.run / "profile").rglob("op_summary*.csv"))
    if len(profiles) != 1:
        raise ValueError("Expected one op_summary CSV")
    rows = [
        r
        for r in csv.DictReader(profiles[0].open())
        if r["Op Name"] == "att_mte2_fixed_pool_kernel"
    ]
    rows.sort(key=lambda r: float(r["Task Start Time(us)"].strip()))
    if len(rows) != len(launches) or {r["sync_mode"] for r in launches} not in (
        {"span_store"},
        {"tail_store"},
    ):
        raise ValueError("Launch/profile count or consumer mode mismatch")
    groups = {}
    for record, row in zip(launches, rows):
        if (
            int(row["Block Num"]) != record["cores"]
            or record["verify"]
            or record["api"] != factor_apis[record["factor"]]
        ):
            raise ValueError("Launch/profile identity mismatch")
        if record["warmup"]:
            continue
        cycles = (
            float(row["aiv_total_cycles"])
            * float(row["aiv_mte2_ratio"])
            / record["cores"]
        )
        groups.setdefault(record["factor"], {}).setdefault(record["calls"], []).append(
            cycles
        )
    output = []
    for factor in factors:
        samples = groups[factor["factor_id"]]
        if set(samples) != {1, 2, 4, 10} or any(
            len(s) != args.samples_per_k for s in samples.values()
        ):
            raise ValueError("Unexpected measured launch count per K=1/2/4/10")
        medians = {k: statistics.median(v) for k, v in sorted(samples.items())}
        mean_k = statistics.mean(medians)
        mean_cycles = statistics.mean(medians.values())
        slope = sum((k - mean_k) * (v - mean_cycles) for k, v in medians.items()) / sum(
            (k - mean_k) ** 2 for k in medians
        )
        if slope <= 0:
            raise ValueError("Non-positive MTE2 slope")
        output.append({**factor, "medians": medians, "cycles_per_api": slope})
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
