#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Extract audited group medians and replay the v35 common launch cost fit."""

import argparse
import csv
import json
import math
import sys
from decimal import Decimal
from itertools import groupby
from pathlib import Path
from statistics import mean, median

ROUNDS = dict(
    zip(("p1", "p2", "p3", "p4"), ("forward", "reverse", "reverse", "forward"))
)
BRIDGE_ROLES = (
    dict.fromkeys((128, 512, 2048), "train")
    | dict.fromkeys((256, 1024), "holdout")
    | {0: "control"}
)
HEAD_ROLES = dict.fromkeys((3, 6, 12, 24, 36, 48), "train") | dict.fromkeys(
    (5, 15, 18, 30, 42, 50), "holdout"
)
CASES = ("bench", "round", "order", "cores", "work", "role")
SAMPLES = (*CASES, "samples", "task_us", "pmu_cycles_per_block")


def plan():
    for tag, order in ROUNDS.items():
        reverse = order == "reverse"
        for cores in sorted((3, 24, 48), reverse=reverse):
            for repeats in sorted(BRIDGE_ROLES, reverse=reverse):
                yield ("bridge", tag, order, cores, str(repeats), BRIDGE_ROLES[repeats])
        for cores in sorted(HEAD_ROLES, reverse=reverse):
            for variant in ("empty", "stamp"):
                yield (
                    "head",
                    tag,
                    order,
                    cores,
                    variant,
                    HEAD_ROLES[cores] if variant == "empty" else "control",
                )


def require(ok, message):
    if not ok:
        raise ValueError(message)


def medians(run):
    for (bench, tag), cases in groupby(plan(), lambda row: row[:2]):
        prefix = run / bench / tag
        with prefix.with_suffix(".jsonl").open() as stream:
            launches = [json.loads(line) for line in stream]
        (summary,) = prefix.glob("PROF*/mindstudio_profiler_output/op_summary*.csv")
        with summary.open(newline="") as stream:
            prof = list(csv.DictReader(stream))
        receipts = [
            line
            for line in prefix.with_suffix(".log").read_text().splitlines()
            if line.startswith("CASE_PASS ")
        ]
        cases = list(cases)
        require(
            len(launches) == len(prof) == len(cases) * 34,
            f"{prefix}: launch/profile count mismatch",
        )
        require(
            receipts == [f"CASE_PASS {c} {work}" for _, _, _, c, work, _ in cases],
            f"{prefix}: output check mismatch",
        )
        starts = [Decimal(row["Task Start Time(us)"]) for row in prof]
        require(
            all(a < b for a, b in zip(starts, starts[1:]))
            and len({(row["Device_id"], row["Stream ID"]) for row in prof}) == 1,
            f"{prefix}: profile contains unordered samples or multiple streams",
        )
        for index, case in enumerate(cases):
            _, _, order, cores, work, _ = case
            name = f"_Z11core_launchILb{int(work == 'stamp')}EEvPh"
            times, cycles = [], []
            for sample in range(34):
                launch, row = launches[index * 34 + sample], prof[index * 34 + sample]
                require(
                    (
                        launch["order"],
                        launch["cores"],
                        launch["sample"],
                        launch["warmup"],
                    )
                    == (order, cores, sample, sample < 3)
                    and str(
                        launch["repeats"] if bench == "bridge" else launch["variant"]
                    )
                    == work
                    and int(row["Block Num"]) == cores
                    and row["Task Type"] == "AI_VECTOR_CORE"
                    and int(row["Mix Block Num"]) == 0
                    and (
                        ("cycle_bridge" in row["Op Name"])
                        if bench == "bridge"
                        else row["Op Name"] == name
                    ),
                    f"{prefix}: task identity mismatch at sample {index * 34 + sample}",
                )
                if sample >= 3:
                    t, p = (
                        float(row["Task Duration(us)"]),
                        float(row["aiv_total_cycles"]) / cores,
                    )
                    require(
                        math.isfinite(t) and t > 0 and math.isfinite(p) and p > 0,
                        f"{prefix}: invalid counters",
                    )
                    times.append(t)
                    cycles.append(p)
            yield (*case, len(times), median(times), median(cycles))


def affine(points):
    x, y = zip(*points)
    xm, ym = mean(x), mean(y)
    slope = sum((a - xm) * (b - ym) for a, b in points) / sum((a - xm) ** 2 for a in x)
    return slope, ym - slope * xm


def fit(samples):
    require(
        [row[:6] for row in samples] == list(plan()),
        "sample identities or splits differ from the protocol",
    )
    require(
        all(
            n == 31 and 0 < t < math.inf and 0 < p < math.inf for *_, n, t, p in samples
        ),
        "expected 31 valid observations per group",
    )
    ratios, errors = [], []
    bridge = (row for row in samples if row[0] == "bridge")
    for _, group in groupby(bridge, lambda row: (row[1], row[3])):
        rows = list(group)
        train = [row for row in rows if row[5] == "train"]
        time_slope, time_offset = affine([(int(row[4]), row[7]) for row in train])
        pmu_slope, pmu_offset = affine([(int(row[4]), row[8]) for row in train])
        require(time_slope > 0 and pmu_slope > 0, "invalid cycle bridge fit")
        ratio = pmu_slope / time_slope
        ratios.append(ratio)
        offset = time_offset - pmu_offset / ratio
        errors.extend(
            abs(offset + row[8] / ratio - row[7]) / row[7]
            for row in rows
            if row[5] == "holdout"
        )
    ratio = median(ratios)
    head = [row for row in samples if row[0] == "head" and row[4] == "empty"]
    output = {
        "bridge_cycles_per_us": ratio,
        "bridge_range": [min(ratios), max(ratios)],
        "bridge_holdout_max_relative_error": max(errors),
        "models": {},
    }
    for name, metric in {
        "task_cycles": lambda row: row[7] * ratio,
        "pmu_cycles": lambda row: row[8],
        "task_minus_pmu": lambda row: row[7] * ratio - row[8],
    }.items():
        slope, intercept = affine(
            [(row[3], metric(row)) for row in head if row[5] == "train"]
        )
        holdout = [(row[3], metric(row)) for row in head if row[5] == "holdout"]
        output["models"][name] = {
            "slope": slope,
            "intercept": intercept,
            "holdout_max_relative_error": max(
                abs(slope * x + intercept - y) / abs(y) for x, y in holdout
            ),
        }
    return output


def read_samples(path):
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            yield (
                row["bench"],
                row["round"],
                row["order"],
                int(row["cores"]),
                row["work"],
                row["role"],
                int(row["samples"]),
                float(row["task_us"]),
                float(row["pmu_cycles_per_block"]),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "fit"))
    parser.add_argument(
        "input", type=Path, help="run directory for collect; samples.csv for fit"
    )
    args = parser.parse_args()
    try:
        if args.action == "collect":
            samples = list(medians(args.input))
            writer = csv.writer(sys.stdout, lineterminator="\n")
            writer.writerow(SAMPLES)
            writer.writerows(samples)
        else:
            print(json.dumps(fit(list(read_samples(args.input))), indent=2))
    except (ValueError, OSError, KeyError, ArithmeticError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
