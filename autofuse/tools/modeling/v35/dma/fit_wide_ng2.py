#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

"""Refit the two bandwidth terms of wide, packed FP16 NG2."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

STARTUP = 454.57389906919423
PAYLOAD_HINGE = 1324.1407381801332
OLD_COSTS = (1139.3068621999473, 208.02520354552965)


def features(row):
    rows, extent, width, tile, cores, rounds, inputs, padding = (
        int(row[key])
        for key in (
            "rows",
            "row_length",
            "inner_width",
            "tile_rows",
            "cores",
            "rounds",
            "inputs",
            "pitch_padding_elt",
        )
    )
    pitch = rows * extent + padding
    if not (
        1282048 <= pitch <= 1648640
        and width in (16, 32)
        and 1 <= tile <= 752
        and 32 <= cores <= 56
    ):
        return None
    tiles_per_row = (extent + tile - 1) // tile
    total_tiles = rows * tiles_per_row
    tiles_per_core = (total_tiles + cores - 1) // cores
    full_payload = []
    for core in range(cores):
        for offset in range(min(rounds, tiles_per_core)):
            slot = core * tiles_per_core + offset
            if slot < total_tiles:
                length = min(tile, extent - (slot % tiles_per_row) * tile)
                full_payload.append(length * width / 8192)
    scale = inputs / cores
    payload = sum(full_payload) * scale
    shared = payload * max(0, cores - 32) / 24
    baseline = scale * (
        STARTUP * len(full_payload)
        + PAYLOAD_HINGE * sum(max(0, value - 1) for value in full_payload)
    )
    return baseline, payload, shared


def fit(train):
    aa, ab, bb, ay, by = (0.0,) * 5
    for _, (baseline, payload, shared), measured in train:
        a, b, y = payload / measured, shared / measured, 1 - baseline / measured
        aa += a * a
        ab += a * b
        bb += b * b
        ay += a * y
        by += b * y
    det = aa * bb - ab * ab
    if det <= 0:
        raise ValueError("Training data must identify both bandwidth terms")
    params = (round((ay * bb - by * ab) / det, 8), round((aa * by - ab * ay) / det, 8))
    if min(params) < 0:
        raise ValueError("Bandwidth terms must have nonnegative costs")
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--train-batch", default="balanced-grid")
    args = parser.parse_args()
    records = defaultdict(list)
    with args.samples.open() as stream:
        for row in csv.DictReader(stream):
            if row["policy"] == "normal":
                records[row["batch"], row["id"]].append(
                    float(row["mte2_cycles_per_core"])
                )

    points = []
    with args.cases.open() as stream:
        for row in csv.DictReader(stream):
            fact = features(row)
            if fact is None:
                continue
            samples = records[row["batch"], row["id"]]
            if len(samples) != 10 or min(samples) <= 0:
                raise ValueError(
                    f"Expected ten positive samples for {row['batch']}/{row['id']}"
                )
            points.append((row, fact, median(samples)))

    params = fit(
        [
            point
            for point in points
            if point[0]["batch"] == args.train_batch and point[0]["role"] == "train"
        ]
    )
    outcomes = []
    for row, (baseline, payload, shared), measured in points:
        before = baseline + payload * OLD_COSTS[0] + shared * OLD_COSTS[1]
        after = baseline + payload * params[0] + shared * params[1]
        outcomes.append(
            {
                "batch": row["batch"],
                "id": row["id"],
                "role": row["role"],
                "measured": measured,
                "before": before,
                "after": after,
                "old_error": abs(before / measured - 1),
                "new_error": abs(after / measured - 1),
            }
        )
    holdout = [row for row in outcomes if row["role"] == "holdout"]
    if not holdout:
        raise ValueError("Independent in-domain holdout is required")
    max_error = max(row["new_error"] for row in holdout)
    print(
        json.dumps(
            {
                "coefficients": params,
                "holdout_count": len(holdout),
                "holdout_max_absolute_relative_error": max_error,
                "points": outcomes,
            },
            indent=2,
        )
    )
    return int(max_error > 0.5)


if __name__ == "__main__":
    raise SystemExit(main())
