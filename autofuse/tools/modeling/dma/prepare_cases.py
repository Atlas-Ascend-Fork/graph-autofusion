#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Translate archived ATT model descriptors into batch bench input."""

import argparse
import csv
import math
import re
from pathlib import Path

LOAD_APIS = {
    "NG2 narrow source FP32": "NDDMA",
    "LoadSmallPayload": "LOAD_NORMAL",
    "LoadSmallPayloadCompact": "LOAD",
    "LoadAlignedSingleRow": "LOAD_NORMAL",
    "LoadAlignedDensePayload": "LOAD_NORMAL",
    "LoadAlignedWideGap": "LOAD_NORMAL",
    "LoadWidePitch128B": "LOAD_NORMAL",
    "LoadAlignedLoopPayload": "LOAD_NORMAL_LOOP2D",
    "LoadNarrowLoopPayload": "LOAD_NORMAL_LOOP2D",
}
STORE_MODEL = "StoreSmallPayload"
LOAD_FIELDS = (
    "factor_id",
    "experiment",
    "role",
    "dtype",
    "dtype_size",
    "block_dim",
    "api",
    "api_rank",
    "dims",
    "gm_or_input_strides",
    "ub_or_output_strides",
    "bytes_per_call",
    "input_span_bytes",
    "output_span_bytes",
    "point_flush",
    "nd_dci",
    "packing_extra",
)
STORE_FIELDS = (
    "id",
    "dtype",
    "block_count",
    "block_bytes",
    "stride_bytes",
    "cores",
    "repeat",
    "role",
)


def axes(value):
    return tuple(map(int, value.split("x")))


def axis_string(values):
    return "x".join(map(str, values))


def prepare(cases, model, policy, roles, point_limit):
    result = []
    seen = set()
    for row in cases:
        if (
            row["model"] != model
            or row["policy"] != policy
            or (roles and row["role"] not in roles)
        ):
            continue
        dims, gm, ub = (
            axes(row[col])
            for col in ("dims", "gm_stride_elements", "ub_stride_elements")
        )
        cores = int(row["blocks"])
        if (
            row["dtype"] != "float32"
            or not (1 <= cores <= 56)
            or not len(dims) == len(gm) == len(ub)
            or any(x <= 0 for x in (*dims, *gm, *ub))
            or (LOAD_APIS.get(model) != "NDDMA" and gm[-1] != 1)
            or ub[-1] != 1
        ):
            raise ValueError(f"Unsupported descriptor: {row['case_id']}")
        key = (dims, gm, ub, cores, row["role"])
        if key in seen:
            continue
        seen.add(key)
        case = re.sub(r"[^a-zA-Z0-9_.-]", "_", row["case_id"])
        payload = math.prod(dims) * 4
        if model == STORE_MODEL:
            if (
                len(dims) != 1
                or not 1024 <= payload <= 10240
                or payload % 32
                or cores < 7
            ):
                raise ValueError(f"Unsupported contiguous store: {row['case_id']}")
            result.extend(
                (
                    f"{case}_{repeat}",
                    "float32",
                    1,
                    payload,
                    0,
                    cores,
                    repeat,
                    row["role"],
                )
                for repeat in (4, 16)
            )
        else:
            if len(dims) == 1:
                dims, gm, ub = (1, dims[0]), (dims[0], 1), (dims[0], 1)
            api = LOAD_APIS[model]
            if len(dims) != (3 if api == "LOAD_NORMAL_LOOP2D" else 2):
                raise ValueError(f"Unexpected descriptor rank: {row['case_id']}")
            gm_span = 4 * (1 + sum((d - 1) * s for d, s in zip(dims, gm)))
            ub_span = 4 * (1 + sum((d - 1) * s for d, s in zip(dims, ub)))
            result.append(
                (
                    case,
                    model,
                    row["role"],
                    "float32",
                    4,
                    cores,
                    api,
                    len(dims),
                    axis_string(dims),
                    axis_string(gm),
                    axis_string(ub),
                    payload,
                    gm_span,
                    ub_span,
                    1,
                    0,
                    0,
                )
            )
        if point_limit is not None and len(seen) >= point_limit:
            break
    if not result:
        raise ValueError(f"No cases for {model}, policy {policy!r}, roles {roles}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("model", choices=(*LOAD_APIS, STORE_MODEL))
    parser.add_argument(
        "policy",
        help="Policy value in the archived CSV, for example N-forward or normal",
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="Limit to an archived role; repeat to select more",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit descriptor count for smoke checks"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        with args.cases.open(newline="") as src:
            rows = prepare(
                csv.DictReader(src), args.model, args.policy, args.role, args.limit
            )
        with args.output.open("x", newline="") as out:
            writer = csv.writer(out, lineterminator="\n")
            writer.writerow(STORE_FIELDS if args.model == STORE_MODEL else LOAD_FIELDS)
            writer.writerows(rows)
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
