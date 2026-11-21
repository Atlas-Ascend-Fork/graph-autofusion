#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Collect the existing 3D grids at single, dual and multiple core counts."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round5"
BLOCK_DIMS = (1, 2, 4, 8, 16, 32, 56)

spec = importlib.util.spec_from_file_location(
    "round5_collection_base", SCRIPT_DIR.parents[1] / "round3/scripts/collect.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def build_rows(execution_repeat_count: int) -> list[dict[str, str]]:
    rows = []
    for source in common.build_rows(execution_repeat_count):
        if source["dim"] != "3":
            continue
        for core in BLOCK_DIMS:
            row = dict(source)
            token = source["token"].replace("r3_", "r5_", 1) + f"_bd{core}"
            row.update(
                {
                    "experiment_idx": str(len(rows) + 1),
                    "token": token,
                    "config_id": token,
                    "sample_id": token,
                    "round_id": "r5_3d_specialized",
                    "stage_define": "51",
                    "sensitivity_id": "R5D3",
                    "block_dim": str(core),
                    "logical_total_bytes": str(int(row["bytes_per_core"]) * core),
                    "fit_role": "train",
                    "model_target": "3D N_G/N_GU/rho refit",
                    "parent_model": "Round2 (frozen N_base)",
                    "notes": "Round3 3D grids; refit corrections, keep Base fixed; no d_0.",
                }
            )
            rows.append(row)
    return rows


def main() -> int:
    args = common.parse_args(
        defaults={
            "factor_csv": str(DEFAULT_ANA_DIR / "round5_3d_factor.csv"),
            "output_dir": str(DEFAULT_ANA_DIR / "collection"),
        },
        description=__doc__,
    )
    return common.run_collection(args, build_rows(args.execution_repeat_count))


if __name__ == "__main__":
    raise SystemExit(main())
