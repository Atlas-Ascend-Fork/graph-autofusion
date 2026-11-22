#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Plot 3D refitted and inherited model residuals by dtype and core count."""

import argparse
import csv
import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR.parents[2] / "Ana/round5"
spec = importlib.util.spec_from_file_location(
    "round5_plot_base", SCRIPT_DIR.parents[1] / "round3/scripts/draw.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    source = Path(args.input_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else source / "figures"
    with (source / "round5_3d_specialized_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("[ERROR] no Round5 predictions found")
    for dtype in sorted({row["dtype"] for row in rows}):
        for core in sorted(
            {int(row["block_dim"]) for row in rows if row["dtype"] == dtype}
        ):
            selected = [
                row
                for row in rows
                if row["dtype"] == dtype and int(row["block_dim"]) == core
            ]
            stem = f"round5_3d_{dtype}_bd{core}"
            for field, suffix, label in (
                ("predicted_cycles", "refit", "3D refit"),
                ("baseline_predicted_cycles", "inherited", "inherited Round2"),
            ):
                common.draw_svg(
                    selected,
                    output / f"{stem}_{suffix}_residual.svg",
                    f"Round5 3D {dtype} k={core}: {label}",
                    True,
                    prediction_field=field,
                )
    print(f"[INFO] wrote plots to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
