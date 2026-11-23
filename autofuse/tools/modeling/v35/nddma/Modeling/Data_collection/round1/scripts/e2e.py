#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COLLECT = SCRIPT_DIR / "collect.py"
FIT = SCRIPT_DIR / "fit.py"
DRAW = SCRIPT_DIR / "draw.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NDDMA2 Round1 1D single/multi-core end-to-end workflow."
    )
    parser.add_argument("action", choices=("all", "collect", "fit", "draw"))
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--measurement-csv", default="")
    parser.add_argument("--model-output-dir", default="")
    parser.add_argument("--msprof-bin", default="")
    parser.add_argument(
        "--npu-arch",
        default="",
        help="Collection architecture; overrides NPU_ARCH in the environment.",
    )
    parser.add_argument("--factor-csv", default="")
    parser.add_argument("--run-output-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def invoke(
    script: Path,
    args: argparse.Namespace,
    extra: list[str],
    data_dir: str = "",
    output_dir: str = "",
    collect_options: bool = False,
) -> int:
    command = [sys.executable, str(script)]
    if data_dir and script == FIT:
        command.append(f"--data-dir={data_dir}")
    elif data_dir and script == DRAW:
        command.append(f"--input-dir={data_dir}")
    if args.measurement_csv and script == FIT:
        command.append(f"--measurement-csv={args.measurement_csv}")
    if output_dir:
        command.append(f"--output-dir={output_dir}")
    if args.npu_arch and collect_options:
        command.append(f"--npu-arch={args.npu_arch}")
    if args.msprof_bin and collect_options:
        command.append(f"--msprof-bin={args.msprof_bin}")
    if args.factor_csv and collect_options:
        command.append(f"--factor-csv={args.factor_csv}")
    command.extend(extra)
    print("[INFO]", " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command).returncode


def main() -> int:
    args = parse_args()
    run_output_dir = args.run_output_dir
    model_output_dir = args.model_output_dir
    if args.action in ("all", "collect"):
        status = invoke(
            COLLECT,
            args,
            ["--dry-run"] if args.dry_run else [],
            output_dir=run_output_dir,
            collect_options=True,
        )
        if status:
            return status
    if args.action in ("all", "fit"):
        status = invoke(
            FIT,
            args,
            [],
            data_dir=args.data_dir or run_output_dir,
            output_dir=model_output_dir,
        )
        if status:
            return status
    if args.action in ("all", "draw"):
        status = invoke(
            DRAW,
            args,
            [],
            data_dir=model_output_dir,
        )
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
