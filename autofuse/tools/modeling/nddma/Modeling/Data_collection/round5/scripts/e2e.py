#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Collect, fit and draw the Round5 3D specialization."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("all", "collect", "fit", "draw"))
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--measurement-csv", action="append", default=[])
    parser.add_argument("--model-output-dir", default="")
    parser.add_argument("--round2-model", default="")
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


def commands(args):
    result = []
    if args.action in ("all", "collect"):
        command = [sys.executable, str(SCRIPT_DIR / "collect.py")]
        for name, value in (
            ("output-dir", args.run_output_dir),
            ("msprof-bin", args.msprof_bin),
            ("npu-arch", args.npu_arch),
            ("factor-csv", args.factor_csv),
        ):
            if value:
                command.append(f"--{name}={value}")
        result.append(command)
    if args.action in ("all", "fit"):
        command = [sys.executable, str(SCRIPT_DIR / "fit.py")]
        for name, value in (
            ("data-dir", args.data_dir or args.run_output_dir),
            ("output-dir", args.model_output_dir),
            ("round2-model", args.round2_model),
        ):
            if value:
                command.append(f"--{name}={value}")
        command.extend(f"--measurement-csv={path}" for path in args.measurement_csv)
        result.append(command)
    if args.action in ("all", "draw"):
        command = [sys.executable, str(SCRIPT_DIR / "draw.py")]
        if args.model_output_dir:
            command.append(f"--input-dir={args.model_output_dir}")
        result.append(command)
    return result


def main():
    args = parse_args()
    for command in commands(args):
        print("[INFO]", shlex.join(command), flush=True)
        if not args.dry_run:
            status = subprocess.run(command).returncode
            if status:
                return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
