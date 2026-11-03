#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
ASCIR_TOOL = (
    REPO_ROOT / "autofuse" / "tests" / "st" / "codegen" / "ascir_tool" / "test_ascir.sh"
)
ARTIFACT_ROOT = REPO_ROOT / "autofuse" / "mlir" / ".artifacts"


def status_filename(case_name: str) -> str:
    return case_name.replace("/", "_").replace("\\", "_") + ".status"


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--jobs must be a positive integer.") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("--jobs must be a positive integer.")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Autofuse ascir_tool case and record its status."
    )
    parser.add_argument(
        "--case", required=True, help="ascir_tool testcase directory name."
    )
    parser.add_argument("--mode", default="0", help="ascir_tool mode, default: 0.")
    parser.add_argument(
        "-j",
        "--jobs",
        type=positive_int,
        default=8,
        help="ascir_tool build parallelism, default: 8.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(ASCIR_TOOL), f"--mode={args.mode}", f"--case={args.case}"]
    env = os.environ.copy()
    existing_makeflags = env.get("MAKEFLAGS", "")
    env["MAKEFLAGS"] = f"{existing_makeflags} -j{args.jobs}".strip()
    result = subprocess.run(
        cmd, cwd=str(ASCIR_TOOL.parent), env=env, text=True, check=False
    )

    report = ARTIFACT_ROOT / status_filename(args.case)
    report.write_text(
        f"case={args.case}\n"
        f"mode={args.mode}\n"
        f"jobs={args.jobs}\n"
        f"returncode={result.returncode}\n",
        encoding="utf-8",
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
