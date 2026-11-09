#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Public CLI for independent graph parsing and offline Graph / DFX reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _retired(args):
    print(
        "Analysis View / diagnose-performance 已下线；请使用 analyze 查看 Graph / DFX。真实耗时分析暂不提供。",
        file=sys.stderr,
    )
    return 2


def _trace(args):
    command = [
        sys.executable,
        str(Path(__file__).with_name("sk_node_tracing.py")),
        args.input,
    ]
    if args.output:
        command.extend(["-o", str(args.output)])
    if args.model_instance_id:
        command.extend(["--model-instance-id", args.model_instance_id])
    return subprocess.run(command, check=False).returncode


def build_parser():
    from graph_report.cli import add_arguments, generate

    parser = argparse.ArgumentParser(description="SK execution graph and DFX analysis")
    commands = parser.add_subparsers(dest="subcommand", required=True)
    for name, help_text in (
        ("analyze", "Generate unified offline Graph / DFX report"),
        (
            "diagnose-hang-crash",
            "Generate the same report with runtime and parser diagnostics",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        add_arguments(command)
        command.set_defaults(func=generate)
    retired = commands.add_parser(
        "diagnose-performance", help="Retired: Analysis View has been removed"
    )
    retired.add_argument("input", nargs="?")
    retired.set_defaults(func=_retired)
    trace = commands.add_parser(
        "trace-nodes", help="Export standalone node tracing (no timing analysis)"
    )
    trace.add_argument("input")
    trace.add_argument("-o", "--output", type=Path)
    trace.add_argument(
        "--model-instance-id", help="Select optimize instance for tracing"
    )
    trace.set_defaults(func=_trace)
    return parser


def main():
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
