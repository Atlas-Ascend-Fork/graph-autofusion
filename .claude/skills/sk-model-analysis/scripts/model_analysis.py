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
    return parser


def main():
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
