#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""att.py - ATT-Analyze 统一 CLI 入口"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="att", description="ATT-Analyze 工具集")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # summary 子命令
    p_summary = subparsers.add_parser("summary", help="解析 ATT 日志，输出算子汇总")
    p_summary.add_argument("log_path", help="日志文件或目录")
    p_summary.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="汇总所有 result 下所有 group 的最佳 case",
    )
    p_summary.add_argument(
        "-f", "--format", choices=["console", "csv", "excel"], default="console"
    )
    p_summary.add_argument("-o", "--output", default=None)

    # compare 子命令
    p_compare = subparsers.add_parser("compare", help="对比两个 CSV 文件")
    p_compare.add_argument("csv1", help="基准 CSV")
    p_compare.add_argument("csv2", help="对比 CSV")
    p_compare.add_argument(
        "-f", "--format", choices=["console", "text", "excel"], default="console"
    )
    p_compare.add_argument("-o", "--output", default=None)

    # verify-tiling 子命令
    p_vt = subparsers.add_parser("verify-tiling", help="TilingFunc 编译+执行验证")
    p_vt.add_argument("source_dir", help="源文件目录（tf 或 inductor）")
    p_vt.add_argument(
        "--scene", choices=["tf", "inductor"], default=None, help="不填则自动检测"
    )
    input_source = p_vt.add_mutually_exclusive_group()
    input_source.add_argument("--preset", choices=["A", "B"], default="A")
    input_source.add_argument(
        "--input-json", default=None, help="输入参数 JSON，与 --preset 互斥"
    )
    p_vt.add_argument(
        "--aiv-num",
        type=int,
        default=None,
        help="覆盖 preset 或 input-json 中的 AI Vector 核数配置",
    )
    p_vt.add_argument(
        "--log", default=None, help="ATT 日志，用于提取输入参数和 --case 默认值"
    )
    p_vt.add_argument("--case", default=None)
    p_vt.add_argument("--compile-config", default=None)
    p_vt.add_argument("--keep-build", action="store_true")
    p_vt.add_argument("-o", "--output", default="output/verify/")

    # split-slog 子命令
    p_ss = subparsers.add_parser("split-slog", help="slog 日志按算子/模板拆分")
    p_ss.add_argument("log_path", help="日志文件或目录")
    p_ss.add_argument("--op", default=None, help="只处理指定算子")
    p_ss.add_argument("--case", default=None)
    p_ss.add_argument("-o", "--output", default="output/split/")

    # perf-formula 子命令
    p_pf = subparsers.add_parser("perf-formula", help="性能公式分析+SVG 可视化")
    p_pf.add_argument("source_dir", help="tiling_func 源文件目录")
    p_pf.add_argument("log_path", help="ATT 结果日志")
    p_pf.add_argument("--case", default=None)
    p_pf.add_argument("-o", "--output", default="output/perf/")

    # evidence 子命令
    p_ev = subparsers.add_parser("evidence", help="导出机器可读 ATT evidence JSONL")
    p_ev.add_argument("log_path", help="日志文件或目录")
    p_ev.add_argument("-o", "--output", required=True, help="输出目录")

    # topn-templates 子命令
    p_topn = subparsers.add_parser(
        "topn-templates", help="生成静态 Top1/TopN 模板与性能表"
    )
    p_topn.add_argument("log_path", help="Autofuse worker 日志文件")
    p_topn.add_argument("-o", "--output", required=True, help="输出目录")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    exit_code = 0
    if args.command == "summary":
        from commands.summary import run

        result = run(args)
    elif args.command == "compare":
        from commands.compare import run

        result = run(args)
    elif args.command == "verify-tiling":
        from commands.verify_tiling import run

        result = run(args)
    elif args.command == "split-slog":
        from commands.split_slog import run

        result = run(args)
    elif args.command == "perf-formula":
        from commands.perf_formula import run

        result = run(args)
    elif args.command == "evidence":
        from commands.evidence import run

        result = run(args)
    elif args.command == "topn-templates":
        from commands.template_topn import run

        result = run(args)
    if isinstance(result, int):
        exit_code = result
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
