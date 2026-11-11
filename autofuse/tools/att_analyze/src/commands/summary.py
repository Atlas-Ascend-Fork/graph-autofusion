#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""summary 子命令：调度 summary_templates 逻辑"""

import sys


def run(args):
    import summary_templates

    argv = [args.log_path, "-f", args.format]
    if getattr(args, "all", False):
        argv.append("--all")
    if getattr(args, "topn", False):
        argv.append("--topn")
    if args.output:
        argv += ["-o", args.output]
    old_argv = sys.argv[:]
    sys.argv = ["summary_templates"] + argv
    try:
        return summary_templates.main()
    finally:
        sys.argv = old_argv
