#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

"""跨条件块的 Vector 写入 → GetValue → 广播回归。"""

import re
import sys

from support.npu_runner import NpuRunner

# 大张量沿广播轴分 tile，使 Abs 和 Broadcast 分别进入缓存条件块。
# 不能缩成单 tile：那样数值即使通过，也没有覆盖跨条件块的标量读取。
INPUT_SHAPES = ((8, 1), (8, 16384))


def build_module(torch):
    class ScalarReadSync(torch.nn.Module):
        def forward(self, source, full):
            return torch.abs(source) * full

    return ScalarReadSync()


def check_scalar_read_sync(code):
    """只匹配算子调用所在的兄弟条件块，不接受头文件中的同名 helper。"""
    pattern = (
        r"if\s*\([^{}]+\)\s*\{[^{}]*(?:AscendC::Abs|AbsExtend<float>)\([^{}]*\}"
        r"[^{}]*if\s*\([^{}]+\)\s*\{[^{}]*"
        r"(?P<event>\w+)\s*=\s*\w+\.FetchEventID<AscendC::HardEvent::V_S>\(\);\s*"
        r"AscendC::SetFlag<AscendC::HardEvent::V_S>\((?P=event)\);\s*"
        r"AscendC::WaitFlag<AscendC::HardEvent::V_S>\((?P=event)\);\s*"
        r"float\s+(?P<scalar>\w+)\s*=\s*\w+\.GetValue\(\w+\);\s*"
        r"AscendC::Duplicate\(\w+,\s*(?P=scalar),[^{}]*\}"
        r"[^{}]*AscendC::Mul\("
    )
    if re.search(pattern, code) is None:
        raise RuntimeError(
            "未命中跨条件块的 Abs → V_S → GetValue → Duplicate → Mul，"
            "请检查调度、MLIR 接管和生成的 kernel"
        )


class ScalarReadSyncRunner(NpuRunner):
    def report_pass(self, case_name, backend_route, actual, **kwargs):
        if backend_route == "mlir":
            kernels = [
                path
                for artifact in self.collect_autofuse_artifacts()
                for path in artifact.glob("asc_kernel_*.py")
            ]
            if len(kernels) != 1 or kwargs.get("new_artifact_count", 0) != 1:
                raise RuntimeError(
                    "需要本轮新生成的单个融合 kernel，不能用旧缓存或未融合执行代替："
                    f"kernels={len(kernels)}, new_artifacts={kwargs.get('new_artifact_count')}"
                )
            check_scalar_read_sync(kernels[0].read_text(encoding="utf-8"))
            print(f"scalar_read_sync_kernel={kernels[0]}")
        super().report_pass(case_name, backend_route, actual, **kwargs)


if __name__ == "__main__":
    sys.exit(
        ScalarReadSyncRunner.main(
            "LOAD_BROADCAST_SCALAR_READ_SYNC", build_module, INPUT_SHAPES
        )
    )
