#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

from support.npu_runner import NpuRunner


def build_module(torch):
    class TransposeSum(torch.nn.Module):
        def forward(self, x):
            return x.permute(0, 2, 1).sum(dim=-1, keepdim=True)

    return TransposeSum()


if __name__ == "__main__":
    sys.exit(
        NpuRunner.main(
            "TRANSPOSE_SUM",
            build_module,
            input_shapes=((2, 37, 23),),
            rtol=1e-5,
            atol=1e-5,
        )
    )
