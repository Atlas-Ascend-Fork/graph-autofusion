#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.

import sys

from support.npu_runner import NpuRunner


def build_module(torch):
    class CastTranspose(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float32).permute(1, 0).contiguous()

    return CastTranspose()


if __name__ == "__main__":
    sys.exit(NpuRunner.main("CAST_TRANSPOSE", build_module, input_shapes=((64, 64),)))
