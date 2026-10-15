#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------
# fmt: off
# Imports
import torch.nn as nn
import numpy as np
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

def superkernel_scope():
    _src_graph = '''
    |o>--------------------------------------------------
    |o>test case: %s
    |o>                    data
    |o>                      |
    |o>      sk1:GroupedMatmul+GroupedMatmul+MoeGatingTopK
    |o>                      |
    |o>                reshape-square-concat
    |o>                      |
    |o>     sk2:DequantSwigluQuant+QuantBatchMatmulV3
    |o>                      |
    |o>                  netoutput
    |o>--------------------------------------------------
    '''

    torch.npu.set_device(0)

    gmm1_x1 = torch.from_numpy(np.random.uniform(-1, 1, size=(64, 128))).to(torch.float16).npu()
    gmm1_x2 = torch.from_numpy(np.random.uniform(-1, 1, size=(256, 180))).to(torch.float16).npu()
    gmm1_x = [gmm1_x1, gmm1_x2]

    gmm1_weight1 = torch.from_numpy(np.random.uniform(-1, 1, size=(128, 64))).to(torch.float16).npu()
    gmm1_weight2 = torch.from_numpy(np.random.uniform(-1, 1, size=(180, 320))).to(torch.float16).npu()
    gmm1_weight = [gmm1_weight1, gmm1_weight2]

    gmm1_bias1 = torch.from_numpy(np.random.uniform(-2, 2, size=(64))).to(torch.float16).npu()
    gmm1_bias2 = torch.from_numpy(np.random.uniform(-2, 2, size=(320))).to(torch.float16).npu()
    gmm1_bias = [gmm1_bias1, gmm1_bias2]

    gmm2_weight1 = torch.from_numpy(np.random.uniform(1, 1, size=(64, 56))).to(torch.float16).npu()
    gmm2_weight2 = torch.from_numpy(np.random.uniform(1, 1, size=(320, 256))).to(torch.float16).npu()
    gmm2_weight = [gmm2_weight1, gmm2_weight2]

    moe1_bias = torch.from_numpy(np.random.uniform(-2, 2, size=(256, ))).to(torch.float16).npu()

    dsq1_weight_scale = torch.from_numpy(np.random.uniform(1, 1, size=(128, ))).to(torch.float32).npu()
    dsq1_activate_scale = torch.from_numpy(np.random.uniform(1, 1, size=(48, 1))).to(torch.float32).npu()
    _dsq1_bias = None
    dsq1_quant_scale = torch.from_numpy(np.random.uniform(1, 1, size=(1, 64))).to(torch.float32).npu()
    dsq_input = [dsq1_weight_scale, dsq1_activate_scale, dsq1_quant_scale]

    data1 = torch.from_numpy(np.random.uniform(-5, 5, size=(4, 8, 128))).to(torch.int32).npu()
    data2 = torch.from_numpy(np.random.uniform(-5, 5, size=(6, 64, 64))).to(torch.int8).npu()
    scale = torch.from_numpy(np.random.uniform(1, 1, size=(1, ))).to(torch.int64).npu()

    # Custom model
    class Network(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, gmm1_x, gmm1_weight, gmm1_bias, gmm2_weight, moe1_bias, dsq_input, data1, data2, scale):
            with tng.scope.super_kernel("sk1"):
                grouped_matmul_01 = torch_npu.npu_grouped_matmul(group_type=-1, x=gmm1_x, weight=gmm1_weight,
                                                                 bias=gmm1_bias)
                grouped_matmul_02 = torch_npu.npu_grouped_matmul(group_type=-1, x=grouped_matmul_01, weight=gmm2_weight)
                moe_gating_top_k_01 = torch_npu.npu_moe_gating_top_k(x=grouped_matmul_02[1], bias=moe1_bias,
                                                        k=8, k_group=4, group_count=8, group_select_mode=1, norm_type=1)

            reshape_01 = torch.reshape(moe_gating_top_k_01[1], (2, 8, 128))
            square_01 = torch.square(reshape_01)
            concat_01 = torch.cat((square_01, data1), dim=0)
            reshape_02 = torch.reshape(concat_01, (-1, 128))

            with tng.scope.super_kernel("sk2"):
                dequant_swiglu_quant_01 = torch_npu.npu_dequant_swiglu_quant(x=reshape_02, weight_scale=dsq_input[0],
                    activation_scale=dsq_input[1], bias=None, quant_scale=dsq_input[2], quant_offset=None,
                    group_index=None, activate_left=False, quant_mode=1)
                quant_matmul_01 = torch_npu.npu_quant_matmul(x1=dequant_swiglu_quant_01[0], x2=data2, scale=scale,
                    offset=None, bias=None, pertoken_scale=None, output_dtype=torch.float16)
            return quant_matmul_01

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    model = Network().npu()

    # Run the model configured with SuperKernel on the NPU
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    _npu_output = model(gmm1_x, gmm1_weight, gmm1_bias, gmm2_weight, moe1_bias, dsq_input, data1, data2, scale)
    print("execute sample success")

superkernel_scope()
# fmt: on
