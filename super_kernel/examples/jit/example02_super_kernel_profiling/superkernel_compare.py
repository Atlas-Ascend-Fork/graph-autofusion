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

import numpy as np
import torch
import torch.nn as nn
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig


class FakeContextManager:
    def __init__(self) -> None:
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def super_kernel_scope(enable_superkernel: bool, scope: str, options: str):
    if enable_superkernel:
        return tng.scope.super_kernel(scope, options)
    else:
        return FakeContextManager()



def superkernel_compare():
    _src_graph = '''
    |o>--------------------------------------------------
    |o>test case: %s
    |o>              split+matmul+concat+permute+reducemean
    |o>                      |
    |o>     sk1:GroupedMatmul+MoeGatingTopK+GroupedMatmul+DequantSwigluQuant+cast+GroupedMatmul
    |o>                      |
    |o>              cast+matmul+permute+concat+narrow+reducemean
    |o>                      |
    |o>     sk2:GroupedMatmul+MoeGatingTopK+GroupedMatmul+DequantSwigluQuant+cast+GroupedMatmul
    |o>                      |
    |o>              concat+transpose+repeat+cast+reducemean
    |o>                      |
    |o>     sk3: DequantSwigluQuant+DequantSwigluQuant+DequantSwigluQuant
    |o>                      |
    |o>     sk4:GroupedMatmul+MoeGatingTopK+GroupedMatmul+GroupedMatmul
    |o>                      |
    |o>              narrow+concat+transpose+repeat
    |o>                      |
    |o>     sk5:GroupedMatmul+MoeGatingTopK+GroupedMatmul+GroupedMatmul
    |o>                      |
    |o>              narrow+concat+transpose+repeat
    |o>                      |
    |o>     sk6:GroupedMatmul+MoeGatingTopK+GroupedMatmul+DequantSwigluQuant+GroupedMatmul
    |o>                      |
    |o>              netoutput
    |o>--------------------------------------------------
    |o> sk fusion results：
    |o>     sk1:  scope1[GroupedMatmul+MoeGatingTopK+GroupedMatmul+GroupedMatmul]   scope2[DequantSwigluQuant]
    |o>     sk2:  scope1[GroupedMatmul+MoeGatingTopK+GroupedMatmul+GroupedMatmul]   scope2[DequantSwigluQuant]
    |o>     sk3:  scope1[DequantSwigluQuant]   scope2[DequantSwigluQuant+DequantSwigluQuant]
    |o>     sk4:GroupedMatmul+MoeGatingTopK+GroupedMatmul+GroupedMatmul
    |o>     sk5:GroupedMatmul+MoeGatingTopK+GroupedMatmul+GroupedMatmul
    |o>     sk6:GroupedMatmul+MoeGatingTopK+GroupedMatmul+GroupedMatmul
    |o>--------------------------------------------------
    '''

    torch.npu.set_device(0)
    data1 = torch.from_numpy(np.random.uniform(-1, 1, size=(128, 64))).to(torch.float32).npu()
    data2 = torch.from_numpy(np.random.uniform(-1, 1, size=(8, 320))).to(torch.float32).npu()

    gmm1_x1 = torch.from_numpy(np.random.uniform(-1, 1, size=(256, 8))).to(torch.float32).npu()
    gmm1_weight = torch.from_numpy(np.random.uniform(-1, 1, size=(320, 32))).to(torch.float32).npu()

    moe1_bias = torch.from_numpy(np.random.uniform(-2, 2, size=(256, ))).to(torch.float32).npu()
    arn1_x2 = torch.from_numpy(np.random.uniform(-3, 3, size=(256, 8))).to(torch.float32).npu()
    arn1_gamma = torch.from_numpy(np.random.uniform(-3, 3, size=(256, 8))).to(torch.float32).npu()

    gmm2_x1 = torch.from_numpy(np.random.uniform(-3, 3, size=(128, 256))).to(torch.float32).npu()
    gmm2_x2 = torch.from_numpy(np.random.uniform(-3, 3, size=(256, 256))).to(torch.float32).npu()
    gmm2_x3 = torch.from_numpy(np.random.uniform(-3, 3, size=(1, 256))).to(torch.float32).npu()
    gmm2_x = [gmm2_x1, gmm2_x2, gmm2_x3]

    data3 = torch.from_numpy(np.random.uniform(1, 1, size=(16, 256))).to(torch.int32).npu()
    dsq1_activate_scale = torch.from_numpy(np.random.uniform(1, 1, size=(16, 1))).to(torch.float32).npu()
    dsq1_group_index = None

    gmm3_weight1 = torch.from_numpy(np.random.uniform(-3, 3, size=(32, 8))).to(torch.float32).npu()
    gmm3_weight2 = torch.from_numpy(np.random.uniform(-3, 3, size=(128, 320))).to(torch.float32).npu()
    gmm3_weight3 = torch.from_numpy(np.random.uniform(-3, 3, size=(8, 256))).to(torch.float32).npu()
    gmm3_weight = [gmm3_weight1, gmm3_weight2, gmm3_weight3]

    class Network(nn.Module):
        def __init__(self, enable_superkernel: bool):
            super().__init__()
            self._enable_superkernel = enable_superkernel

        def forward(self, data1, data2, gmm1_x1, gmm1_weight, moe1_bias, arn1_x2, arn1_gamma, gmm2_x, data3,
                    dsq1_activate_scale, dsq1_group_index, gmm3_weight):
            split = torch.split(data1, 8, dim=1)
            matmul_01 = torch.matmul(split[0], data2)
            concat_01 = torch.cat([split[1], split[2]], 0)
            permute_01 = concat_01.permute(1, 0)
            permute_02 = split[3].permute(1, 0)
            permute_03 = split[4].permute(1, 0)
            concat_dsq_01 = torch.cat([permute_02, permute_03], 1)
            mean_01 = torch.mean(concat_dsq_01, 0, False)
            mean_02 = torch.mean(permute_03, 0, True)

            with super_kernel_scope(self._enable_superkernel, "sp1", ""):
                grouped_matmul_01 = torch_npu.npu_grouped_matmul(group_type=-1, x=[matmul_01, gmm1_x1],
                                                                 weight=[gmm1_weight, permute_01])
                moe_gating_top_k_01 = torch_npu.npu_moe_gating_top_k(x=grouped_matmul_01[1], bias=moe1_bias, k=8,
                    k_group=4, group_count=8, group_select_mode=1, norm_type=1)
                grouped_matmul_02 = torch_npu.npu_grouped_matmul(group_type=-1, x=gmm2_x,
                    weight=[moe_gating_top_k_01[0], moe_gating_top_k_01[0], moe_gating_top_k_01[2]])
                dequant_swiglu_quant_01 = torch_npu.npu_dequant_swiglu_quant(x=data3, weight_scale=mean_01,
                    activation_scale=dsq1_activate_scale, bias=None, quant_scale=mean_02, quant_offset=None,
                    group_index=None, activate_left=False, quant_mode=1)
                cast_dsq_1 = dequant_swiglu_quant_01[0].to(torch.float32)
                grouped_matmul_03 = torch_npu.npu_grouped_matmul(group_type=-1, x=[grouped_matmul_01[0],
                    cast_dsq_1, grouped_matmul_02[0]], weight=gmm3_weight)
            cast_01 = grouped_matmul_03[0].to(torch.float32)
            matmul_02 = torch.matmul(cast_01, data2)
            permute_04 = grouped_matmul_03[1].permute(1, 0)
            concat_02 = torch.cat([permute_04, permute_04], 1)
            narrow_01 = grouped_matmul_03[2].narrow(0, 0, 8)
            permute_05 = split[5].permute(1, 0)
            permute_06 = split[6].permute(1, 0)
            concat_dsq_02 = torch.cat([permute_05, permute_06], 1)
            mean_03 = torch.mean(concat_dsq_02, 0, False)
            mean_04 = torch.mean(permute_06, 0, True)

            with super_kernel_scope(self._enable_superkernel, "sp2", ""):
                grouped_matmul_04 = torch_npu.npu_grouped_matmul(group_type=-1, x=[matmul_02, gmm1_x1],
                                                                 weight=[concat_02, narrow_01])
                moe_gating_top_k_02 = torch_npu.npu_moe_gating_top_k(x=grouped_matmul_04[1], bias=moe1_bias,
                    k=8, k_group=4, group_count=8, group_select_mode=1, norm_type=1)
                grouped_matmul_05 = torch_npu.npu_grouped_matmul(group_type=-1, x=gmm2_x,
                    weight=[moe_gating_top_k_02[0], moe_gating_top_k_02[0], moe_gating_top_k_02[2]])
                dequant_swiglu_quant_02 = torch_npu.npu_dequant_swiglu_quant(x=data3, weight_scale=mean_03,
                    activation_scale=dsq1_activate_scale, bias=None, quant_scale=mean_04, quant_offset=None,
                    group_index=None, activate_left=False, quant_mode=1)
                cast_dsq_2 = dequant_swiglu_quant_02[0].to(torch.float32)
                grouped_matmul_06 = torch_npu.npu_grouped_matmul(group_type=-1, x=[grouped_matmul_04[0],
                                                                cast_dsq_2, grouped_matmul_05[0]], weight=gmm3_weight)
            concat_03 = torch.cat([grouped_matmul_03[1], grouped_matmul_06[1]], 0)
            transpose_01 = torch.transpose(concat_03, 1, 0)
            repeat_01 = split[7].repeat([1, 40])
            cast_02 = grouped_matmul_06[0].to(torch.float32)
            repeat_02 = cast_02.repeat([2, 1])
            mean_05 = torch.mean(concat_02, 0, False)

            with super_kernel_scope(self._enable_superkernel, "sp3", ""):
                dequant_swiglu_quant_03 = torch_npu.npu_dequant_swiglu_quant(x=data3, weight_scale=mean_03,
                    activation_scale=dsq1_activate_scale, bias=None, quant_scale=mean_04, quant_offset=None,
                    group_index=None, activate_left=False, quant_mode=1)
                dequant_swiglu_quant_04 = torch_npu.npu_dequant_swiglu_quant(x=data3, weight_scale=mean_01,
                    activation_scale=dsq1_activate_scale, bias=None, quant_scale=mean_04, quant_offset=None,
                    group_index=None, activate_left=False, quant_mode=1)
                dequant_swiglu_quant_05 = torch_npu.npu_dequant_swiglu_quant(x=data3, weight_scale=mean_01,
                    activation_scale=dsq1_activate_scale, bias=None, quant_scale=mean_02, quant_offset=None,
                    group_index=None, activate_left=False, quant_mode=1)
                add_01 = torch.add(dequant_swiglu_quant_03[0], dequant_swiglu_quant_04[0])
                add_02 = torch.add(add_01, dequant_swiglu_quant_05[0])
                cast_dsq_3 = add_02.to(torch.float32)

            with super_kernel_scope(self._enable_superkernel, "sp4", ""):
                grouped_matmul_07 = torch_npu.npu_grouped_matmul(group_type=-1, x=[repeat_01, repeat_02],
                    weight=[transpose_01, narrow_01], bias=[mean_05, moe1_bias])
                moe_gating_top_k_03 = torch_npu.npu_moe_gating_top_k(x=grouped_matmul_07[1], bias=moe1_bias, k=8,
                    k_group=4, group_count=8, group_select_mode=1, norm_type=1)
                grouped_matmul_08 = torch_npu.npu_grouped_matmul(group_type=-1, x=gmm2_x,
                    weight=[moe_gating_top_k_03[0], moe_gating_top_k_03[0], moe_gating_top_k_03[2]])
                grouped_matmul_09 = torch_npu.npu_grouped_matmul(group_type=-1, x=[grouped_matmul_07[0],
                    cast_dsq_3, grouped_matmul_08[0]], weight=gmm3_weight)
            narrow_02 = grouped_matmul_06[2].narrow(0, 0, 8)
            concat_04 = torch.cat([grouped_matmul_09[1], grouped_matmul_09[1]], 0)
            transpose_02 = torch.transpose(concat_04, 1, 0)
            transpose_03 = torch.transpose(narrow_02, 1, 0)
            repeat_03 = concat_04.repeat([4, 1])

            with super_kernel_scope(self._enable_superkernel, "sp5", ""):
                grouped_matmul_10 = torch_npu.npu_grouped_matmul(group_type=-1, x=[repeat_03, transpose_03],
                                                                 weight=[transpose_02, narrow_02])
                moe_gating_top_k_04 = torch_npu.npu_moe_gating_top_k(x=grouped_matmul_10[1], bias=moe1_bias, k=8,
                    k_group=4, group_count=8, group_select_mode=1, norm_type=1)
                grouped_matmul_11 = torch_npu.npu_grouped_matmul(group_type=-1, x=gmm2_x,
                    weight=[moe_gating_top_k_04[0], moe_gating_top_k_04[0], moe_gating_top_k_04[2]])
                grouped_matmul_12 = torch_npu.npu_grouped_matmul(group_type=-1, x=[grouped_matmul_10[0], cast_dsq_3,
                                                                 grouped_matmul_11[0]], weight=gmm3_weight)
            narrow_03 = grouped_matmul_12[2].narrow(0, 10, 8)
            transpose_04 = torch.transpose(grouped_matmul_12[1], 1, 0)
            concat_05 = torch.cat([transpose_04, transpose_04], 1)
            repeat_04 = grouped_matmul_12[0].repeat([2, 1])

            with super_kernel_scope(self._enable_superkernel, "sp6", ""):
                grouped_matmul_13 = torch_npu.npu_grouped_matmul(group_type=-1, x=[matmul_01, repeat_04],
                                                                 weight=[concat_05, narrow_03])
                moe_gating_top_k_05 = torch_npu.npu_moe_gating_top_k(x=grouped_matmul_13[1], bias=moe1_bias,
                    k=8, k_group=4, group_count=8, group_select_mode=1, norm_type=1)
                grouped_matmul_14 = torch_npu.npu_grouped_matmul(group_type=-1, x=gmm2_x,
                    weight=[moe_gating_top_k_05[0], moe_gating_top_k_05[0], moe_gating_top_k_05[2]])
                grouped_matmul_15 = torch_npu.npu_grouped_matmul(group_type=-1, x=[grouped_matmul_13[0], cast_dsq_3,
                                                                 grouped_matmul_14[0]], weight=gmm3_weight)
            return grouped_matmul_15[0], grouped_matmul_15[1], grouped_matmul_15[2]

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    # Run the model configured with SuperKernel on the NPU
    no_sk_model = Network(False).npu()
    sk_model = Network(True).npu()

    # Enable profiling
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[
            torch_npu.profiler.ExportType.Text
        ],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        record_op_args=False,
        data_simplification=False
    )

    # Run the model without SuperKernel and collect the data of the second execution through profiling
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof_result/no_sk_model/"),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config) as prof:
        print("------------------------------------ run no sk --------------------------------")
        for _ in range(4):
            no_sk_model = torch.compile(no_sk_model, fullgraph=True, backend=npu_backend, dynamic=False)
            no_sk_model(data1, data2, gmm1_x1, gmm1_weight, moe1_bias, arn1_x2, arn1_gamma,
                        gmm2_x, data3, dsq1_activate_scale, dsq1_group_index, gmm3_weight)
            prof.step()

    # Run the model with SuperKernel and collect the data of the second execution through profiling
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof_result/sk_model/"),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config) as prof:
        print("------------------------------------ run sk --------------------------------")
        for _ in range(4):
            sk_model = torch.compile(sk_model, fullgraph=True, backend=npu_backend, dynamic=False)
            sk_model(data1, data2, gmm1_x1, gmm1_weight, moe1_bias, arn1_x2, arn1_gamma, gmm2_x, data3,
                     dsq1_activate_scale, dsq1_group_index, gmm3_weight)
            prof.step()

    print("execute sample success")

superkernel_compare()
# fmt: on
