#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import sys

import numpy as np
import torch
import torch_npu


class Network(torch.nn.Module):
    """
    Network topology:

        query, key, value, length
            |
            v
        [fia_01] npu_fused_infer_attention_score
            |
            v
        [res] npu_moe_gating_top_k_softmax_v2
            |
            v
        [quant_01] npu_dynamic_quant -> to(float16)
            |
            v
        [ifa_01] npu_incre_flash_attention  <--- key, value, length
            |
            v
        [fia_02] npu_fused_infer_attention_score <--- key, value, length
            |                         |
            |                         v
            |                 npu_grouped_matmul <--- weight
            |                         |
            +-----------> torch.add <-+
                              |
                              v
                         add_01 output

        outputs: (add_01, ifa_01)
    """

    def __init__(self):
        super(Network, self).__init__()

    def forward(self, query, key, value, length, smooth_scales, weight):
        fia_01, _ = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=16, actual_seq_lengths=length
        )
        res = torch_npu.npu_moe_gating_top_k_softmax_v2(fia_01, finished=None, k=1024)
        quant_01, _ = torch_npu.npu_dynamic_quant(res[0], smooth_scales=smooth_scales)
        quant_01 = quant_01.to(torch.float16)
        ifa_01 = torch_npu.npu_incre_flash_attention(
            quant_01, key, value, num_heads=16, actual_seq_lengths=length
        )
        fia_02, _ = torch_npu.npu_fused_infer_attention_score(
            ifa_01, key, value, num_heads=16, actual_seq_lengths=length
        )
        grouped_matmul_01 = torch_npu.npu_grouped_matmul(
            [fia_02], [weight], group_type=-1
        )
        add_01 = torch.add(grouped_matmul_01[0], fia_02)
        return add_01, ifa_01


def gen_data():
    batch = 3
    seq_len = 256
    hidden = 1024
    query = (
        torch.tensor(np.random.uniform(1, 2, (batch, 1, hidden)))
        .to(torch.float16)
        .npu()
    )
    key = (
        torch.tensor(np.random.uniform(1, 5, (batch, seq_len, hidden)))
        .to(torch.float16)
        .npu()
    )
    value = (
        torch.tensor(np.random.uniform(1, 5, (batch, seq_len, hidden)))
        .to(torch.float16)
        .npu()
    )
    length = [99, 199, 180]
    smooth_scales = torch.randn(1024).to(torch.float16).npu()
    weight = torch.randn(1024, 1024).to(torch.float16).npu()
    return query, key, value, length, smooth_scales, weight


def clone_inputs(data):
    return tuple(
        item.clone() if isinstance(item, torch.Tensor) else list(item) for item in data
    )


def run_model(model, data):
    with torch.no_grad():
        output = model(*data)
        torch.npu.synchronize()
        return tuple(item.detach().clone() for item in output)


def validate_outputs(actual, expected):
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-3, atol=1e-2)


def print_output_summary(name, outputs):
    add_out, ifa_out = outputs
    print(
        f"{name} add_out: shape={tuple(add_out.shape)}, dtype={add_out.dtype}, "
        f"mean={add_out.float().mean().item():.6f}"
    )
    print(
        f"{name} ifa_out: shape={tuple(ifa_out.shape)}, dtype={ifa_out.dtype}, "
        f"mean={ifa_out.float().mean().item():.6f}"
    )


def build_model():
    return torch.compile(
        Network().npu().eval(),
        backend="npugraph_ex",
        fullgraph=True,
        dynamic=True,
        options={
            "static_kernel_compile": True,
            "super_kernel_optimize": True,
            "super_kernel_optimize_options": {
                "auto_op_parallel": 0,
                "dcci_before_kernel_start": [".*"],
                "dcci_after_kernel_end": [".*"],
                "dcci_disable_on_kernel": [".*"],
                "early_start": 1,
                "aggressive_opt_strategies": {
                    "value_breaker_bypass": 0b10,
                    "task_breaker_bypass": 0b00,
                },
            },
            "super_kernel_debug_options": {
                "debug_sync_all": 0,
                "debug_op_exec_trace": 0,
                "debug_cross_core_sync_check": 0,
                "debug_per_op_max_core_num": 0,
            },
        },
    )


def main():
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = gen_data()

    try:
        eager_model = Network().npu().eval()
        expected = run_model(eager_model, clone_inputs(data))

        model = build_model()
        actual = run_model(model, clone_inputs(data))
        validate_outputs(actual, expected)
    except AssertionError as err:
        print("Golden check failed", file=sys.stderr)
        print(err, file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Test run failed: {err}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2

    print_output_summary("eager", expected)
    print_output_summary("compiled", actual)
    print("Golden check passed")
    print("Test completed!")
    print("execute sample success")
    return 0


if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    raise SystemExit(main())
