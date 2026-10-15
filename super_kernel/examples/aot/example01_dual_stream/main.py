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

"""
Dual-stream + SuperKernel + NPU event manual control-edge sample

Compares output consistency with super_kernel_optimize enabled (SK) and disabled (Non-SK),
by comparing the dq_res_2 results.

Features:
1. Stream 1 records an NPU event after its intermediate operators finish
2. Stream 2 waits for the NPU event (a control edge instead of a data dependency)
3. Pure aclnn operator calls
Usage: run it through build.sh in the repository root
"""

import torch
import torch.nn as nn
import torch_npu
import numpy as np
import sys


# ============================================================================
class DualStreamModel(nn.Module):
    """
    Dual-stream model: manual control edge with NPU events

    Stream 1 (primary):
        quant_matmul → grouped_matmul → [record event]
        → swiglu_quant → dynamic_quant

    Stream 2 (secondary):
        [wait event] → quant_matmul → add_rms_norm
        → dynamic_quant → [record event]
    """

    def forward(
        self,
        x1_1,
        x2_1,
        scale_1,
        offset_1,
        bias_1,
        pertoken_scale_1,
        smooth_scales_1,
        x1_2,
        x2_2,
        scale_2,
        offset_2,
        bias_2,
        pertoken_scale_2,
        arn2_x2,
        arn2_gamma,
        smooth_scales_2,
        swiglu_quant_scale_1,
        gmm1_weight_1,
        gmm1_bias_1,
        stream1,
        stream2,
        event1,
        event2,
    ):
        # ========== Stream 1 (primary) ==========
        event1.record()
        with torch.npu.stream(stream1):
            # torch.npu.super_kernel_scope_begin("sk0")
            event1.wait(stream1)

            qm_out_1 = torch_npu.npu_quant_matmul(
                x1_1,
                x2_1,
                scale_1,
                offset=offset_1,
                bias=bias_1,
                pertoken_scale=pertoken_scale_1,
                output_dtype=torch.bfloat16,
            )
            gmm_out_1 = torch_npu.npu_grouped_matmul(
                x=[qm_out_1],
                weight=gmm1_weight_1,
                bias=gmm1_bias_1,
                split_item=0,
                group_list=None,
                group_type=-1,
                output_dtype=torch.bfloat16,
            )
            # Notify stream2: the intermediate operators of stream1 have finished
            event1.record()

            swiglu_res_1 = torch_npu.npu_dequant_swiglu_quant(
                gmm_out_1[0], quant_scale=swiglu_quant_scale_1, quant_mode=1
            )
            event2.record()

            dq_res_1 = torch_npu.npu_dynamic_quant(
                gmm_out_1[0], smooth_scales=smooth_scales_1
            )
            # torch.npu.super_kernel_scope_end("sk0")

        # ========== Stream 2 (secondary) - waits for the stream1 event ==========
        with torch.npu.stream(stream2):
            # torch.npu.super_kernel_scope_begin("sk1")
            event1.wait(stream2)

            qm_out_2 = torch_npu.npu_quant_matmul(
                x1_2,
                x2_2,
                scale_2,
                offset=offset_2,
                bias=bias_2,
                pertoken_scale=pertoken_scale_2,
                output_dtype=torch.bfloat16,
            )
            arn2_out, _, _ = torch_npu.npu_add_rms_norm(
                qm_out_2, arn2_x2, arn2_gamma, epsilon=1e-5
            )
            event2.wait(stream2)

            dq_res_2 = torch_npu.npu_dynamic_quant(
                arn2_out, smooth_scales=smooth_scales_2
            )
            # torch.npu.super_kernel_scope_end("sk1")

        return dq_res_1, dq_res_2, swiglu_res_1


# ============================================================================
# Data preparation
# ============================================================================


def prepare_data(seed=1236):
    """Prepare all input data required by the dual-stream model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Stream 1 parameters
    m1, k1, n1 = 864, 7168, 4096
    x1_1 = torch.randint(-10, 10, (m1, k1), dtype=torch.int8)
    x2_1 = (
        torch.randint(-10, 10, (n1, k1), dtype=torch.int8)
        .npu()
        .transpose(1, 0)
        .contiguous()
    )
    scale_1 = torch.randn((n1,), dtype=torch.float32)
    pertoken_scale_1 = torch.randn((m1,), dtype=torch.float32)
    smooth_scales_1 = torch.randn((200,), dtype=torch.bfloat16)
    swiglu_quant_scale_1 = torch.randn((1, 100), dtype=torch.float32)
    gmm1_weight_1 = [torch.rand(n1, 200, dtype=torch.bfloat16).npu()]
    gmm1_bias_1 = [torch.rand(200, dtype=torch.float32).npu()]

    # Stream 2 parameters
    m2, k2, n2 = 864, 7168, 4096
    x1_2 = torch.randint(-10, 10, (m2, k2), dtype=torch.int8)
    x2_2 = (
        torch.randint(-10, 10, (n2, k2), dtype=torch.int8)
        .npu()
        .transpose(1, 0)
        .contiguous()
    )
    scale_2 = torch.randn((n2,), dtype=torch.float32)
    pertoken_scale_2 = torch.randn((m2,), dtype=torch.float32)
    arn2_x2 = torch.rand(m2, n2, dtype=torch.bfloat16).npu()
    arn2_gamma = torch.rand(n2, dtype=torch.bfloat16).npu()
    smooth_scales_2 = torch.randn((n2,), dtype=torch.bfloat16).npu()

    # NPU streams/events
    stream1 = torch.npu.Stream()
    stream2 = torch.npu.Stream()
    event1 = torch.npu.Event()
    event2 = torch.npu.Event()

    return dict(
        x1_1=x1_1.npu(),
        x2_1=x2_1,
        scale_1=scale_1.npu(),
        offset_1=None,
        bias_1=None,
        pertoken_scale_1=pertoken_scale_1.npu(),
        smooth_scales_1=smooth_scales_1.npu(),
        swiglu_quant_scale_1=swiglu_quant_scale_1.npu(),
        x1_2=x1_2.npu(),
        x2_2=x2_2,
        scale_2=scale_2.npu(),
        offset_2=None,
        bias_2=None,
        pertoken_scale_2=pertoken_scale_2.npu(),
        arn2_x2=arn2_x2,
        arn2_gamma=arn2_gamma,
        smooth_scales_2=smooth_scales_2,
        gmm1_weight_1=gmm1_weight_1,
        gmm1_bias_1=gmm1_bias_1,
        stream1=stream1,
        stream2=stream2,
        event1=event1,
        event2=event2,
    )


# ============================================================================
# Compile and run
# ============================================================================


def build_compile_options(enable_sk=True):
    """Build npugraph_ex compile options; enable_sk toggles SuperKernel."""
    options = {}
    if enable_sk:
        options.update(
            {
                "static_kernel_compile": True,
                "super_kernel_optimize": True,
                "super_kernel_optimize_options": {"auto_op_parallel": 1},
            }
        )
    return options


def run_model(enable_sk, data):
    """Run the model and return dq_res_2; print the error and exit on exception."""
    tag = "SK" if enable_sk else "Non-SK"
    print(f"\n{'=' * 60}")
    print(f"  Running the {tag} version (super_kernel_optimize={enable_sk})")
    print(f"{'=' * 60}")

    try:
        model = torch.compile(
            DualStreamModel(),
            backend="npugraph_ex",
            options=build_compile_options(enable_sk),
            dynamic=False,
        )
        _, dq_res_2, _ = model(**data)
        torch_npu.npu.synchronize()
    except Exception as e:
        print(f"[ERROR] Failed to run the {tag} version: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(2)
    print(f"{tag} version finished")
    return dq_res_2


# ============================================================================
# Accuracy comparison
# ============================================================================


def compare_results(dq_res_2_non_sk, dq_res_2_sk, atol=1e-3, rtol=1e-3):
    """Compare the dq_res_2 outputs of the Non-SK and SK versions."""
    print(f"\n{'=' * 60}")
    print("  Accuracy comparison (dq_res_2)")
    print(f"{'=' * 60}")

    if len(dq_res_2_non_sk) != len(dq_res_2_sk):
        print(
            f"  Output count mismatch: Non-SK={len(dq_res_2_non_sk)}, SK={len(dq_res_2_sk)}"
        )
        return False

    all_outputs_close = True
    for output_idx, (non_sk, sk) in enumerate(zip(dq_res_2_non_sk, dq_res_2_sk)):
        non_sk_np = non_sk.cpu().float().numpy()
        sk_np = sk.cpu().float().numpy()
        abs_diff = np.abs(non_sk_np - sk_np)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        output_close = np.allclose(non_sk_np, sk_np, atol=atol, rtol=rtol)
        all_outputs_close = all_outputs_close and output_close

        print(f"  output[{output_idx}] max absolute difference: {max_diff:.6e}")
        print(f"  output[{output_idx}] mean absolute difference: {mean_diff:.6e}")
        print(
            f"  output[{output_idx}] allclose(atol={atol}, rtol={rtol}): "
            f"{'PASS' if output_close else 'FAIL'}"
        )

    return all_outputs_close


# ============================================================================
# Main entry
# ============================================================================

if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    torch_npu.npu.set_op_timeout_ms(10000)

    print("=" * 60)
    print("  Dual-stream + SuperKernel + NPU event comparison sample")
    print("=" * 60)
    print("Stream 1: quant_matmul → grouped_matmul → [record]")
    print("          → swiglu_quant → dynamic_quant")
    print("Stream 2: [wait] → quant_matmul → add_rms_norm")
    print("          → dynamic_quant → [record]")
    print("Compared item: dq_res_2 (SK vs Non-SK)")
    print("=" * 60)

    try:
        # Prepare data
        data = prepare_data()

        # Run the SK version
        dq_res_2_sk = run_model(enable_sk=True, data=data)

        # Prepare data
        data = prepare_data()

        # Run the Non-SK baseline version
        dq_res_2_non_sk = run_model(enable_sk=False, data=data)

        # Accuracy comparison
        passed = compare_results(dq_res_2_non_sk, dq_res_2_sk)
    except Exception as e:
        print(f"\n[ERROR] Sample exited with an exception: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(2)

    if passed:
        print("\nTest passed: SK and Non-SK outputs match!")
        print("execute sample success")
        sys.exit(0)

    print("\nTest failed: SK and Non-SK outputs differ!")
    sys.exit(1)
