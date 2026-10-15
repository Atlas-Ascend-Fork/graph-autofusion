#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import sys
import traceback

import numpy as np
import torch
import torch_npu  # noqa: F401
from op_extension import register_torch_ops


def log(message):
    print(message, flush=True)


class CustomAddModel(torch.nn.Module):
    def __init__(self, enable_sk_scope=False):
        super().__init__()
        self.enable_sk_scope = enable_sk_scope

    def forward(self, x, y):
        if self.enable_sk_scope:
            torch.npu.super_kernel_scope_begin("custom_add_sk")
        z = torch.ops.ascendc_ops.add_custom(x, y)
        if self.enable_sk_scope:
            torch.npu.super_kernel_scope_end("custom_add_sk")
        return z


def compile_options():
    return {
        "static_kernel_compile": True,
        "super_kernel_optimize": True,
        "super_kernel_debug_options": {
            "debug_per_op_max_core_num": 1,
        },
    }


def create_cpu_inputs(shape, cpu_generator):
    log("create cpu inputs begin")
    x_cpu = torch.rand(
        shape, generator=cpu_generator, device="cpu", dtype=torch.float16
    )
    y_cpu = torch.rand(
        shape, generator=cpu_generator, device="cpu", dtype=torch.float16
    )
    log("create cpu inputs end")
    return x_cpu, y_cpu


def move_inputs_to_npu(tag, x_cpu, y_cpu):
    log(f"{tag}: move inputs to npu begin")
    x_npu = x_cpu.npu()
    y_npu = y_cpu.npu()
    torch.npu.synchronize()
    log(f"{tag}: move inputs to npu end")
    return x_npu, y_npu


def run_sk_custom_add(x_cpu, y_cpu):
    tag = "with sk"
    log("----------------------- run with sk -----------------------------")
    try:
        log(f"{tag}: torch.compile begin")
        model = torch.compile(
            CustomAddModel(enable_sk_scope=True),
            backend="npugraph_ex",
            fullgraph=True,
            options=compile_options(),
            dynamic=False,
        )
        x, y = move_inputs_to_npu(tag, x_cpu, y_cpu)
        log(f"{tag}: compiled call begin")
        output_npu = model(x, y)
        torch.npu.synchronize()
        log(f"{tag}: copy output to cpu begin")
        output = output_npu.cpu()
        log(f"{tag} end")
        return output
    except Exception as err:
        log(f"Failed to run CustomAddModel ({tag}): {err}")
        if os.getenv("ASCENDC_PRINT_PY_STACK", "1") == "1":
            traceback.print_exc()
        sys.exit(2)


def main():
    register_torch_ops()
    seed = 1236
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_npu.npu.set_device(0)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)

    shape = [8, 2048]
    x_cpu, y_cpu = create_cpu_inputs(shape, cpu_generator)
    golden = torch.add(x_cpu, y_cpu)

    output_sk = run_sk_custom_add(x_cpu, y_cpu)
    if not torch.allclose(output_sk, golden, rtol=1e-3, atol=1e-3):
        log("Error: the with sk output differs from the golden result")
        sys.exit(1)

    log("Test passed: the with sk output matches the golden result")
    log("execute sample success")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as err:
        log(f"Failed to run the custom add sk sample: {err}")
        if os.getenv("ASCENDC_PRINT_PY_STACK", "1") == "1":
            traceback.print_exc()
        sys.exit(2)
