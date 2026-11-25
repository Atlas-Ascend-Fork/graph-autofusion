# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Real-device regression. Run in an installed Autofuse/CANN environment with
AUTOFUSE_RUN_DEVICE_E2E=1 python -m pytest <this file> --basetemp=<new artifact directory> -s.
Tests compile unmodified generated Host/Device sources. Launch tests also check device results;
codegen_bin tests check compilation and embedded kernel exports.
"""

import ctypes
import json
import os
import re

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("AUTOFUSE_RUN_DEVICE_E2E") != "1",
    reason="requires an installed Autofuse build, CANN and an NPU",
)


def _make_graph(ascir, dynamic, batch, shape, variant="plain"):
    graph = ascir.HintGraph("fixpip_matmul")
    names = ("b", "m", "k", "n") if batch else ("m", "k", "n")
    sizes = [
        graph.create_size(name) if dynamic else ascir.SizeExpr(value)
        for name, value in zip(names, shape)
    ]
    axes = [graph.create_axis(name, size) for name, size in zip(names, sizes)]
    m, k, n = axes[-3:]
    m_size, k_size, n_size = sizes[-3:]
    one = ascir.SizeExpr(1)
    zero = ascir.SizeExpr(0)

    def layout(op, op_axes, repeats, strides):
        op.attr.sched.axis = op_axes
        op.y.axis, op.y.size, op.y.strides = op_axes, repeats, strides
        op.y.dtype = ascir.dtypes.float16

    layouts = [
        ([m, k], [m_size, k_size], [k_size, one]),
        ([k, n], [k_size, n_size], [n_size, one]),
    ]
    output_layout = ([m, n], [m_size, n_size], [n_size, one])
    if batch:
        b, batch_size = axes[0], sizes[0]
        batch_layouts = []
        for op_axes, repeats, strides in layouts:
            batch_layouts.append(
                (
                    [b] + op_axes,
                    [batch_size] + repeats,
                    [repeats[0] * repeats[1]] + strides,
                )
            )
        layouts = batch_layouts
        output_layout = (
            [b, m, n],
            [batch_size, m_size, n_size],
            [m_size * n_size, n_size, one],
        )
    if variant == "b_view":
        a, r, _ = layouts[1]
        layouts[1] = (a, r, [n_size, batch_size * n_size, one])
    inputs = []
    for index, desc in enumerate(layouts):
        data = ascir.ops.Data(f"data{index}", graph)
        data.attr.ir_attr.index = index
        layout(data, *desc)
        load = ascir.ops.Load(f"load{index}", graph)
        load.x = data.y
        load.attr.ir_attr.offset = zero
        layout(load, *desc)
        inputs.append(load)
    op_type = (
        (ascir.ops.BatchMatMulBias if batch else ascir.ops.MatMulBias)
        if variant == "bias"
        else (ascir.ops.BatchMatMul if batch else ascir.ops.MatMul)
    )
    mm = op_type("matmul", graph)
    mm.x1, mm.x2 = inputs[0].y, inputs[0 if variant == "shared_input" else 1].y
    mm.attr.api.compute_type = "cube"
    for name in ("adj_x1", "adj_x2") if batch else ("transpose_x1", "transpose_x2"):
        setattr(mm.attr.ir_attr, name, 0)
    mm.attr.ir_attr.has_relu = 0
    mm.attr.ir_attr.enable_hf32 = 0
    mm.attr.ir_attr.offset_x = 0
    if variant == "bias":
        bias = ascir.ops.Data("bias", graph)
        bias.attr.ir_attr.index = 2
        layout(bias, [n], [n_size], [one])
        bias_load = ascir.ops.Load("bias_load", graph)
        bias_load.x, bias_load.attr.ir_attr.offset = bias.y, zero
        layout(bias_load, [n], [n_size], [one])
        mm.bias = bias_load.y
    layout(mm, *output_layout)
    mm.attr.sched.axis = axes
    store = ascir.ops.Store("store", graph)
    store.x, store.attr.ir_attr.offset = mm.y, zero
    layout(store, *output_layout)
    output = ascir.ops.Output("output", graph)
    output.attr.ir_attr.index, output.x = 0, store.y
    layout(output, *output_layout)
    if variant == "unused_input":
        unused = ascir.ops.Data("unused", graph)
        unused.attr.ir_attr.index = 2
        layout(unused, *layouts[0])
    if variant in ("multiple_outputs", "shared_store_outputs"):
        store2 = (
            store
            if variant == "shared_store_outputs"
            else ascir.ops.Store("store2", graph)
        )
        store2.x, store2.attr.ir_attr.offset = mm.y, zero
        layout(store2, *output_layout)
        output2 = ascir.ops.Output("output2", graph)
        output2.attr.ir_attr.index, output2.x = 1, store2.y
        layout(output2, *output_layout)
    return graph


def _compile_kernel(
    tmp_path, monkeypatch, dynamic, batch, shape, variant, bin_mode=False
):
    # PGO is deliberately disabled: neither missing headers nor ATT-only calls may be emitted.
    monkeypatch.setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=false")
    monkeypatch.setenv("MAX_JOBS", "8")
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "8")
    for variable, subdir in [
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_NPU_EXT_CACHE_DIR", "npu"),
    ]:
        monkeypatch.setenv(variable, str(tmp_path / subdir))
    import torch
    import torch_npu  # noqa: F401
    from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions
    from autofuse.compile_adapter import jit_compile, jit_compile_bin
    from asc_op_compile_base.common.platform import platform_info

    torch.npu.set_device(0)
    soc = torch.npu.get_device_name(0)
    platform_info.set_current_compile_soc_info(soc)
    ascir.utils.set_platform(
        str(platform_info.get_soc_spec("NpuArch")),
        int(platform_info.get_soc_spec("VECTOR_CORE_CNT")),
        int(platform_info.get_soc_spec("UB_SIZE")),
        soc,
    )
    graph = _make_graph(ascir, dynamic, batch, shape, variant)
    fuser = Autofuser(AutofuserOptions())
    tiling_def, host, device = (
        fuser.codegen_bin(fuser.schedule(graph))
        if bin_mode
        else fuser.autofuse_backend(graph)
    )
    for name, content in [
        ("autofuse_tiling_data.h", tiling_def),
        ("host.cpp", host),
        ("device.cpp", device),
    ]:
        (tmp_path / name).write_text(content)
    assert "struct CVAutofuseTilingData {" in tiling_def
    if variant not in ("multiple_outputs", "shared_store_outputs"):
        assert "autofuse_tiling_func_pgo.h" not in host
        assert "optiling::EmitFinalTilingByCase(" not in host
        assert "optiling::GetTilingCore(" not in host
        assert "mat_mul_v3_fusion" not in device
        assert "class TensorCVBlockMmad" not in device
    library_path = tmp_path / "kernel.so"
    (jit_compile_bin if bin_mode else jit_compile)(
        tiling_def,
        host,
        device,
        [
            "--graph_name=fixpip_matmul",
            f"--output_file={library_path}",
            f"--output_path={tmp_path / 'compile'}",
            f"--soc_version={soc}",
        ],
    )
    library = ctypes.CDLL(str(library_path))
    return torch, library, host, soc


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    "batch,shape,variant",
    [
        (False, (128, 128, 128), "plain"),
        (False, (17, 31, 33), "plain"),
        (True, (4, 16, 64, 16), "plain"),
        (True, (4, 16, 64, 16), "b_view"),
        (True, (32, 1, 64, 1), "b_view"),
        (True, (4, 128, 256, 128), "b_view"),
        (False, (128, 128, 128), "unused_input"),
        (False, (128, 128, 128), "bias"),
        (False, (128, 128, 128), "shared_input"),
        (False, (128, 128, 128), "multiple_outputs"),
        (True, (4, 16, 64, 16), "multiple_outputs"),
        (False, (128, 128, 128), "shared_store_outputs"),
        (True, (4, 16, 64, 16), "shared_store_outputs"),
    ],
    ids=[
        "mm",
        "mm_tail",
        "bmm",
        "bmm_view",
        "bmm_view_unit_n",
        "bmm_view_large",
        "unused_input",
        "bias",
        "shared_input",
        "mm_two_outputs",
        "bmm_two_outputs",
        "mm_shared_store_outputs",
        "bmm_shared_store_outputs",
    ],
)
def test_fixpip_host_device_compile_and_launch(
    tmp_path, monkeypatch, dynamic, batch, shape, variant
):
    torch, library, host, soc = _compile_kernel(
        tmp_path, monkeypatch, dynamic, batch, shape, variant
    )
    library.GetTilingDataSize.restype = ctypes.c_size_t
    tiling = ctypes.create_string_buffer(library.GetTilingDataSize())
    workspace_bytes, blocks = ctypes.c_uint64(), ctypes.c_uint32()
    # Follow the generated ABI order rather than assuming alphabetical shape-variable order.
    signature = re.search(
        r'extern "C" int64_t AutofuseTiling\((.*?)\)', host, re.S
    ).group(1)
    dim_names = re.findall(r"int64_t\s+([a-z][a-z0-9_]*)", signature)
    dimensions = dict(zip(("b", "m", "k", "n") if batch else ("m", "k", "n"), shape))
    library.AutofuseTiling.argtypes = [ctypes.c_int64] * len(dim_names) + [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    library.AutofuseTiling.restype = ctypes.c_int64
    assert (
        library.AutofuseTiling(
            *[dimensions[name] for name in dim_names],
            tiling,
            ctypes.byref(workspace_bytes),
            ctypes.byref(blocks),
            None,
        )
        == 0
    )
    if variant == "b_view":
        # Re-run into a poisoned buffer: cached tiling must restore the whole input plan.
        ctypes.memset(tiling, 0xA5, len(tiling))
        assert (
            library.AutofuseTiling(
                *[dimensions[name] for name in dim_names],
                tiling,
                ctypes.byref(workspace_bytes),
                ctypes.byref(blocks),
                None,
            )
            == 0
        )
    assert blocks.value > 0
    generator = torch.Generator().manual_seed(2026)
    m_size, k_size, n_size = shape[-3:]
    prefix = shape[:-3]
    lhs = (torch.randn(*prefix, m_size, k_size, generator=generator) * 0.25).half()
    rhs = (torch.randn(*prefix, k_size, n_size, generator=generator) * 0.25).half()
    golden = torch.matmul(
        lhs.float(), (lhs if variant == "shared_input" else rhs).float()
    )
    bias = None
    if variant == "bias":
        bias = (torch.randn(n_size, generator=generator) * 0.25).half()
        golden += bias.float()
    lhs_npu, rhs_npu = lhs.npu(), rhs.npu()
    if variant == "b_view":
        rhs_npu = rhs.permute(1, 0, 2).contiguous().npu().permute(1, 0, 2)
        assert rhs_npu.stride() == (n_size, prefix[0] * n_size, 1)
    result = torch.full(
        (*prefix, m_size, n_size), float("nan"), dtype=torch.float16, device="npu"
    )
    workspace = torch.empty(
        max(1, workspace_bytes.value), dtype=torch.uint8, device="npu"
    )
    inputs = [lhs_npu, rhs_npu]
    if bias is not None:
        inputs.append(bias.npu())
    if variant == "unused_input":
        inputs.append(torch.full_like(lhs_npu, float("nan")))
    outputs = [result]
    if variant in ("multiple_outputs", "shared_store_outputs"):
        outputs.append(torch.full_like(result, float("nan")))
    library.AutofuseLaunch.argtypes = [ctypes.c_uint32] + [ctypes.c_void_p] * (
        len(inputs) + len(outputs) + 3
    )
    library.AutofuseLaunch.restype = ctypes.c_int64
    assert (
        library.AutofuseLaunch(
            blocks,
            torch.npu.current_stream().npu_stream,
            *[tensor.data_ptr() for tensor in inputs + outputs],
            workspace.data_ptr(),
            tiling,
        )
        == 0
    )
    torch.npu.synchronize()
    # FP16 mixed tolerance; every element of every output must pass.
    tolerance = 2**-9
    max_abs_diff = 0.0
    for output in outputs:
        actual = output.cpu().float()
        torch.testing.assert_close(actual, golden, atol=tolerance, rtol=tolerance)
        difference = (actual - golden).abs().max().item()
        assert difference <= 0.1
        max_abs_diff = max(max_abs_diff, difference)
    summary = {
        "dynamic": dynamic,
        "batch": batch,
        "shape": shape,
        "soc": soc,
        "max_abs_diff": max_abs_diff,
        "variant": variant,
        "output_count": len(outputs),
        "block_dim": blocks.value,
        "workspace_size": workspace_bytes.value,
        "ok": True,
    }
    (tmp_path / "result.json").write_text(json.dumps(summary, indent=2))
    print("CASE_RESULT " + json.dumps(summary), flush=True)


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    "variant", ["plain", "unused_input", "multiple_outputs", "shared_store_outputs"]
)
def test_codegen_bin_matmul_operand_boundary_compiles(
    tmp_path, monkeypatch, dynamic, variant
):
    _, library, _, _ = _compile_kernel(
        tmp_path, monkeypatch, dynamic, False, (128, 128, 128), variant, bin_mode=True
    )
    assert hasattr(library, "GetKernelBinAic")
    assert not hasattr(library, "AutofuseLaunch")


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_codegen_bin_bmm_input_view_compiles(tmp_path, monkeypatch, dynamic):
    _, library, _, _ = _compile_kernel(
        tmp_path, monkeypatch, dynamic, True, (4, 16, 64, 16), "b_view", bin_mode=True
    )
    assert hasattr(library, "GetKernelBinAic")
    assert not hasattr(library, "AutofuseLaunch")
