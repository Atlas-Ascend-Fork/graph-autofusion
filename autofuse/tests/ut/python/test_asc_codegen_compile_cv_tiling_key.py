# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace

import pytest

from compile_test_utils import load_asc_codegen_compile_module


@pytest.fixture
def compile_module(monkeypatch):
    with load_asc_codegen_compile_module() as module:
        monkeypatch.setattr(
            module,
            "static_shape_cv_common_compile",
            lambda **kwargs: (56, 0, False, False),
        )
        yield module


def decide_template(
    module,
    monkeypatch,
    cube_key,
    vector_key,
    is_batch=False,
    is_conv=False,
    has_relu=False,
    cube_block_dim=28,
):
    monkeypatch.setattr(module, "static_shape_cv_compile", lambda **kwargs: vector_key)
    context = module.StaticCompileContext("kernel", "temp", "graph", 56, False, "", "")
    tiling_info = SimpleNamespace(tiling_key=cube_key, file_content="")
    cube_info = [4, is_batch, cube_block_dim, [False], has_relu]
    header = module.template_decider(context, tiling_info, cube_info, is_conv=is_conv)
    return header, tiling_info.file_content


@pytest.mark.parametrize(
    "cube_key,vector_key,is_batch,has_relu",
    [
        (4161, 1, False, True),
        (65, 0, False, True),
        (2, 0, False, False),
        (66, 1, False, False),
        (4162, 0, False, False),
        (257, 0, True, False),
        (258, 1, True, False),
        (833, -1, True, False),
    ],
)
def test_matmul_host_key_preserves_native_kernel_entry(
    compile_module, monkeypatch, cube_key, vector_key, is_batch, has_relu
):
    header, _ = decide_template(
        compile_module,
        monkeypatch,
        cube_key,
        vector_key,
        is_batch=is_batch,
        has_relu=has_relu,
    )
    assert f"#define CUBE_TILING_KEY {cube_key}\n" in header


@pytest.mark.parametrize("vector_key,mode", [(0, "CV_UB_NO_DB"), (1, "CV_UB_DB")])
def test_conv_host_key_preserves_native_kernel_entry(
    compile_module, monkeypatch, vector_key, mode
):
    header, device_header = decide_template(
        compile_module, monkeypatch, 4097, vector_key, is_conv=True
    )
    assert "#define CUBE_TILING_KEY 4097\n" in header
    assert f"#define {mode} 1\n" in device_header


@pytest.mark.parametrize("vector_key,mode", [(0, "CV_UB_NO_DB"), (1, "CV_UB_DB")])
def test_matmul_vector_key_still_selects_ub_template(
    compile_module, monkeypatch, vector_key, mode
):
    header, device_header = decide_template(compile_module, monkeypatch, 65, vector_key)
    assert "#define CUBE_TILING_KEY 65\n" in header
    assert f"#define {mode} 1\n" in device_header


@pytest.mark.parametrize("is_aiv_only,expected_block_dim", [(True, 56), (False, 28)])
def test_static_common_sync_mode_preserves_vector_block_count(
    compile_module, monkeypatch, is_aiv_only, expected_block_dim
):
    monkeypatch.setattr(
        compile_module,
        "static_shape_cv_common_compile",
        lambda **kwargs: (56, 1024, is_aiv_only, False),
    )
    header, device_header = decide_template(
        compile_module, monkeypatch, 513, -1, is_batch=True
    )
    assert ("#define CV_SAFETY_FUSION_AIV_ONLY 1\n" in device_header) == is_aiv_only
    assert f"#define CUBE_BLOCK_DIM {expected_block_dim}\n" in header
    assert "#define CV_SAFETY_FUSION_MIX_MODE 1\n" in device_header


@pytest.mark.parametrize(
    "cube_key,is_batch,native_blocks,vector_blocks,use_launch_aic_num,expected_launch,expected_aic",
    [
        (257, True, 4, 56, True, 28, 28),
        (769, True, 4, 56, True, 28, 28),
        (2097409, True, 4, 56, True, 28, 28),
        (769, True, 28, 56, True, 28, 28),
        (769, True, 28, 4, True, 28, 28),
        (4097, True, 4, 56, False, 28, 4),
        (4097, False, 4, 56, False, 28, 4),
    ],
)
def test_common_cube_limit_matches_scheduler(
    compile_module,
    monkeypatch,
    cube_key,
    is_batch,
    native_blocks,
    vector_blocks,
    use_launch_aic_num,
    expected_launch,
    expected_aic,
):
    monkeypatch.setattr(
        compile_module,
        "static_shape_cv_common_compile",
        lambda **kwargs: (vector_blocks, 1024, False, use_launch_aic_num),
    )
    header, device_header = decide_template(
        compile_module,
        monkeypatch,
        cube_key,
        -1,
        is_batch=is_batch,
        cube_block_dim=native_blocks,
    )
    assert f"#define CUBE_BLOCK_DIM {expected_launch}\n" in header
    assert f"#define CV_AIC_NUM {expected_aic}\n" in device_header
    assert f"#define CV_AIV_NUM {vector_blocks}\n" in device_header
    assert "#define CV_SAFETY_FUSION_AIV_ONLY 1\n" not in device_header


@pytest.mark.parametrize("workspaces", [[], [7340032], [0x120000000]])
def test_native_matmul_workspace_is_preserved(compile_module, monkeypatch, workspaces):
    monkeypatch.setattr(compile_module, "TilingInfo", SimpleNamespace)
    monkeypatch.setattr(
        compile_module,
        "do_op_tiling",
        lambda *args: {
            "tiling_data": b"",
            "tiling_key": 4161,
            "block_dim": 28,
            "workspaces": workspaces,
        },
    )
    info, block_dim = compile_module._process_tiling_info(False, {}, [], [], [], [0])
    assert info.static_workspace_size == sum(workspaces)
    assert info.tiling_key == 4161
    assert block_dim == 28


@pytest.mark.parametrize("workspace_size", [0, 7340032, 0x120000000])
def test_matmul_host_headers_include_native_workspace(
    compile_module, monkeypatch, tmp_path, workspace_size
):
    monkeypatch.setattr(compile_module, "static_shape_cv_compile", lambda **kwargs: 0)
    monkeypatch.setattr(
        compile_module, "_write_matmul_device_tiling_files", lambda *args: None
    )
    context = compile_module.StaticCompileContext(
        "kernel", str(tmp_path), "graph", 56, False, "", ""
    )
    info = SimpleNamespace(
        tiling_key=65, file_content="", static_workspace_size=workspace_size
    )
    cube_info = [4, False, 28, [False], True, [], []]
    compile_module.create_matmul_tiling_data(context, info, cube_info)
    for directory in [tmp_path / "host", tmp_path / "host/cv_common"]:
        header = (directory / "autofuse_cube_tiling_data.h").read_text()
        assert f"#define CUBE_WORKSPACE_SIZE {workspace_size}ULL\n" in header
        assert "#define CUBE_TILING_KEY 65\n" in header
