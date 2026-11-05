# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from types import SimpleNamespace

import pytest

from compile_test_utils import load_asc_codegen_compile_module


@pytest.fixture
def compiler():
    with load_asc_codegen_compile_module() as module:
        module.AscendCLogLevel.LOG_INFO = "info"
        yield module


@pytest.mark.parametrize(
    "a,b,specialized,transpose_a,transpose_b,bias,expected",
    [
        ((12, -1, 64), (12, 64, 100), (12, 50, 64), False, False, False, (12, 50, 100)),
        ((-1, 64), (100, 64), (50, 64), False, True, False, (50, 100)),
        ((64, -1), (64, 100), (64, 50), True, False, False, (50, 100)),
        ((-1, 64), (64, 100), (1, 50, 64), False, False, False, (50, 100)),
        ((12, -1, 64), (12, 64, 100), (12, 50, 64), False, False, True, (12, 50, 100)),
        ((50, 64), (64, 100), None, False, False, False, (50, 100)),
    ],
)
def test_static_native_tiling_uses_specialized_inputs(
    compiler, monkeypatch, a, b, specialized, transpose_a, transpose_b, bias, expected
):
    inputs = [dict(shape=a, dtype="float16"), dict(shape=b, dtype="float16")]
    if bias:
        inputs.append(dict(shape=(100,), dtype="float16"))
    shapes = [specialized, b, (1, 1, 100)] if specialized else None
    if not bias and shapes:
        shapes = shapes[:2]
    captured = {}

    def native_tiling(is_batch, compile_info, actual_inputs, outputs, attrs, keys):
        captured.update(inputs=actual_inputs, output=outputs[0]["shape"])
        return SimpleNamespace(tiling_key=2, tiling_data=[], file_content=""), 1

    monkeypatch.setattr(compiler, "_process_tiling_info", native_tiling)
    monkeypatch.setattr(
        compiler, "create_matmul_tiling_data", lambda *args, **kwargs: None
    )
    compiler.ascbc_matmul_kernel_tiling_pro(
        (*inputs, dict(shape=expected, dtype="float32"), "kernel"),
        temp_dir="unused",
        graph_name="graph",
        kernel_name="kernel",
        input_num=len(inputs),
        output_num=1,
        use_list_tensor_desc=False,
        cube_attrs={
            "cube_attributes": dict(
                input_num=len(inputs),
                is_batch=len(a) == 3,
                transpose_x1=transpose_a,
                transpose_x2=transpose_b,
                input_shapes=shapes,
            )
        },
        tiling_key_list=[-1],
    )
    assert captured["output"] == expected
    assert all(dim >= 0 for tensor in captured["inputs"] for dim in tensor["shape"])
    assert captured["inputs"][0]["shape"] != a or -1 not in a
    assert inputs[0]["shape"] == a
    if bias:
        assert captured["inputs"][2]["shape"] == (100,)


@pytest.mark.parametrize(
    "shapes", [None, [], [[12, 50, 65]], [[12, -1, 64]], [[2, 12, 50, 64]]]
)
def test_static_shape_mismatch_is_rejected_before_native_tiling(compiler, shapes):
    with pytest.raises(ValueError, match="[Ss]tatic|[Ss]pecialized"):
        compiler._resolve_static_mm_args([dict(shape=(12, -1, 64))], 1, shapes)


def test_dynamic_native_shapes_remain_runtime_values(compiler, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        compiler,
        "create_matmul_dynamic_tiling_data",
        lambda directory, cube_info: captured.update(inputs=cube_info[4]),
    )
    compiler.ascbc_matmul_kernel_dynamic_tiling_pro(
        (
            dict(shape=(12, -1, 64), dtype="float16"),
            dict(shape=(12, 64, 100), dtype="float16"),
            dict(shape=(12, -1, 100), dtype="float16"),
            "kernel",
        ),
        temp_dir="unused",
        graph_name="graph",
        kernel_name="kernel",
        input_num=2,
        output_num=1,
        use_list_tensor_desc=False,
        cube_attrs={
            "cube_attributes": dict(
                input_num=2, is_batch=True, input_shapes=[[12, 50, 64], [12, 64, 100]]
            )
        },
    )
    assert captured["inputs"][0]["shape"] == (12, -1, 64)


@pytest.mark.parametrize(
    "native_key,vector_key,is_batch,is_conv,expected",
    [
        (2, 0, False, False, 2),
        (18, 1, False, False, 18),
        (66, 0, True, False, 66),
        (131074, 1, False, False, 131074),
        (258, -1, True, False, 258),
        (1, 1, False, False, 1),
        (66, 0, False, True, 66),
        (66, 1, False, True, 66),
    ],
)
def test_static_runtime_key_matches_device_entry(
    compiler, monkeypatch, native_key, vector_key, is_batch, is_conv, expected
):
    monkeypatch.setattr(
        compiler, "static_shape_cv_compile", lambda **kwargs: vector_key
    )
    monkeypatch.setattr(
        compiler,
        "static_shape_cv_common_compile",
        lambda **kwargs: (4, 0, False, False),
    )
    info = SimpleNamespace(tiling_key=native_key, file_content="")
    context = compiler.StaticCompileContext(
        "kernel", "unused", "graph", 4, False, "", ""
    )
    cube_info = [4, is_batch, 4, [False], False]
    content = compiler.template_decider(context, info, cube_info, is_conv=is_conv)
    assert f"#define CUBE_TILING_KEY {expected}\n" in content
    if is_conv or native_key in (2, 18, 66, 131074, 1):
        assert ("#define CV_UB_NO_DB 1" in info.file_content) == (vector_key == 0)
        assert ("#define CV_UB_DB 1" in info.file_content) == (vector_key != 0)
    else:
        assert "#define CV_SAFETY_FUSION 1" in info.file_content
