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
def module(monkeypatch):
    with load_asc_codegen_compile_module() as module:
        monkeypatch.setattr(
            module,
            "get_soc_spec",
            lambda key: {"cube_core_cnt": 28, "vector_core_cnt": 56}[key],
        )
        yield module


@pytest.mark.parametrize(
    "aic,aiv,ratio,expected",
    [
        (5, 7, 2, (3, 6)),
        (5, 7, 1, (5, 5)),
        (5, 6, 2, (3, 6)),
        (4, 7, 2, (3, 6)),
        (None, 7, 2, (3, 6)),
        (5, None, 2, (5, 10)),
        (0, 0, 2, (28, 56)),
        (100, 200, 2, (28, 56)),
        (1, 2, 2, (1, 2)),
    ],
)
def test_limits_choose_largest_complete_task_group(module, aic, aiv, ratio, expected):
    assert (
        module.normalize_cv_core_limits(aic, aiv, f"KERNEL_TYPE_MIX_AIC_1_{ratio}")
        == expected
    )


@pytest.mark.parametrize("aic,aiv", [(1, 1), (2, 1), (-1, 7), (5, -1)])
def test_incomplete_or_negative_limits_are_not_treated_as_unlimited(module, aic, aiv):
    with pytest.raises(ValueError):
        module.normalize_cv_core_limits(aic, aiv, "KERNEL_TYPE_MIX_AIC_1_2")


class OpContext:
    def __init__(self, aic, aiv):
        self.values = {"_op_aicore_num": aic, "_op_vectorcore_num": aiv}

    def get_addition(self, key):
        return self.values.get(key)

    def add_addition(self, key, value):
        self.values[key] = value


@pytest.mark.parametrize("fail", [False, True])
def test_native_context_restores_limits_on_success_and_failure(module, fail):
    context = OpContext(5, 7)
    try:
        with module.cv_core_limit_context(context, 3, 6):
            assert context.values == {"_op_aicore_num": "3", "_op_vectorcore_num": "6"}
            if fail:
                raise RuntimeError("native tiler failed")
    except RuntimeError:
        assert fail
    assert context.values == {"_op_aicore_num": 5, "_op_vectorcore_num": 7}


@pytest.mark.parametrize("ratio,expected", [(1, (5, 5)), (2, (3, 6))])
def test_tf_compile_native_and_vector_receive_same_limits(
    module, monkeypatch, ratio, expected
):
    context = OpContext(5, 7)
    schedule = SimpleNamespace(is_cube_type=lambda: True)
    codegen = SimpleNamespace(
        get_kernel_task_type=lambda _: f"KERNEL_TYPE_MIX_AIC_1_{ratio}"
    )
    monkeypatch.setattr(module, "CodeGen", lambda: codegen)
    monkeypatch.setattr(module, "get_context", lambda: context)
    params = {"schedule_results": schedule, "vector_core_num": 7}
    received = []

    def compile_impl(*args, temp_dir, params):
        received.append((params["ai_core_num"], params["vector_core_num"]))
        assert received[-1] == expected
        assert context.get_addition("_op_aicore_num") == str(expected[0])
        assert context.get_addition("_op_vectorcore_num") == str(expected[1])
        assert params["cv_task_ratio"] == ratio

    monkeypatch.setattr(module, "_asc_graph_compile_impl", compile_impl)
    module.asc_graph_compile(temp_dir="unused", params=params)
    assert received == [expected]
    assert "ai_core_num" not in params
    assert params["vector_core_num"] == 7
    assert context.get_addition("_op_aicore_num") == 5
    assert context.get_addition("_op_vectorcore_num") == 7


@pytest.mark.parametrize("is_cube", [False, True])
def test_non_mixed_compile_keeps_original_limits(module, monkeypatch, is_cube):
    params = {
        "schedule_results": SimpleNamespace(is_cube_type=lambda: is_cube),
        "vector_core_num": 7,
    }
    monkeypatch.setattr(
        module,
        "CodeGen",
        lambda: SimpleNamespace(get_kernel_task_type=lambda _: "KERNEL_TYPE_AIC_ONLY"),
    )
    received = []
    monkeypatch.setattr(
        module,
        "_asc_graph_compile_impl",
        lambda *args, **kwargs: received.append(kwargs["params"]),
    )
    module.asc_graph_compile(temp_dir="unused", params=params)
    assert received == [params]
    assert received[0]["vector_core_num"] == 7


def test_common_launch_uses_kernel_ratio_instead_of_fixed_two(module, monkeypatch):
    monkeypatch.setattr(
        module, "static_shape_cv_common_compile", lambda **kwargs: (5, 0, False, False)
    )
    context = module.SafetyFusionCtx(
        {},
        None,
        SimpleNamespace(tiling_key=2, file_content=""),
        False,
        3,
        -1,
        [False],
        1,
    )
    assert module._apply_safety_fusion(context) == "#define CUBE_BLOCK_DIM 5\n"


def test_session_platform_limits_are_normalized_without_operator_overrides(
    module, monkeypatch
):
    context = OpContext(None, None)
    schedule = SimpleNamespace(is_cube_type=lambda: True)
    monkeypatch.setattr(
        module,
        "get_soc_spec",
        lambda key: {"cube_core_cnt": 5, "vector_core_cnt": 7}[key],
    )
    monkeypatch.setattr(module, "get_context", lambda: context)
    monkeypatch.setattr(
        module,
        "CodeGen",
        lambda: SimpleNamespace(
            get_kernel_task_type=lambda _: "KERNEL_TYPE_MIX_AIC_1_2"
        ),
    )
    received = []
    monkeypatch.setattr(
        module,
        "_asc_graph_compile_impl",
        lambda *args, **kwargs: received.append(
            (
                kwargs["params"]["ai_core_num"],
                kwargs["params"]["vector_core_num"],
                context.get_addition("_op_aicore_num"),
                context.get_addition("_op_vectorcore_num"),
            )
        ),
    )
    module.asc_graph_compile(
        temp_dir="unused",
        params={"schedule_results": schedule, "vector_core_num": None},
    )
    assert received == [(3, 6, "3", "6")]
    assert context.get_addition("_op_aicore_num") is None
    assert context.get_addition("_op_vectorcore_num") is None
