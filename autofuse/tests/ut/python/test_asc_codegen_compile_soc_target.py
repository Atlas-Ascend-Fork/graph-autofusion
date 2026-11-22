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

import sys
import types

import pytest

from autofuse.pyautofuse import ascir
from compile_test_utils import load_asc_codegen_compile_module


def test_requires_full_soc_version_uses_registered_architecture():
    assert ascir.utils.requires_full_soc_version("2201") is False
    assert ascir.utils.requires_full_soc_version("5102") is True
    with pytest.raises(RuntimeError, match="invalid npu arch"):
        ascir.utils.requires_full_soc_version("3102")


class FakeSocUtils:
    def __init__(self, required):
        self.required = required
        self.set_platform_calls = []

    @staticmethod
    def requires_full_soc_version(npu_arch):
        return npu_arch in {"3510", "5102", "9201", "9202"}

    def set_platform(self, npu_arch, vector_core_num, ub_size, soc_version=""):
        self.set_platform_calls.append(
            (npu_arch, vector_core_num, ub_size, soc_version)
        )


def _make_context(module):
    extra_params = {"compute_graph": "", "symbol_source_info": ""}
    return module.CodegenCompileContext(extra_params, 1, 1, 0, 8)


def test_compile_target_passes_full_soc_version_for_v35(monkeypatch):
    with load_asc_codegen_compile_module() as loaded:
        module = loaded
        queried = []
        values = {"NpuArch": "5102", "SOC_VERSION": "MC62CM12AA", "ub_size": "245760"}

        def fake_get_soc_spec(key):
            queried.append(key)
            return values[key]

        fake_utils = FakeSocUtils(True)
        monkeypatch.setattr(module, "get_soc_spec", fake_get_soc_spec)
        monkeypatch.setattr(module, "ascir", types.SimpleNamespace(utils=fake_utils))

        module._set_platform_from_compile_target(_make_context(module))

        assert queried == ["NpuArch", "SOC_VERSION", "ub_size"]
        assert fake_utils.set_platform_calls == [("5102", 8, 245760, "MC62CM12AA")]


def test_compile_target_does_not_query_full_soc_version_for_v2(monkeypatch):
    with load_asc_codegen_compile_module() as loaded:
        module = loaded
        queried = []
        values = {"NpuArch": "2201", "ub_size": "245760"}

        def fake_get_soc_spec(key):
            queried.append(key)
            return values[key]

        fake_utils = FakeSocUtils(False)
        monkeypatch.setattr(module, "get_soc_spec", fake_get_soc_spec)
        monkeypatch.setattr(module, "ascir", types.SimpleNamespace(utils=fake_utils))

        module._set_platform_from_compile_target(_make_context(module))

        assert "SOC_VERSION" not in queried
        assert fake_utils.set_platform_calls == [("2201", 8, 245760, "")]


@pytest.mark.parametrize("npu_arch", ["3510", "5102", "9201", "9202"])
def test_compile_target_keeps_v2_baseline_without_full_soc_version(
    monkeypatch, npu_arch
):
    with load_asc_codegen_compile_module() as loaded:
        module = loaded
        values = {"NpuArch": npu_arch, "ub_size": "245760"}
        monkeypatch.setattr(module, "get_soc_spec", lambda key: values.get(key))
        fake_utils = FakeSocUtils(True)
        monkeypatch.setattr(module, "ascir", types.SimpleNamespace(utils=fake_utils))

        assert (
            module._set_platform_from_compile_target(_make_context(module)) == npu_arch
        )
        assert fake_utils.set_platform_calls == [(npu_arch, 8, 245760, "")]


def test_compile_target_is_set_before_graph_compile(monkeypatch, tmpdir):
    with load_asc_codegen_compile_module() as loaded:
        module = loaded
        events = []
        values = {"NpuArch": "5102", "SOC_VERSION": "MC62CM12AA", "ub_size": "245760"}
        fake_utils = FakeSocUtils(True)

        def fake_set_platform(*args, **kwargs):
            events.append(("set_platform",) + args)

        def fake_compute_graph_compile(*args, **kwargs):
            events.append("compute_graph_compile")

        fake_utils.set_platform = fake_set_platform
        monkeypatch.setattr(module, "get_soc_spec", lambda key: values[key])
        monkeypatch.setattr(module, "ascir", types.SimpleNamespace(utils=fake_utils))
        monkeypatch.setattr(
            module, "get_codegen_compile_context", lambda: _make_context(module)
        )
        monkeypatch.setattr(module, "compute_graph_compile", fake_compute_graph_compile)
        monkeypatch.setattr(module, "get_current_build_config", lambda key: str(tmpdir))
        monkeypatch.setattr(module, "is_valid_path", lambda value: True)
        monkeypatch.setattr(module.AscendCLogLevel, "LOG_INFO", "info", raising=False)
        monkeypatch.setitem(
            sys.modules,
            "autofuse.compile_adapter",
            types.SimpleNamespace(get_debug_flag=lambda: False),
        )

        module.asc_codegen_compile("kernel")

        assert events == [
            ("set_platform", "5102", 8, 245760, "MC62CM12AA"),
            "compute_graph_compile",
        ]
