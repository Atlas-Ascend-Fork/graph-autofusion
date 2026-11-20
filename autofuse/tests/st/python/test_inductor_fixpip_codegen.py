# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""CPU-only code generation regressions for the installed Autofuse extension."""

import pytest
from test_inductor_fixpip_device_e2e import _make_graph

from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions


@pytest.fixture()
def fixpip_codegen_platform(monkeypatch):
    monkeypatch.setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=false")
    ascir.utils.set_platform("3510", 1, 245760)
    try:
        yield
    finally:
        ascir.utils.set_platform("2201", 1, 245760)


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("bin_mode", [False, True], ids=["launch", "bin"])
@pytest.mark.parametrize("variant", ["plain", "b_view", "shared_store_outputs"])
def test_fixpip_generated_source_contract(
    fixpip_codegen_platform, dynamic, bin_mode, variant
):
    batch = variant == "b_view"
    shape = (4, 16, 64, 16) if batch else (128, 128, 128)
    graph = _make_graph(ascir, dynamic, batch, shape, variant)
    fuser = Autofuser(AutofuserOptions())
    scheduled = fuser.schedule(graph)
    tiling, host, device = (
        fuser.codegen_bin(scheduled) if bin_mode else fuser.codegen(scheduled)
    )
    assert "struct CVAutofuseTilingData {" in tiling
    if variant == "shared_store_outputs":
        assert "output2" in device
        assert "CV_FUSION_ADDR" in device
        return
    assert "autofuse_tiling_func_pgo.h" not in host
    assert "optiling::EmitFinalTilingByCase(" not in host
    assert "optiling::GetTilingCore(" not in host
    assert "mat_mul_v3_fusion" not in device
    assert "class TensorCVBlockMmad" not in device
    if batch:
        declaration = device.index("GM_ADDR cv_bmm_input = ")
        materialize = device.index("CvMaterializeBmmInput<", declaration)
        native_call = device.index("batch_mat_mul_v3<", materialize)
        assert declaration < materialize < native_call
        reset = host.index("tiling->cv_tiling_data = {};")
        call = host.index("if (!CallCubeTiling(", reset)
        assert reset < call
        assert "tiling->cv_tiling_data = {};" not in host[call:]
        assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)" in device
