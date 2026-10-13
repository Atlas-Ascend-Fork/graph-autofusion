# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from jit_adapter import (
    patch_device_kernel,
    patch_single_operator_tiling,
    resolve_soc_version,
)

SIX_ARG_TILING = """
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData* tiling, uint64_t* workspaceSize, uint32_t *blockDim, ResLimit *res_limit)
{
 return 0;
}
struct ResLimit {
  uint32_t valid_num = 0;
};
"""

FOUR_ARG_TILING = """
extern "C" int64_t AutofuseTiling(AutofuseTilingData* tiling, uint64_t* workspaceSize, uint32_t *blockDim, ResLimit *res_limit)
{
 return 0;
}
"""


class TestPatchSingleOperatorTiling:
    @staticmethod
    def test_rewrites_six_arg_entry_into_forwarding_four_arg():
        patched = patch_single_operator_tiling(SIX_ARG_TILING, 128, 130)
        assert "AutofuseTilingS0S1" in patched
        assert "AutofuseTiling(AutofuseTilingData *tiling," in patched
        assert "AutofuseTilingS0S1(128, 130, tiling, workspaceSize," in patched
        assert "int64_t AutofuseTiling(uint32_t s0, uint32_t s1" not in patched

    @staticmethod
    def test_leaves_four_arg_entry_untouched():
        patched = patch_single_operator_tiling(FOUR_ARG_TILING, 128, 130)
        assert patched == FOUR_ARG_TILING

    @staticmethod
    def test_leaves_unrelated_source_untouched():
        source = (
            'extern "C" int64_t AutofuseTilingWithConfig(const char *cfg) { return 0; }'
        )
        assert patch_single_operator_tiling(source, 128, 130) == source

    @staticmethod
    def test_keeps_adapter_inside_split_host_markers():
        patched = patch_single_operator_tiling(SIX_ARG_TILING, 8, 8)
        opening = patched.find("struct ResLimit;")
        closing = patched.rfind("struct ResLimit {")
        assert opening != -1 and closing != -1 and opening < closing


class TestResolveSocVersion:
    @staticmethod
    def test_profile_field_wins_without_cli_value(tmp_path):
        profile = tmp_path / "device.json"
        profile.write_text(json.dumps({"soc_version": "Ascend910B4"}), encoding="utf-8")
        assert resolve_soc_version(str(profile), "") == "Ascend910B4"

    @staticmethod
    def test_cli_value_overrides_profile(tmp_path):
        profile = tmp_path / "device.json"
        profile.write_text(json.dumps({"soc_version": "Ascend910B4"}), encoding="utf-8")
        assert resolve_soc_version(str(profile), "Ascend950") == "Ascend950"

    @staticmethod
    def test_missing_profile_field_is_an_error(tmp_path):
        profile = tmp_path / "device.json"
        profile.write_text(json.dumps({"profile": "x"}), encoding="utf-8")
        with pytest.raises(ValueError):
            resolve_soc_version(str(profile), "")


class TestPatchDeviceKernel:
    @staticmethod
    def test_names_generated_kernel_and_launch_site():
        source = (
            'extern "C" __global__ __aicore__ void (GM_ADDR data, GM_ADDR output, '
            "GM_ADDR workspace, AutofuseTilingData t) {}\n"
            "  <<<blockDim, nullptr, stream>>>((uint8_t*)input0);\n"
        )
        patched = patch_device_kernel(source)
        assert "void autofuse_kernel(GM_ADDR data" in patched
        assert "autofuse_kernel<<<blockDim, nullptr, stream>>>" in patched

    @staticmethod
    def test_leaves_named_kernel_untouched():
        source = 'extern "C" __global__ __aicore__ void named(GM_ADDR data) {}'
        assert patch_device_kernel(source) == source
