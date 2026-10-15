#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

"""Unit tests of the stub sub-kernels used by the SuperKernel system tests."""

import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from st.stub.stub_kernel import (
    SCENARIO_OVERRIDES,
    SPLIT_MEMBER_COUNT,
    STUB_TEXT_SIZE,
    SUB_KERNEL_BASE,
    _build_elf,
    materialize_sub_kernels,
    render_sub_kernel_json,
    resolve_spec,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "st" / "data"


def _text_size_of(elf_bytes):
    """Read the size of the first section (``.text``) of a generated ELF."""
    section_header_offset = struct.unpack_from("<Q", elf_bytes, 0x28)[0]
    return struct.unpack_from("<Q", elf_bytes, section_header_offset + 64 + 32)[0]


@pytest.mark.parametrize("kernel", sorted(SUB_KERNEL_BASE))
def test_stub_text_size_is_pinned(kernel):
    """The .text size drives the preload count of the generated kernel."""
    assert _text_size_of(_build_elf(STUB_TEXT_SIZE[kernel])) == STUB_TEXT_SIZE[kernel]


def test_stub_object_is_an_archive(tmp_path):
    """The stub sub-kernel must be extractable by ``ar x``."""
    if shutil.which("ar") is None:
        pytest.skip("ar is not available")

    sub_kernels = materialize_sub_kernels(tmp_path)
    for kernel, path in sub_kernels.items():
        members = subprocess.run(
            ["ar", "t", str(path.o())], check=True, capture_output=True, text=True
        ).stdout.split()
        expected = [f"{kernel}.o"] + [
            f"{kernel}_split{index}.o" for index in range(1, SPLIT_MEMBER_COUNT + 1)
        ]
        assert sorted(members) == sorted(expected)


def test_default_spec_and_overrides():
    """Scenarios without an entry fall back to the default specification."""
    default = resolve_spec("test_sk_1_stream_2_ops_default_aic_only", "is_inf")
    assert (default.kernel_type, default.split_mode, default.sync_all) == (
        "KERNEL_TYPE_AIC_ONLY",
        4,
        False,
    )

    mix = resolve_spec("test_sk_2_stream_2_ops_default_aic_1_1", "is_inf")
    assert mix.kernel_type == "KERNEL_TYPE_MIX_AIC_1_1"

    without_split = resolve_spec(
        "test_sk_1_stream_2_ops_json_split_none_aic_only", "is_inf"
    )
    assert without_split.split_mode is None


def test_rendered_metadata_follows_spec():
    """The rendered metadata must carry the scenario specific variation."""
    bin_path = Path("kernel_meta") / "is_inf.o"

    mix_meta = render_sub_kernel_json(
        "is_inf",
        resolve_spec("test_sk_2_stream_2_ops_default_aic_1_1", "is_inf"),
        bin_path,
    )
    assert "dav-c220-cube" in mix_meta["sub_operator_kernel_name"]
    assert "dav-c220-vec" in mix_meta["sub_operator_kernel_name"]

    no_split_meta = render_sub_kernel_json(
        "is_inf",
        resolve_spec("test_sk_1_stream_2_ops_json_split_none_aic_only", "is_inf"),
        bin_path,
    )
    assert "split_mode" not in no_split_meta

    dynamic_meta = render_sub_kernel_json(
        "is_inf",
        resolve_spec(
            "test_sk_2_stream_2_ops_dynamic_send_recv_default_aic_only", "is_inf"
        ),
        bin_path,
    )
    assert "dynamic_func_names" in dynamic_meta["sub_operator_kernel_name"]


def test_overrides_reference_existing_scenarios():
    """A typo in a scenario name would silently fall back to the default."""
    known = {path.name for path in DATA_DIR.iterdir() if path.is_dir()}
    unknown = sorted(set(SCENARIO_OVERRIDES) - known)
    assert not unknown, f"SCENARIO_OVERRIDES references unknown scenarios: {unknown}"
