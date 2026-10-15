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

"""Fixtures providing stub sub-kernels to the SuperKernel system tests.

The sub-kernels are generated at test time (see ``st.stub.stub_kernel``) instead
of being compiled from an operator implementation, so the system tests neither
commit binaries nor depend on a CANN operator package.
"""

from pathlib import Path

import pytest

from st.stub.stub_kernel import materialize_sub_kernels

# Names used in the ``parametrize`` tables mapped to the sub-kernel they select.
FIXTURE_KERNELS = {
    "subkernel_is_inf": "is_inf",
    "subkernel_is_inf_default": "is_inf",
    "subkernel_is_inf_split_mode1": "is_inf",
    "subkernel_is_finite": "is_finite",
    "subkernel_is_finite_default": "is_finite",
    "subkernel_is_finite_split_mode1": "is_finite",
    "subkernel_pows": "pows",
    "subkernel_pows_default": "pows",
}


@pytest.fixture(scope="session")
def stub_subkernels(tmp_dir):
    """Generate every stub sub-kernel once per test session."""
    return materialize_sub_kernels(Path(tmp_dir))


def make_subkernel_fixture(kernel):
    """Create a fixture returning the stub sub-kernel ``kernel``."""

    @pytest.fixture(scope="function")
    def fixture_func(stub_subkernels):
        return stub_subkernels[kernel]

    return fixture_func


subkernel_is_inf = make_subkernel_fixture("is_inf")
subkernel_is_inf_default = make_subkernel_fixture("is_inf")
subkernel_is_inf_split_mode1 = make_subkernel_fixture("is_inf")
subkernel_is_finite = make_subkernel_fixture("is_finite")
subkernel_is_finite_default = make_subkernel_fixture("is_finite")
subkernel_is_finite_split_mode1 = make_subkernel_fixture("is_finite")
subkernel_pows_default = make_subkernel_fixture("pows")


@pytest.fixture
def subkernel_inf(request, stub_subkernels):
    """Resolve the sub-kernel selected by the current parametrisation."""
    return stub_subkernels[FIXTURE_KERNELS[request.param]]


@pytest.fixture
def subkernel_finite(request, stub_subkernels):
    """Resolve the sub-kernel selected by the current parametrisation."""
    return stub_subkernels[FIXTURE_KERNELS[request.param]]


@pytest.fixture
def subkernel_pows(request, stub_subkernels):
    """Resolve the sub-kernel selected by the current parametrisation."""
    return stub_subkernels[FIXTURE_KERNELS[request.param]]
