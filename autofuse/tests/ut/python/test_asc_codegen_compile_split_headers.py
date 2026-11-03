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

import pytest

from compile_test_utils import (
    load_asc_codegen_compile_module,
    process_tiling_funcs_and_infershape,
)


@pytest.fixture()
def asc_codegen_compile_module():
    with load_asc_codegen_compile_module() as module:
        yield module


def test_pgo_compile_command_links_autofuse_collector(
    asc_codegen_compile_module,
):
    asc_codegen_compile_module.ASCEND_PATH = "/opt/cann"
    command = asc_codegen_compile_module.build_pgo_compile_command(
        "/tmp/graph_pgo.cpp", "/tmp/graph_pgo"
    )

    assert "-laihac_codegen" in command
    assert "-L/opt/cann/lib64" in command
    assert "-Wl,-rpath,/opt/cann/lib64" in command


def test_process_tiling_funcs_writes_split_headers(tmpdir, asc_codegen_compile_module):
    template_dict = {
        "TilingHead": "common",
        "TilingStateHeader": "state",
        "TilingLogHeader": "log",
        "TilingPgoHeader": "pgo",
        "TilingSolverHeader": "solver header",
        "TilingApiHeader": "api",
        "ACubeKernelTilingWrapperHpp": "wrapper header",
        "BCubeKernelTilingWrapperCpp": '#include "cube_kernel_tiling_wrapper.h"\nint Wrapper() { return 0; }',
        "solver_func": '#include "autofuse_tiling_func_solver.h"\nint Solver() { return 0; }',
    }

    process_tiling_funcs_and_infershape(
        asc_codegen_compile_module,
        {"main": template_dict},
        "demo_graph",
        str(tmpdir),
        "infer",
    )

    assert not tmpdir.join("autofuse_tiling_func_common.h").check()
    assert tmpdir.join("autofuse_tiling_func_state.h").read() == "state"
    assert tmpdir.join("autofuse_tiling_func_log.h").read() == "log"
    assert tmpdir.join("autofuse_tiling_func_pgo.h").read() == "pgo"
    assert tmpdir.join("autofuse_tiling_func_solver.h").read() == "solver header"
    assert tmpdir.join("autofuse_tiling_func_api.h").read() == "api"
    assert tmpdir.join("cube_kernel_tiling_wrapper.h").read() == "wrapper header"
    assert tmpdir.join("cube_kernel_tiling_wrapper.cpp").check()
    cpp = tmpdir.join("demo_graph_tiling_func_solver_func.cpp").read()
    assert cpp.startswith('#include "autofuse_tiling_func_solver.h"')


def test_process_tiling_funcs_writes_split_headers_for_cv_common(
    tmpdir, asc_codegen_compile_module
):
    template_dict = {
        "TilingHead": "common",
        "TilingStateHeader": "state",
        "schedule_group_tail": '#include "autofuse_tiling_func_state.h"\nint Tail() { return 0; }',
    }

    process_tiling_funcs_and_infershape(
        asc_codegen_compile_module,
        {"common": template_dict},
        "demo_graph",
        str(tmpdir),
        "infer",
    )

    cv_common = tmpdir.join("cv_common")
    assert not cv_common.join("autofuse_tiling_func_common.h").check()
    assert cv_common.join("autofuse_tiling_func_state.h").read() == "state"
    assert cv_common.join("demo_graph_tiling_func_schedule_group_tail.cpp").check()


def test_process_tiling_funcs_keeps_common_header_for_legacy_format(
    tmpdir, asc_codegen_compile_module
):
    template_dict = {
        "TilingHead": "common",
        "solver_func": "int Solver() { return 0; }",
    }

    process_tiling_funcs_and_infershape(
        asc_codegen_compile_module,
        {"main": template_dict},
        "demo_graph",
        str(tmpdir),
        "infer",
    )

    assert tmpdir.join("autofuse_tiling_func_common.h").read() == "common"
