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

from compile_test_utils import (
    load_asc_codegen_compile_module,
    process_tiling_funcs_and_infershape,
)


@pytest.fixture()
def asc_codegen_compile_module():
    with load_asc_codegen_compile_module() as module:
        yield module


@pytest.mark.parametrize(
    "vector_key,cube_key,large_k,expected_common",
    [
        (0, 1, False, False),
        (0, 2, False, False),
        (0, 0x20001, False, False),
        (0, 0x40001, False, True),
        (-1, 0x20001, False, True),
        (0, 1, True, True),
        (0, 0x20001, True, True),
    ],
)
def test_static_cv_ub_keys_match_runtime_policy_and_keep_fallbacks(
    asc_codegen_compile_module,
    monkeypatch,
    vector_key,
    cube_key,
    large_k,
    expected_common,
):
    module = asc_codegen_compile_module
    monkeypatch.setattr(module, "is_fp32_large_k_matmul", lambda *args: large_k)

    assert (
        module._should_use_safety_fusion(vector_key, cube_key, None, None, False)
        is expected_common
    )


@pytest.mark.parametrize("cube_key", [257, 258, 274, 322, 338, 769, 770, 2097410])
def test_static_iterbatch_covers_all_launch_blocks(
    asc_codegen_compile_module, monkeypatch, cube_key
):
    module = asc_codegen_compile_module
    monkeypatch.setattr(
        module, "static_shape_cv_common_compile", lambda **kwargs: (56, 4096)
    )
    tiling = SimpleNamespace(tiling_key=cube_key, file_content="")
    context = module.SafetyFusionCtx({}, None, tiling, True, 1, -1, [False])

    macros = module._apply_safety_fusion(context)

    assert "#define CUBE_BLOCK_DIM 28" in macros
    assert "#define CV_AIC_NUM 28" in tiling.file_content
    assert "#define CV_AIV_NUM 56" in tiling.file_content


@pytest.mark.parametrize(
    "cube_key,needs_mix", [(2, False), (258, False), (4114, True), (1048578, True)]
)
def test_tensor_common_keeps_native_mix_requirements(
    asc_codegen_compile_module, monkeypatch, cube_key, needs_mix
):
    module = asc_codegen_compile_module
    monkeypatch.setattr(
        module, "static_shape_cv_common_compile", lambda **kwargs: (56, 4096)
    )
    tiling = SimpleNamespace(tiling_key=cube_key, file_content="")
    context = module.SafetyFusionCtx({}, None, tiling, cube_key == 258, 1, -1, [False])

    module._apply_safety_fusion(context)

    assert ("#define CV_SAFETY_FUSION_MIX_MODE 1" in tiling.file_content) is needs_mix
    assert (
        "#define CV_AIC_NUM " + ("28" if cube_key == 258 else "1")
        in tiling.file_content
    )


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


@pytest.mark.parametrize("is_static_shape", [False, True])
@pytest.mark.parametrize("is_cube", [False, True])
@pytest.mark.parametrize("is_cross_compile", [False, True])
def test_cv_host_dependencies_only_apply_to_dynamic_cube(
    asc_codegen_compile_module, tmpdir, is_static_shape, is_cube, is_cross_compile
):
    module = asc_codegen_compile_module
    module.ASCEND_PATH = "/opt/cann"
    context = module.HostCompileContext(
        "graph", "kernel", str(tmpdir), False, is_static_shape, is_cube
    )
    cross_info = module.CrossCompileInfo(is_cross_compile, "cross-", "aarch64")

    module.generate_cmake_lists(context, cross_info=cross_info)

    cmake = tmpdir.join("CMakeLists.txt").read()
    expected = not is_static_shape and is_cube
    for dependency in (
        "_GLIBCXX_USE_CXX11_ABI=0",
        "    /opt/cann/pkg_inc\n",
        "    /opt/cann/include/external\n",
        "target_link_libraries(kernel PRIVATE opp_registry)",
        "    cube_kernel_tiling_wrapper.cpp\n",
    ):
        assert (dependency in cmake) is expected


def test_dynamic_shape_host_uses_dynamic_compile_context(
    asc_codegen_compile_module, monkeypatch, tmpdir
):
    module = asc_codegen_compile_module
    contexts = []

    class CompileStopped(Exception):
        pass

    def stop_before_compilation(context, **kwargs):
        contexts.append(context)
        raise CompileStopped()

    monkeypatch.setattr(module, "ascbc_host_compile", stop_before_compilation)
    with pytest.raises(CompileStopped):
        module.dynamic_shape_compile(("kernel", str(tmpdir), "graph"), is_cube=True)
    assert contexts[0].is_static_shape is False


@pytest.mark.parametrize("use_cv_common", [False, True])
@pytest.mark.parametrize("has_stage_size", [False, True])
def test_dynamic_shape_preserves_device_tiling_definition(
    asc_codegen_compile_module, monkeypatch, tmpdir, use_cv_common, has_stage_size
):
    import ctypes
    from types import SimpleNamespace

    module = asc_codegen_compile_module
    device_dir = tmpdir.mkdir("device")
    if use_cv_common:
        device_dir = device_dir.mkdir("cv_common")
    original = "struct AutofuseTilingData { unsigned block_dim; };\n"
    device_dir.join("autofuse_tiling_data.h").write(original)
    monkeypatch.setattr(module, "ascbc_host_compile", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "ascendc_clean", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.AscendCLogLevel, "LOG_INFO", 0, raising=False)
    lib = SimpleNamespace()
    if has_stage_size:
        lib.GetCVUBFusionStageSizeName = object()
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: lib)
    monkeypatch.setattr(
        module, "get_cv_ub_fusion_stage_size_name", lambda **kwargs: "tile_size"
    )

    module.dynamic_shape_compile(
        ("kernel", str(tmpdir), "graph"), use_cv_common=[use_cv_common], is_cube=True
    )
    assert device_dir.join("autofuse_tiling_data_bak.h").read() == original
    expected = "\n#define STAGE_SIZE_NAME tile_size\n" if has_stage_size else ""
    assert device_dir.join("autofuse_tiling_data.h").read() == expected + original


@pytest.mark.parametrize("use_cv_common", [False, True])
def test_static_shape_preserves_const_only_device_header(
    asc_codegen_compile_module, monkeypatch, tmpdir, use_cv_common
):
    import ctypes
    from types import SimpleNamespace

    module = asc_codegen_compile_module
    device_dir = tmpdir.mkdir("device")
    if use_cv_common:
        device_dir = device_dir.mkdir("cv_common")
    original = "struct AutofuseTilingData { unsigned block_dim; };\n"
    device_dir.join("autofuse_tiling_data.h").write(original)
    const_tiling_data = b"constexpr unsigned block_dim = 1;\n"
    lib = SimpleNamespace(
        GenConstTilingData=lambda *args: const_tiling_data,
        GetCVUBFusionStageSizeName=object(),
    )
    monkeypatch.setattr(module, "ascbc_host_compile", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "ascendc_clean", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module, "static_shape_kernel_proc", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(module, "get_static_compile_vector_core_num", lambda _: 1)
    monkeypatch.setattr(module, "get_soc_spec", lambda _: 1024)
    monkeypatch.setattr(module.AscendCLogLevel, "LOG_INFO", 0, raising=False)
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: lib)
    monkeypatch.setattr(
        module, "get_cv_ub_fusion_stage_size_name", lambda **kwargs: "tile_size"
    )

    module.static_shape_compile(
        "kernel", str(tmpdir), "graph", use_cv_common=[use_cv_common], is_cube=True
    )

    assert device_dir.join("autofuse_tiling_data_bak.h").read() == original
    assert device_dir.join("autofuse_tiling_data.h").read() == (
        const_tiling_data.decode() + "\n#define STAGE_SIZE_NAME tile_size\n"
    )


def test_dynamic_shape_cross_compile_preserves_dynamic_context(
    asc_codegen_compile_module, monkeypatch, tmpdir
):
    module = asc_codegen_compile_module
    contexts = []

    class CompileStopped(Exception):
        pass

    def capture_native(context):
        contexts.append(context)
        raise CompileStopped()

    monkeypatch.setattr(
        module, "ascbc_host_compile", lambda context, **kwargs: contexts.append(context)
    )
    monkeypatch.setattr(module, "compile_native_host_so", capture_native)
    cross = module.CrossCompileInfo(True, "cross-", "aarch64")
    with pytest.raises(CompileStopped):
        module.dynamic_shape_compile(
            ("kernel", str(tmpdir), "graph"), is_cube=True, cross_info=cross
        )
    assert len(contexts) == 2
    assert all(context.is_static_shape is False for context in contexts)


def test_dynamic_cv_header_contains_runtime_stage_fields(
    asc_codegen_compile_module, tmpdir
):
    module = asc_codegen_compile_module
    inputs = [{"dtype": "float32"}]
    outputs = [{"dtype": "float32"}]
    module.create_matmul_dynamic_tiling_data(
        str(tmpdir), [4, False, [False], False, inputs, outputs]
    )
    host = tmpdir.join("host", "autofuse_cube_tiling_data.h").read()
    device = tmpdir.join("device", "autofuse_cube_tiling_data.h").read()
    for content in (host, device):
        assert "uint32_t stage_size_name;" in content
        assert "uint32_t cube_ub_stage_size;" in content
        assert "uint32_t cube_m_stage_size;" in content
        assert "#define CV_DYNAMIC_TILING_MAX_SIZE(lhs, rhs)" in content
        assert "alignas(8) uint8_t matmul_tiling_data[" in content
        assert "sizeof(MatMulV3TilingDataCopy)" in content

    batch_dir = tmpdir.mkdir("batch")
    module.create_matmul_dynamic_tiling_data(
        str(batch_dir), [4, True, [False], False, inputs, outputs]
    )
    batch_host = batch_dir.join("host", "autofuse_cube_tiling_data.h").read()
    assert "sizeof(BatchMatMulV3MergeBatchBasicTilingData)" in batch_host


@pytest.mark.parametrize("is_batch", [False, True])
@pytest.mark.parametrize("vector_words", [1, 2])
def test_dynamic_cv_vector_view_preserves_runtime_abi(
    asc_codegen_compile_module, tmpdir, is_batch, vector_words
):
    import subprocess

    module = asc_codegen_compile_module
    inputs = [{"dtype": "float32"}]
    module.create_matmul_dynamic_tiling_data(
        str(tmpdir), [4, is_batch, [False], False, inputs, inputs]
    )
    host = tmpdir.join("host")
    arch = host.mkdir("arch35")
    native_types = {
        "MatMulV3BasicTilingData": 92,
        "MatMulV3TilingDataCopy": 744,
        "MatMulV3TilingData": 232,
        "MatMulToMulBasicTilingData": 112,
        "MatMulV3KEqZeroBasicTilingData": 64,
        "BatchMatMulV3BasicTilingData": 100,
        "BatchMatMulV3TilingData": 384,
        "BatchMatMulV3IterBatchBasicTilingData": 72,
        "BatchMatMulV3MergeBatchBasicTilingData": 128,
        "BatchMatMulToMulBasicTilingData": 160,
    }
    arch.join("mat_mul_tiling_data.h").write(
        "#include <cstdint>\n"
        + "\n".join(
            "struct %s { uint8_t bytes[%d]; };" % (name, size)
            for name, size in native_types.items()
        )
    )
    host.join("autofuse_tiling_data.h").write(
        "struct AutofuseTilingData { unsigned tiling_key; unsigned words[%d]; };\n"
        % vector_words
    )
    source = host.join("check_prefix.cpp")
    source.write(r"""
#include <cassert>
#include <cstddef>
#include <cstring>
#include "autofuse_cube_tiling_data.h"
struct PreviousCVTilingData {
    AutofuseTilingData tiling_data;
    CVTilingData cv_tiling_data;
    uint32_t stage_size_name;
    uint32_t cube_ub_stage_size;
    uint32_t cube_m_stage_size;
    alignas(8) uint8_t matmul_tiling_data[sizeof(CVAutofuseTilingData::matmul_tiling_data)];
};
static_assert(sizeof(CVAutofuseTilingData) == sizeof(PreviousCVTilingData), "GM ABI changed");
#define CHECK_OFFSET(field) \
    static_assert(offsetof(CVVectorTilingData, field) == offsetof(PreviousCVTilingData, field), "prefix moved"); \
    static_assert(offsetof(CVAutofuseTilingData, field) == offsetof(PreviousCVTilingData, field), "GM field moved")
CHECK_OFFSET(tiling_data);
CHECK_OFFSET(cv_tiling_data);
CHECK_OFFSET(stage_size_name);
CHECK_OFFSET(cube_ub_stage_size);
CHECK_OFFSET(cube_m_stage_size);
static_assert(sizeof(CVVectorTilingData) == offsetof(PreviousCVTilingData, matmul_tiling_data), "wrong copy size");
static_assert(sizeof(CVVectorTilingData) < sizeof(CVAutofuseTilingData), "native payload still copied");
int main() {
    CVAutofuseTilingData gm{};
    gm.tiling_data.tiling_key = 7;
    gm.cv_tiling_data.fusion_mode = 1;
    gm.cv_tiling_data.cv_aic_num = 28;
    gm.cv_tiling_data.cv_aiv_num = 56;
    gm.stage_size_name = 11776;
    gm.cube_ub_stage_size = 26880;
    gm.cube_m_stage_size = 105;
    gm.matmul_tiling_data[0] = 0xa5;
    CVVectorTilingData view{};
    std::memcpy(&view, &gm, sizeof(view));
    assert(view.tiling_data.tiling_key == 7);
    assert(view.cv_tiling_data.fusion_mode == 1);
    assert(view.cv_tiling_data.cv_aic_num == 28 && view.cv_tiling_data.cv_aiv_num == 56);
    assert(view.stage_size_name == 11776 && view.cube_ub_stage_size == 26880 && view.cube_m_stage_size == 105);
    const auto *payload = reinterpret_cast<const uint8_t *>(&gm) + sizeof(view);
    assert(payload[0] == 0xa5);
}
""")
    binary = host.join("check_prefix")
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-I" + str(host), str(source), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


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


@pytest.mark.parametrize(
    "cube_info,expected_common",
    [
        (None, True),
        ([], True),
        ([4, True], True),
        ([2, False], True),
        ([4, False], False),
    ],
)
def test_tensor_bfull_ub_requires_matmul_fp32_output(
    asc_codegen_compile_module, cube_info, expected_common
):
    assert (
        asc_codegen_compile_module._should_use_safety_fusion(
            0, 0x20002, None, cube_info, False
        )
        is expected_common
    )
