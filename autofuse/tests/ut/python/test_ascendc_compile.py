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

import hashlib
import json
import os
import time
import types
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import pytest

from compile_test_utils import (
    PYTHON_DIR,
    load_compile_module,
    make_tbe_log_stub_modules,
)

MODULE_NAME = "autofuse.compiler.python.ascendc_compile"
MODULE_PATH = os.path.join(PYTHON_DIR, "ascendc_compile.py")

_LAUNCH_FUNC = (
    'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, void* output0, '
    "void* workspace, AutofuseTilingData* tiling_data)\n"
    "{\n"
    "  static_kernel<<<blockDim, nullptr, stream>>>(\n"
    "      (uint8_t*)input0, (uint8_t*)output0,\n"
    "      (uint8_t*)workspace, *tiling_data);\n"
    "  return 0;\n"
    "}\n"
)


def fake_get_soc_spec(key):
    return {
        "vector_core_cnt": 20,
        "ub_size": 262144,
    }.get(key, "")


@pytest.fixture()
def ascendc_compile_module():
    platform_info_module = types.ModuleType(
        "asc_op_compile_base.common.platform.platform_info"
    )
    platform_info_module.get_soc_spec = fake_get_soc_spec
    extra_modules = {
        "asc_op_compile_base": types.ModuleType("asc_op_compile_base"),
        "asc_op_compile_base.common": types.ModuleType("asc_op_compile_base.common"),
        "asc_op_compile_base.common.platform": types.ModuleType(
            "asc_op_compile_base.common.platform"
        ),
        "asc_op_compile_base.common.platform.platform_info": platform_info_module,
        # Stub tbe log so the fixture never triggers the real (slow, and
        # process-global-state polluting) TBE package init.
        **make_tbe_log_stub_modules(),
    }
    with load_compile_module(
        MODULE_NAME, MODULE_PATH, extra_modules=extra_modules
    ) as loaded_module:
        yield loaded_module


def _noop_run_compile_command(cmd, stage_name):
    return None


def _make_host_pgo_args(tmpdir):
    return type(
        "Args",
        (),
        {
            "stage": "host",
            "temp_dir": str(tmpdir),
            "output_file": str(tmpdir.join("tiling.so")),
            "pgo_runner_file": str(tmpdir.join("runner.cpp")),
            "pgo_device_file": str(tmpdir.join("device.cpp")),
        },
    )()


def _make_pgo_bundle(module, artifacts, output_file, generation, ld_preload=""):
    return module.PgoBundle(*artifacts, output_file, generation, ld_preload=ld_preload)


@dataclass
class CopySoToOutputCase:
    src_file: object
    dst_file: object
    output_dir: object
    wrapper_file: object
    src_directory: str
    args: object


def _make_copy_so_to_output_case(tmpdir, wrapper_in_output_dir=False):
    src_file = tmpdir.join("source.so")
    src_file.write("kernel")
    output_dir = tmpdir.mkdir("kernel_meta")
    if wrapper_in_output_dir:
        wrapper_dir = output_dir.mkdir("cv_tiling_wrapper_cache")
    else:
        wrapper_dir = tmpdir.mkdir("cache")
    wrapper_file = wrapper_dir.join("libautofuse_cv_tiling_wrapper_abc.so")
    wrapper_file.write("wrapper")
    dst_file = output_dir.join("target.so")
    src_directory = os.getcwd()
    args = type(
        "Args",
        (),
        {
            "output_file": str(dst_file),
            "shared_cv_wrapper_so": str(wrapper_file),
            "stage": "host",
            "graph_name": "graph",
        },
    )()
    return CopySoToOutputCase(
        src_file, dst_file, output_dir, wrapper_file, src_directory, args
    )


def _make_soc_type_args(soc_version):
    return type("Args", (), {"soc_version": soc_version})()


def test_get_soc_type_maps_supported_soc_versions(ascendc_compile_module):
    get_soc_type = ascendc_compile_module.get_soc_type

    assert get_soc_type(_make_soc_type_args("Ascend910B4")) == "dav-2201"
    assert get_soc_type(_make_soc_type_args("Ascend910_9391")) == "dav-2201"
    assert get_soc_type(_make_soc_type_args("Ascend950PR_9579")) == "dav-3510"
    assert get_soc_type(_make_soc_type_args("Ascend350_355e")) == "dav-3510"
    assert get_soc_type(_make_soc_type_args("Ascend350_354f")) == "dav-3510"


def test_get_soc_type_rejects_unknown_soc_version(ascendc_compile_module):
    with pytest.raises(ValueError, match="Unsupported soc_version"):
        ascendc_compile_module.get_soc_type(_make_soc_type_args("Ascend310P"))
    # Ascend350_355e 仅支持全等匹配，其他 Ascend350 前缀变体不受支持
    with pytest.raises(ValueError, match="Unsupported soc_version"):
        ascendc_compile_module.get_soc_type(_make_soc_type_args("Ascend350_355"))


def test_link_shared_adds_requested_libraries(ascendc_compile_module):
    captured = {}

    def fake_run_compile_command(cmd, stage_name):
        captured["cmd"] = cmd
        captured["stage_name"] = stage_name

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command
    ascendc_compile_module.module.ASCEND_PATH = "/usr/local/Ascend/cann"
    ascendc_compile_module.module.machine = "x86_64"

    result = ascendc_compile_module.link_shared(
        "kernel.so", ["host.o"], link_libraries=["graph_base", "register"]
    )

    assert result == "kernel.so"
    assert captured["stage_name"] == "LinkObj"
    assert "-L" in captured["cmd"]
    assert "/usr/local/Ascend/cann/x86_64-linux/lib64" in captured["cmd"]
    assert "-lgraph_base" in captured["cmd"]
    assert "-lregister" in captured["cmd"]
    assert captured["cmd"].index("-lgraph_base") < captured["cmd"].index("-lregister")


def test_link_shared_skips_libraries_by_default(ascendc_compile_module):
    captured = {}

    def fake_run_compile_command(cmd, stage_name):
        captured["cmd"] = cmd

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command

    ascendc_compile_module.link_shared("kernel.so", ["device.o"])

    assert "-lgraph_base" not in captured["cmd"]


def test_link_shared_appends_extra_link_options(ascendc_compile_module):
    captured = {}

    def fake_run_compile_command(cmd, stage_name):
        captured["cmd"] = cmd

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command

    ascendc_compile_module.link_shared(
        "kernel.so",
        ["host.o"],
        extra_link_options=["-Wl,-rpath,$ORIGIN/cv_tiling_wrapper_cache"],
    )

    assert "-Wl,-rpath,$ORIGIN/cv_tiling_wrapper_cache" in captured["cmd"]


def test_link_pgo_executable_uses_host_runtime_without_profiler_libraries(
    ascendc_compile_module,
):
    captured = {}

    def fake_run_compile_command(cmd, stage_name):
        captured["cmd"] = cmd
        captured["stage_name"] = stage_name

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command
    ascendc_compile_module.module.ASCEND_PATH = "/usr/local/Ascend/cann"
    ascendc_compile_module.module.machine = "x86_64"

    result = ascendc_compile_module.link_pgo_executable(
        "pgo_runner", ["solver.o", "runner.o"]
    )

    assert result == "pgo_runner"
    assert captured["stage_name"] == "LinkPgoExecutable"
    assert "--shared" not in captured["cmd"]
    assert captured["cmd"][:3] == [
        "/usr/local/Ascend/cann/tools/bisheng_compiler/bin/bisheng",
        "solver.o",
        "runner.o",
    ]
    for option in [
        "-ltiling_api",
        "-lplatform",
        "-lgraph_base",
        "-lregister",
        "-lascendcl",
        "-lruntime",
        "-lunified_dlog",
        "-lascendalog",
        "-lc_sec",
        "-lm",
        "-lstdc++",
        "-ldl",
        "-lpthread",
    ]:
        assert option in captured["cmd"]


def test_link_pgo_executable_propagates_link_failure(ascendc_compile_module):
    def fake_run_compile_command(cmd, stage_name):
        raise ascendc_compile_module.CompileError("link failed")

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command

    with pytest.raises(ascendc_compile_module.CompileError, match="link failed"):
        ascendc_compile_module.link_pgo_executable("pgo_runner", ["runner.o"])


def test_extract_aicore_binary_uses_bisheng_objcopy_fallback(
    ascendc_compile_module, tmpdir, monkeypatch
):
    ascend_path = tmpdir.mkdir("cann")
    objcopy = (
        ascend_path.mkdir("tools")
        .mkdir("bisheng_compiler")
        .mkdir("bin")
        .join("llvm-objcopy")
    )
    objcopy.write("tool")
    output_file = str(tmpdir.join("kernel.elf"))
    captured = {}
    ascendc_compile_module.module.ASCEND_PATH = str(ascend_path)
    monkeypatch.setattr(ascendc_compile_module.shutil, "which", lambda _: None)

    def fake_run_compile_command(cmd, stage_name):
        captured["cmd"] = cmd
        captured["stage_name"] = stage_name
        open(output_file, "wb").write(b"device-elf")

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command

    result = ascendc_compile_module.extract_aicore_binary("device.o", output_file)

    assert result == output_file
    assert captured == {
        "cmd": [
            str(objcopy),
            "--dump-section",
            f".aicore_binary={output_file}",
            "device.o",
        ],
        "stage_name": "ExtractPgoDeviceBinary",
    }


def test_extract_aicore_binary_requires_objcopy(
    ascendc_compile_module, tmpdir, monkeypatch
):
    ascendc_compile_module.module.ASCEND_PATH = str(tmpdir.mkdir("cann"))
    monkeypatch.setattr(ascendc_compile_module.shutil, "which", lambda _: None)

    with pytest.raises(ascendc_compile_module.CompileError, match="llvm-objcopy"):
        ascendc_compile_module.extract_aicore_binary(
            "device.o", str(tmpdir.join("kernel.elf"))
        )


def test_extract_aicore_binary_rejects_empty_output(
    ascendc_compile_module, tmpdir, monkeypatch
):
    objcopy = tmpdir.join("llvm-objcopy")
    objcopy.write("tool")
    output_file = str(tmpdir.join("kernel.elf"))
    monkeypatch.setattr(ascendc_compile_module.shutil, "which", lambda _: str(objcopy))
    ascendc_compile_module.module.run_compile_command = _noop_run_compile_command

    with pytest.raises(ascendc_compile_module.CompileError, match="empty"):
        ascendc_compile_module.extract_aicore_binary("device.o", output_file)


def test_get_pgo_sidecar_paths_are_generation_scoped(ascendc_compile_module, tmpdir):
    output_file = str(tmpdir.join("tiling.so"))

    paths = ascendc_compile_module.get_pgo_sidecar_paths(output_file, "generation1")

    assert paths["generation_dir"] == output_file + ".pgo_v2/generation1"
    assert paths["tiling_so"].endswith("/tiling.so")
    assert paths["runner"].endswith("/tiling.so.pgo_runner")
    assert paths["kernel"].endswith("/tiling.so.pgo_kernel.aicore_binary_elf_v1")
    assert paths["manifest"].endswith("/manifest.json")


def test_build_pgo_sidecars_compiles_runner_then_device_binary(
    ascendc_compile_module, tmpdir
):
    events = []
    host_dir = tmpdir.mkdir("host")
    device_dir = tmpdir.mkdir("device")
    runner_source = str(host_dir.join("runner.cpp"))
    device_source = str(device_dir.join("device.cpp"))
    host_dir.join("runner.cpp").write("runner")
    device_dir.join("device.cpp").write("device")
    args = type(
        "Args",
        (),
        {
            "pgo_runner_file": runner_source,
            "pgo_device_file": device_source,
        },
    )()

    def fake_compile_host_obj_file(_, __, source):
        events.append(("compile_runner", source))
        return source + ".o"

    def fake_link_runner(target, objects):
        events.append(("link_runner", target, objects))
        open(target, "wb").write(b"runner")
        return target

    def fake_compile_device_obj(_, __):
        events.append(("compile_device", args.device_files))
        path = str(device_dir.join("device.o"))
        open(path, "wb").write(b"fat-object")
        return path

    def fake_extract(source, target):
        events.append(("extract_device", source, target))
        open(target, "wb").write(b"device-elf")
        return target

    ascendc_compile_module.module.compile_host_obj_file = fake_compile_host_obj_file
    ascendc_compile_module.module.link_pgo_executable = fake_link_runner
    ascendc_compile_module.module.compile_device_obj = fake_compile_device_obj
    ascendc_compile_module.module.extract_aicore_binary = fake_extract

    runner, kernel = ascendc_compile_module.build_pgo_sidecars(args, str(tmpdir))

    assert [event[0] for event in events] == [
        "compile_runner",
        "link_runner",
        "compile_device",
        "extract_device",
    ]
    assert open(runner, "rb").read() == b"runner"
    assert open(kernel, "rb").read() == b"device-elf"


def test_publish_pgo_bundle_binds_hashes_and_replaces_tiling_last(
    ascendc_compile_module, tmpdir, monkeypatch
):
    output_file = str(tmpdir.join("tiling.so"))
    built_tiling = str(tmpdir.join("built_tiling.so"))
    built_runner = str(tmpdir.join("built_runner"))
    built_kernel = str(tmpdir.join("built_kernel"))
    for path, data in (
        (built_tiling, b"tiling"),
        (built_runner, b"runner"),
        (built_kernel, b"kernel"),
    ):
        open(path, "wb").write(data)
    replace_events = []
    real_replace = os.replace

    def record_replace(source, target):
        replace_events.append((source, target))
        real_replace(source, target)

    monkeypatch.setattr(ascendc_compile_module.os, "replace", record_replace)

    bundle = _make_pgo_bundle(
        ascendc_compile_module,
        [built_tiling, built_runner, built_kernel],
        output_file,
        "generation1",
        "",
    )
    paths = ascendc_compile_module.publish_pgo_bundle(bundle)

    assert replace_events[-1][1] == output_file
    assert replace_events[-2][1] == paths["generation_dir"]
    assert open(paths["tiling_so"], "rb").read() == b"tiling"
    manifest = json.loads(open(paths["manifest"]).read())
    assert manifest["bundle_schema_version"] == 2
    assert manifest["cache_root"] == "pgo_v2"
    assert manifest["generation"] == "generation1"
    assert manifest["result_protocol_version"] == 2
    assert manifest["profiling_backend"] == "mspti_equivalent"
    assert manifest["requires_mspti"] is False
    assert (
        not {
            "protocol",
            "version",
            "runner_abi",
            "proxy_abi",
            "device_source_abi",
            "kernel_format",
        }
        & manifest.keys()
    )
    assert manifest["ld_preload"] == ""
    assert manifest["artifacts"]["tiling_so"]["file"] == "tiling.so"
    assert (
        manifest["artifacts"]["tiling_so"]["sha256"]
        == hashlib.sha256(b"tiling").hexdigest()
    )
    assert (
        manifest["artifacts"]["runner"]["sha256"]
        == hashlib.sha256(b"runner").hexdigest()
    )
    assert (
        manifest["artifacts"]["kernel"]["sha256"]
        == hashlib.sha256(b"kernel").hexdigest()
    )
    assert open(output_file, "rb").read() == b"tiling"


def test_publish_pgo_bundle_keeps_current_and_previous_generation(
    ascendc_compile_module, tmpdir
):
    output_file = str(tmpdir.join("tiling.so"))
    cache_root = output_file + ".pgo_v2"
    old_generation = cache_root + "/generation1"
    previous_generation = cache_root + "/generation2"
    os.makedirs(old_generation)
    os.makedirs(previous_generation)
    os.utime(old_generation, ns=(1, 1))
    os.utime(previous_generation, ns=(2, 2))
    artifacts = []
    for name in ("built_tiling.so", "built_runner", "built_kernel"):
        path = str(tmpdir.join(name))
        open(path, "wb").write(name.encode())
        artifacts.append(path)

    paths = ascendc_compile_module.publish_pgo_bundle(
        _make_pgo_bundle(ascendc_compile_module, artifacts, output_file, "generation3")
    )

    assert not os.path.exists(old_generation)
    assert os.path.isdir(previous_generation)
    assert os.path.isdir(paths["generation_dir"])


def test_publish_pgo_bundle_keeps_previous_generation_tiling_immutable(
    ascendc_compile_module, tmpdir
):
    output_file = str(tmpdir.join("tiling.so"))
    artifacts = []
    for name in ("built_tiling.so", "built_runner", "built_kernel"):
        path = str(tmpdir.join(name))
        open(path, "wb").write(name.encode())
        artifacts.append(path)

    previous = ascendc_compile_module.publish_pgo_bundle(
        _make_pgo_bundle(ascendc_compile_module, artifacts, output_file, "generation1")
    )
    open(artifacts[0], "wb").write(b"new-tiling")
    current = ascendc_compile_module.publish_pgo_bundle(
        _make_pgo_bundle(ascendc_compile_module, artifacts, output_file, "generation2")
    )

    assert open(previous["tiling_so"], "rb").read() == b"built_tiling.so"
    assert open(current["tiling_so"], "rb").read() == b"new-tiling"
    assert open(output_file, "rb").read() == b"new-tiling"


def test_publish_pgo_bundle_failure_keeps_previous_tiling_and_removes_new_generation(
    ascendc_compile_module, tmpdir, monkeypatch
):
    output_file = str(tmpdir.join("tiling.so"))
    open(output_file, "wb").write(b"old-tiling")
    artifacts = []
    for name in ("built_tiling.so", "built_runner", "built_kernel"):
        path = str(tmpdir.join(name))
        open(path, "wb").write(name.encode())
        artifacts.append(path)
    real_replace = os.replace

    def fail_tiling_replace(source, target):
        if target == output_file:
            raise OSError("publish interrupted")
        real_replace(source, target)

    monkeypatch.setattr(ascendc_compile_module.os, "replace", fail_tiling_replace)

    with pytest.raises(ascendc_compile_module.CompileError, match="publish"):
        ascendc_compile_module.publish_pgo_bundle(
            _make_pgo_bundle(
                ascendc_compile_module, artifacts, output_file, "generation2"
            )
        )

    assert open(output_file, "rb").read() == b"old-tiling"
    assert not os.path.exists(output_file + ".pgo_v2/generation2")


def test_main_host_pgo_builds_bundle_and_skips_plain_copy(
    ascendc_compile_module, tmpdir
):
    events = []
    args = _make_host_pgo_args(tmpdir)

    def fake_link_tiling_so(*_):
        return str(tmpdir.join("built_tiling.so"))

    def fake_build_pgo_sidecars(*_):
        return str(tmpdir.join("built_runner")), str(tmpdir.join("built_kernel"))

    ascendc_compile_module.module.compile_host_objs = lambda *_: ["/tmp/build/host.o"]
    ascendc_compile_module.module.link_tiling_so = fake_link_tiling_so
    ascendc_compile_module.module.build_pgo_sidecars = fake_build_pgo_sidecars
    args.pgo_ld_preload = ""

    def fake_publish(bundle):
        events.append(
            (
                bundle.tiling_file,
                bundle.runner_file,
                bundle.kernel_file,
                bundle.output_file,
                bundle.generation,
                bundle.ld_preload,
            )
        )

    def fail_plain_copy(*_):
        pytest.fail("plain copy must not run")

    ascendc_compile_module.module.publish_pgo_bundle = fake_publish
    ascendc_compile_module.module.copy_so_to_output = fail_plain_copy

    ascendc_compile_module.main(args)

    assert len(events) == 1
    assert events[0][:4] == (
        str(tmpdir.join("built_tiling.so")),
        str(tmpdir.join("built_runner")),
        str(tmpdir.join("built_kernel")),
        str(tmpdir.join("tiling.so")),
    )
    assert len(events[0][4]) == 32
    assert args.pgo_generation == events[0][4]
    assert events[0][5] == ""


def test_main_all_pgo_builds_bundle_after_kernel_link(ascendc_compile_module, tmpdir):
    events = []
    args = _make_host_pgo_args(tmpdir)
    args.stage = "all"
    args.device_files = str(tmpdir.join("device.cpp"))

    class NoPchBatch:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return False

    def fake_build_pgo_sidecars(*_):
        return str(tmpdir.join("built_runner")), str(tmpdir.join("built_kernel"))

    def fake_publish(bundle):
        events.append(
            (
                bundle.tiling_file,
                bundle.runner_file,
                bundle.kernel_file,
                bundle.output_file,
            )
        )

    ascendc_compile_module.module.host_compile_batch = lambda *_: NoPchBatch()
    ascendc_compile_module.module.compile_host_objs = lambda *_: ["host.o"]
    ascendc_compile_module.module.build_kernel_target = lambda *_: str(
        tmpdir.join("built_kernel_so")
    )
    ascendc_compile_module.module.build_pgo_sidecars = fake_build_pgo_sidecars
    ascendc_compile_module.module.publish_pgo_bundle = fake_publish
    ascendc_compile_module.module.copy_so_to_output = lambda *_: pytest.fail(
        "plain copy must not run"
    )
    args.pgo_ld_preload = ""

    ascendc_compile_module.main(args)

    assert events == [
        (
            str(tmpdir.join("built_kernel_so")),
            str(tmpdir.join("built_runner")),
            str(tmpdir.join("built_kernel")),
            str(tmpdir.join("tiling.so")),
        )
    ]
    assert len(args.pgo_generation) == 32


def test_build_host_output_resolves_generation_before_host_compile(
    ascendc_compile_module, tmpdir
):
    args = _make_host_pgo_args(tmpdir)
    compiled_generations = []

    def fake_compile_host_objs(*_):
        # Capture the generation as seen at host compile time: the tiling source
        # embeds AUTOFUSE_PGO_GENERATION here, so it must already be resolved.
        compiled_generations.append(getattr(args, "pgo_generation", None))
        return ["/tmp/build/host.o"]

    def fake_link_tiling_so(*_):
        return str(tmpdir.join("built_tiling.so"))

    def fake_build_pgo_sidecars(*_):
        return str(tmpdir.join("built_runner")), str(tmpdir.join("built_kernel"))

    published_generations = []
    ascendc_compile_module.module.compile_host_objs = fake_compile_host_objs
    ascendc_compile_module.module.link_tiling_so = fake_link_tiling_so
    ascendc_compile_module.module.build_pgo_sidecars = fake_build_pgo_sidecars
    ascendc_compile_module.module.publish_pgo_bundle = lambda bundle: (
        published_generations.append(bundle.generation)
    )
    ascendc_compile_module.module.copy_so_to_output = lambda *_: pytest.fail(
        "plain copy must not run"
    )
    args.pgo_ld_preload = ""

    ascendc_compile_module.build_host_output(args)

    assert len(compiled_generations) == 1
    assert compiled_generations[0] is not None
    assert len(compiled_generations[0]) == 32
    # The published manifest must keep the same generation the .so was compiled
    # with, otherwise the runtime proxy rejects the bundle.
    assert published_generations == [compiled_generations[0]]


def test_main_host_pgo_failure_falls_back_to_plain_tiling(
    ascendc_compile_module, tmpdir
):
    original_dir = os.getcwd()
    copied = []
    args = _make_host_pgo_args(tmpdir)

    def fake_link_tiling_so(*_):
        return str(tmpdir.join("built_tiling.so"))

    def fail_build_pgo_sidecars(*_):
        raise ascendc_compile_module.CompileError("sidecar failed")

    def record_copy(so_file, compile_args, src_dir):
        copied.append((so_file, compile_args.output_file, src_dir))

    ascendc_compile_module.module.compile_host_objs = lambda *_: ["/tmp/build/host.o"]
    ascendc_compile_module.module.link_tiling_so = fake_link_tiling_so
    ascendc_compile_module.module.build_pgo_sidecars = fail_build_pgo_sidecars
    ascendc_compile_module.module.copy_so_to_output = record_copy

    ascendc_compile_module.main(args)

    assert copied == [
        (
            str(tmpdir.join("built_tiling.so")),
            str(tmpdir.join("tiling.so")),
            original_dir,
        )
    ]
    assert os.getcwd() == original_dir


def test_build_host_output_passes_pch_to_host_compile(ascendc_compile_module, tmpdir):
    args = _make_host_pgo_args(tmpdir)
    args.pgo_runner_file = None
    captured = {}

    def fake_compile_host_objs(compile_args, temp_dir, pch_path):
        captured["pch_path"] = pch_path
        return ["/tmp/build/host.o"]

    def fake_link_tiling_so(compile_args, tiling_obj_paths, temp_dir):
        return str(tmpdir.join("built_tiling.so"))

    ascendc_compile_module.module.compile_host_objs = fake_compile_host_objs
    ascendc_compile_module.module.link_tiling_so = fake_link_tiling_so

    result = ascendc_compile_module.build_host_output(args, "/tmp/cache/host.pch")

    assert result == str(tmpdir.join("built_tiling.so"))
    assert captured["pch_path"] == "/tmp/cache/host.pch"


def test_host_pgo_without_mspti_builds_acl_sidecars(ascendc_compile_module, tmpdir):
    published = []
    args = _make_host_pgo_args(tmpdir)

    def fake_compile_host_objs(*_):
        return [str(tmpdir.join("host.o"))]

    def fake_link_tiling_so(*_):
        return str(tmpdir.join("built_tiling.so"))

    def fake_build_pgo_sidecars(*_):
        return str(tmpdir.join("built_runner")), str(tmpdir.join("built_kernel"))

    ascendc_compile_module.module.compile_host_objs = fake_compile_host_objs
    ascendc_compile_module.module.link_tiling_so = fake_link_tiling_so
    ascendc_compile_module.module.build_pgo_sidecars = fake_build_pgo_sidecars
    ascendc_compile_module.module.publish_pgo_bundle = published.append

    result = ascendc_compile_module.build_host_output(args)

    assert result is None
    assert len(published) == 1
    assert published[0].runner_file.endswith("built_runner")
    assert published[0].kernel_file.endswith("built_kernel")
    assert not hasattr(args, "pgo_mspti_config")


def test_host_target_records_compile_and_link_stage(
    ascendc_compile_module, tmpdir, capsys
):
    host_dir = tmpdir.mkdir("host")
    host_file = host_dir.join("graph_tiling_func.cpp")
    host_file.write("host")
    args = type(
        "Args",
        (),
        {
            "host_files": str(host_file),
            "compile_options": "",
            "soc_version": "Ascend910B",
            "stage": "host",
            "graph_name": "graph",
            "output_file": str(tmpdir.join("host.so")),
            "temp_dir": str(tmpdir),
        },
    )()

    ascendc_compile_module.module.run_compile_command = _noop_run_compile_command
    ascendc_compile_module.build_host_output(args, str(tmpdir))

    labels = [item[0] for item in ascendc_compile_module.duration_records]
    assert ["InductorCompile", "host", "CompileHostObj", "graph"] in labels
    assert ["InductorCompile", "host", "LinkTilingSo", "graph"] in labels
    assert capsys.readouterr().out == ""


def test_kernel_target_records_device_compile_and_link_stage(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    device_file = device_dir.join("graph_op_kernel.cpp")
    device_file.write(
        'extern "C" __global__ __aicore__ void graph_kernel(GM_ADDR input0) {}\n'
    )
    output_file = tmpdir.join("kernel.so")
    args = type(
        "Args",
        (),
        {
            "device_files": str(device_file),
            "output_file": str(output_file),
            "soc_version": "Ascend910B",
            "stage": "device",
            "tiling_repr": None,
            "force_unknown": True,
            "graph_name": "graph",
        },
    )()

    ascendc_compile_module.module.run_compile_command = _noop_run_compile_command
    ascendc_compile_module.build_kernel_target(args, None, str(tmpdir))

    labels = [item[0] for item in ascendc_compile_module.duration_records]
    assert ["InductorCompile", "device", "CompileDeviceObj", "graph"] in labels
    assert ["InductorCompile", "device", "LinkKernelSo", "graph"] in labels


def test_compile_device_obj_includes_machine_asc_headers(
    ascendc_compile_module, tmpdir
):
    calls = []
    device_file = tmpdir.mkdir("device").join("graph_op_kernel.cpp")
    args = type(
        "Args",
        (),
        {
            "device_files": str(device_file),
            "soc_version": "Ascend910B",
            "stage": "device",
            "graph_name": "graph",
        },
    )()

    def fake_run_compile_command(cmd, stage_name):
        calls.append((cmd, stage_name))

    ascendc_compile_module.module.ASCEND_PATH = "/usr/local/Ascend/cann"
    ascendc_compile_module.module.machine = "aarch64"
    ascendc_compile_module.module.run_compile_command = fake_run_compile_command

    ascendc_compile_module.compile_device_obj(args, str(tmpdir))

    assert len(calls) == 1
    cmd, stage_name = calls[0]
    assert stage_name == "Device"
    assert "/usr/local/Ascend/cann/aarch64-linux/asc/include" in cmd


def test_try_static_shape_compile_records_stage_when_force_unknown(
    ascendc_compile_module, tmpdir
):
    args = type(
        "Args", (), {"force_unknown": True, "stage": "all", "graph_name": "graph"}
    )()

    assert (
        ascendc_compile_module.try_static_shape_compile(args, str(tmpdir), "kernel.so")
        is False
    )

    labels = [item[0] for item in ascendc_compile_module.duration_records]
    assert ["InductorCompile", "all", "PrepareStaticShapeRecompile", "graph"] in labels


def test_try_static_shape_compile_rejects_empty_const_tiling(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    device_dir.join("autofuse_tiling_data.h").write("old tiling")

    def is_static_shape():
        return True

    def gen_const_tiling_data(*args):
        return b""

    fake_lib = types.SimpleNamespace(
        AutofuseIsStaticShape=is_static_shape,
        GenConstTilingData=gen_const_tiling_data,
    )
    ascendc_compile_module.module.ctypes.CDLL = lambda path: fake_lib
    ascendc_compile_module.module.static_shape_kernel_proc = lambda *args: None
    ascendc_compile_module.module.init_torch_npu_for_const_tiling = lambda: None
    args = type(
        "Args",
        (),
        {
            "force_unknown": False,
            "stage": "all",
            "graph_name": "graph",
            "config_file": "config.txt",
        },
    )()

    with pytest.raises(
        ascendc_compile_module.CompileError, match="GenConstTilingData returned empty"
    ):
        ascendc_compile_module.try_static_shape_compile(args, str(tmpdir), "kernel.so")


def test_copy_so_to_output_records_stage(ascendc_compile_module, tmpdir):
    src_file = tmpdir.join("source.so")
    dst_file = tmpdir.mkdir("out").join("target.so")
    src_file.write("binary")
    src_directory = os.getcwd()
    args = type(
        "Args",
        (),
        {
            "output_file": str(dst_file),
            "stage": "host",
            "graph_name": "graph",
        },
    )()

    ascendc_compile_module.copy_so_to_output(str(src_file), args, src_directory)

    labels = [item[0] for item in ascendc_compile_module.duration_records]
    assert ["InductorCompile", "host", "CopyOutput", "graph"] in labels
    assert dst_file.read() == "binary"


def test_copy_so_to_output_copies_shared_cv_wrapper_next_to_output(
    ascendc_compile_module, tmpdir
):
    case = _make_copy_so_to_output_case(tmpdir)
    ascendc_compile_module.copy_so_to_output(
        str(case.src_file), case.args, case.src_directory
    )

    copied_wrapper = case.output_dir.join(
        "cv_tiling_wrapper_cache", "libautofuse_cv_tiling_wrapper_abc.so"
    )
    assert case.dst_file.read() == "kernel"
    assert copied_wrapper.read() == "wrapper"


def test_copy_so_to_output_keeps_existing_shared_cv_wrapper_in_output_dir(
    ascendc_compile_module, tmpdir
):
    case = _make_copy_so_to_output_case(tmpdir, wrapper_in_output_dir=True)
    ascendc_compile_module.copy_so_to_output(
        str(case.src_file), case.args, case.src_directory
    )

    assert case.dst_file.read() == "kernel"
    assert case.wrapper_file.read() == "wrapper"


def test_static_shape_kernel_proc_removes_tiling_data_from_launch(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR output0, '
        "GM_ADDR workspace, AutofuseTilingData t) {\n"
        "  use(t);\n"
        "}\n"
        "void init_static_kernel(void) {}\n"
        'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, void* output0, '
        "void* workspace, AutofuseTilingData* tiling_data)\n"
        "{\n"
        "  static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)output0, "
        "(uint8_t*)workspace, *tiling_data);\n"
        "  return 0;\n"
        "}\n"
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(
        args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}"
    )

    content = kernel_file.read()
    assert (
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR output0, GM_ADDR workspace) {'
        in content
    )
    assert (
        "constexpr AutofuseTilingData t = AutofuseTilingData{.block_dim = 8};"
        in content
    )
    assert (
        "static_kernel<<<blockDim, nullptr, stream>>>("
        "(uint8_t*)input0, (uint8_t*)output0, (uint8_t*)workspace);" in content
    )
    assert "*tiling_data" not in content


def test_static_shape_kernel_proc_removes_multiline_tiling_data_from_launch(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR output0, '
        "GM_ADDR workspace, AutofuseTilingData t) {\n"
        "  use(t);\n"
        "}\n" + _LAUNCH_FUNC
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(
        args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}"
    )

    content = kernel_file.read()
    assert "*tiling_data" not in content
    assert "(uint8_t*)workspace);" in content


def test_static_shape_kernel_proc_removes_multiline_kernel_param(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(\n'
        "    GM_ADDR input0,\n"
        "    GM_ADDR output0,\n"
        "    GM_ADDR workspace,\n"
        "    AutofuseTilingData t) {\n"
        "  use(t);\n"
        "}\n" + _LAUNCH_FUNC
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(
        args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}"
    )

    content = kernel_file.read()
    assert "GM_ADDR workspace) {" in content
    assert "AutofuseTilingData t) {" not in content
    assert (
        "constexpr AutofuseTilingData t = AutofuseTilingData{.block_dim = 8};"
        in content
    )
    assert "(uint8_t*)workspace);" in content


def test_static_shape_kernel_proc_removes_indented_kernel_param(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        '  extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR output0, '
        "GM_ADDR workspace, AutofuseTilingData t) {\n"
        "  use(t);\n"
        "}\n" + _LAUNCH_FUNC
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(
        args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}"
    )

    content = kernel_file.read()
    assert "AutofuseTilingData t) {" not in content
    assert (
        "constexpr AutofuseTilingData t = AutofuseTilingData{.block_dim = 8};"
        in content
    )
    assert "(uint8_t*)workspace);" in content


def test_static_shape_kernel_proc_keeps_launch_when_kernel_definition_not_rewritten(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        "__global__ __aicore__ __launch_bounds__(1) void static_kernel("
        "GM_ADDR input0, GM_ADDR workspace, AutofuseTilingData t) {\n"
        "  use(t);\n"
        "}\n"
        'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, '
        "void* workspace, AutofuseTilingData* tiling_data)\n"
        "{\n"
        "  static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, "
        "(uint8_t*)workspace, *tiling_data);\n"
        "  return 0;\n"
        "}\n"
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(
        args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}"
    )

    content = kernel_file.read()
    assert "AutofuseTilingData t) {" in content
    assert "constexpr AutofuseTilingData t =" not in content
    assert (
        "static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace, *tiling_data);"
        in content
    )


def test_static_shape_kernel_proc_only_rewrites_launch_for_rewritten_kernel(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR workspace, '
        "AutofuseTilingData t) {\n"
        "  use(t);\n"
        "}\n"
        'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, '
        "void* workspace, AutofuseTilingData* tiling_data)\n"
        "{\n"
        "  static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace, *tiling_data);\n"
        "  other_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace, *tiling_data);\n"
        "  return 0;\n"
        "}\n"
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(
        args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}"
    )

    content = kernel_file.read()
    assert (
        "static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace);"
        in content
    )
    assert (
        "other_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace, *tiling_data);"
        in content
    )


def test_static_shape_kernel_proc_keeps_dynamic_inductor_tiling_when_tiling_repr_is_none(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR workspace, '
        "AutofuseTilingData t) {\n"
        "#ifdef INDUCTOR_CONST_TILING_DATA\n"
        "  const AutofuseTilingData t;\n"
        "#else\n"
        "  use(t);\n"
        "#endif\n"
        "}\n"
        'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, '
        "void* workspace, AutofuseTilingData* tiling_data)\n"
        "{\n"
        "  static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, "
        "(uint8_t*)workspace, *tiling_data);\n"
        "  return 0;\n"
        "}\n"
    )
    args = type("Args", (), {"device_files": str(kernel_file), "stage": "device"})()

    ascendc_compile_module.static_shape_kernel_proc(args, str(tmpdir), None)

    content = kernel_file.read()
    assert "INDUCTOR_CONST_TILING_DATA" not in content
    assert "AutofuseTilingData t) {" in content
    assert "const AutofuseTilingData t;" not in content
    assert (
        "static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, "
        "(uint8_t*)workspace, *tiling_data);" in content
    )


def _make_device_stage_args():
    return type("Args", (), {"stage": "device"})()


def _make_inductor_const_tiling_branch_lines():
    return [
        "#ifdef INDUCTOR_CONST_TILING_DATA\n",
        "const CVAutofuseTilingData& gm_tiling_data = kConstTilingData;\n",
        "#else\n",
        "const TILING_DATA_T& gm_tiling_data = t;\n",
        "#endif\n",
    ]


def test_expand_inductor_const_tiling_data_selects_dynamic_branch_without_tiling_repr(
    ascendc_compile_module,
):
    args = _make_device_stage_args()
    lines = _make_inductor_const_tiling_branch_lines()

    result = ascendc_compile_module.expand_inductor_const_tiling_data(args, lines, None)

    assert result == ["const TILING_DATA_T& gm_tiling_data = t;\n"]


def test_expand_inductor_const_tiling_data_selects_const_branch_with_tiling_repr(
    ascendc_compile_module,
):
    args = _make_device_stage_args()
    lines = _make_inductor_const_tiling_branch_lines()

    result = ascendc_compile_module.expand_inductor_const_tiling_data(
        args, lines, "AutofuseTilingData{}"
    )

    assert result == [
        "const CVAutofuseTilingData& gm_tiling_data = kConstTilingData;\n"
    ]


def _make_compile_args(host_files=None):
    return type(
        "Args",
        (),
        {
            "host_files": host_files,
            "device_files": "/tmp/build/device/kernel.cpp",
            "soc_version": "Ascend910B",
            "compile_options": "-Werror",
            "output_file": "/tmp/build/kernel.so",
            "force_unknown": True,
            "stage": "all",
            "tiling_repr": None,
        },
    )()


def test_generate_pch_source_uses_only_stable_common_headers(
    ascendc_compile_module, tmpdir
):
    host_dir = tmpdir.mkdir("host")
    host_dir.join("autofuse_tiling_func_solver.h").write("graph-specific header")

    pch_source = ascendc_compile_module.generate_pch_source(str(host_dir))
    contents = open(pch_source, encoding="utf-8").read()

    assert "autofuse_tiling_func_solver.h" not in contents
    assert "tiling/platform/platform_ascendc.h" not in contents


def test_single_host_file_creates_pch(ascendc_compile_module, monkeypatch, tmpdir):
    ascendc_compile_module.module.PCH_CACHE_ROOT = str(tmpdir.mkdir("pch-cache"))
    args = _make_compile_args("/tmp/build/host/graph_tiling_func.cpp")
    paths = ascendc_compile_module.get_host_pch_paths(args)
    monkeypatch.setattr(
        ascendc_compile_module.module,
        "prepare_host_pch_with_cleanup",
        lambda unused_args, unused_paths: paths,
    )

    with ascendc_compile_module.host_compile_batch(args) as pch_path:
        assert pch_path == paths[2]


def test_host_compile_batch_falls_back_when_pch_build_fails(
    ascendc_compile_module, monkeypatch, tmpdir
):
    ascendc_compile_module.module.PCH_CACHE_ROOT = str(tmpdir.mkdir("pch-cache"))
    args = _make_compile_args(
        [
            "/tmp/build/host/graph_tiling_func_a.cpp",
            "/tmp/build/host/graph_tiling_func_b.cpp",
        ]
    )

    monkeypatch.setattr(
        ascendc_compile_module.module,
        "generate_pch_source",
        lambda *unused_args: (_ for _ in ()).throw(OSError("no write access")),
    )

    with ascendc_compile_module.host_compile_batch(args) as pch_path:
        assert pch_path is None


def test_host_compile_batch_falls_back_when_pch_cache_root_is_file(
    ascendc_compile_module, tmpdir
):
    cache_root = tmpdir.join("pch-cache")
    cache_root.write("not a directory")
    ascendc_compile_module.module.PCH_CACHE_ROOT = str(cache_root)
    args = _make_compile_args(
        [
            "/tmp/build/host/graph_tiling_func_a.cpp",
            "/tmp/build/host/graph_tiling_func_b.cpp",
        ]
    )

    with ascendc_compile_module.host_compile_batch(args) as pch_path:
        assert pch_path is None


def test_build_host_compile_cmd_uses_bisheng_without_cmake(ascendc_compile_module):
    ascendc_compile_module.module.ASCEND_PATH = "/usr/local/Ascend/cann"
    ascendc_compile_module.module.machine = "x86_64"
    args = _make_compile_args("/tmp/build/host/graph_tiling_func.cpp")

    cmd = ascendc_compile_module.build_host_compile_cmd(
        args,
        "/tmp/build",
        "/tmp/build/host/graph_tiling_func.cpp",
        "/tmp/build/host/graph_tiling_func.cpp.o",
    )

    assert cmd[0] == "/usr/local/Ascend/cann/tools/bisheng_compiler/bin/bisheng"
    assert "-c" in cmd
    assert "-x" in cmd
    assert "c++" in cmd
    assert "-std=c++17" in cmd
    assert "/tmp/build/host/graph_tiling_func.cpp" in cmd
    assert "/tmp/build/host/graph_tiling_func.cpp.o" in cmd
    assert "cmake" not in cmd
    assert "make" not in cmd


def test_compile_host_objs_compiles_single_file_serially(
    ascendc_compile_module, monkeypatch
):
    compiled = []
    host_files = ["/tmp/build/host/graph_tiling_func.cpp"]

    monkeypatch.setattr(
        ascendc_compile_module.module,
        "prepare_shared_cv_wrapper",
        lambda args, temp_dir, files: host_files,
    )
    monkeypatch.setattr(
        ascendc_compile_module.module,
        "compile_host_obj_file",
        lambda args, temp_dir, source_file, pch_state: (
            compiled.append(source_file) or source_file + ".o"
        ),
    )

    args = _make_compile_args(host_files[0])
    result = ascendc_compile_module.compile_host_objs(args, "/tmp/build")
    assert compiled == host_files
    assert result == [host_files[0] + ".o"]


def test_build_pch_command_uses_cpp17(ascendc_compile_module):
    args = _make_compile_args("/tmp/build/host/graph_tiling_func.cpp")

    cmd = ascendc_compile_module.build_pch_command(
        args,
        "/tmp/cache/autofuse_tiling_pch.h",
        "/tmp/cache/autofuse_tiling_pch.h.gch",
    )

    assert "-std=c++17" in cmd
    assert "c++-header" in cmd


def test_compile_diagnostics_write_trace_to_default_directory(
    ascendc_compile_module, monkeypatch, tmpdir
):
    trace_dir = tmpdir.mkdir("trace")
    ascendc_compile_module.module.COMPILE_TRACE_ROOT = str(trace_dir)
    monkeypatch.setenv("AUTOFUSE_DFX_FLAGS", "codegen_compile_debug=true")
    log_messages = []
    monkeypatch.setattr(
        ascendc_compile_module.module.logger,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )

    flags = ascendc_compile_module.get_compile_diagnostic_flags("/tmp/host.o")

    assert "-ftime-report=per-pass" in flags
    trace_flag = next(flag for flag in flags if flag.startswith("-ftime-trace="))
    assert trace_flag.startswith(f"-ftime-trace={trace_dir}/host.o.")
    assert trace_flag.endswith(".json")
    assert f"[CompileTrace] {trace_flag.removeprefix('-ftime-trace=')}" in log_messages


def test_compile_diagnostics_use_unique_trace_files(
    ascendc_compile_module, monkeypatch, tmpdir
):
    trace_dir = tmpdir.mkdir("trace")
    ascendc_compile_module.module.COMPILE_TRACE_ROOT = str(trace_dir)
    monkeypatch.setenv("AUTOFUSE_DFX_FLAGS", "codegen_compile_debug=true")

    first = ascendc_compile_module.get_compile_diagnostic_flags("/tmp/a/graph.o")
    second = ascendc_compile_module.get_compile_diagnostic_flags("/tmp/b/graph.o")

    assert first != second


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_compile_host_obj_falls_back_without_pch(
    ascendc_compile_module, monkeypatch, cleanup_fails
):
    args = _make_compile_args("/tmp/build/host/graph_tiling_func.cpp")
    calls = []

    def fake_run_compile_command(cmd, stage_name):
        calls.append(cmd)
        if "-include-pch" in cmd:
            raise ascendc_compile_module.CompileError("invalid PCH")

    if cleanup_fails:
        monkeypatch.setattr(
            ascendc_compile_module.module,
            "invalidate_host_pch",
            lambda *unused_args: (_ for _ in ()).throw(OSError("cache unavailable")),
        )
    ascendc_compile_module.module.run_compile_command = fake_run_compile_command
    pch_state = {"path": "/tmp/cache/host.pch", "lock": ascendc_compile_module.Lock()}

    result = ascendc_compile_module.compile_host_obj_file(
        args,
        "/tmp/build",
        "/tmp/build/host/graph_tiling_func.cpp",
        pch_state,
    )

    assert result == "/tmp/build/host/graph_tiling_func.cpp.o"
    assert pch_state["path"] is None
    assert len(calls) == 2
    assert "-include-pch" in calls[0]
    assert "-include-pch" not in calls[1]


def test_compile_host_obj_removes_rejected_cached_pch(ascendc_compile_module, tmpdir):
    args = _make_compile_args("/tmp/build/host/graph_tiling_func.cpp")
    ascendc_compile_module.module.PCH_CACHE_ROOT = str(tmpdir.mkdir("pch-cache"))
    _, _, pch_path, _ = ascendc_compile_module.get_host_pch_paths(args)
    os.makedirs(os.path.dirname(pch_path))
    open(pch_path, "w", encoding="utf-8").write("invalid PCH")

    def fake_run_compile_command(cmd, stage_name):
        if "-include-pch" in cmd:
            raise ascendc_compile_module.CompileError("invalid PCH")

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command
    pch_state = {"path": pch_path, "lock": ascendc_compile_module.Lock()}

    ascendc_compile_module.compile_host_obj_file(
        args,
        "/tmp/build",
        "/tmp/build/host/graph_tiling_func.cpp",
        pch_state,
    )

    assert pch_state["path"] is None
    assert not os.path.exists(pch_path)


def test_build_host_compile_cmd_includes_pkg_inc_roots(ascendc_compile_module):
    ascendc_compile_module.module.ASCEND_PATH = "/usr/local/Ascend/cann"
    ascendc_compile_module.module.machine = "x86_64"

    include_options = ascendc_compile_module.build_host_include_options("/tmp/build")

    assert "/usr/local/Ascend/cann/pkg_inc" in include_options
    assert "/usr/local/Ascend/cann/x86_64-linux/pkg_inc" in include_options


def test_build_host_include_options_prefers_machine_pkg_inc_base(
    ascendc_compile_module,
):
    ascendc_compile_module.module.ASCEND_PATH = "/usr/local/Ascend/cann"
    ascendc_compile_module.module.machine = "x86_64"

    include_options = ascendc_compile_module.build_host_include_options("/tmp/build")

    machine_base = include_options.index(
        "/usr/local/Ascend/cann/x86_64-linux/pkg_inc/base"
    )
    generic_base = include_options.index("/usr/local/Ascend/cann/pkg_inc/base")
    assert machine_base < generic_base


def _assert_compile_host_objs_skips_shared_cv_wrapper_source(
    ascendc_compile_module, tmpdir, monkeypatch, source_case
):
    graph_file_name, wrapper_file_name, wrapper_content = source_case
    host_dir = tmpdir.mkdir("host")
    graph_file = host_dir.join(graph_file_name)
    wrapper_file = host_dir.join(wrapper_file_name)
    graph_file.write("CVAutofuseTilingData graph tiling")
    wrapper_file.write(wrapper_content)
    args = _make_compile_args([str(graph_file), str(wrapper_file)])
    compiled_sources = []

    def fake_compile_host_obj_file(compile_args, temp_dir, source_file, pch_state=None):
        compiled_sources.append(source_file)
        return source_file + ".o"

    def fake_ensure_shared_cv_wrapper_so(compile_args, temp_dir, source_file):
        assert source_file == str(wrapper_file)
        return "/tmp/run/cv_tiling_wrapper_cache/libautofuse_cv_tiling_wrapper.so"

    ascendc_compile_module.module.compile_host_obj_file = fake_compile_host_obj_file
    ascendc_compile_module.module.ensure_shared_cv_wrapper_so = (
        fake_ensure_shared_cv_wrapper_so
    )
    monkeypatch.setattr(ascendc_compile_module.os, "cpu_count", lambda: 2)

    result = ascendc_compile_module.compile_host_objs(args, str(tmpdir))

    assert result == [str(graph_file) + ".o"]
    assert compiled_sources == [str(graph_file)]
    assert (
        args.shared_cv_wrapper_so
        == "/tmp/run/cv_tiling_wrapper_cache/libautofuse_cv_tiling_wrapper.so"
    )


def test_build_host_compile_cmd_excludes_mspti_include(ascendc_compile_module):
    args = _make_compile_args("/tmp/build/host/graph_tiling_func_PgoRunner.cpp")
    args.pgo_mspti_dir = "/usr/local/Ascend/cann/tools/mspti"
    args.pgo_generation = "generation1"
    args.compile_options = "-Werror -D_GLIBCXX_USE_CXX11_ABI=1"

    cmd = ascendc_compile_module.build_host_compile_cmd(
        args,
        "/tmp/build",
        "/tmp/build/host/graph_tiling_func_PgoRunner.cpp",
        "/tmp/build/host/graph_tiling_func_PgoRunner.cpp.o",
    )

    assert not any("mspti" in str(option).lower() for option in cmd)
    assert "-D_GLIBCXX_USE_CXX11_ABI=1" in cmd
    assert 'AUTOFUSE_PGO_GENERATION="generation1"' in cmd


def test_compile_host_objs_keeps_single_file_compatible(ascendc_compile_module):
    calls = []
    args = _make_compile_args("/tmp/build/host/graph_tiling_func.cpp")

    def fake_run_compile_command(cmd, stage_name):
        calls.append((cmd, stage_name))

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command

    result = ascendc_compile_module.compile_host_objs(args, "/tmp/build")

    assert result == ["/tmp/build/host/graph_tiling_func.cpp.o"]
    assert len(calls) == 1
    assert calls[0][1] == "Host"


def test_compile_host_objs_compiles_multiple_files(ascendc_compile_module, monkeypatch):
    calls = []
    args = _make_compile_args(
        [
            "/tmp/build/host/graph_tiling_func_a.cpp",
            "/tmp/build/host/graph_tiling_func_b.cpp",
        ]
    )

    def fake_run_compile_command(cmd, stage_name):
        calls.append((cmd, stage_name))

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command
    monkeypatch.setattr(ascendc_compile_module.os, "cpu_count", lambda: 2)

    result = ascendc_compile_module.compile_host_objs(args, "/tmp/build")

    assert result == [
        "/tmp/build/host/graph_tiling_func_a.cpp.o",
        "/tmp/build/host/graph_tiling_func_b.cpp.o",
    ]
    assert [call[1] for call in calls] == ["Host", "Host"]
    compile_cmds = [call[0] for call in calls]
    assert any("/tmp/build/host/graph_tiling_func_a.cpp" in cmd for cmd in compile_cmds)
    assert any("/tmp/build/host/graph_tiling_func_b.cpp" in cmd for cmd in compile_cmds)
    assert all("cmake" not in cmd and "make" not in cmd for cmd in compile_cmds)


def test_get_shared_cv_wrapper_cache_dir_prefers_run_dir_env(
    ascendc_compile_module, tmpdir, monkeypatch
):
    run_dir = tmpdir.mkdir("run")
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    args = _make_compile_args()
    args.output_file = None

    result = ascendc_compile_module.get_shared_cv_wrapper_cache_dir(args, "")

    assert result == os.path.join(str(run_dir), "cv_tiling_wrapper_cache")


def test_get_shared_cv_wrapper_cache_dir_uses_inductor_cache_without_run_dir(
    ascendc_compile_module, tmpdir, monkeypatch
):
    cache_dir = tmpdir.mkdir("inductor_cache")
    monkeypatch.delenv("RUN_DIR", raising=False)
    monkeypatch.setenv("TORCHINDUCTOR_NPU_EXT_CACHE_DIR", str(cache_dir))
    args = _make_compile_args()
    args.output_file = None

    result = ascendc_compile_module.get_shared_cv_wrapper_cache_dir(args, "")

    assert result == os.path.join(str(cache_dir), "cv_tiling_wrapper_cache")


def test_get_shared_cv_wrapper_cache_dir_prefers_output_file_dir(
    ascendc_compile_module, tmpdir, monkeypatch
):
    output_dir = tmpdir.mkdir("kernel_meta")
    output_file = output_dir.join("kernel.so")
    run_dir = tmpdir.mkdir("run")
    cache_dir = tmpdir.mkdir("inductor_cache")
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    monkeypatch.setenv("TORCHINDUCTOR_NPU_EXT_CACHE_DIR", str(cache_dir))
    args = _make_compile_args()
    args.output_file = str(output_file)

    result = ascendc_compile_module.get_shared_cv_wrapper_cache_dir(args, "")

    assert result == os.path.join(str(output_dir), "cv_tiling_wrapper_cache")


def test_get_shared_cv_wrapper_cache_dir_prefers_temp_dir(
    ascendc_compile_module, tmpdir, monkeypatch
):
    temp_dir = tmpdir.mkdir("temp")
    output_dir = tmpdir.mkdir("kernel_meta")
    output_file = output_dir.join("kernel.so")
    run_dir = tmpdir.mkdir("run")
    cache_dir = tmpdir.mkdir("inductor_cache")
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    monkeypatch.setenv("TORCHINDUCTOR_NPU_EXT_CACHE_DIR", str(cache_dir))
    args = _make_compile_args()
    args.output_file = str(output_file)

    result = ascendc_compile_module.get_shared_cv_wrapper_cache_dir(args, str(temp_dir))

    assert result == os.path.join(str(temp_dir), "cv_tiling_wrapper_cache")


def test_compile_host_objs_skips_shared_cv_wrapper_source(
    ascendc_compile_module, tmpdir, monkeypatch
):
    _assert_compile_host_objs_skips_shared_cv_wrapper_source(
        ascendc_compile_module,
        tmpdir,
        monkeypatch,
        (
            "graph_tiling_func.cpp",
            "cube_kernel_tiling_wrapper.cpp",
            "CVAutofuseTilingData wrapper tiling",
        ),
    )


def test_compile_host_objs_skips_split_shared_cv_wrapper_source(
    ascendc_compile_module, tmpdir, monkeypatch
):
    _assert_compile_host_objs_skips_shared_cv_wrapper_source(
        ascendc_compile_module,
        tmpdir,
        monkeypatch,
        (
            "autofused_tiling_func_tiling_def_and_tiling_const.cpp",
            "autofused_tiling_func_BCubeKernelTilingWrapperCpp.cpp",
            "AutofuseDoCubeMatMulTiling wrapper tiling",
        ),
    )


def test_ensure_shared_cv_wrapper_so_reuses_existing_so(
    ascendc_compile_module, tmpdir, monkeypatch
):
    run_dir = tmpdir.mkdir("run")
    host_dir = tmpdir.mkdir("host")
    wrapper_file = host_dir.join("cube_kernel_tiling_wrapper.cpp")
    wrapper_file.write("CVAutofuseTilingData wrapper tiling")
    args = _make_compile_args([str(wrapper_file)])
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    so_path = ascendc_compile_module.get_shared_cv_wrapper_so_path(
        args, str(tmpdir), str(wrapper_file)
    )
    os.makedirs(os.path.dirname(so_path), exist_ok=True)
    with open(so_path, "w") as f:
        f.write("cached")

    def fail_compile(*_args, **_kwargs):
        pytest.fail("cached wrapper so should not be recompiled")

    ascendc_compile_module.module.compile_host_obj_file = fail_compile
    ascendc_compile_module.module.link_shared = fail_compile

    result = ascendc_compile_module.ensure_shared_cv_wrapper_so(
        args, str(tmpdir), str(wrapper_file)
    )

    assert result == so_path


def test_ensure_shared_cv_wrapper_so_serializes_concurrent_first_compile(
    ascendc_compile_module, tmpdir, monkeypatch
):
    run_dir = tmpdir.mkdir("run")
    host_dir = tmpdir.mkdir("host")
    wrapper_file = host_dir.join("cube_kernel_tiling_wrapper.cpp")
    wrapper_file.write("CVAutofuseTilingData wrapper tiling")
    args = _make_compile_args([str(wrapper_file)])
    args.output_file = str(tmpdir.mkdir("kernel_meta").join("kernel.so"))
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    compile_calls = []

    def fake_compile_host_obj_file(compile_args, temp_dir, source_file):
        compile_calls.append(source_file)
        time.sleep(0.05)
        return source_file + ".o"

    def fake_link_shared(
        target_file, obj_files, link_libraries=None, extra_link_options=None
    ):
        with open(target_file, "w") as f:
            f.write("linked")
        return target_file

    ascendc_compile_module.module.compile_host_obj_file = fake_compile_host_obj_file
    ascendc_compile_module.module.link_shared = fake_link_shared

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                ascendc_compile_module.ensure_shared_cv_wrapper_so,
                args,
                str(tmpdir),
                str(wrapper_file),
            )
            for _ in range(2)
        ]
        results = [future.result() for future in futures]

    assert results[0] == results[1]
    assert os.path.exists(results[0])
    assert compile_calls == [str(wrapper_file)]


def test_ensure_shared_cv_wrapper_so_sets_soname(
    ascendc_compile_module, tmpdir, monkeypatch
):
    output_dir = tmpdir.mkdir("kernel_meta")
    host_dir = tmpdir.mkdir("host")
    wrapper_file = host_dir.join("cube_kernel_tiling_wrapper.cpp")
    wrapper_file.write("CVAutofuseTilingData wrapper tiling")
    args = _make_compile_args([str(wrapper_file)])
    args.output_file = str(output_dir.join("kernel.so"))
    monkeypatch.delenv("RUN_DIR", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_NPU_EXT_CACHE_DIR", raising=False)
    captured = {}

    def fake_compile_host_obj_file(compile_args, temp_dir, source_file):
        return source_file + ".o"

    def fake_link_shared(
        target_file, obj_files, link_libraries=None, extra_link_options=None
    ):
        captured["target_file"] = target_file
        captured["extra_link_options"] = extra_link_options
        with open(target_file, "w") as f:
            f.write("linked")
        return target_file

    ascendc_compile_module.module.compile_host_obj_file = fake_compile_host_obj_file
    ascendc_compile_module.module.link_shared = fake_link_shared

    so_path = ascendc_compile_module.ensure_shared_cv_wrapper_so(
        args, str(tmpdir), str(wrapper_file)
    )

    assert captured["extra_link_options"] == [
        f"-Wl,-soname,{os.path.basename(so_path)}"
    ]


def test_clean_before_modify_keeps_shared_cv_wrapper_cache(
    ascendc_compile_module, tmpdir
):
    tmpdir.mkdir("host")
    tmpdir.mkdir("device")
    cache_dir = tmpdir.mkdir("cv_tiling_wrapper_cache")
    cache_file = cache_dir.join("libautofuse_cv_tiling_wrapper.so")
    cache_file.write("cached")
    tmpdir.mkdir("stale")

    ascendc_compile_module.clean_before_modify(str(tmpdir))

    assert os.path.exists(str(cache_file))
    assert not os.path.exists(os.path.join(str(tmpdir), "stale"))


def test_get_host_compile_worker_count_uses_32_worker_limit(
    ascendc_compile_module, monkeypatch
):
    monkeypatch.setattr(ascendc_compile_module.os, "cpu_count", lambda: 64)
    assert ascendc_compile_module.get_host_compile_worker_count(32) == 32
    assert ascendc_compile_module.get_host_compile_worker_count(12) == 12
    monkeypatch.setattr(ascendc_compile_module.os, "cpu_count", lambda: 16)
    assert ascendc_compile_module.get_host_compile_worker_count(32) == 16
    monkeypatch.setattr(ascendc_compile_module.os, "cpu_count", lambda: None)
    assert ascendc_compile_module.get_host_compile_worker_count(3) == 1


def test_compile_host_objs_reports_failed_source(ascendc_compile_module, monkeypatch):
    args = _make_compile_args(
        [
            "/tmp/build/host/graph_tiling_func_a.cpp",
            "/tmp/build/host/graph_tiling_func_b.cpp",
        ]
    )

    def fake_run_compile_command(cmd, stage_name):
        if "/tmp/build/host/graph_tiling_func_b.cpp" in cmd:
            raise ascendc_compile_module.CompileError("stderr: fail")

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command
    monkeypatch.setattr(ascendc_compile_module.os, "cpu_count", lambda: 1)

    with pytest.raises(ascendc_compile_module.CompileError) as exc_info:
        ascendc_compile_module.compile_host_objs(args, "/tmp/build")

    assert "/tmp/build/host/graph_tiling_func_b.cpp" in str(exc_info.value)
    assert "stderr: fail" in str(exc_info.value)


def test_compile_host_obj_rejects_multiple_sources_without_compile(
    ascendc_compile_module,
):
    args = _make_compile_args(
        [
            "/tmp/build/host/graph_tiling_func_a.cpp",
            "/tmp/build/host/graph_tiling_func_b.cpp",
        ]
    )

    def fake_run_compile_command(cmd, stage_name):
        pytest.fail("should not compile")

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command

    with pytest.raises(ascendc_compile_module.CompileError) as exc_info:
        ascendc_compile_module.compile_host_obj(args, "/tmp/build")

    assert "expects exactly one host source" in str(exc_info.value)


def _capture_link_kernel_so(ascendc_compile_module, args, host_obj_path, temp_dir):
    captured = {}

    def fake_link_shared(
        target_file, obj_files, link_libraries=None, extra_link_options=None
    ):
        captured["obj_files"] = obj_files
        captured["link_libraries"] = link_libraries
        captured["extra_link_options"] = extra_link_options
        return target_file

    ascendc_compile_module.module.link_shared = fake_link_shared
    ascendc_compile_module.link_kernel_so(
        args, host_obj_path, temp_dir, "/tmp/build/device/kernel.cpp.o"
    )
    return captured


def test_link_kernel_so_links_all_host_objects(ascendc_compile_module):
    captured = {}
    args = _make_compile_args()

    def fake_link_shared(
        target_file, obj_files, link_libraries=None, extra_link_options=None
    ):
        captured["target_file"] = target_file
        captured["obj_files"] = obj_files
        captured["link_libraries"] = link_libraries
        captured["extra_link_options"] = extra_link_options
        return target_file

    ascendc_compile_module.module.link_shared = fake_link_shared

    result = ascendc_compile_module.link_kernel_so(
        args, ["a.o", "b.o"], "/tmp/build", "/tmp/build/device/kernel.cpp.o"
    )

    assert result == "/tmp/build/kernel.so"
    assert captured["obj_files"] == ["a.o", "b.o", "/tmp/build/device/kernel.cpp.o"]
    assert captured["link_libraries"] == ascendc_compile_module.HOST_LINK_LIBRARIES


def test_link_kernel_so_links_shared_cv_wrapper_so_for_cv_compile(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    device_file = device_dir.join("kernel.cpp")
    device_file.write("CVAutofuseTilingData device tiling")
    args = _make_compile_args()
    args.device_files = str(device_file)
    args.shared_cv_wrapper_so = (
        "/tmp/run/cv_tiling_wrapper_cache/libautofuse_cv_tiling_wrapper.so"
    )
    captured = _capture_link_kernel_so(
        ascendc_compile_module, args, ["graph.o"], "/tmp/build"
    )

    assert captured["obj_files"] == [
        "graph.o",
        "/tmp/build/device/kernel.cpp.o",
        "/tmp/run/cv_tiling_wrapper_cache/libautofuse_cv_tiling_wrapper.so",
    ]
    assert captured["link_libraries"] == ascendc_compile_module.CV_HOST_LINK_LIBRARIES
    assert captured["extra_link_options"] == [
        ascendc_compile_module.CV_WRAPPER_RPATH_OPTION
    ]


def test_link_kernel_so_ignores_shared_cv_wrapper_so_for_non_cv_compile(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    device_file = device_dir.join("kernel.cpp")
    device_file.write("regular device tiling")
    args = _make_compile_args()
    args.device_files = str(device_file)
    args.shared_cv_wrapper_so = (
        "/tmp/run/cv_tiling_wrapper_cache/libautofuse_cv_tiling_wrapper.so"
    )
    captured = _capture_link_kernel_so(
        ascendc_compile_module, args, ["graph.o"], str(tmpdir)
    )

    assert captured["obj_files"] == ["graph.o", "/tmp/build/device/kernel.cpp.o"]
    assert captured["link_libraries"] == ascendc_compile_module.HOST_LINK_LIBRARIES


def test_build_host_output_links_multiple_host_objects(ascendc_compile_module):
    captured = {}
    args = _make_compile_args(
        [
            "/tmp/build/host/graph_tiling_func_a.cpp",
            "/tmp/build/host/graph_tiling_func_b.cpp",
        ]
    )

    def fake_compile_host_objs(compile_args, temp_dir):
        return ["a.o", "b.o"]

    ascendc_compile_module.module.compile_host_objs = fake_compile_host_objs

    def fake_link_shared(
        target_file, obj_files, link_libraries=None, extra_link_options=None
    ):
        captured["target_file"] = target_file
        captured["obj_files"] = obj_files
        captured["link_libraries"] = link_libraries
        captured["extra_link_options"] = extra_link_options
        return target_file

    ascendc_compile_module.module.link_shared = fake_link_shared

    result = ascendc_compile_module.link_tiling_so(args, ["a.o", "b.o"], "/tmp/build")

    assert result == "/tmp/build/kernel.so"
    assert captured["target_file"] == "/tmp/build/kernel.so"
    assert captured["obj_files"] == ["a.o", "b.o"]
    assert captured["link_libraries"] == ascendc_compile_module.HOST_LINK_LIBRARIES
    assert "acl_rt" in captured["link_libraries"]


def _capture_build_host_output_link(ascendc_compile_module, args, temp_dir):
    captured = {}

    def fake_compile_host_objs(compile_args, temp_dir):
        return ["graph.o"]

    ascendc_compile_module.module.compile_host_objs = fake_compile_host_objs

    def fake_link_shared(
        target_file, obj_files, link_libraries=None, extra_link_options=None
    ):
        captured["obj_files"] = obj_files
        captured["link_libraries"] = link_libraries
        captured["extra_link_options"] = extra_link_options
        return target_file

    ascendc_compile_module.module.link_shared = fake_link_shared
    args.temp_dir = temp_dir
    ascendc_compile_module.build_host_output(args)
    return captured


def test_build_host_output_links_shared_cv_wrapper_so_for_cv_compile(
    ascendc_compile_module, tmpdir
):
    host_dir = tmpdir.mkdir("host")
    host_file = host_dir.join("graph_tiling_func.cpp")
    host_file.write("CVAutofuseTilingData graph tiling")
    args = _make_compile_args([str(host_file)])
    args.shared_cv_wrapper_so = (
        "/tmp/run/cv_tiling_wrapper_cache/libautofuse_cv_tiling_wrapper.so"
    )
    captured = _capture_build_host_output_link(
        ascendc_compile_module, args, "/tmp/build"
    )

    assert captured["obj_files"] == [
        "graph.o",
        "/tmp/run/cv_tiling_wrapper_cache/libautofuse_cv_tiling_wrapper.so",
    ]
    assert captured["link_libraries"] == ascendc_compile_module.CV_HOST_LINK_LIBRARIES
    assert captured["extra_link_options"] == [
        ascendc_compile_module.CV_WRAPPER_RPATH_OPTION
    ]


def test_build_host_output_ignores_shared_cv_wrapper_so_for_non_cv_compile(
    ascendc_compile_module, tmpdir
):
    host_dir = tmpdir.mkdir("host")
    host_file = host_dir.join("graph_tiling_func.cpp")
    host_file.write("regular graph tiling")
    args = _make_compile_args([str(host_file)])
    args.shared_cv_wrapper_so = (
        "/tmp/run/cv_tiling_wrapper_cache/libautofuse_cv_tiling_wrapper.so"
    )
    captured = _capture_build_host_output_link(
        ascendc_compile_module, args, str(tmpdir)
    )

    assert captured["obj_files"] == ["graph.o"]
    assert captured["link_libraries"] == ascendc_compile_module.HOST_LINK_LIBRARIES


def test_build_host_output_adds_acl_runtime_for_pgo_proxy(ascendc_compile_module):
    args = _make_compile_args(["/tmp/build/host/graph_tiling_func.cpp"])
    args.pgo_runner_file = "/tmp/build/host/graph_tiling_func_PgoRunner.cpp"
    captured = _capture_build_host_output_link(
        ascendc_compile_module, args, "/tmp/build"
    )

    assert captured["link_libraries"] == ascendc_compile_module.HOST_LINK_LIBRARIES + [
        "ascendcl",
        "runtime",
    ]


def test_build_kernel_target_reuses_host_objects_for_static_recompile(
    ascendc_compile_module,
):
    calls = []
    args = _make_compile_args()
    args.force_unknown = False

    def fake_try_static_shape_compile(compile_args, temp_dir, so_path):
        return True

    ascendc_compile_module.module.try_static_shape_compile = (
        fake_try_static_shape_compile
    )

    def fake_compile_device_obj(compile_args, temp_dir):
        return f"/tmp/build/device/kernel_{len(calls) + 1}.o"

    def fake_link_kernel_so(compile_args, tiling_obj_paths, temp_dir, kernel_obj_path):
        calls.append((list(tiling_obj_paths), kernel_obj_path))
        return f"/tmp/build/kernel_{len(calls)}.so"

    ascendc_compile_module.module.compile_device_obj = fake_compile_device_obj
    ascendc_compile_module.module.link_kernel_so = fake_link_kernel_so

    result = ascendc_compile_module.build_kernel_target(
        args, ["a.o", "b.o"], "/tmp/build"
    )

    assert result == "/tmp/build/kernel_2.so"
    assert [tiling_obj_paths for tiling_obj_paths, _ in calls] == [
        ["a.o", "b.o"],
        ["a.o", "b.o"],
    ]


def test_bake_const_tiling_before_device_compile_skips_non_cv(
    ascendc_compile_module,
):
    args = _make_compile_args()
    args.force_unknown = False
    link_calls = []
    ascendc_compile_module.module.link_tiling_so = (
        lambda *unused_args, **unused_kwargs: link_calls.append(unused_args)
    )

    assert (
        ascendc_compile_module.bake_const_tiling_before_device_compile(
            args, ["a.o"], "/tmp/build"
        )
        is False
    )
    assert link_calls == []


def test_bake_const_tiling_before_device_compile_keeps_dynamic_when_not_static(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    device_dir.join("kernel_op.cpp").write("int main() { return 0; }")
    args = _make_compile_args()
    args.force_unknown = False
    args.device_files = str(device_dir.join("kernel_op.cpp"))
    rewrite_calls = []
    ascendc_compile_module.module.link_tiling_so = lambda *unused_args: "tiling.so"
    fake_lib = types.SimpleNamespace(
        AutofuseIsStaticShape=lambda: False,
    )
    ascendc_compile_module.module.ctypes.CDLL = lambda path: fake_lib
    ascendc_compile_module.module.rewrite_device_kernel_with_const_tiling = (
        lambda *unused_args: rewrite_calls.append(unused_args)
    )

    assert (
        ascendc_compile_module.bake_const_tiling_before_device_compile(
            args, ["a.o"], str(tmpdir)
        )
        is False
    )
    assert rewrite_calls == []


def test_bake_const_tiling_before_device_compile_bakes_const_tiling(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    device_dir.join("kernel_op.cpp").write("CVAutofuseTilingData")
    device_dir.join("autofuse_tiling_data.h").write("old tiling")
    args = _make_compile_args()
    args.force_unknown = False
    args.device_files = str(device_dir.join("kernel_op.cpp"))
    args.config_file = "config.txt"
    link_calls = []
    rewrite_calls = []
    ascendc_compile_module.module.link_tiling_so = (
        lambda compile_args, tiling_obj_paths, temp_dir: (
            link_calls.append((list(tiling_obj_paths), temp_dir)) or "tiling.so"
        )
    )
    fake_lib = types.SimpleNamespace(
        AutofuseIsStaticShape=lambda: True,
        GenConstTilingData=lambda *unused_args: b"const tiling",
    )
    ascendc_compile_module.module.ctypes.CDLL = lambda path: fake_lib
    ascendc_compile_module.module.static_shape_kernel_proc = lambda *unused_args: (
        rewrite_calls.append("proc")
    )
    ascendc_compile_module.module.init_torch_npu_for_const_tiling = lambda: None

    assert (
        ascendc_compile_module.bake_const_tiling_before_device_compile(
            args, ["a.o", "b.o"], str(tmpdir)
        )
        is True
    )
    assert link_calls == [(["a.o", "b.o"], str(tmpdir))]
    assert rewrite_calls == ["proc"]
    assert device_dir.join("autofuse_tiling_data.h").read() == "const tiling"


def test_build_kernel_target_compiles_device_once_when_const_tiling_baked(
    ascendc_compile_module,
):
    compile_calls = []
    args = _make_compile_args()
    args.force_unknown = False
    ascendc_compile_module.module.bake_const_tiling_before_device_compile = (
        lambda unused_args, unused_paths, unused_temp_dir: True
    )
    ascendc_compile_module.module.try_static_shape_compile = lambda *unused_args: (
        pytest.fail("try_static_shape_compile should be skipped")
    )

    def fake_compile_device_obj(unused_args, unused_temp_dir):
        compile_calls.append("compile")
        return "/tmp/build/device/kernel.o"

    ascendc_compile_module.module.compile_device_obj = fake_compile_device_obj
    ascendc_compile_module.module.link_kernel_so = lambda *unused_args: (
        "/tmp/build/kernel.so"
    )

    assert (
        ascendc_compile_module.build_kernel_target(args, ["a.o"], "/tmp/build")
        == "/tmp/build/kernel.so"
    )
    assert compile_calls == ["compile"]


def test_bake_skips_device_obj_without_tiling_paths(ascendc_compile_module):
    args = _make_compile_args()
    args.stage = "device_obj"
    args.force_unknown = False
    link_calls = []
    ascendc_compile_module.module.link_tiling_so = (
        lambda *unused_args, **unused_kwargs: link_calls.append(unused_args)
    )

    assert (
        ascendc_compile_module.bake_const_tiling_before_device_compile(
            args, [], "/tmp/build"
        )
        is False
    )
    assert link_calls == []


def test_bake_const_tiling_in_device_obj_stage(ascendc_compile_module, tmpdir):
    device_dir = tmpdir.mkdir("device")
    device_dir.join("kernel_op.cpp").write("CVAutofuseTilingData")
    device_dir.join("autofuse_tiling_data.h").write("old tiling")
    args = _make_compile_args()
    args.stage = "device_obj"
    args.force_unknown = False
    args.device_files = str(device_dir.join("kernel_op.cpp"))
    args.config_file = "config.txt"
    ascendc_compile_module.module.link_tiling_so = lambda *unused_args: "tiling.so"
    fake_lib = types.SimpleNamespace(
        AutofuseIsStaticShape=lambda: True,
        GenConstTilingData=lambda *unused_args: b"const tiling",
    )
    ascendc_compile_module.module.ctypes.CDLL = lambda path: fake_lib
    ascendc_compile_module.module.static_shape_kernel_proc = lambda *unused_args: None
    ascendc_compile_module.module.init_torch_npu_for_const_tiling = lambda: None

    assert (
        ascendc_compile_module.bake_const_tiling_before_device_compile(
            args, ["a.o"], str(tmpdir)
        )
        is True
    )
    assert device_dir.join("autofuse_tiling_data.h").read() == "const tiling"


def test_build_kernel_target_from_objects_skips_recompile_when_baked(
    ascendc_compile_module,
):
    args = _make_compile_args()
    args.const_tiling_baked = True
    ascendc_compile_module.module.link_kernel_so = lambda *unused_args: (
        "/tmp/build/kernel.so"
    )
    ascendc_compile_module.module.try_static_shape_compile = lambda *unused_args: (
        pytest.fail("try_static_shape_compile should be skipped when baked")
    )

    assert (
        ascendc_compile_module.build_kernel_target_from_objects(
            args, ["a.o"], "/tmp/build/device/kernel.o", "/tmp/build"
        )
        == "/tmp/build/kernel.so"
    )


def test_main_device_obj_returns_baked_flag(ascendc_compile_module, tmpdir):
    work = tmpdir.mkdir("work")
    args = _make_compile_args()
    args.stage = "device_obj"
    args.temp_dir = str(work)
    args.device_files = str(work.join("k_op_kernel.cpp"))
    args.tiling_obj_paths = "a.o;b.o"
    ascendc_compile_module.module.bake_const_tiling_before_device_compile = (
        lambda *unused_args, **unused_kwargs: True
    )
    ascendc_compile_module.module.compile_device_obj = lambda *unused_args: "/tmp/k.o"

    out = ascendc_compile_module.main(args)

    assert out["stage"] == "device_obj"
    assert out["kernel_obj_path"] == "/tmp/k.o"
    assert out["kernel_source_path"] == args.device_files
    assert out["baked"] is True


def test_build_kernel_target_from_objects_derives_baked_from_const_header(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    device_dir.join("autofuse_tiling_data.h").write(
        "#define INDUCTOR_CONST_TILING_DATA\n"
        "constexpr CVAutofuseTilingData kConstTilingData = {};\n"
    )
    args = _make_compile_args()
    args.const_tiling_baked = False
    ascendc_compile_module.module.link_kernel_so = lambda *unused_args: "/tmp/kernel.so"
    ascendc_compile_module.module.try_static_shape_compile = lambda *unused_args: (
        pytest.fail(
            "try_static_shape_compile should be skipped when const header present"
        )
    )

    assert (
        ascendc_compile_module.build_kernel_target_from_objects(
            args, ["a.o"], "/tmp/build/device/kernel.o", str(tmpdir)
        )
        == "/tmp/kernel.so"
    )


def test_main_link_stage_skips_recompile_when_baked(ascendc_compile_module, tmpdir):
    work = tmpdir.mkdir("work")
    work.mkdir("device")
    tiling_obj = work.join("t.o")
    tiling_obj.write("obj")
    kernel_obj = work.join("k.o")
    kernel_obj.write("obj")
    kernel_src = work.join("k_op_kernel.cpp")
    kernel_src.write("src")
    so_file = work.join("kernel.so")
    so_file.write("so")
    out_file = work.join("out.so")

    link_calls = []
    ascendc_compile_module.module.link_kernel_so = lambda *unused_args: (
        link_calls.append(1) or str(so_file)
    )
    ascendc_compile_module.module.try_static_shape_compile = lambda *unused_args: (
        pytest.fail("try_static_shape_compile must be skipped when baked")
    )

    args = _make_compile_args()
    args.stage = "link"
    args.temp_dir = str(work)
    args.tiling_obj_paths = str(tiling_obj)
    args.kernel_obj_path = str(kernel_obj)
    args.kernel_source_path = str(kernel_src)
    args.tiling_source_paths = ""
    args.shared_cv_wrapper_so = ""
    args.const_tiling_baked = True
    args.output_file = str(out_file)
    args.device_files = str(kernel_src)
    args.host_files = None

    ascendc_compile_module.main(args)

    assert link_calls == [1]
    assert out_file.check()
