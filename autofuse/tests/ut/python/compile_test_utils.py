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

import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[4]
PYTHON_DIR = os.fspath(BASE_DIR / "autofuse/compiler/python")
ASC_CODEGEN_COMPILE_MODULE_NAME = "autofuse.compiler.python.asc_codegen_compile"
ASC_CODEGEN_COMPILE_MODULE_PATH = os.path.join(PYTHON_DIR, "asc_codegen_compile.py")
TILING_HEADER_FILES = dict(
    TilingHead="autofuse_tiling_func_common.h",
    TilingStateHeader="autofuse_tiling_func_state.h",
    TilingLogHeader="autofuse_tiling_func_log.h",
    TilingPgoHeader="autofuse_tiling_func_pgo.h",
    TilingBaseHeader="autofuse_tiling_func_base.h",
    TilingSolverHeader="autofuse_tiling_func_solver.h",
    TilingApiHeader="autofuse_tiling_func_api.h",
    TilingEntryHeader="autofuse_tiling_func_entry.h",
    TilingTailHeader="autofuse_tiling_func_tail.h",
    ACubeKernelTilingWrapperHpp="cube_kernel_tiling_wrapper.h",
    CubeKernelTilingWrapperHpp="cube_kernel_tiling_wrapper.h",
)


@dataclass
class LoadedCompileModule:
    module: object
    duration_records: list = field(default_factory=list)
    duration_reports: list = field(default_factory=list)

    def __getattr__(self, name):
        return getattr(self.module, name)


@contextmanager
def load_compile_module(
    module_name, module_path, extra_autofuse_attrs=None, extra_modules=None
):
    duration_records = []
    duration_reports = []
    original_modules = {}
    extra_modules = extra_modules or {}

    class FakeUtils:
        @staticmethod
        def duration_record(labels, start, duration):
            duration_records.append((labels, start, duration))

        @staticmethod
        def report_durations():
            duration_reports.append(True)

    pyautofuse_module = types.ModuleType("autofuse.pyautofuse")
    pyautofuse_module.ascir = types.SimpleNamespace(utils=FakeUtils)
    autofuse_module = types.ModuleType("autofuse")
    autofuse_module.__path__ = [PYTHON_DIR]
    autofuse_module.pyautofuse = pyautofuse_module
    if extra_autofuse_attrs:
        for name, value in extra_autofuse_attrs.items():
            setattr(autofuse_module, name, value)

    modules_to_patch = {
        "autofuse": autofuse_module,
        "autofuse.pyautofuse": pyautofuse_module,
        **extra_modules,
    }
    for name, value in modules_to_patch.items():
        original_modules[name] = sys.modules.get(name)
        sys.modules[name] = value

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        if spec is not None and spec.loader is not None:
            spec.loader.exec_module(module)
        yield LoadedCompileModule(module, duration_records, duration_reports)
    finally:
        _restore_modules(original_modules)


class _AscCodegenDummyLogger(object):
    @staticmethod
    def info(*args, **kwargs):
        return None

    @staticmethod
    def error(*args, **kwargs):
        return None

    @staticmethod
    def warning(*args, **kwargs):
        return None


def _patch_module(original_modules, name, module):
    if name not in original_modules:
        original_modules[name] = sys.modules.get(name)
    sys.modules[name] = module
    return module


def make_tbe_log_stub_modules():
    """Build sys.modules stubs for tbe.common.utils.log.

    Loading the real TBE log module takes ~1 minute and its package init
    pollutes process-global state, which makes ascir Optimize tests that run
    later in the same pytest process fail with "Optimize fail". Compile
    modules under test only use info/warning/error logging, so a stub keeps
    them hermetic and fast.
    """
    stubs = {}

    def _stub(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        stubs[name] = module
        return module

    tbe_module = _stub("tbe")
    _stub("tbe.common")
    _stub("tbe.common.utils")
    log_module = _stub(
        "tbe.common.utils.log",
        info=_AscCodegenDummyLogger.info,
        warning=_AscCodegenDummyLogger.warning,
        error=_AscCodegenDummyLogger.error,
        debug=lambda *args, **kwargs: None,
    )
    tbe_module.common = stubs["tbe.common"]
    stubs["tbe.common"].utils = stubs["tbe.common.utils"]
    stubs["tbe.common.utils"].log = log_module
    return stubs


def _stub_module(original_modules, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return _patch_module(original_modules, name, module)


def _stub_tbe_common_modules(original_modules):
    _stub_module(original_modules, "tbe")
    _stub_module(original_modules, "tbe.common")
    _stub_module(
        original_modules, "tbe.common.buildcfg", get_current_build_config=lambda: {}
    )
    _stub_module(original_modules, "tbe.common.utils")
    _stub_module(
        original_modules,
        "tbe.common.utils.log",
        info=_AscCodegenDummyLogger.info,
        error=_AscCodegenDummyLogger.error,
        warning=_AscCodegenDummyLogger.warning,
    )
    _stub_module(
        original_modules,
        "tbe.common.utils.op_tiling",
        do_op_tiling=lambda *args, **kwargs: None,
    )
    _stub_module(
        original_modules,
        "tbe.common.context",
        get_context=lambda: types.SimpleNamespace(get_compile_info=lambda: {}),
    )


def _stub_tikcpp_modules(original_modules):
    tikcpp_module = _stub_module(original_modules, "tbe.tikcpp")
    _stub_module(
        original_modules,
        "tbe.tikcpp.compile_op",
        CommonUtility=types.SimpleNamespace(
            print_compile_log=lambda *args, **kwargs: None
        ),
        AscendCLogLevel=types.SimpleNamespace(
            LOG_ERROR="error", LOG_DEBUG="debug", LOG_WARNING="warning"
        ),
    )
    _stub_module(
        original_modules,
        "tbe.tikcpp.get_op_tiling",
        TilingInfo=object,
        _change_param_name_to_name=lambda *args, **kwargs: None,
        gen_static_shape_v2=lambda *args, **kwargs: None,
    )
    tikcpp_module.OpInfo = object


def _stub_asc_compile_base_modules(original_modules):
    _stub_module(original_modules, "asc_op_compile_base")
    _stub_module(original_modules, "asc_op_compile_base.common")
    _stub_module(original_modules, "asc_op_compile_base.common.platform")
    _stub_module(
        original_modules,
        "asc_op_compile_base.common.platform.platform_info",
        get_soc_spec=lambda *args, **kwargs: None,
    )


def _stub_autofuse_compile_modules(original_modules):
    package = _stub_module(original_modules, "autofuse")
    compiler_pkg = _stub_module(original_modules, "autofuse.compiler")
    python_pkg = _stub_module(original_modules, "autofuse.compiler.python")
    package.__path__ = [os.fspath(BASE_DIR / "autofuse")]
    compiler_pkg.__path__ = [os.fspath(BASE_DIR / "autofuse/compiler")]
    python_pkg.__path__ = [PYTHON_DIR]
    package.compiler = compiler_pkg
    compiler_pkg.python = python_pkg

    package_prefix = ASC_CODEGEN_COMPILE_MODULE_NAME.rsplit(".", 1)[0]
    _stub_module(
        original_modules,
        package_prefix + ".pyautofuse",
        Schedule=object,
        CodeGen=object,
        ascir=types.SimpleNamespace(),
    )
    _stub_module(
        original_modules,
        package_prefix + ".ascbc_kernel_compile",
        ascbc_kernel_compile=lambda *args, **kwargs: ("kernel.o", "kernel.json"),
        camel_to_snake=lambda value: value,
    )
    _stub_module(
        original_modules,
        package_prefix + ".compile_adapter",
        TILING_HEADER_FILES=TILING_HEADER_FILES,
        get_pgo_env_flag=lambda: False,
        get_pgo_topn=lambda: 5,
    )


def _load_asc_codegen_compile_module(original_modules):
    spec = importlib.util.spec_from_file_location(
        ASC_CODEGEN_COMPILE_MODULE_NAME, ASC_CODEGEN_COMPILE_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot load module {ASC_CODEGEN_COMPILE_MODULE_NAME} from {ASC_CODEGEN_COMPILE_MODULE_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    _patch_module(original_modules, ASC_CODEGEN_COMPILE_MODULE_NAME, module)
    spec.loader.exec_module(module)
    return module


def _restore_modules(original_modules):
    saved_modules = {
        name: module for name, module in original_modules.items() if module is not None
    }
    sys.modules.update(saved_modules)
    for name in original_modules.keys() - saved_modules.keys():
        sys.modules.pop(name, None)


@contextmanager
def load_asc_codegen_compile_module():
    original_modules = {}
    try:
        _stub_tbe_common_modules(original_modules)
        _stub_tikcpp_modules(original_modules)
        _stub_asc_compile_base_modules(original_modules)
        _stub_autofuse_compile_modules(original_modules)
        yield _load_asc_codegen_compile_module(original_modules)
    finally:
        _restore_modules(original_modules)


def process_tiling_funcs_and_infershape(
    module, tiling_func_srcs, graph_name, host_build_dir, infershape_src
):
    process_func = getattr(module, "_process_tiling_funcs_and_infershape")
    return process_func(tiling_func_srcs, graph_name, host_build_dir, infershape_src)
