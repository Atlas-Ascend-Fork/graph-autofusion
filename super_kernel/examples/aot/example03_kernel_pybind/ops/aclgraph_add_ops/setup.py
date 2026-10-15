# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import glob
import os
import sysconfig
from distutils.errors import CompileError
from distutils.spawn import find_executable

import torch
import torch_npu
import torch.utils.cpp_extension as cpp_extension
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SOURCE_FILES = sorted(
    glob.glob(os.path.join(BASE_DIR, "csrc", "*.asc"), recursive=True)
)


def _select_npu_arch():
    npu_arch = os.getenv("SK_NPU_ARCH", "")
    if npu_arch not in ("dav-2201", "dav-3510"):
        raise RuntimeError("set SK_NPU_ARCH to dav-2201 or dav-3510")
    return npu_arch


NPU_ARCH = _select_npu_arch()


def _torch_abi_compile_flags():
    flags = [f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    for name in ("COMPILER_TYPE", "STDLIB", "BUILD_ABI"):
        value = getattr(torch._C, f"_PYBIND11_{name}", None)
        if value is not None:
            flags.append(f'-DPYBIND11_{name}="{value}"')
    return flags


def get_dependency_paths():
    python_include = sysconfig.get_config_var("INCLUDEPY")
    python_lib = sysconfig.get_config_var("LIBDIR")

    torch_include_paths = cpp_extension.include_paths()
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_lib = os.path.join(torch_npu_path, "lib")

    return {
        "all_includes": [*torch_include_paths, python_include, torch_npu_include],
        "all_libs": [python_lib, torch_lib, torch_npu_lib],
    }


class AscendBuildExtension(build_ext):
    def _check_bisheng_compiler(self):
        if not find_executable("bisheng"):
            raise RuntimeError(
                "bisheng command not found. Please source CANN set_env.sh first."
            )

    def build_extension(self, ext):
        self._check_bisheng_compiler()
        dep_paths = get_dependency_paths()

        ext_fullpath = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)

        compile_cmd = [
            "bisheng",
            "-x",
            "asc",
            f"--npu-arch={NPU_ARCH}",
            "-shared",
            "-fPIC",
            "-std=c++17",
            *_torch_abi_compile_flags(),
            "-ltorch_npu",
            "-ltorch",
            "-lc10",
            *ext.sources,
            "-o",
            ext_fullpath,
        ]

        for include_dir in dep_paths["all_includes"]:
            compile_cmd.append(f"-I{include_dir}")

        for lib_dir in dep_paths["all_libs"]:
            compile_cmd.append(f"-L{lib_dir}")

        try:
            self.spawn(compile_cmd)
        except Exception as exc:
            raise CompileError(str(exc)) from exc


setup(
    name="op_extension",
    version="0.1",
    ext_modules=[
        Extension(
            name="op_extension.custom_ops_lib",
            sources=SOURCE_FILES,
            language="asc",
        )
    ],
    packages=find_packages(),
    cmdclass={"build_ext": AscendBuildExtension},
)
