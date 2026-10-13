#!/usr/bin/env python3
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
"""Device validation JIT adapter.

Compiles an ASCIR codegen artifact (``tiling.h``, ``host_impl.cpp``,
``device_impl.cpp`` produced under ``--codegen-dir``) into a
``kernel_module.so`` consumable by the device validation runner. It is the
default ``AUTOFUSE_DEVICE_JIT`` implementation and is device-agnostic:

- ``--soc_version`` is read from the device profile (``soc_version`` field)
  when not passed explicitly, falling back to the compiler adapter default.
- ``--graph-name`` is passed by the orchestrator from the case contract.
- Single-operator codegen may emit a 6-argument
  ``AutofuseTiling(s0, s1, ...)`` entry while the runner calls the 4-argument
  ABI; the adapter rewrites the host source into a 4-argument forwarding
  entry bound to ``--rows/--cols`` so unfused steps execute on device.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path


def patch_single_operator_tiling(host_code, rows, cols):
    """Return host_impl.cpp with a 4-arg tiling entry when codegen emitted a
    6-arg ``AutofuseTiling(s0, s1, ...)`` signature.

    The original 6-arg entry is renamed to ``AutofuseTilingS0S1`` and a
    4-arg ``AutofuseTiling`` forwards the requested shape into it. The
    injection is placed inside the split-host markers so the compiler's
    host-source splitter accepts it.
    """
    marker = 'extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1'
    if marker not in host_code:
        return host_code
    host_code = host_code.replace(
        marker, 'extern "C" int64_t AutofuseTilingS0S1(uint32_t s0, uint32_t s1', 1
    )
    adapter = (
        "struct ResLimit;\n"
        'extern "C" int64_t AutofuseTilingS0S1(uint32_t s0, uint32_t s1, '
        "AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim, "
        "ResLimit *res_limit);\n"
        'extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, '
        "uint64_t *workspaceSize, uint32_t *blockDim, ResLimit *res_limit) {\n"
        f"  return AutofuseTilingS0S1({rows}, {cols}, tiling, workspaceSize, "
        "blockDim, res_limit);\n"
        "}\n"
    )
    anchor = "struct ResLimit {"
    if anchor not in host_code:
        return host_code
    return host_code.replace(anchor, adapter + anchor, 1)


def patch_device_kernel(device_code):
    """Name the generated AscendC entry so the device compiler accepts it."""
    unnamed_kernel = 'extern "C" __global__ __aicore__ void (GM_ADDR'
    named_kernel = 'extern "C" __global__ __aicore__ void autofuse_kernel(GM_ADDR'
    if unnamed_kernel not in device_code:
        return device_code
    device_code = device_code.replace(unnamed_kernel, named_kernel, 1)
    launch = "  <<<blockDim, nullptr, stream>>>("
    named_launch = "  autofuse_kernel<<<blockDim, nullptr, stream>>>("
    return device_code.replace(launch, named_launch, 1)


def resolve_soc_version(profile_path, cli_value):
    if cli_value:
        return cli_value
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    version = profile.get("soc_version")
    if not version or not isinstance(version, str):
        raise ValueError(
            f"device profile {profile_path} must declare a string "
            "'soc_version' (or pass --soc-version)"
        )
    return version


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--codegen-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--cols", required=True, type=int)
    parser.add_argument("--profile", default="")
    parser.add_argument("--graph-name", default="autofuse")
    parser.add_argument("--soc-version", default="")
    args = parser.parse_args(argv)
    codegen_dir = Path(args.codegen_dir)
    try:
        soc_version = (
            resolve_soc_version(args.profile, args.soc_version)
            if args.profile
            else (args.soc_version or "Ascend910B")
        )
    except ValueError as error:
        logging.error(str(error))
        return 2
    host_code = (codegen_dir / "host_impl.cpp").read_text(encoding="utf-8")
    host_code = patch_single_operator_tiling(host_code, args.rows, args.cols)
    device_code = patch_device_kernel(
        (codegen_dir / "device_impl.cpp").read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory(prefix="device-validation-jit-") as temp_dir:
        with tempfile.TemporaryDirectory(
            prefix="device-validation-jit-out-"
        ) as out_dir:
            output_file = Path(out_dir) / "device-validation-module.so"
            try:
                from autofuse.compiler.python.compile_adapter import jit_compile
            except ImportError:
                from autofuse.compile_adapter import jit_compile
            # A repository codegen package may live outside the CANN toolkit.
            # Let the explicit toolkit environment control device compilation.
            toolkit = os.environ.get("ASCEND_HOME_PATH")
            if toolkit:
                try:
                    from autofuse.compiler.python import ascendc_compile
                except ImportError:
                    from autofuse import ascendc_compile
                ascendc_compile.ASCEND_PATH = toolkit
            jit_compile(
                (codegen_dir / "tiling.h").read_text(encoding="utf-8"),
                host_code,
                device_code,
                [
                    f"--graph_name={args.graph_name}",
                    f"--output_file={output_file}",
                    f"--output_path={temp_dir}",
                    "--force_unknown=True",
                    f"--soc_version={soc_version}",
                ],
            )
            shutil.copy2(output_file, args.output)
            output_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
