#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License"). You may not use this file except in
# compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import argparse
import ctypes
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

UINT32_MAX = 0xFFFFFFFF


def run_codegen(command, repeat):
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        subprocess.run(shlex.split(command), check=True)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def run_tiling(so_path, abi_width, warmup, repeat, shape, expected_shape_count):
    if len(shape) != expected_shape_count:
        raise ValueError(
            f"shape count mismatch: expected {expected_shape_count}, got {len(shape)}"
        )
    if any(not 0 <= dim <= UINT32_MAX for dim in shape):
        raise ValueError("shape dimensions must fit uint32_t")
    lib = ctypes.CDLL(so_path)
    lib.GetTilingDataSize.restype = ctypes.c_size_t
    tiling_size = lib.GetTilingDataSize()
    if tiling_size <= 0:
        raise ValueError("GetTilingDataSize returned an invalid size")
    tiling = ctypes.create_string_buffer(tiling_size)
    workspace_type = ctypes.c_uint32 if abi_width == 32 else ctypes.c_uint64
    lib.AutofuseTiling.argtypes = [ctypes.c_uint32] * len(shape) + [
        ctypes.c_void_p,
        ctypes.POINTER(workspace_type),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    lib.AutofuseTiling.restype = ctypes.c_int64

    def call():
        workspace = workspace_type(0)
        block_dim = ctypes.c_uint32(0)
        result = lib.AutofuseTiling(
            *(
                list(shape)
                + [tiling, ctypes.byref(workspace), ctypes.byref(block_dim), None]
            )
        )
        if result != 0:
            raise RuntimeError(f"AutofuseTiling failed: {result}")

    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codegen-command")
    parser.add_argument("--tiling-so")
    parser.add_argument("--abi-width", type=int, choices=(32, 64), required=True)
    parser.add_argument("--shape", nargs="+", type=int, default=[])
    parser.add_argument("--expected-shape-count", type=int, default=0)
    parser.add_argument("--codegen-repeat", type=int, default=5)
    parser.add_argument("--tiling-warmup", type=int, default=100)
    parser.add_argument("--tiling-repeat", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = {
        "abi_width": args.abi_width,
        "environment": {
            "machine": platform.platform(),
            "python": sys.version,
            "compiler": shutil.which(os.environ.get("CXX", "c++")),
            "cann_path": os.environ.get(
                "ASCEND_HOME_PATH", os.environ.get("ASCEND_INSTALL_PATH", "")
            ),
            "so_path": str(Path(args.tiling_so).resolve()) if args.tiling_so else None,
        },
        "codegen_median_ms": run_codegen(args.codegen_command, args.codegen_repeat)
        if args.codegen_command
        else None,
        "tiling_median_us": run_tiling(
            args.tiling_so,
            args.abi_width,
            args.tiling_warmup,
            args.tiling_repeat,
            args.shape,
            args.expected_shape_count,
        )
        if args.tiling_so
        else None,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
