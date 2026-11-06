#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from __future__ import annotations

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DATA_COLLECTION_DIR = MODELING_DIR / "Data_collection"
HARNESS_DIR = DATA_COLLECTION_DIR / "common" / "executables" / "standalone_nddma"
ANALYSIS_SCRIPT = HARNESS_DIR / "analyze_profiling_with_params.py"
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round2"
DEFAULT_OUTPUT_DIR = DEFAULT_ANA_DIR / "collection"
DEFAULT_FACTOR_CSV = DEFAULT_ANA_DIR / "round2_1d_noncontiguous_factor.csv"

DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
DTYPES = tuple(DTYPE_SIZES)
OUTPUT_DIMS = (8, 16, 32, 64, 128, 512, 1024, 2048, 4196, 10001, 20001, 30001, 42302)
BLOCK_DIMS = (1, 2, 4, 8, 15, 32, 56)
INPUT_STRIDES = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
OUTPUT_STRIDES = (2, 4, 6, 8, 10, 12, 14, 16, 32)
BYTES_PER_CORE = (64, 128, 512, 1024, 4096, 16384)
BASE_PAIRS = ((1, 1), (1, 2), (1, 16))
C_INPUT_STRIDES = (2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
MULTICORE_INPUT_STRIDES = (2, 8, 32, 128, 256)
MULTICORE_OS2_INPUT_STRIDES = MULTICORE_INPUT_STRIDES
BASE_REFERENCE_BLOCK_DIMS = (1, 2, 4, 8, 16, 32, 56, 6, 24, 40)
VALIDATION_BLOCK_DIMS = (6, 24, 40)
REFERENCE_BYTES_PER_CORE = (64, 512, 4096, 16384)
OUTPUT_VALIDATION_PAIRS = ((2, 4), (32, 16), (128, 32))
GUARD_ELEMS = 64
MAX_GM_SPAN_BYTES = 4 * 1024 * 1024
MAX_UB_SPAN_BYTES = 256 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and collect NDDMA2 Round2 1D non-contiguous data."
    )
    parser.add_argument("--factor-csv", default=str(DEFAULT_FACTOR_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--executable-dir", default=str(HARNESS_DIR))
    parser.add_argument("--msprof-bin", default=os.environ.get("MSPROF_BIN", ""))
    parser.add_argument("--analysis-pipes", default="mte2")
    parser.add_argument("--op-name-filter", default="nddma")
    parser.add_argument("--kernel-repeat", type=int, default=200)
    parser.add_argument("--execution-repeat-count", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--run-mode", default="")
    parser.add_argument("--npu-arch", default=os.environ.get("NPU_ARCH", ""))
    parser.add_argument("--cmake-extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.npu_arch = args.npu_arch.strip()
    if not args.npu_arch and args.run_mode != "cpu":
        message = "NPU_ARCH is not configured. Set it for your actual v35 device, e.g. export NPU_ARCH=dav-3510, or pass --npu-arch dav-3510 to collect.py / run_nddma_ad_hoc.py. dav-3510 is an example, not a default for every device. e2e.py also accepts --npu-arch and forwards it to collection."
        if args.dry_run:
            print(f"[WARN] {message}")
        else:
            parser.error(message)
    return args


def span(dim: int, stride: int) -> int:
    return 1 + (dim - 1) * stride


def safe(dtype: str, dim: int, input_stride: int, output_stride: int) -> bool:
    size = DTYPE_SIZES[dtype]
    return (span(dim, input_stride) + GUARD_ELEMS) * size <= MAX_GM_SPAN_BYTES and (
        span(dim, output_stride) + GUARD_ELEMS
    ) * size <= MAX_UB_SPAN_BYTES


def row(
    index: int,
    group: str,
    dtype: str,
    block_dim: int,
    bytes_per_core: int,
    input_stride: int,
    output_stride: int,
    repeat: int,
    execution_repeat_count: int,
) -> dict[str, str]:
    size = DTYPE_SIZES[dtype]
    output_dim = bytes_per_core // size
    token = (
        f"r2_{group.lower()}_{dtype}_k{block_dim}_b{bytes_per_core}"
        f"_i{input_stride}_o{output_stride}"
    )
    input_span = span(output_dim, input_stride)
    output_span = span(output_dim, output_stride)
    return {
        "experiment_idx": str(index),
        "token": token,
        "config_id": token,
        "round_id": "r2_1d_noncontiguous",
        "sample_id": token,
        "execution_repeat_count": str(execution_repeat_count),
        "sensitivity_id": f"R2{group}",
        "sensitivity_key": f"group_{group.lower()}",
        "group_id": group,
        "group_name": {
            "A": "single_core_input",
            "B": "single_core_output",
            "C": "single_core_joint",
            "F": "multicore_base",
            "G": "multicore_input",
            "H": "multicore_joint",
            "I": "multicore_output_validation",
            "J": "multicore_requested_validation",
        }[group],
        "model_target": "NDDMA round4 1D non-contiguous staged model",
        "metric_target": "nddma_mte2_cycles_per_block",
        "fit_role": "fit" if group in {"A", "C", "G", "H"} else "validation",
        "dtype": dtype,
        "dtype_size": str(size),
        "dim": "1",
        "block_dim": str(block_dim),
        "repeat": str(repeat),
        "output_dims": str(output_dim),
        "output_stride": str(output_stride),
        "input_stride": str(input_stride),
        "input_stride_axis": "0",
        "output_stride_axis": "0",
        "src_offset_elem": "0",
        "dst_offset_elem": "0",
        "inner_elems": str(output_dim),
        "inner_bytes": str(bytes_per_core),
        "total_elems": str(output_dim),
        "total_bytes": str(bytes_per_core),
        "bytes_per_core": str(bytes_per_core),
        "logical_total_bytes": str(bytes_per_core * block_dim),
        "gm_span_elems": str(input_span),
        "ub_span_elems": str(output_span),
        "input_span_bytes": str(input_span * size),
        "output_span_bytes": str(output_span * size),
        "input_stride_pattern": "contiguous" if input_stride == 1 else "strided",
        "output_stride_pattern": "contiguous" if output_stride == 1 else "strided",
        "layout_pattern": f"i{input_stride}_o{output_stride}",
        "parent_model": "round2_base_then_N_G_then_N_GU",
        "notes": "Round4 1D non-contiguous model; B is bytes per core.",
    }


def build_rows(repeat: int, execution_repeat_count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(
        group: str, dtype: str, k: int, b: int, i: int, o: int, role: str = ""
    ) -> None:
        del role
        dim = b // DTYPE_SIZES[dtype]
        if safe(dtype, dim, i, o):
            rows.append(
                row(
                    len(rows) + 1,
                    group,
                    dtype,
                    k,
                    b,
                    i,
                    o,
                    repeat,
                    execution_repeat_count,
                )
            )

    for dtype in DTYPES:
        for dim in OUTPUT_DIMS:
            b = dim * DTYPE_SIZES[dtype]
            add("A", dtype, 1, b, 2, 1)
            for i in INPUT_STRIDES[1:]:
                add("A", dtype, 1, b, i, 1)
            for o in OUTPUT_STRIDES:
                add("B", dtype, 1, b, 1, o)
            for i in C_INPUT_STRIDES:
                for o in OUTPUT_STRIDES:
                    add("C", dtype, 1, b, i, o)

    for dtype in DTYPES:
        for k in BLOCK_DIMS:
            for b in BYTES_PER_CORE:
                for i in MULTICORE_INPUT_STRIDES:
                    add("G", dtype, k, b, i, 1)
                for i in MULTICORE_OS2_INPUT_STRIDES:
                    add("H", dtype, k, b, i, 2)
        for k in BASE_REFERENCE_BLOCK_DIMS:
            for b in REFERENCE_BYTES_PER_CORE:
                for i, o in BASE_PAIRS:
                    add("F", dtype, k, b, i, o)
        for k in VALIDATION_BLOCK_DIMS:
            for b in REFERENCE_BYTES_PER_CORE:
                for i, o in OUTPUT_VALIDATION_PAIRS:
                    add("I", dtype, k, b, i, o)

    add("J", "int32_t", 55, 9656 * 4, 3, 1)
    return rows


def write_factor(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote factor csv: {path} ({len(rows)} rows)")


def resolve_msprof_bin(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    found = shutil.which(text)
    if found:
        return found
    path = Path(text).expanduser()
    if path.exists():
        return str(path.resolve())
    raise ValueError(f"msprof binary not found: {text}")


def run(command: list[str], dry_run: bool) -> int:
    print("[INFO]", shlex.join(command))
    return 0 if dry_run else subprocess.run(command).returncode


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def reset_output(path: Path) -> None:
    for name in ("profiling_raw", "experiment", "analysis"):
        shutil.rmtree(path / name, ignore_errors=True)
    for name in ("command.log", "stdout.log", "stderr.log", "analysis.log"):
        (path / name).unlink(missing_ok=True)
    (path / "experiment").mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    try:
        args.msprof_bin = (
            (args.msprof_bin if args.dry_run else resolve_msprof_bin(args.msprof_bin))
            if args.msprof_bin
            else ""
        )
    except ValueError as error:
        print(f"[ERROR] {error}")
        return 1
    factor_csv = Path(args.factor_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = build_rows(args.kernel_repeat, args.execution_repeat_count)
    if not args.dry_run:
        write_factor(factor_csv, rows)
    executable_dir = Path(args.executable_dir).resolve()
    if not args.dry_run:
        reset_output(output_dir)
    build_dir = output_dir / "build"
    executable = build_dir / "demo_nddma"
    configure = [
        "cmake",
        "-S",
        str(executable_dir),
        "-B",
        str(build_dir),
        "-DNDDMA_STAGE=1",
        f"-DNPU_ARCH={args.npu_arch}",
    ]
    if args.run_mode:
        configure.append(f"-DRUN_MODE={args.run_mode}")
    configure.extend(args.cmake_extra_arg)
    status = run(configure, args.dry_run)
    if status:
        return status
    status = run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "demo_nddma",
            "-j",
            str(args.jobs),
        ],
        args.dry_run,
    )
    if status:
        return status
    experiment_dir = output_dir / "experiment"
    raw_dir = output_dir / "profiling_raw"
    app = [
        str(executable),
        f"--factor-csv={factor_csv}",
        f"--output-dir={experiment_dir}",
    ]
    command = (
        [args.msprof_bin, f"--output={raw_dir}", f"--application={shlex.join(app)}"]
        if args.msprof_bin
        else app
    )
    if not args.dry_run:
        write_text(output_dir / "command.log", shlex.join(command) + "\n")
    status = run(command, args.dry_run)
    if status or args.dry_run or not args.msprof_bin:
        if not args.msprof_bin:
            print("[WARN] msprof was not provided; profiling analysis was skipped.")
        return status
    if not any(raw_dir.rglob("op_summary_*.csv")):
        print(f"[ERROR] no op_summary_*.csv found under {raw_dir}")
        return 1
    analysis_dir = output_dir / "analysis"
    analysis_command = [
        sys.executable,
        str(ANALYSIS_SCRIPT),
        f"--profiling-dir={raw_dir}",
        f"--experiment-log={experiment_dir / 'experiment.log'}",
        f"--factor-csv={factor_csv}",
        f"--output-dir={analysis_dir}",
        f"--pipes={args.analysis_pipes}",
        f"--op-name-filter={args.op_name_filter}",
    ]
    completed = subprocess.run(analysis_command, text=True, capture_output=True)
    write_text(
        output_dir / "analysis.log",
        "COMMAND\n"
        + shlex.join(analysis_command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
