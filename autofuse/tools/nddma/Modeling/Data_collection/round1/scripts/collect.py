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
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DATA_COLLECTION_DIR = MODELING_DIR / "Data_collection"
HARNESS_DIR = DATA_COLLECTION_DIR / "common" / "executables" / "standalone_nddma"
ANALYSIS_SCRIPT = HARNESS_DIR / "analyze_profiling_with_params.py"
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round1"
DEFAULT_OUTPUT_DIR = DEFAULT_ANA_DIR / "collection"
DEFAULT_FACTOR_CSV = DEFAULT_ANA_DIR / "round1_1d_single_core_factor.csv"
ROUND_ID = "r1_1d_round2_compatible"
GROUP_ID = "R2_AC_1D"
DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
C_X_BY_DTYPE = {
    "int64_t": (4096, 8192, 16384, 32768, 49152, 61440, 90112, 130048, 180224, 256000),
    "int32_t": (4096, 8192, 16384, 32768, 49152, 61440, 90112, 130048, 180224, 256000),
    "int16_t": (4096, 8192, 16384, 32768, 49152, 61440, 90112, 130048),
    "int8_t": (4096, 8192, 16384, 32768, 49152, 61440),
}
# HW_GE_ATT Round2 A+C fit grid; B/D are validation, not Base training.
STAGE1_X_BY_DTYPE = {
    "int64_t": (
        4096,
        6144,
        8192,
        12288,
        16384,
        24576,
        32768,
        40960,
        49152,
        57344,
        61440,
        73728,
        90112,
        106496,
        122880,
        147456,
        163840,
        180224,
        212992,
        229376,
        256000,
    ),
    "int32_t": (
        4096,
        6144,
        8192,
        12288,
        16384,
        24576,
        32768,
        40960,
        49152,
        57344,
        61440,
        73728,
        90112,
        106496,
        122880,
        147456,
        163840,
        180224,
        212992,
        229376,
        256000,
    ),
    "int16_t": (
        4096,
        6144,
        8192,
        12288,
        16384,
        24576,
        32768,
        40960,
        49152,
        57344,
        61440,
        73728,
        90112,
        106496,
        122880,
        130048,
    ),
    "int8_t": (
        4096,
        6144,
        8192,
        12288,
        16384,
        24576,
        32768,
        40960,
        49152,
        57344,
        61440,
    ),
}
A_BLOCK_DIMS = (1, 2, 4, 8, 16, 32, 56)
BLOCK_DIMS = tuple(range(1, 57))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and collect focused NDDMA2 Round1 1D data."
    )
    parser.add_argument("--factor-csv", default=str(DEFAULT_FACTOR_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--executable-dir", default=str(HARNESS_DIR))
    parser.add_argument("--msprof-bin", default=os.environ.get("MSPROF_BIN", ""))
    parser.add_argument("--analysis-pipes", default="mte2")
    parser.add_argument("--op-name-filter", default="nddma")
    parser.add_argument("--kernel-repeat", type=int, default=200)
    parser.add_argument("--execution-repeat-count", type=int, default=1)
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


def join_dims(values: Iterable[int]) -> str:
    return "x".join(str(int(value)) for value in values)


def generate_factor(
    path: Path, kernel_repeat: int, execution_repeat_count: int
) -> None:
    rows = []
    index = 1
    groups = (
        (
            "A",
            "group_a_stage1_linear_fit",
            "stage1_linear_fit",
            "stage1_representative_block_linear_fit",
            STAGE1_X_BY_DTYPE,
            A_BLOCK_DIMS,
        ),
        (
            "C",
            "group_c_t_alpha_block_model",
            "t_alpha_block_model",
            "all_block_dim_t_alpha_model_fit",
            C_X_BY_DTYPE,
            BLOCK_DIMS,
        ),
    )
    for (
        group_id,
        sensitivity_key,
        group_name,
        sensitivity_name,
        x_by_dtype,
        block_dims,
    ) in groups:
        for dtype in DTYPES:
            dtype_size = DTYPE_SIZES[dtype]
            for bytes_per_core in x_by_dtype[dtype]:
                output_dims = bytes_per_core // dtype_size
                for block_dim in block_dims:
                    logical_total_bytes = bytes_per_core * block_dim
                    token = f"{ROUND_ID}_{group_id.lower()}_{dtype}_b{bytes_per_core}_c{block_dim}"
                    rows.append(
                        {
                            "experiment_idx": str(index),
                            "token": token,
                            "round_id": ROUND_ID,
                            "sample_id": token,
                            "execution_repeat_index": "",
                            "execution_repeat_count": str(execution_repeat_count),
                            "sensitivity_id": f"{GROUP_ID}_{group_id}",
                            "sensitivity_key": sensitivity_key,
                            "sensitivity_name": sensitivity_name,
                            "stage_define": "1" if group_id == "A" else "3",
                            "group_id": group_id,
                            "group_name": group_name,
                            "model_target": "cycles = bytes_per_core / T + H",
                            "metric_target": "nddma_mte2_cycles_per_block",
                            "scan_variable": "dtype,logical_total_bytes",
                            "controlled_variables": (
                                f"dim=1,block_dim={','.join(str(value) for value in block_dims)},input_stride=1,output_stride=1,"
                                "GM/UB contiguous,src/dst aligned"
                            ),
                            "dtype": dtype,
                            "dtype_size": str(dtype_size),
                            "dim": "1",
                            "block_dim": str(block_dim),
                            "repeat": str(kernel_repeat),
                            "output_dims": str(output_dims),
                            "output_stride": "1",
                            "input_stride": "1",
                            "input_stride_axis": "-1",
                            "input_stride_multiplier": "1",
                            "output_stride_axis": "-1",
                            "output_stride_multiplier": "1",
                            "src_offset_elem": "0",
                            "dst_offset_elem": "0",
                            "src_align_mod32": "0",
                            "dst_align_mod32": "0",
                            "inner_elems": str(output_dims),
                            "inner_bytes": str(bytes_per_core),
                            "total_elems": str(output_dims),
                            "total_bytes": str(logical_total_bytes),
                            "bytes_per_core": str(bytes_per_core),
                            "logical_total_bytes": str(logical_total_bytes),
                            "gm_span_elems": str(output_dims),
                            "ub_span_elems": str(output_dims),
                            "input_stride_pattern": "contiguous",
                            "output_stride_pattern": "contiguous",
                            "layout_pattern": "contiguous",
                            "notes": f"NDDMA2 Round1 uses HW_GE_ATT round2 {group_id}-group 1D contiguous fit sample.",
                            "fit_role": "fit",
                            "bytes_region": "stage1_representative"
                            if group_id == "A"
                            else "t_alpha_block_model",
                            "shape_policy": "dim1",
                            "model_family": "contiguous_baseline",
                        }
                    )
                    index += 1

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
    parts = shlex.split(text)
    if len(parts) == 2 and parts[0] == "which":
        found = shutil.which(parts[1])
        if found:
            return found
        raise ValueError(f"unable to resolve msprof command: {text}")
    found = shutil.which(text)
    if found:
        return found
    path = Path(text).expanduser()
    if path.exists():
        return str(path.resolve())
    raise ValueError(f"msprof binary not found: {text}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_step(command: list[str], dry_run: bool) -> int:
    print("[INFO]", shlex.join(command))
    if dry_run:
        return 0
    return subprocess.run(command).returncode


def reset_output_dir(output_dir: Path) -> None:
    for name in ("profiling_raw", "experiment", "analysis"):
        shutil.rmtree(output_dir / name, ignore_errors=True)
    for name in ("command.log", "stdout.log", "stderr.log", "analysis.log"):
        (output_dir / name).unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def has_op_summary(path: Path) -> bool:
    return any(path.rglob("op_summary_*.csv"))


def collect_measurements(
    args: argparse.Namespace, factor_csv: Path, output_dir: Path, executable: Path
) -> int:
    if not args.dry_run:
        reset_output_dir(output_dir)
    experiment_dir = output_dir / "experiment"
    raw_dir = output_dir / "profiling_raw"
    analysis_dir = output_dir / "analysis"
    # The ASC harness opens experiment.log directly and does not create its
    # output directory.
    if not args.dry_run:
        experiment_dir.mkdir(parents=True, exist_ok=True)
    app_command = [
        str(executable),
        f"--factor-csv={factor_csv}",
        f"--output-dir={experiment_dir}",
    ]
    command = (
        [
            args.msprof_bin,
            f"--output={raw_dir}",
            f"--application={shlex.join(app_command)}",
        ]
        if args.msprof_bin
        else app_command
    )
    if not args.dry_run:
        write_text(output_dir / "command.log", shlex.join(command) + "\n")
    status = run_step(command, args.dry_run)
    if status != 0:
        return status
    if args.dry_run:
        return 0
    if not args.msprof_bin:
        print("[WARN] msprof was not provided; profiling analysis was skipped.")
        return 0

    if not has_op_summary(raw_dir):
        print(f"[ERROR] no op_summary_*.csv found under {raw_dir}")
        return 1
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
    print("[INFO]", shlex.join(analysis_command))
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
    if completed.returncode != 0:
        print(f"[ERROR] profiling analysis failed; see {output_dir / 'analysis.log'}")
    return completed.returncode


def main() -> int:
    args = parse_args()
    if args.msprof_bin and not args.dry_run:
        try:
            args.msprof_bin = resolve_msprof_bin(args.msprof_bin)
        except ValueError as error:
            print(f"[ERROR] {error}")
            return 1

    factor_csv = Path(args.factor_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    executable_dir = Path(args.executable_dir).resolve()
    if not args.dry_run:
        generate_factor(factor_csv, args.kernel_repeat, args.execution_repeat_count)

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
    status = run_step(configure, args.dry_run)
    if status:
        return status
    status = run_step(
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
    return collect_measurements(args, factor_csv, output_dir, executable)


if __name__ == "__main__":
    raise SystemExit(main())
