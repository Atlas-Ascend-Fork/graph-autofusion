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
from itertools import product
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DATA_COLLECTION_DIR = MODELING_DIR / "Data_collection"
HARNESS_DIR = DATA_COLLECTION_DIR / "common" / "executables" / "standalone_nddma"
ANALYSIS_SCRIPT = HARNESS_DIR / "analyze_profiling_with_params.py"
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round4"
DEFAULT_OUTPUT_DIR = DEFAULT_ANA_DIR / "collection"
DEFAULT_FACTOR_CSV = DEFAULT_ANA_DIR / "round4_2d_ub_contiguous_ng2_factor.csv"

DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
DTYPES = tuple(DTYPE_SIZES)
BLOCK_DIMS = (1,)
M_VALUES = (2, 3, 4, 8, 16, 32, 64, 128, 156, 256)
N_VALUES = (8, 16, 32, 64, 128, 512, 1024, 2048, 4196, 10001, 20001, 30001, 42302)
INPUT_STRIDES = (2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 3333, 4444)
OUTER_INPUT_STRIDES = (2, 4, 6, 8, 16)
HIGH_STRIDE_INCREMENTAL_VALUES = (130, 132, 136, 144, 160, 192, 512)
MAX_GM_SPAN_BYTES = 4 * 1024 * 1024
MAX_UB_SPAN_BYTES = 256 * 1024
GUARD_ELEMS = 64
KERNEL_REPEAT = 200

CSV_COLUMNS = (
    "experiment_idx",
    "token",
    "config_id",
    "round_id",
    "sample_id",
    "execution_repeat_count",
    "sensitivity_id",
    "sensitivity_key",
    "sensitivity_name",
    "stage_define",
    "group_id",
    "group_name",
    "fit_stage",
    "parent_model",
    "model_target",
    "metric_target",
    "scan_variable",
    "controlled_variables",
    "dtype",
    "dtype_size",
    "dim",
    "block_dim",
    "repeat",
    "output_dims",
    "output_stride",
    "input_stride",
    "input_stride_axis",
    "input_stride_multiplier",
    "output_stride_axis",
    "output_stride_multiplier",
    "src_offset_elem",
    "dst_offset_elem",
    "src_align_mod32",
    "dst_align_mod32",
    "inner_elems",
    "inner_bytes",
    "total_elems",
    "total_bytes",
    "bytes_per_core",
    "logical_total_bytes",
    "gm_span_elems",
    "ub_span_elems",
    "input_stride_pattern",
    "output_stride_pattern",
    "layout_pattern",
    "notes",
    "fit_role",
    "bytes_region",
    "shape_policy",
    "model_family",
    "data_split",
    "shape_family_id",
    "input_layout_policy",
    "output_layout_policy",
    "input_span_elems",
    "output_span_elems",
    "input_span_bytes",
    "output_span_bytes",
    "input_gap_elems",
    "output_gap_elems",
    "input_gap_bytes",
    "output_gap_bytes",
    "basis_variable_count",
    "basis_candidate_count",
    "sampling_seed",
    "m",
    "n",
    "is2",
    "is1",
    "ng2_input_stride",
    "inner_input_stride",
    "s_byte_stride",
    "pair_role",
    "address_set_id",
    "api_output_dims",
    "api_input_stride",
    "api_output_stride",
    "loop_output_dims_reversed",
    "loop_input_stride_reversed",
    "loop_output_stride_reversed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and collect NDDMA2 Round4 data aligned to HW Round5 D/F/G N_G2 training data."
    )
    parser.add_argument("--factor-csv", default=str(DEFAULT_FACTOR_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--executable-dir", default=str(HARNESS_DIR))
    parser.add_argument("--msprof-bin", default=os.environ.get("MSPROF_BIN", ""))
    parser.add_argument("--analysis-pipes", default="mte2")
    parser.add_argument("--op-name-filter", default="nddma")
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


def span_elems(dims: Sequence[int], strides: Sequence[int]) -> int:
    return 1 + sum((int(dim) - 1) * int(stride) for dim, stride in zip(dims, strides))


def safe(dtype: str, m: int, n: int, is2: int, is1: int) -> bool:
    size = DTYPE_SIZES[dtype]
    gm_span = span_elems((m, n), (is2, is1))
    ub_span = span_elems((m, n), (n, 1))
    return (gm_span + GUARD_ELEMS) * size <= MAX_GM_SPAN_BYTES and (
        ub_span + GUARD_ELEMS
    ) * size <= MAX_UB_SPAN_BYTES


def build_row(
    index: int,
    dtype: str,
    block_dim: int,
    m: int,
    n: int,
    is2: int,
    is1: int,
    execution_repeat_count: int,
) -> dict[str, str]:
    size = DTYPE_SIZES[dtype]
    gm_span = span_elems((m, n), (is2, is1))
    ub_span = span_elems((m, n), (n, 1))
    total_elems = m * n
    total_bytes = total_elems * size
    token = f"r5g_{dtype}_m{m}_n{n}_i0{is2}_i1{is1}"
    return {
        "experiment_idx": str(index),
        "token": token,
        "config_id": token,
        "round_id": "v4_round5_transpose_repeat_with_input_stride",
        "sample_id": token,
        "execution_repeat_count": str(execution_repeat_count),
        "sensitivity_id": "R5G",
        "sensitivity_key": "group_g_gm_outer_stride_sensitivity",
        "sensitivity_name": "gm_outer_stride_sensitivity",
        "stage_define": "27",
        "group_id": "G",
        "group_name": "group_g_gm_outer_stride_sensitivity",
        "fit_stage": "gm_outer_stride_sensitivity",
        "parent_model": "none",
        "model_target": "measure whether GM outer stride is2 changes cycles",
        "metric_target": "nddma_mte2_cycles_per_block/repeat",
        "scan_variable": "is2 with fixed dtype,is1,M,N",
        "controlled_variables": "block_dim=1,identical GM/UB address sets",
        "dtype": dtype,
        "dtype_size": str(size),
        "dim": "2",
        "block_dim": str(block_dim),
        "repeat": str(KERNEL_REPEAT),
        "output_dims": f"{m}x{n}",
        "input_stride": f"{is2}x{is1}",
        "output_stride": f"{n}x1",
        "input_stride_axis": "1",
        "input_stride_multiplier": str(is1),
        "output_stride_axis": "1",
        "output_stride_multiplier": "1",
        "src_offset_elem": "0",
        "dst_offset_elem": "0",
        "src_align_mod32": "0",
        "dst_align_mod32": "0",
        "inner_elems": str(n),
        "inner_bytes": str(n * size),
        "total_elems": str(total_elems),
        "total_bytes": str(total_bytes),
        "bytes_per_core": str(total_bytes),
        "logical_total_bytes": str(total_bytes * block_dim),
        "gm_span_elems": str(gm_span),
        "ub_span_elems": str(ub_span),
        "input_stride_pattern": "api_gm_two_stride_noncontiguous",
        "output_stride_pattern": "api_contiguous",
        "layout_pattern": "gm_outer_stride_sensitivity",
        "notes": "Aligned to HW_GE_ATT Round5 v4 E2E group G gm_outer_stride_sensitivity.",
        "fit_role": "fit",
        "bytes_region": "safety_filtered",
        "shape_policy": "is2_in_2_4_6_8_16_and_is2_times_M_le_is1",
        "model_family": "PAIRED_2D_LEADING_UNIT_AXIS_VS_1D",
        "data_split": "G",
        "shape_family_id": f"m{m}_n{n}_i1{is1}",
        "input_layout_policy": "api_is0_is1",
        "output_layout_policy": "api_N_1",
        "input_span_elems": str(gm_span),
        "output_span_elems": str(ub_span),
        "input_span_bytes": str(gm_span * size),
        "output_span_bytes": str(ub_span * size),
        "input_gap_elems": str(gm_span - total_elems),
        "output_gap_elems": "0",
        "input_gap_bytes": str((gm_span - total_elems) * size),
        "output_gap_bytes": "0",
        "basis_variable_count": "2",
        "basis_candidate_count": "4",
        "sampling_seed": "",
        "m": str(m),
        "n": str(n),
        "is2": str(is2),
        "is1": str(is1),
        "ng2_input_stride": str(is2),
        "inner_input_stride": str(is1),
        "s_byte_stride": str(min(is1 * size, 128)),
        "pair_role": "gm_outer_stride_sensitivity",
        "address_set_id": f"{dtype}_m{m}_n{n}_i1{is1}",
        "api_output_dims": f"{m}x{n}",
        "api_input_stride": f"{is2}x{is1}",
        "api_output_stride": f"{n}x1",
        "loop_output_dims_reversed": f"{n}x{m}",
        "loop_input_stride_reversed": f"{is1}x{is2}",
        "loop_output_stride_reversed": f"1x{n}",
    }


def build_rows(execution_repeat_count: int) -> list[dict[str, str]]:
    rows = []
    for dtype, block_dim, m, n, is2, is1 in product(
        DTYPES, BLOCK_DIMS, M_VALUES, N_VALUES, OUTER_INPUT_STRIDES, INPUT_STRIDES
    ):
        if is2 * m > is1:
            continue
        if safe(dtype, m, n, is2, is1):
            rows.append(
                build_row(
                    len(rows) + 1,
                    dtype,
                    block_dim,
                    m,
                    n,
                    is2,
                    is1,
                    execution_repeat_count,
                )
            )
    for group, strides in (("D", INPUT_STRIDES), ("F", HIGH_STRIDE_INCREMENTAL_VALUES)):
        for dtype, m, n, is1 in product(DTYPES, M_VALUES, N_VALUES, strides):
            if is1 < m or not safe(dtype, m, n, 1, is1):
                continue
            item = build_row(
                len(rows) + 1, dtype, 1, m, n, 1, is1, execution_repeat_count
            )
            token = f"r5{group.lower()}_{dtype}_m{m}_n{n}_i{is1}"
            key = (
                "group_d_2d_transpose_input_stride"
                if group == "D"
                else "group_f_high_stride_2d_transpose"
            )
            item.update(
                token=token,
                config_id=token,
                sample_id=token,
                group_id=group,
                data_split=group,
                sensitivity_id=f"R5{group}",
                sensitivity_key=key,
                sensitivity_name=key,
                group_name=key,
                stage_define="29",
                fit_stage="transpose_ng2_fit",
                fit_role="fit",
                notes="HW Round5 D/F N_G2 training point; frozen model supplies N_G1",
                shape_policy="is1_ge_M",
                pair_role="strided_transpose_repeat",
            )
            rows.append(item)
    return rows


def write_factor(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_COLUMNS, lineterminator="\n")
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


def reset_output(path: Path) -> None:
    for name in ("profiling_raw", "experiment", "analysis"):
        shutil.rmtree(path / name, ignore_errors=True)
    for name in ("command.log", "analysis.log"):
        (path / name).unlink(missing_ok=True)
    (path / "experiment").mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    if not args.dry_run:
        write_factor(factor_csv, build_rows(args.execution_repeat_count))
    if not args.dry_run:
        reset_output(output_dir)
    build_dir = output_dir / "build"
    executable = build_dir / "demo_nddma"
    configure = [
        "cmake",
        "-S",
        str(Path(args.executable_dir).resolve()),
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
    analysis_command = [
        sys.executable,
        str(ANALYSIS_SCRIPT),
        f"--profiling-dir={raw_dir}",
        f"--experiment-log={experiment_dir / 'experiment.log'}",
        f"--factor-csv={factor_csv}",
        f"--output-dir={output_dir / 'analysis'}",
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
