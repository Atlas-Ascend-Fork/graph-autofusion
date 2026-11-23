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
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DATA_COLLECTION_DIR = MODELING_DIR / "Data_collection"
HARNESS_DIR = DATA_COLLECTION_DIR / "common" / "executables" / "standalone_nddma"
ANALYSIS_SCRIPT = HARNESS_DIR / "analyze_profiling_with_params.py"
DEFAULT_ANA_DIR = MODELING_DIR / "Ana" / "round3"
DEFAULT_OUTPUT_DIR = DEFAULT_ANA_DIR / "collection"
DEFAULT_FACTOR_CSV = DEFAULT_ANA_DIR / "round3_multidim_factor.csv"

DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
DTYPES = tuple(DTYPE_SIZES)
MAX_GM_SPAN_BYTES = 4 * 1024 * 1024
MAX_UB_SPAN_BYTES = 256 * 1024
GUARD_ELEMS = 64
KERNEL_REPEAT = 200

BASE_COLUMNS = (
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
    "model_target",
    "metric_target",
    "fit_role",
    "dtype",
    "dtype_size",
    "dim",
    "block_dim",
    "repeat",
    "output_dims",
    "output_stride",
    "input_stride",
    "input_stride_axis",
    "output_stride_axis",
    "src_offset_elem",
    "dst_offset_elem",
    "inner_elems",
    "inner_bytes",
    "total_elems",
    "total_bytes",
    "bytes_per_core",
    "logical_total_bytes",
    "gm_span_elems",
    "ub_span_elems",
    "input_span_bytes",
    "output_span_bytes",
    "input_stride_pattern",
    "output_stride_pattern",
    "layout_pattern",
    "parent_model",
    "notes",
)
EXTRA_COLUMNS = ("l4", "l3", "l2", "l1", "l0", "source_round")
CSV_COLUMNS = BASE_COLUMNS + EXTRA_COLUMNS


def parse_args(
    defaults: dict | None = None, description: str | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description
        or "Generate and collect NDDMA2 Round3 2D/3D/4D/5D data."
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
    parser.set_defaults(**(defaults or {}))
    args = parser.parse_args()
    args.npu_arch = args.npu_arch.strip()
    if not args.npu_arch and args.run_mode != "cpu":
        message = "NPU_ARCH is not configured. Set it for your actual v35 device, e.g. export NPU_ARCH=dav-3510, or pass --npu-arch dav-3510 to collect.py / run_nddma_ad_hoc.py. dav-3510 is an example, not a default for every device. e2e.py also accepts --npu-arch and forwards it to collection."
        if args.dry_run:
            print(f"[WARN] {message}")
        else:
            parser.error(message)
    return args


def product_int(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def span_elems(dims: Sequence[int], strides: Sequence[int]) -> int:
    return 1 + sum((int(dim) - 1) * int(stride) for dim, stride in zip(dims, strides))


def safe(
    dtype: str,
    dims: Sequence[int],
    input_stride: Sequence[int],
    output_stride: Sequence[int],
) -> bool:
    size = DTYPE_SIZES[dtype]
    return (
        span_elems(dims, input_stride) + GUARD_ELEMS
    ) * size <= MAX_GM_SPAN_BYTES and (
        span_elems(dims, output_stride) + GUARD_ELEMS
    ) * size <= MAX_UB_SPAN_BYTES


def continuous_stride(loop_sizes: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        product_int(loop_sizes[:axis]) for axis in range(len(loop_sizes) - 1, -1, -1)
    )


def layouts_3d(
    m: int, n: int, k: int
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    ub = (n * k, k, 1)
    return {
        "continuous": ((n * k, n, 1), (n * k, n, 1)),
        "transpose_dim2": ((k, m * k, 1), ub),
        "transpose_dim1_dim3": ((1, m, n * m), ub),
        "transpose_3d": ((1, m, k * m), ub),
    }


def layouts_4d(
    l0: int, l1: int, l2: int, l3: int
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    output = (l2 * l1 * l0, l1 * l0, l0, 1)
    return {
        "continuous": (output, output),
        "transpose_dim1_dim4": ((1, l1 * l3, l3, l1 * l2 * l3), output),
        "transpose_dim3_dim4": ((l1 * l0, l1 * l0 * l3, l0, 1), output),
    }


def layouts_5d(
    l0: int, l1: int, l2: int, l3: int, l4: int
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    output = continuous_stride((l0, l1, l2, l3, l4))
    return {
        "continuous": (output, output),
        "transpose_dim1_dim5": (
            (1, l4 * l2 * l1, l4 * l1, l4, l4 * l1 * l2 * l3),
            output,
        ),
        "transpose_dim1_dim2": (
            (l3 * l2 * l1 * l0, l2 * l1 * l0, l1 * l0, 1, l1),
            output,
        ),
    }


def make_row(
    index: int,
    dtype: str,
    dims: Sequence[int],
    input_stride: Sequence[int],
    output_stride: Sequence[int],
    layout: str,
    source_round: str,
    execution_repeat_count: int,
) -> dict[str, str]:
    size = DTYPE_SIZES[dtype]
    dim = len(dims)
    loop_sizes = tuple(reversed(dims))
    total_elems = product_int(dims)
    total_bytes = total_elems * size
    token = f"r3_{source_round}_d{dim}_{layout}_{dtype}_" + "x".join(
        str(value) for value in dims
    )
    token += (
        "_i"
        + "x".join(map(str, input_stride))
        + "_o"
        + "x".join(map(str, output_stride))
    )
    row = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "experiment_idx": str(index),
            "token": token,
            "config_id": token,
            "round_id": "r3_multidim",
            "sample_id": token,
            "execution_repeat_count": str(execution_repeat_count),
            "sensitivity_id": f"R3D{dim}",
            "sensitivity_key": layout,
            "sensitivity_name": layout,
            "stage_define": "31",
            "group_id": f"D{dim}",
            "group_name": f"{dim}d_{layout}",
            "model_target": "NDDMA multidimensional extension",
            "metric_target": "nddma_mte2_cycles_per_block",
            "fit_role": "validation",
            "dtype": dtype,
            "dtype_size": str(size),
            "dim": str(dim),
            "block_dim": "1",
            "repeat": str(KERNEL_REPEAT),
            "output_dims": "x".join(str(value) for value in dims),
            "input_stride": "x".join(str(value) for value in input_stride),
            "output_stride": "x".join(str(value) for value in output_stride),
            "input_stride_axis": ",".join(str(axis) for axis in range(dim)),
            "output_stride_axis": ",".join(str(axis) for axis in range(dim)),
            "src_offset_elem": "0",
            "dst_offset_elem": "0",
            "inner_elems": str(loop_sizes[0]),
            "inner_bytes": str(loop_sizes[0] * size),
            "total_elems": str(total_elems),
            "total_bytes": str(total_bytes),
            "bytes_per_core": str(total_bytes),
            "logical_total_bytes": str(total_bytes),
            "gm_span_elems": str(span_elems(dims, input_stride)),
            "ub_span_elems": str(span_elems(dims, output_stride)),
            "input_span_bytes": str(span_elems(dims, input_stride) * size),
            "output_span_bytes": str(span_elems(dims, output_stride) * size),
            "input_stride_pattern": layout,
            "output_stride_pattern": layout if layout == "continuous" else "continuous",
            "layout_pattern": layout,
            "parent_model": "DOC/round3_multidim_model.md",
            "notes": "Round3 multidimensional extension dataset migrated from legacy NDDMA round6/round7 plus 2D transpose.",
            "source_round": source_round,
        }
    )
    padded = ("", "", "", "", "") + tuple(str(value) for value in loop_sizes)
    row.update(
        {
            "l4": padded[-5],
            "l3": padded[-4],
            "l2": padded[-3],
            "l1": padded[-2],
            "l0": padded[-1],
        }
    )
    return row


def build_rows(execution_repeat_count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dtype in DTYPES:
        # HW Round4 D/E: transpose descriptors with byte-based GM padding/UB gaps.
        for outer, inner, padding, gap in product(
            (4, 8, 16, 32, 64, 128, 153, 167, 224),
            (4, 8, 16, 32, 64, 128, 153, 167),
            (0, 32, 128, 512, 2048, 8192),
            (0, 16, 64, 256, 512),
        ):
            dims = (outer, inner)
            gm = (1, outer + padding // DTYPE_SIZES[dtype])
            ub = (inner + gap // DTYPE_SIZES[dtype], 1)
            layout = (
                "group_d_2d_gm_noncontig_ub_contig"
                if gap == 0
                else "group_e_2d_gm_noncontig_ub_noncontig"
            )
            if safe(dtype, dims, gm, ub):
                rows.append(
                    make_row(
                        len(rows) + 1,
                        dtype,
                        dims,
                        gm,
                        ub,
                        layout,
                        "round4_2d",
                        execution_repeat_count,
                    )
                )
        # HW Round5 J, also used by the Round7 3D/4D/5D combined evaluation.
        for outer, middle, inner, is2, is1 in product(
            (2, 3, 4, 8, 16, 32),
            (2, 3, 4, 8, 16, 32, 64, 128, 156),
            (8, 16, 32, 64, 128, 512, 1024, 2048),
            (1, 2, 4, 8, 16, 32, 64, 128),
            (2, 4, 8, 16, 32, 64, 128, 160, 512, 3333),
        ):
            dims, gm, ub = (
                (outer, middle, inner),
                (1, is2, is1),
                (middle * inner, inner, 1),
            )
            if (
                outer <= is2
                and (outer - 1) + (middle - 1) * is2 < is1
                and safe(dtype, dims, gm, ub)
            ):
                rows.append(
                    make_row(
                        len(rows) + 1,
                        dtype,
                        dims,
                        gm,
                        ub,
                        "group_j_3d_parameter_inheritance",
                        "round5_j",
                        execution_repeat_count,
                    )
                )
        for m, n, k in product(
            (2, 4, 8, 16, 32), (4, 8, 16, 32, 64), (4, 8, 16, 32, 64, 128)
        ):
            dims = (m, n, k)
            for layout, (gm, ub) in layouts_3d(m, n, k).items():
                if safe(dtype, dims, gm, ub):
                    rows.append(
                        make_row(
                            len(rows) + 1,
                            dtype,
                            dims,
                            gm,
                            ub,
                            layout,
                            "round6",
                            execution_repeat_count,
                        )
                    )
        for m, n, k, stride in product(
            (2, 4, 8, 16, 32),
            (4, 8, 16, 32, 64),
            (4, 8, 16, 32, 64, 128),
            (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
        ):
            dims, gm, ub = (
                (m, n, k),
                (n * k * stride, k * stride, stride),
                (n * k, k, 1),
            )
            if safe(dtype, dims, gm, ub):
                rows.append(
                    make_row(
                        len(rows) + 1,
                        dtype,
                        dims,
                        gm,
                        ub,
                        "gm_noncontiguous",
                        "round6",
                        execution_repeat_count,
                    )
                )
        for l3, l2, l1, l0 in product(
            (2, 4, 8, 16), (2, 4, 8, 16), (4, 8, 16, 32), (4, 8, 16, 32, 64, 128)
        ):
            dims = (l3, l2, l1, l0)
            for layout, (gm, ub) in layouts_4d(l0, l1, l2, l3).items():
                if safe(dtype, dims, gm, ub):
                    rows.append(
                        make_row(
                            len(rows) + 1,
                            dtype,
                            dims,
                            gm,
                            ub,
                            layout,
                            "round7",
                            execution_repeat_count,
                        )
                    )
        for l4, l3, l2, l1, l0 in product(
            (2, 4, 8, 16),
            (2, 4, 8, 16),
            (2, 4, 8, 16),
            (4, 8, 16, 32),
            (4, 8, 16, 32, 64, 128),
        ):
            dims = (l4, l3, l2, l1, l0)
            for layout, (gm, ub) in layouts_5d(l0, l1, l2, l3, l4).items():
                if safe(dtype, dims, gm, ub):
                    rows.append(
                        make_row(
                            len(rows) + 1,
                            dtype,
                            dims,
                            gm,
                            ub,
                            layout,
                            "round7",
                            execution_repeat_count,
                        )
                    )
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
    return run_collection(parse_args())


def run_collection(
    args: argparse.Namespace, rows: list[dict[str, str]] | None = None
) -> int:
    """Run the shared harness with the default grid or explicitly supplied rows."""
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
    rows = build_rows(args.execution_repeat_count) if rows is None else rows
    if not args.dry_run:
        write_factor(factor_csv, rows)
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
