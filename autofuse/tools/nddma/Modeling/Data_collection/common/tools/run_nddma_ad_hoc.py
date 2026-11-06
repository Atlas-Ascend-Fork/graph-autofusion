#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run one ad-hoc NDDMA config and print the measured cycles."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
EXECUTABLE_SOURCE = COMMON_DIR / "executables" / "standalone_nddma"
PROFILE_ANALYZER = EXECUTABLE_SOURCE / "analyze_profiling_with_params.py"
DEFAULT_BASE_MODEL = (
    REPO_ROOT / "Modeling" / "Ana" / "round1" / "round1_1d_single_multi_core_model.json"
)
BUILD_DIR_NAME = "standalone_nddma"
EXECUTABLE_NAME = "demo_nddma"
DEFAULT_NPU_ARCH = os.environ.get("NPU_ARCH", "")
DTYPE_SIZE = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
DTYPE_ALIASES = {
    "int8_t": "int8_t",
    "int8": "int8_t",
    "i8": "int8_t",
    "b8": "int8_t",
    "int16_t": "int16_t",
    "int16": "int16_t",
    "i16": "int16_t",
    "b16": "int16_t",
    "int32_t": "int32_t",
    "int32": "int32_t",
    "i32": "int32_t",
    "b32": "int32_t",
    "int64_t": "int64_t",
    "int64": "int64_t",
    "i64": "int64_t",
    "b64": "int64_t",
}
CSV_COLUMNS = [
    "experiment_idx",
    "token",
    "round_id",
    "sample_id",
    "execution_repeat_index",
    "execution_repeat_count",
    "sensitivity_id",
    "sensitivity_key",
    "sensitivity_name",
    "stage_define",
    "group_id",
    "group_name",
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
    "config_id",
    "fit_stage",
    "parent_model",
]
GUARD_ELEMS = 64
MAX_UB_SPAN_BYTES = 256 * 1024


def parse_vector(text: str) -> list[int]:
    values = [
        int(part)
        for part in text.replace(",", "x").replace(";", "x").split("x")
        if part
    ]
    if not values:
        raise argparse.ArgumentTypeError("vector must not be empty")
    return values


def parse_positive_vector(text: str) -> list[int]:
    values = parse_vector(text)
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all vector entries must be positive")
    return values


def parse_non_negative_vector(text: str) -> list[int]:
    values = parse_vector(text)
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("all vector entries must be non-negative")
    return values


def join_vector(values: Sequence[int]) -> str:
    return "x".join(str(value) for value in values)


def parse_dtype(text: str) -> str:
    key = text.strip().lower()
    if key in DTYPE_ALIASES:
        return DTYPE_ALIASES[key]
    supported = ", ".join(sorted(DTYPE_ALIASES))
    raise argparse.ArgumentTypeError(
        f"unsupported dtype {text!r}; supported: {supported}"
    )


def product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def load_base_model(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file_obj:
        model = json.load(file_obj)
    parameters = model.get("parameters", {})
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(f"base model has no parameters: {path}")
    return model


def span_elems(dims: Sequence[int], strides: Sequence[int]) -> int:
    if any(int(dim) == 0 for dim in dims):
        return 0
    return 1 + sum((int(dim) - 1) * int(stride) for dim, stride in zip(dims, strides))


def resolve_program(value: str) -> str:
    found = shutil.which(value)
    if found:
        return found
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    raise ValueError(f"program not found: {value}")


def run(command: Sequence[str], *, dry_run: bool, log_path: Path | None = None) -> int:
    text = shlex.join([str(part) for part in command])
    print(f"[INFO] {text}", flush=True)
    if dry_run:
        return 0
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    if log_path is not None:
        log_path.write_text(
            "COMMAND\n"
            + text
            + "\n\nSTDOUT\n"
            + completed.stdout
            + "\n\nSTDERR\n"
            + completed.stderr,
            encoding="utf-8",
        )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode and completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return completed.returncode


def reset_run_outputs(output_dir: Path, result_file: str) -> None:
    for child in ("experiment", "profiling_raw", "analysis"):
        target = output_dir / child
        if target.exists():
            shutil.rmtree(target)
    for child in (
        "configure.log",
        "build.log",
        "profile.log",
        "analyze_profile.log",
        result_file,
    ):
        target = output_dir / child
        if target.exists():
            target.unlink()


def op_summary_files(profiling_dir: Path) -> list[Path]:
    def is_op_summary(path: Path) -> bool:
        name = path.name.lower()
        return path.suffix.lower() == ".csv" and (
            name == "op_summary.csv" or name.startswith("op_summary_")
        )

    return sorted(
        (path for path in profiling_dir.rglob("*.csv") if is_op_summary(path)),
        key=lambda path: path.relative_to(profiling_dir).as_posix(),
    )


def sample_profiler_csvs(profiling_dir: Path, limit: int = 20) -> list[Path]:
    return sorted(
        profiling_dir.rglob("*.csv"),
        key=lambda path: path.relative_to(profiling_dir).as_posix(),
    )[:limit]


def build_msprof_command(
    msprof: str,
    profiling_dir: Path,
    application: Sequence[str],
    args: argparse.Namespace,
) -> list[str]:
    return [
        msprof,
        f"--output={profiling_dir}",
        *args.msprof_extra_arg,
        f"--application={shlex.join(application)}",
    ]


def build_row(args: argparse.Namespace) -> dict[str, str]:
    dims = args.output_dims
    input_stride = args.input_stride
    output_stride = args.output_stride
    dim = len(dims)
    if len(input_stride) != dim or len(output_stride) != dim:
        raise ValueError(
            "output_dims, input_stride, and output_stride must have the same length"
        )
    if dim < 1 or dim > 5:
        raise ValueError("standalone_nddma supports dim in [1, 5]")
    if any(int(value) <= 0 for value in dims):
        raise ValueError("output_dims values must be positive")
    if any(int(value) < 0 for value in input_stride):
        raise ValueError("input_stride values must be non-negative")
    if any(int(value) < 0 for value in output_stride):
        raise ValueError("output_stride values must be non-negative")
    dtype_size = DTYPE_SIZE[args.dtype]
    total_elems = product(dims)
    bytes_per_core = total_elems * dtype_size
    gm_span = span_elems(dims, input_stride)
    ub_span = span_elems(dims, output_stride)
    ub_required_bytes = (ub_span + GUARD_ELEMS) * dtype_size
    if ub_required_bytes > MAX_UB_SPAN_BYTES:
        raise ValueError(
            f"UB span exceeds safe default: {(ub_span + GUARD_ELEMS)} elems, "
            f"{ub_required_bytes} bytes > {MAX_UB_SPAN_BYTES}"
        )
    token = (
        f"adhoc_{args.dtype}_d{dim}_k{args.block_dim}"
        f"_shape{join_vector(dims)}_in{join_vector(input_stride)}"
        f"_out{join_vector(output_stride)}"
    )
    row = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "experiment_idx": "0",
            "token": token,
            "config_id": token,
            "round_id": "adhoc_nddma",
            "sample_id": token,
            "execution_repeat_index": "0",
            "execution_repeat_count": str(args.execution_repeat_count),
            "sensitivity_id": "ADHOC",
            "sensitivity_key": "adhoc_nddma",
            "sensitivity_name": "ad-hoc NDDMA measurement",
            "stage_define": str(args.stage),
            "group_id": "ADHOC",
            "group_name": "adhoc_nddma",
            "model_target": "ad-hoc measurement compared with Ana/round1 d/T+H base model",
            "metric_target": (
                "nddma_mte2_cycles_per_block/repeat"
                if args.cycle_output == "per-repeat"
                else "nddma_mte2_cycles_per_block"
            ),
            "scan_variable": "output_dims,input_stride,output_stride,block_dim",
            "controlled_variables": (
                f"dtype={args.dtype},cycle_output={args.cycle_output}"
            ),
            "dtype": args.dtype,
            "dtype_size": str(dtype_size),
            "dim": str(dim),
            "block_dim": str(args.block_dim),
            "repeat": str(args.kernel_repeat),
            "output_dims": join_vector(dims),
            "output_stride": join_vector(output_stride),
            "input_stride": join_vector(input_stride),
            "input_stride_axis": ",".join(str(axis) for axis in range(dim)),
            "input_stride_multiplier": join_vector(input_stride),
            "output_stride_axis": ",".join(str(axis) for axis in range(dim)),
            "output_stride_multiplier": join_vector(output_stride),
            "src_offset_elem": str(args.src_offset_elem),
            "dst_offset_elem": str(args.dst_offset_elem),
            "src_align_mod32": str((args.src_offset_elem * dtype_size) % 32),
            "dst_align_mod32": str((args.dst_offset_elem * dtype_size) % 32),
            "inner_elems": str(dims[-1]),
            "inner_bytes": str(dims[-1] * dtype_size),
            "total_elems": str(total_elems),
            "total_bytes": str(bytes_per_core),
            "bytes_per_core": str(bytes_per_core),
            "logical_total_bytes": str(bytes_per_core * args.block_dim),
            "gm_span_elems": str(gm_span),
            "ub_span_elems": str(ub_span),
            "input_stride_pattern": "explicit",
            "output_stride_pattern": "explicit",
            "layout_pattern": "adhoc_explicit",
            "notes": "Generated by common/scripts/run_nddma_ad_hoc.py.",
            "fit_role": "ad_hoc_measurement",
            "bytes_region": "manual_safe",
            "shape_policy": "requested_exact",
            "model_family": "ADHOC_NDDMA_ACTUAL_CYCLES",
            "data_split": "ADHOC",
            "shape_family_id": f"d{dim}_{join_vector(dims)}",
            "input_layout_policy": "explicit_api_order_stride",
            "output_layout_policy": "explicit_api_order_stride",
            "input_span_elems": str(gm_span),
            "output_span_elems": str(ub_span),
            "input_span_bytes": str(gm_span * dtype_size),
            "output_span_bytes": str(ub_span * dtype_size),
            "input_gap_elems": str(gm_span - total_elems),
            "output_gap_elems": str(ub_span - total_elems),
            "input_gap_bytes": str((gm_span - total_elems) * dtype_size),
            "output_gap_bytes": str((ub_span - total_elems) * dtype_size),
            "basis_variable_count": "0",
            "basis_candidate_count": "0",
            "sampling_seed": "",
            "fit_stage": "ad_hoc",
            "parent_model": str(args.base_model),
        }
    )
    return row


def write_factor_csv(path: Path, row: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_COLUMNS})


def read_single_mean(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one mean row in {path}, found {len(rows)}")
    return rows[0]


def numeric(row: Mapping[str, str], field: str) -> float:
    text = str(row.get(field, "")).strip()
    if not text or text.upper() == "N/A":
        raise ValueError(f"missing numeric field {field!r}")
    return float(text)


def run_pipeline(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    result_file = f"{args.result_name}.{args.result_suffix}"
    try:
        base_model = load_base_model(args.base_model.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] unable to load base model: {error}")
        return 2
    factor_csv = output_dir / "factor.csv"
    experiment_dir = output_dir / "experiment"
    profiling_dir = output_dir / "profiling_raw"
    analysis_dir = output_dir / "analysis"
    if not args.dry_run and not args.generate_only and not args.keep_output:
        reset_run_outputs(output_dir, result_file)
    row = build_row(args)
    if not args.dry_run:
        write_factor_csv(factor_csv, row)
    print(f"[INFO] factor CSV: {factor_csv}")
    print(
        "[INFO] config: "
        + json.dumps(
            {
                "token": row["token"],
                "dtype": row["dtype"],
                "dim": row["dim"],
                "block_dim": row["block_dim"],
                "kernel_repeat": row["repeat"],
                "cycle_output": args.cycle_output,
                "output_dims": row["output_dims"],
                "input_stride": row["input_stride"],
                "output_stride": row["output_stride"],
                "bytes_per_core": row["bytes_per_core"],
                "logical_total_bytes": row["logical_total_bytes"],
                "gm_span_elems": row["gm_span_elems"],
                "ub_span_elems": row["ub_span_elems"],
            },
            sort_keys=True,
        )
    )
    if args.generate_only:
        return 0
    build_dir = output_dir / "build" / BUILD_DIR_NAME
    executable = args.exe.resolve() if args.exe else build_dir / EXECUTABLE_NAME
    if not args.exe:
        configure = [
            "cmake",
            "-S",
            str(args.executable_source.resolve()),
            "-B",
            str(build_dir),
            f"-DNDDMA_STAGE={args.stage}",
            f"-DNPU_ARCH={args.npu_arch}",
        ]
        if args.run_mode:
            configure.append(f"-DRUN_MODE={args.run_mode}")
        configure.extend(args.cmake_extra_arg)
        status = run(
            configure, dry_run=args.dry_run, log_path=output_dir / "configure.log"
        )
        if status:
            return status
        status = run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                EXECUTABLE_NAME,
                "-j",
                str(args.jobs),
            ],
            dry_run=args.dry_run,
            log_path=output_dir / "build.log",
        )
        if status:
            return status
    elif not args.dry_run and not executable.is_file():
        print(f"[ERROR] executable not found: {executable}")
        return 2
    msprof = args.msprof_bin or "msprof"
    if not args.dry_run:
        try:
            msprof = resolve_program(msprof)
        except ValueError as error:
            print(f"[ERROR] {error}")
            return 2
    if not args.dry_run:
        experiment_dir.mkdir(parents=True, exist_ok=True)
    application = [
        str(executable.resolve()) if not args.dry_run else str(executable),
        f"--factor-csv={factor_csv}",
        f"--output-dir={experiment_dir}",
    ]
    status = run(
        build_msprof_command(msprof, profiling_dir, application, args),
        dry_run=args.dry_run,
        log_path=output_dir / "profile.log",
    )
    if status:
        return status
    experiment_log = experiment_dir / "experiment.log"
    if not args.dry_run and (
        not experiment_log.exists() or experiment_log.stat().st_size == 0
    ):
        print(
            f"[ERROR] profiling application did not produce a non-empty experiment log: {experiment_log}"
        )
        print(f"[ERROR] inspect profiling log: {output_dir / 'profile.log'}")
        return 2
    summary_files = [] if args.dry_run else op_summary_files(profiling_dir)
    if not args.dry_run and not summary_files:
        print(f"[ERROR] msprof produced no op_summary CSV under: {profiling_dir}")
        csv_samples = sample_profiler_csvs(profiling_dir)
        if csv_samples:
            print("[ERROR] profiler CSV files found, but none looked like op_summary:")
            for path in csv_samples:
                print(f"[ERROR]   {path.relative_to(profiling_dir)}")
            print(
                "[ERROR] This ad-hoc runner matches the Round2-style msprof "
                "invocation: msprof --output=... --application=... . Only "
                "API-level CSVs usually mean the profiler did not capture "
                "task-level op_summary data for this run; pass site-specific "
                "msprof switches with --msprof-extra-arg=<flag> if your "
                "environment requires them."
            )
        else:
            print("[ERROR] no profiler CSV files were found under profiling_raw.")
        print(f"[ERROR] inspect profiling log: {output_dir / 'profile.log'}")
        return 2
    analyzer = args.profile_analyzer.resolve()
    if not args.dry_run and not analyzer.is_file():
        print(f"[ERROR] profile analyzer not found: {analyzer}")
        return 2
    status = run(
        [
            sys.executable,
            str(analyzer),
            f"--profiling-dir={profiling_dir}",
            f"--experiment-log={experiment_log}",
            f"--factor-csv={factor_csv}",
            f"--output-dir={analysis_dir}",
            f"--pipes={args.analysis_pipes}",
            *(
                [f"--op-name-filter={args.op_name_filter}"]
                if args.op_name_filter
                else []
            ),
        ],
        dry_run=args.dry_run,
        log_path=output_dir / "analyze_profile.log",
    )
    if status or args.dry_run:
        return status
    mean_row = read_single_mean(analysis_dir / "profiling_with_params_mean.csv")
    raw_cycles = numeric(mean_row, "nddma_mte2_cycles_per_block")
    kernel_repeat = numeric(mean_row, "repeat")
    per_repeat_cycles = raw_cycles / kernel_repeat
    actual_cycles = (
        per_repeat_cycles if args.cycle_output == "per-repeat" else raw_cycles
    )
    result = {
        "token": mean_row.get("token", row["token"]),
        "metric": (
            "nddma_mte2_cycles_per_block/repeat"
            if args.cycle_output == "per-repeat"
            else "nddma_mte2_cycles_per_block"
        ),
        "actual_cycles": actual_cycles,
        "actual_cycles_total_kernel_repeat": raw_cycles,
        "actual_cycles_per_kernel_repeat": per_repeat_cycles,
        "raw_nddma_mte2_cycles_per_block": raw_cycles,
        "kernel_repeat": kernel_repeat,
        "cycle_output": args.cycle_output,
        "base_model": base_model.get("model", ""),
        "profiling_mean_sample_count": mean_row.get("profiling_mean_sample_count", ""),
        "analysis_csv": str(analysis_dir / "profiling_with_params_mean.csv"),
    }
    (output_dir / result_file).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    print(f"[RESULT] actual_cycles={actual_cycles:.12g}")
    print(f"[RESULT] actual_cycles_total_kernel_repeat={raw_cycles:.12g}")
    print(f"[RESULT] actual_cycles_per_kernel_repeat={per_repeat_cycles:.12g}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one standalone_nddma ad-hoc measurement. Vectors are in "
            "NDDMA API order, for example: 63x2 2x1 3x1 72 b64."
        )
    )
    parser.add_argument("output_dims", type=parse_positive_vector)
    parser.add_argument("input_stride", type=parse_non_negative_vector)
    parser.add_argument("output_stride", type=parse_non_negative_vector)
    parser.add_argument("block_dim", type=int)
    parser.add_argument(
        "dtype_positional",
        nargs="?",
        type=parse_dtype,
        help="Optional dtype positional argument: b8/b16/b32/b64 or int*_t.",
    )
    parser.add_argument(
        "--dtype",
        type=parse_dtype,
        default=None,
        help="Dtype used when the optional positional dtype is omitted.",
    )
    parser.add_argument("--kernel-repeat", type=int, default=200)
    parser.add_argument(
        "--cycle-output",
        choices=("total", "per-repeat"),
        default="total",
        help=(
            "Which value to expose as actual_cycles. 'total' keeps the "
            "measured cycles for the whole in-kernel repeat loop; "
            "'per-repeat' divides by --kernel-repeat."
        ),
    )
    parser.add_argument("--execution-repeat-count", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path, default=COMMON_DIR / "results_analysis" / "ad_hoc"
    )
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--executable-source", type=Path, default=EXECUTABLE_SOURCE)
    parser.add_argument("--profile-analyzer", type=Path, default=PROFILE_ANALYZER)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=DEFAULT_BASE_MODEL,
        help="Ana/round1 JSON providing the one-dimensional d/T+H base-model parameters.",
    )
    parser.add_argument(
        "--msprof-bin",
        default=os.environ.get("MSPROF_BIN") or shutil.which("msprof") or "msprof",
        help="msprof executable; defaults to MSPROF_BIN or the msprof found on PATH.",
    )
    parser.add_argument("--analysis-pipes", default="mte2")
    parser.add_argument(
        "--op-name-filter",
        default="",
        help=(
            "Case-insensitive Op Name substring for profiling rows. "
            "Default keeps all op_summary rows, matching the Round6 E2E behavior."
        ),
    )
    parser.add_argument(
        "--msprof-extra-arg",
        action="append",
        default=[],
        help=(
            "Extra argument passed to msprof before --application. For values "
            "that start with '-', use --msprof-extra-arg=<flag>. May repeat."
        ),
    )
    parser.add_argument("--stage", type=int, default=99)
    parser.add_argument(
        "--npu-arch",
        default=DEFAULT_NPU_ARCH,
        help=(
            "Ascend NPU architecture passed to CMake. "
            "Defaults to NPU_ARCH from the environment; required for device collection."
        ),
    )
    parser.add_argument("--run-mode", default="")
    parser.add_argument("--cmake-extra-arg", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--src-offset-elem", type=int, default=0)
    parser.add_argument("--dst-offset-elem", type=int, default=0)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--result-name",
        default="result",
        help="Basename of the ad-hoc measurement JSON, written as <result-name>.<result-suffix>.",
    )
    parser.add_argument(
        "--result-suffix",
        default="json",
        help="Suffix of the ad-hoc measurement JSON.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Do not remove old experiment/profiling_raw/analysis outputs before a real run.",
    )
    args = parser.parse_args()
    args.npu_arch = args.npu_arch.strip()
    if (
        not args.npu_arch
        and args.run_mode != "cpu"
        and not args.generate_only
        and not args.exe
    ):
        message = "NPU_ARCH is not configured. Set it for your actual v35 device, e.g. export NPU_ARCH=dav-3510, or pass --npu-arch dav-3510 to collect.py / run_nddma_ad_hoc.py. dav-3510 is an example, not a default for every device. e2e.py also accepts --npu-arch and forwards it to collection."
        if args.dry_run:
            print(f"[WARN] {message}")
        else:
            parser.error(message)
    if args.block_dim <= 0:
        parser.error("block_dim must be positive")
    if args.kernel_repeat <= 0:
        parser.error("--kernel-repeat must be positive")
    if args.execution_repeat_count <= 0:
        parser.error("--execution-repeat-count must be positive")
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.src_offset_elem < 0 or args.dst_offset_elem < 0:
        parser.error("offsets must be non-negative")
    if args.dtype_positional and args.dtype and args.dtype_positional != args.dtype:
        parser.error("positional dtype and --dtype disagree")
    args.dtype = args.dtype_positional or args.dtype or "int64_t"
    return args


def main() -> int:
    try:
        return run_pipeline(parse_args())
    except (csv.Error, OSError, TypeError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
