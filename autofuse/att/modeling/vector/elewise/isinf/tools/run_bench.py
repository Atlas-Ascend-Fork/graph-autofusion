#!/usr/bin/env python3
"""
IsInf perf bench 运行工具：编译、从 cases.csv 读取用例、批量运行、采集数据到 samples.csv。

npu 模式经 msprof --application 开启 profiling（--parse=on），每用例测试完后在其 profiling
子目录递归查找 op_summary_*.csv，按
    aiv_total_cycles / Block Num * aiv_vec_ratio
计算本用例实测值（多次运行取中位数），刷新 samples.csv 的 measured_cycles。
cpu 模式仅做正确性校验（VERIFY），不采集性能数据，measured_cycles 留空。

samples.csv 与 cases.csv 用例逐行一致，追加两列：
- measured_cycles：仅 npu 模式填写（来自 op_summary 解析）
- model_predict_cycles：初始留空，待拟合出 a、b 并完成 perf 计算代码后由后续工具回填
"""

import argparse
import csv
import datetime
import glob
import os
import re
import shutil
import statistics
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_NAME = "isinf_perf_bench"
API_NAME = "isinf"

COMBOS = {
    "bool_float": 64,
    "bool_half": 128,
}
RUNS_PER_CASE = 5

VERIFY_RE = re.compile(r"^VERIFY,(?P<status>PASS|FAILED)")


def sh(cmd, cwd=None):
    print("+ {}".format(" ".join(cmd)))
    proc = subprocess.run(
        cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError("command failed: {}".format(" ".join(cmd)))
    return proc.stdout


def build(run_mode, arch, jobs):
    if not os.environ.get("ASCEND_HOME_PATH"):
        raise RuntimeError(
            "ASCEND_HOME_PATH is not set, run: source ${ASCEND_HOME_PATH}/set_env.sh"
        )
    build_dir = "build" if run_mode == "npu" else "build_cpu"
    build_dir = os.path.join(SCRIPT_DIR, build_dir)
    os.makedirs(build_dir, exist_ok=True)
    sh(
        [
            "cmake",
            "-DCMAKE_ASC_RUN_MODE={}".format(run_mode),
            "-DCMAKE_ASC_ARCHITECTURES={}".format(arch),
            "..",
        ],
        cwd=build_dir,
    )
    sh(["make", "-j{}".format(jobs)], cwd=build_dir)
    return os.path.join(build_dir, BENCH_NAME)


def check_device_env(arch):
    """npu 模式环境自检：npu-smi 可用时打印设备信息（环境证据）；不存在时默认给定 arch。"""
    npu_smi = shutil.which("npu-smi")
    if npu_smi is None:
        print("npu-smi not found, default arch = {}".format(arch))
        return
    try:
        proc = subprocess.run(
            [npu_smi, "info"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print("npu-smi info timeout, default arch = {}".format(arch))
        return
    if proc.returncode != 0:
        print("npu-smi info failed, default arch = {}".format(arch))
        return
    print("device info (npu-smi info):")
    for line in proc.stdout.splitlines()[:10]:
        if line.strip():
            print("  | {}".format(line.rstrip()))


def load_cases(path):
    """读取用例文件，列：combo,count,repeat,loop_count；repeat 必须与 ceil(count/repeat_elm) 一致。"""
    cases = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for field in ("combo", "count", "repeat", "loop_count"):
            if field not in fields:
                raise SystemExit("cases file {} missing column: {}".format(path, field))
        for line_no, row in enumerate(reader, start=2):
            combo = (row.get("combo") or "").strip()
            count_text = (row.get("count") or "").strip()
            repeat_text = (row.get("repeat") or "").strip()
            loop_text = (row.get("loop_count") or "").strip()
            if not combo and not count_text:
                continue
            if combo not in COMBOS:
                raise SystemExit(
                    "cases file {} line {}: unknown combo: {}".format(
                        path, line_no, combo
                    )
                )
            if not count_text or int(count_text) <= 0:
                raise SystemExit(
                    "cases file {} line {}: invalid count: {}".format(
                        path, line_no, count_text
                    )
                )
            if not repeat_text or int(repeat_text) <= 0:
                raise SystemExit(
                    "cases file {} line {}: invalid repeat: {}".format(
                        path, line_no, repeat_text
                    )
                )
            if not loop_text or int(loop_text) <= 0:
                raise SystemExit(
                    "cases file {} line {}: invalid loop_count: {}".format(
                        path, line_no, loop_text
                    )
                )
            count = int(count_text)
            repeat = int(repeat_text)
            repeat_elm = COMBOS[combo]
            expected_repeat = (count + repeat_elm - 1) // repeat_elm
            if repeat != expected_repeat:
                raise SystemExit(
                    "cases file {} line {}: repeat {} mismatch, count {} with repeat_elm {} expects {}".format(
                        path, line_no, repeat, count, repeat_elm, expected_repeat
                    )
                )
            cases.append(
                {
                    "combo": combo,
                    "count": count,
                    "repeat": repeat,
                    "loop_count": int(loop_text),
                }
            )
    if not cases:
        raise SystemExit("cases file {} has no case".format(path))
    return cases


def run_one(exe, combo, count, loop_count, run_mode, profile_dir=None):
    """执行单个用例；npu 模式经 msprof --application 包裹并开启 --parse 以产出 op_summary。"""
    if run_mode == "npu":
        app_cmd = "{} {} {} {}".format(exe, combo, count, loop_count)
        cmd = ["msprof", "--parse=on", "--application={}".format(app_cmd)]
        if profile_dir:
            os.makedirs(profile_dir, exist_ok=True)
            cmd.append("--output={}".format(profile_dir))
    else:
        cmd = [exe, combo, str(count), str(loop_count)]
    proc = subprocess.run(
        cmd,
        cwd=os.path.dirname(exe),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    verify = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        match_verify = VERIFY_RE.match(line)
        if match_verify:
            verify = match_verify.group("status")
    if proc.returncode != 0 and verify is None:
        print("  bench exited with {}: {}".format(proc.returncode, proc.stdout[-500:]))
        return None
    return verify


def parse_ratio(text):
    """解析 aiv_vec_ratio：兼容 '0.85' 与 '85%' 两种格式，统一为 0~1 小数。"""
    text = text.strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def parse_case_profiling(case_dir):
    """递归查找用例 profiling 目录下的 op_summary_*.csv，返回各次运行的实测值列表。

    实测值 = aiv_total_cycles / Block Num * aiv_vec_ratio，取 OP Type 包含被测 API 名称
    （API_NAME，忽略大小写）的行。
    """
    values = []
    for path in sorted(
        glob.glob(os.path.join(case_dir, "**", "op_summary_*.csv"), recursive=True)
    ):
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            missing = [
                field
                for field in (
                    "OP Type",
                    "Block Num",
                    "aiv_total_cycles",
                    "aiv_vec_ratio",
                )
                if field not in fields
            ]
            if missing:
                print(
                    "  warn: {} missing columns: {}, skipped".format(
                        path, ",".join(missing)
                    )
                )
                continue
            for row in reader:
                if API_NAME not in (row.get("OP Type") or "").lower():
                    continue
                try:
                    block_num = float(row["Block Num"])
                    total_cycles = float(row["aiv_total_cycles"])
                    ratio = parse_ratio(row["aiv_vec_ratio"])
                except (ValueError, TypeError):
                    continue
                if block_num <= 0 or total_cycles <= 0:
                    continue
                values.append(total_cycles / block_num * ratio)
    return values


def main():
    parser = argparse.ArgumentParser(description="IsInf perf bench runner")
    parser.add_argument("--run-mode", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--arch", default="dav-3510")
    parser.add_argument(
        "--cases",
        default=os.path.join(SCRIPT_DIR, "cases.csv"),
        help="cases csv, columns: combo,count,repeat,loop_count",
    )
    parser.add_argument("--output", default="samples.csv")
    parser.add_argument(
        "--profile-dir",
        default=os.path.join(SCRIPT_DIR, "profiling"),
        help="msprof profiling output root, npu mode only",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="profiling batch id, default run_YYYYmmdd_HHMMSS; "
        "profiling data goes to {profile-dir}/{run-id}/{combo}_{count}/",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    if not os.path.isfile(args.cases):
        raise SystemExit("cases file not found: {}".format(args.cases))
    cases = load_cases(args.cases)
    print("loaded {} cases from {}".format(len(cases), args.cases))

    if args.skip_build:
        build_dir = "build" if args.run_mode == "npu" else "build_cpu"
        exe = os.path.join(SCRIPT_DIR, build_dir, BENCH_NAME)
        if not os.path.isfile(exe):
            raise SystemExit("bench binary not found: {}".format(exe))
    else:
        exe = build(args.run_mode, args.arch, args.jobs)

    run_id = args.run_id
    if args.run_mode == "cpu":
        print(
            "cpu mode: correctness only (VERIFY), no performance data; loop_count clamped to <=2"
        )
    else:
        if not run_id:
            run_id = "run_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(
            "npu mode: each case runs under `msprof --application`, profiling data in {}/{}/".format(
                args.profile_dir, run_id
            )
        )
        check_device_env(args.arch)

    rows = []
    total = measured = failed = invalid = 0
    for case in cases:
        combo = case["combo"]
        count = case["count"]
        repeat = case["repeat"]
        loop_count = case["loop_count"]
        if args.run_mode == "cpu":
            loop_count = min(loop_count, 2)
        verify_status = None
        profile_dir = None
        if args.run_mode == "npu":
            profile_dir = os.path.join(
                args.profile_dir, run_id, "{}_{}".format(combo, count)
            )
        for _ in range(RUNS_PER_CASE):
            verify = run_one(exe, combo, count, loop_count, args.run_mode, profile_dir)
            if verify == "PASS":
                verify_status = "PASS"
            elif verify == "FAILED":
                verify_status = "FAILED"
        total += 1
        candidate = None
        measured_src = "NA"
        if args.run_mode == "npu":
            prof_values = parse_case_profiling(profile_dir) if profile_dir else []
            if prof_values:
                candidate = round(statistics.median(prof_values), 3)
                measured_src = "msprof({} runs)".format(len(prof_values))
        measured_cycles = None
        if verify_status == "PASS":
            if args.run_mode == "npu" and (not candidate or candidate <= 0):
                status = "INVALID"
                invalid += 1
            else:
                status = "PASS"
                measured += 1
                measured_cycles = candidate
        else:
            status = "FAILED"
            failed += 1
        rows.append([combo, count, repeat, loop_count, None, measured_cycles])
        print(
            "{},{},repeat={},measured_cycles={},src={},{}".format(
                combo,
                count,
                repeat,
                measured_cycles if measured_cycles is not None else "NA",
                measured_src,
                status,
            )
        )

    output = (
        args.output
        if os.path.isabs(args.output)
        else os.path.join(SCRIPT_DIR, args.output)
    )
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "combo",
                "count",
                "repeat",
                "loop_count",
                "model_predict_cycles",
                "measured_cycles",
            ]
        )
        writer.writerows(rows)
    print(
        "summary: total={}, measured={}, invalid={}, failed={}".format(
            total, measured, invalid, failed
        )
    )
    print("output: {}".format(output))
    if failed or invalid:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
