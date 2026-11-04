#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""上板用例的统一入口：按清单跑、与预期对账、汇总成一张表、给一个退出码。

    python3 autofuse/mlir/test/e2e/npu/run.py --list
    python3 autofuse/mlir/test/e2e/npu/run.py --group smoke
    python3 autofuse/mlir/test/e2e/npu/run.py --group reduce --jobs 2
    python3 autofuse/mlir/test/e2e/npu/run.py --id load_sum_store --calibrate

它解决的是四件此前没有归口的事：

1. **跑哪些**——清单在 `cases.py`，一条项是 `(脚本, argv, env)`；`ls *.py` 说不出
   「同一个脚本配不同 `--force_schedule_result` 是不同用例」这件事。
2. **环境**——`support/device_env.py` 按 `AGENTS.md` §2.4 装配，并在跑之前断言
   `autofuse.pyautofuse` 真的来自 overlay。这条断言是硬的：overlay 没生效时用例会用
   CANN 自带的 2 月版跑完并打 PASS，**不报任何错**，而那个结论与本次改动无关。
3. **判级**——所有脚本都打同一种机读结果行（`support/case_status.py`），这里解析并汇总。
4. **对账**——与清单里的预期等级比，比预期差才红。见 `case_status` 的模块说明。

## 为什么每条用例一个子进程

不是为了隔离崩溃（虽然也有这个好处），是因为**没法不这样**：每个脚本都要自己 `chdir`
到实验目录、改一批环境变量、再 `import torch`，而 ATT 的 DFX 配置是进程级的、只初始化
一次。同进程串跑第二条时，钉的 plan 还是第一条那个，所以每条清单项必须使用独立子进程。

## 关于 `--jobs 2`

910B 上有两颗 Ascend910 芯（Phy-ID 4/5），平时只用一颗。`--jobs 2` 把用例分成两股，各用
`ASCEND_RT_VISIBLE_DEVICES` 钉一颗。**钉芯没有经过实测验证**，第一次用先跑
`--jobs 2 --id load_store_direct,load_add_store`，跑完确认两颗芯的 AICore 都动过
（`npu-smi info`）；确认不了就退回默认的串行。
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cases as case_manifest  # noqa: E402
from support import case_status, device_env  # noqa: E402


class Outcome(object):
    """一条清单项跑完之后的结果。"""

    __slots__ = ("case", "returncode", "results", "log", "seconds", "env_error")

    def __init__(self, case, returncode, results, log, seconds, env_error=None):
        self.case = case
        self.returncode = returncode
        self.results = results
        self.log = log
        self.seconds = seconds
        self.env_error = env_error

    @property
    def worst(self) -> Optional[str]:
        return case_status.worst_of(self.results)

    @property
    def tally(self) -> Dict[str, int]:
        return case_status.tally_of(self.results)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按清单跑 MLIR 上板 E2E 用例并汇总",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="只列清单，不跑。")
    parser.add_argument(
        "--id",
        default="",
        help="逗号分隔的清单项 id，只跑这些。",
    )
    parser.add_argument(
        "--group",
        default="",
        help="逗号分隔的分组，只跑这些。可用分组见 --list。",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="实验根目录。默认 $AF_NPU_AGENT_DIR/npu-e2e-<sha>-<时间戳>（§2.4.1）。",
    )
    parser.add_argument("--build-dir", default="", help="autofuse 的 build 目录。")
    parser.add_argument("--overlay", default="", help="overlay 落点。")
    parser.add_argument(
        "--rebuild-overlay",
        action="store_true",
        help="先删掉 overlay 再建。常规重编不需要（.py 与 .so 每次都会覆盖），"
        "只在仓里删过 .py、或 overlay 落点被别的 sha 用过时需要。",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过模块来源核对。只在确认过环境、且要省掉每条用例几秒的前置开销时用。",
    )
    parser.add_argument(
        "--mode",
        choices=("mlir", "legacy", "both"),
        default="",
        help="透传给用例脚本的 --mode。不给则各脚本用自己的默认（mlir）。",
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="并发条数；2 表示钉两颗芯。"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="单条用例的超时秒数，超时记 ERROR。默认 3600。",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="按本轮实测生成可粘回 cases.py 的 expect 块。只输出，不改文件。",
    )
    return parser.parse_args(argv)


def selected_cases(args: argparse.Namespace) -> Tuple[case_manifest.Case, ...]:
    ids = tuple(item.strip() for item in args.id.split(",") if item.strip())
    groups = tuple(item.strip() for item in args.group.split(",") if item.strip())
    unknown = [name for name in groups if name not in case_manifest.all_groups()]
    if unknown:
        raise SystemExit(
            "没有这些分组：{}\n可用：{}".format(
                ", ".join(unknown), ", ".join(case_manifest.all_groups())
            )
        )
    chosen = case_manifest.select(ids, groups)
    if not chosen:
        raise SystemExit("没有选中任何用例")
    return chosen


def print_listing(chosen: Sequence[case_manifest.Case]) -> None:
    print(
        "清单共 {} 项，分组：{}".format(
            len(chosen), ", ".join(case_manifest.all_groups())
        )
    )
    for case in chosen:
        argv = " ".join(case.argv)
        env = " ".join(
            "{}={}".format(key, value) for key, value in sorted(case.env.items())
        )
        pinned = len(case.expect)
        print(
            "  {:<28} {:<26} {}{}{}".format(
                case.id,
                case.script,
                argv,
                ("  [" + env + "]") if env else "",
                ("  expect:{}格".format(pinned)) if pinned else "",
            )
        )
        if case.note:
            print("  {:<28} └ {}".format("", case.note))


def run_one(
    case: case_manifest.Case,
    repo: Path,
    build: Path,
    overlay: Path,
    work_root: Path,
    args: argparse.Namespace,
    device_index: int,
) -> Outcome:
    work_dir = work_root / case.id
    work_dir.mkdir(parents=True, exist_ok=True)

    extra = dict(case.env)
    if args.jobs > 1:
        # 两颗芯各跑一股。`ASCEND_RT_VISIBLE_DEVICES` 决定进程能看见哪颗。
        extra["ASCEND_RT_VISIBLE_DEVICES"] = str(device_index)
    env = device_env.environ_for_run(repo, build, overlay, extra=extra)

    argv = [device_env.python_bin(), str(device_env.cases_dir(repo) / case.script)]
    argv.extend(case.argv)
    argv.extend(["--work-dir", str(work_dir)])
    if args.mode:
        argv.extend(["--mode", args.mode])

    started = time.time()
    try:
        completed = subprocess.run(
            argv,
            env=env,
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
        text = completed.stdout.decode("utf-8", "replace")
        returncode = completed.returncode
    except subprocess.TimeoutExpired as expired:
        text = (expired.output or b"").decode("utf-8", "replace")
        text += "\n[run.py] 超时 {} 秒，已杀掉\n".format(args.timeout)
        returncode = -1

    seconds = time.time() - started
    (work_dir / "run.log").write_text(text, encoding="utf-8")
    results = case_status.parse_cases(text)

    if not results:
        # 一行机读结果都没有：脚本在进到用例之前就死了（import 失败、参数不认、
        # codegen abort 把进程带走）。这种情况必须自己造一格 ERROR——否则它在汇总里
        # 是「0 格」，看起来像没选中，而不是失败。
        reason = "没有任何机读结果行；退出码 {}".format(returncode)
        tail = [line for line in text.strip().splitlines() if line.strip()][-1:]
        if tail:
            reason += "；末行：" + tail[0][:200]
        results = [
            case_status.CaseResult(
                case.id, "", "", case_status.ERROR, "no-report", "", reason
            )
        ]
    return Outcome(case, returncode, results, work_dir / "run.log", seconds)


def reconcile(outcome: Outcome) -> List[Tuple[str, str, str, str]]:
    """把一条用例的每一格与清单预期比。返回 (格子名, 实测, 预期, 判定)。"""
    rows = []
    expect = outcome.case.expect
    for result in outcome.results:
        cell = result.cell or outcome.case.id
        expected = expect.get(cell)
        if expected is None:
            verdict = "defect" if case_status.is_defect(result.status) else "ungraded"
            rows.append((cell, result.status, "", verdict))
        else:
            rows.append(
                (
                    cell,
                    result.status,
                    expected,
                    case_status.compare(result.status, expected),
                )
            )
    observed = {result.cell or outcome.case.id for result in outcome.results}
    for cell in sorted(expect.keys() - observed):
        rows.append((cell, case_status.ERROR, expect[cell], "regressed"))
    # 进程完成状态独立于格子预期；即使 ERROR 是已知预期，也不能接受中途退出。
    if outcome.returncode != 0:
        rows.append(
            (
                "<process exit {}>".format(outcome.returncode),
                case_status.ERROR,
                "",
                "defect",
            )
        )
    return rows


def print_report(outcomes: Sequence[Outcome]) -> int:
    print()
    print("=" * 96)
    print("AF_NPU_RUN 汇总")
    print("=" * 96)

    regressed: List[Tuple[str, str, str, str, str]] = []
    improved: List[Tuple[str, str, str, str]] = []
    defects: List[Tuple[str, str, str, str]] = []

    for outcome in outcomes:
        rows = reconcile(outcome)
        tally = outcome.tally
        counts = " ".join(
            "{}={}".format(status, tally[status])
            for status in case_status.ALL_STATUSES
            if tally.get(status)
        )
        worst = outcome.worst or case_status.SKIP
        print(
            "{:<28} {:<14} {:<44} {:>6.0f}s".format(
                outcome.case.id, worst, counts, outcome.seconds
            )
        )
        for cell, actual, expected, verdict in rows:
            if verdict == "regressed":
                regressed.append((outcome.case.id, cell, actual, expected, "比预期差"))
            elif verdict == "changed":
                regressed.append(
                    (outcome.case.id, cell, actual, expected, "与预期不可比")
                )
            elif verdict == "improved":
                improved.append((outcome.case.id, cell, actual, expected))
            elif verdict == "defect":
                reason = next(
                    (
                        r.reason
                        for r in outcome.results
                        if (r.cell or outcome.case.id) == cell
                    ),
                    "",
                )
                defects.append((outcome.case.id, cell, actual, reason))

    print("-" * 96)
    if defects:
        print("真缺陷（无预期声明，实测 MISMATCH / ERROR）：")
        for case_id, cell, actual, reason in defects:
            print("  {} / {}  {}  {}".format(case_id, cell, actual, reason[:120]))
    if regressed:
        print("比预期差（清单声明过，实测更差）：")
        for case_id, cell, actual, expected, why in regressed:
            print(
                "  {} / {}  实测 {} ← 预期 {}  {}".format(
                    case_id, cell, actual, expected, why
                )
            )
    if improved:
        print("比预期好（支持面变宽了，清单可以收紧——跑 --calibrate 生成新预期）：")
        for case_id, cell, actual, expected in improved:
            print(
                "  {} / {}  实测 {} ← 预期 {}".format(case_id, cell, actual, expected)
            )
    if not (defects or regressed or improved):
        print("没有与预期不符的格子。")

    cells = sum(len(outcome.results) for outcome in outcomes)
    print(
        "AF_NPU_RUN_SUMMARY 用例={} 格={} 缺陷={} 比预期差={} 比预期好={}".format(
            len(outcomes), cells, len(defects), len(regressed), len(improved)
        )
    )
    # 退出码：真缺陷与「比预期差」非零。「比预期好」不红——支持面变宽不是回归，
    # 但它必须被看见，所以单列一节提示去收紧清单。
    return 1 if (defects or regressed) else 0


def print_calibration(outcomes: Sequence[Outcome]) -> None:
    print()
    print("=" * 96)
    print("按本轮实测生成的 expect（粘回 cases.py；只有实测过的格子才在里面）")
    print("=" * 96)
    for outcome in outcomes:
        if outcome.returncode != 0:
            print("# {}：进程异常退出，不能作为校准基线".format(outcome.case.id))
            continue
        graded = [
            result
            for result in outcome.results
            if result.status in case_status.ALL_STATUSES
        ]
        if not graded:
            print("# {}：没有可校准的结果".format(outcome.case.id))
            continue
        print("# {}".format(outcome.case.id))
        print("expect={")
        for result in graded:
            cell = result.cell or outcome.case.id
            comment = "  # {}".format(result.tag) if result.tag else ""
            print('    "{}": case_status.{},{}'.format(cell, result.status, comment))
        print("},")


def run_parallel(chosen, repo, build, overlay, work_root, args):
    """每个 worker 固定占用一颗设备；完成后再领取下一条用例。"""
    pending = Queue()
    for index, case in enumerate(chosen):
        pending.put((index, case))
    for _ in range(args.jobs):
        pending.put(None)
    outcomes = [None] * len(chosen)

    def worker(device_index):
        while True:
            item = pending.get()
            if item is None:
                return
            index, case = item
            outcome = run_one(case, repo, build, overlay, work_root, args, device_index)
            outcomes[index] = outcome
            print(
                "---- [{}/{}] {}  {} ({:.0f}s)".format(
                    index + 1, len(chosen), case.id, outcome.worst, outcome.seconds
                ),
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(worker, device_index) for device_index in range(args.jobs)
        ]
        for future in futures:
            future.result()
    return outcomes


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    chosen = selected_cases(args)

    if args.list:
        print_listing(chosen)
        return 0

    repo = device_env.repo_root()
    try:
        build = device_env.build_dir(repo, args.build_dir or None)
        overlay = device_env.overlay_dir(repo, args.overlay or None)
        device_env.ensure_overlay(repo, build, overlay, rebuild=args.rebuild_overlay)
    except device_env.DeviceEnvError as error:
        print("环境没搭起来，一条用例都没跑：\n{}".format(error), file=sys.stderr)
        return 2

    if args.work_dir:
        work_root = Path(args.work_dir).resolve()
    else:
        base = Path(os.environ.get("AF_NPU_AGENT_DIR", "/mnt/workspace/agent"))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        work_root = base / "npu-e2e-{}-{}".format(device_env.git_sha(repo), stamp)
    work_root.mkdir(parents=True, exist_ok=True)

    print("仓     : {}".format(repo))
    print("build  : {}".format(build))
    print("overlay: {}".format(overlay))
    print("产物   : {}".format(work_root))
    print("用例   : {} 条，并发 {}".format(len(chosen), args.jobs))

    if not args.skip_preflight:
        try:
            origins = device_env.preflight(
                device_env.environ_for_run(repo, build, overlay), overlay
            )
        except device_env.DeviceEnvError as error:
            print("\n前置检查没过，一条用例都没跑：\n{}".format(error), file=sys.stderr)
            return 2
        for name in device_env.PROBED_MODULES:
            print("  {:<28} {}".format(name, origins.get(name, "None")))

    outcomes: List[Outcome] = []
    if args.jobs <= 1:
        for index, case in enumerate(chosen):
            print(
                "\n---- [{}/{}] {}".format(index + 1, len(chosen), case.id), flush=True
            )
            outcomes.append(run_one(case, repo, build, overlay, work_root, args, 0))
            print(
                "     {} ({:.0f}s)".format(outcomes[-1].worst, outcomes[-1].seconds),
                flush=True,
            )
    else:
        outcomes = run_parallel(chosen, repo, build, overlay, work_root, args)

    code = print_report(outcomes)
    if args.calibrate:
        print_calibration(outcomes)
    print("日志   : {}/<id>/run.log".format(work_root))
    return code


if __name__ == "__main__":
    sys.exit(main())
