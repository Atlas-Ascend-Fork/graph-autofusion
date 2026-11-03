#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""统一入口里**不需要 NPU** 的那部分的自测：机读行解析、与预期对账、退出码、环境装配。

为什么单独有它：`run.py` 的判级与对账逻辑决定了整套的退出码，而上板跑一轮要几十分钟、
还得占着设备。这些逻辑本身是纯函数，拿假脚本就能测完；真上板那部分（torch / torch_npu /
overlay）测不了，也不该在这里测。

跑法（两种都行，不需要额外的 pytest 配置）：

    python3 autofuse/mlir/test/e2e/npu/support/test_run_entry.py
    python3 -m pytest autofuse/mlir/test/e2e/npu/support/test_run_entry.py
"""

import os
import io
from contextlib import redirect_stdout
from threading import Event, Lock
from unittest.mock import patch
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cases as case_manifest  # noqa: E402
import run as entry  # noqa: E402
from support import case_status, device_env  # noqa: E402


def test_manifest_matches_torch_e2e_scope():
    root = HERE.parent
    registered = {case.script for case in case_manifest.CASES}
    runnable = {path.name for path in root.glob("*.py")} - {"run.py", "cases.py"}
    assert runnable == registered
    assert all((root / script).is_file() for script in registered)
    assert {
        "load_add_store.py",
        "mixed_consumption.py",
        "load_broadcast_add_direct.py",
        "load_broadcast_to_abs.py",
        "load_broadcast_to_cascade.py",
        "load_broadcast_diamond_cascade.py",
        "load_broadcast_inner_brc_cascade.py",
        "load_broadcast_interleaved_axes_cascade.py",
        "load_broadcast_tail_pitch_cascade.py",
    } <= registered
    assert not {"broadcast_matrix.py", "elewise_matrix.py"} & runnable
    assert not any(
        (root / name).exists()
        for name in ("direct_ascir", "generated_artifact", "reduce_padding")
    )


def test_canonical_line_roundtrip():
    line = case_status.emit_case(
        "ELEWISE",
        "relu_f32",
        case_status.PASS_FALLBACK,
        "binding-conflict",
        ((16, 32),),
        "boom\tgoes\nthe reason",
        "unary",
    )
    # 制表符和换行必须被洗掉，否则一行结果会被拆成几行、字段数对不上而被整行丢弃。
    assert line.count("\t") == 7, line
    parsed = case_status.parse_cases("噪声\n" + line + "\n更多噪声")
    assert len(parsed) == 1
    got = parsed[0]
    assert got.suite == "ELEWISE"
    assert got.cell == "relu_f32"
    assert got.group == "unary"
    assert got.status == case_status.PASS_FALLBACK
    assert got.tag == "binding-conflict"
    assert got.key == "ELEWISE/relu_f32"
    assert "boom goes the reason" in got.reason


def test_malformed_lines_are_ignored():
    # 字段数不对、等级不认识的行一律丢掉：用例的 stdout 里混着 inductor 的大量输出，
    # 宽松匹配会把别的东西认成结果。
    bad = [
        "AF_NPU_CASE\tA\tb\tg\tPASS",  # 字段少
        "AF_NPU_CASE\tA\tb\tg\tNOPE\t\t\t",  # 等级不认识
        "prefix AF_NPU_CASE\tA\tb\tg\tPASS\t\t\t",  # 不在行首
    ]
    assert case_status.parse_cases("\n".join(bad)) == []


def test_severity_comparison():
    assert case_status.compare(case_status.PASS, case_status.PASS) == "match"
    assert (
        case_status.compare(case_status.PASS_FALLBACK, case_status.PASS) == "regressed"
    )
    assert (
        case_status.compare(case_status.REFUSED, case_status.PASS_FALLBACK)
        == "regressed"
    )
    assert case_status.compare(case_status.PASS, case_status.REFUSED) == "improved"
    assert case_status.compare(case_status.ERROR, case_status.MISMATCH) == "regressed"
    # SKIP 不在严重度阶梯上，只认相等，不比好坏。
    assert case_status.compare(case_status.SKIP, case_status.PASS) == "changed"
    assert case_status.compare(case_status.SKIP, case_status.SKIP) == "match"


def test_worst_of_skips_ungraded():
    results = [
        case_status.CaseResult("S", "a", "", case_status.SKIP, "", "", ""),
        case_status.CaseResult("S", "b", "", case_status.PASS, "", "", ""),
        case_status.CaseResult("S", "c", "", case_status.REFUSED, "", "", ""),
    ]
    assert case_status.worst_of(results) == case_status.REFUSED
    only_skip = [case_status.CaseResult("S", "a", "", case_status.SKIP, "", "", "")]
    assert case_status.worst_of(only_skip) is None


def _outcome(case, statuses):
    results = [
        case_status.CaseResult(case.id, cell, "", status, "", "", "")
        for cell, status in statuses
    ]
    return entry.Outcome(case, 0, results, Path("/dev/null"), 1.0)


def test_reconcile_and_exit_code():
    pinned = case_manifest.Case(
        id="pinned",
        script="x.py",
        expect={"cell_a": case_status.REFUSED, "cell_b": case_status.PASS},
    )

    # 实测与预期一致 -> 绿
    same = _outcome(
        pinned, [("cell_a", case_status.REFUSED), ("cell_b", case_status.PASS)]
    )
    assert entry.print_report([same]) == 0

    # 声明 PASS 实测 PASS_FALLBACK -> 红。这正是此前测不出来的那类漂移：
    # 旧规则只看 MISMATCH/ERROR，PASS 退化成靠容错捡回来是全绿的。
    worse = _outcome(
        pinned, [("cell_a", case_status.REFUSED), ("cell_b", case_status.PASS_FALLBACK)]
    )
    assert entry.print_report([worse]) == 1

    # 声明 REFUSED 实测 PASS -> 不红，但要被看见（报告里单列一节）
    better = _outcome(
        pinned, [("cell_a", case_status.PASS), ("cell_b", case_status.PASS)]
    )
    assert entry.print_report([better]) == 0
    verdicts = {row[0]: row[3] for row in entry.reconcile(better)}
    assert verdicts["cell_a"] == "improved"

    # 没有预期声明：沿用旧规则，只有真缺陷红
    bare = case_manifest.Case(id="bare", script="x.py")
    assert entry.print_report([_outcome(bare, [("c", case_status.REFUSED)])]) == 0
    assert entry.print_report([_outcome(bare, [("c", case_status.MISMATCH)])]) == 1


def test_manifest_is_consistent():
    case_manifest.by_id()  # id 重复会抛
    npu_dir = HERE.parent
    for case in case_manifest.CASES:
        assert (npu_dir / case.script).is_file(), "清单指向不存在的脚本：" + case.script
        for cell, status in case.expect.items():
            assert status in case_status.ALL_STATUSES, (case.id, cell, status)
    # 每个脚本必须注册；防止意外漏掉用例。
    # 未说明的漏项就是「统一入口跑不到的用例」，
    # 而它看起来和跑到了没有区别。
    scripts = {
        path.name
        for path in npu_dir.glob("*.py")
        if path.name not in ("run.py", "cases.py")
    }
    listed = {case.script for case in case_manifest.CASES}
    assert scripts == listed, "清单与目录不符，差集：{}".format(scripts ^ listed)


def _fake_repo(tmp: Path, script_body: str) -> tuple:
    """搭一个假仓 + 假 build + 假 CANN，够 run_one 跑起来。"""
    repo = tmp / "repo"
    npu = repo / "autofuse" / "mlir" / "test" / "e2e" / "npu"
    npu.mkdir(parents=True)
    (repo / "autofuse" / "mlir" / "AGENTS.md").write_text("fake", encoding="utf-8")
    (repo / "autofuse" / "compiler" / "python").mkdir(parents=True)
    (repo / "autofuse" / "compiler" / "python" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (npu / "fake_case.py").write_text(script_body, encoding="utf-8")

    build = tmp / "build"
    (build / "compiler" / "py_module").mkdir(parents=True)
    (build / "compiler" / "py_module" / "pyautofuse.so").write_bytes(b"\0")
    (build / "libaihac_codegen.so").write_bytes(b"\0")

    cann = tmp / "cann"
    (cann / "lib64").mkdir(parents=True)
    os.environ["ASCEND_HOME_PATH"] = str(cann)
    # 默认的 PYTHONHOME=/usr 是设备上的配方，本机解释器会因此起不来——那样两条
    # 端到端断言测的就是「python 没启动」，而不是机读行解析。
    os.environ["AF_NPU_PYTHONHOME"] = sys.base_prefix
    return repo, build


def test_run_one_parses_a_real_subprocess():
    tmp = Path(tempfile.mkdtemp())
    try:
        body = (
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "from support import case_status\n"
            "print('noise from inductor')\n"
            "print(case_status.emit_case('FAKE', 'c0', case_status.PASS, '', (1,), '', 'g'))\n"
            "print(case_status.emit_case('FAKE', 'c1', case_status.REFUSED, 'no-impl', (2,), 'why', 'g'))\n"
            "sys.exit(0)\n" % str(HERE.parent)
        )
        repo, build = _fake_repo(tmp, body)
        overlay = tmp / "ov"
        device_env.ensure_overlay(repo, build, overlay)
        case = case_manifest.Case(
            id="fake", script="fake_case.py", argv=("--steps", "0")
        )
        args = entry.parse_args(["--timeout", "120"])
        # run_one 会追加 --work-dir，所以假脚本不能用 argparse 卡在未知参数上——
        # 这里的假脚本直接忽略 argv，真脚本靠 NpuRunner 解析。
        outcome = entry.run_one(case, repo, build, overlay, tmp / "work", args, 0)
        statuses = [(r.cell, r.status) for r in outcome.results]
        assert statuses == [("c0", case_status.PASS), ("c1", case_status.REFUSED)], (
            statuses
        )
        assert outcome.worst == case_status.REFUSED
        assert (
            outcome.log.is_file() and "noise from inductor" in outcome.log.read_text()
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_silent_death_becomes_an_error_cell():
    """脚本在进到用例之前就死：必须变成一格 ERROR，不能是「0 格」。

    这条是真踩得到的：codegen abort 会把整个进程带走，历史随机探针中就
    记着这件事。0 格在汇总里看起来像「没选中」，和失败完全分不开。
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        repo, build = _fake_repo(
            tmp, "import sys\nprint('about to die')\nsys.exit(134)\n"
        )
        overlay = tmp / "ov"
        device_env.ensure_overlay(repo, build, overlay)
        case = case_manifest.Case(id="dies", script="fake_case.py")
        args = entry.parse_args([])
        outcome = entry.run_one(case, repo, build, overlay, tmp / "work", args, 0)
        assert len(outcome.results) == 1
        assert outcome.results[0].status == case_status.ERROR
        assert outcome.results[0].tag == "no-report"
        assert "about to die" in outcome.results[0].reason
        assert entry.print_report([outcome]) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_library_path_covers_every_so_dir():
    """LD_LIBRARY_PATH 要带上 build 树里每个放 .so 的目录，漏一个就 undefined symbol。"""
    tmp = Path(tempfile.mkdtemp())
    try:
        build = tmp / "build"
        for sub in ("", "graph_metadef/graph", "ascir/meta", "nested/deep/dir"):
            target = build / sub if sub else build
            target.mkdir(parents=True, exist_ok=True)
            (target / "lib.so").write_bytes(b"\0")
        (build / "versioned").mkdir()
        (build / "versioned" / "lib.so.1.2").write_bytes(b"\0")
        (build / "nolibs").mkdir()
        entries = device_env.library_path_entries(build)
        assert entries[0] == str(build.resolve()), "build 根必须排最前"
        for sub in (
            "graph_metadef/graph",
            "ascir/meta",
            "nested/deep/dir",
            "versioned",
        ):
            assert str((build / sub).resolve()) in entries, sub
        assert str((build / "nolibs").resolve()) not in entries
        # 顺序必须确定：不确定的 LD_LIBRARY_PATH 顺序会让符号解析在两次运行之间漂。
        assert entries[1:] == sorted(entries[1:])
        assert device_env.library_path_entries(build) == entries
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_preflight_rejects_fallback_to_installed_cann():
    """overlay 没生效必须报错而不是放过——这是整套里唯一能抓住静默回落的判据。"""
    tmp = Path(tempfile.mkdtemp())
    try:
        repo, build = _fake_repo(tmp, "")
        overlay = tmp / "ov"
        device_env.ensure_overlay(repo, build, overlay)
        # 假 autofuse 包放在别处，PYTHONPATH 指向它 -> origin 不在 overlay 里
        elsewhere = tmp / "elsewhere"
        (elsewhere / "autofuse").mkdir(parents=True)
        (elsewhere / "autofuse" / "__init__.py").write_text("", encoding="utf-8")
        (elsewhere / "autofuse" / "pyautofuse.py").write_text("", encoding="utf-8")
        (elsewhere / "torch.py").write_text("", encoding="utf-8")
        (elsewhere / "torch_npu.py").write_text("", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(elsewhere)
        try:
            device_env.preflight(env, overlay, python=sys.executable)
        except device_env.DeviceEnvError as error:
            assert "不在 overlay" in str(error), str(error)
        else:
            raise AssertionError("overlay 没生效却没报错")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_partial_report_then_exit_or_timeout_is_failure():
    for ending in ("sys.exit(134)", "import time; time.sleep(30)"):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            body = (
                "import sys\n"
                "print('AF_NPU_CASE\\tFAKE\\tc0\\tg\\tPASS\\t\\t\\t', flush=True)\n"
                + ending
                + "\n"
            )
            repo, build = _fake_repo(tmp, body)
            overlay = tmp / "ov"
            device_env.ensure_overlay(repo, build, overlay)
            case = case_manifest.Case(id="partial", script="fake_case.py")
            outcome = entry.run_one(
                case,
                repo,
                build,
                overlay,
                tmp / "work",
                entry.parse_args(["--timeout", "1"]),
                0,
            )
            assert outcome.results[0].status == case_status.PASS
            assert outcome.returncode != 0
            assert entry.print_report([outcome]) == 1
            output = io.StringIO()
            with redirect_stdout(output):
                entry.print_calibration([outcome])
            assert "expect={" not in output.getvalue()


def test_numeric_failure_wins_over_candidate_refusal():
    diagnostic = "BindingConflict: rejected candidate"
    assert case_status.classify(
        AssertionError("Tensor-likes are not close!"), diagnostic
    ) == (case_status.MISMATCH, "numeric")
    assert case_status.classify(None, diagnostic)[0] == case_status.PASS_FALLBACK
    assert (
        case_status.classify(RuntimeError("compilation failed"), diagnostic)[0]
        == case_status.REFUSED
    )
    assert (
        case_status.classify(RuntimeError("unexpected failure"), "")[0]
        == case_status.ERROR
    )


def test_calibration_retains_pass_and_detects_degradation():
    case = case_manifest.Case(id="baseline", script="x.py")
    output = io.StringIO()
    with redirect_stdout(output):
        entry.print_calibration(
            [_outcome(case, [("a", case_status.PASS), ("b", case_status.SKIP)])]
        )
    assert '"a": case_status.PASS,' in output.getvalue()
    assert '"b": case_status.SKIP,' in output.getvalue()
    calibrated = case_manifest.Case(
        id=case.id, script=case.script, expect={"a": case_status.PASS}
    )
    for actual in (case_status.PASS_FALLBACK, case_status.REFUSED):
        assert entry.print_report([_outcome(calibrated, [("a", actual)])]) == 1


def test_missing_expected_cell_is_failure_even_when_expected_error():
    for expected in (case_status.PASS, case_status.ERROR):
        case = case_manifest.Case(
            id="missing", script="x.py", expect={"a": case_status.PASS, "b": expected}
        )
        assert entry.print_report([_outcome(case, [("a", case_status.PASS)])]) == 1
        assert (
            entry.print_report(
                [_outcome(case, [("a", case_status.PASS), ("b", expected)])]
            )
            == 0
        )


def test_parallel_workers_hold_device_until_case_finishes():
    cases = [case_manifest.Case(id=str(i), script="x.py") for i in range(4)]
    first_started, third_finished = Event(), Event()
    lock = Lock()
    active, assignments = set(), {}

    def fake_run(case, repo, build, overlay, work_root, args, device):
        with lock:
            assert device not in active, "two cases used the same device concurrently"
            active.add(device)
            assignments[case.id] = device
        if case.id == "0":
            first_started.set()
            assert third_finished.wait(3)
        else:
            assert first_started.wait(3)
            if case.id == "2":
                third_finished.set()
        with lock:
            active.remove(device)
        return _outcome(case, [("a", case_status.PASS)])

    with patch.object(entry, "run_one", side_effect=fake_run):
        outcomes = entry.run_parallel(
            cases, None, None, None, None, entry.parse_args(["--jobs", "2"])
        )
    assert [outcome.case.id for outcome in outcomes] == [case.id for case in cases]
    assert assignments["0"] != assignments["2"]


def main() -> int:
    tests = [
        value for name, value in sorted(globals().items()) if name.startswith("test_")
    ]
    failures = []
    for test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001
            failures.append((test.__name__, error))
            print("FAIL {}: {}: {}".format(test.__name__, type(error).__name__, error))
        else:
            print("ok   {}".format(test.__name__))
    print("\n{}/{} 通过".format(len(tests) - len(failures), len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
