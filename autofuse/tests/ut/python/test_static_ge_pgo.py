# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from compile_test_utils import load_asc_codegen_compile_module


@pytest.fixture
def ge_compile(compile_adapter_module, monkeypatch):
    with load_asc_codegen_compile_module() as module:
        name = module.__package__ + ".compile_adapter"
        monkeypatch.setitem(sys.modules, name, compile_adapter_module)
        yield module


@pytest.mark.parametrize(
    "first_status,first_duration,retest_duration,attempted,expected_level,expected_config",
    [
        (1, "1.79769e+308", "21", True, "info", "2 # 21"),
        (1, "14", None, True, "info", "2 # 14"),
        (1, "1.79769e+308", None, True, "error", None),
        (1, "1.79769e+308", None, False, "warning", None),
        (1, "14", "21", True, "warning", None),
        (None, "14", None, True, "info", "2 # 14"),
        (-signal.SIGSEGV, "1.79769e+308", None, True, "error", None),
        (250, "14", None, True, "info", "2 # 14"),
    ],
)
def test_pgo_uses_only_valid_phase_results(
    ge_compile,
    monkeypatch,
    tmp_path,
    first_status,
    first_duration,
    retest_duration,
    attempted,
    expected_level,
    expected_config,
):
    module = ge_compile
    temp_dir = tmp_path / "build" / "compile"
    for directory in ("host", "device"):
        (temp_dir / directory).mkdir(parents=True)
    pgo_dir = tmp_path / "pgo"
    reported = []
    stages = []
    kernel = tmp_path / "graph.o"
    metadata = tmp_path / "graph.json"
    kernel.touch()
    metadata.touch()
    monkeypatch.setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_pgo_topn=10")
    monkeypatch.setattr(module, "timestamp_set", lambda *args: None)
    monkeypatch.setattr(module, "pgo_kernel_compile", lambda *args, **kwargs: True)
    monkeypatch.setattr(module, "generate_pgo_code", lambda *args: None)
    monkeypatch.setattr(module, "pgo_compile", lambda *args: "pgo")
    monkeypatch.setattr(module, "get_pgo_exec_params", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        module, "pgo_compile_candidate", lambda *args, **kwargs: (kernel, metadata)
    )
    monkeypatch.setattr(module.AscendCLogLevel, "LOG_INFO", "info", raising=False)
    monkeypatch.setattr(
        module.CommonUtility,
        "print_compile_log",
        lambda _, message, level: reported.append((message, level)),
    )
    if attempted and retest_duration and expected_config is None:

        def fail_publish(*args):
            raise OSError("commit failed")

        adapter = sys.modules[module.__package__ + ".compile_adapter"]
        monkeypatch.setattr(adapter, "pgo_generate_config", fail_publish)

    def run(trial_dir, stage, *, pgo_exec_params, lock_dir):
        stages.append(stage)
        assert Path(lock_dir) == pgo_dir
        assert not (pgo_dir / "graph_config.txt").exists()
        search = Path(trial_dir).parent / "graph_search.txt"
        if stage == 0:
            search.write_text(f"2 # {first_duration}\n1 # 1.79769e+308\n")
            if attempted:
                search.with_suffix(".txt.attempted").touch()
            return first_status
        assert search.read_text().splitlines() == [
            f"2 # {first_duration}",
            "1 # 1.79769e+308",
        ]
        if retest_duration and not search.with_suffix(".txt.retest_written").exists():
            search.with_suffix(".txt.retest").write_text(f"2 # {retest_duration}\n")
            search.with_suffix(".txt.retest_written").touch()
        return 0

    monkeypatch.setattr(module, "pgo_program_exec", run)
    module.asc_pgo_exec(
        "graph",
        temp_dir=str(temp_dir),
        params={"schedule_results": SimpleNamespace(get_name=lambda: "graph")},
        op_kernel_src="",
        code_gen=None,
    )

    assert stages == ([0, 1, 1] if first_status == 1 else [0])
    assert [level for _, level in reported] == [expected_level]
    config = pgo_dir / "graph_config.txt"
    if expected_config:
        assert config.read_text() == f"1\n{expected_config}\n"
    else:
        assert not config.exists()
    assert list(pgo_dir.glob(".pgo-tune-*")) == []


def test_pgo_rejects_stale_or_differently_identified_retests(
    ge_compile, monkeypatch, tmp_path
):
    search = tmp_path / "graph_search.txt"
    retest = tmp_path / "graph_search.txt.retest"
    retest.write_text("9 # 1\n")
    monkeypatch.setattr(ge_compile, "get_pgo_topn", lambda: 0)
    candidates = ["2 # 21", "2 # 20", "1 # 22"]
    compiled = []

    def compile_trial(line):
        compiled.append(line)
        return True

    def run_trial():
        assert not retest.exists()
        retest.write_text("9 # 1\n")
        return 0

    measured, identities = ge_compile.pgo_second_optimization(
        compile_trial, run_trial, candidates, search
    )

    assert compiled == ["2 # 20", "1 # 22"]
    assert identities == {(2,), (1,)}
    assert measured == []


@pytest.mark.parametrize(
    "status,expected_attempts",
    [(None, 2), (-signal.SIGSEGV, 2), (243, 2), (244, 2), (250, 2), (1, 3)],
)
def test_pgo_retest_preserves_measurements_when_runner_stops(
    ge_compile, monkeypatch, tmp_path, status, expected_attempts
):
    search = tmp_path / "graph_search.txt"
    retest = tmp_path / "graph_search.txt.retest"
    monkeypatch.setattr(ge_compile, "get_pgo_topn", lambda: 10)
    candidates = ["2 # 10", "3 # 12", "1 # 14"]
    compiled = []
    statuses = iter([0, status, 0])

    def compile_trial(line):
        compiled.append(line)
        return True

    def run_trial():
        result = next(statuses)
        if result == 0:
            identity, _, _ = compiled[-1].partition(" # ")
            retest.write_text(f"{identity} # 20\n")
        return result

    measured, identities = ge_compile.pgo_second_optimization(
        compile_trial, run_trial, candidates, search
    )

    assert compiled == candidates[:expected_attempts]
    assert identities == (
        {(2,), (3,)} if expected_attempts == 2 else {(2,), (3,), (1,)}
    )
    assert measured == (["2 # 20"] if expected_attempts == 2 else ["2 # 20", "1 # 20"])


def test_replay_does_not_start_pgo_runner(ge_compile, monkeypatch):
    monkeypatch.setenv("AF_BATCH_PGO_REPLAY_ONLY", "1")
    monkeypatch.setattr(ge_compile, "get_pgo_env_flag", lambda: True)
    monkeypatch.setattr(
        ge_compile, "asc_pgo_exec", lambda **kwargs: pytest.fail("runner started")
    )

    ge_compile._run_pgo_and_clean_cache(
        temp_dir=".",
        params={},
        op_kernel_src="",
        cross_compile_flag=False,
        host_build_dir=".",
    )


def test_reused_worker_observes_the_session_replay_marker(
    ge_compile, monkeypatch, tmp_path
):
    monkeypatch.delenv("AF_BATCH_PGO_REPLAY_ONLY", raising=False)
    monkeypatch.setattr(ge_compile, "get_pgo_env_flag", lambda: True)
    monkeypatch.setattr(
        ge_compile,
        "asc_pgo_exec",
        lambda *args, **kwargs: pytest.fail("runner started"),
    )
    pgo_dir = tmp_path / "pgo"
    pgo_dir.mkdir()
    (pgo_dir / ".replay-only").touch()

    ge_compile._run_pgo_and_clean_cache(
        temp_dir=str(tmp_path / "build" / "compile"),
        params={},
        op_kernel_src="",
        cross_compile_flag=False,
        host_build_dir=".",
    )


def test_ge_search_reuses_a_valid_committed_config(ge_compile, monkeypatch, tmp_path):
    config_dir = tmp_path / "pgo"
    config_dir.mkdir()
    config = config_dir / "graph_config.txt"
    config.write_text("1\n1 2 # 20\n")
    monkeypatch.setattr(
        ge_compile, "_asc_pgo_tune", lambda *args: pytest.fail("cache was ignored")
    )

    ge_compile.asc_pgo_exec(
        "graph",
        temp_dir=str(tmp_path / "build" / "compile"),
        params={"schedule_results": SimpleNamespace(get_name=lambda: "graph")},
        op_kernel_src="",
        code_gen=None,
    )

    assert config.read_text() == "1\n1 2 # 20\n"
