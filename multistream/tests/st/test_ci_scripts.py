# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Static checks for the multistream CI dispatch plumbing."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


_ROOT = Path(__file__).parents[3]
_BUILD = _ROOT / "build.sh"
_UT = _ROOT / ".gitcode" / "scripts" / "ut.sh"
_WORKFLOW = _ROOT / ".gitcode" / "workflows" / "graph-autofusion_action.yml"


def test_multistream_build_entry_and_coverage_artifact_contract_are_present():
    text = _BUILD.read_text(encoding="utf-8")
    assert '"multistream:py_ut"' in text
    assert "coverage_multistream.info" in text
    assert "--cov-report=lcov:" in text
    assert (
        "fail_under = 90" in (_ROOT / "multistream/scripts/ms_ut_cfg.toml").read_text()
    )
    assert "--no-autofuse" in text


def test_ci_dispatch_has_multistream_job_and_both_branch_cases():
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    assert "JOB_UT_TEST_Python_multistream:" in workflow
    assert "ut_type: 'UT_Test_Python_multistream'" in workflow
    text = _UT.read_text(encoding="utf-8")
    assert text.count("UT_Test_Python_multistream)") >= 2
    assert text.count("--module=multistream --no-autofuse") >= 2


@pytest.mark.parametrize("entry_name", ["multistream_py_ut", "multistream_py_st"])
def test_multistream_installs_dev_extra(entry_name):
    """UT/ST use the project's dev extra, as SuperKernel does."""

    text = _BUILD.read_text(encoding="utf-8")
    entry = text.split(f"{entry_name}() {{", 1)[1].split("\n}", 1)[0]
    assert 'pip install -e ".[dev]" --force-reinstall -q' in entry


@pytest.mark.parametrize("coverage", ["off", "on"])
def test_multistream_st_writes_junit_report(tmp_path, coverage):
    root = tmp_path / "checkout with spaces"
    source = root / "multistream"
    (source / "tests/st").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "probe.py").write_text("def answer():\n    return 42\n")
    (source / "tests/st/test_probe.py").write_text(
        "import pytest\nfrom probe import answer\n"
        "@pytest.mark.st\ndef test_answer():\n    assert answer() == 42\n"
    )
    (source / "pytest.ini").write_text("[pytest]\nmarkers = st: system test\n")
    (source / "scripts/ms_st_cfg.toml").write_text(
        (_ROOT / "multistream/scripts/ms_st_cfg.toml").read_text()
    )
    # Skip dependency installation, but execute the real pytest command from
    # build.sh so the test checks the report file rather than argument spelling.
    python = tmp_path / "python-wrapper"
    python.write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "pip" ]; then exit 0; fi\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    python.chmod(0o755)
    entry = _BUILD.read_text().split("multistream_py_st() {", 1)[1].split("\n}", 1)[0]
    result = subprocess.run(
        ["bash", "-c", "multistream_py_st() {" + entry + "\n}\nmultistream_py_st"],
        env=dict(
            os.environ,
            BASEPATH=str(root),
            PYTHON_CMD=str(python),
            ENABLE_COVERAGE=coverage,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = root / "build_out/multistream_st/junit.xml"
    assert report.is_file(), result.stdout + result.stderr
    suite = ET.parse(report).getroot().find("testsuite")
    assert suite is not None
    assert suite.get("tests") == "1"
    assert suite.get("failures") == "0"
    assert suite.get("errors") == "0"
