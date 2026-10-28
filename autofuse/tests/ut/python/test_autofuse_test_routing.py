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

import os
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[4]


def run_shell_main(script, setup, tmp_path):
    """Exercise shell orchestration with build tools and test executors stubbed."""
    text = script.read_text()
    assert text.rstrip().endswith('main "$@"')
    definitions = text.rsplit('\nmain "$@"', 1)[0]
    build_path = tmp_path / "build"
    program = r"""
cmake() {
  printf 'TRACE cmake %s\n' "$*"
  [[ "$PWD" == "$BUILD_PATH" ]] || return 92
  return "${MOCK_CMAKE_STATUS:-0}"
}
make() {
  printf 'TRACE make %s\n' "$*"
  [[ -d "$BUILD_PATH" ]] || return 91
  return "${MOCK_MAKE_STATUS:-0}"
}
ctest() { printf 'TRACE ctest %s\n' "$*"; }
g++() { printf 'TRACE compiler %s\n' "$*"; }
env() { :; }
find() { printf 'TRACE find\n' >&2; }
cp() { printf 'TRACE cp %s\n' "$*"; }
"""
    program += definitions
    program += f"""
BUILD_PATH={shlex.quote(str(build_path))}
AUTOFUSE_BUILD_PATH="${{BUILD_PATH}}/autofuse"
OUTPUT_PATH={shlex.quote(str(tmp_path / "output"))}
METADEF_LIB_PATH="${{OUTPUT_PATH}}/metadef/lib"
THREAD_NUM=8
{setup}
main
"""
    return subprocess.run(
        ["bash", "-c", program, str(script)],
        cwd=tmp_path,
        env={**os.environ, "ASCEND_HOME_PATH": str(tmp_path / "cann")},
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )


@pytest.mark.parametrize(
    "package,actions,product_required",
    [
        ("off", ["autofuse_framework:all_ut"], False),
        ("off", ["autofuse_ascendc_api:all_ut"], False),
        ("off", ["autofuse_e2e:all_st"], False),
        ("off", ["autofuse_framework:all_ut", "autofuse_e2e:all_st"], False),
        ("on", ["autofuse_framework:all_ut"], True),
        ("off", ["superkernel:py_ut"], True),
        ("off", ["superkernel:py_ut", "autofuse_framework:all_ut"], True),
        ("off", ["multistream:py_ut"], True),
    ],
)
@pytest.mark.parametrize("autofuse_enabled", ["on", "off"])
def test_product_build_is_only_skipped_for_test_owned_autofuse_targets(
    tmp_path, package, actions, product_required, autofuse_enabled
):
    action_words = " ".join(shlex.quote(action) for action in actions)
    setup = f"""
checkopts() {{
  ENABLE_AUTOFUSE={autofuse_enabled}
  ENABLE_BUILD_PACKAGE={package}
  EXEC_ACTIONS=({action_words})
  CHANGED_FILES=
}}
clean_coverage_artifacts() {{ :; }}
autofuse_module_test_suite() {{ printf 'TRACE handler %s:%s\\n' "$1" "$2"; }}
superkernel_py_ut() {{ printf 'TRACE handler %s:%s\\n' "$1" "$2"; }}
multistream_py_ut() {{ printf 'TRACE handler %s:%s\\n' "$1" "$2"; }}
build_package() {{ printf 'TRACE package\\n'; }}
"""
    result = run_shell_main(ROOT / "build.sh", setup, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("TRACE cmake --build" in result.stdout) == (
        product_required and autofuse_enabled == "on"
    )
    assert ("TRACE package" in result.stdout) == (package == "on")
    for action in actions:
        assert f"TRACE handler {action}" in result.stdout


@pytest.mark.parametrize(
    "affinity,quota,expected",
    [
        (96, "1600000 100000", 16),
        (96, "3200000 100000", 32),
        (8, "1600000 100000", 8),
        (96, "max 100000", 96),
        (96, "-1 100000", 96),
        (96, "50000 100000", 1),
        (96, "150000 100000", 2),
        (96, "9223372036854775807 100000", 96),
        (96, "", 96),
        (96, "100000 0", 96),
        (96, "invalid 100000", 96),
    ],
)
def test_default_parallelism_respects_affinity_and_container_quota(
    tmp_path, affinity, quota, expected
):
    setup = f"""
nproc() {{ printf '%s\\n' {affinity}; }}
get_cgroup_cpu_quota() {{ printf '%s\\n' {shlex.quote(quota)}; }}
main() {{ printf 'TRACE cpu=%s\\n' "$(get_available_cpu_count)"; }}
"""
    result = run_shell_main(ROOT / "build.sh", setup, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"TRACE cpu={expected}\n" in result.stdout


def test_cpu_discovery_ignores_openmp_without_changing_runtime_settings(tmp_path):
    setup = r"""
OMP_NUM_THREADS=1
OMP_THREAD_LIMIT=2
nproc() {
  [[ ! -v OMP_NUM_THREADS && ! -v OMP_THREAD_LIMIT ]] || return 93
  printf '96\n'
}
get_cgroup_cpu_quota() { printf '1600000 100000\n'; }
main() {
  printf 'TRACE cpu=%s\n' "$(get_available_cpu_count)"
  printf 'TRACE omp=%s:%s\n' "$OMP_NUM_THREADS" "$OMP_THREAD_LIMIT"
}
"""
    result = run_shell_main(ROOT / "build.sh", setup, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TRACE cpu=16\n" in result.stdout
    assert "TRACE omp=1:2\n" in result.stdout


def test_cpu_discovery_failure_does_not_produce_unbounded_parallelism(tmp_path):
    setup = "nproc() { return 94; }\nmain() { get_available_cpu_count; }\n"
    result = run_shell_main(ROOT / "build.sh", setup, tmp_path)
    assert result.returncode != 0


def autofuse_setup(module, ut, st):
    return f"""
checkopts() {{
  MODEL_NAME={module}
  ENABLE_UT={ut}
  ENABLE_ST={st}
  ENABLE_COV=on
  RUN_V35_TESTS=on
  ASCEND_INSTALL_PATH={shlex.quote(str(ROOT))}
}}
build_ascgen-dev() {{ printf 'TRACE pyautofuse dependency build\\n'; mkdir -p "$BUILD_PATH"; }}
get_coverage() {{ printf 'TRACE coverage\\n'; }}
"""


def test_numeric_api_ut_does_not_build_codegen_or_pyautofuse(tmp_path):
    result = run_shell_main(
        ROOT / "scripts/test/run_autofuse_test.sh",
        autofuse_setup("ascendc_api", "on", "off") + "BUILD_TYPE=Debug\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "build").is_dir()
    trace = [line for line in result.stdout.splitlines() if line.startswith("TRACE ")]
    make_calls = [line for line in trace if line.startswith("TRACE make ")]
    assert make_calls == ["TRACE make -j8 test_ascendc_api test_ascendc_api_v35"]
    configure = next(line for line in trace if line.startswith("TRACE cmake "))
    assert "-D BUILD_AUTOFUSE=ON" in configure
    assert "-D ENABLE_CPP_UTEST=OFF" in configure
    assert "-D ENABLE_CPP_STEST=OFF" in configure
    assert "-D CMAKE_BUILD_TYPE=Debug" in configure
    assert "TRACE pyautofuse dependency build" not in trace
    assert "TRACE find" not in result.stderr
    ctest_calls = [line for line in trace if line.startswith("TRACE ctest ")]
    assert len(ctest_calls) == 2
    assert "/tests/ut/ascendc" in ctest_calls[0]
    assert "/tests/v35/ut/ascendc" in ctest_calls[1]
    assert all("/tests/ut/e2e" not in call for call in ctest_calls)
    assert "TRACE coverage" in trace
    assert "[autofuse resources][exit]" in result.stdout


def test_autofuse_handler_preserves_requested_build_type(tmp_path):
    setup = r"""
BUILD_TYPE=Debug
bash() { printf 'TRACE child BUILD_TYPE=%s args=%s\n' "$BUILD_TYPE" "$*"; }
main() { autofuse_module_test_suite autofuse_framework all_ut; }
"""
    result = run_shell_main(ROOT / "build.sh", setup, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TRACE child BUILD_TYPE=Debug" in result.stdout
    assert "-u -m framework -j 8" in result.stdout


@pytest.mark.parametrize(
    "module,ut,st",
    [
        ("ascendc_api", "off", "on"),
        ("ascendc_api", "on", "on"),
        ("framework", "on", "off"),
    ],
)
def test_other_suites_keep_their_pyautofuse_dependencies(tmp_path, module, ut, st):
    setup = autofuse_setup(module, ut, st)
    setup += r"""
build_ut() { printf 'TRACE ut suite\n'; }
build_st() { printf 'TRACE st suite\n'; }
"""
    result = run_shell_main(ROOT / "scripts/test/run_autofuse_test.sh", setup, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TRACE pyautofuse dependency build" in result.stdout
    assert "TRACE coverage" in result.stdout


def test_broadcast_codegen_cases_remain_in_e2e_suite(tmp_path):
    setup = autofuse_setup("e2e", "off", "on")
    setup += "codegen_e2e_st() { printf 'TRACE codegen e2e st\\n'; }\n"
    result = run_shell_main(ROOT / "scripts/test/run_autofuse_test.sh", setup, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TRACE pyautofuse dependency build" in result.stdout
    assert "TRACE codegen e2e st" in result.stdout
    make_calls = [
        line for line in result.stdout.splitlines() if line.startswith("TRACE make ")
    ]
    assert len(make_calls) == 1
    assert (
        "test_load_broadcast_store_codegen test_load_broadcast_multi_axis_store_codegen"
        in make_calls[0]
    )
    assert "/tests/ut/e2e" in result.stdout
    assert "TRACE coverage" in result.stdout


def test_all_ut_still_runs_broadcast_codegen_cases(tmp_path):
    setup = autofuse_setup("all", "on", "off")
    for helper in (
        "build_ut_att",
        "build_ut_optimize",
        "build_ut_autofuse_utils",
        "build_ut_common",
        "build_test",
        "build_test_ascendc_api_test",
        "py_module_ut",
    ):
        setup += f"{helper}() {{ printf 'TRACE {helper}\\n'; }}\n"
    result = run_shell_main(ROOT / "scripts/test/run_autofuse_test.sh", setup, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TRACE pyautofuse dependency build" in result.stdout
    make_calls = [
        line for line in result.stdout.splitlines() if line.startswith("TRACE make ")
    ]
    assert len(make_calls) == 1
    assert (
        "test_load_broadcast_store_codegen test_load_broadcast_multi_axis_store_codegen"
        in make_calls[0]
    )
    assert "/tests/ut/e2e" in result.stdout
    assert "TRACE coverage" in result.stdout


def test_build_failure_is_not_hidden_by_resource_logging(tmp_path):
    setup = autofuse_setup("ascendc_api", "on", "off") + "MOCK_MAKE_STATUS=17\n"
    result = run_shell_main(ROOT / "scripts/test/run_autofuse_test.sh", setup, tmp_path)
    assert result.returncode != 0
    assert "TRACE make -j8 test_ascendc_api test_ascendc_api_v35" in result.stdout
    assert "TRACE ctest " not in result.stdout
    assert "TRACE coverage" not in result.stdout
    assert "[autofuse resources][exit]" in result.stdout


def test_configuration_failure_does_not_build_stale_targets(tmp_path):
    setup = autofuse_setup("ascendc_api", "on", "off") + "MOCK_CMAKE_STATUS=19\n"
    result = run_shell_main(ROOT / "scripts/test/run_autofuse_test.sh", setup, tmp_path)
    assert result.returncode != 0
    assert (tmp_path / "build").is_dir()
    assert "TRACE cmake " in result.stdout
    assert "TRACE make " not in result.stdout
    assert "TRACE ctest " not in result.stdout
    assert "[autofuse resources][exit]" in result.stdout
