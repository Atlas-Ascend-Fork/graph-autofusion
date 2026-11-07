# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build-resource integrity checks; algorithm behavior is exercised by ATT ST."""

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "embed_axes_solver", ROOT / "autofuse/att/cmake/embed_axes_solver.py"
)
EMBED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EMBED)


def resource(path, body):
    path.write_text(EMBED.BEGIN + body + EMBED.END, encoding="utf-8")
    return path


def callback_scope_guard():
    condition = " || ".join(
        f"defined({name})" for name in sorted(EMBED.CALLBACK_MACROS)
    )
    return f'\n#if {condition}\n#error "callback policy leaked outside shared body"\n#endif\n'


@pytest.mark.parametrize(
    "equal_order,expected",
    [(False, "before\nnormal\nafter\n"), (True, "before\nequal\nafter\n")],
)
def test_selects_declared_compile_variant_without_changing_whitespace(
    tmp_path, equal_order, expected
):
    path = resource(
        tmp_path / "core.inc",
        "before\n#if AF_AXES_EQUAL_ORDER\nequal\n#else\nnormal\n#endif\nafter\n",
    )
    assert EMBED.select_source(path, equal_order) == expected


@pytest.mark.parametrize(
    "body",
    [
        "#else\n",
        "#endif\n",
        "#if AF_AXES_EQUAL_ORDER\nx\n",
        "#if OTHER_FLAG\nx\n#endif\n",
        "#include <other.h>\n",
        "#define OTHER_MACRO 1\n",
        "#undef OTHER_MACRO\n",
        "#if AF_AXES_EQUAL_ORDER\n#else\n#else\n#endif\n",
    ],
)
def test_malformed_or_unhandled_directive_fails_build(tmp_path, body):
    path = resource(tmp_path / "core.inc", body)
    with pytest.raises(ValueError):
        EMBED.select_source(path, False)


@pytest.mark.parametrize(
    "text",
    [
        "",
        EMBED.BEGIN,
        EMBED.END,
        EMBED.BEGIN + EMBED.BEGIN + EMBED.END,
        EMBED.END + EMBED.BEGIN,
    ],
)
def test_invalid_resource_boundaries_fail_build(tmp_path, text):
    path = tmp_path / "core.inc"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        EMBED.select_source(path, True)


def test_inactive_nested_branch_does_not_leak(tmp_path):
    body = "#if AF_AXES_EQUAL_ORDER\n#if AF_AXES_EQUAL_ORDER\na\n#else\nb\n#endif\n#else\nc\n#endif\n"
    path = resource(tmp_path / "core.inc", body)
    assert EMBED.select_source(path, False) == "c\n"
    assert EMBED.select_source(path, True) == "a\n"


@pytest.mark.parametrize("explicit", [False, True])
def test_callback_policy_preserves_directives_in_selected_branch(tmp_path, explicit):
    body = (
        "#if AF_AXES_EXPLICIT_EVAL\n"
        "#define AF_AXES_GET_PERF CheckedGetPerf\n"
        "#else\n#define AF_AXES_GET_PERF GetPerf\n#endif\n"
        "AF_AXES_GET_PERF();\n#undef AF_AXES_GET_PERF\n"
    )
    path = resource(tmp_path / "core.inc", body)
    expected = "CheckedGetPerf" if explicit else "GetPerf"
    assert EMBED.select_source(path, False, explicit) == (
        f"#define AF_AXES_GET_PERF {expected}\n"
        "AF_AXES_GET_PERF();\n#undef AF_AXES_GET_PERF\n"
    )


def test_generation_is_reproducible_and_refreshes_changed_source(tmp_path):
    scalar_dir = tmp_path / "scalar"
    scalar_dir.mkdir()
    for name in ("expression_helpers.inc", "scalar_helpers.inc", "axes_ceil_div.inc"):
        resource(scalar_dir / name, "// helper\n")
    tmp_path = tmp_path / "axes_reorder"
    tmp_path.mkdir()
    for part in ("head", "body"):
        resource(tmp_path / f"axes_reorder_{part}.inc", part + "\n")
    output = tmp_path / "generated.h"
    EMBED.generate(tmp_path, output)
    first, timestamp = output.read_bytes(), output.stat().st_mtime_ns
    EMBED.generate(tmp_path, output)
    assert output.read_bytes() == first
    assert output.stat().st_mtime_ns == timestamp
    resource(tmp_path / "axes_reorder_body.inc", "changed\n")
    EMBED.generate(tmp_path, output)
    assert output.read_bytes() != first
    assert b"changed\n" in output.read_bytes()


def test_raw_delimiter_collision_is_rejected(tmp_path):
    scalar_dir = tmp_path / "scalar"
    scalar_dir.mkdir()
    for name in ("expression_helpers.inc", "scalar_helpers.inc", "axes_ceil_div.inc"):
        resource(scalar_dir / name, "// helper\n")
    tmp_path = tmp_path / "axes_reorder"
    tmp_path.mkdir()
    for part in ("head", "body"):
        resource(tmp_path / f"axes_reorder_{part}.inc", ')AF_AXES"\n')
    with pytest.raises(ValueError):
        EMBED.generate(tmp_path, tmp_path / "generated.h")


@pytest.mark.parametrize("ordinary_source", [False, True])
@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_native_and_embedded_scalar_contract(tmp_path, ordinary_source, optimization):
    core = ROOT / "autofuse/att/core"
    generated = tmp_path / "resources.h"
    EMBED.generate(core / "axes_reorder", generated)
    source = "#include <cmath>\n#include <cstdint>\n#include <type_traits>\n"
    for filename, symbol in (
        ("expression_helpers.inc", "ExpressionHelpers"),
        ("scalar_helpers.inc", "ScalarHelpers"),
        ("axes_ceil_div.inc", "AxesCeilDiv"),
    ):
        if ordinary_source:
            source += f'#include "{core / "scalar" / filename}"\n'
        else:
            match = re.search(
                rf'constexpr char k{symbol}\[\] = R"AF_AXES\((.*?)\)AF_AXES";',
                generated.read_text(),
                re.S,
            )
            assert match is not None
            source += match.group(1)
    source += r"""
static_assert(std::is_same<decltype(Max(int32_t{0}, uint32_t{0})), uint32_t>::value, "conditional type");
static_assert(std::is_same<decltype(Ceiling(float{0})), float>::value, "float width");
static_assert(std::is_same<decltype(TernaryOp(true, 1, 2)), double>::value, "ternary conversion");
int main() {
    // Preserve historical truncation/epsilon semantics, including negative values.
    if (Ceiling(1.0 + 1e-9) != 1.0 || Ceiling(1.25) != 2.0 || Ceiling(-1.25) != 0.0 || Floor(-1.25) != -1.0) {
        return 1;
    }
    if (CeilDiv(17, 8) != 3 || Mod(-7, 3) != -1 || Mod(-7.5, 3) != -1.5 || Mod(7, 2.5) != 2.0) {
        return 2;
    }
    if (Max(-1, 2U) != 2U || Min(-1, 2U) != UINT32_MAX || TernaryOp(true, -1, 2U) != 4294967295.0) {
        return 3;
    }
    int calls = 0;
    if (LogicAnd(false, ++calls) || !LogicOr(true, ++calls) || calls != 0 || Rational(1, 4) != 0.25) {
        return 4;
    }
    return 0;
}
"""
    path, binary = tmp_path / "probe.cpp", tmp_path / "probe"
    path.write_text(source)
    subprocess.run(
        [
            os.environ.get("CXX", "g++"),
            "-std=c++17",
            optimization,
            "-Wall",
            "-Wextra",
            "-Werror",
            str(path),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=10
    )


@pytest.fixture(scope="session")
def pgo_solver_source(tmp_path_factory):
    output = tmp_path_factory.mktemp("pgo_generator")
    att = ROOT / "autofuse/att"
    EMBED.generate(att / "core/axes_reorder", output / "axes_solver_resources.h")
    exporter = output / "export.cpp"
    exporter.write_text(
        '#include <iostream>\n#include "generator/solver_pass/axes_reorder_solver_code.h"\n'
        "int main() { std::cout << att::GetAxesSolverPgoSolverHead(16) << att::GetAxesSolverPgoSolverFunc(); }\n"
    )
    binary = output / "export"
    result = subprocess.run(
        [
            os.environ.get("CXX", "g++"),
            "-std=c++17",
            "-I",
            str(att),
            "-I",
            str(output),
            str(exporter),
            str(att / "generator/solver_pass/axes_reorder_solver_code.cpp"),
            str(att / "generator/solver_pass/axes_reorder_pgo_solver_code.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=10, check=True
    ).stdout


@pytest.mark.parametrize("equal_order", [False, True])
@pytest.mark.parametrize("ordinary_source", [False, True])
@pytest.mark.parametrize("sanitize", [False, True])
def test_evaluation_failure_stops_solver_and_preserves_first_error(
    tmp_path, equal_order, ordinary_source, pgo_solver_source, sanitize
):
    core = ROOT / "autofuse/att/core"
    sources = ""
    for filename in (
        "expression_helpers.inc",
        "scalar_helpers.inc",
        "axes_ceil_div.inc",
    ):
        path = core / "scalar" / filename
        sources += (
            f'#include "{path}"\n' if ordinary_source else EMBED.read_resource(path)
        )
    for filename in ("axes_reorder_head.inc", "axes_reorder_body.inc"):
        path = core / "axes_reorder" / filename
        sources += (
            f'#include "{path}"\n'
            if ordinary_source
            else EMBED.select_source(path, equal_order, explicit_eval=True)
        )
    (tmp_path / "axes_eval_core.h").write_text(
        sources + callback_scope_guard() + pgo_solver_source
    )
    binary = tmp_path / "eval_contract"
    source = Path(__file__).parent / "data/axes_eval_contract.cpp"
    compile_result = subprocess.run(
        [
            os.environ.get("CXX", "g++"),
            "-std=c++17",
            *(
                ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
                if sanitize
                else ["-O2"]
            ),
            "-pthread",
            f"-DAF_AXES_EQUAL_ORDER={int(equal_order)}",
            "-DAF_AXES_EXPLICIT_EVAL=1",
            "-I",
            str(tmp_path),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    environment = os.environ.copy()
    if sanitize:
        environment.update(
            ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
            UBSAN_OPTIONS="halt_on_error=1",
        )
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=30, env=environment
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("fault_points=") == 12


@pytest.mark.parametrize("equal_order", [False, True])
@pytest.mark.parametrize("ordinary_source", [False, True])
def test_legacy_native_keeps_no_exception_compile_contract(
    tmp_path, equal_order, ordinary_source, pgo_solver_source
):
    core = ROOT / "autofuse/att/core"
    sources = ""
    for filename in (
        "expression_helpers.inc",
        "scalar_helpers.inc",
        "axes_ceil_div.inc",
    ):
        path = core / "scalar" / filename
        sources += (
            f'#include "{path}"\n' if ordinary_source else EMBED.read_resource(path)
        )
    for filename in ("axes_reorder_head.inc", "axes_reorder_body.inc"):
        path = core / "axes_reorder" / filename
        sources += (
            f'#include "{path}"\n'
            if ordinary_source
            else EMBED.select_source(path, equal_order)
        )
    (tmp_path / "axes_native_core.h").write_text(
        sources + callback_scope_guard() + pgo_solver_source
    )
    binary = tmp_path / "native_contract"
    result = subprocess.run(
        [
            os.environ.get("CXX", "g++"),
            "-std=c++17",
            "-O2",
            "-fno-exceptions",
            f"-DAF_AXES_EQUAL_ORDER={int(equal_order)}",
            "-DAF_AXES_EXPLICIT_EVAL=0",
            "-I",
            str(tmp_path),
            str(Path(__file__).parent / "data/axes_native_contract.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
