# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for the private Runtime core-count wrapper.

Nothing here loads a real ``libruntime.so``: every case replaces
``ctypes.CDLL`` so the return codes, the buffer handling and the value checks
can be driven from Python.  The real query belongs to the deployment smoke
test on a machine with CANN installed.
"""

from __future__ import annotations

import ctypes

import pytest

from multistream.common import runtime_platform


# ``RT_ERROR_NOT_FOUND`` as the wrapper sees it: any non-zero code is a
# failure, so the exact value only has to be distinguishable in the message.
_RT_ERROR_NOT_FOUND = 0x07010002


class _FakeGetSocSpec:
    """Stand-in for the bound ``rtGetSocSpec`` C function."""

    def __init__(self, answers):
        # ``answers`` maps a key to either the text to return or an error code.
        self._answers = answers
        self.restype = None
        self.argtypes = None
        self.calls = []

    def __call__(self, label, key, buffer, max_len):
        self.calls.append((label, key, max_len))
        answer = self._answers[key.decode("utf-8")]
        if isinstance(answer, int):
            return answer
        if answer is None:
            # An unterminated value: fill the buffer without a NUL byte.
            buffer.raw = b"7" * max_len
            return 0
        buffer.value = answer.encode("utf-8")
        return 0


class _FakeLibrary:
    def __init__(self, function):
        self.rtGetSocSpec = function


class _LibraryWithoutSymbol:
    def __getattr__(self, name):
        raise AttributeError(name)


def _install(monkeypatch, answers):
    function = _FakeGetSocSpec(answers)
    loads = []

    def fake_cdll(name):
        loads.append(name)
        return _FakeLibrary(function)

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    return function, loads


def test_query_returns_both_counts_and_binds_the_c_signature(monkeypatch):
    function, loads = _install(
        monkeypatch, {"ai_core_cnt": "24", "vector_core_cnt": "48"}
    )
    limits = runtime_platform._query_core_limits()
    assert (limits.ai_core_num, limits.vector_core_num) == (24, 48)
    assert loads == ["libruntime.so"]
    assert function.restype is ctypes.c_int32
    assert function.argtypes == (
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
    )
    assert [key for _label, key, _max_len in function.calls] == [
        b"ai_core_cnt",
        b"vector_core_cnt",
    ]
    assert {label for label, _key, _max_len in function.calls} == {b"SoCInfo"}


def test_library_is_loaded_lazily_and_the_binding_is_reused(monkeypatch):
    function, loads = _install(
        monkeypatch, {"ai_core_cnt": "20", "vector_core_cnt": "40"}
    )
    # Importing the module must not have loaded anything.
    assert loads == []
    runtime_platform._query_core_limits()
    runtime_platform._query_core_limits()
    assert loads == ["libruntime.so"]
    # Only the handle is cached, so every allocation queries the counts again.
    assert len(function.calls) == 4


def test_each_call_uses_its_own_buffer(monkeypatch):
    seen = []

    class _RecordingCDLL(_FakeGetSocSpec):
        def __call__(self, label, key, buffer, max_len):
            # Keep the object alive: a freed buffer's id can be reused.
            seen.append(buffer)
            return super().__call__(label, key, buffer, max_len)

    function = _RecordingCDLL({"ai_core_cnt": "24", "vector_core_cnt": "48"})
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _FakeLibrary(function))
    runtime_platform._query_core_limits()
    assert len(seen) == 2
    assert seen[0] is not seen[1]


def test_missing_library_and_missing_symbol_raise_runtime_error(monkeypatch):
    def failing_cdll(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", failing_cdll)
    with pytest.raises(RuntimeError, match="libruntime.so"):
        runtime_platform._query_core_limits()

    monkeypatch.setattr(runtime_platform, "_get_soc_spec", None)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _LibraryWithoutSymbol())
    with pytest.raises(RuntimeError, match="rtGetSocSpec"):
        runtime_platform._query_core_limits()


@pytest.mark.parametrize("error_code", [_RT_ERROR_NOT_FOUND, 1, -1])
def test_query_error_code_is_reported_with_the_key(monkeypatch, error_code):
    # A Runtime that was never initialized and a SoC whose INI lacks the
    # field both arrive here as a non-zero return code.
    _install(monkeypatch, {"ai_core_cnt": error_code, "vector_core_cnt": "48"})
    with pytest.raises(RuntimeError) as failure:
        runtime_platform._query_core_limits()
    message = str(failure.value)
    assert "ai_core_cnt" in message and str(error_code) in message


def test_vector_core_count_field_error_is_reported(monkeypatch):
    _install(monkeypatch, {"ai_core_cnt": "24", "vector_core_cnt": _RT_ERROR_NOT_FOUND})
    with pytest.raises(RuntimeError, match="vector_core_cnt"):
        runtime_platform._query_core_limits()


def test_unterminated_and_empty_values_are_rejected(monkeypatch):
    _install(monkeypatch, {"ai_core_cnt": None, "vector_core_cnt": "48"})
    with pytest.raises(RuntimeError, match="unterminated"):
        runtime_platform._query_core_limits()

    monkeypatch.setattr(runtime_platform, "_get_soc_spec", None)
    _install(monkeypatch, {"ai_core_cnt": "  ", "vector_core_cnt": "48"})
    with pytest.raises(RuntimeError, match="empty"):
        runtime_platform._query_core_limits()


@pytest.mark.parametrize("text", ["abc", "24.5", "", "0x18"])
def test_non_integer_value_is_rejected(monkeypatch, text):
    _install(monkeypatch, {"ai_core_cnt": text, "vector_core_cnt": "48"})
    with pytest.raises(RuntimeError, match="ai_core_cnt"):
        runtime_platform._query_core_limits()


def test_zero_ai_core_count_is_invalid_while_zero_vector_count_is_valid(monkeypatch):
    # A failed query must never look like "this chip has no cores".
    _install(monkeypatch, {"ai_core_cnt": "0", "vector_core_cnt": "48"})
    with pytest.raises(RuntimeError, match="out-of-range"):
        runtime_platform._query_core_limits()

    # Ascend910A and other chips without separate Vector cores report 0 here.
    monkeypatch.setattr(runtime_platform, "_get_soc_spec", None)
    _install(monkeypatch, {"ai_core_cnt": "32", "vector_core_cnt": "0"})
    limits = runtime_platform._query_core_limits()
    assert (limits.ai_core_num, limits.vector_core_num) == (32, 0)


@pytest.mark.parametrize("key", ["ai_core_cnt", "vector_core_cnt"])
def test_negative_count_is_rejected(monkeypatch, key):
    answers = {"ai_core_cnt": "24", "vector_core_cnt": "48"}
    answers[key] = "-1"
    _install(monkeypatch, answers)
    with pytest.raises(RuntimeError, match="out-of-range"):
        runtime_platform._query_core_limits()


def test_core_limits_snapshot_is_immutable(monkeypatch):
    _install(monkeypatch, {"ai_core_cnt": "24", "vector_core_cnt": "48"})
    limits = runtime_platform._query_core_limits()
    with pytest.raises(AttributeError):
        limits.ai_core_num = 1
    with pytest.raises(AttributeError):
        del limits.vector_core_num
    assert limits == runtime_platform._CoreLimits(24, 48)
    assert hash(limits) == hash(runtime_platform._CoreLimits(24, 48))
    assert "ai_core_num=24" in repr(limits)


def test_plan_fails_when_the_runtime_query_fails(monkeypatch):
    import multistream

    _install(monkeypatch, {"ai_core_cnt": _RT_ERROR_NOT_FOUND, "vector_core_cnt": "48"})
    graph = multistream.DAGGraph("no_runtime")
    graph.add_node("n0", "Op")
    with pytest.raises(RuntimeError, match="rtGetSocSpec"):
        multistream.plan(graph)
