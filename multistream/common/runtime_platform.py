# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Private ctypes wrapper reading the hardware core counts from Runtime.

The package stays pure Python: ``libruntime.so`` comes from the host's CANN
installation and is loaded lazily, so ``import multistream`` never touches
Runtime.  The queried SoC is whichever target Runtime currently holds; the
host sets it up (online initialization, or ``set_soc_version`` for GE's
offline entry) before calling ``plan``.  This module never changes that
shared state.
"""

from __future__ import annotations

import ctypes
import threading

from .log import logger


_RUNTIME_LIBRARY = "libruntime.so"
_SOC_INFO_LABEL = "SoCInfo"
_AI_CORE_KEY = "ai_core_cnt"
_VECTOR_CORE_KEY = "vector_core_cnt"
# ``rtGetSocSpec`` copies a decimal string plus its terminator.  The INI
# values are small counts; the buffer only has to be larger than the longest
# spec text so that the call cannot fail with RT_ERROR_INVALID_VALUE.
_VALUE_BUFFER_SIZE = 128

_binding_lock = threading.Lock()
_get_soc_spec = None


class _CoreLimits:
    """Immutable snapshot of one SoC's core counts."""

    __slots__ = ("ai_core_num", "vector_core_num")

    def __init__(self, ai_core_num: int, vector_core_num: int) -> None:
        object.__setattr__(self, "ai_core_num", ai_core_num)
        object.__setattr__(self, "vector_core_num", vector_core_num)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("_CoreLimits is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("_CoreLimits is immutable")

    def __repr__(self) -> str:
        return (
            f"_CoreLimits(ai_core_num={self.ai_core_num}, "
            f"vector_core_num={self.vector_core_num})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _CoreLimits):
            return NotImplemented
        return (
            self.ai_core_num == other.ai_core_num
            and self.vector_core_num == other.vector_core_num
        )

    def __hash__(self) -> int:
        return hash((self.ai_core_num, self.vector_core_num))


def _bind_get_soc_spec():
    """Load Runtime and bind ``rtGetSocSpec`` once per process.

    Only the library handle and the function binding are cached.  Caching the
    counts themselves would need an invalidation protocol for a host that
    switches targets, and Runtime already caches the platform configuration.
    The lock covers the binding alone; keeping the target stable during a
    query remains the host's responsibility.
    """

    global _get_soc_spec
    if _get_soc_spec is not None:
        return _get_soc_spec
    with _binding_lock:
        if _get_soc_spec is not None:
            return _get_soc_spec
        try:
            library = ctypes.CDLL(_RUNTIME_LIBRARY)
        except OSError as exc:
            raise RuntimeError(
                f"cannot load {_RUNTIME_LIBRARY}; multistream needs the CANN "
                f"Runtime of the host environment: {exc}"
            ) from exc
        try:
            function = library.rtGetSocSpec
        except AttributeError as exc:
            raise RuntimeError(
                f"{_RUNTIME_LIBRARY} does not provide rtGetSocSpec; the "
                "deployed CANN version is too old for multistream"
            ) from exc
        function.restype = ctypes.c_int32
        function.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
        )
        _get_soc_spec = function
        return _get_soc_spec


def _query_spec_text(label: str, key: str) -> str:
    function = _bind_get_soc_spec()
    # A fresh buffer per call: the value is only valid for this query and a
    # shared buffer would be visible to another thread mid-update.
    buffer = ctypes.create_string_buffer(_VALUE_BUFFER_SIZE)
    error = function(
        label.encode("utf-8"), key.encode("utf-8"), buffer, _VALUE_BUFFER_SIZE
    )
    if error != 0:
        raise RuntimeError(
            f"rtGetSocSpec({label}, {key}) failed with error code {error}; "
            "check that Runtime is initialized and that the target SoC is set"
        )
    raw = buffer.raw
    if b"\x00" not in raw:
        raise RuntimeError(
            f"rtGetSocSpec({label}, {key}) returned an unterminated value"
        )
    text = raw[: raw.index(b"\x00")].decode("utf-8", errors="replace").strip()
    if not text:
        raise RuntimeError(f"rtGetSocSpec({label}, {key}) returned an empty value")
    return text


def _query_core_count(key: str, *, allow_zero: bool) -> int:
    text = _query_spec_text(_SOC_INFO_LABEL, key)
    try:
        value = int(text)
    except ValueError as exc:
        raise RuntimeError(
            f"rtGetSocSpec({_SOC_INFO_LABEL}, {key}) returned a non-integer "
            f"value: {text!r}"
        ) from exc
    lower_bound = 0 if allow_zero else 1
    if value < lower_bound:
        raise RuntimeError(
            f"rtGetSocSpec({_SOC_INFO_LABEL}, {key}) returned an out-of-range "
            f"value: {value}"
        )
    return value


def _query_core_limits() -> _CoreLimits:
    """Return the current target SoC's AI Core and Vector Core counts.

    The Cube limit comes from ``ai_core_cnt`` rather than ``cube_core_cnt``:
    the latter only exists in the newer INI files, so reading it would make
    the query fail on Ascend910A, 310P and other deployed platforms.
    ``vector_core_cnt`` is defined everywhere and is legitimately 0 on chips
    without separate Vector cores, while ``ai_core_cnt`` of 0 is rejected so
    that a failed query can never look like "no resources".

    Any failure raises ``RuntimeError``.  Failing here rather than skipping
    the query establishes the Runtime dependency from the first release, so
    a later version does not have to turn a tolerated situation into an
    error.
    """

    ai_core_num = _query_core_count(_AI_CORE_KEY, allow_zero=False)
    vector_core_num = _query_core_count(_VECTOR_CORE_KEY, allow_zero=True)
    limits = _CoreLimits(ai_core_num, vector_core_num)
    logger.info(
        "queried core limits from Runtime: ai_core_cnt=%d, vector_core_cnt=%d",
        limits.ai_core_num,
        limits.vector_core_num,
    )
    return limits


__all__ = []
