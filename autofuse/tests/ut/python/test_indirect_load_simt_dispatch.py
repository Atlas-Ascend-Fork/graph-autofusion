# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Execute production dispatch control flow with launch/MTE observation stubs.

This host-only test checks thread selection, not SIMT execution or performance.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
HEADER = ROOT / "autofuse/v35/ascendc/api_regbase/indirect_load_simt.h"


def _definition(source, name):
    end = source.index(name + "(")
    start = source.rfind("template <", 0, end)
    opening = source.index("{", end)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_dispatch_thread_tiers_and_mte_priority(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required")
    source = HEADER.read_text()
    driver = tmp_path / "dispatch.cpp"
    driver.write_text(
        r"""
#include <cstdint>
#include <cstdio>
#define __aicore__
#define __gm__
static uint32_t launched;
static uint32_t mte_calls;
static bool mte_handles;
struct Body { struct OutputTargets {}; };
template <typename OffsetT, bool Embedding> struct Policy {
  using OffsetType = OffsetT;
  static constexpr bool kEmbedding = Embedding;
};
template <uint32_t Threads, typename X, typename FusedBody, typename Context, typename AddressPolicy>
void LaunchIndirectLoadSimt(X *, typename FusedBody::OutputTargets, Context, uint32_t,
                           typename AddressPolicy::OffsetType, AddressPolicy) {
  launched = Threads;
}
template <typename X, typename FusedBody, typename Context, typename AddressPolicy>
bool TryIndirectLoadSimtEmbeddingMte(X *, typename FusedBody::OutputTargets, Context, uint32_t,
                                    typename AddressPolicy::OffsetType, AddressPolicy) {
  ++mte_calls;
  return mte_handles;
}
"""
        + _definition(source, "IndirectLoadUse2048Threads")
        + "\n"
        + _definition(source, "DispatchIndirectLoadSimt")
        + r"""
template <typename OffsetT, bool Embedding> int Check() {
  const uint32_t sizes[] = {1, 2, 31, 32, 33, 64, 127, 128, 129, 256, 257, 512,
                            513, 1024, 1025, 2048, 2049};
  for (auto n : sizes) {
    const uint32_t expected = n <= 32 ? 32 : n <= 128 ? 128 : n <= 256 ? 256 :
                              n <= 512 ? 512 : n <= 1024 || sizeof(OffsetT) == 8 ? 1024 : 2048;
    launched = mte_calls = 0;
    mte_handles = false;
    DispatchIndirectLoadSimt<float, Body>(nullptr, {}, 0, n, OffsetT(17), Policy<OffsetT, Embedding>{});
    if (launched != expected || mte_calls != uint32_t(Embedding)) {
      printf("FAIL size=%u width=%zu embedding=%d launched=%u expected=%u\n",
             n, sizeof(OffsetT), Embedding, launched, expected);
      return 1;
    }
    if constexpr (Embedding) {
      launched = mte_calls = 0;
      mte_handles = true;
      DispatchIndirectLoadSimt<float, Body>(nullptr, {}, 0, n, OffsetT(17), Policy<OffsetT, Embedding>{});
      if (launched != 0 || mte_calls != 1) return 2;
    }
  }
  return 0;
}
int main() {
  int result = Check<uint32_t, false>() | Check<uint64_t, false>() |
               Check<uint32_t, true>() | Check<uint64_t, true>();
  if (!result) puts("PASS: 68 dispatch cases + 34 MTE priority cases");
  return result;
}
"""
    )
    binary = tmp_path / "dispatch"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(driver),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 68 dispatch cases + 34 MTE priority cases" in result.stdout
