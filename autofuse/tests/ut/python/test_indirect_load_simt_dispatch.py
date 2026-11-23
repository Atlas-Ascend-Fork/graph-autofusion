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
bool TryIndirectLoadSimtEmbeddingRows(X *, typename FusedBody::OutputTargets, Context, uint32_t,
                                     typename AddressPolicy::OffsetType, AddressPolicy) { return false; }
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


def test_embedding_rows_launch_guards(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required")
    source = HEADER.read_text()
    driver = tmp_path / "rows.cpp"
    driver.write_text(
        r"""
#include <cstdint>
#include <type_traits>
#include <initializer_list>
#define __aicore__
#define __gm__
static int launches;
namespace Simt {
struct Dim3 { explicit Dim3(unsigned) {} };
template <auto Fn, typename... Args> void VF_CALL(Args...) { ++launches; }
}
template <uint32_t, typename, typename, typename, typename>
void IndirectLoadSimtEmbeddingRowsKernel() {}
struct Body {
  static constexpr unsigned kGmOutputCount=1, kUbOutputCount=0;
  struct OutputTargets {};
};
struct Policy {
  using OffsetType=uint32_t;
  static constexpr bool kEmbedding=true;
  uint32_t shape[6]={64,512,512,1,1,0};
  uint32_t input_axis_stride=512;
};
"""
        + _definition(source, "TryIndirectLoadSimtEmbeddingRows")
        + r"""
int main() {
 Policy p;
 auto run=[&](uint32_t size,uint32_t offset) {
  return TryIndirectLoadSimtEmbeddingRows<float,Body>(nullptr,{},0,size,offset,p);
 };
 if(!run(32768,0)||launches!=1) return 1;
 if(!run(65536,512)||launches!=2) return 2;
 if(run(32767,0)||run(32769,0)||run(32768,1)) return 3;
 p.shape[1]=257;if(run(32768,0))return 4;p.shape[1]=512;
 p.shape[3]=2;if(run(32768,0))return 5;p.shape[3]=1;
 p.shape[4]=2;if(run(32768,0))return 6;p.shape[4]=1;
 p.input_axis_stride=513;if(run(32768,0))return 7;
 if(TryIndirectLoadSimtEmbeddingRows<double,Body>(nullptr,{},0,16384,0,p))return 8;
 for (uint32_t width : {128U, 256U, 512U, 1024U}) {
  p.shape[1]=width; p.input_axis_stride=width;
  if(!run(64*width,width))return 9;
 }
 return launches==6?0:10;
}
"""
    )
    binary = tmp_path / "rows"
    subprocess.run([compiler, "-std=c++17", str(driver), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("payload", [128, 256, 512, 1024])
def test_embedding_rows_kernel_coordinates(tmp_path, payload):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required")
    source = HEADER.read_text()
    driver = tmp_path / "coordinates.cpp"
    driver_source = (
        r"""
#include <cstdint>
#include <vector>
#include <cassert>
#define __simt_vf__
#define __aicore__
#define __gm__
#define LAUNCH_BOUND(x)
struct { uint32_t x; } threadIdx, blockDim;
struct Policy { using OffsetType=uint32_t; };
struct Context { const int64_t *indices; };
struct Body {
 struct OutputTargets { float *output; };
 static int64_t Index(uint32_t oi,uint32_t row,Context c) {
  assert(oi==row*512); return c.indices[row];
 }
 static float Outputs(float v,uint32_t oi,uint32_t index,uint32_t axis,uint32_t prefix,Context) {
  assert(index==axis && axis==prefix);
  uint32_t col=oi-prefix*512; assert(col<512);
  return v+float(col);
 }
 static void Store(OutputTargets t,uint32_t oi,uint32_t local,Context,float v) {
  assert(oi-local==512);t.output[oi]=v;
 }
};
"""
        + _definition(source, "IndirectLoadSimtEmbeddingRowsKernel")
        + r"""
int main() {
 std::vector<float> input(40*512),output(40*512,-1);
 std::vector<int64_t> indices(40);
 for(unsigned r=0;r<40;++r)indices[r]=39-r;
 for(unsigned i=0;i<input.size();++i)input[i]=float(i);
 blockDim.x=2048;
 for(threadIdx.x=0;threadIdx.x<2048;++threadIdx.x)
  IndirectLoadSimtEmbeddingRowsKernel<512,float,Body,Context,Policy>(
    input.data(),{output.data()},{indices.data()},32*512,512);
 for(unsigned r=1;r<33;++r)for(unsigned col=0;col<512;++col)
  assert(output[r*512+col]==input[indices[r]*512+col]+float(col));
 assert(output[0]==-1 && output[33*512]==-1);
}
"""
    )
    driver.write_text(driver_source.replace("512", str(payload)))
    binary = tmp_path / "coordinates"
    subprocess.run([compiler, "-std=c++17", str(driver), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
