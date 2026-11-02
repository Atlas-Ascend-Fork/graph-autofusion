/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <tuple>
#include <type_traits>
#include "gtest/gtest.h"

// Compile the actual dispatch helper with native kernel declarations in a private namespace.
// No device instructions execute: the StreamK stub records the batch passed by the helper.
#ifndef __aicore__
#define __aicore__
#define AF_TEST_DEFINED_AICORE
#endif
namespace tensor_common_dispatch_test {
using GM_ADDR = uint8_t *;
constexpr int8_t MAT_MUL_BASIC = 0;
constexpr int8_t MAT_MUL_STREAM_K = 1;
constexpr int8_t MAT_MUL_SLICE = 5;
constexpr int8_t MAT_MUL_BASIC_SPLIT_K = 6;
constexpr int8_t MAT_MUL_SK_SPLIT_K = 7;
constexpr int8_t MAT_MUL_NO_FULL_LOAD = 0;
constexpr int8_t MAT_MUL_A_FULL_LOAD = 1;
constexpr int8_t MAT_MUL_B_FULL_LOAD = 2;
constexpr int8_t MAT_MUL_ON_THE_FLY = 0;
struct MatMulV3BasicTilingData {
  uint32_t m, n, k, mL1, nL1, kL1, baseM, baseN, baseK, mTailCnt, nTailCnt;
  uint32_t mBaseTailSplitCnt, nBaseTailSplitCnt, mTailMain, nTailMain, l1BufferNum, l0cDB;
  uint32_t rowStride, l2CacheDisable, sliceM, srcNdStride, usedCoreNum, innerBatch;
  uint8_t mmadParam;
};
namespace AscendC {
namespace Std = std;
inline uint32_t GetBlockNum() {
  return 1;
}
}  // namespace AscendC
namespace asc {
namespace te {
template <class... T>
using shape = std::tuple<T...>;
}  // namespace te
}  // namespace asc
namespace Blaze {
namespace Gemm {
enum class MatMulL0C2Out : uint8_t { ON_THE_FLY = 0, ND_FIXPIPE_1_2 = 2 };
struct KernelMmadMultiBlockBasic;
struct KernelMmadMultiBlockAFullLoad;
struct KernelMmadMultiBlockBFullLoad;
template <class T>
struct IsWeightNz : std::false_type {};
template <int8_t, int8_t, class, uint64_t>
struct MatmulMultiBlockBasic;
template <int8_t, bool, class, uint64_t>
struct MatmulMultiBlockBasicSplitK;
template <int8_t, int8_t, class>
struct MatmulMultiBlockAFullLoad;
template <int8_t, int8_t, class>
struct MatmulMultiBlockBFullLoad;
namespace Block {
template <class, int8_t, bool, bool>
class TensorCommonBlockScheduler;
template <class...>
class BlockMmad;
}  // namespace Block
namespace Kernel {
template <class...>
class GemmUniversal;
}  // namespace Kernel
}  // namespace Gemm
namespace Epilogue {
namespace Block {
struct BlockEpilogueEmpty;
}  // namespace Block
}  // namespace Epilogue
}  // namespace Blaze

int64_t observed_batch = -1;
GM_ADDR observed_workspace = nullptr;
namespace MatmulV3Advanced {
template <class, class, class, class, class, class, class, int8_t>
void MatMulAL1FullLoadKernel(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, const MatMulV3BasicTilingData &, int64_t);
template <class, class, class, class, class, class, class, int8_t, int8_t = 0>
void MatMulBL1FullLoadKernel(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, const MatMulV3BasicTilingData &, int64_t);
template <class, class, class, class, class, class, class, int8_t, int8_t, uint64_t>
void MatMulBasicKernel(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, const MatMulV3BasicTilingData &, int64_t);
template <class, class, class, class, class, class, class>
void MatMulBasicSplitKKernel(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, const MatMulV3BasicTilingData &, int64_t);
template <class, class, class, class, class, class, Blaze::Gemm::MatMulL0C2Out>
void MatMulStreamKKernel(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR workspace, const MatMulV3BasicTilingData &,
                         int64_t batch = 1) {
  observed_batch = batch;
  observed_workspace = workspace;
}
template <class, class, class, class, class, class, Blaze::Gemm::MatMulL0C2Out>
void MatMulStreamKSplitKKernel(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, const MatMulV3BasicTilingData &);
template <class, class, class, class, class, class, class, int8_t, int8_t>
void MatMulFixpipeOptiTensorKernel(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, const MatMulV3BasicTilingData &,
                                   int64_t);
}  // namespace MatmulV3Advanced

#include "../../../../v35/ascendc/api_cube/matmul/mat_mul_tensor_common.h"

struct MmadParamsBase {
  GM_ADDR aGmAddr{}, bGmAddr{}, cGmAddr{}, biasGmAddr{};
  uint64_t mL1{}, kL1{}, mL0{}, nL0{}, kL0{}, l1Stages{}, l0cStages{}, rowStride{};
};
struct BasicParams : MmadParamsBase {
  uint64_t nL1{};
};
struct SplitKParams : BasicParams {
  uint64_t k{};
};
struct AFullParams : MmadParamsBase {
  uint64_t oriK{};
};
struct BFullParams : BasicParams {
  uint64_t oriK{};
};

TEST(TensorCommonDispatch, NativeMmadLayoutsKeepTheirKAndNTileFields) {
  MatMulV3BasicTilingData tiling{};
  tiling.k = 512;
  tiling.nL1 = 64;
  BasicParams basic{};
  SplitKParams split{};
  AFullParams aFull{};
  BFullParams bFull{};
  MatmulV3Advanced::FillTensorCommonMmadParams<MAT_MUL_BASIC, MAT_MUL_NO_FULL_LOAD>(basic, nullptr, nullptr, nullptr,
                                                                                    nullptr, tiling);
  MatmulV3Advanced::FillTensorCommonMmadParams<MAT_MUL_BASIC_SPLIT_K, MAT_MUL_NO_FULL_LOAD>(split, nullptr, nullptr,
                                                                                            nullptr, nullptr, tiling);
  MatmulV3Advanced::FillTensorCommonMmadParams<MAT_MUL_BASIC, MAT_MUL_A_FULL_LOAD>(aFull, nullptr, nullptr, nullptr,
                                                                                   nullptr, tiling);
  MatmulV3Advanced::FillTensorCommonMmadParams<MAT_MUL_BASIC, MAT_MUL_B_FULL_LOAD>(bFull, nullptr, nullptr, nullptr,
                                                                                   nullptr, tiling);
  EXPECT_EQ(basic.nL1, 64U);
  EXPECT_EQ(split.nL1, 64U);
  EXPECT_EQ(split.k, 512U);
  EXPECT_EQ(aFull.oriK, 512U);
  EXPECT_EQ(bFull.nL1, 64U);
  EXPECT_EQ(bFull.oriK, 512U);
}

struct Layout {};
TEST(TensorCommonDispatch, MatmulKeepsNativeStreamKDefaultBatch) {
  MatMulV3BasicTilingData tiling{};
  uint8_t workspace = 0;
  observed_batch = -1;
  MatmulV3Advanced::MatMulTensorCommonKernel<float, float, float, float, Layout, Layout, Layout, MAT_MUL_STREAM_K,
                                             MAT_MUL_NO_FULL_LOAD, MAT_MUL_ON_THE_FLY>(nullptr, nullptr, nullptr,
                                                                                       nullptr, &workspace, tiling);
  EXPECT_EQ(observed_batch, 1);
  EXPECT_EQ(observed_workspace, &workspace);
}

TEST(TensorCommonDispatch, BatchMatmulPreservesExplicitBatch) {
  MatMulV3BasicTilingData tiling{};
  observed_batch = -1;
  MatmulV3Advanced::MatMulTensorCommonKernel<float, float, float, float, Layout, Layout, Layout, MAT_MUL_STREAM_K,
                                             MAT_MUL_NO_FULL_LOAD, 2>(nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                      tiling, 5);
  EXPECT_EQ(observed_batch, 5);
}
}  // namespace tensor_common_dispatch_test
#ifdef AF_TEST_DEFINED_AICORE
#undef __aicore__
#undef AF_TEST_DEFINED_AICORE
#endif
