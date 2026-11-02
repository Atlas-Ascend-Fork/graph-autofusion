/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MAT_MUL_TENSOR_COMMON_H
#define MAT_MUL_TENSOR_COMMON_H

namespace MatmulV3Advanced {

// Native block parameter layouts differ between Basic and full-load MMADs.
template <int8_t Model, int8_t FullLoad, class Params>
__aicore__ inline void FillTensorCommonMmadParams(Params &params, GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                                  const MatMulV3BasicTilingData &tiling) {
  params.aGmAddr = aGM;
  params.bGmAddr = bGM;
  params.cGmAddr = cGM;
  params.biasGmAddr = biasGM;
  params.mL1 = tiling.mL1;
  params.kL1 = tiling.kL1;
  params.mL0 = tiling.baseM;
  params.nL0 = tiling.baseN;
  params.kL0 = tiling.baseK;
  params.l1Stages = tiling.l1BufferNum;
  params.l0cStages = tiling.l0cDB;
  params.rowStride = tiling.rowStride;
  if constexpr (FullLoad != MAT_MUL_A_FULL_LOAD) {
    params.nL1 = tiling.nL1;
  }
  if constexpr (Model == MAT_MUL_BASIC_SPLIT_K) {
    params.k = tiling.k;
  } else if constexpr (FullLoad != MAT_MUL_NO_FULL_LOAD) {
    params.oriK = tiling.k;
  }
}

template <class AType, class BType, class CType, class BiasType, class LayoutA, class LayoutB, class LayoutC,
          int8_t Model, int8_t FullLoad, uint64_t NonContiguous>
__aicore__ inline void MatMulTensorCommonScheduledKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                                         const MatMulV3BasicTilingData &tiling, int64_t batch) {
  using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
  using BlockScheduler =
      Blaze::Gemm::Block::TensorCommonBlockScheduler<ProblemShape, FullLoad, AscendC::Std::is_same_v<BType, float>,
                                                     !Blaze::Gemm::IsWeightNz<LayoutB>::value>;
  using BasicPolicy =
      Blaze::Gemm::MatmulMultiBlockBasic<FullLoad, 0, Blaze::Gemm::KernelMmadMultiBlockBasic, NonContiguous>;
  using SplitKPolicy =
      Blaze::Gemm::MatmulMultiBlockBasicSplitK<FullLoad, true, Blaze::Gemm::KernelMmadMultiBlockBasic, NonContiguous>;
  using AFullPolicy = Blaze::Gemm::MatmulMultiBlockAFullLoad<FullLoad, 0, Blaze::Gemm::KernelMmadMultiBlockAFullLoad>;
  using BFullPolicy =
      Blaze::Gemm::MatmulMultiBlockBFullLoad<MAT_MUL_ON_THE_FLY, 0, Blaze::Gemm::KernelMmadMultiBlockBFullLoad>;
  using BasicOrSplitKPolicy = AscendC::Std::conditional_t<Model == MAT_MUL_BASIC_SPLIT_K, SplitKPolicy, BasicPolicy>;
  using Policy = AscendC::Std::conditional_t<
      FullLoad == MAT_MUL_A_FULL_LOAD, AFullPolicy,
      AscendC::Std::conditional_t<FullLoad == MAT_MUL_B_FULL_LOAD, BFullPolicy, BasicOrSplitKPolicy>>;
  using BlockMmad =
      Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutC>;
  using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, Blaze::Epilogue::Block::BlockEpilogueEmpty,
                                                    BlockScheduler>;
  typename Kernel::Params params{};
  params.problemShape = {tiling.m, tiling.n, tiling.k, batch};
  FillTensorCommonMmadParams<Model, FullLoad>(params.mmadParams, aGM, bGM, biasGM, cGM, tiling);
  params.schParams = {tiling.mL1,
                      tiling.nL1,
                      tiling.kL1,
                      tiling.baseM,
                      tiling.baseN,
                      tiling.baseK,
                      tiling.mTailCnt,
                      tiling.nTailCnt,
                      tiling.mBaseTailSplitCnt,
                      tiling.nBaseTailSplitCnt,
                      tiling.mTailMain,
                      tiling.nTailMain,
                      tiling.mmadParam,
                      static_cast<uint32_t>(tiling.l2CacheDisable),
                      tiling.sliceM,
                      tiling.srcNdStride,
                      tiling.usedCoreNum,
                      tiling.innerBatch};
  Kernel mm;
  mm(params);
}

template <class AType, class BType, class CType, class BiasType, class LayoutA, class LayoutB, class LayoutC,
          int8_t Model, int8_t FullLoad, uint64_t NonContiguous>
__aicore__ inline void MatMulTensorOnTheFlyCommonKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                                        GM_ADDR workspaceGM, const MatMulV3BasicTilingData &tiling,
                                                        int64_t batch) {
  if (AscendC::GetBlockNum() > tiling.usedCoreNum && tiling.usedCoreNum > 0) {
    MatMulTensorCommonScheduledKernel<AType, BType, CType, BiasType, LayoutA, LayoutB, LayoutC, Model, FullLoad,
                                      NonContiguous>(aGM, bGM, biasGM, cGM, tiling, batch);
  } else if constexpr (FullLoad == MAT_MUL_A_FULL_LOAD) {
    MatMulAL1FullLoadKernel<AType, BType, CType, BiasType, LayoutA, LayoutB, LayoutC, FullLoad>(
        aGM, bGM, biasGM, cGM, workspaceGM, tiling, batch);
  } else if constexpr (FullLoad == MAT_MUL_B_FULL_LOAD) {
    MatMulBL1FullLoadKernel<AType, BType, CType, BiasType, LayoutA, LayoutB, LayoutC, FullLoad>(
        aGM, bGM, biasGM, cGM, workspaceGM, tiling, batch);
  } else if constexpr (Model == MAT_MUL_BASIC_SPLIT_K) {
    MatMulBasicSplitKKernel<AType, BType, CType, BiasType, LayoutA, LayoutB, LayoutC>(aGM, bGM, biasGM, cGM,
                                                                                      workspaceGM, tiling, batch);
  } else {
    MatMulBasicKernel<AType, BType, CType, BiasType, LayoutA, LayoutB, LayoutC, 0, 0, NonContiguous>(
        aGM, bGM, biasGM, cGM, workspaceGM, tiling, batch);
  }
}

// An omitted batch denotes one matrix; the native StreamK kernel uses this count directly.
// Static and dynamic Common fusion preserve the native Tensor kernel selected by its tiling key.
template <class AType, class BType, class CType, class BiasType, class LayoutA, class LayoutB, class LayoutC,
          int8_t Model, int8_t FullLoad, int8_t L0C2Out>
__aicore__ inline void MatMulTensorCommonKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                                GM_ADDR workspaceGM, const MatMulV3BasicTilingData &tilingData,
                                                int64_t batch = 1) {
  if constexpr (L0C2Out == MAT_MUL_ON_THE_FLY &&
                (Model == MAT_MUL_BASIC || Model == MAT_MUL_SLICE || Model == MAT_MUL_BASIC_SPLIT_K)) {
    MatMulTensorOnTheFlyCommonKernel<AType, BType, CType, BiasType, LayoutA, LayoutB, LayoutC, Model, FullLoad,
                                     Model == MAT_MUL_SLICE ? 1 : 0>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData,
                                                                     batch);
  } else if constexpr (FullLoad == MAT_MUL_NO_FULL_LOAD && Model == MAT_MUL_STREAM_K) {
    MatMulStreamKKernel<AType, BType, CType, BiasType, LayoutA, LayoutB,
                        static_cast<Blaze::Gemm::MatMulL0C2Out>(L0C2Out)>(aGM, bGM, biasGM, cGM, workspaceGM,
                                                                          tilingData, batch);
  } else if constexpr (FullLoad == MAT_MUL_NO_FULL_LOAD && Model == MAT_MUL_SK_SPLIT_K) {
    MatMulStreamKSplitKKernel<AType, BType, CType, BiasType, LayoutA, LayoutB,
                              static_cast<Blaze::Gemm::MatMulL0C2Out>(L0C2Out)>(aGM, bGM, biasGM, cGM, workspaceGM,
                                                                                tilingData);
  } else if constexpr (FullLoad == MAT_MUL_B_FULL_LOAD && Model == MAT_MUL_BASIC) {
    MatMulBL1FullLoadKernel<AType, BType, CType, BiasType, LayoutA, LayoutB, LayoutC, FullLoad, L0C2Out>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData, batch);
  } else if constexpr (FullLoad == MAT_MUL_NO_FULL_LOAD && Model == MAT_MUL_BASIC) {
    MatMulFixpipeOptiTensorKernel<AType, BType, CType, BiasType, LayoutA, LayoutB, LayoutC, FullLoad, L0C2Out>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData, batch);
  }
}

}  // namespace MatmulV3Advanced
#endif  // MAT_MUL_TENSOR_COMMON_H
