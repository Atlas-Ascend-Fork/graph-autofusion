/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAT_MUL_BASIC_CV_PIPELINE_H
#define MAT_MUL_BASIC_CV_PIPELINE_H

#include "cmct/block/block_mmad_builder.h"

namespace MatmulV3Advanced {
template <typename NativeMmad>
class BasicCVPipelineMmad : public NativeMmad {
 public:
  using AType = typename NativeMmad::A_T;
  using BType = typename NativeMmad::B_T;
  using BiasType = typename NativeMmad::Bias_T;
  using TileShape = typename NativeMmad::TupleL1L0Shape;

  template <typename Tensor, CubeFormat LayoutB = CubeFormat::ND>
  __aicore__ inline void operator()(Tensor output, AscendC::GlobalTensor<AType> a, AscendC::GlobalTensor<BType> b,
                                    AscendC::GlobalTensor<BiasType> bias, TileShape shape, uint64_t mOffset,
                                    uint64_t nOffset, bool isFirstTile = false, bool isAllLoc2Ub = false,
                                    uint64_t blkK = 0, bool isFirstSplitK = false, bool isEndSplitK = false) {
    if (isAllLoc2Ub) {
      // FP32 Fixpipe can split the output between both AIVs while retaining the native K-load pipeline.
      this->SetDualParam(true);
      // Reuse the native L1 transfer count; no transfer has occurred before the first tile.
      NativeMmad::template operator()<Tensor, LayoutB>(output, a, b, bias, shape, mOffset, nOffset,
                                                       this->abL1LoopCnt_ == 0, false, blkK, isFirstSplitK,
                                                       isEndSplitK);
      this->SetDualParam(false);
      return;
    }
    NativeMmad::template operator()<Tensor, LayoutB>(output, a, b, bias, shape, mOffset, nOffset, isFirstTile,
                                                     isAllLoc2Ub, blkK, isFirstSplitK, isEndSplitK);
  }
};

template <typename NativeBuilder>
struct BasicCVPipelineBuilder : NativeBuilder {
  using BlockMmadOp = BasicCVPipelineMmad<typename NativeBuilder::BlockMmadOp>;
};

template <typename NativeBuilder, typename OutputType, uint64_t FullLoadMode>
using BasicCVBlockMmadBuilder =
    std::conditional_t<std::is_same_v<OutputType, float> && (FullLoadMode == Cmct::Gemm::NONE_FULL_LOAD_MODE) &&
                           (NativeBuilder::formatB == CubeFormat::ND),
                       BasicCVPipelineBuilder<NativeBuilder>, NativeBuilder>;
}  // namespace MatmulV3Advanced

#endif
