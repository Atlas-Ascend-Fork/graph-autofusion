/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mat_mul_pingpong_tensor_cv.h
 * \brief Tensor API matmul wrapper with a CV epilogue.
 */
#ifndef MAT_MUL_PINGPONG_TENSOR_CV_H
#define MAT_MUL_PINGPONG_TENSOR_CV_H

#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "kernel_basic_intf.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <typename DataTypeOut_, typename DataTypeIn_, typename DispatchPolicy_, typename FusionOp_>
class BlockEpilogueTensorCV {
 public:
  using DataTypeOut = DataTypeOut_;
  using DataTypeIn = DataTypeIn_;
  using DispatchPolicy = DispatchPolicy_;
  using FusionOp = FusionOp_;
  using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
  using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
  using FusionParams = typename FusionOp::Params;

  static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
  static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;
  static constexpr uint16_t AIC_SYNC_AIV_FLAG = 6;
  static constexpr uint16_t CUBE_OUT_VECTOR_IN = 1;
  static constexpr uint16_t ONLY_VECTOR_IN = 2;

  struct Params {
    GM_ADDR outGmAddr{nullptr};
    FusionParams fusionParams{};
    int64_t l1M{0};
    int64_t l1N{0};
  };

  __aicore__ inline BlockEpilogueTensorCV() {}

  __aicore__ inline void Init(Params const &params, ProblemShape &problemShape) {
    problemShape_ = problemShape;
    const int64_t l1M = params.l1M > 0 ? params.l1M : asc::te::get<Gemm::MNK_M>(problemShape_);
    const int64_t l1N = params.l1N > 0 ? params.l1N : asc::te::get<Gemm::MNK_N>(problemShape_);
    const int64_t l1NAlign = Gemm::CeilAlign(l1N, static_cast<int64_t>(asc::te::c0_element<DataTypeOut>));
    const int64_t ubOffset = l1M * l1NAlign;
    fusionOp_.Init(params.fusionParams, cLocal_, l1M, l1NAlign, ubOffset, cubeMStageRows_);
  }

  __aicore__ inline void Run(BlockShape const &blockShape, int64_t dstOffset, bool splitM, int64_t baseM, int64_t baseN,
                             uint64_t ubDB = 1) {
    int64_t curM = asc::te::get<Gemm::MNK_M>(blockShape);
    if (baseM != 0) {
      curM = Gemm::Min(curM, baseM);
    }
    const int64_t halfBlockShapeM = Gemm::CeilDiv(curM, AscendC::GetTaskRation());
    int64_t blockShapeM = curM;
    if (splitM) {
      blockShapeM =
          (static_cast<uint64_t>(curM) & 1UL) > 0UL ? (halfBlockShapeM - AscendC::GetSubBlockIdx()) : halfBlockShapeM;
    }

    const int64_t nL1 = asc::te::get<Gemm::MNK_N>(blockShape);
    const int64_t curBaseN = baseN != 0 ? Gemm::Min(nL1, baseN) : nL1;
    const int64_t nL1Iter = Gemm::CeilDiv(nL1, curBaseN);
    const int64_t n = asc::te::get<Gemm::MNK_N>(problemShape_);
    const int64_t m = asc::te::get<Gemm::MNK_M>(problemShape_);
    constexpr int64_t c0Size = static_cast<int64_t>(asc::te::c0_element<DataTypeOut>);
    constexpr int64_t ubHalfElems =
        static_cast<int64_t>(AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn) / Gemm::DOUBLE_BUFFER_COUNT);
    const bool enablePp = ubDB > 1;

    for (int64_t nIdx = 0; nIdx < nL1Iter; ++nIdx) {
      const int64_t tileN = (nIdx + 1 == nL1Iter) ? (nL1 - curBaseN * nIdx) : curBaseN;
      const int64_t blockShapeNAlign = Gemm::CeilAlign(tileN, c0Size);
      const int64_t inputSize = blockShapeM * blockShapeNAlign;
      const uint16_t slot = enablePp ? static_cast<uint16_t>(cvPingPong_ & 1UL) : 0U;

      Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG + slot);
      AscendC::LocalTensor<DataTypeIn> ubLocal{AscendC::TPosition::VECIN, 0,
                                               AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn)};
      cLocal_ = ubLocal[slot * ubHalfElems];

      const int64_t stageM = Gemm::Min(cubeMStageRows_, blockShapeM);
      const int64_t stageCapacity = stageM * blockShapeNAlign;
      if (stageCapacity > 0) {
        int64_t stageOffset = 0;
        int64_t loop = 0;
        while (stageOffset < inputSize) {
          const int64_t offset = dstOffset + nIdx * curBaseN + loop * stageCapacity / blockShapeNAlign * n +
                                 halfBlockShapeM * n * (AscendC::GetSubBlockIdx() & 0x1);
          const int64_t stageSize = Gemm::Min(stageCapacity, inputSize - stageOffset);
          fusionOp_(offset, stageSize / blockShapeNAlign, tileN, n, m, blockShapeNAlign, stageSize, stageOffset,
                    CUBE_OUT_VECTOR_IN);
          fusionOp_(offset, stageSize / blockShapeNAlign, tileN, n, m, blockShapeNAlign, stageSize, stageOffset,
                    ONLY_VECTOR_IN);
          stageOffset += stageSize;
          ++loop;
        }
      }

      Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + slot);
      cvPingPong_++;
    }
  }

  __aicore__ inline void operator()(BlockShape const &blockShape, int64_t dstOffset = 0, bool splitM = false,
                                    int64_t baseM = 0, int64_t baseN = 0, uint64_t ubDB = 1) {
    Run(blockShape, dstOffset, splitM, baseM, baseN, ubDB);
  }

 private:
  AscendC::LocalTensor<DataTypeIn> cLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn)};
  FusionOp fusionOp_;
  ProblemShape problemShape_{};
  int64_t cubeMStageRows_{0};
  uint64_t cvPingPong_{0};
};

}  // namespace Block
}  // namespace Epilogue
}  // namespace Blaze

namespace MatmulV3Advanced {

// Keep the native Tensor MMAD and scheduler, with batch coordinates shared by MMAD and CV epilogue.
template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class TensorCVKernel {
 public:
  using NativeKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
  using Params = typename NativeKernel::Params;
  using AType = typename BlockMmad::AType;
  using BType = typename BlockMmad::BType;
  using CType = typename BlockMmad::CType;
  using BiasType = typename BlockMmad::BiasType;
  using LayoutA = typename BlockMmad::LayoutA;
  using LayoutB = typename BlockMmad::LayoutB;
  using LayoutC = typename BlockMmad::LayoutC;
  using MakeLayoutA = asc::te::frame_layout_format<LayoutA>;
  using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
  using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
  using MakeLayoutBias = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<BiasType>>>;

  __aicore__ inline TensorCVKernel() {
    if ASCEND_IS_AIV {
      Blaze::Gemm::Sync::NotifyCube<4, PIPE_MTE3>(4);
      Blaze::Gemm::Sync::NotifyCube<4, PIPE_MTE3>(5);
    }
  }

  __aicore__ inline ~TensorCVKernel() {
    if ASCEND_IS_AIC {
      Blaze::Gemm::Sync::WaitForVector<4, PIPE_FIX>(4, true);
      Blaze::Gemm::Sync::WaitForVector<4, PIPE_FIX>(5, true);
    }
  }

  __aicore__ inline void operator()(Params const &params) {
    int64_t coreIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
      coreIdx /= AscendC::GetTaskRation();
    }
    BlockScheduler scheduler(params.problemShape, params.schParams);
    if (coreIdx >= scheduler.GetCoreNums()) {
      return;
    }
    auto problemShape = params.problemShape;
    BlockEpilogue epilogue;
    BlockMmad blockMmad;
    if ASCEND_IS_AIV {
      epilogue.Init(params.epilogueParams, problemShape);
    }
    if ASCEND_IS_AIC {
      blockMmad.Init(params.mmadParams);
    }
    Blaze::Gemm::SetHF32(params.schParams.isHf32);
    Process(params, scheduler, blockMmad, epilogue, coreIdx);
    Blaze::Gemm::UnsetHF32(params.schParams.isHf32);
  }

 private:
  __aicore__ inline auto MakeInputLayout(Params const &params, int64_t batch, int64_t m, int64_t k) {
    if constexpr (!BlockMmad::TRANS_A) {
      const int64_t rowStride = params.mmadParams.rowStride == 0 ? k : params.mmadParams.rowStride;
      return asc::te::make_pattern_layout<LayoutA, asc::te::layout_trait_default<>>(
          asc::te::make_shape(
              batch, asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, m), asc::te::make_shape(asc::te::_1{}, k))),
          asc::te::make_stride(m * rowStride,
                               asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, rowStride),
                                                    asc::te::make_stride(asc::te::_0{}, asc::te::_1{}))));
    } else {
      return MakeLayoutA{}(batch, m, k);
    }
  }

  __aicore__ inline void Process(Params const &params, BlockScheduler &scheduler, BlockMmad &blockMmad,
                                 BlockEpilogue &epilogue, int64_t coreIdx) {
    const int64_t m = asc::te::get<Blaze::Gemm::MNK_M>(params.problemShape);
    const int64_t n = asc::te::get<Blaze::Gemm::MNK_N>(params.problemShape);
    const int64_t k = asc::te::get<Blaze::Gemm::MNK_K>(params.problemShape);
    const int64_t batch = Blaze::Gemm::Max(asc::te::get<Blaze::Gemm::MNK_B>(params.problemShape), int64_t{1});
    auto gmA = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ AType *>(params.mmadParams.aGmAddr)),
        MakeInputLayout(params, batch, m, k));
    auto gmB = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BType *>(params.mmadParams.bGmAddr)),
        MakeLayoutB{}(batch, k, n));
    auto gmBias = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BiasType *>(params.mmadParams.biasGmAddr)),
        MakeLayoutBias{}(1L, n));
    if (params.schParams.l2CacheMode == Blaze::Gemm::ALL_L2_CACHE_DISABLE ||
        params.schParams.l2CacheMode == Blaze::Gemm::A_L2_CACHE_DISABLE) {
      gmA.set_l2_cache_hint(asc::te::cache_mode::disable);
    }
    if (params.schParams.l2CacheMode == Blaze::Gemm::ALL_L2_CACHE_DISABLE ||
        params.schParams.l2CacheMode == Blaze::Gemm::B_L2_CACHE_DISABLE) {
      gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
    }
    for (int64_t blockIdx = coreIdx; blockIdx < scheduler.GetBlockNums(); blockIdx += AscendC::GetBlockNum()) {
      auto shape = scheduler.template GetBlockShape<BlockMmad::TRANS_B, BType>(blockIdx);
      auto coord = scheduler.GetBlockCoord(blockIdx);
      const int64_t coordM = asc::te::get<Blaze::Gemm::MNK_M>(coord);
      const int64_t coordN = asc::te::get<Blaze::Gemm::MNK_N>(coord);
      const int64_t coordB = asc::te::get<Blaze::Gemm::MNK_B>(coord);
      const int64_t shapeM = asc::te::get<Blaze::Gemm::MNK_M>(shape);
      const int64_t shapeN = Blaze::Gemm::Min(asc::te::get<Blaze::Gemm::MNK_N>(shape), n - coordN);
      auto subA = gmA.slice(AscendC::MakeCoord(coordB, AscendC::MakeCoord(coordM, 0L)),
                            AscendC::MakeShape(1L, AscendC::MakeShape(shapeM, k)));
      auto subB = gmB.slice(AscendC::MakeCoord(coordB, AscendC::MakeCoord(0L, coordN)),
                            AscendC::MakeShape(1L, AscendC::MakeShape(k, shapeN)));
      auto tileA = asc::te::squeeze<0>(subA);
      auto tileB = asc::te::squeeze<0>(subB);
      auto tileBias = gmBias.slice(AscendC::MakeCoord(0L, coordN), AscendC::MakeShape(1L, shapeN));
      ProblemShape validShape{shapeM, shapeN, k, asc::te::get<Blaze::Gemm::MNK_B>(shape)};
      auto ub =
          asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, CType>(0), MakeLayoutC{}(shapeM, shapeN));
      if ASCEND_IS_AIC {
        if constexpr (BlockMmad::DispatchPolicy::FULL_LOAD_MODE == Blaze::Gemm::B_FULL_LOAD_MODE) {
          auto fullB = gmB.slice(AscendC::MakeCoord(coordB, AscendC::MakeCoord(0L, 0L)),
                                 AscendC::MakeShape(1L, AscendC::MakeShape(k, n)));
          auto matrixB = asc::te::squeeze<0>(fullB);
          blockMmad(tileA, matrixB, gmBias, ub, validShape);
        } else {
          blockMmad(tileA, tileB, tileBias, ub, validShape);
        }
      }
      if ASCEND_IS_AIV {
        const int64_t offset = (coordB * m + coordM) * n + coordN;
        epilogue(validShape, offset, params.mmadParams.splitM, params.schParams.baseM, params.schParams.baseN,
                 params.mmadParams.ubDB);
      }
    }
  }
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
          uint64_t FULL_LOAD_MODE = 0, uint64_t L0C2OUT_MODEL, typename FusionOp>
__aicore__ inline void MatMulTensorCVKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                            const MatMulV3BasicTilingData &tilingData, typename FusionOp::Params *param,
                                            int64_t batch = 0) {
  using AType = A_TYPE;
  using BType = B_TYPE;
  using BiasType = BIAS_TYPE;
  using OutType = C_TYPE;
  using LayoutA = A_LAYOUT;
  using LayoutB = B_LAYOUT;
  using LayoutC = C_LAYOUT;
  using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
  static constexpr bool isFp32 = AscendC::Std::is_same_v<BType, float>;
  static constexpr bool isNDFormat = !(Blaze::Gemm::IsWeightNz<LayoutB>::value);
  using BlockScheduler =
      Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE, isFp32, isNDFormat>;
  // CV consumes the MM result from UB even when native tiling selects on-the-fly GM output.
  // Native split-M copying is valid for FP32 output; converting outputs retain the Common fallback.
  constexpr uint64_t bFullCvOutMode =
      L0C2OUT_MODEL == Blaze::Gemm::ON_THE_FLY && AscendC::Std::is_same_v<OutType, float> ? MAT_MUL_1V2_ND_ALIG_FIXPIPE
                                                                                          : L0C2OUT_MODEL;
  using DispatchPolicy = AscendC::Std::conditional_t<
      FULL_LOAD_MODE == Blaze::Gemm::B_FULL_LOAD_MODE,
      Blaze::Gemm::MatmulMultiBlockBFullLoad<bFullCvOutMode, 0, Blaze::Gemm::KernelMmadMultiBlockBFullLoad>,
      Blaze::Gemm::MatmulMultiBlockFixpipeOpti<L0C2OUT_MODEL, 0, Blaze::Gemm::KernelMmadMultiBlockFixpipeOpti>>;
  using BlockMmad =
      AscendC::Std::conditional_t<FULL_LOAD_MODE == Blaze::Gemm::B_FULL_LOAD_MODE,
                                  Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType,
                                                                LayoutC, BiasType, LayoutC>,
                                  Blaze::Gemm::Block::TensorCVBlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB,
                                                                        OutType, LayoutC, BiasType, LayoutC>>;
  using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueTensorCV<OutType, OutType, DispatchPolicy, FusionOp>;
  using MatmulKernel = TensorCVKernel<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
  using Params = typename MatmulKernel::Params;
  MatmulKernel mm;
  Params params = {{tilingData.m, tilingData.n, tilingData.k, batch},
                   {aGM, bGM, cGM, biasGM, nullptr, nullptr, tilingData.k, tilingData.mL1, tilingData.nL1,
                    tilingData.kL1, tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.l1BufferNum,
                    tilingData.l0cDB, 1, 1, tilingData.rowStride},
                   {cGM, *param, tilingData.mL1, tilingData.nL1},
                   {tilingData.mL1, tilingData.nL1, tilingData.kL1, tilingData.baseM, tilingData.baseN,
                    tilingData.baseK, tilingData.mTailCnt, tilingData.nTailCnt, tilingData.mBaseTailSplitCnt,
                    tilingData.nBaseTailSplitCnt, tilingData.mTailMain, tilingData.nTailMain, tilingData.mmadParam,
                    static_cast<uint32_t>(tilingData.l2CacheDisable), tilingData.sliceM, tilingData.srcNdStride,
                    tilingData.innerBatch}};
  mm(params);
}

}  // namespace MatmulV3Advanced

#endif  // MAT_MUL_PINGPONG_TENSOR_CV_H
