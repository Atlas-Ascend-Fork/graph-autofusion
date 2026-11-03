/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDUCTOR_CV_FUSION
#include "autofuse_cube_tiling_data.h"
#endif
#include "arch35/mat_mul_streamk_basic_cmct.h"

using namespace AscendC;
using namespace matmul;
// 此处和act模板代码有差异，注意不要 DTYPE_BIAS

#if defined(FORMAT_X1) && FORMAT_X1 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x1 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x1 = CubeFormat::ND;
#endif

#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x2 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x2 = CubeFormat::ND;
#endif

#if defined(FORMAT_Y) && FORMAT_Y == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_y = CubeFormat::NZ;
#else
constexpr CubeFormat format_y = CubeFormat::ND;
#endif

#define BMMV3_IMPL_CLASS(templateClass, ...)                                                                      \
  do {                                                                                                            \
    using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y, false, LayoutMode::NORMAL>;               \
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS, false, LayoutMode::NORMAL>;   \
    TPipe pipe;                                                                                                   \
    if (tilingData.matmulTiling.matmulRunInfo.transA == 0 && tilingData.matmulTiling.matmulRunInfo.transB == 0) { \
      using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false, LayoutMode::NORMAL>;           \
      using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false, LayoutMode::NORMAL>;           \
      templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                               \
      op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                        \
      op.Process();                                                                                               \
    } else if (tilingData.matmulTiling.matmulRunInfo.transA == 1 &&                                               \
               tilingData.matmulTiling.matmulRunInfo.transB == 0) {                                               \
      using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true, LayoutMode::NORMAL>;            \
      using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false, LayoutMode::NORMAL>;           \
      templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                               \
      op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                        \
      op.Process();                                                                                               \
    } else if (tilingData.matmulTiling.matmulRunInfo.transA == 0 &&                                               \
               tilingData.matmulTiling.matmulRunInfo.transB == 1) {                                               \
      using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false, LayoutMode::NORMAL>;           \
      using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true, LayoutMode::NORMAL>;            \
      templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                               \
      op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                        \
      op.Process();                                                                                               \
    } else {                                                                                                      \
      using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true, LayoutMode::NORMAL>;            \
      using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true, LayoutMode::NORMAL>;            \
      templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                               \
      op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                        \
      op.Process();                                                                                               \
    }                                                                                                             \
  } while (0)
// 此处和act模板代码有差异，注意保留GET_TILING_DATA_WITH_STRUCT
#define BMMV3_IMPL_CLASS_TRANS(transA, transB, templateClass, ...)                                              \
  do {                                                                                                          \
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3TilingData, tilingData, tilingGM);                                 \
    using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y, false, LayoutMode::NORMAL>;             \
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS, false, LayoutMode::NORMAL>; \
    TPipe pipe;                                                                                                 \
    using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, transA, LayoutMode::NORMAL>;          \
    using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, transB, LayoutMode::NORMAL>;          \
    templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                               \
    op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                        \
    op.Process();                                                                                               \
  } while (0)

#define BMMV3_IMPL_CLASS_COMMON(templateClass, ...)                                                               \
  do {                                                                                                            \
    using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                          \
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                              \
    TPipe pipe;                                                                                                   \
    if (tilingData.matmulTiling.matmulRunInfo.transA == 0 && tilingData.matmulTiling.matmulRunInfo.transB == 0) { \
      using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                               \
      using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                               \
      templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                               \
      op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                        \
      op.Process();                                                                                               \
    } else if (tilingData.matmulTiling.matmulRunInfo.transA == 1 &&                                               \
               tilingData.matmulTiling.matmulRunInfo.transB == 0) {                                               \
      using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                                \
      using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                               \
      templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                               \
      op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                        \
      op.Process();                                                                                               \
    } else if (tilingData.matmulTiling.matmulRunInfo.transA == 0 &&                                               \
               tilingData.matmulTiling.matmulRunInfo.transB == 1) {                                               \
      using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                               \
      using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                                \
      templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                               \
      op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                        \
      op.Process();                                                                                               \
    } else {                                                                                                      \
      using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                                \
      using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                                \
      templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                               \
      op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                        \
      op.Process();                                                                                               \
    }                                                                                                             \
  } while (0)
// 此处和act模板代码有差异，注意保留GET_TILING_DATA_WITH_STRUCT
#define BMMV3_IMPL_CLASS_COMMON_TRNAS(transA, transB, templateClass, ...)            \
  do {                                                                               \
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3TilingData, tilingData, tilingGM);      \
    using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;             \
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>; \
    TPipe pipe;                                                                      \
    using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, transA>;   \
    using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, transB>;   \
    templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                    \
    op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);             \
    op.Process();                                                                    \
  } while (0)
#ifdef INDUCTOR_TILING_DATA
template <typename CVTilingT, int8_t BATCH_API_LEVEL, int8_t BATCH_A_TRANS, int8_t BATCH_B_TRANS,
          int8_t BATCH_ITER_MODEL, int8_t BMODEL, int8_t BATCH_FULL_LOAD, int8_t BATCH_L0C2OUT_MODEL>
#else
template <int8_t BATCH_API_LEVEL, int8_t BATCH_A_TRANS, int8_t BATCH_B_TRANS, int8_t BATCH_ITER_MODEL, int8_t BMODEL,
          int8_t BATCH_FULL_LOAD, int8_t BATCH_L0C2OUT_MODEL>
#endif
__aicore__ void batch_mat_mul_v3(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR cGM,
                                 GM_ADDR workspaceGM,
#ifdef INDUCTOR_TILING_DATA
                                 const CVTilingT &tilingGM
#else
                                 GM_ADDR tilingGM
#endif
) {
  __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

  constexpr bool aTran = (BATCH_A_TRANS == 1);
  constexpr bool bTran = (BATCH_B_TRANS == 1);

#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
  using aLayout = std::conditional_t<aTran, layout::ColumnMajor, layout::RowMajor>;
  using bLayout = std::conditional_t<(format_x2 == CubeFormat::NZ), std::conditional_t<bTran, layout::Zn, layout::Nz>,
                                     std::conditional_t<bTran, layout::ColumnMajor, layout::RowMajor> >;
#endif

  REGISTER_TILING_DEFAULT(BatchMatMulV3TilingData);

  if constexpr (BATCH_API_LEVEL == MAT_MUL_HIGH_LEVEL && BMODEL == MAT_MUL_BASIC &&
                BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    BMMV3_IMPL_CLASS_COMMON_TRNAS(aTran, bTran, BatchMatMulV3Advanced::BatchMatMulAswKernel,
                                  BatchMatMulV3Advanced::BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, 0,
                                      OP_TYPE_RELU_VALUE>(aGM, bGM, biasGM, cGM, workspaceGM,
                                                          tilingData.matMulTilingData, tilingData.batchDimAll);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_A_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                                      A_FULL_LOAD_MODE, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_B_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                                      B_FULL_LOAD_MODE, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
    BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_STREAM_K &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                           MatMulL0C2Out::ON_THE_FLY, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_STREAM_K &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                           MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
#endif
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_HIGH_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
    BMMV3_IMPL_CLASS_TRANS(aTran, bTran, BatchMatMulV3Advanced::BatchMatMulMultiBatchKernel,
                           BatchMatMulV3Advanced::BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT_SING_BIAS);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE &&
                       BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
    BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                                  MatMulL0C2Out::ND_FIXPIPE_1_2>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_MERGE_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3MergeBatchBasicTilingData, tilingData, tilingGM);
    BatchMatMulActMergeBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_BATCH_MATMUL_TO_MUL && IsSameType<DTYPE_X1, DTYPE_Y>::value) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulToMulBasicTilingData, tilingData, tilingGM);
    BatchMatMulToMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_K_EQUAL_ZERO &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MatMulV3KEqZeroBasicTilingData, tilingData, tilingGM);
    MatmulV3Advanced::MatMulInputKEqZeroClearOutput(biasGM, cGM, tilingData);
#endif
  }
}

#ifdef INDUCTOR_TILING_DATA
template <typename CVTilingT, int8_t BATCH_API_LEVEL, int8_t BATCH_A_TRANS, int8_t BATCH_B_TRANS,
          int8_t BATCH_ITER_MODEL, int8_t BMODEL, int8_t BATCH_FULL_LOAD, int8_t BATCH_L0C2OUT_MODEL>
#else
template <int8_t BATCH_API_LEVEL, int8_t BATCH_A_TRANS, int8_t BATCH_B_TRANS, int8_t BATCH_ITER_MODEL, int8_t BMODEL,
          int8_t BATCH_FULL_LOAD, int8_t BATCH_L0C2OUT_MODEL>
#endif
__aicore__ void batch_mat_mul_v3_fusion(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR cGM,
                                        GM_ADDR workspaceGM,
#ifdef INDUCTOR_TILING_DATA
                                        const CVTilingT &tilingGM,
#else
                                        GM_ADDR tilingGM,
#endif
                                        AutoFusionVector::Params *param) {
  __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

  constexpr bool aTran = (BATCH_A_TRANS == 1);
  constexpr bool bTran = (BATCH_B_TRANS == 1);

#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
  using aLayout = std::conditional_t<aTran, layout::ColumnMajor, layout::RowMajor>;
  using bLayout = std::conditional_t<(format_x2 == CubeFormat::NZ), std::conditional_t<bTran, layout::Zn, layout::Nz>,
                                     std::conditional_t<bTran, layout::ColumnMajor, layout::RowMajor> >;
#endif

  REGISTER_TILING_DEFAULT(BatchMatMulV3TilingData);

  if constexpr (BATCH_API_LEVEL == MAT_MUL_HIGH_LEVEL && BMODEL == MAT_MUL_BASIC &&
                BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    BMMV3_IMPL_CLASS_COMMON_TRNAS(aTran, bTran, BatchMatMulV3Advanced::BatchMatMulAswKernel,
                                  BatchMatMulV3Advanced::BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatmulV3Advanced::MatMulActKernelFusion<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                                            0, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, param, tilingData.batchDimAll);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_A_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatmulV3Advanced::MatMulActKernelFusion<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                                            A_FULL_LOAD_MODE, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, param, tilingData.batchDimAll);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_B_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatmulV3Advanced::MatMulActKernelFusion<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                                            B_FULL_LOAD_MODE, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, param, tilingData.batchDimAll);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
    BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_STREAM_K &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                           MatMulL0C2Out::ON_THE_FLY, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_STREAM_K &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                           MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
#endif
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_HIGH_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
    BMMV3_IMPL_CLASS_TRANS(aTran, bTran, BatchMatMulV3Advanced::BatchMatMulMultiBatchKernel,
                           BatchMatMulV3Advanced::BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT_SING_BIAS);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE &&
                       BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
    BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                                  MatMulL0C2Out::ND_FIXPIPE_1_2>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_MERGE_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3MergeBatchBasicTilingData, tilingData, tilingGM);
    BatchMatMulActMergeBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_BATCH_MATMUL_TO_MUL && IsSameType<DTYPE_X1, DTYPE_Y>::value) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulToMulBasicTilingData, tilingData, tilingGM);
    BatchMatMulToMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_K_EQUAL_ZERO &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MatMulV3KEqZeroBasicTilingData, tilingData, tilingGM);
    MatmulV3Advanced::MatMulInputKEqZeroClearOutput(biasGM, cGM, tilingData);
#endif
  }
}

#ifdef INDUCTOR_TILING_DATA
template <typename CVTilingT, int8_t BATCH_API_LEVEL, int8_t BATCH_A_TRANS, int8_t BATCH_B_TRANS,
          int8_t BATCH_ITER_MODEL, int8_t BMODEL, int8_t BATCH_FULL_LOAD, int8_t BATCH_L0C2OUT_MODEL>
#else
template <int8_t BATCH_API_LEVEL, int8_t BATCH_A_TRANS, int8_t BATCH_B_TRANS, int8_t BATCH_ITER_MODEL, int8_t BMODEL,
          int8_t BATCH_FULL_LOAD, int8_t BATCH_L0C2OUT_MODEL>
#endif
__aicore__ void batch_mat_mul_v3_fusion_db(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR cGM,
                                           GM_ADDR workspaceGM,
#ifdef INDUCTOR_TILING_DATA
                                           const CVTilingT &tilingGM,
#else
                                           GM_ADDR tilingGM,
#endif
                                           AutoFusionVectorDB::Params *param) {
  __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

  constexpr bool aTran = (BATCH_A_TRANS == 1);
  constexpr bool bTran = (BATCH_B_TRANS == 1);

#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
  using aLayout = std::conditional_t<aTran, layout::ColumnMajor, layout::RowMajor>;
  using bLayout = std::conditional_t<(format_x2 == CubeFormat::NZ), std::conditional_t<bTran, layout::Zn, layout::Nz>,
                                     std::conditional_t<bTran, layout::ColumnMajor, layout::RowMajor> >;
#endif

  REGISTER_TILING_DEFAULT(BatchMatMulV3TilingData);

  if constexpr (BATCH_API_LEVEL == MAT_MUL_HIGH_LEVEL && BMODEL == MAT_MUL_BASIC &&
                BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    BMMV3_IMPL_CLASS_COMMON_TRNAS(aTran, bTran, BatchMatMulV3Advanced::BatchMatMulAswKernel,
                                  BatchMatMulV3Advanced::BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatmulV3Advanced::MatMulActKernelFusion<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                                            0, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, param, tilingData.batchDimAll);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_A_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatmulV3Advanced::MatMulActKernelFusion<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                                            A_FULL_LOAD_MODE, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, param, tilingData.batchDimAll);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_B_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatmulV3Advanced::MatMulActKernelFusion<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                                            B_FULL_LOAD_MODE, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, param, tilingData.batchDimAll);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
    BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_STREAM_K &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                           MatMulL0C2Out::ON_THE_FLY, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_STREAM_K &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
    MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                           MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_RELU_VALUE>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
#endif
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_HIGH_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
    BMMV3_IMPL_CLASS_TRANS(aTran, bTran, BatchMatMulV3Advanced::BatchMatMulMultiBatchKernel,
                           BatchMatMulV3Advanced::BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT_SING_BIAS);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE &&
                       BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
    BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                                  MatMulL0C2Out::ND_FIXPIPE_1_2>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_MERGE_BATCH) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3MergeBatchBasicTilingData, tilingData, tilingGM);
    BatchMatMulActMergeBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_BATCH_MATMUL_TO_MUL && IsSameType<DTYPE_X1, DTYPE_Y>::value) {
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulToMulBasicTilingData, tilingData, tilingGM);
    BatchMatMulToMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
  } else if constexpr (BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_K_EQUAL_ZERO &&
                       BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
                       BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MatMulV3KEqZeroBasicTilingData, tilingData, tilingGM);
    MatmulV3Advanced::MatMulInputKEqZeroClearOutput(biasGM, cGM, tilingData);
#endif
  }
}
