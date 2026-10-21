/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_SHARE_GRAPH_H
#define INC_SHARE_GRAPH_H
#include <cstdint>

#include "graph/compute_graph.h"
#include "ascendc_ir.h"

namespace ascir {
enum class IndirectLoadInputPreType : uint32_t {
  kNone = 0U,
  kRelu,
  kExp2,
  kReluExp2,
  kReluRelu,
  kPrecisionCast,
};

enum class IndirectLoadOutputPostType : uint32_t {
  kDefault = 0U,
  kModifiedBesselK0,
  kDirectStore,
  kCastExp2CastBack,
  kSum,
  kSumKeepTail,
  kSumLastAxis,
  kExp2Sum,
  kAbsExp2Sum,
  kCastSum,
  kAddSum,
  kModifiedBesselK0Sum,
};

struct ShareGraph {
  static af::ComputeGraphPtr LoadLog2StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ModFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadLShiftStoreFusedGraph(size_t dims_size, af::DataType in_dtype, af::DataType out_dtype);
  static af::ComputeGraphPtr AddAbsFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr SubAbsFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr SubTransposeAbsFusedGraph(size_t dims_size, vector<size_t> perms);
  static af::ComputeGraphPtr ScalarInfAddFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ScalarDivInfFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr IsInfMaskedFillFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AddGeluFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr CompareFusedGraph(size_t dims_size, bool is_second_input_tensor, af::DataType dtype,
                                               std::string mode);
  static af::ComputeGraphPtr AddNegFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadToStoreAndAbsFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadUnalignPadFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadNeedLoopModeFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr BrcInlineFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadWhereStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadWhereX2X3IsUbscalarStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadWhereX2IsUbscalarStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadWhereX3IsUbscalarStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadLogicalNotStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadLogicalNotStoreFusedGraph(size_t dims_size, af::DataType dt_in, af::DataType dt_out);
  static af::ComputeGraphPtr AddRsqrtFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadBitwiseAndStoreFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                           af::DataType out_dtype);
  static af::ComputeGraphPtr ContinuesBrcFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ScalarBrcFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadBrcFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr CastCastFusedGraph(size_t dims_size, af::DataType in_dtype, af::DataType out_dtype);
  static af::ComputeGraphPtr ScalarCastAddFusedGraph(size_t dims_size, af::DataType in_dtype, af::DataType out_dtype);
  static af::ComputeGraphPtr CastCastNanFusedGraph(size_t dims_size, af::DataType in_dtype, af::DataType out_dtype);
  static af::ComputeGraphPtr CastCastIsFiniteFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                        af::DataType out_dtype);
  static af::ComputeGraphPtr CastCastReciprocalFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                          af::DataType out_dtype);
  static af::ComputeGraphPtr LoadLeakyReluStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadSigmoidStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadErfStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AddExp2FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AddFloorFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AddFloorBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AbsFmaFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AbsFmaBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AddExpBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr FloordivAbsFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadTanhStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadModifiedBesselI0StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadModifiedBesselI1StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadModifiedBesselK0StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadModifiedBesselK1StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LaguerrePolynomialLStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LegendrePolynomialPStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadAiryAiStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadErfinvStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr I0StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr I0eStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr I1eStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr BesselJ0StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr BesselJ1StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr BesselY0StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr BesselY1StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ScaledModifiedBesselK0StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ScaledModifiedBesselK1StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr SphericalBesselJ0StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr NdtrStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr NdtriStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LogNdtrStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr NextAfterStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr PolyGammaStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ZetaStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr SignBitStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr FrexpStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ShiftedChebyshevPolynomialTStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ShiftedChebyshevPolynomialUStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ShiftedChebyshevPolynomialVStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ShiftedChebyshevPolynomialWStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ChebyshevPolynomialTStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ChebyshevPolynomialUStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ChebyshevPolynomialVStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ChebyshevPolynomialWStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr HermitePolynomialHStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr HermitePolynomialHeStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr RemainderInt32StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr FloorDivInt32StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadTanhBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr RandStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr RandnStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AddAbsScalarFusedGraph(size_t dims_size, af::DataType dtype);
  static af::ComputeGraphPtr AbsBrcAddFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr UbScalarBrcAbsAddFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr BrcReduceFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr FloorDivMulLessEqualSelectFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr IsfiniteBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr IsnanBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr FmaInt8FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AxpyAbsFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AxpyAbsHalfFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AxpyAddFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr TailBrcTailReduceFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadPowAllInputIsScalarStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AddAbsFusedConstGraph(size_t dims_size, std::vector<int> dims);
  static af::ComputeGraphPtr SubTransposeAbsFusedConstGraph(size_t dims_size, vector<size_t> perms,
                                                            std::vector<int> dims);
  static af::ComputeGraphPtr LoadLogicalOrStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadLogicalAndStoreFusedGraph(size_t dims_size);
  static void ConcatAscGraph(af::AscGraph &graph, const std::vector<std::string> &dim_sizes, bool align = false);
  static af::ComputeGraphPtr AbsClipFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadGatherAbsStore(int64_t gather_axis, af::DataType data_type);
  static af::ComputeGraphPtr LoadGatherTailAbsStore(int64_t gather_axis, af::DataType data_type);
  static af::ComputeGraphPtr LoadGatherOneAxisAbsStore(int64_t gather_axis, af::DataType data_type);
  static af::ComputeGraphPtr IndirectLoadStoreFusedGraph(size_t rank, int64_t axis, af::DataType data_type,
                                                         af::DataType index_type,
                                                         IndirectLoadInputPreType input_pre_type, bool use_exp2,
                                                         IndirectLoadOutputPostType output_post_type,
                                                         const std::vector<int64_t> &input_shape = {},
                                                         const std::vector<int64_t> &output_shape = {},
                                                         bool mixed_index_pre = false, bool direct_index = false);
  static af::ComputeGraphPtr MatMulFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr GatherReduceStore(int64_t gather_axis, af::DataType data_type);
  static af::ComputeGraphPtr LoadWhereReduceStoreFusedGraph(size_t dims_size, bool x2_scalar, bool x3_scalar);
  static af::ComputeGraphPtr LoadCompareStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadLeBoolStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadBitwiseBoolStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadCompareCastSumStoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadMatmulElewiseBrcFusedGraph(bool is_dynamic = false);
  static af::ComputeGraphPtr LoadBatchMatmulElewiseBrcFusedGraph(bool is_dynamic = false);
  static af::ComputeGraphPtr LoadMatmulCompareScalarFusedGraph();
  static af::ComputeGraphPtr LoadMatmulToIntCastFusedGraph();
  static af::ComputeGraphPtr DivAbsFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr TrueDivBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr TruedivAbsFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr BF16AddFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr BF16NddmaAddFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AbsBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AbsUint8FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ErfBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadBitwiseNotStoreFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                           af::DataType out_dtype);
  static af::ComputeGraphPtr LoadBitwiseOrStoreFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                          af::DataType out_dtype);
  static af::ComputeGraphPtr LoadBitwiseXorStoreFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                           af::DataType out_dtype);
  static af::ComputeGraphPtr CeilBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr CosBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr Expm1Bf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AtanhBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr CoshBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr DigammaBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ErfcBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr BF16SinFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr BF16SqrtFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr BF16RsqrtFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr BF16SigmoidFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadCompareScalarWhereFusedGraph();
  static af::ComputeGraphPtr LoadCompareWhereFusedGraph();
  static af::ComputeGraphPtr BoolMaskFlowFusedGraph();
  static af::ComputeGraphPtr TransposeBoolFusedGraph();
  static af::ComputeGraphPtr SelectBoolFusedGraph();
  static af::ComputeGraphPtr BinaryApiScalarFusedGraph();
  static af::ComputeGraphPtr FloorToIntFloatFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr FmodFloatFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr HypotFloatFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LogicalXorFloatFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LgammaFloatFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr LoadLog10StoreFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr Log1pBfloat16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr FrexpFloatFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AcosFloatFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AcosBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AcoshBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AsinBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AsinhBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr AtanBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr PowBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ReciprocalBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr RoundBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ReluUint8FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr RshiftUint8FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr SignUint8FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr SignBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr Atan2Bf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr CopysignBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr IndirectLoadMixedElementwiseFusedGraph(size_t rank, int64_t axis, bool is_simt,
                                                                    const std::vector<int64_t> &input_shape = {},
                                                                    const std::vector<int64_t> &output_shape = {});
  static af::ComputeGraphPtr IndirectLoadSimtElementwiseCoverageFusedGraph(
      size_t rank, int64_t axis, const std::vector<int64_t> &input_shape = {},
      const std::vector<int64_t> &output_shape = {});
  static af::ComputeGraphPtr Ceil2intBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ErfcxTestFusedGraph(size_t dims_size, af::DataType dtype);
  static af::ComputeGraphPtr SinhBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr TanBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr SquareUint8FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr XorUint8FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr TruncBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr TruncDivBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr MaskRegChainFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr RoundToIntFloatToInt32FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr TruncToIntBf16ToInt32FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr VfScalarFusionComprehensiveFusedGraph();
  static af::ComputeGraphPtr RemainderBf16FusedGraph(size_t dims_size);
  static af::ComputeGraphPtr ArgMaxFusedGraph(size_t dims_size);
  static af::ComputeGraphPtr FusedBackendElewiseGraph(size_t dims_size);
};
}  // namespace ascir
#endif
