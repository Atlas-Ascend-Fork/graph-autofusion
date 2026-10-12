/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascir_register.h"
#include "v2_ascir_codegen_impl.h"
#include "v2_ascir_att_impl.h"
#include "graph/types_af.h"
#include <utility>
#include <type_traits>

template <size_t N, size_t... Is>
static af::OrderedTensorTypeList MakeT1ListImpl(const std::pair<ge::DataType, ge::DataType> (&pairs)[N],
                                                std::index_sequence<Is...>) {
  return af::OrderedTensorTypeList{pairs[Is].first...};
}

template <size_t N>
static af::OrderedTensorTypeList MakeT1List(const std::pair<ge::DataType, ge::DataType> (&pairs)[N]) {
  return MakeT1ListImpl(pairs, std::make_index_sequence<N>{});
}

template <size_t N, size_t... Is>
static af::OrderedTensorTypeList MakeT2ListImpl(const std::pair<ge::DataType, ge::DataType> (&pairs)[N],
                                                std::index_sequence<Is...>) {
  return af::OrderedTensorTypeList{pairs[Is].second...};
}

template <size_t N>
static af::OrderedTensorTypeList MakeT2List(const std::pair<ge::DataType, ge::DataType> (&pairs)[N]) {
  return MakeT2ListImpl(pairs, std::make_index_sequence<N>{});
}

namespace af {
namespace ascir {
EXPORT_GENERATOR()

// todo:: 暂时先定义一个AscIrAttStubV2用于注册, 后面att根据需要替换成具体的impl子类
class AscIrAttStubV2 : public af::ascir::AscIrAtt {
  void *GetApiPerf() const override {
    return nullptr;
  }
  void *GetAscendCApiPerfTable() const override {
    return nullptr;
  }
};

const std::vector<std::string> v2_soc_versions{"3510", "5102", "9201", "9202"};

REG_ASC_IR(Square)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SquareAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::SquareAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_UINT16, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_BF16, DT_UINT8}}}});

constexpr std::pair<ge::DataType, ge::DataType> kRoundToIntTypePairs[] = {
    {DT_FLOAT, DT_INT64},   {DT_FLOAT, DT_INT32},  {DT_FLOAT, DT_INT16},   {DT_FLOAT16, DT_INT16},
    {DT_FLOAT16, DT_INT32}, {DT_FLOAT16, DT_INT8}, {DT_FLOAT16, DT_UINT8}, {DT_BF16, DT_INT32},
};
REG_ASC_IR(RoundToInt)
    .Input("x", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::RoundToIntAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::RoundToIntAscIrCodegenImplV2>(),
                            {{"T1", MakeT1List(kRoundToIntTypePairs)}, {"T2", MakeT2List(kRoundToIntTypePairs)}}});

constexpr std::pair<ge::DataType, ge::DataType> TruncToIntTypePairs[] = {
    {DT_FLOAT, DT_INT64},   {DT_FLOAT, DT_INT32},  {DT_FLOAT, DT_INT16},   {DT_FLOAT16, DT_INT16},
    {DT_FLOAT16, DT_INT32}, {DT_FLOAT16, DT_INT8}, {DT_FLOAT16, DT_UINT8}, {DT_BF16, DT_INT32},
};
REG_ASC_IR(TruncToInt)
    .Input("x", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::TruncToIntAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::TruncToIntAscIrCodegenImplV2>(),
                            {{"T1", MakeT1List(TruncToIntTypePairs)}, {"T2", MakeT2List(TruncToIntTypePairs)}}});

REG_ASC_IR(Xor)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::XorAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::XorAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_UINT8}}}});

REG_ASC_IR(Trunc)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::TruncAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::TruncAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Tan)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::TanAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::TanAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(TruncDiv)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::TruncDivAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::TruncDivAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Sinh)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SinhAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::SinhAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Remainder)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::RemainderAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::RemainderAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}}}});

REG_ASC_IR(ModifiedBesselI0)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ModifiedBesselI0AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ModifiedBesselI0AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ModifiedBesselI1)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ModifiedBesselI1AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ModifiedBesselI1AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ModifiedBesselK0)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ModifiedBesselK0AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ModifiedBesselK0AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ModifiedBesselK1)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ModifiedBesselK1AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ModifiedBesselK1AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(I0)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::I0AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::I0AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(I0e)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::I0eAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::I0eAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(I1e)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::I1eAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::I1eAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(BesselJ0)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::BesselJ0AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::BesselJ0AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(BesselJ1)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::BesselJ1AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::BesselJ1AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(BesselY0)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::BesselY0AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::BesselY0AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(BesselY1)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::BesselY1AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::BesselY1AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ScaledModifiedBesselK0)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ScaledModifiedBesselK0AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ScaledModifiedBesselK0AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ScaledModifiedBesselK1)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ScaledModifiedBesselK1AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ScaledModifiedBesselK1AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(SphericalBesselJ0)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SphericalBesselJ0AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::SphericalBesselJ0AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Ndtr)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::NdtrAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::NdtrAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Ndtri)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::NdtriAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::NdtriAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(LogNdtr)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LogNdtrAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::LogNdtrAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(NextAfter)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::NextAfterAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::NextAfterAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(PolyGamma)
    .Input("x", "T")
    .Input("n", "U")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::PolyGammaAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::PolyGammaAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}, {"U", TensorType{DT_INT32}}}});

REG_ASC_IR(SignBit)
    .Input("x", "T")
    .Output("y", "U")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SignBitAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::SignBitAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_INT32}}, {"U", TensorType{DT_UINT8, DT_BOOL}}}});

REG_ASC_IR(Frexp)
    .Input("x", "T")
    .Output("mantissa", "T")
    .Output("exponent", "U")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::FrexpAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::FrexpAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}, {"U", TensorType{DT_INT32}}}});

REG_ASC_IR(Igamma)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::IgammaAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::IgammaAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Igammac)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::IgammacAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::IgammacAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Zeta)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ZetaAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ZetaAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ShiftedChebyshevPolynomialT)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<int64_t>("n")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ShiftedChebyshevPolynomialTAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ShiftedChebyshevPolynomialTAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ShiftedChebyshevPolynomialU)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<int64_t>("n")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ShiftedChebyshevPolynomialUAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ShiftedChebyshevPolynomialUAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ShiftedChebyshevPolynomialV)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<int64_t>("n")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ShiftedChebyshevPolynomialVAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ShiftedChebyshevPolynomialVAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ShiftedChebyshevPolynomialW)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<int64_t>("n")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ShiftedChebyshevPolynomialWAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ShiftedChebyshevPolynomialWAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ChebyshevPolynomialT)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<int64_t>("n")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ChebyshevPolynomialTAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ChebyshevPolynomialTAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ChebyshevPolynomialU)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<int64_t>("n")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ChebyshevPolynomialUAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ChebyshevPolynomialUAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ChebyshevPolynomialV)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<int64_t>("n")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ChebyshevPolynomialVAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ChebyshevPolynomialVAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(ChebyshevPolynomialW)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<int64_t>("n")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ChebyshevPolynomialWAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ChebyshevPolynomialWAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(HermitePolynomialH)
    .Input("x", "T")
    .Input("n", "U")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::HermitePolynomialHAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::HermitePolynomialHAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}, {"U", TensorType{DT_INT32}}}});

REG_ASC_IR(HermitePolynomialHe)
    .Input("x", "T")
    .Input("n", "U")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::HermitePolynomialHeAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::HermitePolynomialHeAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}, {"U", TensorType{DT_INT32}}}});

REG_ASC_IR(LaguerrePolynomialL)
    .Input("x", "T")
    .Input("n", "U")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::LaguerrePolynomialLAscIrAttImplV2>(),
           af::ascir::AscIrImplCreator<af::ascir::LaguerrePolynomialLAscIrCodegenImplV2>(),
           {{"T", TensorType{DT_FLOAT}},
            {"U", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}}}});

REG_ASC_IR(LegendrePolynomialP)
    .Input("x", "T")
    .Input("n", "U")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::LegendrePolynomialPAscIrAttImplV2>(),
           af::ascir::AscIrImplCreator<af::ascir::LegendrePolynomialPAscIrCodegenImplV2>(),
           {{"T", TensorType{DT_FLOAT}},
            {"U", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}}}});

REG_ASC_IR(AiryAi)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AiryAiAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::AiryAiAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Erfinv)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ErfinvAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ErfinvAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(VectorFunc)
    .DynamicInput("x", "T")
    .DynamicOutput("y", "T")
    .Attr<std::string>("sub_graph_name")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::VectorFuncAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::VfAscIrCodegenImpl>(),
                            {{"T", TensorType::ALL()}}});

REG_ASC_IR(Data).Impl(v2_soc_versions,
                      {af::ascir::AscIrImplCreator<af::ascir::DataAscIrAttImplV2>(),
                       af::ascir::AscIrImplCreator<af::ascir::DataAscIrCodegenImplV2>(),
                       {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                         DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_BOOL}}}});

REG_ASC_IR(Scalar).Impl(v2_soc_versions,
                        {af::ascir::AscIrImplCreator<af::ascir::ScalarAscIrAttImplV2>(),
                         af::ascir::AscIrImplCreator<af::ascir::ScalarAscIrCodegenImplV2>(),
                         {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                           DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_BOOL}}}});

REG_ASC_IR(ScalarData)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ScalarAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ScalarAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BOOL}}}});

REG_ASC_IR(IndexExpr).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::IndexExprAscIrAttImplV2>(),
                                             af::ascir::AscIrImplCreator<af::ascir::IndexExprAscIrCodegenImplV2>(),
                                             {{"T", TensorType{DT_INT32, DT_INT64}}}});

REG_ASC_IR(Arange)
    .Inputs({})
    .Output("y", "T")
    .StartNode()
    .Attr<Expression>("base")
    .Attr<Expression>("step")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AscIrAttStubV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ArangeAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT32, DT_INT64}}}});

REG_ASC_IR(Output).Impl(v2_soc_versions,
                        {af::ascir::AscIrImplCreator<af::ascir::OutputAscIrAttImplV2>(),
                         af::ascir::AscIrImplCreator<af::ascir::OutputAscIrCodegenImplV2>(),
                         {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                           DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_BOOL}}}});

REG_ASC_IR(Rand).Inputs({}).Output("y", "T").StartNode().Impl(
    v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::RandAscIrAttImplV2>(),
                      af::ascir::AscIrImplCreator<af::ascir::RandAscIrCodegenImplV2>(),
                      {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Randn).Inputs({}).Output("y", "T").StartNode().Impl(
    v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::RandnAscIrAttImplV2>(),
                      af::ascir::AscIrImplCreator<af::ascir::RandnAscIrCodegenImplV2>(),
                      {{"T", TensorType{DT_UINT32, DT_INT32}}}});

REG_ASC_IR(Workspace).Impl(v2_soc_versions,
                           {af::ascir::AscIrImplCreator<af::ascir::WorkspaceAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::WorkspaceAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_BOOL}}}});

REG_ASC_IR(Load).Impl(v2_soc_versions,
                      {af::ascir::AscIrImplCreator<af::ascir::LoadAscIrAttImplV2>(),
                       af::ascir::AscIrImplCreator<af::ascir::LoadAscIrCodegenImplV2>(),
                       {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16,
                                         DT_FLOAT, DT_INT64, DT_BF16, DT_UINT64, DT_BOOL}}}});

REG_ASC_IR(Nddma)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<Expression>("offset")
    .ComputeType(ComputeType::kComputeLoad)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::NddmaAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::NddmaAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16,
                                              DT_FLOAT, DT_INT64, DT_BF16, DT_UINT64, DT_BOOL}}}});

REG_ASC_IR(Store).Impl(v2_soc_versions,
                       {af::ascir::AscIrImplCreator<af::ascir::StoreAscIrAttImplV2>(),
                        af::ascir::AscIrImplCreator<af::ascir::StoreAscIrCodegenImplV2>(),
                        {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16,
                                          DT_FLOAT, DT_INT64, DT_BF16, DT_UINT64, DT_BOOL}}}});

REG_ASC_IR(Broadcast).Impl(v2_soc_versions,
                           {af::ascir::AscIrImplCreator<af::ascir::BroadcastAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::BroadcastAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_UINT8, DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT16,
                                              DT_UINT32, DT_UINT64, DT_INT64, DT_BOOL}}}});

REG_ASC_IR(RemovePad).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::RemovePadAscIrAttImplV2>(),
                                             af::ascir::AscIrImplCreator<af::ascir::RemovePadAscIrCodegenImplV2>(),
                                             {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16,
                                                               DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Pad).Impl(v2_soc_versions,
                     {af::ascir::AscIrImplCreator<af::ascir::PadAscIrAttImplV2>(),
                      af::ascir::AscIrImplCreator<af::ascir::PadAscIrCodegenImplV2>(),
                      {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Round)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::RoundAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::RoundAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_BF16, DT_FLOAT16}}}});

// todo: Nop DT_INT64 后面根据需要放开
REG_ASC_IR(Nop).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::NopAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::NopAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                                                         DT_UINT64, DT_FLOAT16, DT_FLOAT}}}});

/* cast 暂时先放开int64->float, 以下类型, 暂不放开
 * T1:DT_INT64, DT_INT64, DT_INT64, DT_INT64,
 * T2:DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_UINT64,
 */
constexpr std::pair<ge::DataType, ge::DataType> kCastTypePairs[] = {
    {DT_FLOAT, DT_FLOAT},   {DT_FLOAT, DT_FLOAT16}, {DT_FLOAT, DT_INT64},   {DT_FLOAT, DT_INT32},
    {DT_FLOAT, DT_INT16},   {DT_FLOAT, DT_BF16},    {DT_FLOAT, DT_INT8},    {DT_FLOAT16, DT_FLOAT},
    {DT_FLOAT16, DT_INT32}, {DT_FLOAT16, DT_INT16}, {DT_FLOAT16, DT_INT8},  {DT_FLOAT16, DT_UINT8},
    {DT_UINT64, DT_INT64},  {DT_FLOAT16, DT_INT64}, {DT_UINT16, DT_INT16},  {DT_UINT8, DT_FLOAT16},
    {DT_UINT8, DT_FLOAT},   {DT_UINT8, DT_INT32},   {DT_UINT8, DT_INT16},   {DT_UINT8, DT_INT8},
    {DT_UINT8, DT_INT64},   {DT_UINT32, DT_INT32},  {DT_INT8, DT_FLOAT16},  {DT_INT8, DT_UINT8},
    {DT_INT8, DT_FLOAT},    {DT_INT8, DT_INT16},    {DT_INT16, DT_FLOAT16}, {DT_INT16, DT_FLOAT},
    {DT_INT16, DT_UINT16},  {DT_INT16, DT_INT8},    {DT_INT16, DT_UINT8},   {DT_INT32, DT_FLOAT},
    {DT_INT32, DT_INT64},   {DT_INT32, DT_INT16},   {DT_INT32, DT_FLOAT16}, {DT_INT32, DT_UINT32},
    {DT_INT32, DT_BOOL},    {DT_INT64, DT_INT32},   {DT_INT64, DT_FLOAT},   {DT_INT64, DT_UINT8},
    {DT_INT64, DT_UINT64},  {DT_INT64, DT_FLOAT16}, {DT_INT64, DT_BF16},    {DT_INT64, DT_BOOL},
    {DT_INT8, DT_BOOL},     {DT_INT16, DT_BOOL},    {DT_BF16, DT_FLOAT},    {DT_BF16, DT_INT32},
    {DT_FLOAT, DT_BOOL}};
REG_ASC_IR(Cast).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::CastAscIrAttImplV2>(),
                                        af::ascir::AscIrImplCreator<af::ascir::CastAscIrCodegenImplV2>(),
                                        {{"T1", MakeT1List(kCastTypePairs)}, {"T2", MakeT2List(kCastTypePairs)}}});

REG_ASC_IR(Abs).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AbsAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::AbsAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                                                         DT_BF16, DT_UINT8}}}});

REG_ASC_IR(Exp).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ExpAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::ExpAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Exp2)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::Exp2AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::Exp2AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Floor)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::FloorAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::FloorAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Fma)
    .Input("x1", "T")
    .Input("x2", "T")
    .Input("x3", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::FmaAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::FmaAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_UINT8}}}});

REG_ASC_IR(Ln).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LnAscIrAttImplV2>(),
                                      af::ascir::AscIrImplCreator<af::ascir::LnAscIrCodegenImplV2>(),
                                      {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Expm1)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::Expm1AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::Expm1AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_BF16, DT_FLOAT16}}}});

REG_ASC_IR(Log2)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::Log2AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::Log2AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

constexpr std::pair<ge::DataType, ge::DataType> kLShiftTypePairs[] = {
    {DT_INT8, DT_INT8},  {DT_INT16, DT_INT16},  {DT_INT32, DT_INT32},  {DT_INT64, DT_INT64},
    {DT_UINT8, DT_INT8}, {DT_UINT16, DT_INT16}, {DT_UINT32, DT_INT32}, {DT_UINT64, DT_INT64},
};
REG_ASC_IR(LShift)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Output("y", "T1")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LShiftAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::LShiftAscIrCodegenImplV2>(),
                            {{"T1", MakeT1List(kLShiftTypePairs)}, {"T2", MakeT2List(kLShiftTypePairs)}}});

REG_ASC_IR(Mod)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ModAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ModAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT16, DT_UINT8}}}});

REG_ASC_IR(Sqrt).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SqrtAscIrAttImplV2>(),
                                        af::ascir::AscIrImplCreator<af::ascir::SqrtAscIrCodegenImplV2>(),
                                        {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Rsqrt).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::RsqrtAscIrAttImplV2>(),
                                         af::ascir::AscIrImplCreator<af::ascir::RsqrtAscIrCodegenImplV2>(),
                                         {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Reciprocal)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ReciprocalAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ReciprocalAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Erf).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ErfAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::ErfAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Sign).Impl(v2_soc_versions,
                      {af::ascir::AscIrImplCreator<af::ascir::SignAscIrAttImplV2>(),
                       af::ascir::AscIrImplCreator<af::ascir::SignAscIrCodegenImplV2>(),
                       {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT8, DT_BF16}}}});

REG_ASC_IR(Tanh).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<AscIrAttStubV2>(),
                                        af::ascir::AscIrImplCreator<af::ascir::TanhAscIrCodegenImplV2>(),
                                        {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Isnan)
    .Input("x", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::IsnanAscIrAttImplV2>(),
           af::ascir::AscIrImplCreator<af::ascir::IsnanAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}, {"T2", TensorType{DT_UINT8, DT_BOOL}}}});

REG_ASC_IR(IsFinite)
    .Input("x", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::IsFiniteAscIrAttImplV2>(),
           af::ascir::AscIrImplCreator<af::ascir::IsFiniteAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}, {"T2", TensorType{DT_UINT8, DT_BOOL}}}});

REG_ASC_IR(IsInf).Impl(v2_soc_versions,
                       {af::ascir::AscIrImplCreator<af::ascir::IsInfAscIrAttImplV2>(),
                        af::ascir::AscIrImplCreator<af::ascir::IsInfAscIrCodegenImplV2>(),
                        {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}, {"T2", TensorType{DT_UINT8, DT_BOOL}}}});

REG_ASC_IR(Relu).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ReluAscIrAttImplV2>(),
                                        af::ascir::AscIrImplCreator<af::ascir::ReluAscIrCodegenImplV2>(),
                                        {{"T", TensorType{DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT64}}}});

REG_ASC_IR(Neg).Impl(v2_soc_versions,
                     {af::ascir::AscIrImplCreator<af::ascir::NegAscIrAttImplV2>(),
                      af::ascir::AscIrImplCreator<af::ascir::NegAscIrCodegenImplV2>(),
                      {{"T", TensorType{DT_BF16, DT_INT8, DT_INT64, DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

// todo: LogicalNot DT_INT64 后面根据需要放开
constexpr std::pair<ge::DataType, ge::DataType> kLogicalNotTypePairs[] = {
    {DT_FLOAT16, DT_UINT8}, {DT_FLOAT, DT_UINT8},  {DT_UINT8, DT_UINT8},  {DT_INT16, DT_UINT8},
    {DT_INT32, DT_UINT8},   {DT_INT8, DT_UINT8},   {DT_INT64, DT_UINT8},  {DT_BF16, DT_UINT8},
    {DT_UINT16, DT_UINT8},  {DT_UINT32, DT_UINT8}, {DT_UINT64, DT_UINT8}, {DT_BOOL, DT_BOOL},
};
REG_ASC_IR(LogicalNot)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LogicalNotAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::LogicalNotAscIrCodegenImplV2>(),
                            {{"T1", MakeT1List(kLogicalNotTypePairs)}, {"T2", MakeT2List(kLogicalNotTypePairs)}}});

REG_ASC_IR(Max).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MaxAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::MaxAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_BF16, DT_FLOAT16,
                                                         DT_FLOAT, DT_INT64}}}});

REG_ASC_IR(Sum).Impl(v2_soc_versions,
                     {af::ascir::AscIrImplCreator<af::ascir::SumAscIrAttImplV2>(),
                      af::ascir::AscIrImplCreator<af::ascir::SumAscIrCodegenImplV2>(),
                      {{"T", TensorType{DT_INT8, DT_INT16, DT_INT32, DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT64}}}});

REG_ASC_IR(Min).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MinAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::MinAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_BF16, DT_FLOAT16,
                                                         DT_FLOAT, DT_INT64}}}});

REG_ASC_IR(Mean).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MeanAscIrAttImplV2>(),
                                        af::ascir::AscIrImplCreator<af::ascir::MeanAscIrCodegenImplV2>(),
                                        {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Prod).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ProdAscIrAttImplV2>(),
                                        af::ascir::AscIrImplCreator<af::ascir::ProdAscIrCodegenImplV2>(),
                                        {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Sigmoid).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SigmoidAscIrAttImplV2>(),
                                           af::ascir::AscIrImplCreator<af::ascir::SigmoidAscIrCodegenImplV2>(),
                                           {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Any).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AnyAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::AnyAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(All).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AllAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::AllAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Add).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AddAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::AddAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8,
                                                         DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}}}});

REG_ASC_IR(Sub).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SubAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::SubAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8,
                                                         DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}}}});

REG_ASC_IR(Div).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::DivAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::DivAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Mul).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MulAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::MulAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_INT8, DT_UINT8, DT_BF16, DT_INT16, DT_INT32, DT_FLOAT16,
                                                         DT_FLOAT, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL}}}});

REG_ASC_IR(Minimum).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MinimumAscIrAttImplV2>(),
                                           af::ascir::AscIrImplCreator<af::ascir::MinimumAscIrCodegenImplV2>(),
                                           {{"T", TensorType{DT_BF16, DT_INT8, DT_INT16, DT_INT64, DT_INT32, DT_FLOAT16,
                                                             DT_FLOAT, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}}}});

REG_ASC_IR(Maximum).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MaximumAscIrAttImplV2>(),
                                           af::ascir::AscIrImplCreator<af::ascir::MaximumAscIrCodegenImplV2>(),
                                           {{"T", TensorType{DT_BF16, DT_INT8, DT_INT16, DT_INT64, DT_INT32, DT_FLOAT16,
                                                             DT_FLOAT, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}}}});

constexpr std::pair<ge::DataType, ge::DataType> kTrueDivTypePairs[] = {
    {DT_FLOAT16, DT_FLOAT16},
    {DT_FLOAT, DT_FLOAT},
    {DT_BF16, DT_BF16},
};
REG_ASC_IR(TrueDiv).Impl(v2_soc_versions,
                         {af::ascir::AscIrImplCreator<af::ascir::TrueDivAscIrAttImplV2>(),
                          af::ascir::AscIrImplCreator<af::ascir::TrueDivAscIrCodegenImplV2>(),
                          {{"T1", MakeT1List(kTrueDivTypePairs)}, {"T2", MakeT2List(kTrueDivTypePairs)}}});

// todo:LogicalOr DT_INT64 后面根据需要放开
REG_ASC_IR(LogicalOr).Impl(v2_soc_versions,
                           {af::ascir::AscIrImplCreator<af::ascir::LogicalOrAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::LogicalOrAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_BF16, DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT8,
                                               DT_UINT32, DT_INT64, DT_UINT16, DT_UINT64, DT_BOOL}},
                             {"T2", TensorType{DT_UINT8, DT_BOOL}}}});

// todo:LogicalAnd DT_INT64 后面根据需要放开
REG_ASC_IR(LogicalAnd)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LogicalAndAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::LogicalAndAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT8, DT_INT64,
                                               DT_UINT32, DT_BF16, DT_UINT16, DT_UINT64, DT_BOOL}},
                             {"T2", TensorType{DT_UINT8, DT_BOOL}}}});

REG_ASC_IR(Pow).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::PowAscIrAttImplV2>(),
                                       af::ascir::AscIrImplCreator<af::ascir::PowAscIrCodegenImplV2>(),
                                       {{"T", TensorType{DT_INT8, DT_UINT8, DT_UINT16, DT_UINT32, DT_INT16, DT_INT32,
                                                         DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(ClipByValue)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ClipByValueAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ClipByValueAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_INT8, DT_INT16, DT_INT64,
                                              DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}}}});

// todo:Ge Eq Ne Gt Le  DT_INT64 后面根据需要放开
REG_ASC_IR(Ge).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::GeAscIrAttImplV2>(),
                                      af::ascir::AscIrImplCreator<af::ascir::GeAscIrCodegenImplV2>(),
                                      {{"T1", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32,
                                                         DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}},
                                       {"T2", TensorType{DT_UINT8, DT_BOOL}}}});

REG_ASC_IR(Eq).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::EqAscIrAttImplV2>(),
                                      af::ascir::AscIrImplCreator<af::ascir::EqAscIrCodegenImplV2>(),
                                      {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_BF16, DT_INT8,
                                                         DT_INT16, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}},
                                       {"T2", TensorType{DT_UINT8, DT_BOOL}}}});

REG_ASC_IR(Ne).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::NeAscIrAttImplV2>(),
                                      af::ascir::AscIrImplCreator<af::ascir::NeAscIrCodegenImplV2>(),
                                      {{"T1", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_BF16, DT_FLOAT16, DT_FLOAT,
                                                         DT_INT32, DT_INT64, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}},
                                       {"T2", TensorType{DT_UINT8, DT_BOOL}}}});

REG_ASC_IR(Gt).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::GtAscIrAttImplV2>(),
                                      af::ascir::AscIrImplCreator<af::ascir::GtAscIrCodegenImplV2>(),
                                      {{"T1", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32,
                                                         DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}},
                                       {"T2", TensorType{DT_UINT8, DT_BOOL}}}});

REG_ASC_IR(Le).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LeAscIrAttImplV2>(),
                                      af::ascir::AscIrImplCreator<af::ascir::LeAscIrCodegenImplV2>(),
                                      {{"T1", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32,
                                                         DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}},
                                       {"T2", TensorType{DT_UINT8, DT_BOOL}}}});

REG_ASC_IR(Lt).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LtAscIrAttImplV2>(),
                                      af::ascir::AscIrImplCreator<af::ascir::LtAscIrCodegenImplV2>(),
                                      {{"T1", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32,
                                                         DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}},
                                       {"T2", TensorType{DT_UINT8, DT_BOOL}}}});

REG_ASC_IR(Concat).Impl(v2_soc_versions,
                        {af::ascir::AscIrImplCreator<af::ascir::ConcatAscIrAttImplV2>(),
                         af::ascir::AscIrImplCreator<af::ascir::ConcatAscIrCodegenImplV2>(),
                         {{"T", TensorType{DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                                           DT_INT64, DT_UINT64, DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Split)
    .Input("x", "T")
    .DynamicOutput("y", "T")
    .Attr<int64_t>("index")
    .Attr<int64_t>("gid")  // global_id, SplitOp的全局编号
    .ComputeType(ComputeType::kComputeSplit)
    .Impl(v2_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::SplitAscIrAttImplV2>(),
           af::ascir::AscIrImplCreator<af::ascir::SplitAscIrCodegenImplV2>(),
           {{"T", TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,   DT_INT32,
                             DT_INT64,      DT_INT8,      DT_QINT16, DT_QINT32, DT_QINT8,   DT_QUINT16, DT_QUINT8,
                             DT_UINT16,     DT_UINT32,    DT_UINT64, DT_UINT8,  DT_BF16,    DT_BOOL}}}});

REG_ASC_IR(Select).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SelectAscIrAttImplV2>(),
                                          af::ascir::AscIrImplCreator<af::ascir::WhereAscIrCodegenImplV2>(),
                                          {{"T1", TensorType{DT_UINT8, DT_BOOL}},
                                           {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT32, DT_INT64}}}});

REG_ASC_IR(Where).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::WhereAscIrAttImplV2>(),
                                         af::ascir::AscIrImplCreator<af::ascir::WhereAscIrCodegenImplV2>(),
                                         {{"T1", TensorType{DT_UINT8, DT_BOOL}},
                                          {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_BF16,
                                                            DT_INT8, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}}}});

REG_ASC_IR(MaskedFill)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MaskedFillAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::MaskedFillAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_UINT8}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

// Ub2ub是在sched阶段添加的，不需要在py构图中对外体现
REG_ASC_IR(Ub2ub).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::Ub2ubAscIrAttImplV2>(),
                                         af::ascir::AscIrImplCreator<af::ascir::Ub2ubAscIrCodegenImplV2>(),
                                         {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                                                           DT_INT64, DT_UINT64, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(LeakyRelu).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LeakyReluAscIrAttImplV2>(),
                                             af::ascir::AscIrImplCreator<af::ascir::LeakyReluAscIrCodegenImplV2>(),
                                             {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

// todo:BitwiseAnd DT_INT64 后面根据需要放开
REG_ASC_IR(BitwiseAnd)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::BitwiseAndAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::BitwiseAndAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT8, DT_INT8, DT_INT64, DT_UINT32,
                                              DT_UINT64, DT_BOOL}}}});

REG_ASC_IR(BitwiseNot)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::BitwiseNotAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::BitwiseNotAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT8, DT_INT8, DT_INT64, DT_UINT32,
                                              DT_UINT64, DT_BOOL}}}});

REG_ASC_IR(BitwiseOr)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::BitwiseOrAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::BitwiseOrAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT8, DT_INT8, DT_INT64, DT_UINT32,
                                              DT_UINT64, DT_BOOL}}}});

REG_ASC_IR(BitwiseXor)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::BitwiseXorAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::BitwiseXorAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT8, DT_INT8, DT_INT64, DT_UINT32,
                                              DT_UINT64, DT_BOOL}}}});

REG_ASC_IR(Ceil)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::CeilAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::CeilAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Cos)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::CosAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::CosAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Acos)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AcosAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::AcosAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Cosh)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::CoshAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::CoshAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Digamma)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::DigammaAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::DigammaAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Erfc)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ErfcAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ErfcAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Erfcx)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ErfcxAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::ErfcxAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Atan2)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::Atan2AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::Atan2AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(CopySign)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::CopySignAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::CopySignAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Ceil2Int)
    .Input("x", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::Ceil2IntAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::Ceil2IntAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}, {"T2", TensorType{DT_INT32}}}});

REG_ASC_IR(Gather).Impl(v2_soc_versions,
                        {af::ascir::AscIrImplCreator<af::ascir::GatherAscIrAttImplV2>(),
                         af::ascir::AscIrImplCreator<af::ascir::GatherAscIrCodegenImplV2>(),
                         {{"T1", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_BF16, DT_FLOAT}},
                          {"T2", TensorType{DT_INT32, DT_INT64}}}});

REG_ASC_IR(IndirectLoad)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Output("y", "T1")
    .Attr<int64_t>("axis")
    .Attr<bool>("negative_index_support")
    .Attr<bool>("need_check_bound")
    .Attr<Expression>("max")
    .ComputeType(ComputeType::kComputeLoad)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::IndirectLoadAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::IndirectLoadAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_INT8, DT_UINT8, DT_BOOL, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                                               DT_INT64, DT_UINT64, DT_FLOAT16, DT_BF16, DT_FLOAT}},
                             {"T2", TensorType{DT_INT32, DT_INT64}}}});

REG_ASC_IR(Transpose).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::TransposeAscIrAttImplV2>(),
                                             af::ascir::AscIrImplCreator<af::ascir::TransposeAscIrCodegenImplV2>(),
                                             {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16,
                                                               DT_FLOAT, DT_BF16, DT_BOOL, DT_INT64}}}});
// todo:目前前端dt构图用到了FlashSoftmax，暂时无法删除
REG_ASC_IR(FlashSoftmax)
    .Impl({}, {af::ascir::AscIrImplCreator<af::ascir::AbsAscIrAttImplV2>(),
               af::ascir::AscIrImplCreator<af::ascir::AscIrCodegen>(),
               {{"T1", TensorType{DT_INT8, DT_INT16}}, {"T2", TensorType{DT_UINT8, DT_INT16}}}});

REG_ASC_IR(FloorDiv).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::FloorDivAscIrAttImplV2>(),
                                            af::ascir::AscIrImplCreator<af::ascir::FloorDivAscIrCodegenImplV2>(),
                                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_UINT8,
                                                              DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}}}});

REG_ASC_IR(FloorToInt)
    .Input("x", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::FloorToIntAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::FloorToIntAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}, {"T2", TensorType{DT_INT32}}}});

REG_ASC_IR(Fmod)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::FmodAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::FmodAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Hypot)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::HypotAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::HypotAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Lgamma)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LgammaAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::LgammaAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Log10)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::Log10AscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::Log10AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(LogicalXor)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LogicalXorAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::LogicalXorAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_BF16, DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT8,
                                               DT_UINT32, DT_INT64, DT_UINT16, DT_UINT64, DT_BOOL}},
                             {"T2", TensorType{DT_UINT8, DT_BOOL}}}});

REG_ASC_IR(Log1p)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::Log1pAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::Log1pAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Gelu).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::GeluAscIrAttImplV2>(),
                                        af::ascir::AscIrImplCreator<af::ascir::GeluAscIrCodegenImplV2>(),
                                        {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Axpy).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AxpyAscIrAttImplV2>(),
                                        af::ascir::AscIrImplCreator<af::ascir::AxpyAscIrCodegenImplV2>(),
                                        {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16, DT_UINT64, DT_INT64}}}});
REG_ASC_IR(MatMul).Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                                          af::ascir::AscIrImplCreator<af::ascir::MatMulAscIrCodegenImplV2>(),
                                          {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                                           {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(MatMulBias)
    .Impl(v2_soc_versions,
          {af::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
           af::ascir::AscIrImplCreator<af::ascir::MatMulAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(MatMulOffset)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::MatMulAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(MatMulOffsetBias)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::MatMulAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(BatchMatMul)
    .Impl(v2_soc_versions,
          {af::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
           af::ascir::AscIrImplCreator<af::ascir::BatchMatMulAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(BatchMatMulBias)
    .Impl(v2_soc_versions,
          {af::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
           af::ascir::AscIrImplCreator<af::ascir::BatchMatMulAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(BatchMatMulOffset)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::BatchMatMulAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(BatchMatMulOffsetBias)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::BatchMatMulAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(Conv2D)
    .Input("x", "T1")
    .Input("filter", "T1")
    .Output("y", "T2")
    .Attr<std::vector<int64_t>>("strides")
    .Attr<std::vector<int64_t>>("pads")
    .Attr<std::vector<int64_t>>("dilations")
    .Attr<int64_t>("groups")
    .Attr<std::string>("pad_mode")
    .Attr<std::string>("data_format")
    .Attr<int64_t>("offset_x")
    .Attr<bool>("enable_hf32")
    .Attr<int64_t>("fixed_shift_value")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<Conv2DAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16}}, {"T2", TensorType{DT_FLOAT16}}}});

REG_ASC_IR(Conv2DBias)
    .Input("x", "T1")
    .Input("filter", "T1")
    .Input("bias", "T2")
    .Output("y", "T2")
    .Attr<std::vector<int64_t>>("strides")
    .Attr<std::vector<int64_t>>("pads")
    .Attr<std::vector<int64_t>>("dilations")
    .Attr<int64_t>("groups")
    .Attr<std::string>("pad_mode")
    .Attr<std::string>("data_format")
    .Attr<int64_t>("offset_x")
    .Attr<bool>("enable_hf32")
    .Attr<int64_t>("fixed_shift_value")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<Conv2DAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16}}, {"T2", TensorType{DT_FLOAT16}}}});

REG_ASC_IR(Conv2DOffset)
    .Input("x", "T1")
    .Input("filter", "T1")
    .Input("offset_w", "T3")
    .Output("y", "T2")
    .Attr<std::vector<int64_t>>("strides")
    .Attr<std::vector<int64_t>>("pads")
    .Attr<std::vector<int64_t>>("dilations")
    .Attr<int64_t>("groups")
    .Attr<std::string>("pad_mode")
    .Attr<std::string>("data_format")
    .Attr<int64_t>("offset_x")
    .Attr<bool>("enable_hf32")
    .Attr<int64_t>("fixed_shift_value")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v2_soc_versions,
          {af::ascir::AscIrImplCreator<Conv2DAscIrAttImplV2>(),
           af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_FLOAT16}}, {"T2", TensorType{DT_FLOAT16}}, {"T3", TensorType{DT_INT8}}}});

REG_ASC_IR(Conv2DOffsetBias)
    .Input("x", "T1")
    .Input("filter", "T1")
    .Input("bias", "T2")
    .Input("offset_w", "T3")
    .Output("y", "T2")
    .Attr<std::vector<int64_t>>("strides")
    .Attr<std::vector<int64_t>>("pads")
    .Attr<std::vector<int64_t>>("dilations")
    .Attr<int64_t>("groups")
    .Attr<std::string>("pad_mode")
    .Attr<std::string>("data_format")
    .Attr<int64_t>("offset_x")
    .Attr<bool>("enable_hf32")
    .Attr<int64_t>("fixed_shift_value")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v2_soc_versions,
          {af::ascir::AscIrImplCreator<Conv2DAscIrAttImplV2>(),
           af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_FLOAT16}}, {"T2", TensorType{DT_FLOAT16}}, {"T3", TensorType{DT_INT8}}}});

REG_ASC_IR(ExtendConv2D)
    .Input("x", "T1")
    .Input("filter", "T1")
    .Output("y", "T2")
    .Attr<std::vector<int64_t>>("strides")
    .Attr<std::vector<int64_t>>("pads")
    .Attr<std::vector<int64_t>>("dilations")
    .Attr<int64_t>("groups")
    .Attr<std::string>("pad_mode")
    .Attr<std::string>("data_format")
    .Attr<int64_t>("offset_x")
    .Attr<std::string>("round_mode")
    .Attr<bool>("enable_hf32")
    .Attr<int64_t>("fixed_shift_value")
    .Attr<bool>("enable_relu0")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<Conv2DAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16}}, {"T2", TensorType{DT_FLOAT16}}}});

REG_ASC_IR(ExtendConv2DBias)
    .Input("x", "T1")
    .Input("filter", "T1")
    .Input("bias", "T2")
    .Output("y", "T3")
    .Attr<std::vector<int64_t>>("strides")
    .Attr<std::vector<int64_t>>("pads")
    .Attr<std::vector<int64_t>>("dilations")
    .Attr<int64_t>("groups")
    .Attr<std::string>("pad_mode")
    .Attr<std::string>("data_format")
    .Attr<int64_t>("offset_x")
    .Attr<std::string>("round_mode")
    .Attr<bool>("enable_hf32")
    .Attr<int64_t>("fixed_shift_value")
    .Attr<bool>("enable_relu0")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v2_soc_versions,
          {af::ascir::AscIrImplCreator<Conv2DAscIrAttImplV2>(),
           af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_FLOAT16}}, {"T2", TensorType{DT_FLOAT16}}, {"T3", TensorType{DT_FLOAT16}}}});

REG_ASC_IR(ExtendConv2DScale)
    .Input("x", "T1")
    .Input("filter", "T1")
    .Input("scale0", "T3")
    .Output("y", "T2")
    .Attr<std::vector<int64_t>>("strides")
    .Attr<std::vector<int64_t>>("pads")
    .Attr<std::vector<int64_t>>("dilations")
    .Attr<int64_t>("groups")
    .Attr<std::string>("pad_mode")
    .Attr<std::string>("data_format")
    .Attr<int64_t>("offset_x")
    .Attr<std::string>("round_mode")
    .Attr<bool>("enable_hf32")
    .Attr<int64_t>("fixed_shift_value")
    .Attr<bool>("enable_relu0")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v2_soc_versions,
          {af::ascir::AscIrImplCreator<Conv2DAscIrAttImplV2>(),
           af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_INT8}}, {"T2", TensorType{DT_FLOAT16}}, {"T3", TensorType{DT_UINT64}}}});

REG_ASC_IR(ExtendConv2DBiasScale)
    .Input("x", "T1")
    .Input("filter", "T1")
    .Input("bias", "T2")
    .Input("scale0", "T4")
    .Output("y", "T3")
    .Attr<std::vector<int64_t>>("strides")
    .Attr<std::vector<int64_t>>("pads")
    .Attr<std::vector<int64_t>>("dilations")
    .Attr<int64_t>("groups")
    .Attr<std::string>("pad_mode")
    .Attr<std::string>("data_format")
    .Attr<int64_t>("offset_x")
    .Attr<std::string>("round_mode")
    .Attr<bool>("enable_hf32")
    .Attr<int64_t>("fixed_shift_value")
    .Attr<bool>("enable_relu0")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<Conv2DAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_INT8}},
                             {"T2", TensorType{DT_INT32}},
                             {"T3", TensorType{DT_FLOAT16}},
                             {"T4", TensorType{DT_UINT64}}}});

REG_ASC_IR(Softmax)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeReduce)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SoftmaxAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::SoftmaxAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Sin)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SinAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::SinAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Acosh)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AcoshAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::AcoshAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Asin)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AsinAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::AsinAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Asinh)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AsinhAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::AsinhAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Atan)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AtanAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::AtanAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Atanh)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AtanhAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::AtanhAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

constexpr std::pair<ge::DataType, ge::DataType> kRShiftTypePairs[] = {
    {DT_INT8, DT_INT8},  {DT_INT16, DT_INT16},  {DT_INT32, DT_INT32},  {DT_INT64, DT_INT64},
    {DT_UINT8, DT_INT8}, {DT_UINT16, DT_INT16}, {DT_UINT32, DT_INT32}, {DT_UINT64, DT_INT64},
};
REG_ASC_IR(RShift)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Output("y", "T1")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::RShiftAscIrAttImplV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::RShiftAscIrCodegenImplV2>(),
                            {{"T1", MakeT1List(kRShiftTypePairs)}, {"T2", MakeT2List(kRShiftTypePairs)}}});

REG_ASC_IR(Unsupported)
    .Inputs({})
    .Output("y", "T")
    .StartNode()
    .Attr<std::string>("error_msg")
    .ComputeType(ComputeType::kComputeInvalid)
    .Impl(v2_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AscIrAttStubV2>(),
                            af::ascir::AscIrImplCreator<af::ascir::UnsupportedAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});
}  // namespace ascir
}  // namespace af
