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
#include "v1_ascir_codegen_impl.h"
#include "v1_ascir_att_impl.h"
#include "graph/types_af.h"
#include "graph/tensor_af.h"
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

class AscIrAttStub : public af::ascir::AscIrAtt {
  void *GetApiPerf() const override {
    return nullptr;
  }
  void *GetAscendCApiPerfTable() const override {
    return nullptr;
  }
};

const std::vector<std::string> v1_soc_versions{"2201"};

REG_ASC_IR(Data)
    .Inputs({})
    .Output("y", "T")
    .StartNode()
    .Attr<int64_t>("index")
    .ComputeType(ComputeType::kComputeInvalid)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::DataAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::DataAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(VectorFunc)
    .DynamicInput("x", "T")
    .DynamicOutput("y", "T")
    .Attr<std::string>("sub_graph_name")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {nullptr, nullptr, {{"T", TensorType::ALL()}}});

REG_ASC_IR(Scalar)
    .Inputs({})
    .Output("y", "T")
    .StartNode()
    .Attr<std::string>("value")
    .Attr<int64_t>("index")
    .ComputeType(ComputeType::kComputeInvalid)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ScalarAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ScalarAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(ScalarData)
    .Inputs({})
    .Output("y", "T")
    .StartNode()
    .Attr<int64_t>("index")
    .ComputeType(ComputeType::kComputeInvalid)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ScalarAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ScalarAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(IndexExpr)
    .Inputs({})
    .Output("y", "T")
    .StartNode()
    .Attr<Expression>("expr")
    .ComputeType(ComputeType::kComputeInvalid)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::IndexExprAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::IndexExprAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Output)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<int64_t>("index")
    .ComputeType(ComputeType::kComputeInvalid)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::OutputAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::OutputAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Workspace)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeInvalid)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::WorkspaceAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::WorkspaceAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Load)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<Expression>("offset")
    .ComputeType(ComputeType::kComputeLoad)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LoadAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::LoadAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16,
                                              DT_FLOAT, DT_INT64, DT_BF16}}}});

REG_ASC_IR(Store)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<Expression>("offset")
    .ComputeType(ComputeType::kComputeStore)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::StoreAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::StoreAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16,
                                              DT_FLOAT, DT_INT64, DT_BF16}}}});

REG_ASC_IR(Broadcast)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeBroadcast)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::BroadcastAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::BroadcastAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_UINT8, DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT16,
                                              DT_UINT32, DT_UINT64, DT_INT64}}}});

REG_ASC_IR(RemovePad)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::RemovePadAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::RemovePadAscIrCodegenImpl>(),
           {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Pad)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::PadAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::PadAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT}}}});

// todo: Nop DT_INT64 后面根据需要放开
REG_ASC_IR(Nop)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::NopAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::NopAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64,
                                              DT_FLOAT16, DT_FLOAT}}}});

/* cast 暂时先放开int64->float, 以下类型, 暂不放开
 * T1:DT_INT64, DT_INT64, DT_INT64, DT_INT64,
 * T2:DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_UINT64,
 */
constexpr std::pair<ge::DataType, ge::DataType> kCastTypePairs[] = {
    {DT_FLOAT, DT_FLOAT},
    {DT_FLOAT, DT_FLOAT16},
    {DT_FLOAT, DT_INT64},
    {DT_FLOAT, DT_INT32},
    {DT_FLOAT, DT_INT16},
    {DT_FLOAT, DT_BF16},
    {DT_FLOAT16, DT_FLOAT},
    {DT_FLOAT16, DT_INT32},
    {DT_FLOAT16, DT_INT16},
    {DT_FLOAT16, DT_INT8},
    {DT_FLOAT16, DT_UINT8},
    {DT_FLOAT16, DT_INT4},
    {DT_FLOAT16, DT_INT64},
    {DT_INT4, DT_FLOAT16},
    {DT_UINT8, DT_FLOAT16},
    {DT_UINT8, DT_FLOAT},
    {DT_UINT8, DT_INT32},
    {DT_UINT8, DT_INT16},
    {DT_UINT8, DT_INT8},
    {DT_UINT8, DT_INT4},
    // uint8 -> int64 由 AscendC CastExtend 通过 half、float 两个中间类型实现。
    {DT_UINT8, DT_INT64},
    {DT_INT8, DT_FLOAT16},
    {DT_INT8, DT_UINT8},
    {DT_INT16, DT_FLOAT16},
    {DT_INT16, DT_FLOAT},
    {DT_INT16, DT_UINT16},
    {DT_INT32, DT_FLOAT},
    {DT_INT32, DT_INT64},
    {DT_INT32, DT_INT16},
    {DT_INT32, DT_FLOAT16},
    {DT_INT32, DT_UINT32},
    {DT_INT64, DT_INT32},
    {DT_BF16, DT_FLOAT},
    {DT_BF16, DT_INT32},
    {DT_UINT32, DT_INT32},
    {DT_UINT16, DT_INT16},
    {DT_UINT64, DT_INT64},
};
REG_ASC_IR(Cast)
    .Input("x", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::CastAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::CastAscIrCodegenImpl>(),
                            {{"T1", MakeT1List(kCastTypePairs)}, {"T2", MakeT2List(kCastTypePairs)}}});

REG_ASC_IR(Abs)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AbsAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::AbsAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32}}}});

REG_ASC_IR(Exp)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ExpAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ExpAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Ln)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LnAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::LnAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(ArgMax)
    .Input("x", "T")
    .Output("y", "U")
    .ComputeType(ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ReduceArgMaxAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ArgMaxAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT, DT_INT32}}, {"U", TensorType{DT_INT64}}}});

// ArgMax R轴分核 Phase1: 输入x，输出value和index
REG_ASC_IR(ArgMaxMultiRPhase1)
    .Input("x", "T")
    .Output("value", "T")
    .Output("index", "U")
    .ComputeType(ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ReduceArgMaxMultiRPhase1AscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ArgMaxMultiRPhase1AscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT, DT_INT32}}, {"U", TensorType{DT_INT64}}}});

// ArgMax R轴分核 Phase2: 输入value和index，输出最终index
REG_ASC_IR(ArgMaxMultiRPhase2)
    .Input("value", "T")
    .Input("index", "U")
    .Output("y", "U")
    .ComputeType(ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ReduceArgMaxMultiRPhase2AscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ArgMaxMultiRPhase2AscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT, DT_INT32}}, {"U", TensorType{DT_INT64}}}});

REG_ASC_IR(Sqrt)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SqrtAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::SqrtAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Rsqrt)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::RsqrtAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::RsqrtAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Reciprocal)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ReciprocalAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ReciprocalAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Erf)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ErfAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ErfAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

// todo: Sign DT_INT64 后面根据需要放开
REG_ASC_IR(Sign)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SignAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::SignAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32}}}});

REG_ASC_IR(Tanh)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::TanhAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::TanhAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Isnan)
    .Input("x", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::IsnanAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::IsnanAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(IsFinite)
    .Input("x", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::IsFiniteAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::IsFiniteAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(IsInf)
    .Input("x", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::IsInfAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::IsInfAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Relu)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ReluAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ReluAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Neg)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::NegAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::NegAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

// todo: LogicalNot DT_INT64 后面根据需要放开
constexpr std::pair<ge::DataType, ge::DataType> kLogicalNotTypePairs[] = {
    {DT_FLOAT16, DT_FLOAT16}, {DT_FLOAT, DT_FLOAT}, {DT_UINT8, DT_UINT8}, {DT_INT16, DT_INT16}, {DT_INT32, DT_INT32},
};
REG_ASC_IR(LogicalNot)
    .Input("x", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LogicalNotAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::LogicalNotAscIrCodegenImpl>(),
                            {{"T1", MakeT1List(kLogicalNotTypePairs)}, {"T2", MakeT2List(kLogicalNotTypePairs)}}});

REG_ASC_IR(Max)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MaxAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::MaxAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Sum)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SumAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::SumAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT, DT_INT32}}}});

REG_ASC_IR(Min)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MinAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::MinAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Mean)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MeanAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::MeanAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Prod)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ProdAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ProdAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Sigmoid)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SigmoidAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::SigmoidAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Any)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AnyAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::AnyAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(All)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AllAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::AllAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Add)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AddAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::AddAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Sub)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::SubAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::SubAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Div)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::DivAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::DivAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Mul)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MulAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::MulAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Minimum)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MinimumAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::MinimumAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Maximum)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MaximumAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::MaximumAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

constexpr std::pair<ge::DataType, ge::DataType> kTrueDivTypePairs[] = {
    {DT_FLOAT16, DT_FLOAT16},
    {DT_FLOAT, DT_FLOAT},
    {DT_INT32, DT_FLOAT},
};
REG_ASC_IR(TrueDiv)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::TrueDivAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::TrueDivAscIrCodegenImpl>(),
                            {{"T1", MakeT1List(kTrueDivTypePairs)}, {"T2", MakeT2List(kTrueDivTypePairs)}}});

REG_ASC_IR(Remainder)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::RemainderAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::RemainderAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT, DT_INT32}}}});
// todo:LogicalOr DT_INT64 后面根据需要放开
// int64 输入由底层 Duplicate/逻辑运算实现支持后，开放对应的 AscIR 类型白名单。
REG_ASC_IR(LogicalOr)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LogicalOrAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::LogicalOrAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT64}},
                             {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(LogicalAnd)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LogicalAndAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::LogicalAndAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT64}},
                             {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Pow)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::PowAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::PowAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(ClipByValue)
    .Input("x1", "T")
    .Input("x2", "T")
    .Input("x3", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ClipByValueAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ClipByValueAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

// todo:Ge Eq Ne Gt Le  DT_INT64 后面根据需要放开
REG_ASC_IR(Ge)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::GeAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::GeAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Eq)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::EqAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::EqAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Ne)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::NeAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::NeAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Gt)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::GtAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::GtAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Le)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::LeAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::LeAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Lt)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LtAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::LtAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32}}, {"T2", TensorType{DT_UINT8}}}});

// todo:Concat DT_INT64 后面根据需要放开
REG_ASC_IR(Concat)
    .DynamicInput("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeConcat)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ConcatAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ConcatAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64,
                                              DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Select)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T2")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::SelectAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::SelectAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_UINT8}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT32, DT_INT64}}}});

REG_ASC_IR(Where)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T2")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::WhereAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::WhereAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_UINT8}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT32, DT_INT64}}}});

REG_ASC_IR(MaskedFill)
    .Input("x", "T2")
    .Input("mask", "T1")
    .Input("value", "T2")
    .Output("y", "T2")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::MaskedFillAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::MaskedFillAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_UINT8}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT}}}});

// Ub2ub是在sched阶段添加的，不需要在py构图中对外体现
// todo:Ub2ub DT_INT64 后面根据需要放开
REG_ASC_IR(Ub2ub).Input("x", "T").Output("y", "T").Impl(
    v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::Ub2ubAscIrAttImpl>(),
                      af::ascir::AscIrImplCreator<af::ascir::Ub2ubAscIrCodegenImpl>(),
                      {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64,
                                        DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(LeakyRelu)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<float>("negative_slope")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::LeakyReluAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::LeakyReluAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

// todo:BitwiseAnd DT_INT64 后面根据需要放开
REG_ASC_IR(BitwiseAnd)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::BitwiseAndAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::BitwiseAndAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT8}}}});

REG_ASC_IR(Gather)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Output("y", "T1")
    .Attr<int64_t>("axis")
    .Attr<bool>("negative_index_support")
    .ComputeType(ComputeType::kComputeGather)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<af::ascir::GatherAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::GatherAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_BF16, DT_FLOAT}},
            {"T2", TensorType{DT_INT32, DT_INT64}}}});

REG_ASC_IR(Transpose)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeTranspose)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::TransposeAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::TransposeAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT}}}});

// todo:目前前端dt构图用到了FlashSoftmax，暂时无法删除
REG_ASC_IR(FlashSoftmax)
    .Inputs({"x1", "x2", "x3"})
    .Outputs({"y1", "y2", "y3"})
    .UseFirstInputDataType()
    .ComputeType(ComputeType::kComputeElewise)
    .Impl({}, {af::ascir::AscIrImplCreator<af::ascir::AbsAscIrAttImpl>(),
               af::ascir::AscIrImplCreator<af::ascir::AscIrCodegen>(),
               {{"T1", TensorType{DT_INT8, DT_INT16}}, {"T2", TensorType{DT_UINT8, DT_INT16}}}});

REG_ASC_IR(FloorDiv)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::FloorDivAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::FloorDivAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Gelu)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::GeluAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::GeluAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Axpy)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .Attr<float>("alpha")
    .ComputeType(ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AxpyAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::AxpyAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});
REG_ASC_IR(MatMul)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("transpose_x1")
    .Attr<int64_t>("transpose_x2")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::MatMulAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(MatMulBias)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Input("bias", "T2")
    .Output("y", "T2")
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("transpose_x1")
    .Attr<int64_t>("transpose_x2")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::MatMulAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(MatMulOffset)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Input("offset_w", "T3")
    .Output("y", "T2")
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("transpose_x1")
    .Attr<int64_t>("transpose_x2")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::MatMulAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(MatMulOffsetBias)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Input("bias", "T2")
    .Input("offset_w", "T3")
    .Output("y", "T2")
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("transpose_x1")
    .Attr<int64_t>("transpose_x2")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::MatMulAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(BatchMatMul)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .Attr<int64_t>("adj_x1")
    .Attr<int64_t>("adj_x2")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::BatchMatMulAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(BatchMatMulBias)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Input("bias", "T2")
    .Output("y", "T2")
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .Attr<int64_t>("adj_x1")
    .Attr<int64_t>("adj_x2")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::BatchMatMulAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(BatchMatMulOffset)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Input("offset_w", "T3")
    .Output("y", "T2")
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .Attr<int64_t>("adj_x1")
    .Attr<int64_t>("adj_x2")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::BatchMatMulAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(BatchMatMulOffsetBias)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Input("bias", "T2")
    .Input("offset_w", "T3")
    .Output("y", "T2")
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .Attr<int64_t>("adj_x1")
    .Attr<int64_t>("adj_x2")
    .ComputeType(ComputeType::kComputeCube)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::BatchMatMulAscIrCodegenImpl>(),
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
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<Conv2DAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImpl>(),
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
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<Conv2DAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImpl>(),
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
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<Conv2DAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImpl>(),
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
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<Conv2DAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImpl>(),
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
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<Conv2DAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImpl>(),
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
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<Conv2DAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImpl>(),
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
    .Impl(v1_soc_versions,
          {af::ascir::AscIrImplCreator<Conv2DAscIrAttImpl>(),
           af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImpl>(),
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
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<Conv2DAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::Conv2DAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_INT8}},
                             {"T2", TensorType{DT_INT32}},
                             {"T3", TensorType{DT_FLOAT16}},
                             {"T4", TensorType{DT_UINT64}}}});

REG_ASC_IR(Split).Input("x", "T").DynamicOutput("y", "T").Attr<int64_t>("index").Attr<int64_t>(
    "gid");  // global_id, SplitOp的全局编号

REG_ASC_IR(Unsupported)
    .Inputs({})
    .Output("y", "T")
    .StartNode()
    .Attr<std::string>("error_msg")
    .ComputeType(ComputeType::kComputeInvalid)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::AscIrAttStub>(),
                            af::ascir::AscIrImplCreator<af::ascir::UnsupportedAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT}}}});
}  // namespace ascir
}  // namespace af
