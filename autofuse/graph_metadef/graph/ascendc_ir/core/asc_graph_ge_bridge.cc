/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ascendc_ir/ascendc_ir_core/asc_graph_ge_bridge.h"
#include "graph/utils/op_desc_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"

namespace af {

AscNodePtr AscGraphAddNodeFromOpDesc(AscGraph &asc_graph, OpDescPtr op_desc) {
  Operator op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  return asc_graph.AddNode(op);
}

AscNodePtr AscGraphAddSplitNode(AscGraph &asc_graph, const char *name, uint32_t out_num) {
  ascir_op::Split op(name);
  op.InstanceOutputy(out_num);
  return asc_graph.AddNode(static_cast<Operator &>(op));
}

// Constructs the appropriate ascir_op by type-string and adds it to the graph.
// All ascir_op constructors initialise AscNodeAttr (and ir_attr) via inline code in ascir_ops.h.
// This file is compiled into libaihac_ir.so (without AUTOFUSE_USE_GE_METADEF), so
// AscNodeAttr::Create resolves to the symbol in the same .so — no cross-library mangle issues.
AscNodePtr AscGraphAddAscirNodeByType(AscGraph &asc_graph, const char *op_type, const char *name,
                                      size_t dynamic_input_num, size_t dynamic_output_num) {
#define MAKE_NODE(OpClass)                                 \
  do {                                                     \
    ascir_op::OpClass op(name);                            \
    if (dynamic_input_num > 0U) {                          \
      op.DynamicInputRegister("x", dynamic_input_num);     \
    }                                                      \
    if (dynamic_output_num > 0U) {                         \
      op.DynamicOutputRegister("y", dynamic_output_num);   \
    }                                                      \
    return asc_graph.AddNode(static_cast<Operator &>(op)); \
  } while (false)

  const std::string type(op_type);
  if (type == ascir_op::Abs::Type) {
    MAKE_NODE(Abs);
  }
  if (type == ascir_op::Acos::Type) {
    MAKE_NODE(Acos);
  }
  if (type == ascir_op::Acosh::Type) {
    MAKE_NODE(Acosh);
  }
  if (type == ascir_op::Add::Type) {
    MAKE_NODE(Add);
  }
  if (type == ascir_op::All::Type) {
    MAKE_NODE(All);
  }
  if (type == ascir_op::Any::Type) {
    MAKE_NODE(Any);
  }
  if (type == ascir_op::Argmax::Type) {
    MAKE_NODE(Argmax);
  }
  if (type == ascir_op::ArgMaxMultiRPhase1::Type) {
    MAKE_NODE(ArgMaxMultiRPhase1);
  }
  if (type == ascir_op::ArgMaxMultiRPhase2::Type) {
    MAKE_NODE(ArgMaxMultiRPhase2);
  }
  if (type == ascir_op::Argmin::Type) {
    MAKE_NODE(Argmin);
  }
  if (type == ascir_op::ArgminWithValue::Type) {
    MAKE_NODE(ArgminWithValue);
  }
  if (type == ascir_op::ArgMinMultiRPhase1::Type) {
    MAKE_NODE(ArgMinMultiRPhase1);
  }
  if (type == ascir_op::ArgMinMultiRPhase2::Type) {
    MAKE_NODE(ArgMinMultiRPhase2);
  }
  if (type == ascir_op::ArgMinMultiRPhase2WithValue::Type) {
    MAKE_NODE(ArgMinMultiRPhase2WithValue);
  }
  if (type == ascir_op::Asin::Type) {
    MAKE_NODE(Asin);
  }
  if (type == ascir_op::Asinh::Type) {
    MAKE_NODE(Asinh);
  }
  if (type == ascir_op::Atan::Type) {
    MAKE_NODE(Atan);
  }
  if (type == ascir_op::Atan2::Type) {
    MAKE_NODE(Atan2);
  }
  if (type == ascir_op::Atanh::Type) {
    MAKE_NODE(Atanh);
  }
  if (type == ascir_op::Axpy::Type) {
    MAKE_NODE(Axpy);
  }
  if (type == ascir_op::BatchMatMul::Type) {
    MAKE_NODE(BatchMatMul);
  }
  if (type == ascir_op::BatchMatMulBias::Type) {
    MAKE_NODE(BatchMatMulBias);
  }
  if (type == ascir_op::BatchMatMulOffset::Type) {
    MAKE_NODE(BatchMatMulOffset);
  }
  if (type == ascir_op::BatchMatMulOffsetBias::Type) {
    MAKE_NODE(BatchMatMulOffsetBias);
  }
  if (type == ascir_op::BitwiseAnd::Type) {
    MAKE_NODE(BitwiseAnd);
  }
  if (type == ascir_op::BitwiseNot::Type) {
    MAKE_NODE(BitwiseNot);
  }
  if (type == ascir_op::BitwiseOr::Type) {
    MAKE_NODE(BitwiseOr);
  }
  if (type == ascir_op::BitwiseXor::Type) {
    MAKE_NODE(BitwiseXor);
  }
  if (type == ascir_op::Broadcast::Type) {
    MAKE_NODE(Broadcast);
  }
  if (type == ascir_op::Cast::Type) {
    MAKE_NODE(Cast);
  }
  if (type == ascir_op::Ceil::Type) {
    MAKE_NODE(Ceil);
  }
  if (type == ascir_op::Ceil2Int::Type) {
    MAKE_NODE(Ceil2Int);
  }
  if (type == ascir_op::ClipByValue::Type) {
    MAKE_NODE(ClipByValue);
  }
  if (type == ascir_op::Concat::Type) {
    MAKE_NODE(Concat);
  }
  if (type == ascir_op::CopySign::Type) {
    MAKE_NODE(CopySign);
  }
  if (type == ascir_op::Cos::Type) {
    MAKE_NODE(Cos);
  }
  if (type == ascir_op::Cosh::Type) {
    MAKE_NODE(Cosh);
  }
  if (type == ascir_op::Data::Type) {
    MAKE_NODE(Data);
  }
  if (type == ascir_op::Digamma::Type) {
    MAKE_NODE(Digamma);
  }
  if (type == ascir_op::Div::Type) {
    MAKE_NODE(Div);
  }
  if (type == ascir_op::Eq::Type) {
    MAKE_NODE(Eq);
  }
  if (type == ascir_op::Erf::Type) {
    MAKE_NODE(Erf);
  }
  if (type == ascir_op::Erfc::Type) {
    MAKE_NODE(Erfc);
  }
  if (type == ascir_op::Erfcx::Type) {
    MAKE_NODE(Erfcx);
  }
  if (type == ascir_op::Exp::Type) {
    MAKE_NODE(Exp);
  }
  if (type == ascir_op::Exp2::Type) {
    MAKE_NODE(Exp2);
  }
  if (type == ascir_op::Expm1::Type) {
    MAKE_NODE(Expm1);
  }
  if (type == ascir_op::FlashSoftmax::Type) {
    MAKE_NODE(FlashSoftmax);
  }
  if (type == ascir_op::Floor::Type) {
    MAKE_NODE(Floor);
  }
  if (type == ascir_op::FloorDiv::Type) {
    MAKE_NODE(FloorDiv);
  }
  if (type == ascir_op::FloorToInt::Type) {
    MAKE_NODE(FloorToInt);
  }
  if (type == ascir_op::Fma::Type) {
    MAKE_NODE(Fma);
  }
  if (type == ascir_op::Fmod::Type) {
    MAKE_NODE(Fmod);
  }
  if (type == ascir_op::Gather::Type) {
    MAKE_NODE(Gather);
  }
  if (type == ascir_op::Ge::Type) {
    MAKE_NODE(Ge);
  }
  if (type == ascir_op::Gelu::Type) {
    MAKE_NODE(Gelu);
  }
  if (type == ascir_op::Gt::Type) {
    MAKE_NODE(Gt);
  }
  if (type == ascir_op::Hypot::Type) {
    MAKE_NODE(Hypot);
  }
  if (type == ascir_op::IndexExpr::Type) {
    MAKE_NODE(IndexExpr);
  }
  if (type == ascir_op::IsFinite::Type) {
    MAKE_NODE(IsFinite);
  }
  if (type == ascir_op::Isnan::Type) {
    MAKE_NODE(Isnan);
  }
  if (type == ascir_op::Le::Type) {
    MAKE_NODE(Le);
  }
  if (type == ascir_op::LeakyRelu::Type) {
    MAKE_NODE(LeakyRelu);
  }
  if (type == ascir_op::Lgamma::Type) {
    MAKE_NODE(Lgamma);
  }
  if (type == ascir_op::Ln::Type) {
    MAKE_NODE(Ln);
  }
  if (type == ascir_op::Load::Type) {
    MAKE_NODE(Load);
  }
  if (type == ascir_op::Log10::Type) {
    MAKE_NODE(Log10);
  }
  if (type == ascir_op::Log1p::Type) {
    MAKE_NODE(Log1p);
  }
  if (type == ascir_op::Log2::Type) {
    MAKE_NODE(Log2);
  }
  if (type == ascir_op::LogicalAnd::Type) {
    MAKE_NODE(LogicalAnd);
  }
  if (type == ascir_op::LogicalNot::Type) {
    MAKE_NODE(LogicalNot);
  }
  if (type == ascir_op::LogicalOr::Type) {
    MAKE_NODE(LogicalOr);
  }
  if (type == ascir_op::LogicalXor::Type) {
    MAKE_NODE(LogicalXor);
  }
  if (type == ascir_op::LShift::Type) {
    MAKE_NODE(LShift);
  }
  if (type == ascir_op::Lt::Type) {
    MAKE_NODE(Lt);
  }
  if (type == ascir_op::MatMul::Type) {
    MAKE_NODE(MatMul);
  }
  if (type == ascir_op::MatMulBias::Type) {
    MAKE_NODE(MatMulBias);
  }
  if (type == ascir_op::MatMulOffset::Type) {
    MAKE_NODE(MatMulOffset);
  }
  if (type == ascir_op::MatMulOffsetBias::Type) {
    MAKE_NODE(MatMulOffsetBias);
  }
  if (type == ascir_op::Max::Type) {
    MAKE_NODE(Max);
  }
  if (type == ascir_op::Maximum::Type) {
    MAKE_NODE(Maximum);
  }
  if (type == ascir_op::Mean::Type) {
    MAKE_NODE(Mean);
  }
  if (type == ascir_op::Min::Type) {
    MAKE_NODE(Min);
  }
  if (type == ascir_op::Minimum::Type) {
    MAKE_NODE(Minimum);
  }
  if (type == ascir_op::Mod::Type) {
    MAKE_NODE(Mod);
  }
  if (type == ascir_op::Mul::Type) {
    MAKE_NODE(Mul);
  }
  if (type == ascir_op::Nddma::Type) {
    MAKE_NODE(Nddma);
  }
  if (type == ascir_op::Ne::Type) {
    MAKE_NODE(Ne);
  }
  if (type == ascir_op::Neg::Type) {
    MAKE_NODE(Neg);
  }
  if (type == ascir_op::Nop::Type) {
    MAKE_NODE(Nop);
  }
  if (type == ascir_op::Output::Type) {
    MAKE_NODE(Output);
  }
  if (type == ascir_op::Pad::Type) {
    MAKE_NODE(Pad);
  }
  if (type == ascir_op::Pow::Type) {
    MAKE_NODE(Pow);
  }
  if (type == ascir_op::Prod::Type) {
    MAKE_NODE(Prod);
  }
  if (type == ascir_op::Reciprocal::Type) {
    MAKE_NODE(Reciprocal);
  }
  if (type == ascir_op::Relu::Type) {
    MAKE_NODE(Relu);
  }
  if (type == ascir_op::Remainder::Type) {
    MAKE_NODE(Remainder);
  }
  if (type == ascir_op::RemovePad::Type) {
    MAKE_NODE(RemovePad);
  }
  if (type == ascir_op::Round::Type) {
    MAKE_NODE(Round);
  }
  if (type == ascir_op::RoundToInt::Type) {
    MAKE_NODE(RoundToInt);
  }
  if (type == ascir_op::RShift::Type) {
    MAKE_NODE(RShift);
  }
  if (type == ascir_op::Rsqrt::Type) {
    MAKE_NODE(Rsqrt);
  }
  if (type == ascir_op::Scalar::Type) {
    MAKE_NODE(Scalar);
  }
  if (type == ascir_op::ScalarData::Type) {
    MAKE_NODE(ScalarData);
  }
  if (type == ascir_op::Select::Type) {
    MAKE_NODE(Select);
  }
  if (type == ascir_op::Sigmoid::Type) {
    MAKE_NODE(Sigmoid);
  }
  if (type == ascir_op::Sign::Type) {
    MAKE_NODE(Sign);
  }
  if (type == ascir_op::Sin::Type) {
    MAKE_NODE(Sin);
  }
  if (type == ascir_op::Sinh::Type) {
    MAKE_NODE(Sinh);
  }
  if (type == ascir_op::Split::Type) {
    MAKE_NODE(Split);
  }
  if (type == ascir_op::Sqrt::Type) {
    MAKE_NODE(Sqrt);
  }
  if (type == ascir_op::Square::Type) {
    MAKE_NODE(Square);
  }
  if (type == ascir_op::Store::Type) {
    MAKE_NODE(Store);
  }
  if (type == ascir_op::Sub::Type) {
    MAKE_NODE(Sub);
  }
  if (type == ascir_op::Sum::Type) {
    MAKE_NODE(Sum);
  }
  if (type == ascir_op::Tan::Type) {
    MAKE_NODE(Tan);
  }
  if (type == ascir_op::Tanh::Type) {
    MAKE_NODE(Tanh);
  }
  if (type == ascir_op::Transpose::Type) {
    MAKE_NODE(Transpose);
  }
  if (type == ascir_op::TrueDiv::Type) {
    MAKE_NODE(TrueDiv);
  }
  if (type == ascir_op::Trunc::Type) {
    MAKE_NODE(Trunc);
  }
  if (type == ascir_op::TruncDiv::Type) {
    MAKE_NODE(TruncDiv);
  }
  if (type == ascir_op::TruncToInt::Type) {
    MAKE_NODE(TruncToInt);
  }
  if (type == ascir_op::Ub2ub::Type) {
    MAKE_NODE(Ub2ub);
  }
  if (type == ascir_op::VectorFunc::Type) {
    MAKE_NODE(VectorFunc);
  }
  if (type == ascir_op::Where::Type) {
    MAKE_NODE(Where);
  }
  if (type == ascir_op::Workspace::Type) {
    MAKE_NODE(Workspace);
  }
  if (type == ascir_op::Xor::Type) {
    MAKE_NODE(Xor);
  }
  if (type == ascir_op::Conv2D::Type) {
    MAKE_NODE(Conv2D);
  }
  if (type == ascir_op::Conv2DBias::Type) {
    MAKE_NODE(Conv2DBias);
  }
  if (type == ascir_op::Conv2DOffset::Type) {
    MAKE_NODE(Conv2DOffset);
  }
  if (type == ascir_op::Conv2DOffsetBias::Type) {
    MAKE_NODE(Conv2DOffsetBias);
  }
  // ExtendConv2D 四个 ASCIR 变体：无附加输入 / bias / scale0 / bias+scale0。
  if (type == ascir_op::ExtendConv2D::Type) {
    MAKE_NODE(ExtendConv2D);
  }
  if (type == ascir_op::ExtendConv2DBias::Type) {
    MAKE_NODE(ExtendConv2DBias);
  }
  if (type == ascir_op::ExtendConv2DScale::Type) {
    MAKE_NODE(ExtendConv2DScale);
  }
  if (type == ascir_op::ExtendConv2DBiasScale::Type) {
    MAKE_NODE(ExtendConv2DBiasScale);
  }

#undef MAKE_NODE

  GELOGE(FAILED, "AscGraphAddAscirNodeByType: unknown op type '%s'", op_type);
  return nullptr;
}

AxisId AscGraphCreateAxisBySerializedExpr(AscGraph &asc_graph, const char *name, const char *serialized_expr) {
  auto expr = Expression::Deserialize(serialized_expr);
  return asc_graph.CreateAxis(name, expr).id;
}

bool IsScalarInputBySerializedExprs(const std::vector<std::string> &serialized_exprs) {
  std::vector<Expression> repeats;
  repeats.reserve(serialized_exprs.size());
  for (const auto &serialized_expr : serialized_exprs) {
    repeats.emplace_back(Expression::Deserialize(serialized_expr.c_str()));
  }
  return ascgen_utils::IsScalarInput(repeats);
}

AscTensorAttr *AscTensorAttrGetOrCreateForOpOutput(void *op_desc_raw, uint32_t index) {
  return AscTensorAttr::GetOrCreateFromOpDescRaw(op_desc_raw, index);
}

AscNodeAttr *AscNodeAttrGetOrCreateForOp(void *op_desc_raw) {
  GE_ASSERT_NOTNULL(op_desc_raw);
  auto *op_desc = static_cast<OpDesc *>(op_desc_raw);
  auto attr_group = op_desc->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(attr_group);
  return attr_group;
}

AscNodePtr AscGraphAddNodeFromOpDescRaw(AscGraph &asc_graph, void *op_desc_raw) {
  GE_ASSERT_NOTNULL(op_desc_raw);
  auto *op_desc = static_cast<OpDesc *>(op_desc_raw);
  return AscGraphAddNodeFromOpDesc(asc_graph, op_desc->shared_from_this());
}

}  // namespace af
