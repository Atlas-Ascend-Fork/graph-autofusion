/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"

#include "graph/ascendc_ir/ascir_registry.h"
#include "v1_ascir_att_impl.h"

namespace af {
namespace ascir {

class AscIrAttImplTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Each AscIrAtt impl exposes the IR name through both perf-table entry points.
#define EXPECT_ATT_IMPL_NAMED(ir_name)                                                \
  {                                                                                   \
    JOIN(ir_name, AscIrAttImpl) impl;                                                 \
    EXPECT_STREQ(static_cast<const char *>(impl.GetApiPerf()), #ir_name);             \
    EXPECT_STREQ(static_cast<const char *>(impl.GetAscendCApiPerfTable()), #ir_name); \
  }

TEST_F(AscIrAttImplTest, GetApiPerfReturnsIrName_AllClasses) {
  EXPECT_ATT_IMPL_NAMED(Add);
  EXPECT_ATT_IMPL_NAMED(Gather);
  EXPECT_ATT_IMPL_NAMED(Abs);
  EXPECT_ATT_IMPL_NAMED(Broadcast);
  EXPECT_ATT_IMPL_NAMED(Cast);
  EXPECT_ATT_IMPL_NAMED(Div);
  EXPECT_ATT_IMPL_NAMED(Erf);
  EXPECT_ATT_IMPL_NAMED(Exp);
  EXPECT_ATT_IMPL_NAMED(LogicalAnd);
  EXPECT_ATT_IMPL_NAMED(LogicalOr);
  EXPECT_ATT_IMPL_NAMED(LogicalNot);
  EXPECT_ATT_IMPL_NAMED(Maximum);
  EXPECT_ATT_IMPL_NAMED(Minimum);
  EXPECT_ATT_IMPL_NAMED(Min);
  EXPECT_ATT_IMPL_NAMED(Mul);
  EXPECT_ATT_IMPL_NAMED(Neg);
  EXPECT_ATT_IMPL_NAMED(Reciprocal);
  EXPECT_ATT_IMPL_NAMED(Relu);
  EXPECT_ATT_IMPL_NAMED(ReduceAll);
  EXPECT_ATT_IMPL_NAMED(ReduceAny);
  EXPECT_ATT_IMPL_NAMED(ReduceMax);
  EXPECT_ATT_IMPL_NAMED(ReduceArgmax);
  EXPECT_ATT_IMPL_NAMED(ReduceArgMaxMultiRPhase1);
  EXPECT_ATT_IMPL_NAMED(ReduceArgMaxMultiRPhase2);
  EXPECT_ATT_IMPL_NAMED(ReduceMean);
  EXPECT_ATT_IMPL_NAMED(ReduceMin);
  EXPECT_ATT_IMPL_NAMED(ReduceSum);
  EXPECT_ATT_IMPL_NAMED(ReduceProd);
  EXPECT_ATT_IMPL_NAMED(RemovePad);
  EXPECT_ATT_IMPL_NAMED(Rsqrt);
  EXPECT_ATT_IMPL_NAMED(Select);
  EXPECT_ATT_IMPL_NAMED(Sign);
  EXPECT_ATT_IMPL_NAMED(Sqrt);
  EXPECT_ATT_IMPL_NAMED(Sub);
  EXPECT_ATT_IMPL_NAMED(Sum);
  EXPECT_ATT_IMPL_NAMED(Tanh);
  EXPECT_ATT_IMPL_NAMED(Where);
  EXPECT_ATT_IMPL_NAMED(Ge);
  EXPECT_ATT_IMPL_NAMED(Eq);
  EXPECT_ATT_IMPL_NAMED(Ne);
  EXPECT_ATT_IMPL_NAMED(Gt);
  EXPECT_ATT_IMPL_NAMED(Le);
  EXPECT_ATT_IMPL_NAMED(Lt);
  EXPECT_ATT_IMPL_NAMED(Ub2ub);
  EXPECT_ATT_IMPL_NAMED(Load);
  EXPECT_ATT_IMPL_NAMED(Store);
  EXPECT_ATT_IMPL_NAMED(Data);
  EXPECT_ATT_IMPL_NAMED(Scalar);
  EXPECT_ATT_IMPL_NAMED(IndexExpr);
  EXPECT_ATT_IMPL_NAMED(Output);
  EXPECT_ATT_IMPL_NAMED(Workspace);
  EXPECT_ATT_IMPL_NAMED(MatMul);
  EXPECT_ATT_IMPL_NAMED(Conv2D);
  EXPECT_ATT_IMPL_NAMED(Pad);
  EXPECT_ATT_IMPL_NAMED(Nop);
  EXPECT_ATT_IMPL_NAMED(Ln);
  EXPECT_ATT_IMPL_NAMED(Isnan);
  EXPECT_ATT_IMPL_NAMED(IsFinite);
  EXPECT_ATT_IMPL_NAMED(IsInf);
  EXPECT_ATT_IMPL_NAMED(MaskedFill);
  EXPECT_ATT_IMPL_NAMED(Max);
  EXPECT_ATT_IMPL_NAMED(Mean);
  EXPECT_ATT_IMPL_NAMED(Prod);
  EXPECT_ATT_IMPL_NAMED(Any);
  EXPECT_ATT_IMPL_NAMED(All);
  EXPECT_ATT_IMPL_NAMED(Sigmoid);
  EXPECT_ATT_IMPL_NAMED(TrueDiv);
  EXPECT_ATT_IMPL_NAMED(Remainder);
  EXPECT_ATT_IMPL_NAMED(Pow);
  EXPECT_ATT_IMPL_NAMED(ClipByValue);
  EXPECT_ATT_IMPL_NAMED(Concat);
  EXPECT_ATT_IMPL_NAMED(LeakyRelu);
  EXPECT_ATT_IMPL_NAMED(BitwiseAnd);
  EXPECT_ATT_IMPL_NAMED(Transpose);
  EXPECT_ATT_IMPL_NAMED(FloorDiv);
  EXPECT_ATT_IMPL_NAMED(Gelu);
  EXPECT_ATT_IMPL_NAMED(Axpy);
}

}  // namespace ascir
}  // namespace af
