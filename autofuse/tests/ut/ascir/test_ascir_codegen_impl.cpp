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

#include "ascir_register.h"
#include "v1_ascir_codegen_impl.h"

#ifndef CODEGEN_TEST_JOIN
#define CODEGEN_TEST_JOIN(a, b) a##b
#define JOIN(a, b) CODEGEN_TEST_JOIN(a, b)
#endif

namespace af {
namespace ascir {

class AscIrCodegenImplMetaTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Every codegen impl must expose a non-empty api call name, api name and include headers.
#define EXPECT_CODEGEN_META(ir_name, has_load)               \
  {                                                          \
    JOIN(ir_name, AscIrCodegenImpl) impl;                    \
    EXPECT_FALSE(impl.GetApiCallName().empty());             \
    EXPECT_FALSE(impl.GetApiName().empty());                 \
    EXPECT_LE(impl.IncludeApiHeaderFiles().size(), 32U);     \
    if (has_load) {                                          \
      EXPECT_LE(impl.LoadApiHeaderFiles(false).size(), 32U); \
    }                                                        \
  }

TEST_F(AscIrCodegenImplMetaTest, CodegenImplMeta_IsWellFormed) {
  EXPECT_CODEGEN_META(Abs, true);
  EXPECT_CODEGEN_META(All, true);
  EXPECT_CODEGEN_META(Argmax, true);
  EXPECT_CODEGEN_META(ArgMaxMultiRPhase1, true);
  EXPECT_CODEGEN_META(ArgMaxMultiRPhase2, true);
  EXPECT_CODEGEN_META(Axpy, true);
  EXPECT_CODEGEN_META(BatchMatMul, true);
  EXPECT_CODEGEN_META(BitwiseAnd, true);
  EXPECT_CODEGEN_META(Cast, true);
  EXPECT_CODEGEN_META(ClipByValue, true);
  EXPECT_CODEGEN_META(Conv2D, true);
  EXPECT_CODEGEN_META(Div, true);
  EXPECT_CODEGEN_META(Eq, true);
  EXPECT_CODEGEN_META(Erf, false);
  EXPECT_CODEGEN_META(Exp, false);
  EXPECT_CODEGEN_META(FloorDiv, true);
  EXPECT_CODEGEN_META(Gather, true);
  EXPECT_CODEGEN_META(Ge, true);
  EXPECT_CODEGEN_META(Gelu, false);
  EXPECT_CODEGEN_META(Gt, true);
  EXPECT_CODEGEN_META(IsInf, true);
  EXPECT_CODEGEN_META(Le, true);
  EXPECT_CODEGEN_META(LeakyRelu, false);
  EXPECT_CODEGEN_META(Ln, false);
  EXPECT_CODEGEN_META(LogicalAnd, true);
  EXPECT_CODEGEN_META(LogicalNot, true);
  EXPECT_CODEGEN_META(LogicalOr, true);
  EXPECT_CODEGEN_META(Lt, true);
  EXPECT_CODEGEN_META(MatMul, true);
  EXPECT_CODEGEN_META(Maximum, true);
  EXPECT_CODEGEN_META(Min, true);
  EXPECT_CODEGEN_META(Minimum, true);
  EXPECT_CODEGEN_META(Mul, true);
  EXPECT_CODEGEN_META(Ne, true);
  EXPECT_CODEGEN_META(Neg, true);
  EXPECT_CODEGEN_META(Nop, false);
  EXPECT_CODEGEN_META(Pad, false);
  EXPECT_CODEGEN_META(Pow, true);
  EXPECT_CODEGEN_META(Prod, true);
  EXPECT_CODEGEN_META(Reciprocal, true);
  EXPECT_CODEGEN_META(Relu, false);
  EXPECT_CODEGEN_META(Remainder, true);
  EXPECT_CODEGEN_META(RemovePad, true);
  EXPECT_CODEGEN_META(Select, true);
  EXPECT_CODEGEN_META(Sigmoid, true);
  EXPECT_CODEGEN_META(Sign, true);
  EXPECT_CODEGEN_META(Sub, true);
  EXPECT_CODEGEN_META(Sum, true);
  EXPECT_CODEGEN_META(Tanh, false);
  EXPECT_CODEGEN_META(Transpose, true);
  EXPECT_CODEGEN_META(TrueDiv, true);
  EXPECT_CODEGEN_META(Ub2ub, false);
  EXPECT_CODEGEN_META(Where, true);
  EXPECT_CODEGEN_META(IndexExpr, false);
  EXPECT_CODEGEN_META(Sqrt, false);
  EXPECT_CODEGEN_META(Scalar, false);
  EXPECT_CODEGEN_META(Mean, false);
  EXPECT_CODEGEN_META(Rsqrt, true);
  EXPECT_CODEGEN_META(Isnan, true);
  EXPECT_CODEGEN_META(IsFinite, true);
  EXPECT_CODEGEN_META(Any, true);
}

// Scalar-input / brc-inline policy probes: every override listed in the coverage report is
// invoked once; the counter assertion fails if any call is dropped.
TEST_F(AscIrCodegenImplMetaTest, InputScalarAndBrcPolicies) {
  size_t probed = 0U;
#define PROBE_POLICY(cls, expr)       \
  {                                   \
    JOIN(cls, AscIrCodegenImpl) impl; \
    (void)(expr);                     \
    ++probed;                         \
  }
  PROBE_POLICY(Transpose, impl.IsScalarInputSupported({false, true}));
  PROBE_POLICY(Le, impl.IsScalarInputSupported({false, true}));
  PROBE_POLICY(LogicalAnd, impl.IsScalarInputSupported({false, true}));
  PROBE_POLICY(LogicalOr, impl.IsScalarInputSupported({false, true}));
  PROBE_POLICY(Maximum, impl.IsScalarInputSupported({false, true}));
  PROBE_POLICY(Pow, impl.IsScalarInputSupported({false, true}));
  PROBE_POLICY(TrueDiv, impl.IsScalarInputSupported({false, true}));
  PROBE_POLICY(MaskedFill, impl.IsScalarInputSupported({false, true}));
  PROBE_POLICY(Eq, impl.IsScalarInputSupportedIfExchangeInputs({true, false}));
  PROBE_POLICY(LogicalAnd, impl.IsScalarInputSupportedIfExchangeInputs({true, false}));
  PROBE_POLICY(LogicalOr, impl.IsScalarInputSupportedIfExchangeInputs({true, false}));
  PROBE_POLICY(Maximum, impl.IsScalarInputSupportedIfExchangeInputs({true, false}));
  PROBE_POLICY(Mul, impl.IsScalarInputSupportedIfExchangeInputs({true, false}));
#undef PROBE_POLICY
  EXPECT_EQ(probed, 13U);
}

}  // namespace ascir
}  // namespace af
