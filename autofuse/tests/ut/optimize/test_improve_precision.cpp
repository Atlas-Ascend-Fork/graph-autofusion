/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "asc_graph_builder.h"
#define private public
#include "optimize/pre_process/improve_precision.h"
#include "optimize/pre_process/pre_process_config.h"
#undef private
#include "tests/framework/improve_precision_test_utils.h"
#include "platform_context.h"
#include "runtime_stub.h"

using namespace af;
using namespace af::testing;
using namespace af::pre_process;
using namespace af::ascir_op;

namespace {

bool HasNodeWithName(AscGraph &graph, const std::string &name) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetName() == name) {
      return true;
    }
  }
  return false;
}

class TestImprovePrecisionUT : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    PreProcessConfig::Instance().Reset();
    auto stub_v1 = std::make_shared<ge::RuntimeStub>();
    ge::RuntimeStub::SetInstance(stub_v1);
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
    unsetenv("AUTOFUSE_FLAGS");
    PreProcessConfig::Instance().Reset();
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};
}  // namespace

TEST_F(TestImprovePrecisionUT, Fp16ToFp16CastBeforeStore_CastDeleted) {
  auto graph = AscGraphBuilder("ut_fp16_to_fp16_cast_before_store")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Cast("cast0", "abs0", ge::DT_FLOAT16)
                   .Store("store0", "cast0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_FALSE(HasNodeWithName(graph, "cast0"));
}

TEST_F(TestImprovePrecisionUT, Fp32ToFp16CastBeforeStore_CastPreserved) {
  auto graph = AscGraphBuilder("ut_fp32_to_fp16_before_store")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Cast("cast0", "abs0", ge::DT_FLOAT16)
                   .Store("store0", "cast0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_TRUE(HasNodeWithName(graph, "cast0"));
  EXPECT_TRUE(HasCastOutputDtype(graph, ge::DT_FLOAT16));
}

TEST_F(TestImprovePrecisionUT, UnsupportedCastChain_PreservesRequiredCast) {
  auto graph = AscGraphBuilder("ut_unsupported_cast_chain")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_BF16)
                   .Load("load0", "data0")
                   .Cast("cast_bf16_to_fp32", "load0", ge::DT_FLOAT)
                   .Cast("cast_fp32_identity", "cast_bf16_to_fp32", ge::DT_FLOAT)
                   .Cast("cast_fp32_to_fp16", "cast_fp32_identity", ge::DT_FLOAT16)
                   .Store("store0", "cast_fp32_to_fp16")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_EQ(CountNodesByType(graph, Cast::Type), 2U);
  EXPECT_TRUE(HasCastOutputDtype(graph, ge::DT_FLOAT));
  EXPECT_TRUE(HasCastOutputDtype(graph, ge::DT_FLOAT16));
}

TEST_F(TestImprovePrecisionUT, NonFloatSourceCastChain_FallbackDeletesIdentity) {
  auto graph = AscGraphBuilder("ut_non_float_source")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_INT32)
                   .Load("load0", "data0")
                   .Cast("cast_int32_to_fp32", "load0", ge::DT_FLOAT)
                   .Cast("cast_fp32_identity", "cast_int32_to_fp32", ge::DT_FLOAT)
                   .Store("store0", "cast_fp32_identity")
                   .Output("output0", "store0", 0, ge::DT_FLOAT)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_FALSE(HasNodeWithName(graph, "cast_fp32_identity"));
}

TEST_F(TestImprovePrecisionUT, LoadTransposeKeepsTransposeInLowPrecision) {
  auto graph = AscGraphBuilder("ut_load_transpose_low_precision")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Transpose("transpose0", "load0", {1, 0})
                   .Abs("abs0", "transpose0")
                   .Store("store0", "abs0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  const auto load = graph.FindNode("load0");
  const auto transpose = graph.FindNode("transpose0");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(transpose, nullptr);
  ASSERT_EQ(load->GetOutDataNodes().size(), 1U);
  EXPECT_EQ(load->GetOutDataNodes().at(0)->GetName(), "transpose0");
  EXPECT_EQ(transpose->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transpose->outputs[0].attr.dtype, ge::DT_FLOAT16);
  ASSERT_EQ(transpose->GetOutDataNodes().size(), 1U);
  EXPECT_EQ(transpose->GetOutDataNodes().at(0)->GetType(), Cast::Type);
  EXPECT_TRUE(HasCastOutputDtype(graph, ge::DT_FLOAT));
}

TEST_F(TestImprovePrecisionUT, TransposeStoreMovesDowncastBeforeTranspose) {
  auto graph = AscGraphBuilder("ut_transpose_store_low_precision")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Transpose("transpose0", "abs0", {1, 0})
                   .Store("store0", "transpose0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  const auto transpose = graph.FindNode("transpose0");
  const auto store = graph.FindNode("store0");
  ASSERT_NE(transpose, nullptr);
  ASSERT_NE(store, nullptr);
  ASSERT_EQ(transpose->GetInDataNodes().size(), 1U);
  EXPECT_EQ(transpose->GetInDataNodes().at(0)->GetType(), Cast::Type);
  ASSERT_EQ(transpose->GetInDataNodes().at(0)->GetInDataNodes().size(), 1U);
  EXPECT_EQ(transpose->GetInDataNodes().at(0)->GetInDataNodes().at(0)->GetOpDesc()->GetOutputDesc(0).GetDataType(),
            ge::DT_FLOAT);
  EXPECT_EQ(transpose->GetInDataNodes().at(0)->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transpose->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transpose->outputs[0].attr.dtype, ge::DT_FLOAT16);
  ASSERT_EQ(store->GetInDataNodes().size(), 1U);
  EXPECT_EQ(store->GetInDataNodes().at(0)->GetName(), "transpose0");
}

TEST_F(TestImprovePrecisionUT, SharedTransposeKeepsStoreLocalDowncast) {
  auto graph = AscGraphBuilder("ut_shared_transpose_store")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0, ge::DT_FLOAT)
                   .Load("load0", "data0")
                   .Transpose("transpose0", "load0", {1, 0})
                   .Store("store0", "transpose0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Abs("abs0", "transpose0")
                   .Store("store1", "abs0")
                   .Output("output1", "store1", 1, ge::DT_FLOAT)
                   .Build();

  const auto store = graph.FindNode("store0");
  ASSERT_NE(store, nullptr);
  store->outputs[0].attr.dtype = ge::DT_FLOAT16;
  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  const auto transpose = graph.FindNode("transpose0");
  const auto abs = graph.FindNode("abs0");
  ASSERT_NE(transpose, nullptr);
  ASSERT_NE(abs, nullptr);
  EXPECT_EQ(transpose->outputs[0].attr.dtype, ge::DT_FLOAT);
  ASSERT_EQ(store->GetInDataNodes().size(), 1U);
  EXPECT_EQ(store->GetInDataNodes().at(0)->GetType(), Cast::Type);
  EXPECT_EQ(store->GetInDataNodes().at(0)->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  ASSERT_EQ(abs->GetInDataNodes().size(), 1U);
  EXPECT_EQ(abs->GetInDataNodes().at(0)->GetName(), "transpose0");
}

TEST_F(TestImprovePrecisionUT, SharedTransposeMovesDowncastWhenAllStoresUseSameLowPrecision) {
  auto graph = AscGraphBuilder("ut_shared_transpose_low_precision_stores")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0, ge::DT_FLOAT)
                   .Load("load0", "data0")
                   .Abs("compute0", "load0")
                   .Transpose("transpose0", "compute0", {1, 0})
                   .Store("store0", "transpose0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Store("store1", "transpose0")
                   .Output("output1", "store1", 1, ge::DT_FLOAT16)
                   .Build();

  const auto store0 = graph.FindNode("store0");
  const auto store1 = graph.FindNode("store1");
  ASSERT_NE(store0, nullptr);
  ASSERT_NE(store1, nullptr);
  store0->outputs[0].attr.dtype = ge::DT_FLOAT16;
  store1->outputs[0].attr.dtype = ge::DT_FLOAT16;
  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  const auto transpose = graph.FindNode("transpose0");
  ASSERT_NE(transpose, nullptr);
  ASSERT_EQ(transpose->GetInDataNodes().size(), 1U);
  EXPECT_EQ(transpose->GetInDataNodes().at(0)->GetType(), Cast::Type);
  EXPECT_EQ(transpose->GetInDataNodes().at(0)->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(transpose->outputs[0].attr.dtype, ge::DT_FLOAT16);
  EXPECT_EQ(store0->GetInDataNodes().at(0)->GetName(), "transpose0");
  EXPECT_EQ(store1->GetInDataNodes().at(0)->GetName(), "transpose0");
  EXPECT_EQ(CountNodesByType(graph, Cast::Type), 1U);
}

// 复现 Inductor frexp 融合图：data(fp32) -> load -> frexp(mantissa:fp32 / exponent:int32)
//   -> store0(fp32) / store1(int32)。多输出异构 dtype 算子的非 0 号输出直连 Store 时，
// 不应按 peer 输出 0 的 dtype 误判插入 Cast（曾导致 Store 输入 half / 输出 int32 校验失败）
TEST_F(TestImprovePrecisionUT, MultiOutputHeteroDtypeOpBeforeStore_NoCastInserted) {
  auto graph = AscGraphBuilder("ut_frexp_hetero_dtype_store")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT)
                   .Load("load0", "data0")
                   .Frexp("frexp0", "load0")
                   .Store("store0", "frexp0:0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT)
                   .Store("store1", "frexp0:1")
                   .Output("output1", "store1", 1, ge::DT_INT32)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_EQ(CountNodesByType(graph, Cast::Type), 0U);
  const auto frexp = graph.FindNode("frexp0");
  ASSERT_NE(frexp, nullptr);
  EXPECT_EQ(frexp->GetOpDesc()->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(frexp->GetOpDesc()->GetOutputDesc(1).GetDataType(), ge::DT_INT32);
  const auto store0 = graph.FindNode("store0");
  const auto store1 = graph.FindNode("store1");
  ASSERT_NE(store0, nullptr);
  ASSERT_NE(store1, nullptr);
  ASSERT_EQ(store0->GetInDataNodes().size(), 1U);
  EXPECT_EQ(store0->GetInDataNodes().at(0)->GetName(), "frexp0");
  ASSERT_EQ(store1->GetInDataNodes().size(), 1U);
  EXPECT_EQ(store1->GetInDataNodes().at(0)->GetName(), "frexp0");
}
