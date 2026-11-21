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
#include "ascir_ops.h"
#include "codegen_kernel.h"

using namespace af::ascir_op;
using namespace codegen;

static void AddBuffers(TPipe &tpipe, af::AscGraph &graph, int64_t count, af::Position position) {
  auto axis = graph.CreateAxis("axis", af::Symbol(128));
  auto tensor = graph.FindNode("data")->outputs[0];
  tensor.attr.axis = {axis.id};
  tensor.attr.repeats = {axis.size};
  tensor.attr.strides = {af::ops::One};
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.mem.position = position;
  for (int64_t id = 0; id < count; ++id) {
    tensor.attr.mem.tensor_id = id;
    tensor.attr.buf.id = id;
    ASSERT_EQ(tpipe.AddTensor(tensor), af::SUCCESS);
  }
}

TEST(CodegenTbufMergeSt, AllBranches) {
  af::AscGraph graph("tbuf_merge_st");
  Data data("data", graph);
  Tiler tiler;

  TPipe merged("merged", tiler);
  AddBuffers(merged, graph, 40, af::Position::kPositionVecCalc);
  merged.bufs.at(39).tmp_buf_reuse = true;
  merged.SetUsingGlobalTpipe(true);
  std::string result;
  ASSERT_EQ(merged.LocalTBufAllocLoopTwice(result), af::SUCCESS);
  EXPECT_NE(result.find("b_all"), std::string::npos);
  EXPECT_NE(result.find("GetWithOffset"), std::string::npos);
  EXPECT_NE(result.find("GetTPipePtr()->InitBuffer"), std::string::npos);

  TPipe small("small", tiler);
  AddBuffers(small, graph, 2, af::Position::kPositionVecCalc);
  result.clear();
  ASSERT_EQ(small.LocalTBufAllocLoopTwice(result), af::SUCCESS);
  EXPECT_EQ(result.find("b_all"), std::string::npos);

  TPipe input("input", tiler);
  AddBuffers(input, graph, 40, af::Position::kPositionVecIn);
  result.clear();
  ASSERT_EQ(input.LocalTBufAllocLoopTwice(result), af::SUCCESS);
  EXPECT_EQ(result.find("b_all"), std::string::npos);

  TPipe ubfuse("ubfuse", tiler);
  AddBuffers(ubfuse, graph, 40, af::Position::kPositionVecCalc);
  ubfuse.cv_fusion_type = ascir::CubeTemplateType::kUBFuse;
  result.clear();
  ASSERT_EQ(ubfuse.LocalTBufAllocLoopTwice(result), af::SUCCESS);
  EXPECT_EQ(result.find("b_all"), std::string::npos);

  TPipe legacy("legacy", tiler);
  AddBuffers(legacy, graph, 40, af::Position::kPositionVecCalc);
  legacy.SetUsingAttCalcQBTSizeConfig(false);
  result.clear();
  ASSERT_EQ(legacy.LocalTBufAllocLoopTwice(result), af::SUCCESS);
  EXPECT_EQ(result.find("b_all"), std::string::npos);
}

TEST(CodegenTbufMergeSt, ScalarDedupAndConstExpr) {
  af::AscGraph graph("scalar_dedup_st");
  Load load("load");
  graph.AddNode(load);
  auto tensor_attr = graph.FindNode("load")->outputs[0];
  tensor_attr.attr.dtype = ge::DT_FLOAT;
  tensor_attr.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  Tiler tiler;
  TPipe tpipe("tpipe", tiler);
  for (int64_t id = 0; id < 2; ++id) {
    tensor_attr.attr.mem.tensor_id = id;
    tensor_attr.attr.buf.id = id;
    std::string dtype = "float";
    std::string name = "scalar_" + std::to_string(id);
    Tensor tensor(tensor_attr, dtype, name);
    ASSERT_EQ(tensor.Init(), 0);
    tensor.const_value = id == 0 ? "1.0" : "";
    tensor.const_value_expr = af::Symbol(1);
    ASSERT_EQ(tpipe.AddTensor(tensor), af::SUCCESS);
    tpipe.need_gen_blk_tensors.push_back(tensor.id);
  }
  std::string result;
  ASSERT_EQ(tpipe.BlkTensorAllocAndInit(result), af::SUCCESS);
  EXPECT_FALSE(result.empty());
}
