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
#include "codegen_kernel_loop.h"
#include "common/platform_context.h"

namespace codegen {
namespace {
ApiCall *FindQueueCall(const Loop &loop, const std::string &name) {
  for (const auto &body : loop.bodys) {
    if (body.type == LoopType::CALL && body.call->node_name == name) {
      return body.call;
    }
    if (body.type == LoopType::LOOP) {
      auto *call = FindQueueCall(*body.loop, name);
      if (call != nullptr) {
        return call;
      }
    }
  }
  return nullptr;
}

class QueueReuseLoopTest : public testing::Test {
 protected:
  af::AscGraph graph{"queue_reuse"};
  Tiler tiler;
  TPipe pipe{"tpipe", tiler};
  Loop root{af::kIdNone};

  void SetUp() override {
    ge::PlatformContext::GetInstance().SetPlatform("2201");
  }

  void TearDown() override {
    root.Destruct();
    ge::PlatformContext::GetInstance().Reset();
  }

  void BuildGraph() {
    using namespace af::ascir_op;
    Data data("data", graph);
    Load first("first");
    graph.AddNode(first);
    first.x = data.y;
    Abs consume("consume");
    graph.AddNode(consume);
    consume.x = first.y;
    Abs outer("outer");
    graph.AddNode(outer);
    outer.x = consume.y;
    Load second("second");
    graph.AddNode(second);
    second.x = data.y;
    Abs reuse("reuse");
    graph.AddNode(reuse);
    reuse.x = second.y;
  }

  void ConfigureAxes() {
    const af::Axis a0{.id = 0, .name = "a0", .type = af::Axis::Type::kAxisTypeOriginal, .size = af::Symbol(8)};
    const af::Axis a1{.id = 1, .name = "a1", .type = af::Axis::Type::kAxisTypeOriginal, .size = af::Symbol(16)};
    tiler.AddAxis(a0);
    tiler.AddAxis(a1);
  }

  void ConfigureNodes(bool separate_loops, bool nested) {
    int64_t id = 0;
    for (const auto &node : graph.GetAllNodes()) {
      node->attr.api.unit = node->GetType() == "Data"   ? af::ComputeUnit::kUnitNone
                            : node->GetType() == "Load" ? af::ComputeUnit::kUnitMTE2
                                                        : af::ComputeUnit::kUnitVector;
      node->attr.api.compute_type =
          node->GetType() == "Load" ? af::ComputeType::kComputeLoad : af::ComputeType::kComputeElewise;
      node->attr.api.type = af::ApiType::kAPITypeCompute;
      node->attr.sched.axis = {0, 1};
      node->attr.sched.loop_axis = 0;
      auto &attr = node->outputs[0].attr;
      attr.dtype = ge::DT_FLOAT;
      attr.axis = {0, 1};
      attr.repeats = {af::Symbol(8), af::Symbol(16)};
      attr.strides = {af::Symbol(16), af::Symbol(1)};
      attr.vectorized_axis = {1};
      attr.vectorized_strides = {af::Symbol(1)};
      attr.mem.tensor_id = id++;
      attr.mem.reuse_id = attr.mem.tensor_id;
      attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
      attr.mem.position = af::Position::kPositionVecCalc;
      attr.buf.id = attr.mem.tensor_id;
      attr.opt.merge_scope = af::kIdNone;
      if (node->GetType() == "Data") {
        attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
        attr.mem.position = af::Position::kPositionGM;
        node->attr.sched.loop_axis = af::kIdNone;
      }
      if (node->GetType() == "Load" || node->GetName() == "reuse") {
        attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
        attr.que.id = node->GetName() == "second" ? 1 : 0;
        attr.que.buf_num = 1;
        attr.mem.position = node->GetType() == "Load" ? af::Position::kPositionVecIn : af::Position::kPositionVecCalc;
      }
      if (node->GetName() == "outer" && separate_loops) {
        node->attr.sched.loop_axis = af::kIdNone;
      }
      if ((node->GetName() == "second" || node->GetName() == "reuse") && nested) {
        node->attr.sched.loop_axis = 1;
      }
    }
  }

  void BuildLoop() {
    ASSERT_EQ(pipe.CollectQues(graph), af::SUCCESS);
    for (const auto &node : graph.GetAllNodes()) {
      ASSERT_EQ(pipe.AddTensor(node->outputs[0]), af::SUCCESS);
    }
    ASSERT_EQ(root.ConstructFromNodes(graph.GetAllNodes(), tiler, pipe), af::SUCCESS);
  }

  void ConfigureGraph(bool separate_loops, bool nested) {
    ConfigureAxes();
    ConfigureNodes(separate_loops, nested);
    BuildLoop();
  }

  void Prepare(bool separate_loops, bool nested) {
    BuildGraph();
    ConfigureGraph(separate_loops, nested);
  }
};

TEST_F(QueueReuseLoopTest, SiblingLoopsWithSameAxisHaveIndependentAllocations) {
  Prepare(true, false);
  auto *first = FindQueueCall(root, "first");
  auto *consume = FindQueueCall(root, "consume");
  auto *reuse = FindQueueCall(root, "reuse");
  ASSERT_NE(first, nullptr);
  ASSERT_NE(consume, nullptr);
  ASSERT_NE(reuse, nullptr);
  EXPECT_EQ(first->axis, reuse->axis);
  EXPECT_NE(first->owner_loop, reuse->owner_loop);
  EXPECT_EQ(first->outputs[0].reuse_next, nullptr);
  EXPECT_EQ(reuse->outputs[0].reuse_from, nullptr);
  std::stringstream release, allocation;
  ASSERT_TRUE(consume->FreeInputs(pipe, release));
  ASSERT_EQ(reuse->AllocOutputs(pipe, allocation), af::SUCCESS);
  EXPECT_NE(release.str().find("q0.FreeTensor(q0_buf)"), std::string::npos);
  EXPECT_NE(allocation.str().find("q0.AllocTensor<uint8_t>()"), std::string::npos);
}

TEST_F(QueueReuseLoopTest, SameLoopPreservesReuse) {
  Prepare(false, false);
  auto *first = FindQueueCall(root, "first");
  auto *reuse = FindQueueCall(root, "reuse");
  ASSERT_NE(first, nullptr);
  ASSERT_NE(reuse, nullptr);
  EXPECT_EQ(first->outputs[0].reuse_next, &reuse->outputs[0]);
  EXPECT_EQ(reuse->outputs[0].reuse_from, &first->outputs[0]);
}

TEST_F(QueueReuseLoopTest, NestedLoopsPreserveReuse) {
  Prepare(false, true);
  auto *first = FindQueueCall(root, "first");
  auto *reuse = FindQueueCall(root, "reuse");
  ASSERT_NE(first, nullptr);
  ASSERT_NE(reuse, nullptr);
  EXPECT_NE(first->owner_loop, reuse->owner_loop);
  EXPECT_EQ(reuse->owner_loop->parent, first->owner_loop);
  EXPECT_EQ(first->outputs[0].reuse_next, &reuse->outputs[0]);
  EXPECT_EQ(reuse->outputs[0].reuse_from, &first->outputs[0]);
}
}  // namespace
}  // namespace codegen
