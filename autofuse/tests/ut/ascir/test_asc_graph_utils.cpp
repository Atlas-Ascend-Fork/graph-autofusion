/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <string>

#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "ascir_ops.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"

namespace af {
namespace {
class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) : graph_(std::make_shared<ComputeGraph>(name)) {}

  NodePtr AddNode(const std::string &name, int32_t input_num, int32_t output_num) {
    auto op_desc = std::make_shared<OpDesc>(name, name);
    for (int32_t i = 0; i < input_num; ++i) {
      op_desc->AddInputDesc(GeTensorDesc());
    }
    for (int32_t i = 0; i < output_num; ++i) {
      op_desc->AddOutputDesc(GeTensorDesc());
    }
    return graph_->AddNode(op_desc);
  }

  void AddEdge(const NodePtr &src, const NodePtr &dst) {
    ASSERT_EQ(GraphUtils::AddEdge(src->GetOutDataAnchor(0), dst->GetInDataAnchor(0)), GRAPH_SUCCESS);
  }

 private:
  ComputeGraphPtr graph_;
};

AscNodeAttr *GetNodeAttr(const NodePtr &node) {
  return node->GetOpDesc()->GetOrCreateAttrsGroup<AscNodeAttr>();
}

AscTensorAttr *GetTensorAttr(const NodePtr &node, size_t index = 0U) {
  return node->GetOpDesc()->MutableOutputDesc(index)->GetOrCreateAttrsGroup<AscTensorAttr>();
}

void SetReferenceAttrs(const NodePtr &node) {
  auto node_attr = GetNodeAttr(node);
  node_attr->sched.exec_order = 7;
  node_attr->sched.axis = {1, 2};
  node_attr->sched.loop_axis = 2;
  node_attr->sched.exec_condition = ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  node_attr->api.type = ApiType::kAPITypeCompute;

  auto tensor_attr = GetTensorAttr(node);
  tensor_attr->axis = {1, 2};
  tensor_attr->repeats = {Symbol(8), Symbol(16)};
  tensor_attr->strides = {Symbol(16), Symbol(1)};
}

void SetInsertedAttrs(const NodePtr &node) {
  auto node_attr = GetNodeAttr(node);
  node_attr->sched.axis = {9};
  node_attr->api.type = ApiType::kAPITypeBuffer;

  auto tensor_attr = GetTensorAttr(node);
  node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  tensor_attr->axis = {9};
  tensor_attr->repeats = {Symbol(4)};
  tensor_attr->strides = {Symbol(1)};
  tensor_attr->vectorized_axis = {9};
  tensor_attr->vectorized_strides = {Symbol(3)};
  tensor_attr->mem.alloc_type = AllocType::kAllocTypeQueue;
  tensor_attr->que.id = 11;
  tensor_attr->buf.id = 12;
  tensor_attr->opt.reuse_id = 13;
}

void ExpectInheritedAttrs(const NodePtr &node_reference, const NodePtr &tensor_reference, const NodePtr &inserted) {
  const auto reference_node_attr = GetNodeAttr(node_reference);
  const auto inserted_node_attr = GetNodeAttr(inserted);
  EXPECT_EQ(inserted_node_attr->sched.exec_order, reference_node_attr->sched.exec_order);
  EXPECT_EQ(inserted_node_attr->sched.axis, reference_node_attr->sched.axis);
  EXPECT_EQ(inserted_node_attr->sched.loop_axis, reference_node_attr->sched.loop_axis);
  EXPECT_EQ(inserted_node_attr->sched.exec_condition, reference_node_attr->sched.exec_condition);
  EXPECT_EQ(inserted_node_attr->api.type, ApiType::kAPITypeBuffer);

  const auto reference_tensor_attr = GetTensorAttr(tensor_reference);
  const auto inserted_tensor_attr = GetTensorAttr(inserted);
  EXPECT_EQ(inserted_tensor_attr->axis, reference_tensor_attr->axis);
  EXPECT_EQ(inserted_tensor_attr->repeats, reference_tensor_attr->repeats);
  EXPECT_EQ(inserted_tensor_attr->strides, reference_tensor_attr->strides);
  EXPECT_EQ(inserted->GetOpDesc()->MutableOutputDesc(0)->GetDataType(), DT_FLOAT16);
  EXPECT_EQ(inserted_tensor_attr->vectorized_axis, std::vector<int64_t>{9});
  EXPECT_EQ(inserted_tensor_attr->vectorized_strides, std::vector<Expression>{Symbol(3)});
  EXPECT_EQ(inserted_tensor_attr->mem.alloc_type, AllocType::kAllocTypeQueue);
  EXPECT_EQ(inserted_tensor_attr->que.id, 11);
  EXPECT_EQ(inserted_tensor_attr->buf.id, 12);
  EXPECT_EQ(inserted_tensor_attr->opt.reuse_id, 13);
}
}  // namespace

TEST(AscGraphUtilsInsertNodeTest, InsertBeforeInheritsConsumerSchedAndProducerTensorAttrs) {
  GraphBuilder builder("before");
  auto producer = builder.AddNode("producer", 0, 1);
  auto consumer = builder.AddNode("consumer", 1, 1);
  auto inserted = builder.AddNode("inserted", 1, 1);
  builder.AddEdge(producer, consumer);
  SetReferenceAttrs(producer);
  SetReferenceAttrs(consumer);
  GetTensorAttr(consumer)->axis = {3};
  GetTensorAttr(consumer)->repeats = {Symbol(32)};
  GetTensorAttr(consumer)->strides = {Symbol(4)};
  SetInsertedAttrs(inserted);

  ASSERT_EQ(AscGraphUtils::InsertNodeBefore(consumer->GetInDataAnchor(0), inserted), GRAPH_SUCCESS);

  EXPECT_EQ(producer->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1U);
  EXPECT_EQ(inserted->GetInDataAnchor(0)->GetPeerOutAnchor(), producer->GetOutDataAnchor(0));
  EXPECT_EQ(inserted->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1U);
  EXPECT_EQ(consumer->GetInDataAnchor(0)->GetPeerOutAnchor(), inserted->GetOutDataAnchor(0));
  ExpectInheritedAttrs(consumer, producer, inserted);
}

TEST(AscGraphUtilsInsertNodeTest, InsertAfterInheritsProducerAutofuseAttrs) {
  GraphBuilder builder("after");
  auto producer = builder.AddNode("producer", 0, 1);
  auto consumer1 = builder.AddNode("consumer1", 1, 1);
  auto consumer2 = builder.AddNode("consumer2", 1, 1);
  auto inserted = builder.AddNode("inserted", 1, 1);
  builder.AddEdge(producer, consumer1);
  builder.AddEdge(producer, consumer2);
  SetReferenceAttrs(producer);
  SetInsertedAttrs(inserted);

  ASSERT_EQ(AscGraphUtils::InsertNodeAfter(producer->GetOutDataAnchor(0), inserted), GRAPH_SUCCESS);

  EXPECT_EQ(producer->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1U);
  EXPECT_EQ(inserted->GetInDataAnchor(0)->GetPeerOutAnchor(), producer->GetOutDataAnchor(0));
  EXPECT_EQ(inserted->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 2U);
  EXPECT_EQ(consumer1->GetInDataAnchor(0)->GetPeerOutAnchor(), inserted->GetOutDataAnchor(0));
  EXPECT_EQ(consumer2->GetInDataAnchor(0)->GetPeerOutAnchor(), inserted->GetOutDataAnchor(0));
  ExpectInheritedAttrs(producer, producer, inserted);
}

TEST(AscGraphUtilsInsertNodeTest, InsertAfterInheritsSelectedSourceOutputToSelectedInsertedOutput) {
  GraphBuilder builder("after_selected_output");
  auto producer = builder.AddNode("producer", 0, 2);
  auto consumer = builder.AddNode("consumer", 1, 1);
  auto inserted = builder.AddNode("inserted", 1, 2);
  ASSERT_EQ(GraphUtils::AddEdge(producer->GetOutDataAnchor(1), consumer->GetInDataAnchor(0)), GRAPH_SUCCESS);

  GetTensorAttr(producer, 0)->axis = {10};
  GetTensorAttr(producer, 1)->axis = {20};
  GetTensorAttr(producer, 1)->repeats = {Symbol(8)};
  GetTensorAttr(producer, 1)->strides = {Symbol(2)};
  GetTensorAttr(inserted, 0)->axis = {30};

  ASSERT_EQ(
      AscGraphUtils::InsertNodeAfter(producer->GetOutDataAnchor(1), {consumer->GetInDataAnchor(0)}, inserted, 0, 1),
      GRAPH_SUCCESS);

  EXPECT_EQ(GetTensorAttr(inserted, 0)->axis, std::vector<int64_t>{30});
  EXPECT_EQ(GetTensorAttr(inserted, 1)->axis, std::vector<int64_t>{20});
  EXPECT_EQ(GetTensorAttr(inserted, 1)->repeats, std::vector<Expression>{Symbol(8)});
  EXPECT_EQ(GetTensorAttr(inserted, 1)->strides, std::vector<Expression>{Symbol(2)});
}

TEST(AscGraphUtilsInsertNodeTest, RejectsInsertNodeFromAnotherGraph) {
  GraphBuilder first_builder("first");
  auto producer = first_builder.AddNode("producer", 0, 1);
  auto consumer = first_builder.AddNode("consumer", 1, 1);
  first_builder.AddEdge(producer, consumer);
  GraphBuilder second_builder("second");
  auto inserted = second_builder.AddNode("inserted", 1, 1);

  EXPECT_NE(AscGraphUtils::InsertNodeBefore(consumer->GetInDataAnchor(0), inserted), GRAPH_SUCCESS);
  EXPECT_EQ(consumer->GetInDataAnchor(0)->GetPeerOutAnchor(), producer->GetOutDataAnchor(0));
}

TEST(AscGraphUtilsSerializeTest, PreservesDisconnectedOptionalInputs) {
  AscGraph graph("disconnected_optional_inputs");
  auto size_var = graph.CreateSizeVar("s0");
  auto axis = graph.CreateAxis("z0", size_var);
  const auto one = Symbol(1);

  ascir_op::Rand rand_op("rand");
  graph.AddNode(rand_op);
  *rand_op.y.axis = {axis.id};
  *rand_op.y.repeats = {size_var};
  *rand_op.y.strides = {one};
  rand_op.y.dtype = DT_FLOAT;

  std::string serialized_graph;
  ASSERT_EQ(AscGraphUtils::SerializeToReadable(graph, serialized_graph), GRAPH_SUCCESS);

  AscGraph deserialized_graph("deserialized");
  ASSERT_EQ(AscGraphUtils::DeserializeFromReadable(serialized_graph, deserialized_graph), GRAPH_SUCCESS);

  const auto rand = deserialized_graph.FindNode("rand");
  ASSERT_NE(rand, nullptr);
  const auto op_desc = rand->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetIrInputs().size(), 2U);
  EXPECT_EQ(op_desc->GetIrInputs()[0].second, IrInputType::kIrInputOptional);
  EXPECT_EQ(op_desc->GetIrInputs()[1].second, IrInputType::kIrInputOptional);

  std::map<size_t, std::pair<size_t, size_t>> input_ranges;
  ASSERT_EQ(OpDescUtils::GetIrInputRawDescRange(op_desc, input_ranges), GRAPH_SUCCESS);
  EXPECT_EQ(input_ranges.size(), 2U);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 2U);
  EXPECT_EQ(op_desc->GetInputDesc("seed").GetDataType(), DT_UNDEFINED);
  EXPECT_EQ(op_desc->GetInputDesc("seed").GetFormat(), FORMAT_RESERVED);
  EXPECT_EQ(op_desc->GetInputDesc("offset").GetDataType(), DT_UNDEFINED);
  EXPECT_EQ(op_desc->GetInputDesc("offset").GetFormat(), FORMAT_RESERVED);
  EXPECT_EQ(rand->inputs.Size(), 0U);
}

TEST(AscGraphUtilsSerializeTest, PreservesOptionalInputAnchorIndexes) {
  AscGraph graph("optional_input_anchor_indexes");
  auto size_var = graph.CreateSizeVar("s0");
  auto axis = graph.CreateAxis("z0", size_var);
  const auto one = Symbol(1);

  ascir_op::Rand offset("offset");
  graph.AddNode(offset);
  *offset.y.axis = {axis.id};
  *offset.y.repeats = {one};
  *offset.y.strides = {one};
  offset.y.dtype = DT_INT32;

  ascir_op::Rand rand_op("rand");
  graph.AddNode(rand_op);
  *rand_op.y.axis = {axis.id};
  *rand_op.y.repeats = {size_var};
  *rand_op.y.strides = {one};
  rand_op.y.dtype = DT_FLOAT;
  rand_op.offset = offset.y;
  std::string serialized_graph;
  ASSERT_EQ(AscGraphUtils::SerializeToReadable(graph, serialized_graph), GRAPH_SUCCESS);

  AscGraph deserialized_graph("deserialized");
  ASSERT_EQ(AscGraphUtils::DeserializeFromReadable(serialized_graph, deserialized_graph), GRAPH_SUCCESS);

  const auto rand = deserialized_graph.FindNode("rand");
  ASSERT_NE(rand, nullptr);
  EXPECT_EQ(rand->GetInDataAnchor(0)->GetPeerOutAnchor(), nullptr);
  const auto offset_peer = rand->GetInDataAnchor(1)->GetPeerOutAnchor();
  ASSERT_NE(offset_peer, nullptr);
  EXPECT_EQ(offset_peer->GetOwnerNodeBarePtr()->GetName(), "offset");
}

}  // namespace af
