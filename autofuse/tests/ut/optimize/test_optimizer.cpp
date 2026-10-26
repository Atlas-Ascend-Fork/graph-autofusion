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

#include <ascendc_ir.h>
#include <array>
#include <map>
#include <set>
#include "ascir.h"
#include <ascir_ops.h>
#include <ascir_utils.h>
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#define private public
#include "optimize.h"
#include "platform_context.h"
#include "optimize/graph_pass/pass_runner_handler.h"
#undef private
#include "asc_tensor_utils.h"
#include "ascgraph_info_complete.h"
#include "ascir_ops_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "codegen.h"
#include "fusion/autofuse_attrs.h"
#include "task_generator/transpose_schedule_case_generator.h"
#include "task_generator/reduce_schedule_case_generator.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"
#include "attribute_group/attr_group_shape_env.h"
#include "autoschedule/tiling_group.h"
#include "expression/testcase/source_stub.h"
#include "util/mem_utils.h"
#include "platform/platform_factory.h"
#include "runtime_stub.h"
#include "asc_graph_builder.h"
#include "codegen.h"
#include "optimize/graph_pass/pass_utils.h"
#include "common/autofuse_backend_spec_api.h"
#include "autoschedule/alignment_handler.h"
#include "../../common/att/soc_perf_test_config.h"
#include "../../v35/ut/optimize/runtime_stub.h"

using namespace af;
using namespace af::ops;
using namespace af::ascir_op;
using ge::GraphInputShapeSourceStub;
using ge::InputValueSumSourceStub;
using ge::RuntimeStub;

namespace {
enum class ArangeConsumer {
  kAdd,
  kSub,
  kMul,
  kCompare,
  kFloorDiv,
  kMinimum,
  kMaximum,
};

std::vector<std::string> GetNodeNames(const af::AscGraph &graph) {
  std::vector<std::string> names;
  for (const auto &node : graph.GetAllNodes()) {
    names.emplace_back(node->GetNamePtr());
  }
  return names;
}

std::string GetLineContaining(const std::string &text, const std::string &needle) {
  const auto offset = text.find(needle);
  if (offset == std::string::npos) {
    return {};
  }
  const auto line_begin = text.rfind('\n', offset);
  const auto line_end = text.find('\n', offset);
  return text.substr(line_begin == std::string::npos ? 0UL : line_begin + 1UL,
                     line_end == std::string::npos ? std::string::npos : line_end - line_begin - 1UL);
}

af::AscGraph CreateArangeConsumerGraph(const std::string &name, ArangeConsumer consumer,
                                       const af::Expression &vector_stride = af::ops::One,
                                       af::DataType dtype = af::DT_INT32,
                                       const std::array<af::Expression, 4> &arange_params = {
                                           af::Symbol(0), af::Symbol(1), af::Symbol(1), af::Symbol(1)}) {
  af::AscGraph graph(name.c_str());
  const auto size = graph.CreateSizeVar("size");
  const auto axis = graph.CreateAxis("axis", size);

  Arange arange("arange");
  arange.ir_attr.SetBase(arange_params[0]);
  arange.ir_attr.SetStep(arange_params[1]);
  arange.attr.sched.axis = {axis.id};
  arange.attr.api.unit = af::ComputeUnit::kUnitVector;
  arange.y.dtype = dtype;
  *arange.y.axis = {axis.id};
  *arange.y.repeats = {size};
  *arange.y.strides = {af::ops::One};
  *arange.y.vectorized_axis = {axis.id};
  *arange.y.vectorized_strides = {af::ops::One};
  (void)graph.AddNode(arange);

  Arange other_arange("other_arange");
  other_arange.ir_attr.SetBase(arange_params[2]);
  other_arange.ir_attr.SetStep(arange_params[3]);
  other_arange.attr.sched.axis = {axis.id};
  other_arange.attr.api.unit = af::ComputeUnit::kUnitVector;
  other_arange.y.dtype = dtype;
  *other_arange.y.axis = {axis.id};
  *other_arange.y.repeats = {size};
  *other_arange.y.strides = {af::ops::One};
  *other_arange.y.vectorized_axis = {axis.id};
  *other_arange.y.vectorized_strides = {af::ops::One};
  (void)graph.AddNode(other_arange);

  Store store("store");
  if (consumer == ArangeConsumer::kCompare) {
    Lt compare("compare");
    compare.x1 = arange.y;
    compare.x2 = other_arange.y;
    compare.attr.sched.axis = {axis.id};
    compare.attr.api.unit = af::ComputeUnit::kUnitVector;
    compare.y.dtype = af::DT_UINT8;
    *compare.y.axis = {axis.id};
    *compare.y.repeats = {size};
    *compare.y.strides = {vector_stride};
    *compare.y.vectorized_axis = {axis.id};
    *compare.y.vectorized_strides = {vector_stride};
    store.x = compare.y;
  } else if (consumer == ArangeConsumer::kFloorDiv) {
    FloorDiv floor_div("floor_div");
    floor_div.x1 = arange.y;
    floor_div.x2 = other_arange.y;
    floor_div.attr.sched.axis = {axis.id};
    floor_div.attr.api.unit = af::ComputeUnit::kUnitVector;
    floor_div.y.dtype = dtype;
    *floor_div.y.axis = {axis.id};
    *floor_div.y.repeats = {size};
    *floor_div.y.strides = {af::ops::One};
    *floor_div.y.vectorized_axis = {axis.id};
    *floor_div.y.vectorized_strides = {af::ops::One};
    store.x = floor_div.y;
  } else {
    auto configure = [&](auto &op) {
      op.attr.sched.axis = {axis.id};
      op.attr.api.unit = af::ComputeUnit::kUnitVector;
      op.y.dtype = dtype;
      *op.y.axis = {axis.id};
      *op.y.repeats = {size};
      *op.y.strides = {af::ops::One};
      *op.y.vectorized_axis = {axis.id};
      *op.y.vectorized_strides = {af::ops::One};
      store.x = op.y;
    };
    if (consumer == ArangeConsumer::kAdd) {
      Add op("add");
      op.x1 = arange.y;
      op.x2 = other_arange.y;
      configure(op);
    } else if (consumer == ArangeConsumer::kSub) {
      Sub op("sub");
      op.x1 = arange.y;
      op.x2 = other_arange.y;
      configure(op);
    } else if (consumer == ArangeConsumer::kMul) {
      Mul op("mul");
      op.x1 = arange.y;
      op.x2 = other_arange.y;
      configure(op);
    } else if (consumer == ArangeConsumer::kMinimum) {
      Minimum op("minimum");
      op.x1 = arange.y;
      op.x2 = other_arange.y;
      configure(op);
    } else if (consumer == ArangeConsumer::kMaximum) {
      Maximum op("maximum");
      op.x1 = arange.y;
      op.x2 = other_arange.y;
      configure(op);
    }
  }

  store.attr.sched.axis = {axis.id};
  store.attr.api.unit = af::ComputeUnit::kUnitMTE3;
  store.y.dtype = consumer == ArangeConsumer::kCompare ? af::DT_UINT8 : dtype;
  *store.y.axis = {axis.id};
  *store.y.repeats = {size};
  *store.y.strides = {af::ops::One};
  *store.y.vectorized_axis = {axis.id};
  *store.y.vectorized_strides = {af::ops::One};

  Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  return graph;
}

af::AscGraph CreateWindowMathGraph(const std::string &name) {
  af::AscGraph graph(name.c_str());
  const auto size = graph.CreateSizeVar("size");
  const auto axis = graph.CreateAxis("axis", size);
  auto set_tensor = [&](auto &tensor) {
    tensor.dtype = af::DT_INT64;
    *tensor.axis = {axis.id};
    *tensor.repeats = {size};
    *tensor.strides = {af::ops::One};
    *tensor.vectorized_axis = {axis.id};
    *tensor.vectorized_strides = {af::ops::One};
  };

  Arange arange("arange");
  arange.ir_attr.SetBase(af::Symbol(-17));
  arange.ir_attr.SetStep(af::Symbol(1));
  arange.attr.sched.axis = {axis.id};
  arange.attr.api.unit = af::ComputeUnit::kUnitVector;
  set_tensor(arange.y);
  (void)graph.AddNode(arange);
  Arange divisor("divisor");
  divisor.ir_attr.SetBase(af::Symbol(3));
  divisor.ir_attr.SetStep(af::Symbol(0));
  divisor.attr.sched.axis = {axis.id};
  divisor.attr.api.unit = af::ComputeUnit::kUnitVector;
  set_tensor(divisor.y);
  (void)graph.AddNode(divisor);
  FloorDiv floor_div("floor_div");
  floor_div.x1 = arange.y;
  floor_div.x2 = divisor.y;
  floor_div.attr.sched.axis = {axis.id};
  floor_div.attr.api.unit = af::ComputeUnit::kUnitVector;
  set_tensor(floor_div.y);
  Minimum minimum("minimum");
  minimum.x1 = floor_div.y;
  minimum.x2 = divisor.y;
  minimum.attr.sched.axis = {axis.id};
  minimum.attr.api.unit = af::ComputeUnit::kUnitVector;
  set_tensor(minimum.y);
  Maximum maximum("maximum");
  maximum.x1 = minimum.y;
  maximum.x2 = arange.y;
  maximum.attr.sched.axis = {axis.id};
  maximum.attr.api.unit = af::ComputeUnit::kUnitVector;
  set_tensor(maximum.y);
  Store store("store");
  store.x = maximum.y;
  store.attr.sched.axis = {axis.id};
  store.attr.api.unit = af::ComputeUnit::kUnitMTE3;
  set_tensor(store.y);
  Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  return graph;
}

af::AscGraph CreateArangeStoreGraph(const std::string &name) {
  af::AscGraph graph(name.c_str());
  const auto size = graph.CreateSizeVar("size");
  const auto axis = graph.CreateAxis("axis", size);

  Arange arange("arange");
  arange.ir_attr.SetBase(af::Symbol(0));
  arange.ir_attr.SetStep(af::Symbol(1));
  arange.attr.sched.axis = {axis.id};
  arange.attr.api.unit = af::ComputeUnit::kUnitVector;
  arange.y.dtype = af::DT_INT32;
  *arange.y.axis = {axis.id};
  *arange.y.repeats = {size};
  *arange.y.strides = {af::ops::One};
  *arange.y.vectorized_axis = {axis.id};
  *arange.y.vectorized_strides = {af::ops::One};
  (void)graph.AddNode(arange);

  Store store("store");
  store.x = arange.y;
  store.attr.sched.axis = {axis.id};
  store.attr.api.unit = af::ComputeUnit::kUnitMTE3;
  store.y.dtype = af::DT_INT32;
  *store.y.axis = {axis.id};
  *store.y.repeats = {size};
  *store.y.strides = {af::ops::One};
  *store.y.vectorized_axis = {axis.id};
  *store.y.vectorized_strides = {af::ops::One};

  Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  return graph;
}

class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) {
    graph_ = std::make_shared<ComputeGraph>(name);
  }

  GraphBuilder(const std::string &name, const std::string &node_type) {
    graph_ = std::make_shared<ComputeGraph>(name);
    node_type_ = node_type;
  }

  NodePtr AddNode(const std::string &name, const std::string &type, const int in_cnt, const int out_cnt,
                  const std::vector<int64_t> shape = {1, 1, 1, 1}) {
    auto tensor_desc = std::make_shared<GeTensorDesc>();
    tensor_desc->SetShape(GeShape(std::move(shape)));
    tensor_desc->SetFormat(ge::FORMAT_NCHW);
    tensor_desc->SetDataType(ge::DT_FLOAT);

    auto op_desc = std::make_shared<OpDesc>(name, (node_type_ == "") ? type : "AscGraph");
    for (std::int32_t i = 0; i < in_cnt; ++i) {
      op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (std::int32_t i = 0; i < out_cnt; ++i) {
      op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    op_desc->AddInferFunc([](Operator &op) { return ge::GRAPH_SUCCESS; });
    return graph_->AddNode(op_desc);
  }

  void AddDataEdge(const NodePtr &src_node, const std::int32_t src_idx, const NodePtr &dst_node,
                   const std::int32_t dst_idx) {
    GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
  }

  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
  std::string node_type_;
};
}  // namespace

class TestOptimizer : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v1 = std::make_shared<RuntimeStub>();
    RuntimeStub::SetInstance(stub_v1);
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  optimize::Optimizer optimizer;

  TestOptimizer() : optimizer(optimize::OptimizerOptions{}) {}

  static void SetUpV2RuntimeStub() {
    RuntimeStub::SetInstance(std::make_shared<af::RuntimeStubV2>());
  }

  static std::stringstream &SizeExprListStr(std::stringstream &ss, const af::AscGraph &graph,
                                            const std::vector<af::Expression> &size_expr_list) {
    for (auto &size_expr : size_expr_list) {
      ss << std::string(size_expr.Str().get()) << ", ";
    }
    return ss;
  }

  static std::stringstream &AxisListStr(std::stringstream &ss, af::AscGraph &graph,
                                        const std::vector<af::AxisId> &axis_list) {
    for (auto axis_id : axis_list) {
      ss << graph.FindAxis(axis_id)->name << ", ";
    }
    return ss;
  }
};

TEST_F(TestOptimizer, TwoWorkspace) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Store store_op1("store1");
  Store store_op2("store2");

  Workspace workspace_op1("workspace1");
  Workspace workspace_op2("workspace2");

  Load load_op1("load1");
  Load load_op2("load2");

  Output y_op1("y1");
  Output y_op2("y2");

  x_op.y.dtype = ge::DT_FLOAT16;
  store_op1.x = x_op.y;
  store_op1.y.dtype = ge::DT_FLOAT16;
  *store_op1.y.axis = {z0.id, z1.id};
  x_op.ir_attr.SetIndex(0);

  workspace_op1.x = store_op1.y;
  workspace_op1.y.dtype = ge::DT_FLOAT16;
  *workspace_op1.y.axis = {z0.id, z1.id};

  store_op2.x = x_op.y;
  store_op2.y.dtype = ge::DT_FLOAT16;
  *store_op2.y.axis = {z0.id, z1.id};

  workspace_op2.x = store_op2.y;
  workspace_op2.y.dtype = ge::DT_FLOAT16;
  *workspace_op2.y.axis = {z0.id, z1.id};

  load_op1.x = workspace_op1.y;
  load_op1.y.dtype = ge::DT_FLOAT16;
  *load_op1.y.axis = {z0.id, z1.id};

  load_op2.x = workspace_op2.y;
  load_op2.y.dtype = ge::DT_FLOAT16;
  *load_op2.y.axis = {z0.id, z1.id};

  y_op1.x = load_op1.y;
  y_op2.x = load_op2.y;
  y_op1.ir_attr.SetIndex(0);
  y_op2.ir_attr.SetIndex(1);

  // graph.SetInputs({x_op});
  // graph.SetOutputs({y_op1, y_op2});

  auto x = graph.FindNode("x");
  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto workspace1 = graph.FindNode("workspace1");
  auto workspace2 = graph.FindNode("workspace2");
  auto store1 = graph.FindNode("store1");
  auto store2 = graph.FindNode("store2");
  auto y1 = graph.FindNode("y1");
  auto y2 = graph.FindNode("y2");

  optimizer.BufQueAlloc(graph, graph);
  EXPECT_EQ(workspace1->outputs[0].attr.mem.tensor_id, store1->outputs[0].attr.mem.tensor_id);
  EXPECT_EQ(workspace2->outputs[0].attr.mem.tensor_id, store2->outputs[0].attr.mem.tensor_id);
}

TEST_F(TestOptimizer, V2ArangeCompareOptimizeProducesDeterministicVectorFunc) {
  SetUpV2RuntimeStub();
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  std::vector<std::string> expected_root_order;
  std::vector<std::string> expected_vf_order;
  for (size_t run = 0UL; run < 3UL; ++run) {
    auto graph = CreateArangeConsumerGraph("arange_compare_vf", ArangeConsumer::kCompare);
    ::ascir::FusedScheduledResult result;
    ASSERT_EQ(optimizer.Optimize(graph, result), af::SUCCESS);
    ASSERT_EQ(result.node_idx_to_scheduled_results.size(), 1UL);
    ASSERT_EQ(result.node_idx_to_scheduled_results[0].size(), 1UL);
    ASSERT_EQ(result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
    ASSERT_EQ(result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
    const auto &impl_graph = result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
    std::vector<AscGraph> subgraphs;
    ASSERT_EQ(impl_graph.GetAllSubGraphs(subgraphs), af::SUCCESS);
    ASSERT_EQ(subgraphs.size(), 1UL);
    const auto root_order = GetNodeNames(impl_graph);
    const auto vf_order = GetNodeNames(subgraphs[0]);
    EXPECT_EQ(root_order, (std::vector<std::string>{"arange_compare_vf_0_B0Y0_S0G0C0_VfNode_0", "store", "output"}));
    EXPECT_EQ(vf_order,
              (std::vector<std::string>{"arange", "other_arange", "compare", "Store_compare0", "Output_compare0"}));
    if (run == 0UL) {
      expected_root_order = root_order;
      expected_vf_order = vf_order;
    } else {
      EXPECT_EQ(root_order, expected_root_order);
      EXPECT_EQ(vf_order, expected_vf_order);
    }
  }
}

TEST_F(TestOptimizer, V2ArangeStoreOptimizeProducesSingletonVectorFunc) {
  SetUpV2RuntimeStub();
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  auto graph = CreateArangeStoreGraph("arange_store_vf");
  ::ascir::FusedScheduledResult result;

  ASSERT_EQ(optimizer.Optimize(graph, result), af::SUCCESS);
  ASSERT_EQ(result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);

  const auto &impl_graph = result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  std::vector<AscGraph> subgraphs;
  ASSERT_EQ(impl_graph.GetAllSubGraphs(subgraphs), af::SUCCESS);
  ASSERT_EQ(subgraphs.size(), 1UL);
  EXPECT_EQ(GetNodeNames(impl_graph),
            (std::vector<std::string>{"arange_store_vf_0_B0Y0_S0G0C0_VfNode_0", "store", "output"}));
  EXPECT_EQ(GetNodeNames(subgraphs[0]), (std::vector<std::string>{"arange", "Store_arange0", "Output_arange0"}));
}

TEST_F(TestOptimizer, V2HighRankArangeOptimizeAndCodegenOuterFor) {
  SetUpV2RuntimeStub();
  ASSERT_EQ(
      att::soc_perf_test::SetSocPerfTestPlatform(att::soc_perf_test::kV35AltNpuArch, att::soc_perf_test::kSocEmpty),
      af::SUCCESS);
  af::AscGraph graph("high_rank_arange_pipeline");
  std::vector<af::Expression> sizes;
  std::vector<af::AxisId> axes;
  for (size_t i = 0UL; i < 5UL; ++i) {
    sizes.push_back(graph.CreateSizeVar("size" + std::to_string(i)));
    axes.push_back(graph.CreateAxis("axis" + std::to_string(i), sizes.back()).id);
  }
  const std::vector<af::Expression> strides = {sizes[1] * sizes[2] * sizes[3] * sizes[4] * af::Symbol(5),
                                               sizes[2] * sizes[3] * sizes[4] * af::Symbol(4),
                                               sizes[3] * sizes[4] * af::Symbol(3), sizes[4] * af::Symbol(2), One};

  Arange arange("arange");
  arange.ir_attr.SetBase(af::Symbol(3));
  arange.ir_attr.SetStep(af::Symbol(2));
  arange.attr.sched.axis = axes;
  arange.attr.api.unit = af::ComputeUnit::kUnitVector;
  arange.y.dtype = af::DT_INT64;
  *arange.y.axis = axes;
  *arange.y.repeats = sizes;
  *arange.y.strides = strides;
  ASSERT_NE(graph.AddNode(arange), nullptr);

  Abs abs("abs");
  abs.x = arange.y;
  abs.attr.sched.axis = axes;
  abs.attr.api.unit = af::ComputeUnit::kUnitVector;
  abs.y.dtype = af::DT_INT64;
  *abs.y.axis = axes;
  *abs.y.repeats = sizes;
  *abs.y.strides = strides;

  Store store("store");
  store.x = abs.y;
  store.attr.sched.axis = axes;
  store.attr.api.unit = af::ComputeUnit::kUnitMTE3;
  store.y.dtype = af::DT_INT64;
  *store.y.axis = axes;
  *store.y.repeats = sizes;
  *store.y.strides = strides;

  Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult scheduled;
  ASSERT_EQ(optimizer.Optimize(graph, scheduled), af::SUCCESS);
  ASSERT_EQ(scheduled.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(scheduled.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(scheduled.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_FALSE(scheduled.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.empty());

  codegen::CodegenResult generated;
  const std::map<std::string, std::string> shape_info = {
      {"size0", "2"}, {"size1", "2"}, {"size2", "2"}, {"size3", "2"}, {"size4", "65"}};
  ASSERT_EQ(codegen::Codegen(codegen::CodegenOptions{}).Generate(shape_info, scheduled, generated), af::SUCCESS);
  EXPECT_NE(generated.kernel.find("outer_for_0"), std::string::npos) << generated.kernel;
  EXPECT_NE(generated.kernel.find("arange_offset_"), std::string::npos) << generated.kernel;
}

TEST_F(TestOptimizer, V2ArangeFloorDivOptimizeProducesDeterministicKernel) {
  SetUpV2RuntimeStub();
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  std::vector<std::string> expected_root_order;
  for (size_t run = 0UL; run < 3UL; ++run) {
    auto graph = CreateArangeConsumerGraph("arange_floor_div_vf", ArangeConsumer::kFloorDiv);
    ::ascir::FusedScheduledResult result;
    ASSERT_EQ(optimizer.Optimize(graph, result), af::SUCCESS);
    ASSERT_EQ(result.node_idx_to_scheduled_results.size(), 1UL);
    ASSERT_EQ(result.node_idx_to_scheduled_results[0].size(), 1UL);
    ASSERT_EQ(result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
    ASSERT_EQ(result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
    const auto &impl_graph = result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
    std::vector<AscGraph> subgraphs;
    ASSERT_EQ(impl_graph.GetAllSubGraphs(subgraphs), af::SUCCESS);
    ASSERT_EQ(subgraphs.size(), 2UL);
    const auto root_order = GetNodeNames(impl_graph);
    EXPECT_EQ(root_order,
              (std::vector<std::string>{"arange_floor_div_vf_0_B0Y0_S0G0C0_VfNode_0",
                                        "arange_floor_div_vf_0_B0Y0_S0G0C0_VfNode_1", "floor_div", "store", "output"}));
    if (run == 0UL) {
      expected_root_order = root_order;
    } else {
      EXPECT_EQ(root_order, expected_root_order);
    }
  }
}

TEST_F(TestOptimizer, V2Int64ArangeFloorDivKeepsInt64Composition) {
  SetUpV2RuntimeStub();
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  auto graph =
      CreateArangeConsumerGraph("arange_int64_floor_div", ArangeConsumer::kFloorDiv, af::ops::One, af::DT_INT64);
  ::ascir::FusedScheduledResult result;
  ASSERT_EQ(optimizer.Optimize(graph, result), af::SUCCESS);
  ASSERT_EQ(result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  const auto &impl_graph = result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  std::vector<AscGraph> subgraphs;
  ASSERT_EQ(impl_graph.GetAllSubGraphs(subgraphs), af::SUCCESS);
  ASSERT_EQ(subgraphs.size(), 2UL);
  EXPECT_EQ(GetNodeNames(impl_graph), (std::vector<std::string>{"arange_int64_floor_div_0_B0Y0_S0G0C0_VfNode_0",
                                                                "arange_int64_floor_div_0_B0Y0_S0G0C0_VfNode_1",
                                                                "floor_div", "store", "output"}));
  EXPECT_EQ(subgraphs[0].FindNode("arange")->outputs[0].attr.dtype, af::DT_INT64);
  EXPECT_EQ(subgraphs[1].FindNode("other_arange")->outputs[0].attr.dtype, af::DT_INT64);
  EXPECT_EQ(impl_graph.FindNode("floor_div")->outputs[0].attr.dtype, af::DT_INT64);
}

TEST_F(TestOptimizer, V2WindowMathOptimizeProducesOneDeterministicKernel) {
  SetUpV2RuntimeStub();
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  std::vector<std::string> expected_root_order;
  std::vector<std::vector<std::string>> expected_vf_orders;
  for (size_t run = 0UL; run < 3UL; ++run) {
    auto graph = CreateWindowMathGraph("window_math");
    ::ascir::FusedScheduledResult result;
    ASSERT_EQ(optimizer.Optimize(graph, result), af::SUCCESS);
    ASSERT_EQ(result.node_idx_to_scheduled_results.size(), 1UL);
    ASSERT_EQ(result.node_idx_to_scheduled_results[0].size(), 1UL);
    ASSERT_EQ(result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
    ASSERT_EQ(result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
    const auto &impl_graph = result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
    std::vector<AscGraph> subgraphs;
    ASSERT_EQ(impl_graph.GetAllSubGraphs(subgraphs), af::SUCCESS);
    ASSERT_EQ(subgraphs.size(), 3UL);
    const auto root_order = GetNodeNames(impl_graph);
    std::vector<std::vector<std::string>> vf_orders;
    for (const auto &subgraph : subgraphs) {
      vf_orders.push_back(GetNodeNames(subgraph));
    }
    if (run == 0UL) {
      expected_root_order = root_order;
      expected_vf_orders = vf_orders;
    } else {
      EXPECT_EQ(root_order, expected_root_order);
      EXPECT_EQ(vf_orders, expected_vf_orders);
    }
    EXPECT_EQ(root_order,
              (std::vector<std::string>{"window_math_0_B0Y0_S0G0C0_VfNode_0", "window_math_0_B0Y0_S0G0C0_VfNode_1",
                                        "floor_div", "window_math_0_B0Y0_S0G0C0_VfNode_2", "store", "output"}));
    EXPECT_EQ(vf_orders[0], (std::vector<std::string>{"arange", "Store_arange0", "Output_arange0"}));
    EXPECT_EQ(vf_orders[1], (std::vector<std::string>{"divisor", "Store_divisor0", "Output_divisor0"}));
    EXPECT_EQ(vf_orders[2], (std::vector<std::string>{"Data_floor_div0", "Data_window_math_0_B0Y0_S0G0C0_VfNode_01",
                                                      "Data_window_math_0_B0Y0_S0G0C0_VfNode_12", "Load_floor_div0",
                                                      "Load_window_math_0_B0Y0_S0G0C0_VfNode_01",
                                                      "Load_window_math_0_B0Y0_S0G0C0_VfNode_12", "minimum", "maximum",
                                                      "Store_maximum0", "Output_maximum0"}));
  }
}

TEST_F(TestOptimizer, V2MultiArangeOptimizeKeepsOrderEdgesAndParamsIsolated) {
  SetUpV2RuntimeStub();
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  const std::vector<std::string> root_golden = {"multi_arange_0_B0Y0_S0G0C0_VfNode_0", "store", "output"};
  const std::vector<std::string> edge_golden = {"arange:0->add:0", "other_arange:0->add:1"};
  std::vector<std::string> expected_root_order;
  std::vector<std::string> expected_vf_order;
  std::vector<std::string> expected_add_edges;
  for (size_t run = 0UL; run < 3UL; ++run) {
    auto graph = CreateArangeConsumerGraph("multi_arange", ArangeConsumer::kAdd, af::ops::One, af::DT_INT64,
                                           {af::Symbol(1), af::Symbol(2), af::Symbol(7), af::Symbol(4)});

    ::ascir::FusedScheduledResult result;
    ASSERT_EQ(optimizer.Optimize(graph, result), af::SUCCESS);
    const auto &impl_graph = result.node_idx_to_scheduled_results.at(0).at(0).schedule_groups.at(0).impl_graphs.at(0);
    std::vector<AscGraph> subgraphs;
    ASSERT_EQ(impl_graph.GetAllSubGraphs(subgraphs), af::SUCCESS);
    ASSERT_EQ(subgraphs.size(), 1UL);
    const auto root_order = GetNodeNames(impl_graph);
    const auto vf_order = GetNodeNames(subgraphs[0]);
    EXPECT_EQ(root_order, root_golden);
    EXPECT_EQ(vf_order, (std::vector<std::string>{"arange", "other_arange", "add", "Store_add0", "Output_add0"}));
    auto vf_arange = subgraphs[0].FindNode("arange");
    auto vf_other_arange = subgraphs[0].FindNode("other_arange");
    auto vf_add = subgraphs[0].FindNode("add");
    ASSERT_NE(vf_arange, nullptr);
    ASSERT_NE(vf_other_arange, nullptr);
    ASSERT_NE(vf_add, nullptr);
    af::Expression base;
    af::Expression step;
    ASSERT_EQ(vf_arange->attr.ir_attr->GetAttrValue("base", base), af::GRAPH_SUCCESS);
    ASSERT_EQ(vf_arange->attr.ir_attr->GetAttrValue("step", step), af::GRAPH_SUCCESS);
    EXPECT_EQ(base, af::Symbol(1));
    EXPECT_EQ(step, af::Symbol(2));
    ASSERT_EQ(vf_other_arange->attr.ir_attr->GetAttrValue("base", base), af::GRAPH_SUCCESS);
    ASSERT_EQ(vf_other_arange->attr.ir_attr->GetAttrValue("step", step), af::GRAPH_SUCCESS);
    EXPECT_EQ(base, af::Symbol(7));
    EXPECT_EQ(step, af::Symbol(4));
    ASSERT_EQ(vf_add->GetAllInDataAnchorsSize(), 2UL);
    ASSERT_NE(vf_add->GetInDataAnchor(0)->GetPeerOutAnchor(), nullptr);
    ASSERT_NE(vf_add->GetInDataAnchor(1)->GetPeerOutAnchor(), nullptr);
    EXPECT_EQ(vf_add->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "arange");
    EXPECT_EQ(vf_add->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "other_arange");
    const std::vector<std::string> add_edges = {
        vf_add->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName() + ":" +
            std::to_string(vf_add->GetInDataAnchor(0)->GetPeerOutAnchor()->GetIdx()) +
            "->add:" + std::to_string(vf_add->GetInDataAnchor(0)->GetIdx()),
        vf_add->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName() + ":" +
            std::to_string(vf_add->GetInDataAnchor(1)->GetPeerOutAnchor()->GetIdx()) +
            "->add:" + std::to_string(vf_add->GetInDataAnchor(1)->GetIdx())};
    EXPECT_EQ(add_edges, edge_golden);
    if (run == 0UL) {
      expected_root_order = root_order;
      expected_vf_order = vf_order;
      expected_add_edges = add_edges;
    } else {
      EXPECT_EQ(root_order, expected_root_order);
      EXPECT_EQ(vf_order, expected_vf_order);
      EXPECT_EQ(add_edges, expected_add_edges);
    }
  }
}

struct Int64ConsumerCodegenCase {
  ArangeConsumer consumer;
  const char *consumer_name;
  const char *expected_api;
  const char *expected_call;
};

class Int64ArangeConsumerCodegenTest : public TestOptimizer,
                                       public ::testing::WithParamInterface<Int64ConsumerCodegenCase> {};

TEST_P(Int64ArangeConsumerCodegenTest, GeneratesInt64ConsumerWithoutNarrowing) {
  SetUpV2RuntimeStub();
  ASSERT_EQ(
      att::soc_perf_test::SetSocPerfTestPlatform(att::soc_perf_test::kV35AltNpuArch, att::soc_perf_test::kSocEmpty),
      af::SUCCESS);
  const auto &test_case = GetParam();
  auto graph = CreateArangeConsumerGraph("int64_consumer", test_case.consumer, af::ops::One, af::DT_INT64);
  ::ascir::FusedScheduledResult scheduled;
  ASSERT_EQ(optimizer.Optimize(graph, scheduled), af::SUCCESS);
  const auto &impl_graph = scheduled.node_idx_to_scheduled_results.at(0).at(0).schedule_groups.at(0).impl_graphs.at(0);
  std::vector<AscGraph> subgraphs;
  ASSERT_EQ(impl_graph.GetAllSubGraphs(subgraphs), af::SUCCESS);
  ASSERT_EQ(subgraphs.size(), 1UL);
  const auto consumer = subgraphs[0].FindNode(test_case.consumer_name);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->inputs.Size(), 2UL);
  ASSERT_EQ(consumer->outputs().size(), 1UL);
  EXPECT_EQ(consumer->inputs[0].attr.dtype, af::DT_INT64);
  EXPECT_EQ(consumer->inputs[1].attr.dtype, af::DT_INT64);
  EXPECT_EQ(consumer->outputs[0].attr.dtype, af::DT_INT64);
  codegen::CodegenResult generated;
  ASSERT_EQ(codegen::Codegen(codegen::CodegenOptions{}).Generate({{"size", "stub_size"}}, scheduled, generated),
            af::SUCCESS);
  const auto consumer_call = GetLineContaining(generated.kernel, test_case.expected_api);
  ASSERT_FALSE(consumer_call.empty());
  EXPECT_EQ(consumer_call, test_case.expected_call);
  if (test_case.consumer == ArangeConsumer::kAdd) {
    const auto vf_signature = GetLineContaining(generated.kernel, "int64_t arange_offset_0");
    ASSERT_FALSE(vf_signature.empty());
    EXPECT_EQ(vf_signature,
              "inline __simd_vf__ void VFCallint64_consumer_0_B0Y0_S0G0C0_VfNode_0(__local_mem__ int64_t "
              "*local_1_addr, uint32_t output_dims_0, uint32_t local_1_stride_0, int64_t arange_offset_0, "
              "int64_t arange_base_0, int64_t arange_step_0, int64_t arange_stride_0_0, int64_t arange_stride_0_1, "
              "int64_t arange_stride_0_2, int64_t arange_offset_1, int64_t arange_base_1, int64_t arange_step_1, "
              "int64_t arange_stride_1_0, int64_t arange_stride_1_1, int64_t arange_stride_1_2)");
  }
}

INSTANTIATE_TEST_SUITE_P(
    Int64Consumers, Int64ArangeConsumerCodegenTest,
    ::testing::Values(Int64ConsumerCodegenCase{ArangeConsumer::kAdd, "add", "AscendC::MicroAPI::Add",
                                               "    AscendC::MicroAPI::Add(vreg_2, vreg_0, vreg_1, preg_0);"},
                      Int64ConsumerCodegenCase{ArangeConsumer::kSub, "sub", "AscendC::MicroAPI::Sub",
                                               "    AscendC::MicroAPI::Sub(vreg_2, vreg_0, vreg_1, preg_0);"},
                      Int64ConsumerCodegenCase{ArangeConsumer::kMul, "mul", "AscendC::MicroAPI::Mul",
                                               "    AscendC::MicroAPI::Mul(vreg_2, vreg_0, vreg_1, preg_0);"},
                      Int64ConsumerCodegenCase{ArangeConsumer::kMinimum, "minimum", "AscendC::MicroAPI::Min",
                                               "    AscendC::MicroAPI::Min(vreg_2, vreg_0, vreg_1, preg_0);"},
                      Int64ConsumerCodegenCase{ArangeConsumer::kMaximum, "maximum", "AscendC::MicroAPI::Max",
                                               "    AscendC::MicroAPI::Max(vreg_2, vreg_0, vreg_1, preg_0);"}));

TEST_F(TestOptimizer, V2ScalarVectorFuncPreservesRootScalarThroughCodegen) {
  SetUpV2RuntimeStub();
  ASSERT_EQ(
      att::soc_perf_test::SetSocPerfTestPlatform(att::soc_perf_test::kV35AltNpuArch, att::soc_perf_test::kSocEmpty),
      af::SUCCESS);
  af::AscGraph graph("scalar_vf_regression");
  const auto size = graph.CreateSizeVar("size");
  const auto axis = graph.CreateAxis("axis", size);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = af::DT_FLOAT;

  Load load("load");
  load.x = data.y;
  load.attr.sched.axis = {axis.id};
  load.attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load.y.dtype = af::DT_FLOAT;
  *load.y.axis = {axis.id};
  *load.y.repeats = {size};
  *load.y.strides = {One};
  *load.y.vectorized_axis = {axis.id};
  *load.y.vectorized_strides = {One};

  Scalar scalar("scalar", graph);
  scalar.ir_attr.SetValue("2.0");
  scalar.y.dtype = af::DT_FLOAT;

  Add add("add");
  add.x1 = load.y;
  add.x2 = scalar.y;
  add.attr.sched.axis = {axis.id};
  add.attr.api.unit = af::ComputeUnit::kUnitVector;
  add.y.dtype = af::DT_FLOAT;
  *add.y.axis = {axis.id};
  *add.y.repeats = {size};
  *add.y.strides = {One};
  *add.y.vectorized_axis = {axis.id};
  *add.y.vectorized_strides = {One};

  Abs abs("abs");
  abs.x = add.y;
  abs.attr.sched.axis = {axis.id};
  abs.attr.api.unit = af::ComputeUnit::kUnitVector;
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {axis.id};
  *abs.y.repeats = {size};
  *abs.y.strides = {One};
  *abs.y.vectorized_axis = {axis.id};
  *abs.y.vectorized_strides = {One};

  Store store("store");
  store.x = abs.y;
  store.attr.sched.axis = {axis.id};
  store.attr.api.unit = af::ComputeUnit::kUnitMTE3;
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {axis.id};
  *store.y.repeats = {size};
  *store.y.strides = {One};
  *store.y.vectorized_axis = {axis.id};
  *store.y.vectorized_strides = {One};

  Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult scheduled;
  ASSERT_EQ(optimizer.Optimize(graph, scheduled), af::SUCCESS);
  ASSERT_EQ(scheduled.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(scheduled.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(scheduled.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(scheduled.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  const auto &impl_graph = scheduled.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];

  const auto root_scalar = impl_graph.FindNode("scalar");
  ASSERT_NE(root_scalar, nullptr);
  ASSERT_NE(root_scalar->outputs[0].attr.mem.tensor_id, af::kIdNone);
  EXPECT_NE(root_scalar->outputs[0].attr.mem.alloc_type, af::AllocType::kAllocTypeBuffer);

  af::AscNodePtr vf_node;
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node->GetType() == VectorFunc::Type) {
      vf_node = node;
      break;
    }
  }
  ASSERT_NE(vf_node, nullptr);
  std::vector<AscGraph> subgraphs;
  ASSERT_EQ(impl_graph.GetAllSubGraphs(subgraphs), af::SUCCESS);
  ASSERT_EQ(subgraphs.size(), 1UL);
  const auto sub_scalar = subgraphs[0].FindNode("Scalar_scalar");
  ASSERT_NE(sub_scalar, nullptr);
  EXPECT_EQ(sub_scalar->outputs[0].attr.mem.tensor_id, root_scalar->outputs[0].attr.mem.tensor_id);

  codegen::CodegenResult generated;
  ASSERT_EQ(codegen::Codegen(codegen::CodegenOptions{}).Generate({{"size", "stub_size"}}, scheduled, generated),
            af::SUCCESS);
  const std::string scalar_name = "scalar_" + std::to_string(root_scalar->outputs[0].attr.mem.tensor_id);
  const std::string function_name = "VFCall" + vf_node->GetName();
  const auto signature_offset = generated.kernel.find("inline __simd_vf__ void " + function_name + "(");
  ASSERT_NE(signature_offset, std::string::npos);
  const auto signature_end = generated.kernel.find('\n', signature_offset);
  const auto signature = generated.kernel.substr(signature_offset, signature_end - signature_offset);
  const std::string scalar_param = "float " + scalar_name;
  const auto scalar_param_offset = signature.find(scalar_param);
  ASSERT_NE(scalar_param_offset, std::string::npos) << signature;
  EXPECT_EQ(signature.find(scalar_param, scalar_param_offset + scalar_param.size()), std::string::npos) << signature;

  const auto call_offset = generated.kernel.find(function_name + "(", signature_end);
  ASSERT_NE(call_offset, std::string::npos);
  const auto call_end = generated.kernel.find('\n', call_offset);
  const auto call = generated.kernel.substr(call_offset, call_end - call_offset);
  EXPECT_NE(call.find(scalar_name), std::string::npos) << call;
}

TEST_F(TestOptimizer, ReOrderMergeAxisGraph_scheduler) {
  // z0, z1, z2 mergeaxis
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data_i("data_i", graph);
  data_i.attr.sched.axis = {z0.id, z1.id};
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id};
  data_i.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data_i.ir_attr.SetIndex(0);

  af::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id};
  *load_i.y.axis = {z0.id, z1.id};
  *load_i.y.repeats = {s0, s1};
  *load_i.y.strides = {s1, One};
  load_i.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load_i.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, One};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Store store("store");
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.ir_attr.SetIndex(0);

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);

  auto optimize_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto abs_sched = optimize_graph.FindNode("abs");
  std::stringstream ss;
  SizeExprListStr(ss, optimize_graph, abs_sched->outputs[0].attr.repeats);
  std::stringstream ss1;
  SizeExprListStr(ss1, optimize_graph, abs_sched->outputs[0].attr.strides);
  std::stringstream ss2;
  AxisListStr(ss2, optimize_graph, abs_sched->outputs[0].attr.axis);

  EXPECT_EQ(ss.str(), "(s0 * s1 / (z0z1Tb_size * z0z1t_size)), z0z1Tb_size, z0z1t_size, ");
  EXPECT_EQ(ss1.str(), "(z0z1Tb_size * z0z1t_size), z0z1t_size, 1, ");
  EXPECT_EQ(ss2.str(), "z0z1TB, z0z1Tb, z0z1t, ");
}

TEST_F(TestOptimizer, BufQueAlloc_WhenOutputNode_WillUseInputTensorAsOutput) {
  // In case: Load -> Vec -> Store -> workspace -> Load2
  // here, load2 will use Store output directly,
  // Output node's output need pass it's input.
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Data x_op("x", graph);
  x_op.y.dtype = ge::DT_FLOAT16;

  Load load_op("load");
  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;
  *load_op.y.axis = {z0.id};

  af::ascir_op::Abs vec_op("vec");
  vec_op.x = load_op.y;
  vec_op.y.dtype = ge::DT_FLOAT16;
  *vec_op.y.axis = {z0.id};

  Store store_op("store");
  store_op.x = vec_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id};

  Workspace y_op("y");
  y_op.x = store_op.y;

  Load load_op2("load2");
  load_op2.x = y_op.y;

  // graph.SetInputs({x_op});
  // graph.SetOutputs({y_op});

  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  optimizer.BufQueAlloc(graph, graph);

  EXPECT_EQ(y->outputs[0].attr.mem.tensor_id, store->outputs[0].attr.mem.tensor_id);
}

TEST_F(TestOptimizer, BufQueAlloc_TempBuffer) {
  // In case: Load -> Vec -> Store -> workspace -> Load2
  // here, load2 will use Store output directly,
  // Output node's output need pass it's input.
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Data x_op("x", graph);
  x_op.y.dtype = ge::DT_FLOAT16;
  x_op.ir_attr.SetIndex(0);

  Load load_op("load");
  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;
  *load_op.y.axis = {z0.id};

  af::ascir_op::Broadcast brc_op("brc");
  brc_op.x = load_op.y;
  brc_op.y.dtype = ge::DT_FLOAT16;
  *brc_op.y.axis = {z0.id};

  Store store_op("store");
  store_op.x = brc_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id};

  Workspace y_op("y");
  y_op.x = store_op.y;

  Load load_op2("load2");
  load_op2.x = y_op.y;

  Status status = optimizer.BufQueAlloc(graph, graph);
  ASSERT_EQ(status, af::SUCCESS);
  auto brc = graph.FindNode("brc");
  ASSERT_NE(brc, nullptr);
  ASSERT_EQ(brc->attr.tmp_buffers.size(), 1);
}

TEST_F(TestOptimizer, ConstantToStoreNeedBroadCast) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar(128);
  auto z0 = graph.CreateAxis("z0", s0);

  Scalar const_op("const", graph);
  Store store_op("store");
  Output output_op("output");

  const_op.attr.sched.axis = {z0.id};
  const_op.ir_attr.SetValue("998.998f");
  const_op.y.dtype = ge::DT_FLOAT;
  *const_op.y.strides = {af::ops::One};
  *const_op.y.repeats = {s0};

  store_op.attr.sched.axis = {z0.id};
  store_op.x = const_op.y;
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id};
  *store_op.y.strides = {af::ops::One};
  *store_op.y.repeats = {s0};

  output_op.attr.sched.axis = {z0.id};
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  auto optimize_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];

  auto cg = af::AscGraphUtils::GetComputeGraph(optimize_graph);
  auto found_broadcast = cg->FindFirstNodeMatchType(af::ascir_op::Broadcast::Type);
  ASSERT_NE(found_broadcast, nullptr);
  auto asc_broadcast = af::AscNode(found_broadcast->GetOpDesc(), nullptr);

  auto found_store = cg->FindFirstNodeMatchType(af::ascir_op::Store::Type);
  ASSERT_NE(found_store, nullptr);
  auto asc_store = af::AscNode(found_store->GetOpDesc(), nullptr);
}

TEST_F(TestOptimizer, ScalarConstantToStore) {
  af::AscGraph graph("scalar_const_graph");
  Scalar const_op("const", graph);
  Store store_op("store");
  Output output_op("output");

  const_op.ir_attr.SetValue("998.998f");
  const_op.y.dtype = ge::DT_FLOAT;
  const_op.attr.api.compute_type = af::ComputeType::kComputeInvalid;

  store_op.x = const_op.y;
  store_op.y.dtype = ge::DT_FLOAT;
  store_op.attr.api.compute_type = af::ComputeType::kComputeElewise;

  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  output_op.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ASSERT_EQ(optimizer.GraphPass(graph), af::SUCCESS);
  auto cg = af::AscGraphUtils::GetComputeGraph(graph);
  auto found_broadcast = cg->FindFirstNodeMatchType(af::ascir_op::Broadcast::Type);
  ASSERT_NE(found_broadcast, nullptr);
  auto asc_broadcast = af::AscNode(found_broadcast->GetOpDesc(), cg);
  EXPECT_TRUE(!asc_broadcast.attr.sched.axis.empty());

  auto found_store = cg->FindFirstNodeMatchType(af::ascir_op::Store::Type);
  ASSERT_NE(found_store, nullptr);
  auto asc_store = af::AscNode(found_store->GetOpDesc(), cg);
  EXPECT_TRUE(!asc_store.attr.sched.axis.empty());
}

TEST_F(TestOptimizer, SplitMultiOutputsData) {
  af::AscGraph graph("multi_outputs");

  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(32);
  auto s2 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.ir_attr.SetIndex(11);

  Load a("a");
  a.attr.sched.axis = {z0.id, z1.id};
  a.x = data1.y;
  *a.y.axis = {z0.id, z1.id};
  a.y.dtype = ge::DT_FLOAT;
  *a.y.strides = {s1, af::ops::One};
  *a.y.repeats = {s0, s1};

  Load d("d");
  d.attr.sched.axis = {z0.id, z1.id};
  d.x = data1.y;
  *d.y.axis = {z0.id, z1.id};
  d.y.dtype = ge::DT_FLOAT;
  *d.y.strides = {s2, af::ops::One};
  *d.y.repeats = {s0, s2};

  Concat c("c");
  c.attr.sched.axis = {z0.id, z1.id};
  c.attr.api.compute_type = af::ComputeType::kComputeConcat;
  c.x = {a.y, d.y};
  *c.y.axis = {z0.id, z1.id};
  c.y.dtype = ge::DT_FLOAT;
  *c.y.strides = {s1 + s2, af::ops::One};
  *c.y.repeats = {s0, s1 + s2};

  Store f("f");
  f.attr.sched.axis = {z0.id, z1.id};
  f.x = c.y;
  *f.y.axis = {z0.id, z1.id};
  f.y.dtype = ge::DT_FLOAT;
  *f.y.strides = {s1, af::ops::One};
  *f.y.repeats = {s0, s1};

  Output output3("output3");
  output3.x = f.y;
  output3.y.dtype = ge::DT_FLOAT;
  output3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output3.ir_attr.SetIndex(2);
  EXPECT_EQ(optimizer.GraphPass(graph), 0);

  auto load0_node = graph.FindNode("a");
  auto load1_node = graph.FindNode("d");
  ASSERT_NE(load0_node, nullptr);
  ASSERT_NE(load1_node, nullptr);
  auto data0_node = dynamic_cast<af::AscNode *>(af::ascir::AscTensorUtils::GetOwner(load0_node->inputs[0]));
  auto data1_node = dynamic_cast<af::AscNode *>(af::ascir::AscTensorUtils::GetOwner(load1_node->inputs[0]));
  ASSERT_NE(data0_node, nullptr);
  ASSERT_NE(data1_node, nullptr);
  auto ir_attr0 = data0_node->attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ASSERT_NE(ir_attr0, nullptr);
  auto ir_attr1 = data1_node->attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ASSERT_NE(ir_attr1, nullptr);
  int64_t idx0;
  int64_t idx1;
  ir_attr0->GetIndex(idx0);
  ir_attr1->GetIndex(idx1);
  EXPECT_EQ(idx0, 11);
  EXPECT_EQ(idx1, 11);
}

TEST_F(TestOptimizer, TestSplitConcatAxisPass) {
  af::AscGraph graph("concat_axis_graph");

  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(32);
  auto s2 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.ir_attr.SetIndex(0);

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data2.ir_attr.SetIndex(1);

  Load a("a");
  a.attr.sched.axis = {z0.id, z1.id};
  a.x = data1.y;
  *a.y.axis = {z0.id, z1.id};
  a.y.dtype = ge::DT_FLOAT;
  *a.y.strides = {s1, af::ops::One};
  *a.y.repeats = {s0, s1};

  Exp b("b");
  b.attr.sched.axis = {z0.id, z1.id};
  b.x = a.y;
  *b.y.axis = {z0.id, z1.id};
  b.y.dtype = ge::DT_FLOAT;
  *b.y.strides = {s1, af::ops::One};
  *b.y.repeats = {s0, s1};

  Load d("d");
  d.attr.sched.axis = {z0.id, z1.id};
  d.x = data2.y;
  *d.y.axis = {z0.id, z1.id};
  d.y.dtype = ge::DT_FLOAT;
  *d.y.strides = {s2, af::ops::One};
  *d.y.repeats = {s0, s2};

  Concat c("c");
  c.attr.sched.axis = {z0.id, z1.id};
  c.attr.api.compute_type = af::ComputeType::kComputeConcat;

  c.x = {b.y, d.y};

  *c.y.axis = {z0.id, z1.id};
  c.y.dtype = ge::DT_FLOAT;
  *c.y.strides = {s1 + s2, af::ops::One};
  *c.y.repeats = {s0, s1 + s2};

  Exp e("e");
  e.attr.sched.axis = {z0.id, z1.id};
  e.x = b.y;
  *e.y.axis = {z0.id, z1.id};
  e.y.dtype = ge::DT_FLOAT;
  *e.y.strides = {s1, af::ops::One};
  *e.y.repeats = {s0, s1};

  Store f("f");
  f.attr.sched.axis = {z0.id, z1.id};
  f.x = e.y;
  *f.y.axis = {z0.id, z1.id};
  f.y.dtype = ge::DT_FLOAT;
  *f.y.strides = {s1, af::ops::One};
  *f.y.repeats = {s0, s1};

  Store i("i");
  i.attr.sched.axis = {z0.id, z1.id};
  i.x = d.y;
  *i.y.axis = {z0.id, z1.id};
  i.y.dtype = ge::DT_FLOAT;
  *i.y.strides = {s2, af::ops::One};
  *i.y.repeats = {s0, s2};

  Exp g("g");
  g.attr.sched.axis = {z0.id, z1.id};
  g.x = c.y;
  *g.y.axis = {z0.id, z1.id};
  g.y.dtype = ge::DT_FLOAT;
  *g.y.strides = {s1 + s2, af::ops::One};
  *g.y.repeats = {s0, s1 + s2};

  Store h("h");
  h.attr.sched.axis = {z0.id, z1.id};
  h.x = g.y;
  *h.y.axis = {z0.id, z1.id};
  h.y.dtype = ge::DT_FLOAT;
  *h.y.strides = {s1 + s2, af::ops::One};
  *h.y.repeats = {s0, s1 + s2};

  Output output1("output1");
  output1.x = f.y;
  output1.y.dtype = ge::DT_FLOAT;
  output1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output1.ir_attr.SetIndex(0);

  Output output2("output2");
  output2.x = h.y;
  output2.y.dtype = ge::DT_FLOAT;
  output2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output2.ir_attr.SetIndex(1);

  Output output3("output3");
  output3.x = i.y;
  output3.y.dtype = ge::DT_FLOAT;
  output3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output3.ir_attr.SetIndex(2);
  ::ascir::utils::DumpGraph(graph, "Before_");
  EXPECT_EQ(optimizer.GraphPass(graph), 0);
  ::ascir::utils::DumpGraph(graph, "After_");

  auto a_node = graph.FindNode("a");
  auto c_node = graph.FindNode("c");
  auto f_node = graph.FindNode("f");
  auto h_node = graph.FindNode("h");
  auto i_node = graph.FindNode("i");

  auto size_c_out = c_node->attr.sched.axis;
  auto size_c_in0 = c_node->inputs[0].attr.axis;
  auto size_c_in1 = c_node->inputs[1].attr.axis;

  EXPECT_EQ(a_node->attr.sched.axis, size_c_in0);
  EXPECT_EQ(a_node->outputs[0].attr.axis, size_c_in0);
  EXPECT_EQ(f_node->attr.sched.axis, size_c_in0);
  EXPECT_EQ(f_node->outputs[0].attr.axis, size_c_in0);

  EXPECT_EQ(h_node->attr.sched.axis, size_c_out);
  EXPECT_EQ(h_node->outputs[0].attr.axis, size_c_out);

  EXPECT_EQ(i_node->attr.sched.axis, size_c_in1);
  EXPECT_EQ(i_node->outputs[0].attr.axis, size_c_in1);
}

TEST_F(TestOptimizer, ConcatFirstDim) {
  af::AscGraph graph("concat_1st_dim_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s0 + s1);

  Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);
  Data x3_op("x3", graph);
  x3_op.ir_attr.SetIndex(2);

  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");

  std::vector<Data> all_data{x1_op, x2_op, x3_op};
  std::vector<Load> all_load{load_op1, load_op2, load_op3};

  for (size_t i = 0U; i < all_data.size(); ++i) {
    auto &x_op = all_data[i];
    auto &load_op = all_load[i];
    x_op.y.dtype = ge::DT_FLOAT16;
    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z3.id, z2.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.axis = {z3.id, z2.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.strides = {s2, af::ops::One};
    *load_op.y.repeats = {s0, s2};
  }
  load_op3.attr.sched.axis = {z3.id, z2.id};
  *load_op3.y.axis = {z3.id, z2.id};
  *load_op3.y.repeats = {s1, s2};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z3.id, z2.id};
  add_op.x1 = load_op1.y;
  add_op.x2 = load_op2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z3.id, z2.id};
  *add_op.y.strides = {s2, af::ops::One};
  *add_op.y.repeats = {s0, s2};

  af::ascir_op::Abs abs_op("abs");
  abs_op.attr.sched.axis = {z3.id, z2.id};
  abs_op.x = load_op3.y;
  abs_op.y.dtype = ge::DT_FLOAT16;
  *abs_op.y.axis = {z3.id, z2.id};
  *abs_op.y.strides = {s2, af::ops::One};
  *abs_op.y.repeats = {s1, s2};

  af::ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z3.id, z2.id};
  concat_op.x = {add_op.y, abs_op.y};
  concat_op.y.dtype = ge::DT_FLOAT16;
  *concat_op.y.axis = {z3.id, z2.id};
  *concat_op.y.repeats = {s0 + s1, s2};
  *concat_op.y.strides = {s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z3.id, z2.id};

  store_op.x = concat_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z3.id, z2.id};
  *store_op.y.repeats = {s0 + s1, s2};
  *store_op.y.strides = {s2, af::ops::One};

  Output y_op("y");
  y_op.x = store_op.y;
  y_op.ir_attr.SetIndex(0);
  auto store_node = graph.FindNode("store");
  setenv("AUTOFUSE_DFX_FLAGS", "codegen_compile_debug=true;debug_dir=./TestDump", 1);
  ::ascir::utils::ResetDumpConfig();
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1);
  auto &schedule_result = fused_scheduled_result.node_idx_to_scheduled_results[0][0];
  std::vector<af::Expression> offsets;
  std::vector<af::Expression> expect = {af::Symbol(0), (s0 * s2), af::Symbol(0), (s0 * s2)};
  for (const auto &schedule_group : schedule_result.schedule_groups) {
    for (auto &sub_impl_graph : schedule_group.impl_graphs) {
      for (const auto &sub_node : sub_impl_graph.GetAllNodes()) {
        if (sub_node->GetType() == "Store") {
          af::Expression offset;
          EXPECT_EQ(sub_node->attr.ir_attr->GetAttrValue("offset", offset), 0);
          offsets.emplace_back(offset);
        }
      }
    }
  }
  unsetenv("AUTOFUSE_DFX_FLAGS");
  ::ascir::utils::ResetDumpConfig();
  for (size_t i = 0; i < offsets.size(); ++i) {
    EXPECT_SYMBOL_EQ(offsets[i], expect[i]);
  }
  EXPECT_EQ(fused_scheduled_result.input_nodes.size(), 3);
  EXPECT_EQ(fused_scheduled_result.output_nodes.size(), 1);
  EXPECT_EQ(fused_scheduled_result.workspace_nodes.size(), 0);
  EXPECT_EQ(fused_scheduled_result.input_nodes[0]->GetName(), "x1");
  EXPECT_EQ(fused_scheduled_result.input_nodes[1]->GetName(), "x2");
  EXPECT_EQ(fused_scheduled_result.input_nodes[2]->GetName(), "x3");
  EXPECT_EQ(fused_scheduled_result.output_nodes[0]->GetName(), "y");

  std::set<std::set<std::string>> axis_name_groups;
  for (const auto &schedule_group : schedule_result.schedule_groups) {
    std::set<std::string> axis_names;
    for (const auto &axis : schedule_group.impl_graphs[0].GetAllAxis()) {
      axis_names.emplace(axis->name);
    }
    axis_name_groups.emplace(std::move(axis_names));
  }

  const std::set<std::set<std::string>> expected_axis_name_groups = {
      {"z3z2_0", "z3z2_0T", "z3z2_0TB", "z3z2_0Tb", "z3z2_0t"},
      {"z3z2_1", "z3z2_1T", "z3z2_1TB", "z3z2_1Tb", "z3z2_1t"},
  };
  EXPECT_EQ(axis_name_groups, expected_axis_name_groups);
}

TEST_F(TestOptimizer, ConcatTailDim) {
  af::AscGraph graph("concat_last_dim_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1 + s2);

  Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);
  Data x3_op("x3", graph);
  x3_op.ir_attr.SetIndex(2);

  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");

  std::vector<Data> all_data{x1_op, x2_op, x3_op};
  std::vector<Load> all_load{load_op1, load_op2, load_op3};

  for (size_t i = 0U; i < all_data.size(); ++i) {
    auto &x_op = all_data[i];
    auto &load_op = all_load[i];
    x_op.y.dtype = ge::DT_FLOAT16;
    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.axis = {z0.id, z1.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.repeats = {s0, s1};
    *load_op.y.strides = {s1, af::ops::One};
  }
  load_op3.attr.sched.axis = {z0.id, z1.id};
  *load_op3.y.axis = {z0.id, z1.id};
  *load_op3.y.repeats = {s0, s2};
  *load_op3.y.strides = {s2, af::ops::One};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = load_op1.y;
  add_op.x2 = load_op2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, af::ops::One};
  *add_op.y.repeats = {s0, s1};

  af::ascir_op::Abs abs_op("abs");
  abs_op.attr.sched.axis = {z0.id, z1.id};
  abs_op.x = load_op3.y;
  abs_op.y.dtype = ge::DT_FLOAT16;
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.strides = {s2, af::ops::One};
  *abs_op.y.repeats = {s0, s2};

  af::ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {add_op.y, abs_op.y};
  concat_op.y.dtype = ge::DT_FLOAT16;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1 + s2};
  *concat_op.y.strides = {s1 + s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};

  store_op.x = concat_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1 + s2};
  *store_op.y.strides = {s1 + s2, af::ops::One};
  store_op.ir_attr.SetOffset(af::Symbol(0));

  Output y_op("y");
  y_op.x = store_op.y;
  y_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(res, 0);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3);
  auto &schedule_result = fused_scheduled_result.node_idx_to_scheduled_results[0][1];

  std::vector<af::Expression> offsets;
  std::vector<af::Expression> expect = {af::Symbol(0), s1};
  for (const auto &schedule_group : schedule_result.schedule_groups) {
    auto &sub_impl_graph = schedule_group.impl_graphs.front();
    for (const auto &sub_node : sub_impl_graph.GetAllNodes()) {
      if (sub_node->GetType() == "Store") {
        af::Expression offset;
        EXPECT_EQ(sub_node->attr.ir_attr->GetAttrValue("offset", offset), 0);
        offsets.emplace_back(offset);
      }
    }
  }
  for (size_t i = 0; i < offsets.size(); ++i) {
    EXPECT_SYMBOL_EQ(offsets[i], expect[i]);
  }
  EXPECT_EQ(fused_scheduled_result.input_nodes.size(), 3);
  EXPECT_EQ(fused_scheduled_result.output_nodes.size(), 1);
  EXPECT_EQ(fused_scheduled_result.workspace_nodes.size(), 0);
  EXPECT_EQ(fused_scheduled_result.input_nodes[0]->GetName(), "x1");
  EXPECT_EQ(fused_scheduled_result.input_nodes[1]->GetName(), "x2");
  EXPECT_EQ(fused_scheduled_result.input_nodes[2]->GetName(), "x3");
  EXPECT_EQ(fused_scheduled_result.output_nodes[0]->GetName(), "y");
}

TEST_F(TestOptimizer, ConcatTailDim_OutputOrderReversed) {
  af::AscGraph graph("concat_last_dim_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1 + s2);

  Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);
  Data x3_op("x3", graph);
  x3_op.ir_attr.SetIndex(2);

  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");

  std::vector<Data> all_data{x1_op, x2_op, x3_op};
  std::vector<Load> all_load{load_op1, load_op2, load_op3};

  for (size_t i = 0U; i < all_data.size(); ++i) {
    auto &x_op = all_data[i];
    auto &load_op = all_load[i];
    x_op.y.dtype = ge::DT_FLOAT16;
    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.axis = {z0.id, z1.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.repeats = {s0, s1};
    *load_op.y.strides = {s1, af::ops::One};
  }
  load_op3.attr.sched.axis = {z0.id, z1.id};
  *load_op3.y.axis = {z0.id, z1.id};
  *load_op3.y.repeats = {s0, s2};
  *load_op3.y.strides = {s2, af::ops::One};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = load_op1.y;
  add_op.x2 = load_op2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, af::ops::One};
  *add_op.y.repeats = {s0, s1};

  af::ascir_op::Abs abs_op("abs");
  abs_op.attr.sched.axis = {z0.id, z1.id};
  abs_op.x = load_op3.y;
  abs_op.y.dtype = ge::DT_FLOAT16;
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.strides = {s2, af::ops::One};
  *abs_op.y.repeats = {s0, s2};

  af::ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {abs_op.y, add_op.y};
  concat_op.y.dtype = ge::DT_FLOAT16;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1 + s2};
  *concat_op.y.strides = {s1 + s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};

  store_op.x = concat_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1 + s2};
  *store_op.y.strides = {s1 + s2, af::ops::One};
  store_op.ir_attr.SetOffset(af::Symbol(0));

  Output y_op("y");
  y_op.x = store_op.y;
  y_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3);
  auto &schedule_result = fused_scheduled_result.node_idx_to_scheduled_results[0][1];

  std::vector<af::Expression> offsets;
  std::vector<af::Expression> expect = {s2, af::Symbol(0)};
  for (const auto &schedule_group : schedule_result.schedule_groups) {
    auto &sub_impl_graph = schedule_group.impl_graphs.front();
    for (const auto &sub_node : sub_impl_graph.GetAllNodes()) {
      if (sub_node->GetType() == "Store") {
        af::Expression offset;
        EXPECT_EQ(sub_node->attr.ir_attr->GetAttrValue("offset", offset), 0);
        offsets.emplace_back(offset);
      }
    }
  }
  for (size_t i = 0; i < offsets.size(); ++i) {
    EXPECT_SYMBOL_EQ(offsets[i], expect[i]);
  }
  EXPECT_EQ(fused_scheduled_result.input_nodes.size(), 3);
  EXPECT_EQ(fused_scheduled_result.output_nodes.size(), 1);
  EXPECT_EQ(fused_scheduled_result.workspace_nodes.size(), 0);
  EXPECT_EQ(fused_scheduled_result.input_nodes[0]->GetName(), "x1");
  EXPECT_EQ(fused_scheduled_result.input_nodes[1]->GetName(), "x2");
  EXPECT_EQ(fused_scheduled_result.input_nodes[2]->GetName(), "x3");
  EXPECT_EQ(fused_scheduled_result.output_nodes[0]->GetName(), "y");
}

TEST_F(TestOptimizer, ConcatTailDim_sharing_input) {
  af::AscGraph graph("concat_last_dim_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto tmp = graph.CreateAxis("tmp", s0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto tmp1 = graph.CreateAxis("tmp1", s0);
  auto z1 = graph.CreateAxis("z1", s1 + s1);
  auto tmp2 = graph.CreateAxis("tmp2", s0);

  Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);

  Load load_op1("load1");
  Load load_op2("load2");

  std::vector<Data> all_data{x1_op, x2_op};
  std::vector<Load> all_load{load_op1, load_op2};

  for (size_t i = 0U; i < all_data.size(); ++i) {
    auto &x_op = all_data[i];
    auto &load_op = all_load[i];
    x_op.y.dtype = ge::DT_FLOAT16;
    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.axis = {z0.id, z1.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.repeats = {s0, s1};
    *load_op.y.strides = {s1, af::ops::One};
  }

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = load_op1.y;
  add_op.x2 = load_op2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, af::ops::One};
  *add_op.y.repeats = {s0, s1};

  af::ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {add_op.y, add_op.y};
  concat_op.y.dtype = ge::DT_FLOAT16;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1 + s1};
  *concat_op.y.strides = {s1 + s1, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};

  store_op.x = concat_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1 + s1};
  *store_op.y.strides = {s1 + s1, af::ops::One};
  store_op.ir_attr.SetOffset(af::Symbol(0));

  Output y_op("y");
  y_op.x = store_op.y;
  y_op.ir_attr.SetIndex(0);

  auto axis = graph.GetAllAxis();
  axis.erase(axis.begin() + 4);
  axis.erase(axis.begin() + 2);
  axis.erase(axis.begin());
  const auto graph_attr = af::AscGraphUtils::GetComputeGraph(graph)->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  graph_attr->axis = axis;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3);
  auto &schedule_result = fused_scheduled_result.node_idx_to_scheduled_results[0][1];

  std::vector<af::Expression> offsets;
  std::vector<af::Expression> expect = {af::Symbol(0), s1};
  for (const auto &schedule_group : schedule_result.schedule_groups) {
    auto &sub_impl_graph = schedule_group.impl_graphs.front();
    auto res_axis = sub_impl_graph.GetAllAxis();
    for (size_t i = 0; i < res_axis.size(); i++) {
      EXPECT_EQ(res_axis[i]->id, i);
    }
    for (const auto &sub_node : sub_impl_graph.GetAllNodes()) {
      if (sub_node->GetType() == "Store") {
        af::Expression offset;
        EXPECT_EQ(sub_node->attr.ir_attr->GetAttrValue("offset", offset), 0);
        offsets.emplace_back(offset);
      }
    }
  }
  for (size_t i = 0; i < offsets.size(); ++i) {
    EXPECT_SYMBOL_EQ(offsets[i], expect[i]);
  }
  EXPECT_EQ(fused_scheduled_result.input_nodes.size(), 2);
  EXPECT_EQ(fused_scheduled_result.output_nodes.size(), 1);
  EXPECT_EQ(fused_scheduled_result.workspace_nodes.size(), 0);
  EXPECT_EQ(fused_scheduled_result.input_nodes[0]->GetName(), "x1");
  EXPECT_EQ(fused_scheduled_result.input_nodes[1]->GetName(), "x2");
  EXPECT_EQ(fused_scheduled_result.output_nodes[0]->GetName(), "y");
}

void CreateAscBackendGraphTwoInTwoOut(std::shared_ptr<af::AscGraph> &graph, const std::string &prefix,
                                      int64_t axis_num = 2) {
  auto ONE = af::Symbol(1);
  std::vector<int64_t> axis_ids;
  std::vector<af::Expression> repeats;
  for (int64_t i = 0; i < axis_num; ++i) {
    const af::Expression exp = graph->CreateSizeVar("s" + std::to_string(i));
    auto axis = graph->CreateAxis("z" + std::to_string(i), exp);
    axis_ids.push_back(i);
    repeats.push_back(exp);
  }

  std::vector<af::Expression> strides(repeats.size(), One);
  if (axis_num > 1) {
    for (int64_t i = axis_num - 2; i >= 0; --i) {
      strides[i] = repeats[i + 1] * strides[i + 1];
    }
  }

  af::ascir_op::Data data0(std::string(prefix + "_data0").c_str(), *graph);
  data0.attr.sched.axis = axis_ids;
  *data0.y.axis = axis_ids;
  *data0.y.repeats = repeats;
  *data0.y.strides = strides;
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load0(std::string(prefix + "_load0").c_str());
  load0.x = data0.y;
  load0.attr.sched.axis = axis_ids;
  *load0.y.axis = axis_ids;
  *load0.y.repeats = repeats;
  *load0.y.strides = strides;
  load0.ir_attr.SetOffset(af::Symbol("s999"));

  af::ascir_op::Data data1(std::string(prefix + "_data1").c_str(), *graph);
  data1.attr.sched.axis = axis_ids;
  *data1.y.axis = axis_ids;
  *data1.y.repeats = repeats;
  *data1.y.strides = strides;
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load1(std::string(prefix + "_load1").c_str());
  load1.x = data1.y;
  load1.attr.sched.axis = axis_ids;
  *load1.y.axis = axis_ids;
  *load1.y.repeats = repeats;
  *load1.y.strides = strides;

  af::ascir_op::Add add(std::string(prefix + "_add").c_str());
  add.x1 = load0.y;
  add.x2 = load1.y;
  add.attr.sched.axis = axis_ids;
  *add.y.axis = axis_ids;
  *add.y.repeats = repeats;
  *add.y.strides = strides;

  af::ascir_op::Store store0(std::string(prefix + "_store0").c_str());
  store0.x = add.y;
  store0.attr.sched.axis = axis_ids;
  *store0.y.axis = axis_ids;
  *store0.y.repeats = repeats;
  *store0.y.strides = strides;

  af::ascir_op::Output y0(std::string(prefix + "_out0").c_str());
  y0.x = store0.y;
  y0.ir_attr.SetIndex(0);
  y0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Store store1(std::string(prefix + "_store1").c_str());
  store1.x = add.y;
  store1.attr.sched.axis = axis_ids;
  *store1.y.axis = axis_ids;
  *store1.y.repeats = repeats;
  *store1.y.strides = strides;

  af::ascir_op::Output y1(std::string(prefix + "_out1").c_str());
  y1.x = store1.y;
  y1.ir_attr.SetIndex(1);
  y1.y.dtype = ge::DT_FLOAT16;
}

void CreateAscBackendGraphTwoInOneOut(std::shared_ptr<af::AscGraph> &graph, const std::string &prefix,
                                      int64_t axis_num = 2) {
  auto ONE = af::Symbol(1);
  std::vector<int64_t> axis_ids;
  std::vector<af::Expression> repeats;
  for (int64_t i = 0; i < axis_num; ++i) {
    const af::Expression exp = graph->CreateSizeVar("s" + std::to_string(i));
    auto axis = graph->CreateAxis("z" + std::to_string(i), exp);
    axis_ids.push_back(i);
    repeats.push_back(exp);
  }

  std::vector<af::Expression> strides(repeats.size(), One);
  if (axis_num > 1) {
    for (int64_t i = axis_num - 2; i >= 0; --i) {
      strides[i] = repeats[i + 1] * strides[i + 1];
    }
  }

  af::ascir_op::Data data0(std::string(prefix + "_data0").c_str(), *graph);
  data0.attr.sched.axis = axis_ids;
  *data0.y.axis = axis_ids;
  *data0.y.repeats = repeats;
  *data0.y.strides = strides;
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load0(std::string(prefix + "_load0").c_str());
  load0.x = data0.y;
  load0.attr.sched.axis = axis_ids;
  *load0.y.axis = axis_ids;
  *load0.y.repeats = repeats;
  *load0.y.strides = strides;

  af::ascir_op::Data data1(std::string(prefix + "_data1").c_str(), *graph);
  data1.attr.sched.axis = axis_ids;
  *data1.y.axis = axis_ids;
  *data1.y.repeats = repeats;
  *data1.y.strides = strides;
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load1(std::string(prefix + "_load1").c_str());
  load1.x = data1.y;
  load1.attr.sched.axis = axis_ids;
  *load1.y.axis = axis_ids;
  *load1.y.repeats = repeats;
  *load1.y.strides = strides;

  af::ascir_op::Add add(std::string(prefix + "_add").c_str());
  add.x1 = load0.y;
  add.x2 = load1.y;
  add.attr.sched.axis = axis_ids;
  *add.y.axis = axis_ids;
  *add.y.repeats = repeats;
  *add.y.strides = strides;

  af::ascir_op::Store store0(std::string(prefix + "_store0").c_str());
  store0.x = add.y;
  store0.attr.sched.axis = axis_ids;
  *store0.y.axis = axis_ids;
  *store0.y.repeats = repeats;
  *store0.y.strides = strides;

  af::ascir_op::Output y0(std::string(prefix + "_out0").c_str());
  y0.x = store0.y;
  y0.ir_attr.SetIndex(0);
  y0.y.dtype = ge::DT_FLOAT16;
}

static NodePtr CreateAscbcToAscGraph(const std::string &name, ComputeGraphPtr &compute_graph, int64_t in_num = 1,
                                     int64_t out_num = 1) {
  OpDescBuilder op_desc_builder(name, "AscBackend");
  op_desc_builder.AddDynamicInput("x", in_num);
  op_desc_builder.AddDynamicOutput("y", out_num);
  const auto &op_desc = op_desc_builder.Build();
  auto node = compute_graph->AddNode(op_desc);
  node->SetOwnerComputeGraph(compute_graph);
  return node;
}

TEST_F(TestOptimizer, optimize_with_fused_ascbacked) {
  std::shared_ptr<af::AscGraph> g0 = std::make_shared<af::AscGraph>("g0");
  CreateAscBackendGraphTwoInTwoOut(g0, "g0", 2);
  std::shared_ptr<af::AscGraph> g1 = std::make_shared<af::AscGraph>("g1");
  CreateAscBackendGraphTwoInOneOut(g1, "g1", 1);
  std::shared_ptr<af::AscGraph> g2 = std::make_shared<af::AscGraph>("g2");
  CreateAscBackendGraphTwoInOneOut(g2, "g2", 2);

  af::AscGraph fused_asc_graph("fused_graph");
  af::ascir_op::Data data0("data0", fused_asc_graph);
  auto ir_attr0 = data0.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ir_attr0->SetIndex(0);

  af::ascir_op::Data data1("data1", fused_asc_graph);
  auto ir_attr1 = data1.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ir_attr1->SetIndex(1);

  af::ascir_op::Data data2("data2", fused_asc_graph);
  auto ir_attr2 = data2.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ir_attr2->SetIndex(2);

  auto fused_graph = af::AscGraphUtils::GetComputeGraph(fused_asc_graph);
  auto data0_node = fused_asc_graph.FindNode("data0");
  auto data1_node = fused_asc_graph.FindNode("data1");
  auto data2_node = fused_asc_graph.FindNode("data2");

  auto ascbc1 = CreateAscbcToAscGraph("ascbc1", fused_graph, 2, 2);
  auto ascbc2 = CreateAscbcToAscGraph("ascbc2", fused_graph, 2, 1);
  auto ascbc3 = CreateAscbcToAscGraph("ascbc3", fused_graph, 2, 1);

  af::GraphUtils::AddEdge(data0_node->GetOutDataAnchor(0), ascbc1->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), ascbc1->GetInDataAnchor(1));
  af::GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), ascbc2->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc1->GetOutDataAnchor(0), ascbc2->GetInDataAnchor(1));
  af::GraphUtils::AddEdge(ascbc2->GetOutDataAnchor(0), ascbc3->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc1->GetOutDataAnchor(1), ascbc3->GetInDataAnchor(1));

  af::ascir_op::Output output0("output0");
  auto out0_ir_attr = output0.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  out0_ir_attr->SetIndex(0);
  auto out0_desc = OpDescUtils::GetOpDescFromOperator(output0);
  auto output0_node = fused_graph->AddNode(out0_desc);

  af::ascir_op::Output output1("output1");
  auto out1_ir_attr = output1.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  out1_ir_attr->SetIndex(1);
  auto out1_desc = OpDescUtils::GetOpDescFromOperator(output1);
  auto output1_node = fused_graph->AddNode(out1_desc);
  af::GraphUtils::AddEdge(ascbc3->GetOutDataAnchor(0), output0_node->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc1->GetOutDataAnchor(1), output1_node->GetInDataAnchor(0));

  auto fuse1_attrs = ascbc1->GetOpDesc()->GetOrCreateAttrsGroup<ge::AutoFuseAttrs>();
  fuse1_attrs->SetAscGraph(g0);
  auto fuse2_attrs = ascbc2->GetOpDesc()->GetOrCreateAttrsGroup<ge::AutoFuseAttrs>();
  fuse2_attrs->SetAscGraph(g1);
  auto fuse3_attrs = ascbc3->GetOpDesc()->GetOrCreateAttrsGroup<ge::AutoFuseAttrs>();
  fuse3_attrs->SetAscGraph(g2);
  fused_graph->TopologicalSorting();

  optimize::Optimizer opt(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kFusedAscBackend});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(opt.Optimize(fused_graph, fused_scheduled_result), 0);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.fused_graph_name.GetString(), fused_graph->GetName());
  ASSERT_EQ(fused_scheduled_result.origin_vars.size(), 3UL);
  EXPECT_EQ(std::string(fused_scheduled_result.origin_vars[0].Serialize().get()), "s0");
  EXPECT_EQ(std::string(fused_scheduled_result.origin_vars[1].Serialize().get()), "s1");
  EXPECT_EQ(std::string(fused_scheduled_result.origin_vars[2].Serialize().get()), "s999");
  ASSERT_EQ(fused_scheduled_result.input_nodes.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.output_nodes.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.workspace_nodes.size(), 2UL);
}

static void CreatSingleConcatAscGraph(af::AscGraph &graph) {
  auto s0 = af::Symbol("10");
  auto s2 = af::Symbol("8");
  auto s1 = af::Symbol("24");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("concat_data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  *data0.y.axis = {z0.id, z1.id};
  *data0.y.repeats = {s0, s2};
  *data0.y.strides = {s2, af::sym::kSymbolOne};

  af::ascir_op::Load load0("concat_load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s2};
  *load0.y.strides = {s2, af::sym::kSymbolOne};

  af::ascir_op::Data data1("concat_data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  *data1.y.repeats = {s0, s2};
  *data1.y.strides = {s2, af::sym::kSymbolOne};

  af::ascir_op::Load load1("concat_load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s2};
  *load1.y.strides = {s2, af::sym::kSymbolOne};

  af::ascir_op::Data data2("concat_data2", graph);
  data2.ir_attr.SetIndex(2);
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  *data2.y.repeats = {s0, s2};
  *data2.y.strides = {s2, af::sym::kSymbolOne};

  af::ascir_op::Load load2("concat_load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s2};
  *load2.y.strides = {s2, af::sym::kSymbolOne};

  af::ascir_op::Concat concat("concat");
  concat.x = {load0.y, load1.y, load2.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, af::sym::kSymbolOne};

  af::ascir_op::Store store("concat_store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, af::sym::kSymbolOne};

  af::ascir_op::Output y("concat_out");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.y.dtype = ge::DT_FLOAT16;
}

TEST_F(TestOptimizer, only_concat_graph_tail_dim1_scene) {
  auto builder = GraphBuilder("test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto ascg1 = builder.AddNode("ascbc1", "AscGraph", 3, 1);
  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 2, 0);
  builder.AddDataEdge(data0, 0, ascg1, 0);
  builder.AddDataEdge(data1, 0, ascg1, 1);
  builder.AddDataEdge(data2, 0, ascg1, 2);
  builder.AddDataEdge(ascg1, 0, netoutput1, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  ASSERT_NE(compute_graph, nullptr);

  auto ascbc1 = compute_graph->FindNode("ascbc1");
  af::AscGraph concat_sub_graph("concat");
  CreatSingleConcatAscGraph(concat_sub_graph);

  std::string add_graph_str1;
  af::AscGraphUtils::SerializeToReadable(concat_sub_graph, add_graph_str1);
  af::AttrUtils::SetStr(ascbc1->GetOpDescBarePtr(), "ascgraph", add_graph_str1);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(optimizer.Optimize(compute_graph, fused_scheduled_result), 0);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.input_nodes.size(), 3UL);
  EXPECT_EQ(fused_scheduled_result.output_nodes.size(), 1UL);

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("concat_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("concat_tiling_data.h", std::ios::out);
  std::fstream kernel_func("concat_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

/**
 *         NetOutput
 *            |
 *          AscBc3
 *         /    \
 *     AscBc1   AscBc2
 *       |        |
 *     data0    data1
 */
static ComputeGraphPtr BuildFusedGraph(const std::string node_type = "") {
  auto builder = GraphBuilder("test", node_type);
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto ascbc1 = builder.AddNode("ascbc1", "AscGraph", 1, 1);
  auto ascbc2 = builder.AddNode("ascbc2", "AscGraph", 1, 1);
  auto ascbc3 = builder.AddNode("ascbc3", "AscGraph", 2, 1);

  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data0, 0, ascbc1, 0);
  builder.AddDataEdge(data1, 0, ascbc2, 0);
  builder.AddDataEdge(ascbc1, 0, ascbc3, 0);
  builder.AddDataEdge(ascbc2, 0, ascbc3, 1);
  builder.AddDataEdge(ascbc3, 0, netoutput1, 0);

  return builder.GetGraph();
}

static void CreateOneNodeAscGraph(af::AscGraph &graph, const std::string &prefix = "g0") {
  const af::Expression s0 = graph.CreateSizeVar("s0");
  const af::Expression s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1(std::string(prefix + "sub_data0").c_str(), graph);
  x1.ir_attr.SetIndex(0);
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::sym::kSymbolOne};

  af::ascir_op::Load load(std::string(prefix + "load0").c_str());
  load.x = x1.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, af::sym::kSymbolOne};

  af::ascir_op::Abs abs(std::string(prefix + "abs0").c_str());
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, af::sym::kSymbolOne};

  af::ascir_op::Store store(std::string(prefix + "store0").c_str());
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, af::sym::kSymbolOne};

  af::ascir_op::Output y(std::string(prefix + "out0").c_str());
  y.x = store.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);
}

static void CreateTailPackAscGraph(af::AscGraph &graph) {
  const af::Expression s0 = graph.CreateSizeVar("s0");
  const af::Expression s1 = graph.CreateSizeVar("s1");
  const af::Expression s2 = af::Symbol(2);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {s0, s1, af::sym::kSymbolOne};
  *data0.y.strides = {s1, af::sym::kSymbolOne, af::sym::kSymbolZero};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, af::sym::kSymbolOne};
  *load0.y.strides = {s1, af::sym::kSymbolOne, af::sym::kSymbolZero};

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data1.y.axis = {z0.id, z1.id, z2.id};
  *data1.y.repeats = {s0, s1, af::sym::kSymbolOne};
  *data1.y.strides = {s1, af::sym::kSymbolOne, af::sym::kSymbolZero};

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, af::sym::kSymbolOne};
  *load1.y.strides = {s1, af::sym::kSymbolOne, af::sym::kSymbolZero};

  af::ascir_op::Concat concat("concat");
  concat.x = {load0.y, load1.y};
  concat.attr.sched.axis = {z0.id, z1.id, z2.id};
  *concat.y.axis = {z0.id, z1.id, z2.id};
  *concat.y.repeats = {s0, s1, s2};
  *concat.y.strides = {s1 * s2, s2, af::sym::kSymbolOne};

  af::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, af::sym::kSymbolOne};

  af::ascir_op::Output y("out0");
  y.x = store.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);
}

static void CreateMidPackAscGraph(af::AscGraph &graph) {
  const af::Expression s0 = graph.CreateSizeVar("s0");
  const af::Expression s2 = af::Symbol(2);
  const af::Expression s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z2.id, z1.id};
  *data0.y.axis = {z0.id, z2.id, z1.id};
  *data0.y.repeats = {s0, af::sym::kSymbolOne, s1};
  *data0.y.strides = {s1, s1, af::sym::kSymbolOne};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z2.id, z1.id};
  *load0.y.axis = {z0.id, z2.id, z1.id};
  *load0.y.repeats = {s0, af::sym::kSymbolOne, s1};
  *load0.y.strides = {s1, s1, af::sym::kSymbolOne};

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z2.id, z1.id};
  *data1.y.axis = {z0.id, z2.id, z1.id};
  *data1.y.repeats = {s0, af::sym::kSymbolOne, s1};
  *data1.y.strides = {s1, s1, af::sym::kSymbolOne};

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z2.id, z1.id};
  *load1.y.axis = {z0.id, z2.id, z1.id};
  *load1.y.repeats = {s0, af::sym::kSymbolOne, s1};
  *load1.y.strides = {s1, s1, af::sym::kSymbolOne};

  af::ascir_op::Concat concat("concat");
  concat.x = {load0.y, load1.y};
  concat.attr.sched.axis = {z0.id, z2.id, z1.id};
  *concat.y.axis = {z0.id, z2.id, z1.id};
  *concat.y.repeats = {s0, s2, s1};
  *concat.y.strides = {s1 * s2, s1, af::sym::kSymbolOne};

  af::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z2.id, z1.id};
  *store.y.axis = {z0.id, z2.id, z1.id};
  *store.y.repeats = {s0, s2, s1};
  *store.y.strides = {s1 * s2, s1, af::sym::kSymbolOne};

  af::ascir_op::Output y("out0");
  y.x = store.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);
}

TEST_F(TestOptimizer, OptimizeWithFusedTailPack) {
  ComputeGraphPtr compute_graph = BuildFusedGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);
  auto ascbc3 = compute_graph->FindNode("ascbc3");
  ASSERT_NE(ascbc3, nullptr);
  af::AscGraph subgraph1("sub1");
  af::AscGraph subgraph2("sub2");
  af::AscGraph subgraph3("sub3");

  CreateOneNodeAscGraph(subgraph1, "g1");
  CreateOneNodeAscGraph(subgraph2, "g2");
  CreateTailPackAscGraph(subgraph3);

  std::string add_graph_str1;
  af::AscGraphUtils::SerializeToReadable(subgraph1, add_graph_str1);
  af::AttrUtils::SetStr(ascbc1->GetOpDescBarePtr(), "ascgraph", add_graph_str1);
  std::string add_graph_str2;
  af::AscGraphUtils::SerializeToReadable(subgraph2, add_graph_str2);
  af::AttrUtils::SetStr(ascbc2->GetOpDescBarePtr(), "ascgraph", add_graph_str2);
  std::string add_graph_str3;
  af::AscGraphUtils::SerializeToReadable(subgraph3, add_graph_str3);
  af::AttrUtils::SetStr(ascbc3->GetOpDescBarePtr(), "ascgraph", add_graph_str3);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(optimizer.Optimize(compute_graph, fused_scheduled_result), 0);
  // 1 aligned
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  //  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2UL);
  //  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups[0].impl_graphs.size(), 1UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto ascbc_1 = impl_graph.FindNode("ascbc1");
  EXPECT_EQ(ascbc_1, nullptr);
  auto ascbc_2 = impl_graph.FindNode("ascbc2");
  EXPECT_EQ(ascbc_2, nullptr);
  auto ascbc_3 = impl_graph.FindNode("ascbc3");
  EXPECT_EQ(ascbc_3, nullptr);
}

TEST_F(TestOptimizer, OptimizeWithFusedMidPack) {
  ComputeGraphPtr compute_graph = BuildFusedGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);
  auto ascbc3 = compute_graph->FindNode("ascbc3");
  ASSERT_NE(ascbc3, nullptr);
  af::AscGraph subgraph1("sub1");
  af::AscGraph subgraph2("sub2");
  af::AscGraph subgraph3("sub3");

  CreateOneNodeAscGraph(subgraph1, "g1");
  CreateOneNodeAscGraph(subgraph2, "g2");
  CreateMidPackAscGraph(subgraph3);

  std::string add_graph_str1;
  af::AscGraphUtils::SerializeToReadable(subgraph1, add_graph_str1);
  af::AttrUtils::SetStr(ascbc1->GetOpDescBarePtr(), "ascgraph", add_graph_str1);
  std::string add_graph_str2;
  af::AscGraphUtils::SerializeToReadable(subgraph2, add_graph_str2);
  af::AttrUtils::SetStr(ascbc2->GetOpDescBarePtr(), "ascgraph", add_graph_str2);
  std::string add_graph_str3;
  af::AscGraphUtils::SerializeToReadable(subgraph3, add_graph_str3);
  af::AttrUtils::SetStr(ascbc3->GetOpDescBarePtr(), "ascgraph", add_graph_str3);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(optimizer.Optimize(compute_graph, fused_scheduled_result), 0);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups[0].impl_graphs.size(), 2UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto ascbc_1 = impl_graph.FindNode("ascbc1");
  EXPECT_EQ(ascbc_1, nullptr);
  auto ascbc_2 = impl_graph.FindNode("ascbc2");
  EXPECT_EQ(ascbc_2, nullptr);
  auto ascbc_3 = impl_graph.FindNode("ascbc3");
  EXPECT_EQ(ascbc_3, nullptr);
}

void Construct_Mul_Consumer_Struct_UT(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto z0 = graph.CreateAxis("z0", s0 * s1 * s2);
  auto z1 = graph.CreateAxis("z1", s3);

  auto axis = {z0.id, z1.id};

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = af::ApiType::kAPITypeBuffer;
  arg4_1.y.dtype = ge::DT_FLOAT16;
  arg4_1.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = axis;
  b0_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0 * s1 * s2, s3};
  *b0_load.y.strides = {s3, One};

  Exp b1_exp("b1_exp");
  b1_exp.x = b0_load.y;
  b1_exp.attr.sched.axis = axis;
  b1_exp.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b1_exp.attr.api.type = af::ApiType::kAPITypeCompute;
  b1_exp.y.dtype = ge::DT_FLOAT16;
  *b1_exp.y.axis = axis;
  *b1_exp.y.repeats = {s0 * s1 * s2, s3};
  *b1_exp.y.strides = {s3, One};

  Abs b0_abs("b0_abs");
  b0_abs.x = b1_exp.y;
  b0_abs.attr.sched.axis = axis;
  b0_abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b0_abs.y.dtype = ge::DT_FLOAT16;
  *b0_abs.y.axis = axis;
  *b0_abs.y.repeats = {s0 * s1 * s2, s3};
  *b0_abs.y.strides = {s3, One};

  af::ascir_op::Max b0_max("b0_max");
  b0_max.x = b0_abs.y;
  b0_max.attr.sched.axis = axis;
  b0_max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT16;
  *b0_max.y.axis = axis;
  *b0_max.y.repeats = {s0 * s1 * s2, s3};
  *b0_max.y.strides = {One, Zero};

  Broadcast b1_broadcast("b1_broadcast");
  b1_broadcast.x = b0_max.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0 * s1 * s2, s3};
  *b1_broadcast.y.strides = {s3, One};

  Store b0_store("b0_store");
  b0_store.x = b1_broadcast.y;
  b0_store.attr.sched.axis = axis;
  b0_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b0_store.y.dtype = ge::DT_FLOAT16;
  *b0_store.y.axis = axis;
  *b0_store.y.repeats = {s0 * s1 * s2, s3};
  *b0_store.y.strides = {s3, One};

  Output buf0("buf0");
  buf0.x = b0_store.y;
  buf0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf0.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf0.y.dtype = ge::DT_FLOAT;
  buf0.ir_attr.SetIndex(1);

  af::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = b1_exp.y;
  b0_relu.attr.sched.axis = axis;
  b0_relu.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT16;
  *b0_relu.y.axis = axis;
  *b0_relu.y.repeats = {s0 * s1 * s2, s3};
  *b0_relu.y.strides = {s3, One};

  Store b1_store("b1_store");
  b1_store.x = b0_relu.y;
  b1_store.attr.sched.axis = axis;
  b1_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b1_store.y.dtype = ge::DT_FLOAT16;
  *b1_store.y.axis = axis;
  *b1_store.y.repeats = {s0 * s1 * s2, s3};
  *b1_store.y.strides = {s3, One};

  Output buf1("buf1");
  buf1.x = b1_store.y;
  buf1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf1.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf1.y.dtype = ge::DT_FLOAT;
  buf1.ir_attr.SetIndex(2);
}

TEST_F(TestOptimizer, REDUCE_MUL_CONSUMER) {
  af::AscGraph graph("REDUCE_MUL_CONSUMER");
  Construct_Mul_Consumer_Struct_UT(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 3UL);
}

void ConstructReduceGraphWithMultiOutputs(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto z0 = graph.CreateAxis("z0", s0 * s1 * s2);
  auto z1 = graph.CreateAxis("z1", s3);

  auto axis = {z0.id, z1.id};

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = af::ApiType::kAPITypeBuffer;
  arg4_1.y.dtype = ge::DT_FLOAT16;
  arg4_1.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = axis;
  b0_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0 * s1 * s2, s3};
  *b0_load.y.strides = {s3, One};

  Exp b1_exp("b1_exp");
  b1_exp.x = b0_load.y;
  b1_exp.attr.sched.axis = axis;
  b1_exp.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b1_exp.attr.api.type = af::ApiType::kAPITypeCompute;
  b1_exp.y.dtype = ge::DT_FLOAT16;
  *b1_exp.y.axis = axis;
  *b1_exp.y.repeats = {s0 * s1 * s2, s3};
  *b1_exp.y.strides = {s3, One};

  Store b1_store("b1_store");
  b1_store.x = b1_exp.y;
  b1_store.attr.sched.axis = axis;
  b1_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b1_store.y.dtype = ge::DT_FLOAT16;
  *b1_store.y.axis = axis;
  *b1_store.y.repeats = {s0 * s1 * s2, s3};
  *b1_store.y.strides = {s3, One};

  Output buf1("buf1");
  buf1.x = b1_store.y;
  buf1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf1.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf1.y.dtype = ge::DT_FLOAT;
  buf1.ir_attr.SetIndex(0);

  af::ascir_op::Max b0_max("b0_max");
  b0_max.x = b1_exp.y;
  b0_max.attr.sched.axis = axis;
  b0_max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT16;
  *b0_max.y.axis = axis;
  *b0_max.y.repeats = {s0 * s1 * s2, s3};
  *b0_max.y.strides = {One, Zero};

  Store b0_store("b0_store");
  b0_store.x = b0_max.y;
  b0_store.attr.sched.axis = axis;
  b0_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b0_store.y.dtype = ge::DT_FLOAT16;
  *b0_store.y.axis = axis;
  *b0_store.y.repeats = {s0 * s1 * s2, s3};
  *b0_store.y.strides = {s3, One};

  Output buf0("buf0");
  buf0.x = b0_store.y;
  buf0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf0.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf0.y.dtype = ge::DT_FLOAT;
  buf0.ir_attr.SetIndex(1);
}

TEST_F(TestOptimizer, ReduceTaskGenerate) {
  af::AscGraph graph("ReduceGraph");
  ConstructReduceGraphWithMultiOutputs(graph);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  std::vector<optimize::ScheduleTask> schedule_tasks;
  optimize::OptimizerOptions options{optimize::GraphType::kAscGraph};
  Status res = optimize::ScheduleTaskGenerator::GenerateTasks(graph, schedule_tasks, options);
  ASSERT_EQ(res, af::SUCCESS);
  ASSERT_EQ(schedule_tasks.size(), 3UL);
  int64_t ir_idx = -1;
  for (auto &task : schedule_tasks) {
    if (task.reduce_type == optimize::ReduceTemplateType::kRCore) {
      ASSERT_EQ(task.grouped_graphs.size(), 2UL);
      for (const auto &node : task.grouped_graphs[1].GetAllNodes()) {
        if (IsOps<Output>(node)) {
          auto ir_attr = node->attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
          ir_attr->GetIndex(ir_idx);
        }
      }
    }
  }
  EXPECT_EQ(ir_idx, 1);
  optimize::ReducePartitionCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, graphs, score_functions), 0);
}

TEST_F(TestOptimizer, TemplateSemanticNamesSurviveOptimize) {
  af::AscGraph graph("ReduceGraph");
  ConstructReduceGraphWithMultiOutputs(graph);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_schedule_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  std::set<std::string> semantic_names;
  for (const auto &scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      semantic_names.insert(scheduled_result.semantic_name);
    }
  }
  EXPECT_NE(semantic_names.find("Reduce/Common"), semantic_names.end());
  EXPECT_NE(semantic_names.find("Reduce/AllLoad"), semantic_names.end());
  for (const auto &scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      for (const auto &schedule_group : scheduled_result.schedule_groups) {
        for (const auto &impl_graph : schedule_group.impl_graphs) {
          semantic_names.insert(::ascgen_utils::GetTemplateSemanticName(impl_graph));
        }
      }
    }
  }
  EXPECT_NE(semantic_names.find("Reduce/RCore/Phase1"), semantic_names.end());
  EXPECT_NE(semantic_names.find("Reduce/RCore/Phase2"), semantic_names.end());
}

static af::AscGraph CreateContinuousAxisGraph(bool with_indirect_load) {
  af::AscGraph graph(with_indirect_load ? "ContinuousAxisWithIndirectLoad" : "ContinuousAxisControl");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  const std::vector<af::AxisId> axes = {z0.id, z1.id};
  const std::vector<af::Expression> repeats = {s0, s1};
  const std::vector<af::Expression> strides = {s1, One};

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.sched.axis = axes;
  *data.y.axis = axes;
  *data.y.repeats = repeats;
  *data.y.strides = strides;

  Load load("load");
  load.x = data.y;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.attr.api.type = af::ApiType::kAPITypeCompute;
  load.y.dtype = ge::DT_FLOAT16;
  load.attr.sched.axis = axes;
  *load.y.axis = axes;
  *load.y.repeats = repeats;
  *load.y.strides = strides;

  Store store("store");
  if (with_indirect_load) {
    Data index("index", graph);
    index.ir_attr.SetIndex(1);
    index.attr.api.type = af::ApiType::kAPITypeBuffer;
    index.y.dtype = ge::DT_INT32;
    index.attr.sched.axis = axes;
    *index.y.axis = axes;
    *index.y.repeats = repeats;
    *index.y.strides = strides;

    IndirectLoad indirect_load("indirect_load");
    indirect_load.x1 = load.y;
    indirect_load.x2 = index.y;
    indirect_load.ir_attr.SetAxis(1);
    indirect_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
    indirect_load.attr.api.type = af::ApiType::kAPITypeCompute;
    indirect_load.y.dtype = ge::DT_FLOAT16;
    indirect_load.attr.sched.axis = axes;
    *indirect_load.y.axis = axes;
    *indirect_load.y.repeats = repeats;
    *indirect_load.y.strides = strides;
    store.x = indirect_load.y;
  } else {
    store.x = load.y;
  }
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.api.type = af::ApiType::kAPITypeCompute;
  store.y.dtype = ge::DT_FLOAT16;
  store.attr.sched.axis = axes;
  *store.y.axis = axes;
  *store.y.repeats = repeats;
  *store.y.strides = strides;

  Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.attr.api.type = af::ApiType::kAPITypeBuffer;
  output.y.dtype = ge::DT_FLOAT16;
  output.attr.sched.axis = axes;
  *output.y.axis = axes;
  *output.y.repeats = repeats;
  *output.y.strides = strides;
  return graph;
}

TEST_F(TestOptimizer, MergeContinuousAxisSkipsWholeGraphContainingIndirectLoad) {
  struct TensorViewSnapshot {
    std::vector<af::AxisId> axis;
    std::vector<af::Expression> repeats;
    std::vector<af::Expression> strides;
  };
  struct NodeViewSnapshot {
    std::vector<af::AxisId> sched_axis;
    std::vector<TensorViewSnapshot> outputs;
  };

  auto indirect_load_graph = CreateContinuousAxisGraph(true);
  std::map<af::AxisId, std::pair<std::string, af::Expression>> original_axes;
  for (const auto &axis : indirect_load_graph.GetAllAxis()) {
    ASSERT_NE(axis, nullptr);
    ASSERT_TRUE(original_axes.emplace(axis->id, std::make_pair(axis->name, axis->size)).second);
  }
  ASSERT_EQ(original_axes.size(), 2UL);

  std::map<std::string, NodeViewSnapshot> original_nodes;
  for (const auto &node : indirect_load_graph.GetAllNodes()) {
    ASSERT_NE(node, nullptr);
    NodeViewSnapshot snapshot{node->attr.sched.axis, {}};
    for (const auto &output : node->outputs()) {
      ASSERT_NE(output, nullptr);
      snapshot.outputs.push_back({output->attr.axis, output->attr.repeats, output->attr.strides});
    }
    ASSERT_TRUE(original_nodes.emplace(node->GetName(), std::move(snapshot)).second);
  }
  ASSERT_EQ(original_nodes.count("indirect_load"), 1UL);

  ASSERT_EQ(optimizer.MergeContinuousAxis(indirect_load_graph), af::SUCCESS);

  const auto remaining_axes = indirect_load_graph.GetAllAxis();
  ASSERT_EQ(remaining_axes.size(), original_axes.size());
  for (const auto &axis : remaining_axes) {
    ASSERT_NE(axis, nullptr);
    const auto original_axis = original_axes.find(axis->id);
    ASSERT_NE(original_axis, original_axes.end());
    EXPECT_EQ(axis->name, original_axis->second.first);
    EXPECT_EQ(axis->size, original_axis->second.second);
  }

  auto remaining_nodes = indirect_load_graph.GetAllNodes();
  size_t remaining_node_count = 0UL;
  for (const auto &node : remaining_nodes) {
    ++remaining_node_count;
    ASSERT_NE(node, nullptr);
    SCOPED_TRACE(node->GetName());
    const auto original_node = original_nodes.find(node->GetName());
    ASSERT_NE(original_node, original_nodes.end());
    EXPECT_EQ(node->attr.sched.axis, original_node->second.sched_axis);
    ASSERT_EQ(node->outputs().size(), original_node->second.outputs.size());
    for (size_t i = 0UL; i < node->outputs().size(); ++i) {
      const auto output = node->outputs()[i];
      ASSERT_TRUE(static_cast<bool>(output));
      EXPECT_EQ(output->attr.axis, original_node->second.outputs[i].axis);
      EXPECT_EQ(output->attr.repeats, original_node->second.outputs[i].repeats);
      EXPECT_EQ(output->attr.strides, original_node->second.outputs[i].strides);
    }
  }
  ASSERT_EQ(remaining_node_count, original_nodes.size());

  auto control_graph = CreateContinuousAxisGraph(false);
  const auto control_original_axes = control_graph.GetAllAxis();
  ASSERT_EQ(control_original_axes.size(), 2UL);
  ASSERT_NE(control_original_axes[0], nullptr);
  ASSERT_NE(control_original_axes[1], nullptr);
  const auto merged_size = control_original_axes[0]->size * control_original_axes[1]->size;

  ASSERT_EQ(optimizer.MergeContinuousAxis(control_graph), af::SUCCESS);

  const auto control_merged_axes = control_graph.GetAllAxis();
  const std::set<af::AxisId> original_axis_ids = {control_original_axes[0]->id, control_original_axes[1]->id};
  const auto merged_axis = std::find_if(
      control_merged_axes.begin(), control_merged_axes.end(),
      [&original_axis_ids](const auto &axis) { return axis != nullptr && original_axis_ids.count(axis->id) == 0UL; });
  ASSERT_NE(merged_axis, control_merged_axes.end());
  EXPECT_EQ((*merged_axis)->size, merged_size);
  const auto merged_axis_id = (*merged_axis)->id;
  for (const auto &node : control_graph.GetAllNodes()) {
    ASSERT_NE(node, nullptr);
    if (node->attr.api.type == af::ApiType::kAPITypeBuffer) {
      continue;
    }
    SCOPED_TRACE(node->GetName());
    ASSERT_EQ(node->attr.sched.axis.size(), 1UL);
    EXPECT_EQ(node->attr.sched.axis[0], merged_axis_id);
    ASSERT_FALSE(node->outputs().empty());
    for (const auto &output : node->outputs()) {
      ASSERT_NE(output, nullptr);
      ASSERT_EQ(output->attr.axis.size(), output->attr.repeats.size());
      ASSERT_EQ(output->attr.axis.size(), output->attr.strides.size());
      ASSERT_EQ(output->attr.axis.size(), 1UL);
      EXPECT_EQ(output->attr.axis[0], merged_axis_id);
      EXPECT_EQ(output->attr.repeats[0], merged_size);
      EXPECT_EQ(output->attr.strides[0], One);
    }
  }
}

TEST_F(TestOptimizer, MergeAxesElewiseOnly) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  data.ir_attr.SetIndex(0);

  af::ascir_op::Abs abs("abs");
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.x = data.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs.y.repeats = {s0, s1, s2, s3};
  *abs.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Store store("store");
  store.attr.api.compute_type = af::ComputeType::kComputeElewise;
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, s2, s3};
  *store.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  ASSERT_EQ(optimizer.MergeContinuousAxis(graph), 0);

  auto new_axis = graph.GetAllAxis();
  ASSERT_EQ(new_axis.size(), 1UL);
  EXPECT_EQ(new_axis[0]->size, s0 * s1 * s2 * s3);
}

TEST_F(TestOptimizer, TailAxisSliceDoNotMerge) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  auto stride = af::Symbol(2);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load");
  load.x = data0.y;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.repeats = {s0, s1, s2, s3};
  *load.y.strides = {s1 * s2 * s3 * stride, s2 * s3 * stride, s3 * stride, stride};

  af::ascir_op::Abs abs("abs");
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs.y.repeats = {s0, s1, s2, s3};
  *abs.y.strides = {s1 * s2 * s3 * stride, s2 * s3 * stride, s3 * stride, stride};

  ASSERT_EQ(optimizer.MergeContinuousAxis(graph), af::SUCCESS);

  auto new_axis = graph.GetAllAxis();
  ASSERT_EQ(new_axis.size(), 2UL);
  EXPECT_EQ(new_axis[0]->size, s0 * s1 * s2);
  EXPECT_EQ(new_axis[1]->size, s3);
}

TEST_F(TestOptimizer, MergeAxesGather) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {s0, s1, s2};
  *data0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z3.id, z4.id};
  *data1.y.axis = {z3.id, z4.id};
  *data1.y.repeats = {s3, s4};
  *data1.y.strides = {s4, One};

  af::ascir_op::Gather gather("gather");
  gather.attr.api.compute_type = af::ComputeType::kComputeGather;
  gather.x1 = data0.y;
  gather.x2 = data1.y;
  gather.ir_attr.SetAxis(2);
  gather.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
  gather.y.dtype = ge::DT_FLOAT;
  *gather.y.axis = {z0.id, z1.id, z3.id, z4.id};
  *gather.y.repeats = {s0, s1, s3, s4};
  *gather.y.strides = {s1 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Abs abs("abs");
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.x = gather.y;
  abs.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z3.id, z4.id};
  *abs.y.repeats = {s0, s1, s3, s4};
  *abs.y.strides = {s1 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Store store("store");
  store.attr.api.compute_type = af::ComputeType::kComputeElewise;
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s3, s4};
  *store.y.strides = {s1 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT;
  y.ir_attr.SetIndex(0);

  ASSERT_EQ(optimizer.MergeContinuousAxis(graph), 0);

  auto new_axis = graph.GetAllAxis();
  EXPECT_EQ(new_axis[5]->size, s0 * s1);
  EXPECT_EQ(new_axis[6]->size, s3 * s4);
}

TEST_F(TestOptimizer, MergeAxesGatherOnlyOneDim) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id};
  *data0.y.axis = {z0.id};
  *data0.y.repeats = {s0};
  *data0.y.strides = {One};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z1.id, z2.id};
  *data1.y.axis = {z1.id, z2.id};
  *data1.y.repeats = {s1, s2};
  *data1.y.strides = {s2, One};

  af::ascir_op::Gather gather("gather");
  gather.attr.api.compute_type = af::ComputeType::kComputeGather;
  gather.x1 = data0.y;
  gather.x2 = data1.y;
  gather.ir_attr.SetAxis(2);
  gather.attr.sched.axis = {z1.id, z2.id};
  gather.y.dtype = ge::DT_FLOAT;
  *gather.y.axis = {z1.id, z2.id};
  *gather.y.repeats = {s1, s2};
  *gather.y.strides = {s2, One};

  af::ascir_op::Abs abs("abs");
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.x = gather.y;
  abs.attr.sched.axis = {z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z1.id, z2.id};
  *abs.y.repeats = {s1, s2};
  *abs.y.strides = {s2, One};

  af::ascir_op::Store store("store");
  store.attr.api.compute_type = af::ComputeType::kComputeElewise;
  store.x = abs.y;
  store.attr.sched.axis = {z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z1.id, z2.id};
  *store.y.repeats = {s1, s2};
  *store.y.strides = {s2, One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT;
  y.ir_attr.SetIndex(0);

  ASSERT_EQ(optimizer.MergeContinuousAxis(graph), 0);

  auto new_axis = graph.GetAllAxis();
  EXPECT_EQ(new_axis[3]->size, s1 * s2);
}

TEST_F(TestOptimizer, CompleteGatherApiInfoSetsDcacheSize) {
  constexpr int64_t kSimtDcacheSize = 40 * 1024;
  af::AscGraph graph("GatherDcacheGraph");

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT16;
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_INT32;
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Gather gather("gather");
  gather.x1 = data0.y;
  gather.x2 = data1.y;
  gather.ir_attr.SetAxis(0);
  gather.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Output output("output");
  output.x = gather.y;
  output.y.dtype = ge::DT_FLOAT16;
  output.ir_attr.SetIndex(0);

  auto gather_node = graph.FindNode("gather");
  ASSERT_NE(gather_node, nullptr);
  gather_node->attr.api.compute_type = af::ComputeType::kComputeGather;
  EXPECT_EQ(::ascir::GetDcacheSize(*gather_node), 0);

  ASSERT_EQ(optimize::AscGraphInfoComplete::CompleteApiInfo(graph), af::SUCCESS);
  EXPECT_EQ(::ascir::GetDcacheSize(*gather_node), kSimtDcacheSize);
}

TEST_F(TestOptimizer, CompleteExpm1ApiInfoSetsDcacheSize) {
  constexpr int64_t kSimtDcacheSize = 40 * 1024;
  af::AscGraph graph("Expm1DcacheGraph");

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_BF16;
  data.ir_attr.SetIndex(0);

  af::ascir_op::Expm1 expm1("expm1");
  expm1.x = data.y;
  expm1.y.dtype = ge::DT_BF16;

  af::ascir_op::Output output("output");
  output.x = expm1.y;
  output.y.dtype = ge::DT_BF16;
  output.ir_attr.SetIndex(0);

  auto expm1_node = graph.FindNode("expm1");
  ASSERT_NE(expm1_node, nullptr);
  EXPECT_EQ(::ascir::GetDcacheSize(*expm1_node), 0);

  ASSERT_EQ(optimize::AscGraphInfoComplete::CompleteApiInfo(graph), af::SUCCESS);
  EXPECT_EQ(::ascir::GetDcacheSize(*expm1_node), kSimtDcacheSize);
}

TEST_F(TestOptimizer, MergeAxesReduce) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto s5 = graph.CreateSizeVar("s5");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  data.ir_attr.SetIndex(0);

  af::ascir_op::Abs abs("abs");
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.x = data.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *abs.y.repeats = {s0, s1, s2, s3, s4, s5};
  *abs.y.strides = {s1 * s2 * s3 * s4 * s5, s2 * s3 * s4 * s5, s3 * s4 * s5, s4 * s5, s5, One};

  af::ascir_op::Sum sum("sum");
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = abs.y;
  sum.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  sum.y.dtype = ge::DT_FLOAT16;
  *sum.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *sum.y.repeats = {s0, s1, One, One, s4, s5};
  *sum.y.strides = {s1 * s4 * s5, s4 * s5, Zero, Zero, s5, One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = sum.y;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  ASSERT_EQ(optimizer.MergeContinuousAxis(graph), 0);

  auto new_axis = graph.GetAllAxis();
  ASSERT_EQ(new_axis.size(), 3UL);
  EXPECT_EQ(new_axis[0]->size, s0 * s1);
  EXPECT_EQ(new_axis[1]->size, s2 * s3);
  EXPECT_EQ(new_axis[2]->size, s4 * s5);
}

TEST_F(TestOptimizer, MergeAxesBrc) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  data.ir_attr.SetIndex(0);

  af::ascir_op::Abs abs("abs");
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.x = data.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs.y.repeats = {One, s1, s2, s3};
  *abs.y.strides = {Zero, s2 * s3, s3, One};

  af::ascir_op::Broadcast brc("brc");
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc.x = abs.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc.y.dtype = ge::DT_FLOAT16;
  *brc.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc.y.repeats = {s0, s1, s2, s3};
  *brc.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = brc.y;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  ASSERT_EQ(optimizer.MergeContinuousAxis(graph), 0);

  auto new_axis = graph.GetAllAxis();
  ASSERT_EQ(new_axis.size(), 2UL);
  EXPECT_EQ(new_axis[0]->size, s0);
  EXPECT_EQ(new_axis[1]->size, s1 * s2 * s3);
}

TEST_F(TestOptimizer, MergeAxesTransose) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  data.ir_attr.SetIndex(0);

  af::ascir_op::Abs abs("abs");
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.x = data.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs.y.repeats = {s0, s1, s2, s3};
  *abs.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Broadcast brc("brc");
  brc.attr.api.compute_type = af::ComputeType::kComputeTranspose;
  brc.x = abs.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc.y.dtype = ge::DT_FLOAT16;
  *brc.y.axis = {z2.id, z3.id, z0.id, z1.id};
  *brc.y.repeats = {s2, s3, s0, s1};
  *brc.y.strides = {s3 * s0 * s1, s0 * s1, s1, One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = brc.y;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  ASSERT_EQ(optimizer.MergeContinuousAxis(graph), 0);

  auto new_axis = graph.GetAllAxis();
  ASSERT_EQ(new_axis.size(), 2UL);
  EXPECT_EQ(new_axis[0]->size, s0 * s1);
  EXPECT_EQ(new_axis[1]->size, s2 * s3);
}

TEST_F(TestOptimizer, MergeAxesConcat) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = s2 * af::Symbol(2);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s4);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  data.ir_attr.SetIndex(0);

  af::ascir_op::Abs abs("abs");
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.x = data.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs.y.repeats = {s0, s1, s2, s3};
  *abs.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Abs abs1("abs1");
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.x = data.y;
  abs1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs1.y.dtype = ge::DT_FLOAT16;
  *abs1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs1.y.repeats = {s0, s1, s2, s3};
  *abs1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Concat concat("concat");
  concat.attr.api.compute_type = af::ComputeType::kComputeConcat;
  concat.x = {abs.y, abs1.y};
  concat.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *concat.y.repeats = {s0, s1, s4, s3};
  *concat.y.strides = {s1 * s4 * s3, s4 * s3, s3, One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = concat.y;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  ASSERT_EQ(optimizer.MergeContinuousAxis(graph), 0);

  auto new_axis = graph.GetAllAxis();
  ASSERT_EQ(new_axis.size(), 2UL);
  EXPECT_EQ(new_axis[0]->size, s0 * s1);
  EXPECT_EQ(new_axis[1]->size, s4 * s3);
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Size_Not_Equal_Failed) {
  af::AscGraph graph("ScalarBroadcastOptimization_Size_Not_Equal_Failed");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1};
  *brc2.y.strides = {af::ops::Zero, One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.x1 = brc3.y;
  add.x2 = load1.y;
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s2, af::ops::One};

  Exp exp("exp");
  exp.attr.sched.axis = {z0.id, z1.id, z2.id};
  exp.x = brc2.y;
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.axis = {z0.id, z1.id, z2.id};
  *exp.y.repeats = {af::ops::One, s1, s2};
  *exp.y.strides = {af::ops::Zero, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 11);
  EXPECT_NE(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc3"), nullptr);
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Single) {
  af::AscGraph graph("ScalarBroadcastOptimization_Single");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.x1 = brc3.y;
  add.x2 = load1.y;
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 7);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);
  EXPECT_NE(compute_graph->FindNode("add"), nullptr);
  auto add_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("add"));
  std::stringstream add0_repeats;
  SizeExprListStr(add0_repeats, graph, add_node->inputs[0].attr.repeats);
  std::stringstream expected0_repeats;
  SizeExprListStr(expected0_repeats, graph, {s0, s1, s2});
  EXPECT_EQ(add0_repeats.str(), expected0_repeats.str());

  std::stringstream add1_repeats;
  SizeExprListStr(add1_repeats, graph, add_node->inputs[1].attr.repeats);
  std::stringstream expected1_repeats;
  SizeExprListStr(expected1_repeats, graph, {af::ops::One, af::ops::One, af::ops::One});
  EXPECT_EQ(add1_repeats.str(), expected1_repeats.str());
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Multi_Out_Success) {
  af::AscGraph graph("ScalarBroadcastOptimization_Multi_Out_Success");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.x1 = brc3.y;
  add.x2 = load1.y;
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s2, af::ops::One};

  Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul.x1 = add.y;
  mul.x2 = brc3.y;
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id, z2.id};
  *mul.y.repeats = {s0, s1, s2};
  *mul.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = mul.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 8);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);
  EXPECT_NE(compute_graph->FindNode("add"), nullptr);
  auto add_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("add"));
  std::stringstream add0_repeats;
  SizeExprListStr(add0_repeats, graph, add_node->inputs[0].attr.repeats);
  std::stringstream expected0_repeats;
  SizeExprListStr(expected0_repeats, graph, {s0, s1, s2});
  EXPECT_EQ(add0_repeats.str(), expected0_repeats.str());
  std::stringstream add1_repeats;
  SizeExprListStr(add1_repeats, graph, add_node->inputs[1].attr.repeats);
  std::stringstream expected1_repeats;
  SizeExprListStr(expected1_repeats, graph, {af::ops::One, af::ops::One, af::ops::One});
  EXPECT_EQ(add1_repeats.str(), expected1_repeats.str());

  EXPECT_NE(compute_graph->FindNode("mul"), nullptr);
  auto mul_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("mul"));
  std::stringstream mul0_repeats;
  SizeExprListStr(mul0_repeats, graph, mul_node->inputs[0].attr.repeats);
  EXPECT_EQ(mul0_repeats.str(), expected0_repeats.str());
  std::stringstream mul1_repeats;
  SizeExprListStr(mul1_repeats, graph, mul_node->inputs[1].attr.repeats);
  EXPECT_EQ(mul1_repeats.str(), expected1_repeats.str());
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Multi_Out_Failed) {
  af::AscGraph graph("ScalarBroadcastOptimization_Multi_Out_Failed");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.x1 = brc3.y;
  add.x2 = load1.y;
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s2, af::ops::One};

  Exp exp("exp");
  exp.attr.sched.axis = {z0.id, z1.id, z2.id};
  exp.x = brc2.y;
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.axis = {z0.id, z1.id, z2.id};
  *exp.y.repeats = {af::ops::One, s1, s2};
  *exp.y.strides = {af::ops::Zero, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 11);
  EXPECT_NE(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc3"), nullptr);
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Api_Not_Support_Scalar) {
  af::AscGraph graph("ScalarBroadcastOptimization_Api_Not_Support_Scalar");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Sub sub("sub");
  sub.attr.sched.axis = {z0.id, z1.id, z2.id};
  sub.x1 = brc3.y;
  sub.x2 = load1.y;
  sub.y.dtype = ge::DT_FLOAT;
  *sub.y.axis = {z0.id, z1.id, z2.id};
  *sub.y.repeats = {s0, s1, s2};
  *sub.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = sub.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 7);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Not_Load) {
  af::AscGraph graph("ScalarBroadcastOptimization_Not_Load");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Exp exp("exp");
  exp.attr.sched.axis = {z0.id, z1.id, z2.id};
  exp.x = load.y;
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.axis = {z0.id, z1.id, z2.id};
  *exp.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *exp.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = exp.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.x1 = brc3.y;
  add.x2 = load1.y;
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 8);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Two_Scalar) {
  af::AscGraph graph("ScalarBroadcastOptimization_Two_Scalar");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc4("brc4");
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc4.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc4.x = load1.y;
  *brc4.y.axis = {z0.id, z1.id, z2.id};
  brc4.y.dtype = ge::DT_FLOAT;
  *brc4.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc4.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc5("brc5");
  brc5.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc5.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc5.x = brc4.y;
  *brc5.y.axis = {z0.id, z1.id, z2.id};
  brc5.y.dtype = ge::DT_FLOAT;
  *brc5.y.repeats = {af::ops::One, s1, s2};
  *brc5.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc6("brc6");
  brc6.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc6.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc6.x = brc5.y;
  *brc6.y.axis = {z0.id, z1.id, z2.id};
  brc6.y.dtype = ge::DT_FLOAT;
  *brc6.y.repeats = {s0, s1, s2};
  *brc6.y.strides = {s1 * s2, s2, af::ops::One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.x1 = brc3.y;
  add.x2 = brc6.y;
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 10);
  const auto retained_brc1 = compute_graph->FindNode("brc1");
  const auto retained_brc2 = compute_graph->FindNode("brc2");
  const auto retained_brc3 = compute_graph->FindNode("brc3");
  ASSERT_NE(retained_brc1, nullptr);
  ASSERT_NE(retained_brc2, nullptr);
  ASSERT_NE(retained_brc3, nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc4"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc5"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc6"), nullptr);
  EXPECT_EQ(retained_brc1->GetInDataNodes().at(0)->GetName(), "add");
  EXPECT_EQ(retained_brc2->GetInDataNodes().at(0)->GetName(), "brc1");
  EXPECT_EQ(retained_brc3->GetInDataNodes().at(0)->GetName(), "brc2");
  EXPECT_EQ(compute_graph->FindNode("store")->GetInDataNodes().at(0)->GetName(), "brc3");
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Same_Input) {
  af::AscGraph graph("ScalarBroadcastOptimization_Same_Input");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.x1 = brc3.y;
  add.x2 = brc3.y;
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 8);
  EXPECT_NE(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc3"), nullptr);
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Compare_2nd_Scalar) {
  af::AscGraph graph("ScalarBroadcastOptimization_Compare_2nd_Scalar");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Ge ge("ge");
  ge.attr.sched.axis = {z0.id, z1.id, z2.id};
  ge.x1 = load1.y;
  ge.x2 = brc3.y;
  ge.y.dtype = ge::DT_FLOAT;
  *ge.y.axis = {z0.id, z1.id, z2.id};
  *ge.y.repeats = {s0, s1, s2};
  *ge.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = ge.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 7);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);
  EXPECT_NE(compute_graph->FindNode("ge"), nullptr);
  auto ge_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("ge"));
  std::stringstream ge_repeats;
  SizeExprListStr(ge_repeats, graph, ge_node->inputs[0].attr.repeats);
  std::stringstream expected0_repeats;
  SizeExprListStr(expected0_repeats, graph, {s0, s1, s2});
  EXPECT_EQ(ge_repeats.str(), expected0_repeats.str());

  std::stringstream ge1_repeats;
  SizeExprListStr(ge1_repeats, graph, ge_node->inputs[1].attr.repeats);
  std::stringstream expected1_repeats;
  SizeExprListStr(expected1_repeats, graph, {af::ops::One, af::ops::One, af::ops::One});
  EXPECT_EQ(ge1_repeats.str(), expected1_repeats.str());
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Compare_1nd_Scalar) {
  af::AscGraph graph("ScalarBroadcastOptimization_Compare_1nd_Scalar");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Gt gt("gt");
  gt.attr.sched.axis = {z0.id, z1.id, z2.id};
  gt.x1 = brc3.y;
  gt.x2 = load1.y;
  gt.y.dtype = ge::DT_FLOAT;
  *gt.y.axis = {z0.id, z1.id, z2.id};
  *gt.y.repeats = {s0, s1, s2};
  *gt.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = gt.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 10);
  EXPECT_NE(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc3"), nullptr);
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Add_Eq_Common_Scalar_Success) {
  af::AscGraph graph("ScalarBroadcastOptimization_Add_Eq_Common_Scalar_Success");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.x1 = brc3.y;
  add.x2 = load1.y;
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s2, af::ops::One};

  Eq eq("eq");
  eq.attr.sched.axis = {z0.id, z1.id, z2.id};
  eq.x1 = add.y;
  eq.x2 = brc3.y;
  eq.y.dtype = ge::DT_FLOAT;
  *eq.y.axis = {z0.id, z1.id, z2.id};
  *eq.y.repeats = {s0, s1, s2};
  *eq.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = eq.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 8);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);
  EXPECT_NE(compute_graph->FindNode("add"), nullptr);
  EXPECT_NE(compute_graph->FindNode("eq"), nullptr);
  auto add_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("add"));
  std::stringstream add0_repeats;
  SizeExprListStr(add0_repeats, graph, add_node->inputs[0].attr.repeats);
  std::stringstream expected0_repeats;
  SizeExprListStr(expected0_repeats, graph, {s0, s1, s2});
  EXPECT_EQ(add0_repeats.str(), expected0_repeats.str());
  std::stringstream add1_repeats;
  SizeExprListStr(add1_repeats, graph, add_node->inputs[1].attr.repeats);
  std::stringstream expected1_repeats;
  SizeExprListStr(expected1_repeats, graph, {af::ops::One, af::ops::One, af::ops::One});
  EXPECT_EQ(add1_repeats.str(), expected1_repeats.str());

  auto eq_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("eq"));
  std::stringstream eq0_repeats;
  SizeExprListStr(eq0_repeats, graph, eq_node->inputs[0].attr.repeats);
  EXPECT_EQ(add0_repeats.str(), expected0_repeats.str());
  std::stringstream eq1_repeats;
  SizeExprListStr(eq1_repeats, graph, eq_node->inputs[1].attr.repeats);
  EXPECT_EQ(add1_repeats.str(), expected1_repeats.str());
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Add_Lt_Common_Scalar_Failed) {
  af::AscGraph graph("ScalarBroadcastOptimization_Add_Lt_Common_Scalar_Failed");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.x1 = brc3.y;
  add.x2 = load1.y;
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s2, af::ops::One};

  Lt lt("lt");
  lt.attr.sched.axis = {z0.id, z1.id, z2.id};
  lt.x1 = brc3.y;
  lt.x2 = add.y;
  lt.y.dtype = ge::DT_FLOAT;
  *lt.y.axis = {z0.id, z1.id, z2.id};
  *lt.y.repeats = {s0, s1, s2};
  *lt.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = lt.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 11);
  EXPECT_NE(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc3"), nullptr);
  EXPECT_NE(compute_graph->FindNode("add"), nullptr);
  EXPECT_NE(compute_graph->FindNode("lt"), nullptr);
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Sub_Eq_Success) {
  af::AscGraph graph("ScalarBroadcastOptimization_Sub_Eq_Success");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Sub sub("sub");
  sub.attr.sched.axis = {z0.id, z1.id, z2.id};
  sub.x1 = brc3.y;
  sub.x2 = load1.y;
  sub.y.dtype = ge::DT_FLOAT;
  *sub.y.axis = {z0.id, z1.id, z2.id};
  *sub.y.repeats = {s0, s1, s2};
  *sub.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(2);
  data2.y.dtype = ge::DT_FLOAT;

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load2.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc21("brc21");
  brc21.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc21.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc21.x = load2.y;
  *brc21.y.axis = {z0.id, z1.id, z2.id};
  brc21.y.dtype = ge::DT_FLOAT;
  *brc21.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc21.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc22("brc22");
  brc22.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc22.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc22.x = brc21.y;
  *brc22.y.axis = {z0.id, z1.id, z2.id};
  brc22.y.dtype = ge::DT_FLOAT;
  *brc22.y.repeats = {af::ops::One, s1, s2};
  *brc22.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc23("brc23");
  brc23.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc23.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc23.x = brc22.y;
  *brc23.y.axis = {z0.id, z1.id, z2.id};
  brc23.y.dtype = ge::DT_FLOAT;
  *brc23.y.repeats = {s0, s1, s2};
  *brc23.y.strides = {s1 * s2, s2, af::ops::One};

  Eq eq("eq");
  eq.attr.sched.axis = {z0.id, z1.id, z2.id};
  eq.x1 = sub.y;
  eq.x2 = brc23.y;
  eq.y.dtype = ge::DT_FLOAT;
  *eq.y.axis = {z0.id, z1.id, z2.id};
  *eq.y.repeats = {s0, s1, s2};
  *eq.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = eq.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 10);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc21"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc22"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc23"), nullptr);
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Min_Ne_Success) {
  af::AscGraph graph("ScalarBroadcastOptimization_Min_Ne_Success");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Minimum mininum("min");
  mininum.attr.sched.axis = {z0.id, z1.id, z2.id};
  mininum.x1 = brc3.y;
  mininum.x2 = load1.y;
  mininum.y.dtype = ge::DT_FLOAT;
  *mininum.y.axis = {z0.id, z1.id, z2.id};
  *mininum.y.repeats = {s0, s1, s2};
  *mininum.y.strides = {s1 * s2, s2, af::ops::One};

  Ne ne("ne");
  ne.attr.sched.axis = {z0.id, z1.id, z2.id};
  ne.x1 = mininum.y;
  ne.x2 = brc3.y;
  ne.y.dtype = ge::DT_FLOAT;
  *ne.y.axis = {z0.id, z1.id, z2.id};
  *ne.y.repeats = {s0, s1, s2};
  *ne.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = ne.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 8);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);

  auto min_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("min"));
  std::stringstream min0_repeats;
  SizeExprListStr(min0_repeats, graph, min_node->inputs[0].attr.repeats);
  std::stringstream expected0_repeats;
  SizeExprListStr(expected0_repeats, graph, {s0, s1, s2});
  EXPECT_EQ(min0_repeats.str(), expected0_repeats.str());
  std::stringstream min1_repeats;
  SizeExprListStr(min1_repeats, graph, min_node->inputs[1].attr.repeats);
  std::stringstream expected1_repeats;
  SizeExprListStr(expected1_repeats, graph, {af::ops::One, af::ops::One, af::ops::One});
  EXPECT_EQ(min1_repeats.str(), expected1_repeats.str());

  auto ne_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("ne"));
  std::stringstream ne0_repeats;
  SizeExprListStr(ne0_repeats, graph, ne_node->inputs[0].attr.repeats);
  EXPECT_EQ(ne0_repeats.str(), expected0_repeats.str());
  std::stringstream ne1_repeats;
  SizeExprListStr(ne1_repeats, graph, ne_node->inputs[1].attr.repeats);
  EXPECT_EQ(ne1_repeats.str(), expected1_repeats.str());
}

/**
 *               where
 *           /1    /0   \2
 *         /      /     \
 *       /   not_equal   \
 *      |     /   \       \
 *      |   /      \       \
 *      |  /        \       \
 *      | /       brc123   brc456
 *      |/          |        |
 *    load0       load1    load2
 *      |          |s       |s
 *    data0      data1    data2
 */
TEST_F(TestOptimizer, ScalarBroadcastOptimization_Where_3S_Success) {
  af::AscGraph graph("ScalarBroadcastOptimization_Where_3S_Success");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, af::ops::One};

  Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load1.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(2);
  data2.y.dtype = ge::DT_FLOAT;

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load2.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc4("brc4");
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc4.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc4.x = load2.y;
  *brc4.y.axis = {z0.id, z1.id, z2.id};
  brc4.y.dtype = ge::DT_FLOAT;
  *brc4.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc4.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc5("brc5");
  brc5.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc5.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc5.x = brc4.y;
  *brc5.y.axis = {z0.id, z1.id, z2.id};
  brc5.y.dtype = ge::DT_FLOAT;
  *brc5.y.repeats = {af::ops::One, s1, s2};
  *brc5.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc6("brc6");
  brc6.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc6.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc6.x = brc5.y;
  *brc6.y.axis = {z0.id, z1.id, z2.id};
  brc6.y.dtype = ge::DT_FLOAT;
  *brc6.y.repeats = {s0, s1, s2};
  *brc6.y.strides = {s1 * s2, s2, af::ops::One};

  Ne ne("ne");
  ne.attr.sched.axis = {z0.id, z1.id, z2.id};
  ne.x1 = brc3.y;
  ne.x2 = load0.y;
  ne.y.dtype = ge::DT_FLOAT;
  *ne.y.axis = {z0.id, z1.id, z2.id};
  *ne.y.repeats = {s0, s1, s2};
  *ne.y.strides = {s1 * s2, s2, af::ops::One};

  Where where("where");
  where.attr.sched.axis = {z0.id, z1.id, z2.id};
  where.x1 = ne.y;
  where.x2 = load0.y;
  where.x3 = brc6.y;
  where.y.dtype = ge::DT_FLOAT;
  *where.y.axis = {z0.id, z1.id, z2.id};
  *where.y.repeats = {s0, s1, s2};
  *where.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = where.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 10);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);

  EXPECT_EQ(compute_graph->FindNode("brc4"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc5"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc6"), nullptr);

  auto ne_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("ne"));
  std::stringstream ne0_repeats;
  SizeExprListStr(ne0_repeats, graph, ne_node->inputs[0].attr.repeats);
  std::stringstream expected0_repeats;
  SizeExprListStr(expected0_repeats, graph, {s0, s1, s2});
  EXPECT_EQ(ne0_repeats.str(), expected0_repeats.str());
  std::stringstream ne1_repeats;
  SizeExprListStr(ne1_repeats, graph, ne_node->inputs[1].attr.repeats);
  std::stringstream expected1_repeats;
  SizeExprListStr(expected1_repeats, graph, {af::ops::One, af::ops::One, af::ops::One});
  EXPECT_EQ(ne1_repeats.str(), expected1_repeats.str());

  auto where_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("where"));
  std::stringstream where1_repeats;
  SizeExprListStr(where1_repeats, graph, where_node->inputs[1].attr.repeats);
  EXPECT_EQ(where1_repeats.str(), expected0_repeats.str());
  std::stringstream where2_repeats;
  SizeExprListStr(where2_repeats, graph, where_node->inputs[2].attr.repeats);
  EXPECT_EQ(where2_repeats.str(), expected1_repeats.str());
}

/**
 *              select
 *           /2    /0  \1
 *         /      /     \
 *       /   not_equal   \
 *      |     /   \       \
 *      |   /      \       \
 *      |  /        \       \
 *      | /       brc123   brc456
 *      |/          |        |
 *    load0       load1    load2
 *      |          |s       |s
 *    data0      data1    data2
 */
TEST_F(TestOptimizer, ScalarBroadcastOptimization_Select_2S_Success) {
  af::AscGraph graph("ScalarBroadcastOptimization_Select_2S_Success");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, af::ops::One};

  Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load1.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(2);
  data2.y.dtype = ge::DT_FLOAT;

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load2.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc4("brc4");
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc4.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc4.x = load2.y;
  *brc4.y.axis = {z0.id, z1.id, z2.id};
  brc4.y.dtype = ge::DT_FLOAT;
  *brc4.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc4.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc5("brc5");
  brc5.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc5.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc5.x = brc4.y;
  *brc5.y.axis = {z0.id, z1.id, z2.id};
  brc5.y.dtype = ge::DT_FLOAT;
  *brc5.y.repeats = {af::ops::One, s1, s2};
  *brc5.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc6("brc6");
  brc6.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc6.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc6.x = brc5.y;
  *brc6.y.axis = {z0.id, z1.id, z2.id};
  brc6.y.dtype = ge::DT_FLOAT;
  *brc6.y.repeats = {s0, s1, s2};
  *brc6.y.strides = {s1 * s2, s2, af::ops::One};

  Ne ne("ne");
  ne.attr.sched.axis = {z0.id, z1.id, z2.id};
  ne.x1 = brc3.y;
  ne.x2 = load0.y;
  ne.y.dtype = ge::DT_FLOAT;
  *ne.y.axis = {z0.id, z1.id, z2.id};
  *ne.y.repeats = {s0, s1, s2};
  *ne.y.strides = {s1 * s2, s2, af::ops::One};

  Select select("select");
  select.attr.sched.axis = {z0.id, z1.id, z2.id};
  select.x1 = ne.y;
  select.x2 = brc6.y;
  select.x3 = load0.y;
  select.y.dtype = ge::DT_FLOAT;
  *select.y.axis = {z0.id, z1.id, z2.id};
  *select.y.repeats = {s0, s1, s2};
  *select.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = select.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 10);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);

  EXPECT_EQ(compute_graph->FindNode("brc4"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc5"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc6"), nullptr);

  auto ne_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("ne"));
  std::stringstream ne0_repeats;
  SizeExprListStr(ne0_repeats, graph, ne_node->inputs[0].attr.repeats);
  std::stringstream expected0_repeats;
  SizeExprListStr(expected0_repeats, graph, {s0, s1, s2});
  EXPECT_EQ(ne0_repeats.str(), expected0_repeats.str());
  std::stringstream ne1_repeats;
  SizeExprListStr(ne1_repeats, graph, ne_node->inputs[1].attr.repeats);
  std::stringstream expected1_repeats;
  SizeExprListStr(expected1_repeats, graph, {af::ops::One, af::ops::One, af::ops::One});
  EXPECT_EQ(ne1_repeats.str(), expected1_repeats.str());

  auto select_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("select"));
  std::stringstream select1_repeats;
  SizeExprListStr(select1_repeats, graph, select_node->inputs[1].attr.repeats);
  EXPECT_EQ(select1_repeats.str(), expected1_repeats.str());
  std::stringstream select2_repeats;
  SizeExprListStr(select2_repeats, graph, select_node->inputs[2].attr.repeats);
  EXPECT_EQ(select2_repeats.str(), expected0_repeats.str());
}

/**
 *                 select
 *               /0  \1  \2
 *              /     \    \
 *         not_equal   \     \
 *          /   \       \      \
 *         /      \      \       \
 *        /        \      \       \
 *       /       brc123  brc456  brc789
 *      /           |       |      |
 *    load0       load1   load2  load3
 *      |          |s      |s      |s
 *    data0      data1   data2   data3
 */
TEST_F(TestOptimizer, ScalarBroadcastOptimization_Select_2S_3S_Success) {
  af::AscGraph graph("ScalarBroadcastOptimization_Select_2S_3S_Success");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, af::ops::One};

  Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.x = load1.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(2);
  data2.y.dtype = ge::DT_FLOAT;

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load2.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc4("brc4");
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc4.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc4.x = load2.y;
  *brc4.y.axis = {z0.id, z1.id, z2.id};
  brc4.y.dtype = ge::DT_FLOAT;
  *brc4.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc4.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc5("brc5");
  brc5.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc5.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc5.x = brc4.y;
  *brc5.y.axis = {z0.id, z1.id, z2.id};
  brc5.y.dtype = ge::DT_FLOAT;
  *brc5.y.repeats = {af::ops::One, s1, s2};
  *brc5.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc6("brc6");
  brc6.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc6.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc6.x = brc5.y;
  *brc6.y.axis = {z0.id, z1.id, z2.id};
  brc6.y.dtype = ge::DT_FLOAT;
  *brc6.y.repeats = {s0, s1, s2};
  *brc6.y.strides = {s1 * s2, s2, af::ops::One};

  Ne ne("ne");
  ne.attr.sched.axis = {z0.id, z1.id, z2.id};
  ne.x1 = brc3.y;
  ne.x2 = load0.y;
  ne.y.dtype = ge::DT_FLOAT;
  *ne.y.axis = {z0.id, z1.id, z2.id};
  *ne.y.repeats = {s0, s1, s2};
  *ne.y.strides = {s1 * s2, s2, af::ops::One};

  Data data3("data3", graph);
  data3.ir_attr.SetIndex(3);
  data3.y.dtype = ge::DT_FLOAT;

  Load load3("load3");
  load3.attr.sched.axis = {z0.id, z1.id, z2.id};
  load3.x = data3.y;
  *load3.y.axis = {z0.id, z1.id, z2.id};
  load3.y.dtype = ge::DT_FLOAT;
  *load3.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load3.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc7("brc7");
  brc7.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc7.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc7.x = load3.y;
  *brc7.y.axis = {z0.id, z1.id, z2.id};
  brc7.y.dtype = ge::DT_FLOAT;
  *brc7.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc7.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc8("brc8");
  brc8.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc8.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc8.x = brc7.y;
  *brc8.y.axis = {z0.id, z1.id, z2.id};
  brc8.y.dtype = ge::DT_FLOAT;
  *brc8.y.repeats = {af::ops::One, s1, s2};
  *brc8.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc9("brc9");
  brc9.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc9.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc9.x = brc8.y;
  *brc9.y.axis = {z0.id, z1.id, z2.id};
  brc9.y.dtype = ge::DT_FLOAT;
  *brc9.y.repeats = {s0, s1, s2};
  *brc9.y.strides = {s1 * s2, s2, af::ops::One};

  Select select("select");
  select.attr.sched.axis = {z0.id, z1.id, z2.id};
  select.x1 = ne.y;
  select.x2 = brc6.y;
  select.x3 = brc9.y;
  select.y.dtype = ge::DT_FLOAT;
  *select.y.axis = {z0.id, z1.id, z2.id};
  *select.y.repeats = {s0, s1, s2};
  *select.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = select.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimizer.GraphPass(graph);
  EXPECT_EQ(res, af::SUCCESS);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 12);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);

  EXPECT_EQ(compute_graph->FindNode("brc4"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc5"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc6"), nullptr);

  EXPECT_EQ(compute_graph->FindNode("brc7"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc8"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc9"), nullptr);

  auto ne_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("ne"));
  std::stringstream ne0_repeats;
  SizeExprListStr(ne0_repeats, graph, ne_node->inputs[0].attr.repeats);
  std::stringstream expected0_repeats;
  SizeExprListStr(expected0_repeats, graph, {s0, s1, s2});
  EXPECT_EQ(ne0_repeats.str(), expected0_repeats.str());
  std::stringstream ne1_repeats;
  SizeExprListStr(ne1_repeats, graph, ne_node->inputs[1].attr.repeats);
  std::stringstream expected1_repeats;
  SizeExprListStr(expected1_repeats, graph, {af::ops::One, af::ops::One, af::ops::One});
  EXPECT_EQ(ne1_repeats.str(), expected1_repeats.str());

  auto select_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("select"));
  std::stringstream select1_repeats;
  SizeExprListStr(select1_repeats, graph, select_node->inputs[1].attr.repeats);
  EXPECT_EQ(select1_repeats.str(), expected1_repeats.str());
  std::stringstream select2_repeats;
  SizeExprListStr(select2_repeats, graph, select_node->inputs[2].attr.repeats);
  EXPECT_EQ(select2_repeats.str(), expected1_repeats.str());
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Scalar) {
  af::AscGraph graph("ScalarBroadcastOptimization_Scalar");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Scalar scalar0("data0", graph);
  scalar0.ir_attr.SetValue("0");
  scalar0.attr.sched.axis = {z0.id, z1.id, z2.id};
  scalar0.y.dtype = ge::DT_FLOAT16;
  *scalar0.y.axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = scalar0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {One, One, s2};
  *brc0.y.strides = {Zero, Zero, One};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {s0, One, s2};
  *brc1.y.strides = {s2, Zero, One};

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(0);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, One, s2};
  *load1.y.strides = {s2, Zero, One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.x1 = brc1.y;
  add.x2 = load1.y;
  *add.y.axis = {z0.id, z1.id, z2.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.repeats = {s0, One, s2};
  *add.y.strides = {s2, Zero, One};

  af::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, One, s2};
  *store.y.strides = {s2, Zero, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_TRUE(!fused_scheduled_result.node_idx_to_scheduled_results.empty());
  auto &schedule_results = fused_scheduled_result.node_idx_to_scheduled_results[0];
  EXPECT_EQ(schedule_results.size(), 1UL);
  EXPECT_EQ(schedule_results[0].schedule_groups.size(), 1UL);
  ASSERT_EQ(schedule_results[0].schedule_groups[0].impl_graphs.size(), 1UL);

  auto const &impl_graphs = schedule_results[0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs[0].FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[0].FindNode("brc1"), nullptr);
}

TEST_F(TestOptimizer, ScalarBroadcastOptimization_Not_Output) {
  af::AscGraph graph("ScalarBroadcastOptimization_Not_Output");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Scalar scalar0("data0", graph);
  scalar0.ir_attr.SetValue("0");
  scalar0.attr.sched.axis = {z0.id, z1.id, z2.id};
  scalar0.y.dtype = ge::DT_FLOAT16;
  *scalar0.y.axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = scalar0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {One, One, s2};
  *brc0.y.strides = {Zero, Zero, One};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {s0, One, s2};
  *brc1.y.strides = {s2, Zero, One};

  af::ascir_op::Store store("store");
  store.x = brc1.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, One, s2};
  *store.y.strides = {s2, Zero, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
}

TEST_F(TestOptimizer, NodeCacheMarkerConcat) {
  af::AscGraph graph("NodeCacheMarkerConcat");
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol("s1");
  auto s2 = af::Symbol("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2 * af::Symbol(2));

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_INT8;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Scalar scalar("scalar", graph);
  scalar.ir_attr.SetValue("0");
  scalar.attr.sched.axis = {z0.id, z1.id, z2.id};
  scalar.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  scalar.attr.api.type = af::ApiType::kAPITypeBuffer;
  scalar.y.dtype = ge::DT_INT8;
  *scalar.y.axis = {z0.id, z1.id, z2.id};
  *scalar.y.repeats = {One, One, One};
  *scalar.y.strides = {Zero, Zero, Zero};

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = scalar.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_INT8;
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {One, One, s2};
  *brc0.y.strides = {Zero, Zero, One};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_INT8;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {One, s1, s2};
  *brc1.y.strides = {Zero, s2, One};

  af::ascir_op::Broadcast brc2("brc2");
  brc2.x = brc1.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_INT8;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  *brc2.y.repeats = {s0, s1, s2};
  *brc2.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Concat concat("concat");
  concat.x = {load0.y, brc2.y};
  concat.attr.sched.axis = {z0.id, z1.id, z2.id};
  concat.y.dtype = ge::DT_INT8;
  *concat.y.axis = {z0.id, z1.id, z2.id};
  *concat.y.repeats = {s0, s1, s2 * af::Symbol(2)};
  *concat.y.strides = {s1 * s2 * af::Symbol(2), s2 * af::Symbol(2), One};
  concat.attr.api.compute_type = af::ComputeType::kComputeConcat;

  af::ascir_op::Store store0("store0");
  store0.x = concat.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_INT8;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2 * af::Symbol(2)};
  *store0.y.strides = {s1 * s2 * af::Symbol(2), s2 * af::Symbol(2), One};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(1);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_INT8;
  *y0.y.axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Store store1("store1");
  store1.x = brc2.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_INT8;
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, s2};
  *store1.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(0);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id};
  y1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_INT8;
  *y1.y.axis = {z0.id, z1.id, z2.id};

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_TRUE(!fused_scheduled_result.node_idx_to_scheduled_results.empty());
  auto &schedule_results = fused_scheduled_result.node_idx_to_scheduled_results[0];
  ASSERT_EQ(schedule_results.size(), 2UL);
  bool found_cached_brc0 = false;
  for (const auto &schedule_result : schedule_results) {
    for (const auto &schedule_group : schedule_result.schedule_groups) {
      for (const auto &impl_graph : schedule_group.impl_graphs) {
        const auto brc0 = impl_graph.FindNode("brc0");
        if (brc0 != nullptr &&
            brc0->attr.sched.exec_condition == af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis) {
          found_cached_brc0 = true;
        }
      }
    }
  }
  EXPECT_TRUE(found_cached_brc0);
}

TEST_F(TestOptimizer, NodeCacheMarkerBroadcast) {
  af::AscGraph graph("NodeCacheMarkerBroadcast");
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol("s1");
  auto s2 = af::Symbol("s2");
  auto s3 = af::Symbol("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_INT8;

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_INT8;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load0.y.repeats = {One, One, One, s3};
  *load0.y.strides = {Zero, Zero, Zero, One};

  Cast cast0("cast0");
  cast0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  cast0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast0.x = load0.y;
  cast0.y.dtype = ge::DT_FLOAT;
  *cast0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast0.y.repeats = {One, One, One, s3};
  *cast0.y.strides = {Zero, Zero, Zero, One};

  Relu relu0("relu0");
  relu0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  relu0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  relu0.x = cast0.y;
  relu0.y.dtype = ge::DT_FLOAT;
  *relu0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *relu0.y.repeats = {One, One, One, s3};
  *relu0.y.strides = {Zero, Zero, Zero, One};

  Abs abs0("abs0");
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.x = relu0.y;
  abs0.y.dtype = ge::DT_FLOAT;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs0.y.repeats = {One, One, One, s3};
  *abs0.y.strides = {Zero, Zero, Zero, One};

  Sqrt sqrt0("sqrt0");
  sqrt0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  sqrt0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  sqrt0.x = abs0.y;
  sqrt0.y.dtype = ge::DT_FLOAT;
  *sqrt0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *sqrt0.y.repeats = {One, One, One, s3};
  *sqrt0.y.strides = {Zero, Zero, Zero, One};

  Exp exp0("exp0");
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  exp0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp0.x = sqrt0.y;
  exp0.y.dtype = ge::DT_FLOAT;
  *exp0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *exp0.y.repeats = {One, One, One, s3};
  *exp0.y.strides = {Zero, Zero, Zero, One};

  af::ascir_op::Broadcast brc00("brc00");
  brc00.x = exp0.y;
  brc00.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc00.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc00.y.dtype = ge::DT_INT8;
  *brc00.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc00.y.repeats = {One, One, s2, s3};
  *brc00.y.strides = {Zero, Zero, s3, One};

  af::ascir_op::Broadcast brc01("brc01");
  brc01.x = brc00.y;
  brc01.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc01.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc01.y.dtype = ge::DT_INT8;
  *brc01.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc01.y.repeats = {One, s1, s2, s3};
  *brc01.y.strides = {Zero, s2 * s3, s3, One};

  af::ascir_op::Broadcast brc02("brc02");
  brc02.x = brc01.y;
  brc02.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc02.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc02.y.dtype = ge::DT_INT8;
  *brc02.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc02.y.repeats = {s0, s1, s2, s3};
  *brc02.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  // ------
  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_INT8;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load1.y.repeats = {One, One, s2, One};
  *load1.y.strides = {Zero, Zero, One, Zero};

  Cast cast1("cast1");
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  cast1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast1.x = load1.y;
  cast1.y.dtype = ge::DT_FLOAT;
  *cast1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast1.y.repeats = {One, One, s2, One};
  *cast1.y.strides = {Zero, Zero, One, Zero};

  Sigmoid sigmoid1("sigmoid1");
  sigmoid1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  sigmoid1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  sigmoid1.x = cast1.y;
  sigmoid1.y.dtype = ge::DT_FLOAT;
  *sigmoid1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *sigmoid1.y.repeats = {One, One, s2, One};
  *sigmoid1.y.strides = {Zero, Zero, One, Zero};

  Sign sign1("sign1");
  sign1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  sign1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  sign1.x = sigmoid1.y;
  sign1.y.dtype = ge::DT_FLOAT;
  *sign1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *sign1.y.repeats = {One, One, s2, One};
  *sign1.y.strides = {Zero, Zero, One, Zero};

  Mul mul1("mul1");
  mul1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  mul1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  mul1.x1 = sign1.y;
  mul1.x2 = sign1.y;
  mul1.y.dtype = ge::DT_FLOAT;
  *mul1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *mul1.y.repeats = {One, One, s2, One};
  *mul1.y.strides = {Zero, Zero, One, Zero};

  Exp exp1("exp1");
  exp1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  exp1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp1.x = mul1.y;
  exp1.y.dtype = ge::DT_FLOAT;
  *exp1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *exp1.y.repeats = {One, One, s2, One};
  *exp1.y.strides = {Zero, Zero, One, Zero};

  af::ascir_op::Broadcast brc10("brc10");
  brc10.x = exp1.y;
  brc10.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc10.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc10.y.dtype = ge::DT_INT8;
  *brc10.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc10.y.repeats = {One, One, s2, s3};
  *brc10.y.strides = {Zero, Zero, s3, One};

  af::ascir_op::Broadcast brc11("brc11");
  brc11.x = brc10.y;
  brc11.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc11.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc11.y.dtype = ge::DT_INT8;
  *brc11.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc11.y.repeats = {One, s1, s2, s3};
  *brc11.y.strides = {Zero, s2 * s3, s3, One};

  af::ascir_op::Broadcast brc12("brc12");
  brc12.x = brc11.y;
  brc12.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc12.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc12.y.dtype = ge::DT_INT8;
  *brc12.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc12.y.repeats = {s0, s1, s2, s3};
  *brc12.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add add0("add0");
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add0.x1 = brc12.y;
  add0.x2 = brc02.y;
  add0.y.dtype = ge::DT_FLOAT;
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add0.y.repeats = {s0, s1, s2, s3};
  *add0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  // ------
  af::ascir_op::Data data2("data2", graph);
  data2.ir_attr.SetIndex(2);
  data2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data2.attr.api.type = af::ApiType::kAPITypeBuffer;
  data2.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load2.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2.y.dtype = ge::DT_INT8;
  *load2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load2.y.repeats = {One, s1, s2, One};
  *load2.y.strides = {Zero, s2, One, Zero};

  Cast cast2("cast2");
  cast2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  cast2.attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast2.x = load2.y;
  cast2.y.dtype = ge::DT_FLOAT;
  *cast2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast2.y.repeats = {One, s1, s2, One};
  *cast2.y.strides = {Zero, s2, One, Zero};

  Abs abs2("abs2");
  abs2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs2.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs2.x = cast2.y;
  abs2.y.dtype = ge::DT_FLOAT;
  *abs2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs2.y.repeats = {One, s1, s2, One};
  *abs2.y.strides = {Zero, s2, One, Zero};

  Sign sign2("sign2");
  sign2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  sign2.attr.api.compute_type = af::ComputeType::kComputeElewise;
  sign2.x = abs2.y;
  sign2.y.dtype = ge::DT_FLOAT;
  *sign2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *sign2.y.repeats = {One, s1, s2, One};
  *sign2.y.strides = {Zero, s2, One, Zero};

  Exp exp2("exp2");
  exp2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  exp2.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp2.x = sign2.y;
  exp2.y.dtype = ge::DT_FLOAT;
  *exp2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *exp2.y.repeats = {One, s1, s2, One};
  *exp2.y.strides = {Zero, s2, One, Zero};

  af::ascir_op::Broadcast brc20("brc20");
  brc20.x = exp2.y;
  brc20.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc20.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc20.y.dtype = ge::DT_INT8;
  *brc20.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc20.y.repeats = {One, s1, s2, s3};
  *brc20.y.strides = {Zero, s2 * s3, s3, One};

  af::ascir_op::Broadcast brc21("brc21");
  brc21.x = brc20.y;
  brc21.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc21.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc21.y.dtype = ge::DT_INT8;
  *brc21.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc21.y.repeats = {s0, s1, s2, s3};
  *brc21.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add add1("add1");
  add1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  add1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add1.x1 = brc21.y;
  add1.x2 = add0.y;
  add1.y.dtype = ge::DT_FLOAT;
  *add1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add1.y.repeats = {s0, s1, s2, s3};
  *add1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  // ------
  af::ascir_op::Data data3("data3", graph);
  data3.ir_attr.SetIndex(3);
  data3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  data3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data3.attr.api.type = af::ApiType::kAPITypeBuffer;
  data3.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load3.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load3.y.dtype = ge::DT_INT8;
  *load3.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load3.y.repeats = {s0, One, One, s3};
  *load3.y.strides = {s3, Zero, Zero, One};

  Cast cast3("cast3");
  cast3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  cast3.attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast3.x = load3.y;
  cast3.y.dtype = ge::DT_FLOAT;
  *cast3.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast3.y.repeats = {s0, One, One, s3};
  *cast3.y.strides = {s3, Zero, Zero, One};

  Abs abs3("abs3");
  abs3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs3.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs3.x = cast3.y;
  abs3.y.dtype = ge::DT_FLOAT;
  *abs3.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs3.y.repeats = {s0, One, One, s3};
  *abs3.y.strides = {s3, Zero, Zero, One};

  Sign sign3("sign3");
  sign3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  sign3.attr.api.compute_type = af::ComputeType::kComputeElewise;
  sign3.x = abs3.y;
  sign3.y.dtype = ge::DT_FLOAT;
  *sign3.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *sign3.y.repeats = {s0, One, One, s3};
  *sign3.y.strides = {s3, Zero, Zero, One};

  Exp exp3("exp3");
  exp3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  exp3.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp3.x = sign3.y;
  exp3.y.dtype = ge::DT_FLOAT;
  *exp3.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *exp3.y.repeats = {s0, One, One, s3};
  *exp3.y.strides = {s3, Zero, Zero, One};

  af::ascir_op::Broadcast brc30("brc30");
  brc30.x = exp3.y;
  brc30.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc30.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc30.y.dtype = ge::DT_INT8;
  *brc30.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc30.y.repeats = {s0, One, s2, s3};
  *brc30.y.strides = {s2 * s3, Zero, s3, One};

  af::ascir_op::Broadcast brc31("brc31");
  brc31.x = brc30.y;
  brc31.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc31.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc31.y.dtype = ge::DT_INT8;
  *brc31.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc31.y.repeats = {s0, s1, s2, s3};
  *brc31.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add add2("add2");
  add2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  add2.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add2.x1 = brc31.y;
  add2.x2 = add1.y;
  add2.y.dtype = ge::DT_FLOAT;
  *add2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add2.y.repeats = {s0, s1, s2, s3};
  *add2.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  // -----
  Cast cast4("cast4");
  cast4.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  cast4.attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast4.x = add2.y;
  cast4.y.dtype = ge::DT_FLOAT16;
  *cast4.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast4.y.repeats = {s0, s1, s2, s3};
  *cast4.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Store store0("store0");
  store0.x = cast4.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store0.y.repeats = {s0, s1, s2, s3};
  *store0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, z2.id, z3.id};

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_TRUE(!fused_scheduled_result.node_idx_to_scheduled_results.empty());
  auto &schedule_results = fused_scheduled_result.node_idx_to_scheduled_results[0];

  for (auto const &result : schedule_results) {
    for (auto const &group : result.schedule_groups) {
      for (auto const &impl_graph : group.impl_graphs) {
        for (auto const &node : impl_graph.GetAllNodes()) {
          if (node != nullptr && IsOps<Broadcast>(node)) {
            bool condition =
                node->attr.sched.exec_condition == af::ExecuteCondition::kNoCache ||
                node->attr.sched.exec_condition == af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
            if (!condition) {
              printf("Graph: %s, Broadcast %s", impl_graph.GetName().c_str(), node->GetNamePtr());
            }
            EXPECT_EQ(condition, true);
          }
        }
      }
    }
  }
}

void Construct_Enable_Cache_Max_Struct(af::AscGraph &graph) {
  static af::Expression Zero = af::Symbol(0);
  static af::Expression One = af::Symbol(1);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  auto axis = {z0.id, z1.id, z2.id, z3.id};

  Data data_0("data_0", graph);
  data_0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data_0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data_0.y.dtype = ge::DT_FLOAT16;
  data_0.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = data_0.y;
  b0_load.attr.sched.axis = axis;
  b0_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0, s1, s2, s3};
  *b0_load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Broadcast b0_broadcast("b0_broadcast");
  b0_broadcast.x = b0_load.y;
  b0_broadcast.attr.sched.axis = axis;
  b0_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  b0_broadcast.y.dtype = ge::DT_FLOAT16;
  *b0_broadcast.y.axis = axis;
  *b0_broadcast.y.repeats = {s0, s1, s2, s3};
  *b0_broadcast.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Data data_1("data_1", graph);
  data_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data_1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data_1.y.dtype = ge::DT_FLOAT16;
  data_1.ir_attr.SetIndex(1);

  Load b1_load("b1_load");
  b1_load.x = data_1.y;
  b1_load.attr.sched.axis = axis;
  b1_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b1_load.y.dtype = ge::DT_FLOAT16;
  *b1_load.y.axis = axis;
  *b1_load.y.repeats = {s0, s1, One, s3};
  *b1_load.y.strides = {s1 * s3, s3, Zero, One};

  Broadcast b1_broadcast("b1_broadcast");
  b1_broadcast.x = b1_load.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0, s1, s2, s3};
  *b1_broadcast.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add b0_add("b0_add");
  b0_add.x1 = b0_broadcast.y;
  b0_add.x2 = b1_broadcast.y;
  b0_add.attr.sched.axis = axis;
  b0_add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b0_add.y.dtype = ge::DT_FLOAT16;
  *b0_add.y.axis = axis;
  *b0_add.y.repeats = {s0, s1, s2, s3};
  *b0_add.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Max b0_max("b0_max");
  b0_max.x = b0_add.y;
  b0_max.attr.sched.axis = axis;
  b0_max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT16;
  *b0_max.y.axis = axis;
  *b0_max.y.repeats = {s0, One, One, s3};
  *b0_max.y.strides = {s3, Zero, Zero, One};

  Store b0_store("b0_store");
  b0_store.x = b0_max.y;
  b0_store.attr.sched.axis = axis;
  b0_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b0_store.y.dtype = ge::DT_FLOAT16;
  *b0_store.y.axis = axis;
  *b0_store.y.repeats = {s0, One, One, s3};
  *b0_store.y.strides = {s3, Zero, Zero, One};

  Output output_0("output_0");
  output_0.x = b0_store.y;
  output_0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output_0.attr.api.type = af::ApiType::kAPITypeBuffer;
  output_0.y.dtype = ge::DT_FLOAT;
  output_0.ir_attr.SetIndex(0);
}

void Construct_Enable_Cache_Sum_Struct(af::AscGraph &graph) {
  static af::Expression Zero = af::Symbol(0);
  static af::Expression One = af::Symbol(1);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  auto axis = {z0.id, z1.id, z2.id, z3.id};

  Data data_0("data_0", graph);
  data_0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data_0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data_0.y.dtype = ge::DT_FLOAT16;
  data_0.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = data_0.y;
  b0_load.attr.sched.axis = axis;
  b0_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0, s1, s2, s3};
  *b0_load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Broadcast b0_broadcast("b0_broadcast");
  b0_broadcast.x = b0_load.y;
  b0_broadcast.attr.sched.axis = axis;
  b0_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  b0_broadcast.y.dtype = ge::DT_FLOAT16;
  *b0_broadcast.y.axis = axis;
  *b0_broadcast.y.repeats = {s0, s1, s2, s3};
  *b0_broadcast.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Data data_1("data_1", graph);
  data_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data_1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data_1.y.dtype = ge::DT_FLOAT16;
  data_1.ir_attr.SetIndex(1);

  Load b1_load("b1_load");
  b1_load.x = data_1.y;
  b1_load.attr.sched.axis = axis;
  b1_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b1_load.y.dtype = ge::DT_FLOAT16;
  *b1_load.y.axis = axis;
  *b1_load.y.repeats = {s0, s1, One, s3};
  *b1_load.y.strides = {s1 * s3, s3, Zero, One};

  Broadcast b1_broadcast("b1_broadcast");
  b1_broadcast.x = b1_load.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0, s1, s2, s3};
  *b1_broadcast.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add b0_add("b0_add");
  b0_add.x1 = b0_broadcast.y;
  b0_add.x2 = b1_broadcast.y;
  b0_add.attr.sched.axis = axis;
  b0_add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b0_add.y.dtype = ge::DT_FLOAT16;
  *b0_add.y.axis = axis;
  *b0_add.y.repeats = {s0, s1, s2, s3};
  *b0_add.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Sum b0_sum("b0_sum");
  b0_sum.x = b0_add.y;
  b0_sum.attr.sched.axis = axis;
  b0_sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b0_sum.y.dtype = ge::DT_FLOAT16;
  *b0_sum.y.axis = axis;
  *b0_sum.y.repeats = {s0, One, One, s3};
  *b0_sum.y.strides = {s3, Zero, Zero, One};

  Store b0_store("b0_store");
  b0_store.x = b0_sum.y;
  b0_store.attr.sched.axis = axis;
  b0_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b0_store.y.dtype = ge::DT_FLOAT16;
  *b0_store.y.axis = axis;
  *b0_store.y.repeats = {s0, One, One, s3};
  *b0_store.y.strides = {s3, Zero, Zero, One};

  Output output_0("output_0");
  output_0.x = b0_store.y;
  output_0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output_0.attr.api.type = af::ApiType::kAPITypeBuffer;
  output_0.y.dtype = ge::DT_FLOAT;
  output_0.ir_attr.SetIndex(0);
}

TEST_F(TestOptimizer, EnableCacheMax) {
  bool gen_success = true;
  af::AscGraph test_graph("enable_cache_max");
  Construct_Enable_Cache_Max_Struct(test_graph);
  try {
    auto codegen = codegen::Codegen(
        codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
    ::ascir::FusedScheduledResult fused_schedule_result;
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    optimizer.Optimize(test_graph, fused_schedule_result);

    codegen::CodegenResult result;
    codegen.Generate(fused_schedule_result, result);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestOptimizer, EnableCacheSum) {
  bool gen_success = true;
  af::AscGraph test_graph("enable_cache_sum");
  Construct_Enable_Cache_Sum_Struct(test_graph);
  try {
    auto codegen = codegen::Codegen(
        codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
    ::ascir::FusedScheduledResult fused_schedule_result;
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    optimizer.Optimize(test_graph, fused_schedule_result);

    codegen::CodegenResult result;
    codegen.Generate(fused_schedule_result, result);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestOptimizer, TransposeLongTailWithoutUB) {
  // Transpose 尾轴非转置, 大尾轴场景UT, transpose消除
  af::AscGraph graph("transpose_long_tail_without_UB");

  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(86);
  auto s2 = graph.CreateSizeVar(1536);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);
  Data x3_op("x3", graph);
  x3_op.ir_attr.SetIndex(2);

  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");

  std::vector<Data> all_data{x1_op, x2_op, x3_op};
  std::vector<Load> all_load{load_op1, load_op2, load_op3};

  for (size_t i = 0U; i < all_data.size(); ++i) {
    auto &x_op = all_data[i];
    auto &load_op = all_load[i];
    x_op.y.dtype = ge::DT_FLOAT16;
    x_op.attr.api.type = af::ApiType::kAPITypeBuffer;
    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.axis = {z0.id, z1.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.repeats = {s0, s1, s2};
    *load_op.y.strides = {s1 * s2, s2, af::ops::One};
  }
  load_op1.attr.sched.axis = {z1.id, z0.id, z2.id};
  *load_op1.y.axis = {z1.id, z0.id, z2.id};
  *load_op1.y.repeats = {s1, s0, s2};
  *load_op1.y.strides = {s0 * s2, s2, af::ops::One};

  af::ascir_op::Transpose transpose_op("transpose");
  transpose_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  transpose_op.x = load_op1.y;
  transpose_op.y.dtype = ge::DT_FLOAT16;
  *transpose_op.y.axis = {z0.id, z1.id, z2.id};
  *transpose_op.y.strides = {s1 * s2, s2, af::ops::One};
  *transpose_op.y.repeats = {s0, s1, s2};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = load_op2.y;
  add_op.x2 = load_op3.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1 * s2, s2, af::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  af::ascir_op::Mul mul_op("mul");
  mul_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul_op.x1 = transpose_op.y;
  mul_op.x2 = add_op.y;
  mul_op.y.dtype = ge::DT_FLOAT16;
  *mul_op.y.axis = {z0.id, z1.id, z2.id};
  *mul_op.y.strides = {s1 * s2, s2, af::ops::One};
  *mul_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};

  store_op.x = mul_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};
  store_op.ir_attr.SetOffset(af::Symbol(0));

  Output y_op("y");
  y_op.x = store_op.y;
  y_op.ir_attr.SetIndex(0);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  std::vector<optimize::ScheduleTask> schedule_tasks;
  optimize::OptimizerOptions options{optimize::GraphType::kAscGraph};
  int res = optimize::ScheduleTaskGenerator::GenerateTasks(graph, schedule_tasks, options);
  ASSERT_EQ(res, 0);
  ASSERT_EQ(schedule_tasks.size(), 2);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(schedule_tasks[1].grouped_graphs[0]);
  ASSERT_EQ(compute_graph->FindNode("transpose"), nullptr);
  ASSERT_EQ(schedule_tasks[0].score_func,
            "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
            "  return -1;\n"
            "}\n");
}

TEST_F(TestOptimizer, PowEqiv) {
  auto graph = af::testing::AscGraphBuilder("PowBrc")
                   .Loops({af::testing::Sym(32)})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Scalar("scalar0", "-1")
                   .template Op<af::ascir_op::Pow>("pow0", {"load0", "scalar0"})
                   .Load("load1", "data0")
                   .Scalar("scalar1", "-2")
                   .template Op<af::ascir_op::Pow>("pow1", {"load1", "scalar1"})
                   .Load("load2", "data0")
                   .Scalar("scalar2", "-0.5")
                   .template Op<af::ascir_op::Pow>("pow2", {"load2", "scalar2"})
                   .Scalar("scalar3", "3")
                   .template Op<af::ascir_op::Pow>("pow3", {"scalar3", "scalar3"})
                   .Load("load4", "data0")
                   .Scalar("scalar4", "4")
                   .template Op<af::ascir_op::Pow>("pow4", {"load4", "scalar4"})
                   .Add("add0", "pow0", "pow1")
                   .Add("add1", "pow2", "pow3")
                   .Add("add2", "add0", "add1")
                   .Add("add3", "add2", "pow4")
                   .Store("store", "add3")
                   .Output("y", "store", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::utils::DumpGraph(graph, "BEFORE");
  Status res = optimize::autoschedule::PassRunnerHandler().RunPasses(graph);
  EXPECT_EQ(res, af::SUCCESS);
  ::ascir::utils::DumpGraph(graph, "AFTER");
  auto pow0_node = graph.FindNode("pow0");
  EXPECT_EQ(pow0_node, nullptr);
  auto pow1_node = graph.FindNode("pow1");
  EXPECT_EQ(pow1_node, nullptr);
  auto pow2_node = graph.FindNode("pow2");
  EXPECT_EQ(pow2_node, nullptr);
  auto pow3_node = graph.FindNode("pow3");
  EXPECT_EQ(pow3_node, nullptr);
  auto pow4_node = graph.FindNode("pow4");
  EXPECT_EQ(pow4_node, nullptr);
  auto cg = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(cg->GetAllNodesSize(), 25UL);
}

TEST_F(TestOptimizer, PowEqivCase2) {
  auto graph = af::testing::AscGraphBuilder("PowBrc")
                   .Loops({af::testing::Sym(32)})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .template Op<af::ascir_op::Pow>("pow0", {"load0", "load0"})
                   .Data("data1", 1)
                   .Load("load1", "data1")
                   .Scalar("scalar1", "0")
                   .template Op<af::ascir_op::Pow>("pow1", {"load1", "scalar1"})
                   .Add("add0", "pow0", "pow1")
                   .Store("store", "add0")
                   .Output("y", "store", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::utils::DumpGraph(graph, "BEFORE");
  Status res = optimize::autoschedule::PassRunnerHandler().RunPasses(graph);
  EXPECT_EQ(res, af::SUCCESS);
  ::ascir::utils::DumpGraph(graph, "AFTER");
  auto pow0_node = graph.FindNode("pow0");
  EXPECT_NE(pow0_node, nullptr);
  auto pow1_node = graph.FindNode("pow1");
  EXPECT_EQ(pow1_node, nullptr);
  auto data1_node = graph.FindNode("data1");
  EXPECT_NE(data1_node, nullptr);
  auto cg = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(cg->GetAllNodesSize(), 8UL);
}

TEST_F(TestOptimizer, SkipPruneGraph) {
  auto graph = af::testing::AscGraphBuilder("PowBrc")
                   .Loops({af::testing::Sym(32)})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .template Op<af::ascir_op::Pow>("pow0", {"load0", "load0"})
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  ::ascir::utils::DumpGraph(graph, "BEFORE");
  Status res = optimize::PassUtils::PruneGraph(graph);
  EXPECT_EQ(res, af::SUCCESS);
  ::ascir::utils::DumpGraph(graph, "After");
  auto cg = af::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(cg->GetAllNodesSize(), 3UL);
}

TEST_F(TestOptimizer, OptimizeRemoveDanglingNodes) {
  auto graph = af::testing::AscGraphBuilder("RemoveDanglingNodes")
                   .Loops({af::testing::Sym(32)})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Store("store", "abs0")
                   .Output("output", "store", 0)
                   .Data("data1", 1)
                   .Load("load1", "data1")
                   .Abs("dangling_abs", "load1")
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto cg = af::AscGraphUtils::GetComputeGraph(graph);
  ASSERT_NE(cg, nullptr);
  ASSERT_NE(graph.FindNode("data1"), nullptr);
  ASSERT_NE(graph.FindNode("load1"), nullptr);
  ASSERT_NE(graph.FindNode("dangling_abs"), nullptr);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);

  auto data1_node = graph.FindNode("data1");
  auto load1_node = graph.FindNode("load1");
  auto dangling_abs_node = graph.FindNode("dangling_abs");
  auto output_node = graph.FindNode("output");
  ASSERT_NE(data1_node, nullptr);
  ASSERT_NE(output_node, nullptr);
  EXPECT_EQ(load1_node, nullptr);
  EXPECT_EQ(dangling_abs_node, nullptr);
  EXPECT_EQ(data1_node->GetOutDataNodesSize(), 0U);
  EXPECT_EQ(data1_node->GetOutControlNodesSize(), 1U);
  EXPECT_EQ(output_node->GetInControlNodesSize(), 1U);
}

TEST_F(TestOptimizer, OptimizeRemoveDanglingNodesWithWorkspaceOutput) {
  auto graph = af::testing::AscGraphBuilder("RemoveDanglingNodesWithWorkspaceOutput")
                   .Loops({af::testing::Sym(32)})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Workspace("workspace", "abs0")
                   .ScalarData("scalar_data", 1)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto workspace_node = graph.FindNode("workspace");
  auto scalar_data_node = graph.FindNode("scalar_data");
  ASSERT_NE(workspace_node, nullptr);
  ASSERT_NE(scalar_data_node, nullptr);
  EXPECT_EQ(workspace_node->GetOutDataNodesSize(), 0U);
  EXPECT_EQ(scalar_data_node->GetOutControlNodesSize(), 0U);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);

  workspace_node = graph.FindNode("workspace");
  scalar_data_node = graph.FindNode("scalar_data");
  ASSERT_NE(workspace_node, nullptr);
  ASSERT_NE(scalar_data_node, nullptr);
  EXPECT_EQ(scalar_data_node->GetOutDataNodesSize(), 0U);
  EXPECT_EQ(scalar_data_node->GetOutControlNodesSize(), 1U);
  EXPECT_EQ(workspace_node->GetInControlNodesSize(), 1U);
}

TEST_F(TestOptimizer, TransposeWithUB) {
  // Transpose 尾轴为动态shape
  af::AscGraph graph("transpose_with_ub");
  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(86);
  auto s2 = graph.CreateSizeVar(200);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);
  Data x3_op("x3", graph);
  x3_op.ir_attr.SetIndex(2);

  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");

  std::vector<Data> all_data{x1_op, x2_op, x3_op};
  std::vector<Load> all_load{load_op1, load_op2, load_op3};

  for (size_t i = 0U; i < all_data.size(); ++i) {
    auto &x_op = all_data[i];
    auto &load_op = all_load[i];
    x_op.y.dtype = ge::DT_FLOAT16;
    x_op.attr.api.type = af::ApiType::kAPITypeBuffer;
    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.axis = {z0.id, z1.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.repeats = {s0, s1, s2};
    *load_op.y.strides = {s1 * s2, s2, af::ops::One};
  }
  load_op1.attr.sched.axis = {z1.id, z0.id, z2.id};
  *load_op1.y.axis = {z1.id, z0.id, z2.id};
  *load_op1.y.repeats = {s1, s0, s2};
  *load_op1.y.strides = {s0 * s2, s2, af::ops::One};

  af::ascir_op::Transpose transpose_op("transpose");
  transpose_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  transpose_op.x = load_op1.y;
  transpose_op.y.dtype = ge::DT_FLOAT16;
  *transpose_op.y.axis = {z0.id, z1.id, z2.id};
  *transpose_op.y.strides = {s1 * s2, s2, af::ops::One};
  *transpose_op.y.repeats = {s0, s1, s2};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = load_op2.y;
  add_op.x2 = load_op3.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1 * s2, s2, af::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  af::ascir_op::Mul mul_op("mul");
  mul_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul_op.x1 = transpose_op.y;
  mul_op.x2 = add_op.y;
  mul_op.y.dtype = ge::DT_FLOAT16;
  *mul_op.y.axis = {z0.id, z1.id, z2.id};
  *mul_op.y.strides = {s1 * s2, s2, af::ops::One};
  *mul_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};

  store_op.x = mul_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};
  store_op.ir_attr.SetOffset(af::Symbol(0));

  Output y_op("y");
  y_op.x = store_op.y;
  y_op.ir_attr.SetIndex(0);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  std::vector<optimize::ScheduleTask> schedule_tasks;
  optimize::OptimizerOptions options{optimize::GraphType::kAscGraph};
  int res = optimize::ScheduleTaskGenerator::GenerateTasks(graph, schedule_tasks, options);
  ASSERT_EQ(res, 0);
  ASSERT_EQ(schedule_tasks.size(), 2);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(schedule_tasks[1].grouped_graphs[0]);
  ASSERT_EQ(compute_graph->FindNode("transpose"), nullptr);
  ASSERT_EQ(schedule_tasks[0].score_func,
            "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
            "  return 1;\n"
            "}\n");
  SetCurShapeEnvContext(nullptr);
}

TEST_F(TestOptimizer, TransposeWithEnvSetStaticTail) {
  // Transpose 尾轴为动态shape
  af::AscGraph graph("transpose_with_env_set_static_tail");
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(86);
  auto s2 = shape_env.CreateSymbol(200, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);
  Data x3_op("x3", graph);
  x3_op.ir_attr.SetIndex(2);

  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");

  std::vector<Data> all_data{x1_op, x2_op, x3_op};
  std::vector<Load> all_load{load_op1, load_op2, load_op3};

  for (size_t i = 0U; i < all_data.size(); ++i) {
    auto &x_op = all_data[i];
    auto &load_op = all_load[i];
    x_op.y.dtype = ge::DT_FLOAT16;
    x_op.attr.api.type = af::ApiType::kAPITypeBuffer;
    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.axis = {z0.id, z1.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.repeats = {s0, s1, s2};
    *load_op.y.strides = {s1 * s2, s2, af::ops::One};
  }
  load_op1.attr.sched.axis = {z1.id, z0.id, z2.id};
  *load_op1.y.axis = {z1.id, z0.id, z2.id};
  *load_op1.y.repeats = {s1, s0, s2};
  *load_op1.y.strides = {s0 * s2, s2, af::ops::One};

  af::ascir_op::Transpose transpose_op("transpose");
  transpose_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  transpose_op.x = load_op1.y;
  transpose_op.y.dtype = ge::DT_FLOAT16;
  *transpose_op.y.axis = {z0.id, z1.id, z2.id};
  *transpose_op.y.strides = {s1 * s2, s2, af::ops::One};
  *transpose_op.y.repeats = {s0, s1, s2};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = load_op2.y;
  add_op.x2 = load_op3.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1 * s2, s2, af::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  af::ascir_op::Mul mul_op("mul");
  mul_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul_op.x1 = transpose_op.y;
  mul_op.x2 = add_op.y;
  mul_op.y.dtype = ge::DT_FLOAT16;
  *mul_op.y.axis = {z0.id, z1.id, z2.id};
  *mul_op.y.strides = {s1 * s2, s2, af::ops::One};
  *mul_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};

  store_op.x = mul_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};
  store_op.ir_attr.SetOffset(af::Symbol(0));

  Output y_op("y");
  y_op.x = store_op.y;
  y_op.ir_attr.SetIndex(0);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  std::vector<optimize::ScheduleTask> schedule_tasks;
  optimize::OptimizerOptions options{optimize::GraphType::kAscGraph};
  int res = optimize::ScheduleTaskGenerator::GenerateTasks(graph, schedule_tasks, options);
  ASSERT_EQ(res, 0);
  ASSERT_EQ(schedule_tasks.size(), 2);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(schedule_tasks[1].grouped_graphs[0]);
  ASSERT_EQ(compute_graph->FindNode("transpose"), nullptr);
  ASSERT_EQ(schedule_tasks[0].score_func,
            "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
            "  return 1;\n"
            "}\n");
  SetCurShapeEnvContext(nullptr);
}

TEST_F(TestOptimizer, TransposeWithDynamicTail) {
  // Transpose 尾轴为动态shape
  af::AscGraph graph("transpose_with_dynamic_tail");
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(86);
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);
  Data x3_op("x3", graph);
  x3_op.ir_attr.SetIndex(2);

  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");

  std::vector<Data> all_data{x1_op, x2_op, x3_op};
  std::vector<Load> all_load{load_op1, load_op2, load_op3};

  for (size_t i = 0U; i < all_data.size(); ++i) {
    auto &x_op = all_data[i];
    auto &load_op = all_load[i];
    x_op.y.dtype = ge::DT_FLOAT16;
    x_op.attr.api.type = af::ApiType::kAPITypeBuffer;
    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.axis = {z0.id, z1.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.repeats = {s0, s1, s2};
    *load_op.y.strides = {s1 * s2, s2, af::ops::One};
  }
  load_op1.attr.sched.axis = {z1.id, z0.id, z2.id};
  *load_op1.y.axis = {z1.id, z0.id, z2.id};
  *load_op1.y.repeats = {s1, s0, s2};
  *load_op1.y.strides = {s0 * s2, s2, af::ops::One};

  af::ascir_op::Transpose transpose_op("transpose");
  transpose_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  transpose_op.x = load_op1.y;
  transpose_op.y.dtype = ge::DT_FLOAT16;
  *transpose_op.y.axis = {z0.id, z1.id, z2.id};
  *transpose_op.y.strides = {s1 * s2, s2, af::ops::One};
  *transpose_op.y.repeats = {s0, s1, s2};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = load_op2.y;
  add_op.x2 = load_op3.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1 * s2, s2, af::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  af::ascir_op::Mul mul_op("mul");
  mul_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul_op.x1 = transpose_op.y;
  mul_op.x2 = add_op.y;
  mul_op.y.dtype = ge::DT_FLOAT16;
  *mul_op.y.axis = {z0.id, z1.id, z2.id};
  *mul_op.y.strides = {s1 * s2, s2, af::ops::One};
  *mul_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};

  store_op.x = mul_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};
  store_op.ir_attr.SetOffset(af::Symbol(0));

  Output y_op("y");
  y_op.x = store_op.y;
  y_op.ir_attr.SetIndex(0);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  std::vector<optimize::ScheduleTask> schedule_tasks;
  optimize::OptimizerOptions options{optimize::GraphType::kAscGraph};
  int res = optimize::ScheduleTaskGenerator::GenerateTasks(graph, schedule_tasks, options);
  ASSERT_EQ(res, 0);
  ASSERT_EQ(schedule_tasks.size(), 2);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(schedule_tasks[1].grouped_graphs[0]);
  ASSERT_EQ(compute_graph->FindNode("transpose"), nullptr);
  ASSERT_EQ(schedule_tasks[0].score_func,
            "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
            "  return 0;\n"
            "}\n");
  SetCurShapeEnvContext(nullptr);
}

TEST_F(TestOptimizer, AllReduce) {
  af::AscGraph graph("all_reduce");

  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {af::ops::One, af::ops::One};
  *sum.y.strides = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = sum.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.axis = {z0.id, z1.id};
  *store_op1.y.repeats = {af::ops::One, af::ops::One};
  *store_op1.y.strides = {af::ops::Zero, af::ops::Zero};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, 0);
}

TEST_F(TestOptimizer, BufQueAllocator_RemovePad_MemUnique) {
  af::AscGraph graph("BufQueAllocator_RemovePad_MemUnique");

  const af::Expression s0 = graph.CreateSizeVar(320);
  const af::Expression s1 = graph.CreateSizeVar(2889);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x0("x", graph);
  x0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x0.attr.api.type = af::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = ge::DataType::DT_FLOAT;

  af::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.api.type = af::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  load0.y.dtype = ge::DataType::DT_FLOAT;
  load0.attr.api.unit = af::ComputeUnit::kUnitMTE2;

  af::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1.attr.api.type = af::ApiType::kAPITypeBuffer;
  x1.y.dtype = ge::DataType::DT_FLOAT;
  x1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.attr.api.type = af::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};
  load1.y.dtype = ge::DataType::DT_FLOAT;
  load1.attr.api.unit = af::ComputeUnit::kUnitMTE2;

  af::ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = load1.y;
  broadcast1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast1.attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  *broadcast1.y.repeats = {s0, s1};
  *broadcast1.y.strides = {s1, One};
  broadcast1.y.dtype = ge::DataType::DT_FLOAT;
  broadcast1.attr.api.unit = af::ComputeUnit::kUnitVector;

  af::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = broadcast1.y;
  add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add0.attr.api.type = af::ApiType::kAPITypeCompute;
  add0.attr.sched.axis = {z0.id, z1.id};
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};
  add0.y.dtype = ge::DataType::DT_FLOAT;
  add0.attr.api.unit = af::ComputeUnit::kUnitVector;

  af::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x2.attr.api.type = af::ApiType::kAPITypeBuffer;
  x2.ir_attr.SetIndex(2);
  x2.y.dtype = ge::DataType::DT_FLOAT;

  af::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2.attr.api.type = af::ApiType::kAPITypeCompute;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, One};
  load2.y.dtype = ge::DataType::DT_FLOAT;
  load2.attr.api.unit = af::ComputeUnit::kUnitMTE2;

  af::ascir_op::Mul mul0("mul0");
  mul0.x1 = load2.y;
  mul0.x2 = add0.y;
  mul0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  mul0.attr.api.type = af::ApiType::kAPITypeCompute;
  mul0.attr.sched.axis = {z0.id, z1.id};
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};
  mul0.y.dtype = ge::DataType::DT_FLOAT;
  mul0.attr.api.unit = af::ComputeUnit::kUnitVector;

  af::ascir_op::Store store("store");
  store.x = mul0.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.api.type = af::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.y.dtype = ge::DataType::DT_FLOAT;
  store.attr.api.unit = af::ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 3UL);

  auto impl_graph2 = af::AscGraphUtils::GetComputeGraph(
      fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1]);
  EXPECT_EQ(impl_graph2->GetAllNodesSize(), 12);
  EXPECT_NE(impl_graph2->FindNode("broadcast1"), nullptr);
  EXPECT_NE(impl_graph2->FindNode("broadcast1_remove_pad_0"), nullptr);
  EXPECT_NE(impl_graph2->FindNode("add0"), nullptr);
  const auto &impl_graph2_mul0 = std::dynamic_pointer_cast<af::AscNode>(impl_graph2->FindNode("mul0"));
  const auto &impl_graph2_add0 = std::dynamic_pointer_cast<af::AscNode>(impl_graph2->FindNode("add0"));
  EXPECT_EQ(impl_graph2_add0->outputs[0].attr.que.id, impl_graph2_mul0->outputs[0].attr.que.id);
}

TEST_F(TestOptimizer, GatherLastAxisTest) {
  af::AscGraph graph("gather_last_axis");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = One;

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.repeats = {s0};
  *(x1.y.axis) = {z0.id};

  Data x2("x2", graph);
  x2.attr.sched.axis = {z1.id, z2.id};
  x2.y.dtype = ge::DT_INT64;
  *x2.y.repeats = {s1, s2};
  *(x2.y.axis) = {z1.id, z2.id};

  Gather gather("gather");
  gather.attr.api.compute_type = af::ComputeType::kComputeGather;
  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.sched.axis = {z1.id, z2.id};
  gather.ir_attr.SetAxis(0);
  gather.y.dtype = ge::DT_FLOAT;
  *gather.y.axis = {z1.id, z2.id};
  *gather.y.repeats = {s1, s2};
  *gather.y.strides = {One, Zero};
  gather.attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = gather.y;
  abs.attr.sched.axis = {z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.repeats = {s1, s2};
  *abs.y.strides = {One, Zero};

  Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z1.id, z2.id};
  *store.y.repeats = {s1, s2};
  *store.y.strides = {One, Zero};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z1.id, z2.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z1.id, z2.id};

  ASSERT_EQ(optimize::Optimizer::RemoveAllZeroStrideLoopAxis(graph), af::SUCCESS);
  auto gather_node = graph.FindNode("gather");
  ASSERT_NE(gather_node, nullptr);
  EXPECT_EQ(gather_node->attr.sched.axis.size(), 2UL);
  EXPECT_EQ(gather_node->outputs[0].attr.axis.size(), 2UL);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  std::vector<optimize::ScheduleTask> schedule_tasks;
  optimize::OptimizerOptions options{optimize::GraphType::kAscGraph};
  int res = optimize::ScheduleTaskGenerator::GenerateTasks(graph, schedule_tasks, options);
  ASSERT_EQ(res, 0);
  ASSERT_EQ(schedule_tasks.size(), 1);
}

TEST_F(TestOptimizer, MergeGroupYAndR) {
  af::AscGraph graph("merge_group_y_and_r");

  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.strides = {af::ops::Zero, af::ops::Zero};
  *sum.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.strides = {af::ops::Zero, af::ops::Zero};
  *abs.y.repeats = {af::ops::One, af::ops::One};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = sum.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.axis = {z0.id, z1.id};
  *store_op1.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op1.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::autoschedule::AxisGroup axes_group;
  EXPECT_EQ(optimize::autoschedule::TilingGroup::GenTilingGroup(graph, axes_group), 0);

  optimize::autoschedule::AxisGroup cur_y_group;
  cur_y_group.y_group = {z0.id, z1.id};
  optimize::autoschedule::AxisGroup cur_r_group;
  cur_r_group.r_group = {z0.id, z1.id};
  EXPECT_TRUE(optimize::autoschedule::TilingGroup::MergeAxesGroup(cur_y_group, cur_r_group, true));
  EXPECT_FALSE(optimize::autoschedule::TilingGroup::MergeAxesGroup(cur_r_group, cur_y_group, true));
}

TEST_F(TestOptimizer, ConcatTailDimStatic) {
  af::AscGraph graph("concat_last_dim_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(412);
  auto s2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(32);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1 + s2 + s3);

  std::vector<af::Expression> input_dims{s1, s2, s3};

  Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);
  Data x3_op("x3", graph);
  x3_op.ir_attr.SetIndex(2);

  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");

  std::vector<Data> all_data{x1_op, x2_op, x3_op};
  std::vector<Load> all_load{load_op1, load_op2, load_op3};

  for (size_t i = 0U; i < all_data.size(); ++i) {
    auto &x_op = all_data[i];
    auto &load_op = all_load[i];
    x_op.y.dtype = ge::DT_FLOAT16;
    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.axis = {z0.id, z1.id};
    load_op.y.dtype = ge::DT_FLOAT16;
    *load_op.y.repeats = {s0, input_dims[i]};
    *load_op.y.strides = {input_dims[i], af::ops::One};
  }

  af::ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {load_op1.y, load_op2.y, load_op3.y};
  concat_op.y.dtype = ge::DT_FLOAT16;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1 + s2};
  *concat_op.y.strides = {s1 + s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};

  store_op.x = concat_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1 + s2};
  *store_op.y.strides = {s1 + s2, af::ops::One};
  store_op.ir_attr.SetOffset(af::Symbol(0));

  Output y_op("y");
  y_op.x = store_op.y;
  y_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(res, 0);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1);
  auto &schedule_result = fused_scheduled_result.node_idx_to_scheduled_results[0][0];

  std::vector<af::Expression> offsets;
  std::vector<af::Expression> expect = {af::Symbol(0), s1};
  for (const auto &schedule_group : schedule_result.schedule_groups) {
    auto &sub_impl_graph = schedule_group.impl_graphs.front();
    for (const auto &sub_node : sub_impl_graph.GetAllNodes()) {
      if (sub_node->GetType() == "Store") {
        af::Expression offset;
        EXPECT_EQ(sub_node->attr.ir_attr->GetAttrValue("offset", offset), 0);
        offsets.emplace_back(offset);
      }
    }
  }
  for (size_t i = 0; i < offsets.size(); ++i) {
    EXPECT_SYMBOL_EQ(offsets[i], expect[i]);
  }
  EXPECT_EQ(fused_scheduled_result.input_nodes.size(), 3);
  EXPECT_EQ(fused_scheduled_result.output_nodes.size(), 1);
  EXPECT_EQ(fused_scheduled_result.workspace_nodes.size(), 0);
  EXPECT_EQ(fused_scheduled_result.input_nodes[0]->GetName(), "x1");
  EXPECT_EQ(fused_scheduled_result.input_nodes[1]->GetName(), "x2");
  EXPECT_EQ(fused_scheduled_result.input_nodes[2]->GetName(), "x3");
  EXPECT_EQ(fused_scheduled_result.output_nodes[0]->GetName(), "y");
}

TEST_F(TestOptimizer, ReducePartition) {
  af::AscGraph graph("reduce_partition");

  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Exp exp("exp");
  exp.x = load.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  *exp.y.axis = {z0.id, z1.id};
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.strides = {s1, af::ops::One};
  *exp.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = exp.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.strides = {af::ops::One, af::ops::One};
  *sum.y.repeats = {af::ops::Zero, af::ops::Zero};

  Broadcast broadcast("broadcast");
  broadcast.x = sum.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.strides = {s1, af::ops::One};
  *broadcast.y.repeats = {s0, s1};

  Sub sub("sub");
  sub.x1 = broadcast.y;
  sub.x2 = exp.y;
  sub.attr.sched.axis = {z0.id, z1.id};
  *sub.y.axis = {z0.id, z1.id};
  sub.y.dtype = ge::DT_FLOAT;
  *sub.y.strides = {s1, af::ops::One};
  *sub.y.repeats = {s0, s1};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = sub.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {s1, af::ops::One};
  *store_op1.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2);
}

TEST_F(TestOptimizer, ReducePartitionLoad) {
  af::AscGraph graph("Reduce_partition_load");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0 * s1 * s2);
  auto z1 = graph.CreateAxis("z1", s3);

  auto axis = {z0.id, z1.id};

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  arg4_1.y.dtype = ge::DT_FLOAT16;
  arg4_1.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = axis;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0 * s1 * s2, s3};
  *b0_load.y.strides = {s3, One};

  af::ascir_op::Max b0_max("b0_max");
  b0_max.x = b0_load.y;
  b0_max.attr.sched.axis = axis;
  b0_max.y.dtype = ge::DT_FLOAT16;
  *b0_max.y.axis = axis;
  *b0_max.y.repeats = {s0 * s1 * s2, One};
  *b0_max.y.strides = {One, Zero};

  Broadcast b1_broadcast("b1_broadcast");
  b1_broadcast.x = b0_max.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0 * s1 * s2, s3};
  *b1_broadcast.y.strides = {s3, One};

  af::ascir_op::Sub b1_sub("b1_sub");
  b1_sub.x1 = b0_load.y;
  b1_sub.x2 = b1_broadcast.y;
  b1_sub.attr.sched.axis = axis;
  b1_sub.y.dtype = ge::DT_FLOAT16;
  *b1_sub.y.axis = axis;
  *b1_sub.y.repeats = {s0 * s1 * s2, s3};
  *b1_sub.y.strides = {s3, One};

  Exp b1_exp("b1_exp");
  b1_exp.x = b1_sub.y;
  b1_exp.attr.sched.axis = axis;
  b1_exp.y.dtype = ge::DT_FLOAT16;
  *b1_exp.y.axis = axis;
  *b1_exp.y.repeats = {s0 * s1 * s2, s3};
  *b1_exp.y.strides = {s3, One};

  Sum b2_sum("b2_sum");
  b2_sum.x = b1_exp.y;
  b2_sum.attr.sched.axis = axis;
  b2_sum.y.dtype = ge::DT_FLOAT16;
  *b2_sum.y.axis = axis;
  *b2_sum.y.repeats = {s0 * s1 * s2, One};
  *b2_sum.y.strides = {One, Zero};

  Output buf3("buf3");
  buf3.ir_attr.SetIndex(2);

  Broadcast b3_broadcast("b3_broadcast");
  b3_broadcast.x = b2_sum.y;
  b3_broadcast.attr.sched.axis = axis;
  b3_broadcast.y.dtype = ge::DT_FLOAT16;
  *b3_broadcast.y.axis = axis;
  *b3_broadcast.y.repeats = {s0 * s1 * s2, s3};
  *b3_broadcast.y.strides = {s3, One};

  af::ascir_op::Div b3_div("b3_div");
  b3_div.x1 = b1_exp.y;
  b3_div.x2 = b3_broadcast.y;
  b3_div.attr.sched.axis = axis;
  b3_div.y.dtype = ge::DT_FLOAT16;
  *b3_div.y.axis = axis;
  *b3_div.y.repeats = {s0 * s1 * s2, s3};
  *b3_div.y.strides = {s3, One};

  Store b3_store("b3_store");
  b3_store.x = b3_div.y;
  b3_store.attr.sched.axis = axis;
  b3_store.y.dtype = ge::DT_FLOAT16;
  *b3_store.y.axis = axis;
  *b3_store.y.repeats = {s0 * s1 * s2, s3};
  *b3_store.y.strides = {s3, One};

  buf3.x = b3_store.y;
  buf3.y.dtype = ge::DT_FLOAT16;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2);
}

TEST_F(TestOptimizer, ReducePartition3) {
  af::AscGraph graph("reduce_partition");

  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Exp exp("exp");
  exp.x = load.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  *exp.y.axis = {z0.id, z1.id};
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.strides = {s1, af::ops::One};
  *exp.y.repeats = {s0, s1};

  Abs abs("abs");
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.x = exp.y;
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {s1, af::ops::One};
  *abs.y.repeats = {s0, s1};

  Sub sub("sub");
  sub.x1 = abs.y;
  sub.x2 = exp.y;
  sub.attr.sched.axis = {z0.id, z1.id};
  *sub.y.axis = {z0.id, z1.id};
  sub.y.dtype = ge::DT_FLOAT;
  *sub.y.strides = {s1, af::ops::One};
  *sub.y.repeats = {s0, s1};

  Sum sum("b2_sum");
  sum.x = sub.y;
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT16;
  *sum.y.axis = {z0.id, z1.id};
  *sum.y.strides = {af::ops::One, af::ops::One};
  *sum.y.repeats = {af::ops::Zero, af::ops::Zero};

  Cast cast("cast");
  cast.x = sum.y;
  cast.attr.sched.axis = {z0.id, z1.id};
  cast.y.dtype = ge::DT_FLOAT;
  *cast.y.axis = {z0.id, z1.id};
  *cast.y.strides = {af::ops::One, af::ops::One};
  *cast.y.repeats = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = cast.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {af::ops::One, af::ops::One};
  *store_op1.y.repeats = {af::ops::Zero, af::ops::Zero};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 4);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][2].schedule_groups.size(), 2);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][3].schedule_groups.size(), 1);

  auto impl_graph_phase1 = fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups[0].impl_graphs[0];
  auto impl_graph_phase2 = fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups[1].impl_graphs[0];
  auto phase1_workspace1 = impl_graph_phase1.FindNode("reduce_partition_0_r_multicore_phase_2_graph_workspace");
  auto phase2_workspace1 = impl_graph_phase2.FindNode("reduce_partition_0_r_multicore_phase_2_graph_workspace");
  ASSERT_NE(phase1_workspace1, nullptr);
  ASSERT_NE(phase2_workspace1, nullptr);
}

TEST_F(TestOptimizer, ReducePartition4) {
  af::AscGraph graph("reduce_partition");

  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Exp exp("exp");
  exp.x = load.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  *exp.y.axis = {z0.id, z1.id};
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.strides = {s1, af::ops::One};
  *exp.y.repeats = {s0, s1};

  Abs abs("abs");
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.x = exp.y;
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {s1, af::ops::One};
  *abs.y.repeats = {s0, s1};

  Sub sub("sub");
  sub.x1 = abs.y;
  sub.x2 = exp.y;
  sub.attr.sched.axis = {z0.id, z1.id};
  *sub.y.axis = {z0.id, z1.id};
  sub.y.dtype = ge::DT_FLOAT;
  *sub.y.strides = {s1, af::ops::One};
  *sub.y.repeats = {s0, s1};

  Sum sum("b2_sum");
  sum.x = sub.y;
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT16;
  *sum.y.axis = {z0.id, z1.id};
  *sum.y.strides = {af::ops::One, af::ops::One};
  *sum.y.repeats = {af::ops::Zero, af::ops::Zero};

  Cast cast("cast");
  cast.x = sum.y;
  cast.attr.sched.axis = {z0.id, z1.id};
  cast.y.dtype = ge::DT_FLOAT;
  *cast.y.axis = {z0.id, z1.id};
  *cast.y.strides = {af::ops::One, af::ops::One};
  *cast.y.repeats = {af::ops::Zero, af::ops::Zero};

  Broadcast broadcast("broadcast");
  broadcast.x = cast.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.strides = {s1, af::ops::One};
  *broadcast.y.repeats = {s0, s1};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = broadcast.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {s1, af::ops::One};
  *store_op1.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 4);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 3);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][2].schedule_groups.size(), 3);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][3].schedule_groups.size(), 1);

  auto impl_graph_phase1 = fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups[0].impl_graphs[0];
  auto impl_graph_phase2 = fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups[1].impl_graphs[0];
  auto impl_graph_brc = fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups[2].impl_graphs[0];
  auto phase1_workspace = impl_graph_phase1.FindNode("reduce_partition_0_r_multicore_phase_2_graph_workspace");
  auto phase2_workspace1 = impl_graph_phase2.FindNode("reduce_partition_0_r_multicore_phase_2_graph_workspace");
  auto phase2_workspace2 = impl_graph_phase2.FindNode("b2_sum_Workspace");
  auto brc_workspace = impl_graph_brc.FindNode("b2_sum_Workspace");
  ASSERT_NE(phase1_workspace, nullptr);
  ASSERT_NE(phase2_workspace1, nullptr);
  ASSERT_NE(phase2_workspace2, nullptr);
  ASSERT_NE(brc_workspace, nullptr);
}

TEST_F(TestOptimizer, ReduceRMulticore) {
  af::AscGraph graph("reduce_r_multicore");

  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Exp exp("exp");
  exp.x = load.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  *exp.y.axis = {z0.id, z1.id};
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.strides = {s1, af::ops::One};
  *exp.y.repeats = {s0, s1};

  Abs abs("abs");
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.x = exp.y;
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {s1, af::ops::One};
  *abs.y.repeats = {s0, s1};

  Sum sum("b2_sum");
  sum.x = abs.y;
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT16;
  *sum.y.axis = {z0.id, z1.id};
  *sum.y.strides = {af::ops::One, af::ops::One};
  *sum.y.repeats = {af::ops::Zero, af::ops::Zero};

  Cast cast("cast");
  cast.x = sum.y;
  cast.attr.sched.axis = {z0.id, z1.id};
  cast.y.dtype = ge::DT_FLOAT;
  *cast.y.axis = {z0.id, z1.id};
  *cast.y.strides = {af::ops::One, af::ops::One};
  *cast.y.repeats = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = cast.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {af::ops::One, af::ops::One};
  *store_op1.y.repeats = {af::ops::Zero, af::ops::Zero};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 4);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][2].schedule_groups.size(), 2);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][3].schedule_groups.size(), 1);

  auto impl_graph_phase1 = fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups[0].impl_graphs[0];
  auto impl_graph_phase2 = fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups[1].impl_graphs[0];
  auto phase1_workspace = impl_graph_phase1.FindNode("reduce_r_multicore_0_r_multicore_phase_2_graph_workspace");
  auto phase2_workspace1 = impl_graph_phase2.FindNode("reduce_r_multicore_0_r_multicore_phase_2_graph_workspace");
  ASSERT_NE(phase1_workspace, nullptr);
  ASSERT_NE(phase2_workspace1, nullptr);
}

TEST_F(TestOptimizer, ReducePartitionScalar) {
  af::AscGraph graph("reduce_partition_scalar");

  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Scalar scalar("scalar", graph);
  scalar.y.dtype = ge::DT_FLOAT;

  Workspace workspace("workspace");
  graph.AddNode(workspace);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = workspace.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Add add("add");
  add.x1 = scalar.y;
  add.x2 = load.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.strides = {s1, af::ops::One};
  *add.y.repeats = {s0, s1};

  Mean mean("b2_mean");
  mean.x = add.y;
  mean.attr.sched.axis = {z0.id, z1.id};
  mean.y.dtype = ge::DT_FLOAT;
  *mean.y.axis = {z0.id, z1.id};
  *mean.y.strides = {af::ops::One, af::ops::One};
  *mean.y.repeats = {af::ops::Zero, af::ops::Zero};

  Broadcast broadcast("broadcast");
  broadcast.x = mean.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.strides = {s1, af::ops::One};
  *broadcast.y.repeats = {s0, s1};

  Sub sub("sub");
  sub.x1 = load.y;
  sub.x2 = broadcast.y;
  sub.attr.sched.axis = {z0.id, z1.id};
  *sub.y.axis = {z0.id, z1.id};
  sub.y.dtype = ge::DT_FLOAT;
  *sub.y.strides = {s1, af::ops::One};
  *sub.y.repeats = {s0, s1};

  Add add1("add1");
  add1.x1 = sub.y;
  add1.x2 = scalar.y;
  add1.attr.sched.axis = {z0.id, z1.id};
  *add1.y.axis = {z0.id, z1.id};
  add1.y.dtype = ge::DT_FLOAT;
  *add1.y.strides = {s1, af::ops::One};
  *add1.y.repeats = {s0, s1};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = add1.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {af::ops::One, af::ops::One};
  *store_op1.y.repeats = {af::ops::Zero, af::ops::Zero};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2);
}

TEST_F(TestOptimizer, ReduceAllLoad) {
  af::AscGraph graph("reduce_all_load");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = af::ApiType::kAPITypeBuffer;
  arg4_1.y.dtype = ge::DT_FLOAT;
  arg4_1.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = {z0.id, z1.id, z2.id};
  b0_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT;
  *b0_load.y.axis = {z0.id, z1.id, z2.id};
  *b0_load.y.repeats = {s0, s1, s2};
  *b0_load.y.strides = {s1 * s2, s2, One};

  Abs abs("abs");
  abs.x = b0_load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1 * s2, s2, One};

  Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul.x1 = abs.y;
  mul.x2 = abs.y;
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id, z2.id};
  *mul.y.repeats = {s0, s1, s2};
  *mul.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Max b0_max("b0_max");
  b0_max.x = mul.y;
  b0_max.attr.sched.axis = {z0.id, z1.id, z2.id};
  b0_max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT;
  *b0_max.y.axis = {z0.id, z1.id, z2.id};
  *b0_max.y.repeats = {s0, One, s2};
  *b0_max.y.strides = {s2, Zero, One};

  Store b3_store("b3_store");
  b3_store.x = b0_max.y;
  b3_store.attr.sched.axis = {z0.id, z1.id, z2.id};
  b3_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b3_store.y.dtype = ge::DT_FLOAT;
  *b3_store.y.axis = {z0.id, z1.id, z2.id};
  *b3_store.y.repeats = {s0, One, s2};
  *b3_store.y.strides = {s2, Zero, One};

  Output buf3("buf3");
  buf3.x = b3_store.y;
  buf3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf3.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf3.y.dtype = ge::DT_FLOAT;
  buf3.ir_attr.SetIndex(0);

  af::ascir_op::Sum b0_sum("b0_sum");
  b0_sum.x = abs.y;
  b0_sum.attr.sched.axis = {z0.id, z1.id, z2.id};
  b0_sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b0_sum.y.dtype = ge::DT_FLOAT;
  *b0_sum.y.axis = {z0.id, z1.id, z2.id};
  *b0_sum.y.repeats = {s0, One, s2};
  *b0_sum.y.strides = {s2, Zero, One};

  Store b4_store("b4_store");
  b4_store.x = b0_sum.y;
  b4_store.attr.sched.axis = {z0.id, z1.id, z2.id};
  b4_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b4_store.y.dtype = ge::DT_FLOAT;
  *b4_store.y.axis = {z0.id, z1.id, z2.id};
  *b4_store.y.repeats = {s0, One, s2};
  *b4_store.y.strides = {s2, Zero, One};

  Output buf4("buf4");
  buf4.x = b4_store.y;
  buf4.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf4.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf4.y.dtype = ge::DT_FLOAT;
  buf4.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2);
}

/**
 *    data0          data1
 *      |              |
 *    load0          load1
 *      |              |
 *     abs         broadcast
 *       \            /
 *        \          /
 *             add
 *              |
 *            store
 *              |
 *           output
 */
TEST_F(TestOptimizer, LoadOpSequenceAdjustCase1) {
  af::AscGraph graph("reorder_load_op");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load0.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, One};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(3);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.strides = {af::ops::One, af::ops::One};
  *load1.y.repeats = {af::ops::One, af::ops::One};

  Broadcast broadcast("broadcast");
  broadcast.x = load1.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.strides = {s1, af::ops::One};
  *broadcast.y.repeats = {s0, s1};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = abs.y;
  add_op.x2 = broadcast.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, af::ops::One};
  *add_op.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Data");
    }
  }

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Data");
    }
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Abs");
    }
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Load");
    }
    // broadcast应该被schedule优化掉了
    if (node->GetOpDesc()->GetId() == 5) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(TestOptimizer, LoadOpSequenceAdjustCase2) {
  af::AscGraph graph("reorder_load_op");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load0.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, One};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.attr.sched.axis = {z0.id, z1.id};
  scalar0.ir_attr.SetValue("0");
  scalar0.y.dtype = ge::DT_FLOAT;
  *scalar0.y.axis = {z0.id, z1.id};
  *scalar0.y.repeats = {One, One};
  *scalar0.y.strides = {Zero, Zero};

  Broadcast broadcast0("broadcast0");
  broadcast0.x = scalar0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = ge::DT_FLOAT;
  *broadcast0.y.repeats = {One, s1};
  *broadcast0.y.strides = {Zero, One};

  Broadcast broadcast1("broadcast1");
  broadcast1.x = broadcast0.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1};
  *broadcast1.y.strides = {s1, af::ops::One};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = abs.y;
  add_op.x2 = broadcast1.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.repeats = {s0, s1};
  *add_op.y.strides = {s1, af::ops::One};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Broadcast broadcast2("broadcast2");
  broadcast2.x = load1.y;
  broadcast2.attr.sched.axis = {z0.id, z1.id};
  *broadcast2.y.axis = {z0.id, z1.id};
  broadcast2.y.dtype = ge::DT_FLOAT;
  *broadcast2.y.repeats = {One, s1};
  *broadcast2.y.strides = {Zero, One};

  Broadcast broadcast3("broadcast3");
  broadcast3.x = broadcast2.y;
  broadcast3.attr.sched.axis = {z0.id, z1.id};
  *broadcast3.y.axis = {z0.id, z1.id};
  broadcast3.y.dtype = ge::DT_FLOAT;
  *broadcast3.y.repeats = {s0, s1};
  *broadcast3.y.strides = {s1, One};

  Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.x1 = add_op.y;
  mul.x2 = broadcast3.y;
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Data");
    }
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Abs");
    }
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Scalar");
    }
    if (node->GetOpDesc()->GetId() == 5) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Load");
    }
    if (node->GetOpDesc()->GetId() == 7) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Mul");
    }
  }
}

TEST_F(TestOptimizer, platform_reg_test) {
  af::AscGraph graph("tmp");
  std::string platform_str;
  ge::PlatformContext::GetInstance().GetCurrentPlatformString(platform_str);
  EXPECT_EQ(platform_str, "2201");
  auto platform_v1 = optimize::PlatformFactory::GetInstance().GetPlatform();
  EXPECT_NE(platform_v1, nullptr);
  auto platform_v1_new = optimize::PlatformFactory::GetInstance().GetPlatform();
  EXPECT_EQ(platform_v1, platform_v1_new);

  EXPECT_EQ(platform_v1->PartitionSubFunctions(graph), af::SUCCESS);

  ge::PlatformContext::GetInstance().SetPlatform("fake");
  auto platform_fake = optimize::PlatformFactory::GetInstance().GetPlatform();
  EXPECT_EQ(platform_fake, nullptr);
}

TEST_F(TestOptimizer, platform_config_test) {
  // "2201" -> PlatformV1, is_default_enabled = false
  std::string platform_str;
  ge::PlatformContext::GetInstance().GetCurrentPlatformString(platform_str);
  EXPECT_EQ(platform_str, "2201");
  auto platform_v1 = optimize::PlatformFactory::GetInstance().GetPlatform();
  ASSERT_NE(platform_v1, nullptr);
  const auto &config_v1 = platform_v1->GetPlatformConfig();
  EXPECT_FALSE(config_v1.is_default_enabled);
  EXPECT_FALSE(config_v1.is_support_compat_mode);
  EXPECT_EQ(config_v1.max_que_num, 4UL);

  // "3510" -> PlatformV2, is_default_enabled = true
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  auto platform_3510 = optimize::PlatformFactory::GetInstance().GetPlatform();
  ASSERT_NE(platform_3510, nullptr);
  const auto &config_3510 = platform_3510->GetPlatformConfig();
  EXPECT_TRUE(config_3510.is_default_enabled);
  EXPECT_TRUE(config_3510.is_support_compat_mode);
  EXPECT_EQ(config_3510.max_que_num, 14UL);

  // "5102" -> PlatformV2, is_default_enabled = false
  ge::PlatformContext::GetInstance().SetPlatform("5102");
  auto platform_5102 = optimize::PlatformFactory::GetInstance().GetPlatform();
  ASSERT_NE(platform_5102, nullptr);
  const auto &config_5102 = platform_5102->GetPlatformConfig();
  EXPECT_FALSE(config_5102.is_default_enabled);
  EXPECT_TRUE(config_5102.is_support_compat_mode);
  EXPECT_EQ(config_5102.max_que_num, 14UL);
}

TEST_F(TestOptimizer, BackendSpec) {
  auto spec = optimize::BackendSpec::GetInstance();
  ASSERT_TRUE(spec != nullptr);
  ASSERT_EQ(spec->concat_max_input_num, 63);
  ASSERT_FALSE(spec->is_default_enabled);
}

TEST_F(TestOptimizer, AutofuseBackendSpecTest) {
  // "2201" -> is_default_enabled = false
  auto spec_2201 = ge::GetAutofuseBackendSpec();
  ASSERT_NE(spec_2201, nullptr);
  EXPECT_FALSE(spec_2201->is_default_enabled);

  // "3510" -> is_default_enabled = true
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  auto spec_3510 = ge::GetAutofuseBackendSpec();
  ASSERT_NE(spec_3510, nullptr);
  EXPECT_TRUE(spec_3510->is_default_enabled);

  // "5102" -> is_default_enabled = false
  ge::PlatformContext::GetInstance().SetPlatform("5102");
  auto spec_5102 = ge::GetAutofuseBackendSpec();
  ASSERT_NE(spec_5102, nullptr);
  EXPECT_FALSE(spec_5102->is_default_enabled);
}

TEST_F(TestOptimizer, BrcCacheReuseOtherMem) {
  af::AscGraph graph("BrcCacheReuseOtherMem");
  const af::Expression s0 = af::Symbol(12);
  const af::Expression s1 = af::Symbol(32);
  const af::Expression s2 = af::Symbol(64);
  const af::Expression s3 = af::Symbol(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data x0("x", graph);
  x0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load0.y.repeats = {s0, s1, One, s3};
  *load0.y.strides = {s1 * s3, s3, Zero, One};

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs0.y.repeats = {s0, s1, One, s3};
  *abs0.y.strides = {s1 * s3, s3, Zero, One};

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc0.y.repeats = {s0, s1, s2, s3};
  *brc0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Data x1("x1", graph);
  x1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load1.y.repeats = {s0, s1, s2, One};
  *load1.y.strides = {s1 * s2, s2, One, Zero};

  af::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs1.y.repeats = {s0, s1, s2, One};
  *abs1.y.strides = {s1 * s2, s2, One, Zero};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = abs1.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc1.y.repeats = {s0, s1, s2, s3};
  *brc1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Add add0("add0");
  add0.x1 = brc0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add0.y.repeats = {s0, s1, s2, s3};
  *add0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Data x2("x2", graph);
  x2.ir_attr.SetIndex(2);

  af::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load2.y.repeats = {One, One, s2, s3};
  *load2.y.strides = {Zero, Zero, s3, One};

  af::ascir_op::Abs abs2("abs2");
  abs2.x = load2.y;
  abs2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs2.y.repeats = {One, One, s2, s3};
  *abs2.y.strides = {Zero, Zero, s3, One};

  af::ascir_op::Exp exp2("exp2");
  exp2.x = abs2.y;
  exp2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *exp2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *exp2.y.repeats = {One, One, s2, s3};
  *exp2.y.strides = {Zero, Zero, s3, One};

  af::ascir_op::Broadcast brc3("brc3");
  brc3.x = exp2.y;
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc3.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc3.y.repeats = {s0, s1, s2, s3};
  *brc3.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Data x3("x3", graph);
  x3.ir_attr.SetIndex(3);

  af::ascir_op::Load load3("load3");
  load3.x = x3.y;
  load3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load3.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load3.y.repeats = {s0, s1, s2, s3};
  *load3.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Abs abs3("abs3");
  abs3.x = load3.y;
  abs3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs3.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs3.y.repeats = {s0, s1, s2, s3};
  *abs3.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Exp exp3("exp3");
  exp3.x = abs3.y;
  exp3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *exp3.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *exp3.y.repeats = {s0, s1, s2, s3};
  *exp3.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Add add1("add1");
  add1.x1 = brc3.y;
  add1.x2 = exp3.y;
  add1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *add1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add1.y.repeats = {s0, s1, s2, s3};
  *add1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Add add2("add2");
  add2.x1 = add0.y;
  add2.x2 = add1.y;
  add2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *add2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add2.y.repeats = {s0, s1, s2, s3};
  *add2.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Store store("store");
  store.x = add2.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, s2, s3};
  *store.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 3);
  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  const auto &impl_graph = impl_graphs[2];
  auto abs2_result = impl_graph.FindNode("abs2");
  auto exp2_result = impl_graph.FindNode("exp2");
  EXPECT_NE(abs2_result->outputs[0].attr.buf.id, exp2_result->outputs[0].attr.buf.id);
}

TEST_F(TestOptimizer, SliceSliceConcatD) {
  af::AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(90);
  auto s2 = graph.CreateSizeVar(1);
  auto s1_0 = graph.CreateSizeVar(60);
  auto s1_1 = graph.CreateSizeVar(30);
  auto s3 = graph.CreateSizeVar(97);
  auto s4 = graph.CreateSizeVar(65);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_0 = graph.CreateAxis("z1_0", s1_0);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {s0, s1_1, One};
  *data0.y.strides = {s1_1, One, One};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data1.y.axis = {z0.id, z1.id, z2.id};
  *data1.y.repeats = {s0, s1_0, One};
  *data1.y.strides = {s1_0, One, One};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1_0.id};
  load0.x = data1.y;
  *load0.y.axis = {z0.id, z1_0.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1_0};
  *load0.y.strides = {s3 * s1_0, s3};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1_1.id};
  load1.x = data0.y;
  *load1.y.axis = {z0.id, z1_1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1_1};
  *load1.y.strides = {s4 * s1_1, s4};

  af::ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {load0.y, load1.y};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1};
  *concat_op.y.strides = {s1, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, af::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  for (auto impl_graph : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs) {
    auto load0_remove_pad_0 = impl_graph.FindNode("load0_remove_pad_0");
    EXPECT_NE(load0_remove_pad_0, nullptr);
    auto load1_remove_pad_0 = impl_graph.FindNode("load1_remove_pad_0");
    EXPECT_NE(load1_remove_pad_0, nullptr);
  }
}

TEST_F(TestOptimizer, LastAxisStoreWithStride) {
  af::AscGraph graph("matmul");
  auto s0 = graph.CreateSizeVar(256);
  auto z0 = graph.CreateAxis("z0", s0);

  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id};
  *load0.y.repeats = {s0};
  *load0.y.strides = {af::ops::One};

  Store store0("store0");
  store0.attr.sched.axis = {z0.id};
  store0.x = load0.y;
  *store0.y.axis = {z0.id};
  *store0.y.repeats = {s0};
  *store0.y.strides = {af::Symbol(134)};

  Output out0("output0");
  out0.x = store0.y;
  out0.ir_attr.SetIndex(0);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  std::vector<af::Expression> golden_stride = {af::Symbol(8)};
  auto load_node = impl_graph.FindNode("load0");
  EXPECT_NE(load_node, nullptr);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden_stride);

  auto store_node = impl_graph.FindNode("store0");
  EXPECT_NE(store_node, nullptr);
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(TestOptimizer, AbsAbsAbsAbsTransposeCast) {
  af::AscGraph graph("trans_int64");
  auto s0 = af::Symbol(3);
  auto s1 = af::Symbol(10);
  auto s2 = af::Symbol(4);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  std::vector<int64_t> axis_ids = {z2.id, z1.id, z0.id};
  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = axis_ids;
  load0.x = data0.y;
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = axis_ids;
  *load0.y.repeats = {s2, s1, s0};
  *load0.y.strides = {s1 * s0, s0, af::ops::One};

  Transpose transpose0("transpose0");
  transpose0.attr.sched.axis = axis_ids;
  transpose0.x = load0.y;
  transpose0.y.dtype = ge::DT_FLOAT;
  *transpose0.y.axis = {z0.id, z1.id, z2.id};
  *transpose0.y.repeats = {s0, s1, s2};
  *transpose0.y.strides = {s1 * s2, s2, af::ops::One};

  Cast cast0("cast0");
  cast0.attr.sched.axis = axis_ids;
  cast0.x = transpose0.y;
  cast0.y.dtype = ge::DT_INT64;
  *cast0.y.axis = {z0.id, z1.id, z2.id};
  *cast0.y.repeats = {s0, s1, s2};
  *cast0.y.strides = {s1 * s2, s2, af::ops::One};

  Store store0("store0");
  store0.attr.sched.axis = axis_ids;
  store0.x = cast0.y;
  store0.y.dtype = ge::DT_INT64;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, af::ops::One};

  Output out0("out0");
  out0.x = store0.y;
  out0.y.dtype = ge::DT_INT64;
  out0.ir_attr.SetIndex(0);

  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
}
