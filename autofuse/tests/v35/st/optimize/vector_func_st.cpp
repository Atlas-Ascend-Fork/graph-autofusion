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

#include "codegen.h"

#include "ascendc_ir.h"
#include "ascendc_ir_def.h"
#include "ascir_ops.h"
#define private public
#include "optimize.h"
#include "platform_context.h"
#undef private
#include "ascir_ops_utils.h"
#include "runtime_stub.h"

using namespace std;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

namespace {
class VectorFuncSt : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("DUMP_GE_GRAPH", "2", 1);
    ge::PlatformContext::GetInstance().Reset();
    af::InstallV35SocPerfRuntimeStub();
  }
  void TearDown() override {
    setenv("DUMP_GE_GRAPH", "0", 1);
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }

  optimize::Optimizer optimizer;

  VectorFuncSt() : optimizer(optimize::OptimizerOptions{}) {}
};
}  // namespace

namespace optimize {
namespace {
void SetTrueDivAddRsqrtTensorAttr(af::AscOpOutput &tensor, const std::vector<af::AxisId> &axes,
                                  const af::Expression &rows, const af::Expression &columns) {
  tensor.dtype = af::DT_FLOAT;
  *tensor.axis = axes;
  *tensor.repeats = {rows, columns};
  *tensor.strides = {columns, One};
  *tensor.vectorized_axis = axes;
  *tensor.vectorized_strides = {columns, One};
}

void SetTrueDivAddRsqrtComputeAttr(af::AscGraph &graph, const std::vector<af::AxisId> &axes, const char *name,
                                   af::ComputeType compute_type) {
  auto node = graph.FindNode(name);
  ASSERT_NE(node, nullptr);
  node->attr.sched.axis = axes;
  node->attr.api.compute_type = compute_type;
  node->attr.api.type = af::ApiType::kAPITypeCompute;
}

af::AscOpOutput BuildTrueDivGraph(af::AscGraph &graph, const std::vector<af::AxisId> &axes, const af::Expression &rows,
                                  const af::Expression &columns) {
  af::ascir_op::Data dividend("dividend", graph);
  dividend.ir_attr.SetIndex(0);
  dividend.attr.api.type = af::ApiType::kAPITypeBuffer;
  SetTrueDivAddRsqrtTensorAttr(dividend.y, axes, rows, columns);

  af::ascir_op::Load load_dividend("load_dividend");
  load_dividend.x = dividend.y;
  SetTrueDivAddRsqrtTensorAttr(load_dividend.y, axes, rows, columns);
  SetTrueDivAddRsqrtComputeAttr(graph, axes, "load_dividend", af::ComputeType::kComputeLoad);

  af::ascir_op::Data divisor("divisor", graph);
  divisor.ir_attr.SetIndex(1);
  divisor.attr.api.type = af::ApiType::kAPITypeBuffer;
  SetTrueDivAddRsqrtTensorAttr(divisor.y, axes, rows, columns);

  af::ascir_op::Load load_divisor("load_divisor");
  load_divisor.x = divisor.y;
  SetTrueDivAddRsqrtTensorAttr(load_divisor.y, axes, rows, columns);
  SetTrueDivAddRsqrtComputeAttr(graph, axes, "load_divisor", af::ComputeType::kComputeLoad);

  af::ascir_op::TrueDiv true_div("true_div");
  true_div.x1 = load_dividend.y;
  true_div.x2 = load_divisor.y;
  SetTrueDivAddRsqrtTensorAttr(true_div.y, axes, rows, columns);
  SetTrueDivAddRsqrtComputeAttr(graph, axes, "true_div", af::ComputeType::kComputeElewise);
  return true_div.y;
}

void BuildRsqrtGraph(af::AscGraph &graph, const std::vector<af::AxisId> &axes, const af::Expression &rows,
                     const af::Expression &columns, const af::AscOpOutput &true_div_output) {
  af::ascir_op::Scalar epsilon("epsilon", graph);
  epsilon.y.dtype = af::DT_FLOAT;
  *epsilon.y.repeats = {One, One};
  *epsilon.y.strides = {Zero, Zero};
  epsilon.ir_attr.SetValue("0.00001");

  af::ascir_op::Add add_epsilon("add_epsilon");
  add_epsilon.x1 = true_div_output;
  add_epsilon.x2 = epsilon.y;
  SetTrueDivAddRsqrtTensorAttr(add_epsilon.y, axes, rows, columns);
  SetTrueDivAddRsqrtComputeAttr(graph, axes, "add_epsilon", af::ComputeType::kComputeElewise);

  af::ascir_op::Rsqrt rsqrt("rsqrt");
  rsqrt.x = add_epsilon.y;
  SetTrueDivAddRsqrtTensorAttr(rsqrt.y, axes, rows, columns);
  SetTrueDivAddRsqrtComputeAttr(graph, axes, "rsqrt", af::ComputeType::kComputeElewise);

  af::ascir_op::Store store("store");
  store.x = rsqrt.y;
  SetTrueDivAddRsqrtTensorAttr(store.y, axes, rows, columns);
  SetTrueDivAddRsqrtComputeAttr(graph, axes, "store", af::ComputeType::kComputeStore);

  Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.attr.api.type = af::ApiType::kAPITypeBuffer;
  SetTrueDivAddRsqrtTensorAttr(output.y, axes, rows, columns);
}

void BuildTrueDivAddRsqrtGraph(af::AscGraph &graph) {
  const auto rows = graph.CreateSizeVar(32);
  const auto columns = graph.CreateSizeVar(128);
  const auto row_axis = graph.CreateAxis("row", rows);
  const auto column_axis = graph.CreateAxis("column", columns);
  const std::vector<af::AxisId> axes{row_axis.id, column_axis.id};
  const auto true_div_output = BuildTrueDivGraph(graph, axes, rows, columns);
  BuildRsqrtGraph(graph, axes, rows, columns, true_div_output);
}

bool HasTrueDivAddRsqrtNodes(const af::AscGraph &graph) {
  return graph.FindNode("true_div") != nullptr && graph.FindNode("add_epsilon") != nullptr &&
         graph.FindNode("rsqrt") != nullptr;
}

template <typename ImplGraph>
bool HasTrueDivAddRsqrtInImplGraph(const ImplGraph &impl_graph) {
  std::vector<af::AscGraph> subgraphs;
  if (impl_graph.GetAllSubGraphs(subgraphs) != af::SUCCESS) {
    return false;
  }
  for (const auto &subgraph : subgraphs) {
    if (HasTrueDivAddRsqrtNodes(subgraph)) {
      return true;
    }
  }
  return false;
}

template <typename ScheduleGroup>
bool HasTrueDivAddRsqrtInScheduleGroup(const ScheduleGroup &group) {
  for (const auto &impl_graph : group.impl_graphs) {
    if (HasTrueDivAddRsqrtInImplGraph(impl_graph)) {
      return true;
    }
  }
  return false;
}

template <typename ScheduledResult>
bool HasTrueDivAddRsqrtInScheduledResult(const ScheduledResult &scheduled_result) {
  for (const auto &group : scheduled_result.schedule_groups) {
    if (HasTrueDivAddRsqrtInScheduleGroup(group)) {
      return true;
    }
  }
  return false;
}

bool HasTrueDivAddRsqrtSubgraph(const ::ascir::FusedScheduledResult &result) {
  for (const auto &scheduled_results : result.node_idx_to_scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      if (HasTrueDivAddRsqrtInScheduledResult(scheduled_result)) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace

TEST_F(VectorFuncSt, TrueDivAddRsqrtHostCodegen) {
  af::AscGraph graph("truediv_add_rsqrt_host_codegen");
  BuildTrueDivAddRsqrtGraph(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_scheduled_result), af::SUCCESS);
  ASSERT_TRUE(HasTrueDivAddRsqrtSubgraph(fused_scheduled_result));

  codegen::Codegen generator(codegen::CodegenOptions{});
  codegen::CodegenResult result;
  ASSERT_EQ(generator.Generate(fused_scheduled_result, result), af::SUCCESS);

  const auto first_vf = result.kernel.find("__simd_vf__");
  ASSERT_NE(first_vf, std::string::npos);
  const auto rsqrt_marker = result.kernel.find("_rsqrt_one", first_vf);
  ASSERT_NE(rsqrt_marker, std::string::npos);
  const auto vf_begin = result.kernel.rfind("__simd_vf__", rsqrt_marker);
  ASSERT_NE(vf_begin, std::string::npos);
  const auto vf_end = result.kernel.find("\n}\n", rsqrt_marker);
  ASSERT_NE(vf_end, std::string::npos);
  const auto vf_body = result.kernel.substr(vf_begin, vf_end - vf_begin);
  const auto true_div_pos = vf_body.find("AscendC::MicroAPI::Div<float, &high_precision_div_mode>(");
  const auto add_pos = vf_body.find("AscendC::MicroAPI::Adds(");
  const auto sqrt_pos = vf_body.find("AscendC::MicroAPI::Sqrt(");
  const auto rsqrt_div_pos =
      vf_body.find("AscendC::MicroAPI::Div<float, &high_precision_div_mode>(", true_div_pos + 1U);
  ASSERT_NE(true_div_pos, std::string::npos);
  ASSERT_NE(add_pos, std::string::npos);
  ASSERT_NE(sqrt_pos, std::string::npos);
  ASSERT_NE(rsqrt_div_pos, std::string::npos);
  EXPECT_LT(true_div_pos, add_pos);
  EXPECT_LT(add_pos, sqrt_pos);
  EXPECT_LT(sqrt_pos, rsqrt_div_pos);
}

TEST_F(VectorFuncSt, vf_partition) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(999);
  auto s1 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, One};
  *load0.y.strides = {One, Zero};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Broadcast brc("brc");
  brc.x = load0.y;
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id, z1.id};
  brc.y.dtype = af::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id};
  *brc.y.repeats = {s0, s1};
  *brc.y.strides = {s1, One};
  *brc.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs("abs");
  abs.x = brc.y;
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, One};
  *abs.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs1("abs1");
  abs1.x = brc.y;
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.sched.axis = {z0.id, z1.id};
  abs1.y.dtype = af::DT_FLOAT;
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};
  *abs1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add("add");
  add.x1 = abs.y;
  add.x2 = abs1.y;
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id, z1.id};
  add.y.dtype = af::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};
  *add.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  Output out("out");
  out.x = store.y;
  out.attr.api.compute_type = ComputeType::kComputeInvalid;
  out.attr.api.type = af::ApiType::kAPITypeBuffer;
  out.y.dtype = af::DT_FLOAT;
  out.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
}

TEST_F(VectorFuncSt, skip_fuse_for_cycle) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(999);
  auto s1 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, One};
  *load0.y.strides = {One, Zero};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs("abs");
  abs.x = load0.y;
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, Zero};
  *abs.y.strides = {One, Zero};
  *abs.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Broadcast cat("cat");
  cat.x = abs.y;
  cat.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  cat.attr.sched.axis = {z0.id, z1.id};
  cat.y.dtype = af::DT_FLOAT;
  *cat.y.axis = {z0.id, z1.id};
  *cat.y.repeats = {s0, s1};
  *cat.y.strides = {s1, One};
  *cat.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add("add");
  add.x1 = abs.y;
  add.x2 = cat.y;
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id, z1.id};
  add.y.dtype = af::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};
  *add.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  Output out("out");
  out.x = store.y;
  out.attr.api.compute_type = ComputeType::kComputeInvalid;
  out.attr.api.type = af::ApiType::kAPITypeBuffer;
  out.y.dtype = af::DT_FLOAT;
  out.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
}

TEST_F(VectorFuncSt, ResetIOLimit) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(999);
  auto s1 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = af::DT_FLOAT;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};
  *load.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load2("load2");
  load2.x = data0.y;
  load2.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = af::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, One};
  *load2.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load3("load3");
  load3.x = data0.y;
  load3.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = af::DT_FLOAT;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, One};
  *load3.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, One};
  *abs.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.sched.axis = {z0.id, z1.id};
  abs1.y.dtype = af::DT_FLOAT;
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};
  *abs1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs2("abs2");
  abs2.x = load2.y;
  abs2.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs2.attr.sched.axis = {z0.id, z1.id};
  abs2.y.dtype = af::DT_FLOAT;
  *abs2.y.axis = {z0.id, z1.id};
  *abs2.y.repeats = {s0, s1};
  *abs2.y.strides = {s1, One};
  *abs2.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs3("abs3");
  abs3.x = load3.y;
  abs3.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs3.attr.sched.axis = {z0.id, z1.id};
  abs3.y.dtype = af::DT_FLOAT;
  *abs3.y.axis = {z0.id, z1.id};
  *abs3.y.repeats = {s0, s1};
  *abs3.y.strides = {s1, One};
  *abs3.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add("add");
  add.x1 = abs.y;
  add.x2 = abs1.y;
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id, z1.id};
  add.y.dtype = af::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};
  *add.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add1("add1");
  add1.x1 = abs2.y;
  add1.x2 = abs3.y;
  add1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add1.attr.sched.axis = {z0.id, z1.id};
  add1.y.dtype = af::DT_FLOAT;
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, One};
  *add1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add2("add2");
  add2.x1 = add1.y;
  add2.x2 = add.y;
  add2.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add2.attr.sched.axis = {z0.id, z1.id};
  add2.y.dtype = af::DT_FLOAT;
  *add2.y.axis = {z0.id, z1.id};
  *add2.y.repeats = {s0, s1};
  *add2.y.strides = {s1, One};
  *add2.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Store store("store");
  store.x = add2.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  Output out("out");
  out.x = store.y;
  out.attr.api.compute_type = ComputeType::kComputeInvalid;
  out.attr.api.type = af::ApiType::kAPITypeBuffer;
  out.y.dtype = af::DT_FLOAT;
  out.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);

  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  std::vector<af::AscGraph> asc_graphs;
  fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllSubGraphs(
      asc_graphs);
  EXPECT_EQ(asc_graphs.size(), 1UL);
}

TEST_F(VectorFuncSt, cast_bit_with) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(999);
  auto s1 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT16;
  data0.ir_attr.SetIndex(0);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.attr.sched.axis = {z0.id, z1.id};
  abs0.y.dtype = af::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};
  *abs0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Cast cast0("cast0");
  cast0.x = abs0.y;
  cast0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast0.attr.sched.axis = {z0.id, z1.id};
  cast0.y.dtype = af::DT_FLOAT;
  *cast0.y.axis = {z0.id, z1.id};
  *cast0.y.repeats = {s0, s1};
  *cast0.y.strides = {s1, One};
  *cast0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = af::DT_FLOAT;
  data1.ir_attr.SetIndex(0);
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.sched.axis = {z0.id, z1.id};
  abs1.y.dtype = af::DT_FLOAT;
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};
  *abs1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Cast cast1("cast1");
  cast1.x = abs1.y;
  cast1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast1.attr.sched.axis = {z0.id, z1.id};
  cast1.y.dtype = af::DT_INT64;
  *cast1.y.axis = {z0.id, z1.id};
  *cast1.y.repeats = {s0, s1};
  *cast1.y.strides = {s1, One};
  *cast1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Cast cast2("cast2");
  cast2.x = cast1.y;
  cast2.attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast2.attr.sched.axis = {z0.id, z1.id};
  cast2.y.dtype = af::DT_FLOAT;
  *cast2.y.axis = {z0.id, z1.id};
  *cast2.y.repeats = {s0, s1};
  *cast2.y.strides = {s1, One};
  *cast2.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add("add");
  add.x1 = cast0.y;
  add.x2 = cast2.y;
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id, z1.id};
  add.y.dtype = af::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};
  *add.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  Output out("out");
  out.x = store.y;
  out.attr.api.compute_type = ComputeType::kComputeInvalid;
  out.attr.api.type = af::ApiType::kAPITypeBuffer;
  out.y.dtype = af::DT_FLOAT;
  out.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);

  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  std::vector<af::AscGraph> asc_graphs;
  fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllSubGraphs(
      asc_graphs);
  // float<->int64 casts cross the 8<->4 byte boundary and stay in the root
  // graph (CastExtend), so the remaining f16/float nodes merge into a single
  // VF cluster instead of being split by the int64 bit-width gap.
  EXPECT_EQ(asc_graphs.size(), 1UL);
}

TEST_F(VectorFuncSt, cycle_bugfix) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(999);
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(0);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = af::DT_FLOAT;
  data1.ir_attr.SetIndex(1);
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id};
  load0.y.dtype = af::DT_FLOAT;
  *load0.y.axis = {z0.id};
  *load0.y.repeats = {s0};
  *load0.y.strides = {One};

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.attr.sched.axis = {z0.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id};
  *load1.y.repeats = {s0};
  *load1.y.strides = {One};

  af::ascir_op::Exp exp("exp");
  exp.x = load1.y;
  exp.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp.attr.sched.axis = {z0.id};
  exp.y.dtype = af::DT_FLOAT;
  *exp.y.axis = {z0.id};
  *exp.y.repeats = {s0};
  *exp.y.strides = {One};

  af::ascir_op::Sub sub("sub");
  sub.x1 = load0.y;
  sub.x2 = exp.y;
  sub.attr.api.compute_type = af::ComputeType::kComputeElewise;
  sub.attr.sched.axis = {z0.id};
  sub.y.dtype = af::DT_FLOAT;
  *sub.y.axis = {z0.id};
  *sub.y.repeats = {s0};
  *sub.y.strides = {One};

  af::ascir_op::Relu relu("relu");
  relu.x = sub.y;
  relu.attr.api.compute_type = af::ComputeType::kComputeElewise;
  relu.attr.sched.axis = {z0.id};
  relu.y.dtype = af::DT_FLOAT;
  *relu.y.axis = {z0.id};
  *relu.y.repeats = {s0};
  *relu.y.strides = {One};

  af::ascir_op::Mul mul("mul");
  mul.x1 = sub.y;
  mul.x2 = exp.y;
  mul.attr.api.compute_type = af::ComputeType::kComputeElewise;
  mul.attr.sched.axis = {z0.id};
  mul.y.dtype = af::DT_FLOAT;
  *mul.y.axis = {z0.id};
  *mul.y.repeats = {s0};
  *mul.y.strides = {One};

  af::ascir_op::Abs abs("abs");
  abs.x = mul.y;
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id};
  *abs.y.repeats = {s0};
  *abs.y.strides = {One};

  af::ascir_op::Add add("add");
  add.x1 = relu.y;
  add.x2 = abs.y;
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id};
  add.y.dtype = af::DT_FLOAT;
  *add.y.axis = {z0.id};
  *add.y.repeats = {s0};
  *add.y.strides = {One};

  af::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  Output out("out");
  out.x = store.y;
  out.attr.api.compute_type = ComputeType::kComputeInvalid;
  out.attr.api.type = af::ApiType::kAPITypeBuffer;
  out.y.dtype = af::DT_FLOAT;
  out.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);

  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  std::vector<af::AscGraph> asc_graphs;
  fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllSubGraphs(
      asc_graphs);
  EXPECT_EQ(asc_graphs.size(), 1UL);
}

TEST_F(VectorFuncSt, CastHorizontolFusion) {
  af::AscGraph graph("cast_graph");
  auto s0 = af::Symbol(128);
  auto s1 = af::Symbol(64);
  auto s2 = af::Symbol(160);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT16;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = af::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, One, s2};
  *load0.y.strides = {Zero, Zero, One};

  Cast cast0("cast0");
  cast0.x = load0.y;
  cast0.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast0.y.dtype = af::DT_FLOAT;
  *cast0.y.axis = {z0.id, z1.id, z2.id};
  *cast0.y.repeats = {One, One, s2};
  *cast0.y.strides = {Zero, Zero, One};

  Broadcast brc0("brc0");
  brc0.x = cast0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc0.y.dtype = af::DT_FLOAT;
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = af::DT_FLOAT16;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = af::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};

  Cast cast1("cast1");
  cast1.x = load1.y;
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast1.y.dtype = af::DT_FLOAT;
  *cast1.y.axis = {z0.id, z1.id, z2.id};
  *cast1.y.repeats = {s0, s1, s2};
  *cast1.y.strides = {s1 * s2, s2, One};

  Mul mul0("mul0");
  mul0.x1 = brc0.y;
  mul0.x2 = cast1.y;
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.y.dtype = af::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Data data2("data2", graph);
  data2.y.dtype = af::DT_FLOAT16;
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = af::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.repeats = {s0, s1, s2};
  *load2.y.strides = {s1 * s2, s2, One};

  Cast cast2("cast2");
  cast2.x = load2.y;
  cast2.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast2.y.dtype = af::DT_FLOAT;
  *cast2.y.axis = {z0.id, z1.id, z2.id};
  *cast2.y.repeats = {s0, s1, s2};
  *cast2.y.strides = {s1 * s2, s2, One};

  Sub sub0("sub0");
  sub0.x1 = mul0.y;
  sub0.x2 = cast2.y;
  sub0.attr.sched.axis = {z0.id, z1.id, z2.id};
  sub0.y.dtype = af::DT_FLOAT;
  *sub0.y.axis = {z0.id, z1.id, z2.id};
  *sub0.y.repeats = {s0, s1, s2};
  *sub0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Data data3("data3", graph);
  data3.y.dtype = af::DT_FLOAT16;
  data3.ir_attr.SetIndex(3);

  Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id, z2.id};
  load3.y.dtype = af::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1.id, z2.id};
  *load3.y.repeats = {One, One, One};
  *load3.y.strides = {Zero, Zero, Zero};

  Cast cast3("cast3");
  cast3.x = load3.y;
  cast3.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast3.y.dtype = af::DT_FLOAT;
  *cast3.y.axis = {z0.id, z1.id, z2.id};
  *cast3.y.repeats = {One, One, One};
  *cast3.y.strides = {Zero, Zero, Zero};

  Broadcast brc3("brc3");
  brc3.x = cast3.y;
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = af::DT_FLOAT;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, One};

  Add add0("add0");
  add0.x1 = brc3.y;
  add0.x2 = sub0.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id};
  add0.y.dtype = af::DT_FLOAT;
  *add0.y.axis = {z0.id, z1.id, z2.id};
  *add0.y.repeats = {s0, s1, s2};
  *add0.y.strides = {s1 * s2, s2, One};

  Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.y.dtype = af::DT_FLOAT;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, One};

  Output out("out");
  out.x = store0.y;
  out.y.dtype = af::DT_FLOAT;
  out.ir_attr.SetIndex(0);

  Prod prod0("prod0");
  prod0.x = add0.y;
  prod0.attr.sched.axis = {z0.id, z1.id, z2.id};
  prod0.y.dtype = af::DT_FLOAT;
  *prod0.y.axis = {z0.id, z1.id, z2.id};
  *prod0.y.repeats = {s0, s1, One};
  *prod0.y.strides = {s1, One, Zero};

  Store store1("store1");
  store1.x = prod0.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store1.y.dtype = af::DT_FLOAT;
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, One};
  *store1.y.strides = {s1, One, Zero};

  Output out1("out1");
  out1.x = store1.y;
  out1.y.dtype = af::DT_FLOAT;
  out1.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);

  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  // R-core phase-1 多 tiling case 收敛为同 group 多 impl, RCore 任务的结果数 2 -> 1, 调度结果由 4 减为 3
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_TRUE(!fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.empty());
  std::vector<af::AscGraph> asc_graphs;
  fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllSubGraphs(
      asc_graphs);
  EXPECT_EQ(asc_graphs.size(), 1UL);
}

TEST_F(VectorFuncSt, vectorized_not_empty) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol("s1");
  auto s2 = af::Symbol("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT16;
  data0.ir_attr.SetIndex(0);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = af::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Cast cast0("cast0");
  cast0.x = load0.y;
  cast0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast0.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast0.y.dtype = af::DT_FLOAT;
  *cast0.y.axis = {z0.id, z1.id, z2.id};
  *cast0.y.repeats = {s0, s1, s2};
  *cast0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Neg neg0("neg0");
  neg0.x = cast0.y;
  neg0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  neg0.attr.sched.axis = {z0.id, z1.id, z2.id};
  neg0.y.dtype = af::DT_FLOAT;
  *neg0.y.axis = {z0.id, z1.id, z2.id};
  *neg0.y.repeats = {s0, s1, s2};
  *neg0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Sum sum0("sum0");
  sum0.x = neg0.y;
  sum0.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum0.attr.sched.axis = {z0.id, z1.id, z2.id};
  sum0.y.dtype = af::DT_FLOAT;
  *sum0.y.axis = {z0.id, z1.id, z2.id};
  *sum0.y.repeats = {s0, s1, One};
  *sum0.y.strides = {s1, One, Zero};

  af::ascir_op::Cast cast1("cast1");
  cast1.x = sum0.y;
  cast1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast1.y.dtype = af::DT_FLOAT16;
  *cast1.y.axis = {z0.id, z1.id, z2.id};
  *cast1.y.repeats = {s0, s1, One};
  *cast1.y.strides = {s1, One, Zero};

  af::ascir_op::Store store("store");
  store.x = cast1.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = af::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, One};
  *store.y.strides = {s1, One, Zero};

  Output out("out");
  out.x = store.y;
  out.attr.api.compute_type = ComputeType::kComputeInvalid;
  out.attr.api.type = af::ApiType::kAPITypeBuffer;
  out.y.dtype = af::DT_FLOAT;
  out.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);

  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  // R-core phase-1 多 tiling case 收敛为同 group 多 impl, RCore 任务的结果数 2 -> 1, 调度结果由 4 减为 3
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  std::vector<af::AscGraph> asc_graphs;
  fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllSubGraphs(
      asc_graphs);
  EXPECT_EQ(asc_graphs.size(), 1UL);
}

TEST_F(VectorFuncSt, scalar_brc_fusion) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(0);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {One, One};
  *load0.y.strides = {Zero, Zero};

  af::ascir_op::Abs abs("abs0");
  abs.x = load0.y;
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {One, One};
  *abs.y.strides = {Zero, Zero};

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = abs.y;
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.attr.sched.axis = {z0.id, z1.id};
  brc0.y.dtype = af::DT_FLOAT;
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};

  af::ascir_op::Scalar scalar("scalar0", graph);
  scalar.y.dtype = af::DT_FLOAT;
  scalar.ir_attr.SetValue("0");
  scalar.attr.api.type = af::ApiType::kAPITypeBuffer;
  *scalar.y.repeats = {One, One};
  *scalar.y.strides = {Zero, Zero};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = scalar.y;
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.y.dtype = af::DT_FLOAT;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  af::ascir_op::Add add0("add0");
  add0.x1 = brc0.y;
  add0.x2 = brc1.y;
  add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.y.dtype = af::DT_FLOAT;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  af::ascir_op::Store store("store");
  store.x = add0.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  Output out("out");
  out.x = store.y;
  out.attr.api.compute_type = ComputeType::kComputeInvalid;
  out.attr.api.type = af::ApiType::kAPITypeBuffer;
  out.y.dtype = af::DT_FLOAT;
  out.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);

  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);

  for (const auto &group : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups) {
    for (const auto &impl_graph : group.impl_graphs) {
      auto brc_node0 = impl_graph.FindNode("brc0");
      EXPECT_EQ(brc_node0, nullptr);
      auto brc_node1 = impl_graph.FindNode("brc1");
      EXPECT_EQ(brc_node1, nullptr);
      std::vector<af::AscGraph> asc_graphs;
      impl_graph.GetAllSubGraphs(asc_graphs);
      EXPECT_EQ(asc_graphs.size(), 1UL);
    }
  }
}

TEST_F(VectorFuncSt, BrcMultiReuse) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(4);
  auto s1 = af::Symbol(28);
  auto s2 = af::Symbol(28);
  auto s3 = af::Symbol(4);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT16;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.y.dtype = af::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load0.y.repeats = {s0, s1, One, One};
  *load0.y.strides = {s1, One, Zero, Zero};

  Cast cast0("cast0");
  cast0.x = load0.y;
  cast0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  cast0.y.dtype = af::DT_FLOAT;
  *cast0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast0.y.repeats = {s0, s1, One, One};
  *cast0.y.strides = {s1, One, Zero, Zero};

  Abs abs0("abs0");
  abs0.x = cast0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs0.y.dtype = af::DT_FLOAT;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs0.y.repeats = {s0, s1, One, One};
  *abs0.y.strides = {s1, One, Zero, Zero};

  Broadcast brc0("brc0");
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc0.y.dtype = af::DT_FLOAT;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc0.y.repeats = {s0, s1, s2, s3};
  *brc0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Data data1("data1", graph);
  data1.y.dtype = af::DT_FLOAT16;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.y.dtype = af::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load1.y.repeats = {s0, s1, s2, s3};
  *load1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Cast cast1("cast1");
  cast1.x = load1.y;
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  cast1.y.dtype = af::DT_FLOAT;
  *cast1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast1.y.repeats = {s0, s1, s2, s3};
  *cast1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Relu relu0("relu0");
  relu0.x = cast1.y;
  relu0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  relu0.y.dtype = af::DT_FLOAT;
  *relu0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *relu0.y.repeats = {s0, s1, s2, s3};
  *relu0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Sub sub0("sub0");
  sub0.x1 = relu0.y;
  sub0.x2 = brc0.y;
  sub0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  sub0.y.dtype = af::DT_FLOAT;
  *sub0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *sub0.y.repeats = {s0, s1, s2, s3};
  *sub0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add add0("add0");
  add0.x1 = brc0.y;
  add0.x2 = relu0.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  add0.y.dtype = af::DT_FLOAT;
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add0.y.repeats = {s0, s1, s2, s3};
  *add0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add add1("add1");
  add1.x1 = add0.y;
  add1.x2 = sub0.y;
  add1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  add1.y.dtype = af::DT_FLOAT;
  *add1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add1.y.repeats = {s0, s1, s2, s3};
  *add1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Abs abs1("abs1");
  abs1.x = add1.y;
  abs1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs1.y.dtype = af::DT_FLOAT;
  *abs1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs1.y.repeats = {s0, s1, s2, s3};
  *abs1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Relu relu1("relu1");
  relu1.x = abs1.y;
  relu1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  relu1.y.dtype = af::DT_FLOAT;
  *relu1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *relu1.y.repeats = {s0, s1, s2, s3};
  *relu1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Cast cast2("cast2");
  cast2.x = relu1.y;
  cast2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  cast2.y.dtype = af::DT_FLOAT16;
  *cast2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast2.y.repeats = {s0, s1, s2, s3};
  *cast2.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Store store("store");
  store.x = cast2.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store.y.dtype = af::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, s2, s3};
  *store.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Output out("out");
  out.x = store.y;
  out.y.dtype = af::DT_FLOAT;
  out.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);

  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  std::vector<af::AscGraph> asc_graphs;
  fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllSubGraphs(
      asc_graphs);
  EXPECT_EQ(asc_graphs.size(), 2UL);
  auto graph1 = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1];
  std::vector<af::AscGraph> asc_graphs1;
  graph1.GetAllSubGraphs(asc_graphs1);
  ASSERT_EQ(asc_graphs1.size(), 1UL);
  std::set<std::string> supported_types = {"Data", "Output", "VectorFunc", "Load", "Store"};
  for (const auto &node : graph1.GetAllNodes()) {
    EXPECT_TRUE(supported_types.count(node->GetType()) > 0UL);
  }
}

TEST_F(VectorFuncSt, CastNotFusion) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(32);
  auto s1 = af::Symbol(60);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id};
  abs0.y.dtype = af::DT_FLOAT;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};

  Add add0("add0");
  add0.x1 = abs0.y;
  add0.x2 = load0.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.y.dtype = af::DT_FLOAT;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  Cast cast0("cast0");
  cast0.x = add0.y;
  cast0.attr.sched.axis = {z0.id, z1.id};
  cast0.y.dtype = af::DT_INT64;
  *cast0.y.axis = {z0.id, z1.id};
  *cast0.y.repeats = {s0, s1};
  *cast0.y.strides = {s1, One};

  Store store("store");
  store.x = cast0.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_INT64;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  Output out("out");
  out.x = store.y;
  out.y.dtype = af::DT_INT64;
  out.ir_attr.SetIndex(0);

  Cast cast1("cast1");
  cast1.x = abs0.y;
  cast1.attr.sched.axis = {z0.id, z1.id};
  cast1.y.dtype = af::DT_INT64;
  *cast1.y.axis = {z0.id, z1.id};
  *cast1.y.repeats = {s0, s1};
  *cast1.y.strides = {s1, One};

  Store store1("store1");
  store1.x = cast1.y;
  store1.attr.sched.axis = {z0.id, z1.id};
  store1.y.dtype = af::DT_INT64;
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {s0, s1};
  *store1.y.strides = {s1, One};

  Output out1("out1");
  out1.x = store1.y;
  out1.y.dtype = af::DT_INT64;
  out1.ir_attr.SetIndex(1);

  Scalar scalar("scalar0", graph);
  scalar.y.dtype = af::DT_INT64;

  Broadcast brc3("brc3");
  brc3.x = scalar.y;
  brc3.attr.sched.axis = {z0.id, z1.id};
  brc3.y.dtype = af::DT_INT64;
  *brc3.y.axis = {z0.id, z1.id};
  *brc3.y.repeats = {s0, s1};
  *brc3.y.strides = {s1, One};

  Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  Broadcast brc4("brc4");
  brc4.x = load1.y;
  brc4.attr.sched.axis = {z0.id, z1.id};
  brc4.y.dtype = af::DT_FLOAT;
  *brc4.y.axis = {z0.id, z1.id};
  *brc4.y.repeats = {s0, s1};
  *brc4.y.strides = {s1, One};

  Sign sign0("sign0");
  sign0.x = brc4.y;
  sign0.attr.sched.axis = {z0.id, z1.id};
  sign0.y.dtype = af::DT_FLOAT;
  *sign0.y.axis = {z0.id, z1.id};
  *sign0.y.repeats = {s0, s1};
  *sign0.y.strides = {s1, One};

  Mul mul0("mul0");
  mul0.x1 = sign0.y;
  mul0.x2 = sign0.y;
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.y.dtype = af::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};

  Abs abs1("Abs1");
  abs1.x = mul0.y;
  abs1.attr.sched.axis = {z0.id, z1.id};
  abs1.y.dtype = af::DT_FLOAT;
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};

  Gt gt0("gt0");
  gt0.x1 = abs0.y;
  gt0.x2 = abs1.y;
  gt0.attr.sched.axis = {z0.id, z1.id};
  gt0.y.dtype = af::DT_FLOAT;
  *gt0.y.axis = {z0.id, z1.id};
  *gt0.y.repeats = {s0, s1};
  *gt0.y.strides = {s1, One};

  Where where0("where0");
  where0.x1 = gt0.y;
  where0.x2 = cast1.y;
  where0.x3 = brc3.y;
  where0.attr.sched.axis = {z0.id, z1.id};
  where0.y.dtype = af::DT_INT64;
  *where0.y.axis = {z0.id, z1.id};
  *where0.y.repeats = {s0, s1};
  *where0.y.strides = {s1, One};

  Store store2("store2");
  store2.x = where0.y;
  store2.attr.sched.axis = {z0.id, z1.id};
  store2.y.dtype = af::DT_INT64;
  *store2.y.axis = {z0.id, z1.id};
  *store2.y.repeats = {s0, s1};
  *store2.y.strides = {s1, One};

  Output out2("out2");
  out2.x = store2.y;
  out2.y.dtype = af::DT_INT64;
  out2.ir_attr.SetIndex(2);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);

  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  std::vector<af::AscGraph> asc_graphs;
  fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllSubGraphs(
      asc_graphs);
  EXPECT_EQ(asc_graphs.size(), 3UL);
  auto graph1 = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1];
  std::vector<af::AscGraph> asc_graphs1;
  graph1.GetAllSubGraphs(asc_graphs1);
  ASSERT_EQ(asc_graphs1.size(), 2UL);
  size_t cast_num = 0UL;
  for (const auto &node : graph1.GetAllNodes()) {
    if (node->GetType() == "Cast") {
      ++cast_num;
    }
  }
  // float->int64 casts cross the 8<->4 byte boundary: the capability gate
  // keeps them in the root graph (CastExtend) instead of fusing into VF,
  // so both casts remain visible in the impl graph.
  EXPECT_EQ(cast_num, 2UL);
}

TEST_F(VectorFuncSt, MaximumNotFusion) {
  af::AscGraph graph("brc_abs");
  auto s1 = af::Symbol(60);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z1.id};
  load0.y.dtype = af::DT_FLOAT;
  *load0.y.axis = {z1.id};
  *load0.y.repeats = {s1};
  *load0.y.strides = {One};

  Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z1.id};
  abs0.y.dtype = af::DT_FLOAT;
  *abs0.y.axis = {z1.id};
  *abs0.y.repeats = {s1};
  *abs0.y.strides = {One};

  Add add0("add0");
  add0.x1 = abs0.y;
  add0.x2 = load0.y;
  add0.attr.sched.axis = {z1.id};
  add0.y.dtype = af::DT_FLOAT;
  *add0.y.axis = {z1.id};
  *add0.y.repeats = {s1};
  *add0.y.strides = {One};

  af::ascir_op::Scalar scalar0("scalar0", graph);
  Maximum maximum("maximum");
  maximum.x1 = add0.y;
  maximum.x2 = scalar0.y;
  maximum.attr.sched.axis = {z1.id};
  maximum.y.dtype = af::DT_FLOAT;
  *maximum.y.axis = {z1.id};
  *maximum.y.repeats = {s1};
  *maximum.y.strides = {One};

  Abs abs1("abs1");
  abs1.x = maximum.y;
  abs1.attr.sched.axis = {z1.id};
  abs1.y.dtype = af::DT_FLOAT;
  *abs1.y.axis = {z1.id};
  *abs1.y.repeats = {s1};
  *abs1.y.strides = {One};

  Store store("store");
  store.x = abs1.y;
  store.attr.sched.axis = {z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z1.id};
  *store.y.repeats = {s1};
  *store.y.strides = {One};

  Output out("out");
  out.x = store.y;
  out.y.dtype = af::DT_FLOAT;
  out.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);

  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  std::vector<af::AscGraph> asc_graphs;
  fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllSubGraphs(
      asc_graphs);
  EXPECT_EQ(asc_graphs.size(), 1UL);
  auto graph1 = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  std::vector<af::AscGraph> asc_graphs1;
  graph1.GetAllSubGraphs(asc_graphs1);
  ASSERT_EQ(asc_graphs1.size(), 1UL);

  auto scalar_node = graph.FindNode("scalar0");
  ASSERT_NE(scalar_node, nullptr);

  auto maximum_node = graph.FindNode("maximum");
  ASSERT_NE(maximum_node, nullptr);

  auto abs1_node = graph.FindNode("abs1");
  ASSERT_NE(abs1_node, nullptr);

  auto abs_node = asc_graphs1[0].FindNode("abs0");
  ASSERT_NE(abs_node, nullptr);

  auto add_node = asc_graphs1[0].FindNode("add0");
  ASSERT_NE(add_node, nullptr);
}
}  // namespace optimize
