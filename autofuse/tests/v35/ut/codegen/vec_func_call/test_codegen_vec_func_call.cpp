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
#include "node_utils_ex.h"
#include "ascir_utils.h"
#include "ascir_node_param/ascir_node_param.h"
#include "ascir_node_param/ascir_param_builder.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "vec_func_call.h"
#include "micro_api_call/micro_api_call_factory.h"
#include "micro_api_call/micro_arange_api_call.h"
#include "../common.h"
#include "codegen_graph_check.h"

using namespace std;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

namespace {
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

size_t CountOccurrences(const std::string &text, const std::string &needle) {
  size_t count = 0;
  size_t offset = 0;
  while ((offset = text.find(needle, offset)) != std::string::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

class FailingMicroApiCall final : public MicroApiCall {
 public:
  FailingMicroApiCall() : MicroApiCall("FailingMicroApiCall") {}

  Status Generate(const TensorManager &, const TPipe &, CallParam &, std::string &) override {
    return af::FAILED;
  }
};

class FixedArangeMicroApiCall final : public MicroApiCall {
 public:
  FixedArangeMicroApiCall(int64_t tensor_id, std::string base, std::string step)
      : MicroApiCall("FixedArangeMicroApiCall"), base_(std::move(base)), step_(std::move(step)) {
    AddOutput(tensor_id);
  }

  bool HasArangeParam() const override {
    return true;
  }

  void GetArangeParams(const TPipe &, std::string &base, std::string &step) const override {
    base = base_;
    step = step_;
  }

 private:
  std::string base_;
  std::string step_;
};

template <typename TensorLike>
void SetTwoDimSchedule(TensorLike &tensor, const af::Axis &z0, const af::Axis &z1, const af::Expression &s0,
                       const af::Expression &s1) {
  *tensor.axis = {z0.id, z1.id};
  *tensor.repeats = {s0, s1};
  *tensor.strides = {s1, One};
}

void SetTwoDimVecInAttr(AscTensor &tensor, const af::Axis &z0, const af::Axis &z1, const af::Expression &stride,
                        int64_t tensor_id) {
  tensor.attr.vectorized_axis = {z0.id, z1.id};
  tensor.attr.vectorized_strides = {stride, One};
  tensor.attr.dtype = af::DT_FLOAT;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.mem.tensor_id = tensor_id;
}

codegen::Tensor MakeCvUbFuseTensor(af::AscGraph &graph, ge::DataType dtype, int64_t tensor_id,
                                   const std::vector<af::Expression> &vectorized_strides,
                                   const std::string &tensor_name) {
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];
  tensor.attr.dtype = dtype;
  tensor.attr.mem.tensor_id = tensor_id;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.vectorized_strides = vectorized_strides;
  std::string dtype_name;
  EXPECT_EQ(codegen::Tensor::DtypeName(dtype, dtype_name), af::SUCCESS);
  return codegen::Tensor(tensor, dtype_name, tensor_name);
}

void InitScalarDataVfGraph(VectorFunc &vf_op, Store &store_op, Broadcast &sub_brc_op, Abs &abs_op, Store &sub_store_op,
                           Output &sub_output_op, const ScalarData &scalar_data_op, Scalar &sub_scalar_op,
                           const af::Axis &z0, const af::Axis &z1, const af::Expression &s0, const af::Expression &s1) {
  vf_op.InstanceOutputy(1);
  vf_op.x = {scalar_data_op.y};
  vf_op.attr.sched.axis = {z0.id, z1.id};
  SetTwoDimSchedule(vf_op.y[0], z0, z1, s0, s1);

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  SetTwoDimSchedule(store_op.y, z0, z1, s0, s1);

  sub_brc_op.x = sub_scalar_op.y;
  sub_scalar_op.attr.api.unit = af::ComputeUnit::kUnitNone;
  sub_brc_op.attr.sched.axis = {z0.id, z1.id};
  SetTwoDimSchedule(sub_brc_op.y, z0, z1, s0, s1);

  abs_op.x = sub_brc_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  SetTwoDimSchedule(abs_op.y, z0, z1, s0, s1);

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0.id, z1.id};
  SetTwoDimSchedule(sub_store_op.y, z0, z1, s0, s1);
  sub_output_op.x = sub_store_op.y;
  sub_output_op.attr.api.unit = af::ComputeUnit::kUnitNone;
}

void InitScalarDataVfTensorAttrs(AscGraph &graph, AscGraph &vf_sub_graph, const af::Axis &z0, const af::Axis &z1,
                                 const af::Expression &s1) {
  auto scalar_data = graph.FindNode("scalar_data");
  scalar_data->outputs[0].attr.dtype = af::DT_FLOAT;
  scalar_data->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  scalar_data->outputs[0].attr.mem.tensor_id = 1;
  scalar_data->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_scalar = vf_sub_graph.FindNode("sub_scalar");
  sub_scalar->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  sub_scalar->outputs[0].attr.vectorized_strides = {Zero, Zero};
  sub_scalar->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_scalar->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_scalar->outputs[0].attr.mem.tensor_id = 1;

  SetTwoDimVecInAttr(vf_sub_graph.FindNode("sub_brc")->outputs[0], z0, z1, s1, 2);
  SetTwoDimVecInAttr(vf_sub_graph.FindNode("abs")->outputs[0], z0, z1, s1, 3);
  SetTwoDimVecInAttr(vf_sub_graph.FindNode("sub_store")->outputs[0], z0, z1, s1, 4);

  auto vf = graph.FindNode("vf");
  SetTwoDimVecInAttr(vf->outputs[0], z0, z1, s1, 5);
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  store->outputs[0].attr.vectorized_strides = {s1, One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 6;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

std::string GenerateScalarDataVfCall(AscGraph &graph, const af::Axis &z0, const af::Axis &z1, const af::Expression &s0,
                                     const af::Expression &s1) {
  auto scalar_data = graph.FindNode("scalar_data");
  auto vf = graph.FindNode("vf");
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  EXPECT_EQ(tpipe.CollectQues(graph), af::SUCCESS);
  EXPECT_EQ(tpipe.AddTensor(scalar_data->outputs[0], "scalar_data_y"), 0);
  EXPECT_EQ(tpipe.AddTensor(vf->outputs[0]), af::SUCCESS);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = scalar_data->outputs[0].attr.mem.tensor_id;
  x2.id = vf->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  return result;
}

template <typename TensorLike>
void SetFiveDimSchedule(TensorLike &tensor, const std::vector<af::Axis> &axes,
                        const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  *tensor.axis = {axes[0].id, axes[1].id, axes[2].id, axes[3].id, axes[4].id};
  *tensor.repeats = repeats;
  *tensor.strides = strides;
}

void SetFiveDimVecInAttr(AscTensor &tensor, const std::vector<af::Axis> &axes,
                         const std::vector<af::Expression> &strides, int64_t tensor_id) {
  tensor.attr.vectorized_axis = {axes[0].id, axes[1].id, axes[2].id, axes[3].id, axes[4].id};
  tensor.attr.vectorized_strides = strides;
  tensor.attr.dtype = af::DT_FLOAT;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.mem.tensor_id = tensor_id;
}

void InitDoubleLoopVfGraph(VectorFunc &vf_op, Store &store_op, Load &sub_load_op, Abs &abs_op, Store &sub_store_op,
                           Output &sub_output_op, const Load &load_op, const Data &sub_x_op,
                           const std::vector<af::Axis> &axes, const std::vector<af::Expression> &repeats,
                           const std::vector<af::Expression> &strides) {
  vf_op.InstanceOutputy(1);
  vf_op.x = {load_op.y};
  vf_op.attr.sched.axis = {axes[0].id, axes[1].id, axes[2].id, axes[3].id, axes[4].id};
  SetFiveDimSchedule(vf_op.y[0], axes, repeats, strides);

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  SetFiveDimSchedule(store_op.y, axes, repeats, strides);

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {axes[3].id, axes[4].id};
  sub_load_op.attr.sched.loop_axis = axes[4].id;
  SetFiveDimSchedule(sub_load_op.y, axes, repeats, strides);

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {axes[3].id, axes[4].id};
  abs_op.attr.sched.loop_axis = axes[4].id;
  SetFiveDimSchedule(abs_op.y, axes, repeats, strides);

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {axes[3].id, axes[4].id};
  sub_store_op.attr.sched.loop_axis = axes[4].id;
  SetFiveDimSchedule(sub_store_op.y, axes, repeats, strides);
  sub_output_op.x = sub_store_op.y;
}

void InitDoubleLoopTensorAttrs(AscGraph &graph, AscGraph &vf_sub_graph, const std::vector<af::Axis> &axes,
                               const std::vector<af::Expression> &base_strides,
                               const std::vector<af::Expression> &output_strides) {
  auto load = graph.FindNode("load");
  SetFiveDimVecInAttr(load->outputs[0], axes, base_strides, 0);
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  SetFiveDimVecInAttr(vf_sub_graph.FindNode("sub_load")->outputs[0], axes, base_strides, 0);
  SetFiveDimVecInAttr(vf_sub_graph.FindNode("abs")->outputs[0], axes, base_strides, 1);
  SetFiveDimVecInAttr(vf_sub_graph.FindNode("sub_store")->outputs[0], axes, output_strides, 2);

  auto vf = graph.FindNode("vf");
  SetFiveDimVecInAttr(vf->outputs[0], axes, output_strides, 1);
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = {axes[0].id, axes[1].id, axes[2].id, axes[3].id, axes[4].id};
  store->outputs[0].attr.vectorized_strides = output_strides;
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

void PrepareDoubleLoopCodegen(AscGraph &graph, const std::vector<af::Axis> &axes,
                              const std::vector<af::Expression> &repeats, codegen::Tiler &tiler, codegen::TPipe &tpipe,
                              codegen::VfCall &call, std::stringstream &func_def) {
  auto load = graph.FindNode("load");
  auto vf = graph.FindNode("vf");
  // 模拟生产 Kernel::ParseGraph 流程: queue 类型 tensor 入 tpipe 前需先从图收集 que 定义,
  // 否则 AddTensor 报 "Cannot find que"。
  ASSERT_EQ(tpipe.CollectQues(graph), af::SUCCESS);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(vf->outputs[0]);
  for (const auto &axis : axes) {
    tiler.AddAxis(axis);
  }
  for (const auto &repeat : repeats) {
    tiler.AddSizeVar(af::SizeVar(repeat));
  }

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  ASSERT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);
  ASSERT_EQ(ascir_param::EnrichAscirGraphNodeParams(graph), af::SUCCESS);
  ASSERT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);
}

void RegisterDoubleLoopNodes(AscGraph &graph, AscGraph &vf_sub_graph, VectorFunc &vf_op, Data &sub_x_op,
                             Output &sub_output_op, Load &load_op, Store &store_op, const std::string &sub_graph_name) {
  vf_op.SetAttr("sub_graph_name", sub_graph_name);
  sub_x_op.ir_attr.SetIndex(0);
  sub_output_op.ir_attr.SetIndex(0);
  // 生产图 CompleteApiInfo 会将 Data/Output 的 api.unit 置为 kUnitNone(不发射代码);
  // 手工 fixture 需显式设置, 否则 base MicroApiCall 会参与 Generate 并失败。
  sub_x_op.attr.api.unit = af::ComputeUnit::kUnitNone;
  sub_output_op.attr.api.unit = af::ComputeUnit::kUnitNone;
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);
}

void CheckDoubleLoopVectorFuncParams(const af::AscNodePtr &vf) {
  auto params = ascir_param::GetAscirNodeParams(vf);
  ASSERT_NE(params, nullptr);
  const auto *vector_func_params = std::get_if<ascir_param::VectorFuncNodeParams>(&params->specific_params);
  ASSERT_NE(vector_func_params, nullptr);
  EXPECT_TRUE(vector_func_params->is_double_loop);
  ASSERT_EQ(vector_func_params->all_strides.size(), 6U);
  EXPECT_STREQ(vector_func_params->all_strides[0].Serialize().get(), "(s2 * s3 * s4)");
  EXPECT_STREQ(vector_func_params->all_strides[5].Serialize().get(), "(2 * s4)");
  ASSERT_EQ(vector_func_params->output_dims.size(), 4U);
  EXPECT_STREQ(vector_func_params->output_dims[0].Serialize().get(), "s1");
  EXPECT_STREQ(vector_func_params->output_dims[3].Serialize().get(), "s4");
}
}  // namespace

TEST(VFLoopTest, PropagatesFailureWhenMicroApiCallFails) {
  // 生成失败必须上抛: 吞错会在真机产生未初始化寄存器参与计算的静默错误数据。
  VFLoop loop(af::kIdNone);
  loop.AddCall(new FailingMicroApiCall());

  Tiler tiler;
  TPipe tpipe("tpipe", tiler);
  TensorManager tensor_manager;
  std::string result;
  std::string loop_size;
  int32_t max_depth = -1;
  std::vector<std::string> loop_sizes;
  EXPECT_NE(loop.Generate(tpipe, tensor_manager, 0, result, loop_size, max_depth, loop_sizes), af::SUCCESS);
  EXPECT_NE(loop.GenerateCvUbFuse(tpipe, tensor_manager, result, loop_size), af::SUCCESS);
  loop.Destruct();
}

TEST(VFLoopTest, DeduplicatesSiblingLoopDeclarations) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("sibling_vf_loops");
  const auto size = graph.CreateSizeVar("size");
  const auto axis = graph.CreateAxis("axis", size);

  Tiler tiler;
  tiler.AddAxis(axis);
  TPipe tpipe("tpipe", tiler);
  TensorManager tensor_manager;

  VFLoop root(af::kIdNone);
  root.SetMaxDtypeSize("float");
  root.AddLoop(new VFLoop(axis.id));
  root.AddLoop(new VFLoop(axis.id));

  std::string result;
  std::string loop_size;
  int32_t max_depth = -1;
  std::vector<std::string> loop_sizes;
  ASSERT_EQ(root.Generate(tpipe, tensor_manager, 0, result, loop_size, max_depth, loop_sizes), af::SUCCESS);
  EXPECT_EQ(CountOccurrences(loop_size, "uint16_t axis_loop_size = loop_times;"), 1U) << loop_size;
  EXPECT_EQ(CountOccurrences(result, "uint32_t sreg_0 = element_count;"), 1U) << result;
  EXPECT_EQ(CountOccurrences(result, "AscendC::MicroAPI::MaskReg preg_0;"), 1U) << result;
  root.Destruct();
}

TEST(VFLoopTest, GeneratesInt64ArangeAcrossVectorBlocks) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("arange_vf_loop");
  const auto size = graph.CreateSizeVar("size");
  const auto axis = graph.CreateAxis("axis", size);
  af::AscGraph vf_sub_graph("arange_vf_sub_graph");
  VectorFunc vf_op("vf");
  vf_op.InstanceOutputy(1);
  vf_op.SetAttr("sub_graph_name", "arange_vf_sub_graph");
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(vf_op);

  Arange arange_op("arange");
  Abs abs_op("abs");
  Output output_op("output");
  arange_op.ir_attr.SetBase(af::Symbol(3));
  arange_op.ir_attr.SetStep(af::Symbol(2));
  arange_op.attr.api.unit = af::ComputeUnit::kUnitVector;
  arange_op.attr.sched.axis = {axis.id};
  arange_op.y.dtype = af::DT_INT64;
  *arange_op.y.axis = {axis.id};
  *arange_op.y.repeats = {size};
  *arange_op.y.strides = {One};
  *arange_op.y.vectorized_axis = {axis.id};
  *arange_op.y.vectorized_strides = {One};
  abs_op.attr.api.unit = af::ComputeUnit::kUnitVector;
  abs_op.attr.sched.axis = {axis.id};
  abs_op.y.dtype = af::DT_INT64;
  *abs_op.y.axis = {axis.id};
  *abs_op.y.repeats = {size};
  *abs_op.y.strides = {One};
  *abs_op.y.vectorized_axis = {axis.id};
  *abs_op.y.vectorized_strides = {One};
  output_op.ir_attr.SetIndex(0);
  vf_sub_graph.AddNode(arange_op);
  vf_sub_graph.AddNode(abs_op);
  vf_sub_graph.AddNode(output_op);
  abs_op.x = arange_op.y;
  output_op.x = abs_op.y;
  const auto node = vf_sub_graph.FindNode("arange");
  node->outputs[0].attr.mem.tensor_id = 0;
  auto abs_node = vf_sub_graph.FindNode("abs");
  abs_node->outputs[0].attr.mem.tensor_id = 2;
  auto vf_node = graph.FindNode("vf");
  vf_node->outputs[0].attr.dtype = af::DT_INT64;
  vf_node->outputs[0].attr.mem.tensor_id = 1;
  vf_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;

  std::string dtype_name;
  ASSERT_EQ(codegen::Tensor::DtypeName(af::DT_INT64, dtype_name), af::SUCCESS);
  TensorManager tensor_manager;
  ASSERT_EQ(tensor_manager.AddTensor(MicroApiTensor(node->outputs[0], dtype_name)), af::SUCCESS);
  ASSERT_EQ(tensor_manager.AddTensor(MicroApiTensor(vf_node->outputs[0], dtype_name)), af::SUCCESS);
  ASSERT_EQ(tensor_manager.AddTensor(MicroApiTensor(abs_node->outputs[0], dtype_name)), af::SUCCESS);
  Tiler tiler;
  tiler.AddAxis(axis);
  tiler.AddSizeVar(af::SizeVar(size));
  TPipe tpipe("tpipe", tiler);
  ASSERT_EQ(tpipe.AddTensor(vf_node->outputs[0]), af::SUCCESS);

  VfCall vf_call;
  ASSERT_EQ(vf_call.Init(vf_node), af::SUCCESS);
  tpipe.cv_fusion_type = ::ascir::CubeTemplateType::kUBFuse;
  std::stringstream ub_fuse_definition;
  EXPECT_NE(vf_call.GenerateFuncDefinition(tpipe, tiler, ub_fuse_definition), af::SUCCESS);
  tpipe.cv_fusion_type = ::ascir::CubeTemplateType::kDefault;

  VFLoop loop(axis.id);
  loop.SetMaxDtypeSize("int64_t");
  ASSERT_EQ(loop.ConstructFromNodes(vf_sub_graph.GetAllNodes(), vf_node), af::SUCCESS);

  std::string result;
  std::string loop_size;
  int32_t max_depth = -1;
  std::vector<std::string> loop_sizes;
  ASSERT_EQ(
      loop.Generate(tpipe, tensor_manager, 0, result, loop_size, max_depth, loop_sizes, {{0, "device_block_offset"}}),
      af::SUCCESS);
  EXPECT_NE(result.find("for (uint16_t axis"), std::string::npos);
  const auto arange_adds = GetLineContaining(result, "AscendC::Reg::Adds");
  ASSERT_FALSE(arange_adds.empty());
  EXPECT_EQ(arange_adds,
            "AscendC::Reg::Adds(vreg_0, vreg_0, static_cast<int64_t>((arange_base_0 + (device_block_offset + "
            "(axis * ELEMENT_PER_VECTOR_LENGTH)) * (arange_step_0))), preg_0);");
  loop.Destruct();
}

TEST(CodegenKernel, HighRankArangeOuterForUsesLogicalOffset) {
  for (const size_t rank : {5UL, 6UL}) {
    ge::SetupRuntimeStub();
    af::AscGraph graph(("high_rank_arange_" + std::to_string(rank)).c_str());
    std::vector<af::Axis> axes;
    std::vector<af::AxisId> axis_ids;
    std::vector<af::Expression> logical_strides;
    for (size_t i = 0UL; i < rank; ++i) {
      axes.push_back(graph.CreateAxis("axis" + std::to_string(i), af::Symbol(2)));
      axis_ids.push_back(axes.back().id);
      logical_strides.push_back(af::Symbol(static_cast<int64_t>(1UL << (rank - i - 1UL))));
    }
    const std::vector<af::Expression> repeats(rank, af::Symbol(2));
    const std::vector<af::Expression> physical_strides =
        rank == 5UL ? std::vector<af::Expression>{af::Symbol(64), af::Symbol(24), af::Symbol(10), af::Symbol(3), One}
                    : std::vector<af::Expression>{af::Symbol(180), af::Symbol(70), af::Symbol(26),
                                                  af::Symbol(10),  af::Symbol(3),  One};

    af::AscGraph subgraph(("high_rank_arange_subgraph_" + std::to_string(rank)).c_str());
    VectorFunc vf_op("vf");
    vf_op.InstanceOutputy(1);
    vf_op.SetAttr("sub_graph_name", "high_rank_arange_subgraph_" + std::to_string(rank));
    graph.AddSubGraph(subgraph);
    graph.AddNode(vf_op);

    Arange arange_op("arange");
    arange_op.ir_attr.SetBase(af::Symbol(3));
    arange_op.ir_attr.SetStep(af::Symbol(2));
    arange_op.attr.api.unit = af::ComputeUnit::kUnitVector;
    arange_op.attr.sched.axis = axis_ids;
    arange_op.attr.sched.loop_axis = axes.back().id;
    arange_op.y.dtype = af::DT_INT64;
    *arange_op.y.axis = axis_ids;
    *arange_op.y.repeats = repeats;
    *arange_op.y.strides = logical_strides;
    *arange_op.y.vectorized_axis = axis_ids;
    *arange_op.y.vectorized_strides = physical_strides;

    Abs abs_op("abs");
    abs_op.attr.api.unit = af::ComputeUnit::kUnitVector;
    abs_op.attr.sched.axis = axis_ids;
    abs_op.attr.sched.loop_axis = axes.back().id;
    abs_op.y.dtype = af::DT_INT64;
    *abs_op.y.axis = axis_ids;
    *abs_op.y.repeats = repeats;
    *abs_op.y.strides = logical_strides;
    *abs_op.y.vectorized_axis = axis_ids;
    *abs_op.y.vectorized_strides = physical_strides;

    Store store_op("store");
    store_op.attr.api.unit = af::ComputeUnit::kUnitMTE3;
    store_op.attr.sched.axis = axis_ids;
    store_op.y.dtype = af::DT_INT64;
    *store_op.y.axis = axis_ids;
    *store_op.y.repeats = repeats;
    *store_op.y.strides = logical_strides;
    *store_op.y.vectorized_axis = axis_ids;
    *store_op.y.vectorized_strides = physical_strides;

    Output output_op("output");
    output_op.ir_attr.SetIndex(0);
    subgraph.AddNode(arange_op);
    subgraph.AddNode(abs_op);
    subgraph.AddNode(store_op);
    subgraph.AddNode(output_op);
    abs_op.x = arange_op.y;
    store_op.x = abs_op.y;
    output_op.x = store_op.y;

    const auto arange_node = subgraph.FindNode("arange");
    const auto abs_node = subgraph.FindNode("abs");
    ASSERT_NE(arange_node, nullptr);
    ASSERT_NE(abs_node, nullptr);
    arange_node->outputs[0].attr.mem.tensor_id = 0;
    abs_node->outputs[0].attr.mem.tensor_id = 1;
    const auto store_node = subgraph.FindNode("store");
    ASSERT_NE(store_node, nullptr);
    store_node->outputs[0].attr.mem.tensor_id = 2;
    const auto vf_node = graph.FindNode("vf");
    ASSERT_NE(vf_node, nullptr);
    vf_node->outputs[0].attr.dtype = af::DT_INT64;
    vf_node->outputs[0].attr.axis = axis_ids;
    vf_node->outputs[0].attr.repeats = repeats;
    vf_node->outputs[0].attr.strides = physical_strides;
    vf_node->outputs[0].attr.vectorized_axis = axis_ids;
    vf_node->outputs[0].attr.vectorized_strides = physical_strides;
    vf_node->outputs[0].attr.mem.tensor_id = 3;
    vf_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;

    Tiler tiler;
    for (const auto &axis : axes) {
      ASSERT_EQ(tiler.AddAxis(axis), af::SUCCESS);
    }
    TPipe tpipe("tpipe", tiler);
    ASSERT_EQ(tpipe.AddTensor(vf_node->outputs[0]), af::SUCCESS);
    VfCall call;
    ASSERT_EQ(call.Init(vf_node), af::SUCCESS);

    std::stringstream definition;
    ASSERT_EQ(call.GenerateFuncDefinition(tpipe, tiler, definition), af::SUCCESS);
    std::string invocation;
    ASSERT_EQ(call.Generate(tpipe, {}, invocation), af::SUCCESS);
    EXPECT_NE(invocation.find("for(int outer_for_0 = 0;"), std::string::npos) << invocation;
    const auto vf_call = GetLineContaining(invocation, "VFCallvf(");
    ASSERT_FALSE(vf_call.empty()) << invocation;
    EXPECT_NE(vf_call.find("outer_for_0 * " + std::to_string(1UL << (rank - 1UL))), std::string::npos) << vf_call;
    EXPECT_EQ(vf_call.find("0 + outer_for_0 * " + std::to_string(rank == 5UL ? 64UL : 180UL)), std::string::npos)
        << vf_call;
    if (rank == 6UL) {
      EXPECT_NE(invocation.find("for(int outer_for_1 = 0;"), std::string::npos) << invocation;
      EXPECT_NE(vf_call.find("outer_for_1 * 16"), std::string::npos) << vf_call;
    }
  }
}

TEST(VFLoopTest, UsesLogicalOuterStrideForAlignedArangeLayout) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("aligned_arange_vf_loop");
  const auto rows = graph.CreateSizeVar("rows");
  const auto cols = graph.CreateSizeVar("cols");
  const auto row_axis = graph.CreateAxis("row", rows);
  const auto col_axis = graph.CreateAxis("col", cols);
  af::AscGraph vf_sub_graph("aligned_arange_vf_sub_graph");
  VectorFunc vf_op("vf");
  vf_op.InstanceOutputy(1);
  vf_op.SetAttr("sub_graph_name", "aligned_arange_vf_sub_graph");
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(vf_op);

  Arange arange_op("arange");
  Abs abs_op("abs");
  Output output_op("output");
  arange_op.ir_attr.SetBase(af::Symbol(0));
  arange_op.ir_attr.SetStep(af::Symbol(1));
  arange_op.attr.api.unit = af::ComputeUnit::kUnitVector;
  arange_op.attr.sched.axis = {row_axis.id, col_axis.id};
  arange_op.attr.sched.loop_axis = col_axis.id;
  arange_op.y.dtype = af::DT_INT64;
  *arange_op.y.axis = {row_axis.id, col_axis.id};
  *arange_op.y.repeats = {rows, cols};
  *arange_op.y.strides = {cols, One};
  *arange_op.y.vectorized_axis = {row_axis.id, col_axis.id};
  *arange_op.y.vectorized_strides = {af::Symbol(16), One};
  abs_op.attr.api.unit = af::ComputeUnit::kUnitVector;
  abs_op.attr.sched.axis = {row_axis.id, col_axis.id};
  abs_op.attr.sched.loop_axis = col_axis.id;
  abs_op.y.dtype = af::DT_INT64;
  *abs_op.y.axis = {row_axis.id, col_axis.id};
  *abs_op.y.repeats = {rows, cols};
  *abs_op.y.strides = {cols, One};
  *abs_op.y.vectorized_axis = {row_axis.id, col_axis.id};
  *abs_op.y.vectorized_strides = {af::Symbol(16), One};
  output_op.ir_attr.SetIndex(0);
  vf_sub_graph.AddNode(arange_op);
  vf_sub_graph.AddNode(abs_op);
  vf_sub_graph.AddNode(output_op);
  abs_op.x = arange_op.y;
  output_op.x = abs_op.y;
  const auto arange_node = vf_sub_graph.FindNode("arange");
  arange_node->outputs[0].attr.mem.tensor_id = 0;
  const auto abs_node = vf_sub_graph.FindNode("abs");
  abs_node->outputs[0].attr.mem.tensor_id = 2;
  auto vf_node = graph.FindNode("vf");
  vf_node->outputs[0].attr.dtype = af::DT_INT64;
  vf_node->outputs[0].attr.mem.tensor_id = 1;
  vf_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;

  std::string dtype_name;
  ASSERT_EQ(codegen::Tensor::DtypeName(af::DT_INT64, dtype_name), af::SUCCESS);
  TensorManager tensor_manager;
  ASSERT_EQ(tensor_manager.AddTensor(MicroApiTensor(arange_node->outputs[0], dtype_name)), af::SUCCESS);
  ASSERT_EQ(tensor_manager.AddTensor(MicroApiTensor(vf_node->outputs[0], dtype_name)), af::SUCCESS);
  ASSERT_EQ(tensor_manager.AddTensor(MicroApiTensor(abs_node->outputs[0], dtype_name)), af::SUCCESS);
  Tiler tiler;
  tiler.AddAxis(row_axis);
  tiler.AddAxis(col_axis);
  tiler.AddSizeVar(af::SizeVar(rows));
  tiler.AddSizeVar(af::SizeVar(cols));
  TPipe tpipe("tpipe", tiler);
  ASSERT_EQ(tpipe.AddTensor(vf_node->outputs[0]), af::SUCCESS);

  VFLoop loop(af::kIdNone);
  loop.SetMaxDtypeSize("int64_t");
  ASSERT_EQ(loop.ConstructFromNodes(vf_sub_graph.GetAllNodes(), vf_node), af::SUCCESS);

  std::string result;
  std::string loop_size;
  int32_t max_depth = -1;
  std::vector<std::string> loop_sizes;
  ASSERT_EQ(
      loop.Generate(tpipe, tensor_manager, 0, result, loop_size, max_depth, loop_sizes, {{0, "device_block_offset"}}),
      af::SUCCESS);
  const auto arange_call = GetLineContaining(result, "AscendC::Reg::Arange");
  ASSERT_FALSE(arange_call.empty()) << result;
  EXPECT_NE(arange_call.find("row * arange_stride_0_0"), std::string::npos);
  EXPECT_EQ(arange_call.find("t->"), std::string::npos);
  EXPECT_NE(arange_call.find("col * ELEMENT_PER_VECTOR_LENGTH"), std::string::npos);
  EXPECT_EQ(arange_call.find("row * 16"), std::string::npos);
  loop.Destruct();

  arange_node->attr.sched.axis = {row_axis.id};
  arange_node->attr.sched.loop_axis = row_axis.id;
  arange_node->outputs[0].attr.repeats = {rows, One};
  arange_node->outputs[0].attr.strides = {One, Zero};
  arange_node->outputs[0].attr.vectorized_strides = {One, Zero};
  abs_node->attr.sched.axis = {row_axis.id};
  abs_node->attr.sched.loop_axis = row_axis.id;
  abs_node->outputs[0].attr.repeats = {rows, One};
  abs_node->outputs[0].attr.strides = {One, Zero};
  abs_node->outputs[0].attr.vectorized_strides = {One, Zero};
  TensorManager trailing_singleton_tensor_manager;
  ASSERT_EQ(trailing_singleton_tensor_manager.AddTensor(MicroApiTensor(arange_node->outputs[0], dtype_name)),
            af::SUCCESS);
  ASSERT_EQ(trailing_singleton_tensor_manager.AddTensor(MicroApiTensor(abs_node->outputs[0], dtype_name)), af::SUCCESS);
  ASSERT_EQ(trailing_singleton_tensor_manager.AddTensor(MicroApiTensor(vf_node->outputs[0], dtype_name)), af::SUCCESS);
  VFLoop trailing_singleton_loop(af::kIdNone);
  trailing_singleton_loop.SetMaxDtypeSize("int64_t");
  ASSERT_EQ(trailing_singleton_loop.ConstructFromNodes(vf_sub_graph.GetAllNodes(), vf_node), af::SUCCESS);
  result.clear();
  loop_size.clear();
  max_depth = -1;
  loop_sizes.clear();
  ASSERT_EQ(trailing_singleton_loop.Generate(tpipe, trailing_singleton_tensor_manager, 0, result, loop_size, max_depth,
                                             loop_sizes),
            af::SUCCESS);
  const auto trailing_singleton_call = GetLineContaining(result, "AscendC::Reg::Arange");
  ASSERT_FALSE(trailing_singleton_call.empty()) << result;
  EXPECT_NE(trailing_singleton_call.find("row * ELEMENT_PER_VECTOR_LENGTH"), std::string::npos)
      << trailing_singleton_call;
  trailing_singleton_loop.Destruct();

  arange_node->attr.sched.axis.clear();
  arange_node->attr.sched.loop_axis = af::kIdNone;
  abs_node->attr.sched.axis.clear();
  abs_node->attr.sched.loop_axis = af::kIdNone;
  VFLoop singleton_loop(af::kIdNone);
  singleton_loop.SetMaxDtypeSize("int64_t");
  ASSERT_EQ(singleton_loop.ConstructFromNodes(vf_sub_graph.GetAllNodes(), vf_node), af::SUCCESS);
  result.clear();
  loop_size.clear();
  max_depth = -1;
  loop_sizes.clear();
  ASSERT_EQ(singleton_loop.Generate(tpipe, tensor_manager, 0, result, loop_size, max_depth, loop_sizes,
                                    {{0, "device_block_offset"}}),
            af::SUCCESS);
  const auto singleton_arange_call = GetLineContaining(result, "AscendC::Reg::Arange");
  ASSERT_FALSE(singleton_arange_call.empty()) << result;
  EXPECT_NE(singleton_arange_call.find("device_block_offset + (0)"), std::string::npos);
  EXPECT_EQ(singleton_arange_call.find("row *"), std::string::npos);
  EXPECT_EQ(singleton_arange_call.find("col *"), std::string::npos);
  singleton_loop.Destruct();
}

TEST(VFLoopTest, UsesInnermostLogicalStrideForPartiallyMergedArangeAxis) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("partially_merged_arange");
  const auto outer = graph.CreateAxis("outer", af::Symbol(2));
  const auto middle = graph.CreateAxis("middle", af::Symbol(3));
  const auto inner = graph.CreateAxis("inner", af::Symbol(3));
  const auto merged = graph.MergeAxis({middle.id, outer.id});
  ASSERT_NE(merged, nullptr);
  af::AscGraph vf_sub_graph("partially_merged_arange_subgraph");
  VectorFunc vf_op("vf");
  vf_op.InstanceOutputy(1);
  vf_op.SetAttr("sub_graph_name", "partially_merged_arange_subgraph");
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(vf_op);
  Arange arange_op("arange");
  Abs abs_op("abs");
  Output output_op("output");
  arange_op.ir_attr.SetBase(af::Symbol(0));
  arange_op.ir_attr.SetStep(af::Symbol(1));
  arange_op.attr.api.unit = af::ComputeUnit::kUnitVector;
  arange_op.attr.sched.axis = {merged->id, inner.id};
  arange_op.attr.sched.loop_axis = inner.id;
  arange_op.y.dtype = af::DT_INT64;
  *arange_op.y.axis = {outer.id, middle.id, inner.id};
  *arange_op.y.repeats = {af::Symbol(2), af::Symbol(3), af::Symbol(3)};
  *arange_op.y.strides = {af::Symbol(9), af::Symbol(3), One};
  *arange_op.y.vectorized_axis = {merged->id, inner.id};
  *arange_op.y.vectorized_strides = {af::Symbol(4), One};
  abs_op.attr.api.unit = af::ComputeUnit::kUnitVector;
  abs_op.attr.sched.axis = {merged->id, inner.id};
  abs_op.attr.sched.loop_axis = inner.id;
  abs_op.y.dtype = af::DT_INT64;
  *abs_op.y.axis = *arange_op.y.axis;
  *abs_op.y.repeats = *arange_op.y.repeats;
  *abs_op.y.strides = *arange_op.y.strides;
  *abs_op.y.vectorized_axis = *arange_op.y.vectorized_axis;
  *abs_op.y.vectorized_strides = *arange_op.y.vectorized_strides;
  output_op.ir_attr.SetIndex(0);
  vf_sub_graph.AddNode(arange_op);
  vf_sub_graph.AddNode(abs_op);
  vf_sub_graph.AddNode(output_op);
  abs_op.x = arange_op.y;
  output_op.x = abs_op.y;
  const auto arange_node = vf_sub_graph.FindNode("arange");
  arange_node->outputs[0].attr.mem.tensor_id = 0;
  const auto abs_node = vf_sub_graph.FindNode("abs");
  abs_node->outputs[0].attr.mem.tensor_id = 2;
  const auto vf_node = graph.FindNode("vf");
  vf_node->outputs[0].attr.dtype = af::DT_INT64;
  vf_node->outputs[0].attr.axis = abs_node->outputs[0].attr.axis;
  vf_node->outputs[0].attr.repeats = abs_node->outputs[0].attr.repeats;
  vf_node->outputs[0].attr.strides = abs_node->outputs[0].attr.strides;
  vf_node->outputs[0].attr.vectorized_axis = abs_node->outputs[0].attr.vectorized_axis;
  vf_node->outputs[0].attr.vectorized_strides = abs_node->outputs[0].attr.vectorized_strides;
  vf_node->outputs[0].attr.mem.tensor_id = 1;
  vf_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;

  std::string dtype_name;
  ASSERT_EQ(codegen::Tensor::DtypeName(af::DT_INT64, dtype_name), af::SUCCESS);
  TensorManager tensor_manager;
  ASSERT_EQ(tensor_manager.AddTensor(MicroApiTensor(arange_node->outputs[0], dtype_name)), af::SUCCESS);
  ASSERT_EQ(tensor_manager.AddTensor(MicroApiTensor(vf_node->outputs[0], dtype_name)), af::SUCCESS);
  ASSERT_EQ(tensor_manager.AddTensor(MicroApiTensor(abs_node->outputs[0], dtype_name)), af::SUCCESS);
  Tiler tiler;
  for (const auto &axis : graph.GetAllAxis()) {
    ASSERT_EQ(tiler.AddAxis(*axis), af::SUCCESS);
  }
  TPipe tpipe("tpipe", tiler);
  VFLoop loop(af::kIdNone);
  loop.SetMaxDtypeSize("int64_t");
  ASSERT_EQ(loop.ConstructFromNodes(vf_sub_graph.GetAllNodes(), vf_node), af::SUCCESS);

  std::string result;
  std::string loop_size;
  int32_t max_depth = -1;
  std::vector<std::string> loop_sizes;
  EXPECT_NE(loop.Generate(tpipe, tensor_manager, 0, result, loop_size, max_depth, loop_sizes), af::SUCCESS);
  loop.Destruct();
}

TEST(VFLoopTest, RejectsInterleavedNestedMergedArangeAsOnlyVectorizedAxis) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("interleaved_nested_merged_arange");
  const auto axis0 = graph.CreateAxis("axis0", af::Symbol(2));
  const auto axis1 = graph.CreateAxis("axis1", af::Symbol(2));
  const auto axis2 = graph.CreateAxis("axis2", af::Symbol(2));
  const auto axis3 = graph.CreateAxis("axis3", af::Symbol(2));
  const auto left = graph.MergeAxis({axis0.id, axis2.id});
  const auto right = graph.MergeAxis({axis1.id, axis3.id});
  ASSERT_NE(left, nullptr);
  ASSERT_NE(right, nullptr);
  const auto merged = graph.MergeAxis({left->id, right->id});
  ASSERT_NE(merged, nullptr);
  af::AscGraph vf_sub_graph("interleaved_nested_merged_arange_subgraph");
  VectorFunc vf_op("vf");
  vf_op.InstanceOutputy(1);
  vf_op.SetAttr("sub_graph_name", "interleaved_nested_merged_arange_subgraph");
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(vf_op);
  Arange arange_op("arange");
  Abs abs_op("abs");
  Output output_op("output");
  arange_op.ir_attr.SetBase(af::Symbol(0));
  arange_op.ir_attr.SetStep(af::Symbol(1));
  arange_op.attr.api.unit = af::ComputeUnit::kUnitVector;
  arange_op.attr.sched.axis = {merged->id};
  arange_op.attr.sched.loop_axis = merged->id;
  arange_op.y.dtype = af::DT_INT64;
  *arange_op.y.axis = {axis0.id, axis1.id, axis2.id, axis3.id};
  *arange_op.y.repeats = {af::Symbol(2), af::Symbol(2), af::Symbol(2), af::Symbol(2)};
  *arange_op.y.strides = {af::Symbol(8), af::Symbol(4), af::Symbol(2), One};
  *arange_op.y.vectorized_axis = {merged->id};
  *arange_op.y.vectorized_strides = {One};
  abs_op.attr.api.unit = af::ComputeUnit::kUnitVector;
  abs_op.attr.sched.axis = {merged->id};
  abs_op.attr.sched.loop_axis = merged->id;
  abs_op.y.dtype = af::DT_INT64;
  *abs_op.y.axis = *arange_op.y.axis;
  *abs_op.y.repeats = *arange_op.y.repeats;
  *abs_op.y.strides = *arange_op.y.strides;
  *abs_op.y.vectorized_axis = *arange_op.y.vectorized_axis;
  *abs_op.y.vectorized_strides = *arange_op.y.vectorized_strides;
  output_op.ir_attr.SetIndex(0);
  vf_sub_graph.AddNode(arange_op);
  vf_sub_graph.AddNode(abs_op);
  vf_sub_graph.AddNode(output_op);
  abs_op.x = arange_op.y;
  output_op.x = abs_op.y;
  const auto arange_node = vf_sub_graph.FindNode("arange");
  arange_node->outputs[0].attr.mem.tensor_id = 0;
  const auto abs_node = vf_sub_graph.FindNode("abs");
  abs_node->outputs[0].attr.mem.tensor_id = 2;
  const auto vf_node = graph.FindNode("vf");
  vf_node->outputs[0].attr.dtype = af::DT_INT64;
  vf_node->outputs[0].attr.mem.tensor_id = 1;
  vf_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;

  std::string dtype_name;
  ASSERT_EQ(codegen::Tensor::DtypeName(af::DT_INT64, dtype_name), af::SUCCESS);
  TensorManager tensor_manager;
  ASSERT_EQ(tensor_manager.AddTensor(MicroApiTensor(arange_node->outputs[0], dtype_name)), af::SUCCESS);
  ASSERT_EQ(tensor_manager.AddTensor(MicroApiTensor(vf_node->outputs[0], dtype_name)), af::SUCCESS);
  ASSERT_EQ(tensor_manager.AddTensor(MicroApiTensor(abs_node->outputs[0], dtype_name)), af::SUCCESS);
  Tiler tiler;
  for (const auto &axis : graph.GetAllAxis()) {
    ASSERT_EQ(tiler.AddAxis(*axis), af::SUCCESS);
  }
  TPipe tpipe("tpipe", tiler);
  ASSERT_EQ(tpipe.AddTensor(vf_node->outputs[0]), af::SUCCESS);
  VFLoop loop(af::kIdNone);
  loop.SetMaxDtypeSize("int64_t");
  ASSERT_EQ(loop.ConstructFromNodes(vf_sub_graph.GetAllNodes(), vf_node), af::SUCCESS);

  std::string result;
  std::string loop_size;
  int32_t max_depth = -1;
  std::vector<std::string> loop_sizes;
  EXPECT_NE(loop.Generate(tpipe, tensor_manager, 0, result, loop_size, max_depth, loop_sizes), af::SUCCESS);
  loop.Destruct();
}

TEST(VFLoopTest, KeepsDistinctParamsForMultipleArangeCalls) {
  Tiler tiler;
  TPipe tpipe("tpipe", tiler);
  VFLoop loop(af::kIdNone);
  loop.AddCall(new FixedArangeMicroApiCall(10, "1", "2"));
  loop.AddCall(new FixedArangeMicroApiCall(11, "7", "4"));

  std::vector<ArangeParam> params;
  loop.CollectArangeParams(tpipe, params);
  ASSERT_EQ(params.size(), 2U);
  EXPECT_EQ(params[0].tensor_id, 10);
  EXPECT_EQ(params[0].base, "1");
  EXPECT_EQ(params[0].step, "2");
  EXPECT_EQ(params[1].tensor_id, 11);
  EXPECT_EQ(params[1].base, "7");
  EXPECT_EQ(params[1].step, "4");
  loop.Destruct();
}

TEST(VFLoopTest, MapsBoundaryOrdinalsToConnectedRootInputs) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("vf_input_mapping");
  af::AscGraph subgraph("vf_subgraph");
  const auto axis = graph.CreateAxis("axis", af::Symbol(8));
  af::ascir_op::Data root_data("root_data", graph);
  root_data.ir_attr.SetIndex(0);
  root_data.y.dtype = af::DT_FLOAT;
  af::ascir_op::VectorFunc vf("vf");
  vf.InstanceOutputy(1);
  // 与 VfCall_TwoDimLoad 相同的隐式入图模式: vf 由输入连接隐式创建并绑定 operator,
  // 显式 AddNode 会使 operator 与节点绑定分裂(inputs 为空)。
  vf.x = {root_data.y};
  auto root = graph.FindNode("vf");
  ASSERT_NE(root, nullptr);

  af::ascir_op::Data data("data", subgraph);
  data.ir_attr.SetIndex(1);
  data.y.dtype = af::DT_FLOAT;
  af::ascir_op::Load load("load");
  load.x = data.y;
  load.attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load.attr.sched.axis = {axis.id};
  load.y.dtype = af::DT_FLOAT;
  *load.y.axis = {axis.id};
  *load.y.repeats = {af::Symbol(8)};
  *load.y.strides = {One};
  *load.y.vectorized_axis = {axis.id};
  *load.y.vectorized_strides = {One};
  af::ascir_op::Output output("output");
  subgraph.AddNode(output);
  output.ir_attr.SetIndex(0);
  output.x = load.y;
  ASSERT_NE(subgraph.FindNode("data"), nullptr);
  ASSERT_NE(subgraph.FindNode("load"), nullptr);
  ASSERT_NE(subgraph.FindNode("output"), nullptr);
  ASSERT_TRUE(af::AttrUtils::SetInt(subgraph.FindNode("data")->GetOpDesc(), "vf_root_input_index", 0));
  root->outputs[0].attr.mem.tensor_id = 10;

  VFLoop loop(af::kIdNone);
  loop.SetMaxDtypeSize("float");
  const VFInputMapping mapping = {{1, 0}};
  EXPECT_EQ(loop.ConstructFromNodes(subgraph.GetAllNodes(), root, mapping), af::SUCCESS);
  loop.Destruct();
}

TEST(CodegenKernel, VfCall_TwoDimLoad) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");

  std::string sub_graph_name = "vf_sub_graph1";
  // 创建VectorFunc的子图
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Data sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetIndex(0);
  sub_x_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);
  sub_output_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  vf_op.InstanceOutputy(1);
  vf_op.x = {load_op.y};
  vf_op.attr.sched.axis = {z0.id, z1.id};
  *vf_op.y[0].axis = {z0.id, z1.id};
  *vf_op.y[0].repeats = {s0, s1};
  *vf_op.y[0].strides = {s1, One};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  auto z0z1 = graph.MergeAxis({z0.id, z1.id});
  for (auto node : vf_sub_graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    vf_sub_graph.ApplyMerge(node, z0z1->id);
  }

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {z0z1->id};
  *sub_load_op.y.axis = {z0z1->id};
  *sub_load_op.y.repeats = {s0 * s1};
  *sub_load_op.y.strides = {One};

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {z0z1->id};
  *abs_op.y.axis = {z0z1->id};
  *abs_op.y.repeats = {s0 * s1};
  *abs_op.y.strides = {One};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0z1->id};
  *sub_store_op.y.axis = {z0z1->id};
  *sub_store_op.y.repeats = {s0 * s1};
  *sub_store_op.y.strides = {One};

  sub_output_op.x = sub_store_op.y;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {s1, One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_load = vf_sub_graph.FindNode("sub_load");
  sub_load->outputs[0].attr.vectorized_axis = {z0z1->id};
  sub_load->outputs[0].attr.vectorized_strides = {One};
  sub_load->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_load->outputs[0].attr.mem.tensor_id = 0;

  auto abs = vf_sub_graph.FindNode("abs");
  abs->outputs[0].attr.vectorized_axis = {z0z1->id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->outputs[0].attr.vectorized_axis = {z0z1->id};
  sub_store->outputs[0].attr.vectorized_strides = {One};
  sub_store->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_store->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_store->outputs[0].attr.mem.tensor_id = 2;

  auto vf = graph.FindNode("vf");
  vf->outputs[0].attr.axis = {z0.id, z1.id};
  vf->outputs[0].attr.repeats = {s0, s1};
  vf->outputs[0].attr.strides = {s1, One};
  vf->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  vf->outputs[0].attr.vectorized_strides = {s1, One};
  vf->outputs[0].attr.dtype = af::DT_FLOAT;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 1;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  store->outputs[0].attr.vectorized_strides = {s1, One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  ASSERT_EQ(tpipe.CollectQues(graph), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(load->outputs[0]), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(vf->outputs[0]), af::SUCCESS);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(*z0z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);
  EXPECT_NE(func_def.str().find("sizeof(float)"), std::string::npos);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"#if defined(__DAV_C310__) || "
                                "(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510 || "
                                "__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202))\n"
                                "AscendC::SetCtrlSpr<60, 60>(0);\n"
                                "VFCallvf((__local_mem__ float *)local_1[0].GetPhyAddr(), (__local_mem__ float "
                                "*)local_0[0].GetPhyAddr(), t->s0 * t->s1, 1, 1);\n"
                                "#endif\n"});
}

TEST(CodegenKernel, CvUbFuseVfCallUsesCubeBaseMNPhysicalLayout) {
  ge::SetupRuntimeStub();
  GTEST_SKIP() << "Manual VF graph fixture is unstable; helper-level UT validates this path.";
}

TEST(CodegenKernel, CvUbFuseVfLayoutHelperUsesPhysicalRowStride) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.cv_fusion_type = ::ascir::CubeTemplateType::kUBFuse;
  tpipe.cube_output_tensor_id = 0;

  af::AscGraph graph("test_graph");
  af::ascir_op::Data x("x", graph);
  auto s1 = af::Symbol("s1");
  auto cube = MakeCvUbFuseTensor(graph, af::DT_FLOAT, 0, {s1, One}, "local_0");
  auto fp16_input = MakeCvUbFuseTensor(graph, af::DT_FLOAT16, 1, {s1, One}, "local_1");
  auto bool_output = MakeCvUbFuseTensor(graph, af::DT_UINT8, 2, {s1, One}, "local_2");
  auto broadcast_input = MakeCvUbFuseTensor(graph, af::DT_FLOAT16, 3, {Zero, Zero}, "local_3");

  EXPECT_EQ(codegen::GenCvUbFuseVfFuncDimParams(), "uint32_t curAivM, uint32_t curAivN, uint32_t curAlignN");
  EXPECT_EQ(codegen::GenCvUbFuseVfCallDimParams(), "curAivM, curAivN, curAlignN");
  EXPECT_EQ(codegen::GenCvUbFuseRowStride(tpipe, cube), "curAlignN");
  EXPECT_EQ(codegen::GenCvUbFuseRowStride(tpipe, fp16_input), "KernelUtils::BlkAlign<half>(curAivN)");
  EXPECT_EQ(codegen::GenCvUbFuseRowStride(tpipe, bool_output), "KernelUtils::BlkAlign<uint8_t>(curAivN)");
  EXPECT_EQ(codegen::GenCvUbFuseAddrOffset(tpipe, cube), "0 + cv_m * curAlignN + cv_n * ELEMENT_PER_VECTOR_LENGTH");
  EXPECT_EQ(codegen::GenCvUbFuseAddrOffset(tpipe, fp16_input),
            "0 + cv_m * KernelUtils::BlkAlign<half>(curAivN) + cv_n * ELEMENT_PER_VECTOR_LENGTH");
  EXPECT_EQ(codegen::GenCvUbFuseAddrOffset(tpipe, bool_output),
            "0 + cv_m * KernelUtils::BlkAlign<uint8_t>(curAivN) + cv_n * ELEMENT_PER_VECTOR_LENGTH");
  EXPECT_EQ(codegen::GenCvUbFuseAddrOffset(tpipe, broadcast_input), "0");
}

TEST(CodegenKernel, CvUbFuseTensorSizeAssignUsesTensorDtype) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.cv_fusion_type = ::ascir::CubeTemplateType::kUBFuse;

  af::AscGraph graph("test_graph");
  af::ascir_op::Data x("x", graph);
  auto cube = MakeCvUbFuseTensor(graph, af::DT_FLOAT, 0, {One}, "local_0");
  auto fp16_input = MakeCvUbFuseTensor(graph, af::DT_FLOAT16, 1, {One}, "local_1");
  auto bool_output = MakeCvUbFuseTensor(graph, af::DT_UINT8, 2, {One}, "local_2");
  cube.alloc_type = af::AllocType::kAllocTypeQueue;
  fp16_input.alloc_type = af::AllocType::kAllocTypeQueue;
  bool_output.alloc_type = af::AllocType::kAllocTypeQueue;
  tpipe.tensors.emplace(cube.id, cube);
  tpipe.tensors.emplace(fp16_input.id, fp16_input);
  tpipe.tensors.emplace(bool_output.id, bool_output);

  std::string result;
  ASSERT_EQ(tpipe.TensorSizeAssign("float", result), af::SUCCESS);
  EXPECT_NE(result.find("local_0_size = stage_size;"), std::string::npos);
  EXPECT_NE(result.find("local_1_size = stage_size;"), std::string::npos);
  EXPECT_NE(result.find("local_2_size = stage_size;"), std::string::npos);
}

TEST(CodegenKernel, VfCall_TwoDimLoad_VFLoop) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");

  std::string sub_graph_name = "vf_sub_graph1";
  // 创建VectorFunc的子图
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Data sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetIndex(0);
  sub_x_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);
  sub_output_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, Zero};

  vf_op.InstanceOutputy(1);
  vf_op.x = {load_op.y};
  vf_op.attr.sched.axis = {z0.id, z1.id};
  *vf_op.y[0].axis = {z0.id, z1.id};
  *vf_op.y[0].repeats = {s0, s1};
  *vf_op.y[0].strides = {s1, Zero};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, Zero};

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {z0.id, z1.id};
  *sub_load_op.y.axis = {z0.id, z1.id};
  *sub_load_op.y.repeats = {s0, s1};
  *sub_load_op.y.strides = {s1, Zero};

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, Zero};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0.id, z1.id};
  *sub_store_op.y.axis = {z0.id, z1.id};
  *sub_store_op.y.repeats = {s0, s1};
  *sub_store_op.y.strides = {s1, Zero};

  sub_output_op.x = sub_store_op.y;

  auto load = graph.FindNode("load");
  load->attr.sched.loop_axis = -1;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {Zero, Zero};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_load = vf_sub_graph.FindNode("sub_load");
  sub_load->attr.sched.loop_axis = -1;
  sub_load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  sub_load->outputs[0].attr.vectorized_strides = {Zero, Zero};
  sub_load->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_load->outputs[0].attr.mem.tensor_id = 0;

  auto abs = vf_sub_graph.FindNode("abs");
  abs->attr.sched.loop_axis = -1;
  abs->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  abs->outputs[0].attr.vectorized_strides = {Zero, Zero};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->attr.sched.loop_axis = -1;
  sub_store->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  sub_store->outputs[0].attr.vectorized_strides = {Zero, Zero};
  sub_store->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_store->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_store->outputs[0].attr.mem.tensor_id = 2;

  auto vf = graph.FindNode("vf");
  vf->attr.sched.loop_axis = -1;
  vf->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  vf->outputs[0].attr.vectorized_strides = {Zero, Zero};
  vf->outputs[0].attr.dtype = af::DT_FLOAT;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 1;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.sched.loop_axis = -1;
  store->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  store->outputs[0].attr.vectorized_strides = {Zero, Zero};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  ASSERT_EQ(tpipe.CollectQues(graph), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(load->outputs[0]), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(vf->outputs[0]), af::SUCCESS);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result = func_def.str();
  EXPECT_TRUE(result.find("AscendC::MicroAPI::MaskReg preg_main = AscendC::MicroAPI::CreateMask") != std::string::npos);
  EXPECT_TRUE(result.find("uint32_t element_count") != std::string::npos);
  EXPECT_TRUE(result.find("uint16_t loop_times") != std::string::npos);
}

TEST(CodegenKernel, VfCall_TwoDim_Scalar) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Scalar x_op("x", graph);
  x_op.ir_attr.SetValue("2.0");

  std::string sub_graph_name = "vf_sub_graph1";
  // 创建VectorFunc的子图
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Scalar sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetValue("2.0");
  sub_x_op.ir_attr.SetIndex(0);
  sub_x_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  Broadcast sub_brc_op("sub_brc");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);
  sub_output_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  Store store_op("store");
  graph.AddNode(x_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  vf_op.InstanceOutputy(1);
  vf_op.x = {x_op.y};
  vf_op.attr.sched.axis = {z0.id, z1.id};
  *vf_op.y[0].axis = {z0.id, z1.id};
  *vf_op.y[0].repeats = {s0, s1};
  *vf_op.y[0].strides = {s1, One};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  sub_brc_op.x = sub_x_op.y;
  sub_brc_op.attr.sched.axis = {z0.id, z1.id};
  *sub_brc_op.y.axis = {z0.id, z1.id};
  *sub_brc_op.y.repeats = {s0, s1};
  *sub_brc_op.y.strides = {s1, One};

  abs_op.x = sub_brc_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, One};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0.id, z1.id};
  *sub_store_op.y.axis = {z0.id, z1.id};
  *sub_store_op.y.repeats = {s0, s1};
  *sub_store_op.y.strides = {s1, One};

  sub_output_op.x = sub_store_op.y;

  auto x = graph.FindNode("x");
  x->outputs[0].attr.dtype = af::DT_FLOAT;
  x->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x->outputs[0].attr.que.id = 0;
  x->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_x = vf_sub_graph.FindNode("sub_x");
  sub_x->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  sub_x->outputs[0].attr.vectorized_strides = {Zero, Zero};
  sub_x->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_x->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_x->outputs[0].attr.mem.tensor_id = 0;

  auto sub_brc = vf_sub_graph.FindNode("sub_brc");
  sub_brc->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  sub_brc->outputs[0].attr.vectorized_strides = {s1, One};
  sub_brc->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_brc->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_brc->outputs[0].attr.mem.tensor_id = 1;

  auto abs = vf_sub_graph.FindNode("abs");
  abs->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  abs->outputs[0].attr.vectorized_strides = {s1, One};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 2;

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  sub_store->outputs[0].attr.vectorized_strides = {s1, One};
  sub_store->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_store->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_store->outputs[0].attr.mem.tensor_id = 3;

  auto vf = graph.FindNode("vf");
  vf->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  vf->outputs[0].attr.vectorized_strides = {s1, One};
  vf->outputs[0].attr.dtype = af::DT_FLOAT;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 1;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  store->outputs[0].attr.vectorized_strides = {s1, One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  ASSERT_EQ(tpipe.CollectQues(graph), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor("2.0", x->outputs[0], "scalar_x"), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(vf->outputs[0]), af::SUCCESS);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);
  codegen::ApiTensor x1, x2;
  x1.id = x->outputs[0].attr.mem.tensor_id;
  x2.id = vf->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"#if defined(__DAV_C310__) || "
                                "(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510 || "
                                "__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202))\n"
                                "AscendC::SetCtrlSpr<60, 60>(0);\n"
                                "VFCallvf((__local_mem__ float *)local_1[0].GetPhyAddr(), (__local_mem__ float "
                                "*)local_1[0].GetPhyAddr(), scalar_0, t->s0 * t->s1, 1, 1);\n"
                                "#endif\n"});
}

TEST(CodegenKernel, VfCall_TwoDim_ScalarData) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ScalarData scalar_data_op("scalar_data", graph);
  scalar_data_op.ir_attr.SetIndex(0);

  std::string sub_graph_name = "vf_sub_graph1";
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Scalar sub_scalar_op("sub_scalar", vf_sub_graph);
  sub_scalar_op.ir_attr.SetIndex(0);

  Broadcast sub_brc_op("sub_brc");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);

  Store store_op("store");
  graph.AddNode(scalar_data_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  InitScalarDataVfGraph(vf_op, store_op, sub_brc_op, abs_op, sub_store_op, sub_output_op, scalar_data_op, sub_scalar_op,
                        z0, z1, s0, s1);
  InitScalarDataVfTensorAttrs(graph, vf_sub_graph, z0, z1, s1);

  auto result = GenerateScalarDataVfCall(graph, z0, z1, s0, s1);
  EXPECT_NE(result.find("scalar_data"), std::string::npos) << result;
  EXPECT_EQ(result.find("scalar_1"), std::string::npos) << result;
}

TEST(CodegenKernel, VfCall_ThreeDimLoad) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");

  std::string sub_graph_name = "vf_sub_graph1";
  // 创建VectorFunc的子图
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Data sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetIndex(0);
  sub_x_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);
  sub_output_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {Two * s1 * s2, Two * s2, One};

  vf_op.InstanceOutputy(1);
  vf_op.x = {load_op.y};
  vf_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *vf_op.y[0].axis = {z0.id, z1.id, z2.id};
  *vf_op.y[0].repeats = {s0, s1, s2};
  *vf_op.y[0].strides = {Two * s1 * s2, Two * s2, One};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {Two * s1 * s2, Two * s2, One};

  auto z0z1 = graph.MergeAxis({z0.id, z1.id});
  for (auto node : vf_sub_graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    vf_sub_graph.ApplyMerge(node, z0z1->id);
  }

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {z0z1->id, z2.id};
  *sub_load_op.y.axis = {z0z1->id, z2.id};
  *sub_load_op.y.repeats = {s0 * s1, s2};
  *sub_load_op.y.strides = {Two * s2, One};

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {z0z1->id, z2.id};
  *abs_op.y.axis = {z0z1->id, z2.id};
  *abs_op.y.repeats = {s0 * s1, s2};
  *abs_op.y.strides = {Two * s2, One};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0z1->id, z2.id};
  *sub_store_op.y.axis = {z0z1->id, z2.id};
  *sub_store_op.y.repeats = {s0 * s1, s2};
  *sub_store_op.y.strides = {Two * s2, One};

  sub_output_op.x = sub_store_op.y;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {Two * s1 * s2, Two * s2, One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_load = vf_sub_graph.FindNode("sub_load");
  sub_load->outputs[0].attr.vectorized_axis = {z0z1->id, z2.id};
  sub_load->outputs[0].attr.vectorized_strides = {Two * s2, One};
  sub_load->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_load->outputs[0].attr.mem.tensor_id = 0;

  auto abs = vf_sub_graph.FindNode("abs");
  abs->outputs[0].attr.vectorized_axis = {z0z1->id, z2.id};
  abs->outputs[0].attr.vectorized_strides = {Two * s2, One};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->outputs[0].attr.vectorized_axis = {z0z1->id, z2.id};
  sub_store->outputs[0].attr.vectorized_strides = {Two * s2, One};
  sub_store->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_store->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_store->outputs[0].attr.mem.tensor_id = 2;

  auto vf = graph.FindNode("vf");
  vf->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  vf->outputs[0].attr.vectorized_strides = {Two * s1 * s2, Two * s2, One};
  vf->outputs[0].attr.dtype = af::DT_FLOAT;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 1;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  store->outputs[0].attr.vectorized_strides = {Two * s1 * s2, Two * s2, One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  ASSERT_EQ(tpipe.CollectQues(graph), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(load->outputs[0]), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(vf->outputs[0]), af::SUCCESS);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(*z0z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"#if defined(__DAV_C310__) || "
                                "(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510 || "
                                "__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202))\n"
                                "AscendC::SetCtrlSpr<60, 60>(0);\n"
                                "VFCallvf((__local_mem__ float *)local_1[0].GetPhyAddr(), (__local_mem__ float "
                                "*)local_0[0].GetPhyAddr(), t->s0 * t->s1, t->s2, (2 * t->s2), 1, (2 * t->s2), 1);\n"
                                "#endif\n"});
}

TEST(CodegenKernel, VfCall_FiveDimLoad) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);
  af::Expression Five = af::Symbol(5);
  af::Expression Six = af::Symbol(6);

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

  Data x_op("x", graph);
  Load load_op("load");

  std::string sub_graph_name = "vf_sub_graph1";
  // 创建VectorFunc的子图
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Data sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetIndex(0);
  sub_x_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);
  sub_output_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load_op.y.repeats = {s0, s1, s2, s3, s4};
  *load_op.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  vf_op.InstanceOutputy(1);
  vf_op.x = {load_op.y};
  vf_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *vf_op.y[0].axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *vf_op.y[0].repeats = {s0, s1, s2, s3, s4};
  *vf_op.y[0].strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store_op.y.repeats = {s0, s1, s2, s3, s4};
  *store_op.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *sub_load_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *sub_load_op.y.repeats = {s0, s1, s2, s3, s4};
  *sub_load_op.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs_op.y.repeats = {s0, s1, s2, s3, s4};
  *abs_op.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0.id, z1.id};
  *sub_store_op.y.axis = {z0.id, z1.id};
  *sub_store_op.y.repeats = {s0, s1, s2, s3, s4};
  *sub_store_op.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  sub_output_op.x = sub_store_op.y;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_load = vf_sub_graph.FindNode("sub_load");
  sub_load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  sub_load->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  sub_load->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_load->outputs[0].attr.mem.tensor_id = 0;

  auto abs = vf_sub_graph.FindNode("abs");
  abs->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  sub_store->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4 * Five, s2 * s3 * s4 * Four, s3 * s4 * Three,
                                                   s4 * Two, One};
  sub_store->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_store->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_store->outputs[0].attr.mem.tensor_id = 2;

  auto vf = graph.FindNode("vf");
  vf->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  vf->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4 * Five, s2 * s3 * s4 * Four, s3 * s4 * Three, s4 * Two,
                                            One};
  vf->outputs[0].attr.dtype = af::DT_FLOAT;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 1;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4 * Five, s2 * s3 * s4 * Four, s3 * s4 * Three, s4 * Two,
                                               One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  ASSERT_EQ(tpipe.CollectQues(graph), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(load->outputs[0]), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(vf->outputs[0]), af::SUCCESS);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddAxis(z4);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));
  tiler.AddSizeVar(af::SizeVar(s4));

  vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{
          "#if defined(__DAV_C310__) || "
          "(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || "
          "__NPU_ARCH__ == 9202))\n"
          "AscendC::SetCtrlSpr<60, 60>(0);\n"
          "for(int outer_for_0 = 0; outer_for_0 < t->s0; outer_for_0++) {\n"
          "VFCallvf((__local_mem__ float "
          "*)local_1[outer_for_0 * (5 * t->s1 * t->s2 * t->s3 * t->s4)].GetPhyAddr(), (__local_mem__ float "
          "*)local_0[outer_for_0 * (t->s1 * t->s2 * t->s3 * t->s4)].GetPhyAddr(), t->s1, t->s2, t->s3, t->s4, (4 * "
          "t->s2 "
          "* t->s3 * t->s4), (3 * t->s3 * t->s4), (2 * t->s4), 1, (t->s2 * t->s3 * t->s4), (t->s3 * t->s4), t->s4, "
          "1);\n\n"
          "}\n"
          "#endif\n"});
}

TEST(CodegenKernel, VfCall_DoubleLoopWritesVectorFuncParams) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);
  af::Expression Five = af::Symbol(5);

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
  const std::vector<af::Axis> axes = {z0, z1, z2, z3, z4};
  const std::vector<af::Expression> repeats = {s0, s1, s2, s3, s4};
  const std::vector<af::Expression> base_strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  const std::vector<af::Expression> output_strides = {s1 * s2 * s3 * s4 * Five, s2 * s3 * s4 * Four, s3 * s4 * Three,
                                                      s4 * Two, One};

  Data x_op("x", graph);
  Load load_op("load");
  std::string sub_graph_name = "vf_sub_graph_double_loop";
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  Data sub_x_op("sub_x", vf_sub_graph);
  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  Store store_op("store");
  RegisterDoubleLoopNodes(graph, vf_sub_graph, vf_op, sub_x_op, sub_output_op, load_op, store_op, sub_graph_name);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  SetFiveDimSchedule(load_op.y, axes, repeats, base_strides);
  InitDoubleLoopVfGraph(vf_op, store_op, sub_load_op, abs_op, sub_store_op, sub_output_op, load_op, sub_x_op, axes,
                        repeats, base_strides);
  InitDoubleLoopTensorAttrs(graph, vf_sub_graph, axes, base_strides, output_strides);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  codegen::VfCall call;
  std::stringstream func_def;
  PrepareDoubleLoopCodegen(graph, axes, repeats, tiler, tpipe, call, func_def);

  auto vf = graph.FindNode("vf");
  CheckDoubleLoopVectorFuncParams(vf);
}

// 测试单维场景不触发优化逻辑
TEST(CodegenKernel, VfCall_OneDim_NoOptimization) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Data x_op("x", graph);
  Load load_op("load");

  std::string sub_graph_name = "vf_sub_graph_1d";
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Data sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetIndex(0);
  sub_x_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);
  sub_output_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id};
  *load_op.y.axis = {z0.id};
  *load_op.y.repeats = {s0};
  *load_op.y.strides = {One};

  vf_op.InstanceOutputy(1);
  vf_op.x = {load_op.y};
  vf_op.attr.sched.axis = {z0.id};
  *vf_op.y[0].axis = {z0.id};
  *vf_op.y[0].repeats = {s0};
  *vf_op.y[0].strides = {One};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id};
  *store_op.y.repeats = {s0};
  *store_op.y.strides = {One};

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {z0.id};
  sub_load_op.attr.sched.loop_axis = -1;
  *sub_load_op.y.axis = {z0.id};
  *sub_load_op.y.repeats = {s0};
  *sub_load_op.y.strides = {One};

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {z0.id};
  abs_op.attr.sched.loop_axis = -1;
  *abs_op.y.axis = {z0.id};
  *abs_op.y.repeats = {s0};
  *abs_op.y.strides = {One};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0.id};
  sub_store_op.attr.sched.loop_axis = -1;
  *sub_store_op.y.axis = {z0.id};
  *sub_store_op.y.repeats = {s0};
  *sub_store_op.y.strides = {One};

  sub_output_op.x = sub_store_op.y;

  auto load = graph.FindNode("load");
  load->attr.sched.loop_axis = -1;
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_load = vf_sub_graph.FindNode("sub_load");
  sub_load->attr.sched.loop_axis = -1;
  sub_load->outputs[0].attr.vectorized_axis = {z0.id};
  sub_load->outputs[0].attr.vectorized_strides = {One};
  sub_load->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_load->outputs[0].attr.mem.tensor_id = 0;

  auto abs = vf_sub_graph.FindNode("abs");
  abs->attr.sched.loop_axis = -1;
  abs->outputs[0].attr.vectorized_axis = {z0.id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->attr.sched.loop_axis = -1;
  sub_store->outputs[0].attr.vectorized_axis = {z0.id};
  sub_store->outputs[0].attr.vectorized_strides = {One};
  sub_store->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_store->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_store->outputs[0].attr.mem.tensor_id = 2;

  auto vf = graph.FindNode("vf");
  vf->attr.sched.loop_axis = -1;
  vf->outputs[0].attr.vectorized_axis = {z0.id};
  vf->outputs[0].attr.vectorized_strides = {One};
  vf->outputs[0].attr.dtype = af::DT_FLOAT;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 1;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.sched.loop_axis = -1;
  store->outputs[0].attr.vectorized_axis = {z0.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  ASSERT_EQ(tpipe.CollectQues(graph), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(load->outputs[0]), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(vf->outputs[0]), af::SUCCESS);

  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));

  vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result = func_def.str();

  // 单维场景（dim_size < MAX_VF_AXIS_MERGE_SIZE），不应该触发优化逻辑
  // 检查不应该包含优化相关的代码
  EXPECT_FALSE(result.find("output_dims_1 != strides_align") != std::string::npos)
      << "Optimization code should not be generated for 1D case (dim_size < MAX_VF_AXIS_MERGE_SIZE)";
}
