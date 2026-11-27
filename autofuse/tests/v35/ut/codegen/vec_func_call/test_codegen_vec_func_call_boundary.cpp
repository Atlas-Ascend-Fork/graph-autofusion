/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
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
#include "../common.h"
#include "codegen_graph_check.h"

using namespace std;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

namespace {
constexpr int64_t kRedirectedTensorId = 77;
const std::string kSubGraphName = "vf_sub_graph_redirected";

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

void SetVecInTensor(AscTensor &tensor, const af::Axis &z0, const int64_t tensor_id) {
  tensor.attr.vectorized_axis = {z0.id};
  tensor.attr.vectorized_strides = {One};
  tensor.attr.dtype = af::DT_FLOAT;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.mem.tensor_id = tensor_id;
}

void SetQueueAlloc(AscTensor &tensor, const int64_t que_id) {
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  tensor.attr.que.id = que_id;
  tensor.attr.opt.merge_scope = af::kIdNone;
}

// 子图链路：sub_x -> sub_load -> abs -> sub_store -> sub_output。
void BuildRedirectedSubGraph(af::AscGraph &vf_sub_graph, const af::Axis &z0, const af::Expression &s0) {
  Data sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetIndex(0);
  sub_x_op.attr.api.unit = af::ComputeUnit::kUnitNone;
  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);
  sub_output_op.attr.api.unit = af::ComputeUnit::kUnitNone;

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
}

// 根图链路：x -> load -> vf -> store，并把子图挂到根图。
void BuildRedirectedRootGraph(af::AscGraph &graph, const af::AscGraph &vf_sub_graph, const af::Axis &z0,
                              const af::Expression &s0) {
  Data x_op("x", graph);
  Data redirected_op("redirected", graph);
  // redirected 输出需有完整调度信息：Tensor Init 校验 vectorized 轴必须存在于输出 axis 中。
  *redirected_op.y.axis = {z0.id};
  *redirected_op.y.repeats = {s0};
  *redirected_op.y.strides = {One};
  Load load_op("load");
  VectorFunc vf_op("vf");
  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);
  vf_op.SetAttr("sub_graph_name", kSubGraphName);

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
}

void SetRedirectedBoundaryTensorAttrs(af::AscGraph &graph, af::AscGraph &vf_sub_graph, const af::Axis &z0) {
  auto load = graph.FindNode("load");
  load->attr.sched.loop_axis = -1;
  SetVecInTensor(load->outputs[0], z0, 0);
  SetQueueAlloc(load->outputs[0], 1);

  auto sub_load = vf_sub_graph.FindNode("sub_load");
  sub_load->attr.sched.loop_axis = -1;
  SetVecInTensor(sub_load->outputs[0], z0, 0);

  auto abs = vf_sub_graph.FindNode("abs");
  abs->attr.sched.loop_axis = -1;
  SetVecInTensor(abs->outputs[0], z0, 1);

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->attr.sched.loop_axis = -1;
  SetVecInTensor(sub_store->outputs[0], z0, 2);

  auto vf = graph.FindNode("vf");
  vf->attr.sched.loop_axis = -1;
  SetVecInTensor(vf->outputs[0], z0, 1);
  SetQueueAlloc(vf->outputs[0], 1);

  auto store = graph.FindNode("store");
  store->attr.sched.loop_axis = -1;
  SetVecInTensor(store->outputs[0], z0, 2);
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  SetQueueAlloc(store->outputs[0], 2);
}

// 注册重定向目标 tensor：与图边快照 id(0) 不同源，模拟生产者 ApiCall 输出 id 被改写后
// 根图侧解析到的最终 id。attr 挂在独立 Data 节点上，避免污染 load 的 attr。
void RegisterRedirectedOutputTensor(af::AscGraph &graph, codegen::TPipe &tpipe, const af::Axis &z0) {
  auto redirected = graph.FindNode("redirected");
  SetVecInTensor(redirected->outputs[0], z0, kRedirectedTensorId);
  redirected->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  redirected->outputs[0].attr.buf.id = 1;
  redirected->outputs[0].attr.opt.merge_scope = af::kIdNone;
  ASSERT_EQ(tpipe.AddTensor(redirected->outputs[0]), af::SUCCESS);
}
}  // namespace

// 生产者 ApiCall 输出被重定向（如 IndirectLoad post-Reduce 改写 outputs[0].id）时，根图侧
// 签名/实参解析到重定向后的 id，而子图 boundary Load 记录的仍是分区边快照 id，二者不同源
// 会导致函数体引用无写入也无声明的旧 buffer（use of undeclared identifier）。函数定义生成
// 必须把 boundary Load 校正到最终 id。
TEST(CodegenKernel, VfCall_RewritesBoundaryLoadWhenProducerOutputRedirected) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  af::AscGraph vf_sub_graph(kSubGraphName.c_str());
  BuildRedirectedSubGraph(vf_sub_graph, z0, s0);
  BuildRedirectedRootGraph(graph, vf_sub_graph, z0, s0);
  SetRedirectedBoundaryTensorAttrs(graph, vf_sub_graph, z0);

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  ASSERT_EQ(tpipe.CollectQues(graph), af::SUCCESS);
  // TBuf tensor 的 Init 需按 vectorized_axis 在 Tiler 中查轴，须先注册轴再 AddTensor。
  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));
  ASSERT_EQ(tpipe.AddTensor(graph.FindNode("load")->outputs[0]), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(graph.FindNode("vf")->outputs[0]), af::SUCCESS);
  RegisterRedirectedOutputTensor(graph, tpipe, z0);

  // 根图侧输入（签名/实参来源）指向重定向后的 id；子图内 boundary Load 记录的仍是边快照 id。
  codegen::ApiTensor x1;
  x1.id = kRedirectedTensorId;
  codegen::VfCall call;
  EXPECT_EQ(call.Init(graph.FindNode("vf")), 0);
  call.inputs.push_back(&x1);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  const std::string result = func_def.str();
  // 签名与函数体 boundary Load 必须同源：均引用重定向后的 buffer，旧 buffer 不再出现。
  EXPECT_NE(result.find("local_77_addr"), std::string::npos) << result;
  const auto load_align_line = GetLineContaining(result, "LoadAlign");
  ASSERT_FALSE(load_align_line.empty()) << result;
  EXPECT_NE(load_align_line.find("local_77"), std::string::npos) << load_align_line;
  EXPECT_EQ(result.find("local_0"), std::string::npos) << result;
}
