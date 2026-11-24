/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* SoftmaxV2(尾轴退化输入, [100,3,1]) + 尾轴扩维 Broadcast_2(1 -> 50) + Add 的调度/代码生成诊断用例。
 * 场景：Broadcast_2 的源来自 Div_0（elementwise 生产者，非直连 Load），调度后 softmax 链 tile
 * 布局由对齐策略决定；本用例转储调度产物中各节点的 vectorized_strides，并生成 kernel 文本，
 * 用于核对 BroadcastExtend 的 src_shape 是否与源 tile 的真实布局一致。 */

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "codegen.h"
#include "optimize.h"
#include "backend_common.h"
#include "ascir_ops.h"
#include "platform_context.h"
#include "runtime_stub.h"

namespace {
using af::AxisId;
using af::Expression;

std::string ExprToStr(const Expression &expr) {
  const auto text = expr.Str();
  return (text.get() != nullptr) ? std::string(text.get()) : std::string("<null>");
}

std::string ExprVecToStr(const std::vector<Expression> &exprs) {
  std::string result = "[";
  for (size_t i = 0; i < exprs.size(); ++i) {
    if (i > 0U) {
      result += ", ";
    }
    result += ExprToStr(exprs[i]);
  }
  result += "]";
  return result;
}

// 建图上下文：AscOpOutput 内部保存栈上 Operator 的裸指针，上游算子对象必须存活到下游连边完成，
// 因此各分段建图 helper 共用的算子对象与尺寸符号统一由本结构持有。
struct SoftmaxExpandDimsAddOps {
  explicit SoftmaxExpandDimsAddOps(af::AscGraph &graph)
      : s0(graph.CreateSizeVar(100)),
        s1(graph.CreateSizeVar(3)),
        s2(graph.CreateSizeVar(50)),
        z0(graph.CreateAxis("z0", s0)),
        z1(graph.CreateAxis("z1", s1)),
        z2(graph.CreateAxis("z2", s2)),
        axis{z0.id, z1.id, z2.id},
        one(af::Symbol(1)),
        zero(af::Symbol(0)),
        load0("Softmax_out0_graph/Load_0"),
        load1("add_out0_graph/Load_1"),
        max0("Softmax_out0_graph/Max_0"),
        broadcast0("Broadcast_0_Softmax_out0_graph/Sub_0_2"),
        sub0("Softmax_out0_graph/Sub_0"),
        exp0("Softmax_out0_graph/Exp_0"),
        sum0("Softmax_out0_graph/Sum_0"),
        broadcast1("Broadcast_0_Softmax_out0_graph/Div_0_3"),
        div0("Softmax_out0_graph/Div_0"),
        broadcast2("Broadcast_0_add_out0_graph/Add_0_4"),
        add0("add_out0_graph/Add_0"),
        store0("add_out0_graph/Store_0"),
        output0("add_out0_graph/Output_0") {}

  const Expression s0;
  const Expression s1;
  const Expression s2;
  const af::Axis z0;
  const af::Axis z1;
  const af::Axis z2;
  const std::vector<AxisId> axis;
  const Expression one;
  const Expression zero;
  af::ascir_op::Load load0;
  af::ascir_op::Load load1;
  af::ascir_op::Max max0;
  af::ascir_op::Broadcast broadcast0;
  af::ascir_op::Sub sub0;
  af::ascir_op::Exp exp0;
  af::ascir_op::Sum sum0;
  af::ascir_op::Broadcast broadcast1;
  af::ascir_op::Div div0;
  af::ascir_op::Broadcast broadcast2;
  af::ascir_op::Add add0;
  af::ascir_op::Store store0;
  af::ascir_op::Output output0;
};

// 建图分段：SoftmaxV2 尾轴退化输入的 Data/Load（Data_0 构造即入图，Load_0 首次连边入图）。
void AddSoftmaxInputLoad(SoftmaxExpandDimsAddOps &ops, af::AscGraph &graph) {
  af::ascir_op::Data data0("Softmax_out0_graph/Data_0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = ops.axis;
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = af::DT_FLOAT;
  *data0.y.axis = ops.axis;
  *data0.y.repeats = {ops.s0, ops.s1, ops.one};
  *data0.y.strides = {ops.s1, ops.one, ops.zero};

  ops.load0.x = data0.y;
  ops.load0.attr.sched.axis = ops.axis;
  ops.load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  ops.load0.y.dtype = af::DT_FLOAT;
  *ops.load0.y.axis = ops.axis;
  *ops.load0.y.repeats = {ops.s0, ops.s1, ops.one};
  *ops.load0.y.strides = {ops.s1, ops.one, ops.zero};
}

// 建图分段：Add 操作数输入的 Data/Load。
void AddOperandInputLoad(SoftmaxExpandDimsAddOps &ops, af::AscGraph &graph) {
  af::ascir_op::Data data1("add_out0_graph/Data_1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = ops.axis;
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.y.dtype = af::DT_FLOAT;
  *data1.y.axis = ops.axis;
  *data1.y.repeats = {ops.s0, ops.s1, ops.s2};
  *data1.y.strides = {ops.s1 * ops.s2, ops.s2, ops.one};

  ops.load1.x = data1.y;
  ops.load1.attr.sched.axis = ops.axis;
  ops.load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  ops.load1.y.dtype = af::DT_FLOAT;
  *ops.load1.y.axis = ops.axis;
  *ops.load1.y.repeats = {ops.s0, ops.s1, ops.s2};
  *ops.load1.y.strides = {ops.s1 * ops.s2, ops.s2, ops.one};
}

// 建图分段：SoftmaxV2 的 Max/Broadcast/Sub/Exp 链。
void AddSoftmaxSubExpChain(SoftmaxExpandDimsAddOps &ops) {
  ops.max0.x = ops.load0.y;
  ops.max0.attr.sched.axis = ops.axis;
  ops.max0.attr.api.compute_type = af::ComputeType::kComputeReduce;
  ops.max0.y.dtype = af::DT_FLOAT;
  *ops.max0.y.axis = ops.axis;
  *ops.max0.y.repeats = {ops.s0, ops.one, ops.one};
  *ops.max0.y.strides = {ops.one, ops.zero, ops.zero};

  ops.broadcast0.x = ops.max0.y;
  ops.broadcast0.attr.sched.axis = ops.axis;
  ops.broadcast0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  ops.broadcast0.y.dtype = af::DT_FLOAT;
  *ops.broadcast0.y.axis = ops.axis;
  *ops.broadcast0.y.repeats = {ops.s0, ops.s1, ops.one};
  *ops.broadcast0.y.strides = {ops.s1, ops.one, ops.one};

  ops.sub0.x1 = ops.load0.y;
  ops.sub0.x2 = ops.broadcast0.y;
  ops.sub0.attr.sched.axis = ops.axis;
  ops.sub0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  ops.sub0.y.dtype = af::DT_FLOAT;
  *ops.sub0.y.axis = ops.axis;
  *ops.sub0.y.repeats = {ops.s0, ops.s1, ops.one};
  *ops.sub0.y.strides = {ops.s1, ops.one, ops.zero};

  ops.exp0.x = ops.sub0.y;
  ops.exp0.attr.sched.axis = ops.axis;
  ops.exp0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  ops.exp0.y.dtype = af::DT_FLOAT;
  *ops.exp0.y.axis = ops.axis;
  *ops.exp0.y.repeats = {ops.s0, ops.s1, ops.one};
  *ops.exp0.y.strides = {ops.s1, ops.one, ops.zero};
}

// 建图分段：SoftmaxV2 的 Sum/Broadcast/Div 链。
void AddSoftmaxSumDivChain(SoftmaxExpandDimsAddOps &ops) {
  ops.sum0.x = ops.exp0.y;
  ops.sum0.attr.sched.axis = ops.axis;
  ops.sum0.attr.api.compute_type = af::ComputeType::kComputeReduce;
  ops.sum0.y.dtype = af::DT_FLOAT;
  *ops.sum0.y.axis = ops.axis;
  *ops.sum0.y.repeats = {ops.s0, ops.one, ops.one};
  *ops.sum0.y.strides = {ops.one, ops.zero, ops.zero};

  ops.broadcast1.x = ops.sum0.y;
  ops.broadcast1.attr.sched.axis = ops.axis;
  ops.broadcast1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  ops.broadcast1.y.dtype = af::DT_FLOAT;
  *ops.broadcast1.y.axis = ops.axis;
  *ops.broadcast1.y.repeats = {ops.s0, ops.s1, ops.one};
  *ops.broadcast1.y.strides = {ops.s1, ops.one, ops.one};

  ops.div0.x1 = ops.exp0.y;
  ops.div0.x2 = ops.broadcast1.y;
  ops.div0.attr.sched.axis = ops.axis;
  ops.div0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  ops.div0.y.dtype = af::DT_FLOAT;
  *ops.div0.y.axis = ops.axis;
  *ops.div0.y.repeats = {ops.s0, ops.s1, ops.one};
  *ops.div0.y.strides = {ops.s1, ops.one, ops.zero};
}

// 建图分段：尾轴扩维 Broadcast_2 及 Add/Store/Output 收尾。
void AddExpandAddEpilogue(SoftmaxExpandDimsAddOps &ops) {
  ops.broadcast2.x = ops.div0.y;
  ops.broadcast2.attr.sched.axis = ops.axis;
  ops.broadcast2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  ops.broadcast2.y.dtype = af::DT_FLOAT;
  *ops.broadcast2.y.axis = ops.axis;
  *ops.broadcast2.y.repeats = {ops.s0, ops.s1, ops.s2};
  *ops.broadcast2.y.strides = {ops.s1 * ops.s2, ops.s2, ops.one};

  ops.add0.x1 = ops.broadcast2.y;
  ops.add0.x2 = ops.load1.y;
  ops.add0.attr.sched.axis = ops.axis;
  ops.add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  ops.add0.y.dtype = af::DT_FLOAT;
  *ops.add0.y.axis = ops.axis;
  *ops.add0.y.repeats = {ops.s0, ops.s1, ops.s2};
  *ops.add0.y.strides = {ops.s1 * ops.s2, ops.s2, ops.one};

  ops.store0.x = ops.add0.y;
  ops.store0.attr.sched.axis = ops.axis;
  ops.store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  ops.store0.y.dtype = af::DT_FLOAT;
  *ops.store0.y.axis = ops.axis;
  *ops.store0.y.repeats = {ops.s0, ops.s1, ops.s2};
  *ops.store0.y.strides = {ops.s1 * ops.s2, ops.s2, ops.one};

  ops.output0.ir_attr.SetIndex(0);
  ops.output0.x = ops.store0.y;
  ops.output0.attr.sched.axis = ops.axis;
  ops.output0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  ops.output0.attr.api.type = af::ApiType::kAPITypeBuffer;
  ops.output0.y.dtype = af::DT_FLOAT;
  *ops.output0.y.axis = ops.axis;
  *ops.output0.y.repeats = {ops.s0, ops.s1, ops.s2};
  *ops.output0.y.strides = {ops.s1 * ops.s2, ops.s2, ops.one};
}

// 与 Python HintGraph 逐节点等价的建图：所有 size/strides 与 Python 复现脚本一致。
af::AscGraph BuildSoftmaxExpandDimsAddGraph() {
  af::AscGraph graph("autofuse_2_SoftmaxV2_ExpandDims_Add");
  SoftmaxExpandDimsAddOps ops(graph);
  AddSoftmaxInputLoad(ops, graph);
  AddOperandInputLoad(ops, graph);
  AddSoftmaxSubExpChain(ops);
  AddSoftmaxSumDivChain(ops);
  AddExpandAddEpilogue(ops);
  return graph;
}

void DumpImplGraphLayouts(const af::AscGraph &impl_graph, const std::string &tag) {
  std::cout << "==== ImplGraph[" << tag << "] node layouts ====" << std::endl;
  for (const auto &ge_node : impl_graph.GetAllNodes()) {
    const auto node = std::dynamic_pointer_cast<af::AscNode>(ge_node);
    if (node == nullptr) {
      continue;
    }
    std::cout << "Node [" << node->GetNamePtr() << "] type [" << node->GetTypePtr() << "]" << std::endl;
    const auto input_tensors = node->inputs();
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      const auto &attr = input_tensors[i]->attr;
      std::cout << "  in[" << i << "] repeats=" << ExprVecToStr(attr.repeats)
                << " strides=" << ExprVecToStr(attr.strides) << " vstrides=" << ExprVecToStr(attr.vectorized_strides)
                << std::endl;
    }
    const auto output_tensors = node->outputs();
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      const auto &attr = output_tensors[i]->attr;
      std::cout << "  out[" << i << "] repeats=" << ExprVecToStr(attr.repeats)
                << " strides=" << ExprVecToStr(attr.strides) << " vstrides=" << ExprVecToStr(attr.vectorized_strides)
                << std::endl;
    }
  }
  std::cout << "==== ImplGraph[" << tag << "] dump end ====" << std::endl;
}

std::vector<std::string> SplitByStr(const std::string &text, const std::string &delimiter) {
  std::vector<std::string> parts;
  size_t begin = 0U;
  size_t end = text.find(delimiter, begin);
  while (end != std::string::npos) {
    parts.push_back(text.substr(begin, end - begin));
    begin = end + delimiter.size();
    end = text.find(delimiter, begin);
  }
  parts.push_back(text.substr(begin));
  return parts;
}

std::vector<std::string> GetCallOperands(const std::string &body, size_t call_pos) {
  const size_t args_begin = body.find('(', call_pos);
  const size_t args_end = body.find(';', call_pos);
  if (args_begin == std::string::npos || args_end == std::string::npos || args_begin >= args_end) {
    return {};
  }
  return SplitByStr(body.substr(args_begin + 1U, args_end - args_begin - 1U), ", ");
}

// 收集单个调度实现图中的全部 RemovePad 节点。
void CollectRemovePadNodes(const af::AscGraph &impl_graph, std::vector<af::AscNodePtr> &remove_pad_nodes) {
  for (const auto &ge_node : impl_graph.GetAllNodes()) {
    const auto node = std::dynamic_pointer_cast<af::AscNode>(ge_node);
    if (node == nullptr || std::string(node->GetTypePtr()) != "RemovePad") {
      continue;
    }
    remove_pad_nodes.push_back(node);
  }
}

// 诊断：转储全部调度产物的节点布局（重点关注 Broadcast_2 输入侧 vstrides），并收集 RemovePad 节点。
void DumpAndCollectRemovePadNodes(const ascir::FusedScheduledResult &fused_schedule_result,
                                  std::vector<af::AscNodePtr> &remove_pad_nodes) {
  size_t group_dump_count = 0;
  for (const auto &node_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (const auto &schedule_result : node_results) {
      for (const auto &schedule_group : schedule_result.schedule_groups) {
        for (const auto &impl_graph : schedule_group.impl_graphs) {
          DumpImplGraphLayouts(impl_graph, std::to_string(group_dump_count++));
          CollectRemovePadNodes(impl_graph, remove_pad_nodes);
        }
      }
    }
  }
}

// 校验 RemovePad 输出为紧凑布局（跳过尾部 "0" 后首个非零 stride 必须是 "1"），且唯一消费者是 Broadcast。
void CheckRemovePadNodeExpectations(const af::AscNodePtr &node) {
  const auto outputs = node->outputs();
  ASSERT_FALSE(outputs.empty());
  const auto &vstrides = outputs[0]->attr.vectorized_strides;
  bool has_inner_stride = false;
  for (auto it = vstrides.rbegin(); it != vstrides.rend(); ++it) {
    if (ExprToStr(*it) == "0") {
      continue;
    }
    EXPECT_EQ(ExprToStr(*it), "1");
    has_inner_stride = true;
    break;
  }
  EXPECT_TRUE(has_inner_stride);
  af::NodePtr consumer = nullptr;
  size_t consumer_count = 0UL;
  for (const auto &out_node : node->GetOutDataNodes()) {
    consumer = out_node;
    consumer_count++;
  }
  ASSERT_EQ(consumer_count, 1UL);
  const auto consumer_asc = std::dynamic_pointer_cast<af::AscNode>(consumer);
  ASSERT_NE(consumer_asc, nullptr);
  EXPECT_EQ(std::string(consumer_asc->GetTypePtr()), "Broadcast");
}

// 校验收集到的 RemovePad 节点逐个满足期望，且至少存在一个（对应原 remove_pad_count 检查）。
void CheckRemovePadExpectations(const std::vector<af::AscNodePtr> &remove_pad_nodes) {
  for (const auto &remove_pad_node : remove_pad_nodes) {
    CheckRemovePadNodeExpectations(remove_pad_node);
  }
  EXPECT_GE(remove_pad_nodes.size(), 1UL);
}

// 截取 kernel 文本中 [begin_marker, end_marker) 之间的函数体片段。
void SliceKernelBody(const std::string &kernel_text, const std::string &begin_marker, const std::string &end_marker,
                     std::string &body) {
  const auto begin_pos = kernel_text.find(begin_marker);
  ASSERT_NE(begin_pos, std::string::npos);
  const auto end_pos = kernel_text.find(end_marker, begin_pos);
  ASSERT_NE(end_pos, std::string::npos);
  body = kernel_text.substr(begin_pos, end_pos - begin_pos);
}

// 校验变体函数体内 RemovePadExtend 的输出被送入尾轴扩维 BroadcastExtend。
void CheckRemovePadFeedsBroadcast(const std::string &body) {
  const auto remove_pad_call_pos = body.find("RemovePadExtend");
  ASSERT_NE(remove_pad_call_pos, std::string::npos);
  const auto remove_pad_operands = GetCallOperands(body, remove_pad_call_pos);
  ASSERT_GE(remove_pad_operands.size(), 2UL);

  const auto broadcast_call_pos = body.find("BroadcastExtend", remove_pad_call_pos);
  ASSERT_NE(broadcast_call_pos, std::string::npos);
  const auto broadcast_operands = GetCallOperands(body, broadcast_call_pos);
  ASSERT_GE(broadcast_operands.size(), 2UL);
  EXPECT_EQ(broadcast_operands[1], remove_pad_operands[0]);
}

// 全载非 nddma 变体中，尾轴扩维广播的源应是新插入 RemovePad 的紧凑输出；
// nddma 全载变体（行数段含符号 tile 轴 z0t）同样必须在尾轴扩维广播前插入 RemovePad。
// 注: 函数名中的 sN 为全局 schedule group 编号, NormRecompute 计划(插入于 RCore 之后)的组
// 占用 s2, 本计划全载/nddma 变体随之由 s2 后移至 s3
void CheckKernelRemovePadFeeds(const std::string &kernel_text) {
  std::string full_load_body;
  SliceKernelBody(kernel_text, "autofuse_2__softmax_v2__expand_dims__add_0_b0_y2_s3_g0_c0(",
                  "autofuse_2__softmax_v2__expand_dims__add_0_b0_y0_load_to_nddma_s3_g0_c1(", full_load_body);
  CheckRemovePadFeedsBroadcast(full_load_body);

  std::string nddma_load_body;
  SliceKernelBody(kernel_text, "autofuse_2__softmax_v2__expand_dims__add_0_b0_y0_load_to_nddma_s3_g0_c1(",
                  "extern \"C\" __global__ __aicore__ void autofuse_2__softmax_v2__expand_dims__add(GM_ADDR",
                  nddma_load_body);
  CheckRemovePadFeedsBroadcast(nddma_load_body);
}
}  // namespace

class TestBackendSoftmaxExpandDimsAddE2e : public testing::Test {
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    af::InstallV35SocPerfRuntimeStub();
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
  }
};

TEST_F(TestBackendSoftmaxExpandDimsAddE2e, SoftmaxExpandDimsAddCodegen) {
  bool gen_success = true;
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  af::AscGraph graph = BuildSoftmaxExpandDimsAddGraph();
  std::map<std::string, std::string> shape_info;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];
  std::string tiling_src_file_name = parts[1];
  std::string tiling_data_src_file_name = parts[2];

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

    std::vector<af::AscNodePtr> remove_pad_nodes;
    DumpAndCollectRemovePadNodes(fused_schedule_result, remove_pad_nodes);
    CheckRemovePadExpectations(remove_pad_nodes);

    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    std::cout << result.kernel << std::endl;

    CheckKernelRemovePadFeeds(result.kernel);

    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
