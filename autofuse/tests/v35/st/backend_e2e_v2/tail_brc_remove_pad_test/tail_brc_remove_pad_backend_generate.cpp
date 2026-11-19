/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* 尾轴广播源的逐元素 32B 槽位布局（[100,8,1] 输入经对齐策略 pad 后最内层有效轴
 * stride==Align(1,32/sizeof(dtype))）+ 尾轴扩维 Broadcast_0(1 -> 50) + Add 的代码生成用例。
 * 场景：Broadcast_0 的源来自 Sub_0（元素算子输出，继承 load0 的槽位布局），调度后 tile 布局由
 * 对齐策略决定，32B 槽位源若直接被 BroadcastExtend 按 src_shape 稠密反推读取会错读槽位间隙，
 * 必须插入 RemovePad 归一化成紧凑布局；本用例核对调度产物中的 RemovePad 节点与生成 kernel 中的
 * RemovePadExtend -> BroadcastExtend 数据流。与 softmax_expanddims_add_test 互补：z1 取 8 元素行
 * （满向量），用更小的链路覆盖同一 RemovePad 插入路径。 */

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

struct TailBrcViews {
  std::vector<AxisId> axis;
  // [100,8,1] 槽位视图（尾轴退化 size==1、stride==0，对齐策略 pad 后形成 32B 槽位布局）
  std::vector<Expression> slot_repeats;  // {s0, s1, 1}
  std::vector<Expression> slot_strides;  // {s1, 1, 0}
  // z1 轴 keep-dims 归约视图
  std::vector<Expression> reduce_repeats;  // {s0, 1, 1}
  std::vector<Expression> reduce_strides;  // {1, 0, 0}
  // z1 维恢复广播视图（repeats 与槽位视图相同）
  std::vector<Expression> restore_strides;  // {s1, 1, 1}
  // [100,8,50] 稠密视图
  std::vector<Expression> dense_repeats;  // {s0, s1, s2}
  std::vector<Expression> dense_strides;  // {s1*s2, s2, 1}
};

TailBrcViews CreateTailBrcViews(af::AscGraph &graph) {
  TailBrcViews views;
  const Expression s0 = graph.CreateSizeVar(100);
  const Expression s1 = graph.CreateSizeVar(8);
  const Expression s2 = graph.CreateSizeVar(50);
  const auto z0 = graph.CreateAxis("z0", s0);
  const auto z1 = graph.CreateAxis("z1", s1);
  const auto z2 = graph.CreateAxis("z2", s2);
  views.axis = {z0.id, z1.id, z2.id};
  const Expression one = af::Symbol(1);
  const Expression zero = af::Symbol(0);
  views.slot_repeats = {s0, s1, one};
  views.slot_strides = {s1, one, zero};
  views.reduce_repeats = {s0, one, one};
  views.reduce_strides = {one, zero, zero};
  views.restore_strides = {s1, one, one};
  views.dense_repeats = {s0, s1, s2};
  views.dense_strides = {s1 * s2, s2, one};
  return views;
}

// 统一设置算子的调度轴、compute_type 与输出张量视图；链路内所有张量 dtype 均为 FP32
template <typename Op>
void SetTensorView(Op &op, af::ComputeType compute_type, const std::vector<AxisId> &axis,
                   const std::vector<Expression> &repeats, const std::vector<Expression> &strides) {
  op.attr.sched.axis = axis;
  op.attr.api.compute_type = compute_type;
  op.y.dtype = af::DT_FLOAT;
  *op.y.axis = axis;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
}

void BuildMaxSubSource(af::AscGraph &graph, const TailBrcViews &views, af::ascir_op::Sub &sub0) {
  // x0 形状 [100,8,1]：尾轴退化（size==1、stride==0），对齐策略 pad 后形成槽位布局
  af::ascir_op::Data data0("maxsub_out0_graph/Data_0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  SetTensorView(data0, af::ComputeType::kComputeInvalid, views.axis, views.slot_repeats, views.slot_strides);

  af::ascir_op::Load load0("maxsub_out0_graph/Load_0");
  load0.x = data0.y;
  SetTensorView(load0, af::ComputeType::kComputeLoad, views.axis, views.slot_repeats, views.slot_strides);

  // z1 轴归约（keep-dims）：[100,8,1] -> [100,1,1]，保持 z1 实轴使后段形成 32B 槽位布局
  af::ascir_op::Max max0("maxsub_out0_graph/Max_0");
  max0.x = load0.y;
  SetTensorView(max0, af::ComputeType::kComputeReduce, views.axis, views.reduce_repeats, views.reduce_strides);

  // z1 维恢复广播：[100,1,1] -> [100,8,1]
  af::ascir_op::Broadcast broadcast0("Broadcast_0_maxsub_out0_graph/Sub_0_2");
  broadcast0.x = max0.y;
  SetTensorView(broadcast0, af::ComputeType::kComputeBroadcast, views.axis, views.slot_repeats, views.restore_strides);

  // 元素算子以 load0 为主输入，Sub 输出继承 load0 的 32B 槽位布局
  sub0.x1 = load0.y;
  sub0.x2 = broadcast0.y;
  SetTensorView(sub0, af::ComputeType::kComputeElewise, views.axis, views.slot_repeats, views.slot_strides);
}

void BuildTailBrcAddOutput(af::AscGraph &graph, const TailBrcViews &views, const af::AscOpOutput &sub_out) {
  // x1 / y 形状 [100,8,50]
  af::ascir_op::Data data1("add_out0_graph/Data_1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  SetTensorView(data1, af::ComputeType::kComputeInvalid, views.axis, views.dense_repeats, views.dense_strides);

  af::ascir_op::Load load1("add_out0_graph/Load_1");
  load1.x = data1.y;
  SetTensorView(load1, af::ComputeType::kComputeLoad, views.axis, views.dense_repeats, views.dense_strides);

  // 尾轴扩维广播：[100,8,1] -> [100,8,50]，源为槽位布局，需要 RemovePad 归一化
  af::ascir_op::Broadcast broadcast1("Broadcast_0_add_out0_graph/Add_0_1");
  broadcast1.x = sub_out;
  SetTensorView(broadcast1, af::ComputeType::kComputeBroadcast, views.axis, views.dense_repeats, views.dense_strides);

  af::ascir_op::Add add0("add_out0_graph/Add_0");
  add0.x1 = broadcast1.y;
  add0.x2 = load1.y;
  SetTensorView(add0, af::ComputeType::kComputeElewise, views.axis, views.dense_repeats, views.dense_strides);

  af::ascir_op::Store store0("add_out0_graph/Store_0");
  store0.x = add0.y;
  SetTensorView(store0, af::ComputeType::kComputeStore, views.axis, views.dense_repeats, views.dense_strides);

  af::ascir_op::Output output0("add_out0_graph/Output_0");
  output0.ir_attr.SetIndex(0);
  output0.x = store0.y;
  output0.attr.api.type = af::ApiType::kAPITypeBuffer;
  SetTensorView(output0, af::ComputeType::kComputeInvalid, views.axis, views.dense_repeats, views.dense_strides);
}

// 建图说明：z1 为 8 元素行（32B），z1 轴归约使 z0/z1 无法合并展平，对齐策略 pad 后
// 后段形成逐元素 32B 槽位布局（vstrides=[8,0]）；z1 取满向量宽度可让 RA 归约命中更简的
// 代码生成分支。链路中 z1 恢复广播与尾轴扩维广播之间必须有元素算子隔断（Sub 以 load0 为主输入，
// 其输出继承槽位布局），否则两个 Broadcast 会被调度器合并。
// 场景链路：Load[100,8,1] -> Max(z1) -> Broadcast(z1 恢复) -> Sub(以 Load 为主输入) -> 尾轴 Broadcast(1->50)
// -> Add -> Store，语义 y[i][j][k] = (x0[i][j] - max(x0[i][:])) + x1[i][j][k]。
af::AscGraph BuildTailBrcRemovePadGraph() {
  af::AscGraph graph("tail_brc_remove_pad_test");
  const TailBrcViews views = CreateTailBrcViews(graph);
  // Sub_0 的输出由后段 helper 消费（broadcast1.x = sub0.y），跨函数链接要求生产者存活，故在顶层构造
  af::ascir_op::Sub sub0("maxsub_out0_graph/Sub_0");
  BuildMaxSubSource(graph, views, sub0);
  BuildTailBrcAddOutput(graph, views, sub0.y);
  return graph;
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

std::string MakeTilingStub() {
  return R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
}

void CollectGraphRemovePadNodes(const af::AscGraph &impl_graph, std::vector<af::AscNodePtr> &remove_pad_nodes) {
  for (const auto &ge_node : impl_graph.GetAllNodes()) {
    const auto node = std::dynamic_pointer_cast<af::AscNode>(ge_node);
    if (node == nullptr || std::string(node->GetTypePtr()) != "RemovePad") {
      continue;
    }
    remove_pad_nodes.push_back(node);
  }
}

void CollectScheduledRemovePadNodes(const ascir::ScheduledResult &schedule_result,
                                    std::vector<af::AscNodePtr> &remove_pad_nodes) {
  for (const auto &schedule_group : schedule_result.schedule_groups) {
    for (const auto &impl_graph : schedule_group.impl_graphs) {
      CollectGraphRemovePadNodes(impl_graph, remove_pad_nodes);
    }
  }
}

std::vector<af::AscNodePtr> CollectRemovePadNodes(const ascir::FusedScheduledResult &fused_schedule_result) {
  std::vector<af::AscNodePtr> remove_pad_nodes;
  for (const auto &node_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (const auto &schedule_result : node_results) {
      CollectScheduledRemovePadNodes(schedule_result, remove_pad_nodes);
    }
  }
  return remove_pad_nodes;
}

void AssertRemovePadNode(const af::AscNodePtr &node) {
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

// 生成 kernel 文本中，尾轴扩维广播必须读取新插入 RemovePad 的紧凑输出：
// RemovePadExtend(dst, src, ...) 之后 BroadcastExtend(..., dst, ...) 依次出现。
// 匹配调用形态（"local_" 起参）以避开 kernel 末尾的 RemovePadExtend 模板定义。
void AssertRemovePadKernelText(const codegen::CodegenResult &result) {
  const auto &kernel_text = result.kernel;
  const auto remove_pad_call_pos = kernel_text.find("RemovePadExtend(local_");
  ASSERT_NE(remove_pad_call_pos, std::string::npos);
  const auto remove_pad_operands = GetCallOperands(kernel_text, remove_pad_call_pos);
  ASSERT_GE(remove_pad_operands.size(), 2UL);
  const auto broadcast_call_pos = kernel_text.find("BroadcastExtend<", remove_pad_call_pos);
  ASSERT_NE(broadcast_call_pos, std::string::npos);
  const auto broadcast_operands = GetCallOperands(kernel_text, broadcast_call_pos);
  ASSERT_GE(broadcast_operands.size(), 2UL);
  EXPECT_EQ(broadcast_operands[1], remove_pad_operands[0]);
}
}  // namespace

class TestBackendTailBrcRemovePadE2e : public testing::Test {
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

TEST_F(TestBackendTailBrcRemovePadE2e, TailBrcRemovePadCodegen) {
  bool gen_success = true;
  const std::string tilig_stub = MakeTilingStub();

  af::AscGraph graph = BuildTailBrcRemovePadGraph();
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

    // 核对调度产物：尾轴扩维广播的 32B 槽位源前必须插入 RemovePad，且输出为紧凑布局
    const std::vector<af::AscNodePtr> remove_pad_nodes = CollectRemovePadNodes(fused_schedule_result);
    const size_t remove_pad_count = remove_pad_nodes.size();
    for (const auto &remove_pad_node : remove_pad_nodes) {
      AssertRemovePadNode(remove_pad_node);
    }
    EXPECT_GE(remove_pad_count, 1UL);

    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    std::cout << result.kernel << std::endl;
    AssertRemovePadKernelText(result);

    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
