/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "ascir_ops.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "fusion/autofuse_attrs.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "indirect_load_utils.h"
#include "optimize.h"
#include "platform_context.h"

using namespace af::ops;
using namespace af::ascir_op;

namespace {

class PlatformContextReset {
 public:
  ~PlatformContextReset() {
    ge::PlatformContext::GetInstance().Reset();
  }
};

// kFusedAscBackend 管线的宿主图包装：Data 输入 + AscBackend 节点 + Output，子图经
// AutoFuseAttrs::SetAscGraph 挂载（同 backend_e2e_v2 的 VariadicBackendGraph）。
class AscBackendGraph {
 public:
  AscBackendGraph(const char *graph_name, const std::vector<af::DataType> &input_dtypes,
                  const std::vector<af::DataType> &output_dtypes)
      : fused_graph_(graph_name), input_dtypes_(input_dtypes), output_dtypes_(output_dtypes) {
    for (size_t index = 0UL; index < input_dtypes_.size(); ++index) {
      const std::string name = "input" + std::to_string(index);
      af::ascir_op::Data input(name.c_str(), fused_graph_);
      input.ir_attr.SetIndex(static_cast<int32_t>(index));
      input.y.dtype = input_dtypes_[index];
    }
    compute_graph_ = af::AscGraphUtils::GetComputeGraph(fused_graph_);
    if (compute_graph_ == nullptr) {
      return;
    }
    auto backend_desc = std::make_shared<af::OpDesc>("asc_backend", "AscBackend");
    for (const auto dtype : input_dtypes_) {
      af::GeTensorDesc desc;
      desc.SetDataType(dtype);
      backend_desc->AddInputDesc(desc);
    }
    for (const auto dtype : output_dtypes_) {
      af::GeTensorDesc desc;
      desc.SetDataType(dtype);
      backend_desc->AddOutputDesc(desc);
    }
    backend_ = compute_graph_->AddNode(backend_desc);
  }

  af::ComputeGraphPtr Finalize(const std::shared_ptr<af::AscGraph> &sub_graph) {
    if (compute_graph_ == nullptr || backend_ == nullptr) {
      return nullptr;
    }
    const auto attrs = backend_->GetOpDesc()->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
    if (attrs == nullptr) {
      return nullptr;
    }
    attrs->SetAscGraph(sub_graph);
    bool ok = true;
    for (size_t index = 0UL; index < input_dtypes_.size(); ++index) {
      const std::string name = "input" + std::to_string(index);
      const auto input = fused_graph_.FindNode(name.c_str());
      ok = ok && input != nullptr &&
           af::GraphUtils::AddEdge(input->GetOutDataAnchor(0),
                                   backend_->GetInDataAnchor(static_cast<int32_t>(index))) == ge::GRAPH_SUCCESS;
    }
    for (size_t index = 0UL; index < output_dtypes_.size(); ++index) {
      const std::string name = "output" + std::to_string(index);
      af::ascir_op::Output output(name.c_str());
      output.ir_attr.SetIndex(static_cast<int32_t>(index));
      const auto output_node = compute_graph_->AddNode(af::OpDescUtils::GetOpDescFromOperator(output));
      ok = ok && output_node != nullptr &&
           af::GraphUtils::AddEdge(backend_->GetOutDataAnchor(static_cast<int32_t>(index)),
                                   output_node->GetInDataAnchor(0)) == ge::GRAPH_SUCCESS;
    }
    return ok && compute_graph_->TopologicalSorting() == ge::GRAPH_SUCCESS ? compute_graph_ : nullptr;
  }

 private:
  af::AscGraph fused_graph_;
  af::ComputeGraphPtr compute_graph_;
  af::NodePtr backend_;
  std::vector<af::DataType> input_dtypes_;
  std::vector<af::DataType> output_dtypes_;
};

// 双 gather 共享 index 形态：两路独立值表 + 共享 index Load，IL_0/IL_1 → Add → Store。
// 静态 shape + int32 index → uint32 offset → fused 发射档位。
std::shared_ptr<af::AscGraph> BuildSharedIndexDualGatherGraph() {
  auto graph = std::make_shared<af::AscGraph>("fused_group_codegen_ut_graph");
  const af::Expression rows = graph->CreateSizeVar(8L);
  const af::Expression width = graph->CreateSizeVar(16L);
  const af::Expression input_width = graph->CreateSizeVar(64L);
  const auto z0 = graph->CreateAxis("z0", rows);
  const auto z1 = graph->CreateAxis("z1", width);
  const std::vector<af::AxisId> axes = {z0.id, z1.id};
  const std::vector<af::Expression> output = {rows, width};
  const std::vector<af::Expression> output_strides = {width, af::ops::One};
  const std::vector<af::Expression> table_repeats = {rows, input_width};
  const std::vector<af::Expression> table_strides = {input_width, af::ops::One};

  af::ascir_op::Data indices("indices", *graph);
  indices.ir_attr.SetIndex(2);
  indices.y.dtype = ge::DT_INT32;
  indices.attr.sched.axis = axes;
  *indices.y.axis = axes;
  *indices.y.repeats = output;
  *indices.y.strides = output_strides;

  af::ascir_op::Load index_load("index_load");
  graph->AddNode(index_load);
  index_load.x = indices.y;
  index_load.y.dtype = ge::DT_INT32;
  index_load.attr.sched.axis = axes;
  *index_load.y.axis = axes;
  *index_load.y.repeats = output;
  *index_load.y.strides = output_strides;

  af::ascir_op::Data input0("input0", *graph);
  input0.ir_attr.SetIndex(0);
  input0.y.dtype = ge::DT_FLOAT16;
  input0.attr.sched.axis = axes;
  *input0.y.axis = axes;
  *input0.y.repeats = table_repeats;
  *input0.y.strides = table_strides;

  af::ascir_op::Load input0_load("input0_load");
  graph->AddNode(input0_load);
  input0_load.x = input0.y;
  input0_load.y.dtype = ge::DT_FLOAT16;
  input0_load.attr.sched.axis = axes;
  *input0_load.y.axis = axes;
  *input0_load.y.repeats = table_repeats;
  *input0_load.y.strides = table_strides;

  af::ascir_op::IndirectLoad indirect_load0("indirect_load0");
  graph->AddNode(indirect_load0);
  indirect_load0.x1 = input0_load.y;
  indirect_load0.x2 = index_load.y;
  indirect_load0.ir_attr.SetAxis(1);
  indirect_load0.ir_attr.SetNegative_index_support(true);
  indirect_load0.ir_attr.SetNeed_check_bound(true);
  indirect_load0.ir_attr.SetMax(input_width);
  indirect_load0.y.dtype = ge::DT_FLOAT16;
  indirect_load0.attr.sched.axis = axes;
  *indirect_load0.y.axis = axes;
  *indirect_load0.y.repeats = output;
  *indirect_load0.y.strides = output_strides;

  af::ascir_op::Data input1("input1", *graph);
  input1.ir_attr.SetIndex(1);
  input1.y.dtype = ge::DT_FLOAT16;
  input1.attr.sched.axis = axes;
  *input1.y.axis = axes;
  *input1.y.repeats = table_repeats;
  *input1.y.strides = table_strides;

  af::ascir_op::Load input1_load("input1_load");
  graph->AddNode(input1_load);
  input1_load.x = input1.y;
  input1_load.y.dtype = ge::DT_FLOAT16;
  input1_load.attr.sched.axis = axes;
  *input1_load.y.axis = axes;
  *input1_load.y.repeats = table_repeats;
  *input1_load.y.strides = table_strides;

  af::ascir_op::IndirectLoad indirect_load1("indirect_load1");
  graph->AddNode(indirect_load1);
  indirect_load1.x1 = input1_load.y;
  indirect_load1.x2 = index_load.y;
  indirect_load1.ir_attr.SetAxis(1);
  indirect_load1.ir_attr.SetNegative_index_support(true);
  indirect_load1.ir_attr.SetNeed_check_bound(true);
  indirect_load1.ir_attr.SetMax(input_width);
  indirect_load1.y.dtype = ge::DT_FLOAT16;
  indirect_load1.attr.sched.axis = axes;
  *indirect_load1.y.axis = axes;
  *indirect_load1.y.repeats = output;
  *indirect_load1.y.strides = output_strides;

  af::ascir_op::Scalar scale("scale", *graph);
  scale.ir_attr.SetValue("0.5");
  scale.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Broadcast scale_broadcast("scale_broadcast");
  graph->AddNode(scale_broadcast);
  scale_broadcast.x = scale.y;
  scale_broadcast.y.dtype = ge::DT_FLOAT16;
  scale_broadcast.attr.sched.axis = axes;
  *scale_broadcast.y.axis = axes;
  *scale_broadcast.y.repeats = output;
  *scale_broadcast.y.strides = output_strides;

  af::ascir_op::Mul scale_mul("scale_mul");
  graph->AddNode(scale_mul);
  scale_mul.x1 = indirect_load0.y;
  scale_mul.x2 = scale_broadcast.y;
  scale_mul.y.dtype = ge::DT_FLOAT16;
  scale_mul.attr.sched.axis = axes;
  *scale_mul.y.axis = axes;
  *scale_mul.y.repeats = output;
  *scale_mul.y.strides = output_strides;

  af::ascir_op::Add add("add");
  graph->AddNode(add);
  add.x1 = scale_mul.y;
  add.x2 = indirect_load1.y;
  add.y.dtype = ge::DT_FLOAT16;
  add.attr.sched.axis = axes;
  *add.y.axis = axes;
  *add.y.repeats = output;
  *add.y.strides = output_strides;

  af::ascir_op::Store store("store");
  graph->AddNode(store);
  store.x = add.y;
  store.y.dtype = ge::DT_FLOAT16;
  store.attr.sched.axis = axes;
  *store.y.axis = axes;
  *store.y.repeats = output;
  *store.y.strides = output_strides;

  af::ascir_op::Output output_node("output");
  graph->AddNode(output_node);
  output_node.ir_attr.SetIndex(0);
  output_node.x = store.y;
  output_node.y.dtype = ge::DT_FLOAT16;
  return graph;
}

TEST(IndirectLoadCodegenFusedGroupTest, GeneratesFusedSimtGroupKernel) {
  // fused 组 SIMT 的 codegen 级端到端（生产同款管线）：Optimizer::Optimize（候选
  // 生成 → 调度 → 内存计划/张量 id 分配 → IO 收集）→ Codegen::Generate。该链路此前
  // 无任何测试覆盖，fused 路径的成员输入解析与 UB 局部目标采集缺陷均由此暴露。
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 256 * 1024;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);

  AscBackendGraph backend("fused_group_codegen_ut_host", {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32},
                          {ge::DT_FLOAT16});
  const auto graph = backend.Finalize(BuildSharedIndexDualGatherGraph());
  ASSERT_NE(graph, nullptr);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kFusedAscBackend});
  ascir::FusedScheduledResult scheduled_result;
  ASSERT_EQ(optimizer.Optimize(graph, scheduled_result), af::SUCCESS);

  bool has_simt = false;
  for (const auto &candidates : scheduled_result.node_idx_to_scheduled_results) {
    for (const auto &candidate : candidates) {
      for (const auto &group : candidate.schedule_groups) {
        for (const auto &impl_graph : group.impl_graphs) {
          const auto indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(impl_graph);
          if (indirect_load != nullptr &&
              ascir::GetTemplateIdOrDefault(*indirect_load) == ascir::TemplateId::kIndirectLoadSimt) {
            has_simt = true;
          }
        }
      }
    }
  }
  ASSERT_TRUE(has_simt) << "Dual gather with shared index should keep the SIMT group candidate";

  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;
  ASSERT_EQ(codegen.Generate({}, scheduled_result, result), af::SUCCESS);
  EXPECT_NE(result.kernel.find("// IndirectLoad SIMT Fused"), std::string::npos);
}

}  // namespace
