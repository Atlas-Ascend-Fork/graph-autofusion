/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// [单 gather 边界化融合·全管线] 输出链含不可标量发射算子（Pow 无 V2 标量实现，
// tc_af_inductor_rand_fuzz_{0002,0014,0030} 的失败算子族）的单 gather 图：SIMT 候选
// 不再因发射期断言失败而回落 SIMD 全量驻留（大表 UB 超限 → afError），而是降级为
// 边界化形态——SIMT 只做 gather（value 逐 tile 写 UB 局部目标），Pow 链以普通 VF
// 代码在同一循环迭代内消费。复刻 fuzz 用例的 SIMT 让位形态：行少列宽表（lookup
// extent 4 < 32）使 SIMD 判据不满足，静态大表（4×47360×2B ≈ 370KB > 256KB UB）
// 使 SIMD 全量驻留不可行，SIMT 候选胜出。
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

// kFusedAscBackend 管线的宿主图包装（同 test_codegen_indirect_load_fused_group）：
// Data 输入 + AscBackend 节点 + Output，子图经 AutoFuseAttrs::SetAscGraph 挂载。
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

// 单 gather 边界化形态图：value 表 [4, 47360]（行少列宽，lookup extent 4 < 32 → SIMD
// 判据不满足；4×47360×2B ≈ 370KB > 256KB UB → SIMD 全量驻留不可行）、index/output
// [8, 64]、IL(axis=0) → Pow（标量指数 2 经 Broadcast 物化）→ Store。
// 统一设置节点的 schedule 轴与输出视图（axis / repeats / strides），避免逐节点重复四行。
template <typename OpT>
void SetView(OpT &op, const std::vector<af::AxisId> &axes, const std::vector<af::Expression> &repeats,
             const std::vector<af::Expression> &strides) {
  op.attr.sched.axis = axes;
  *op.y.axis = axes;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
}

// 输出链：IL(axis=0) → Pow（标量指数 2 经 Broadcast 物化）→ Store → Output。
void AddBoundaryFusionPowChain(af::AscGraph &graph, const af::Expression &table_rows, const af::AscOpOutput &value,
                               const af::AscOpOutput &index, const std::vector<af::AxisId> &out_axes,
                               const std::vector<af::Expression> &out_repeats,
                               const std::vector<af::Expression> &out_strides) {
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  graph.AddNode(indirect_load);
  indirect_load.x1 = value;
  indirect_load.x2 = index;
  indirect_load.ir_attr.SetAxis(0L);
  indirect_load.ir_attr.SetNegative_index_support(true);
  indirect_load.ir_attr.SetNeed_check_bound(true);
  indirect_load.ir_attr.SetMax(table_rows);
  indirect_load.y.dtype = ge::DT_FLOAT16;
  SetView(indirect_load, out_axes, out_repeats, out_strides);

  af::ascir_op::Scalar pow_exp("pow_exp", graph);
  // 指数取 5：PowEquivSubstitutionPass 仅替换 0/±0.5/±1/±2/3/4 次幂（替换后为可标量
  // 折叠的 Mul 等，判定不触发）；5 次幂存活到调度期，复刻 fuzz 的 pow 输出链形态。
  pow_exp.ir_attr.SetValue("5");
  pow_exp.y.dtype = ge::DT_FLOAT16;
  af::ascir_op::Broadcast pow_exp_brc("pow_exp_brc");
  graph.AddNode(pow_exp_brc);
  pow_exp_brc.x = pow_exp.y;
  pow_exp_brc.y.dtype = ge::DT_FLOAT16;
  SetView(pow_exp_brc, out_axes, out_repeats, out_strides);

  af::ascir_op::Pow pow("pow");
  graph.AddNode(pow);
  pow.x1 = indirect_load.y;
  pow.x2 = pow_exp_brc.y;
  pow.y.dtype = ge::DT_FLOAT16;
  SetView(pow, out_axes, out_repeats, out_strides);

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = pow.y;
  store.y.dtype = ge::DT_FLOAT16;
  SetView(store, out_axes, out_repeats, out_strides);

  af::ascir_op::Output output_node("output");
  graph.AddNode(output_node);
  output_node.ir_attr.SetIndex(0);
  output_node.x = store.y;
  output_node.y.dtype = ge::DT_FLOAT16;
}

std::shared_ptr<af::AscGraph> BuildBoundaryFusionGraph() {
  auto graph = std::make_shared<af::AscGraph>("boundary_fusion_codegen_ut_graph");
  const af::Expression table_rows = graph->CreateSizeVar(4L);
  const af::Expression table_width = graph->CreateSizeVar(47360L);
  const af::Expression out_rows = graph->CreateSizeVar(8L);
  const af::Expression out_width = graph->CreateSizeVar(64L);
  const auto table_axis0 = graph->CreateAxis("table0", table_rows);
  const auto table_axis1 = graph->CreateAxis("table1", table_width);
  const auto out_axis0 = graph->CreateAxis("out0", out_rows);
  const auto out_axis1 = graph->CreateAxis("out1", out_width);
  const std::vector<af::AxisId> table_axes = {table_axis0.id, table_axis1.id};
  const std::vector<af::AxisId> out_axes = {out_axis0.id, out_axis1.id};
  const std::vector<af::Expression> table_repeats = {table_rows, table_width};
  const std::vector<af::Expression> table_strides = {table_width, af::ops::One};
  const std::vector<af::Expression> out_repeats = {out_rows, out_width};
  const std::vector<af::Expression> out_strides = {out_width, af::ops::One};

  af::ascir_op::Data x("x", *graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = ge::DT_FLOAT16;
  SetView(x, table_axes, table_repeats, table_strides);
  af::ascir_op::Load input_load("input_load");
  graph->AddNode(input_load);
  input_load.x = x.y;
  input_load.y.dtype = ge::DT_FLOAT16;
  SetView(input_load, table_axes, table_repeats, table_strides);

  af::ascir_op::Data index("index", *graph);
  index.ir_attr.SetIndex(1);
  index.y.dtype = ge::DT_INT32;
  SetView(index, out_axes, out_repeats, out_strides);
  af::ascir_op::Load index_load("index_load");
  graph->AddNode(index_load);
  index_load.x = index.y;
  index_load.y.dtype = ge::DT_INT32;
  SetView(index_load, out_axes, out_repeats, out_strides);

  AddBoundaryFusionPowChain(*graph, table_rows, input_load.y, index_load.y, out_axes, out_repeats, out_strides);
  return graph;
}

// [fuzz 0002 精确复刻] 依据 fuzz graph code dump 逐节点重建：
// data(int64, idx=0) → load([4,1]/strides[1,0]) → broadcast([4,47360]) ─┐
// data1(f32, idx=1) → load1([4,47360]) ────────────────────────────────┴→ indirectload
// (axis=0, neg, bound, max=4, f32 [4,47360]) → reciprocal → fmod ← broadcast2 ← broadcast1
// ← scalar('1.71') → store → output(f32)。全链 fp32、index data-dependent（GM load 后
// broadcast）、fmod 右操作数双级 Broadcast——三条形态均与简化 UT 图不同。
// 输出链：IL → Reciprocal → Fmod（右操作数 scalar → 双级 Broadcast）→ Store → Output。
// fmod 右操作数：scalar('1.71') → broadcast1([4,1]) → broadcast2([4,47360])，末端接到 fmod.x2。
// 注意不能返回 AscOpOutput（局部 op 析构后即悬空），必须在算子存活的作用域内完成接线。
void AddBoundaryFusionFmodRhs(af::AscGraph &graph, const std::vector<af::AxisId> &axes,
                              const std::vector<af::Expression> &full_repeats,
                              const std::vector<af::Expression> &full_strides,
                              const std::vector<af::Expression> &row_repeats,
                              const std::vector<af::Expression> &row_strides, af::ascir_op::Fmod &fmod) {
  af::ascir_op::Scalar fmod_rhs("fmod_rhs", graph);
  fmod_rhs.ir_attr.SetValue("1.71");
  fmod_rhs.y.dtype = ge::DT_FLOAT;
  af::ascir_op::Broadcast fmod_rhs_brc1("fmod_rhs_brc1");
  graph.AddNode(fmod_rhs_brc1);
  fmod_rhs_brc1.x = fmod_rhs.y;
  fmod_rhs_brc1.y.dtype = ge::DT_FLOAT;
  SetView(fmod_rhs_brc1, axes, row_repeats, row_strides);
  af::ascir_op::Broadcast fmod_rhs_brc2("fmod_rhs_brc2");
  graph.AddNode(fmod_rhs_brc2);
  fmod_rhs_brc2.x = fmod_rhs_brc1.y;
  fmod_rhs_brc2.y.dtype = ge::DT_FLOAT;
  SetView(fmod_rhs_brc2, axes, full_repeats, full_strides);
  fmod.x2 = fmod_rhs_brc2.y;
}

void AddBoundaryFusionFmodOutputChain(af::AscGraph &graph, const af::Expression &out_rows, const af::AscOpOutput &table,
                                      const af::AscOpOutput &index, const std::vector<af::AxisId> &axes,
                                      const std::vector<af::Expression> &full_repeats,
                                      const std::vector<af::Expression> &full_strides,
                                      const std::vector<af::Expression> &row_repeats,
                                      const std::vector<af::Expression> &row_strides) {
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  graph.AddNode(indirect_load);
  indirect_load.x1 = table;
  indirect_load.x2 = index;
  indirect_load.ir_attr.SetAxis(0L);
  indirect_load.ir_attr.SetNegative_index_support(true);
  indirect_load.ir_attr.SetNeed_check_bound(true);
  indirect_load.ir_attr.SetMax(out_rows);
  indirect_load.y.dtype = ge::DT_FLOAT;
  SetView(indirect_load, axes, full_repeats, full_strides);

  af::ascir_op::Reciprocal reciprocal("reciprocal");
  graph.AddNode(reciprocal);
  reciprocal.x = indirect_load.y;
  reciprocal.y.dtype = ge::DT_FLOAT;
  SetView(reciprocal, axes, full_repeats, full_strides);

  af::ascir_op::Fmod fmod("fmod");
  graph.AddNode(fmod);
  fmod.x1 = reciprocal.y;
  AddBoundaryFusionFmodRhs(graph, axes, full_repeats, full_strides, row_repeats, row_strides, fmod);
  fmod.y.dtype = ge::DT_FLOAT;
  SetView(fmod, axes, full_repeats, full_strides);

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = fmod.y;
  store.y.dtype = ge::DT_FLOAT;
  SetView(store, axes, full_repeats, full_strides);

  af::ascir_op::Output output_node("output");
  graph.AddNode(output_node);
  output_node.ir_attr.SetIndex(0);
  output_node.x = store.y;
  output_node.y.dtype = ge::DT_FLOAT;
}

std::shared_ptr<af::AscGraph> BuildBoundaryFusionFmodGraph() {
  auto graph = std::make_shared<af::AscGraph>("boundary_fusion_fmod_ut_graph");
  const af::Expression out_rows = graph->CreateSizeVar(4L);
  const af::Expression out_cols = graph->CreateSizeVar(47360L);
  const auto axis0 = graph->CreateAxis("a0", out_rows);
  const auto axis1 = graph->CreateAxis("a1", out_cols);
  const std::vector<af::AxisId> axes = {axis0.id, axis1.id};
  const std::vector<af::Expression> full_repeats = {out_rows, out_cols};
  const std::vector<af::Expression> full_strides = {out_cols, af::ops::One};
  const std::vector<af::Expression> row_repeats = {out_rows, af::ops::One};
  const std::vector<af::Expression> row_strides = {af::ops::One, af::ops::Zero};

  // index 链：int64 GM load（[4,1]，a1 轴 stride 0）后 broadcast 成 [4,47360]。
  af::ascir_op::Data index_data("index_data", *graph);
  index_data.ir_attr.SetIndex(0);
  index_data.y.dtype = ge::DT_INT64;
  SetView(index_data, axes, row_repeats, row_strides);
  af::ascir_op::Load index_load("index_load");
  graph->AddNode(index_load);
  index_load.x = index_data.y;
  index_load.y.dtype = ge::DT_INT64;
  SetView(index_load, axes, row_repeats, row_strides);
  af::ascir_op::Broadcast index_brc("index_brc");
  graph->AddNode(index_brc);
  index_brc.x = index_load.y;
  index_brc.y.dtype = ge::DT_INT64;
  SetView(index_brc, axes, full_repeats, full_strides);

  // value 表：f32 [4,47360]。
  af::ascir_op::Data table_data("table_data", *graph);
  table_data.ir_attr.SetIndex(1);
  table_data.y.dtype = ge::DT_FLOAT;
  SetView(table_data, axes, full_repeats, full_strides);
  af::ascir_op::Load table_load("table_load");
  graph->AddNode(table_load);
  table_load.x = table_data.y;
  table_load.y.dtype = ge::DT_FLOAT;
  SetView(table_load, axes, full_repeats, full_strides);

  AddBoundaryFusionFmodOutputChain(*graph, out_rows, table_load.y, index_brc.y, axes, full_repeats, full_strides,
                                   row_repeats, row_strides);
  return graph;
}

// 调度结果中是否存在携带边界化标记的 SIMT 候选（逐层展开以避免深嵌套扫描）。
template <typename GroupT>
bool HasBoundaryFusionSimtInGroup(const GroupT &group) {
  for (const auto &impl_graph : group.impl_graphs) {
    const auto indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(impl_graph);
    if (indirect_load != nullptr &&
        ascir::GetTemplateIdOrDefault(*indirect_load) == ascir::TemplateId::kIndirectLoadSimt &&
        ascgen_utils::indirect_load::IsSimtBoundaryFusion(indirect_load)) {
      return true;
    }
  }
  return false;
}

template <typename CandidateT>
bool HasBoundaryFusionSimtInCandidate(const CandidateT &candidate) {
  for (const auto &group : candidate.schedule_groups) {
    if (HasBoundaryFusionSimtInGroup(group)) {
      return true;
    }
  }
  return false;
}

template <typename ResultT>
bool HasBoundaryFusionSimtCandidate(const ResultT &scheduled_result) {
  for (const auto &candidates : scheduled_result.node_idx_to_scheduled_results) {
    for (const auto &candidate : candidates) {
      if (HasBoundaryFusionSimtInCandidate(candidate)) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

// [边界化·全管线] 修复前该图在 SIMT 候选的标量求值器发射处断言失败（Pow 无 V2 标量
// 实现）回落 SIMD 全量驻留（370KB > 256KB UB）后 afError；修复后 SIMT 候选携带
// 边界化标记（gather-only），Pow 链以普通 VF 代码发射，整条 Optimize → Codegen
// 管线 SUCCESS。形态断言：SIMT 调用逐 tile（不含整核 loop_size 实参）、value 写
// UB 局部目标（__ubuf__ 目标）、Pow 走向量发射（power API 头被收集）。
TEST(IndirectLoadCodegenBoundaryFusionTest, GeneratesBoundaryFusedSimtKernelWithVfPowChain) {
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 256 * 1024;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);

  AscBackendGraph backend("boundary_fusion_codegen_ut_host", {ge::DT_FLOAT16, ge::DT_INT32}, {ge::DT_FLOAT16});
  const auto graph = backend.Finalize(BuildBoundaryFusionGraph());
  ASSERT_NE(graph, nullptr);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kFusedAscBackend});
  ascir::FusedScheduledResult scheduled_result;
  ASSERT_EQ(optimizer.Optimize(graph, scheduled_result), af::SUCCESS);

  // SIMT 候选存在且携带边界化标记（gather-only：输出链交普通 VF）。
  ASSERT_TRUE(HasBoundaryFusionSimtCandidate(scheduled_result))
      << "SIMT candidate with boundary fusion marker must survive scheduling.";

  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;
  ASSERT_EQ(codegen.Generate({}, scheduled_result, result), af::SUCCESS);

  // SIMT gather 调用存在（单 gather 非 fused）。
  const auto call_pos = result.kernel.find("AscendC::IndirectLoadSimt<");
  ASSERT_NE(call_pos, std::string::npos) << "boundary fusion must keep the SIMT gather invocation.";
  const auto call_end = result.kernel.find(");", call_pos);
  ASSERT_NE(call_end, std::string::npos);
  const std::string call_text = result.kernel.substr(call_pos, call_end - call_pos);
  // 逐 tile 形态：整核 <Tb>_loop_size 实参不得出现（loop_size 是 local target 缺席时
  // 裸 gather/提升形态的专属实参）。
  EXPECT_EQ(call_text.find("_loop_size"), std::string::npos)
      << "boundary fusion must use per-tile element count, not whole-core loop size.";
  // value 写 UB 局部目标（区别于 GM 直写目标）。
  EXPECT_NE(call_text.find("(__ubuf__ "), std::string::npos)
      << "boundary fusion must write the value tensor to a UB local target.";
  // Pow 链以普通 VF 代码发射：power API 头被收集（SIMT 折叠路径不会收集 VF 算子头）。
  EXPECT_NE(result.kernel.find("power.h"), std::string::npos)
      << "the pow chain must be emitted as ordinary vector code.";
}

// [fuzz 0002 复刻·全管线] IL → Reciprocal → Fmod ← Broadcast(Scalar) → Store。fuzz 0002
// 在该形态上报 "Optimize fail"（Autofuser::schedule 内 Optimize 返回非零，第二候选
// 的图重建在 Fmod 节点处静默失败）。本测试先锁定 Optimize SUCCESS，再验证边界化
// SIMT 存活与 Fmod 链的普通 VF 发射。
TEST(IndirectLoadCodegenBoundaryFusionTest, GeneratesBoundaryFusedSimtKernelWithVfFmodChain) {
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 256 * 1024;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);

  AscBackendGraph backend("boundary_fusion_fmod_ut_host", {ge::DT_INT64, ge::DT_FLOAT}, {ge::DT_FLOAT});
  const auto graph = backend.Finalize(BuildBoundaryFusionFmodGraph());
  ASSERT_NE(graph, nullptr);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kFusedAscBackend});
  ascir::FusedScheduledResult scheduled_result;
  ASSERT_EQ(optimizer.Optimize(graph, scheduled_result), af::SUCCESS)
      << "repro of fuzz 0002: Optimize must succeed for the reciprocal+fmod output chain.";

  ASSERT_TRUE(HasBoundaryFusionSimtCandidate(scheduled_result))
      << "SIMT candidate with boundary fusion marker must survive scheduling.";

  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;
  ASSERT_EQ(codegen.Generate({}, scheduled_result, result), af::SUCCESS);

  EXPECT_NE(result.kernel.find("AscendC::IndirectLoadSimt<"), std::string::npos)
      << "boundary fusion must keep the SIMT gather invocation.";
  // Fmod 链以普通 VF 代码发射（SIMT 折叠路径不发射 VF 算子调用）。
  EXPECT_NE(result.kernel.find("Fmod"), std::string::npos) << "the fmod chain must be emitted as ordinary vector code.";
}
