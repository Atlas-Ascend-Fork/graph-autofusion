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

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "runtime_stub.h"
#include "graph_utils_ex.h"

#define private public
#include "buffer_allocate/buf_que_allocator.h"
#include "asc_graph_builder.h"
#include "ascgraph_info_complete.h"
#include "common/l2_cache_hint_manager.h"
#undef private
#include "fusion/autofuse_attrs.h"
#include "schedule_utils.h"
#include "platform_context.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace optimize;
using af::testing::AscGraphBuilder;

namespace optimize {
class SetL2CtrlTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v1 = std::make_shared<RuntimeStub>();
    RuntimeStub::SetInstance(stub_v1);
  }
  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
  }
};
}  // namespace optimize

namespace {
af::AscGraph MakeStaticLoadStoreGraph(const std::string &name, int64_t size, int64_t data_index = 0) {
  auto graph = AscGraphBuilder(name)
                   .Loops({size})
                   .Data("data" + std::to_string(data_index), data_index, af::DT_UINT8)
                   .Load("load", "data" + std::to_string(data_index))
                   .Store("store", "load")
                   .Output("output" + std::to_string(data_index), "store", data_index, af::DT_UINT8)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);
  return graph;
}

ascir::FusedScheduledResult MakeFusedScheduledResultWithGraphs(std::vector<af::AscGraph> &&impl_graphs) {
  ascir::FusedScheduledResult fused_result{};
  fused_result.node_idx_to_scheduled_results.resize(1UL);
  auto &scheduled_result = fused_result.node_idx_to_scheduled_results[0].emplace_back();
  auto &group = scheduled_result.schedule_groups.emplace_back();
  group.impl_graphs = std::move(impl_graphs);
  return fused_result;
}

struct L2CtrlTestResult {
  af::AscGraph hint_graph;
  ascir::FusedScheduledResult fused_result;
};
L2CtrlTestResult BuildResult(std::vector<af::AscGraph> &&impl_graphs) {
  af::AscGraph hint_graph = impl_graphs.front();
  auto fused_result = MakeFusedScheduledResultWithGraphs(std::move(impl_graphs));
  BufQueAllocator allocator;
  EXPECT_EQ(allocator.PrepareImplGraphMemoryPlan(fused_result), af::SUCCESS);
  EXPECT_EQ(allocator.CollectFusedIoNodes(fused_result), af::SUCCESS);
  return {std::move(hint_graph), std::move(fused_result)};
}

af::ComputeGraphPtr ToComputeGraph(const af::AscGraph &hint_graph) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(hint_graph);
  EXPECT_NE(compute_graph, nullptr);
  return compute_graph;
}

bool GetDataSkipHintFromGraph(const af::AscGraph &impl_graph, const std::string &data_name) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node != nullptr && node->GetName() == data_name && af::ops::IsOps<af::ascir_op::Data>(node)) {
      bool value = false;
      af::AttrUtils::GetBool(node->GetOpDesc(), "_skip_l2_cache_hint", value);
      return value;
    }
  }
  return false;
}

bool GetDataSkipHintFromScheduledResult(const ascir::ScheduledResult &scheduled_result, const std::string &data_name) {
  for (const auto &schedule_group : scheduled_result.schedule_groups) {
    for (const auto &impl_graph : schedule_group.impl_graphs) {
      if (GetDataSkipHintFromGraph(impl_graph, data_name)) {
        return true;
      }
    }
  }
  return false;
}

bool GetDataSkipHint(ascir::FusedScheduledResult &fsr, const std::string &data_name) {
  for (auto &scheduled_results : fsr.node_idx_to_scheduled_results) {
    for (auto &scheduled_result : scheduled_results) {
      if (GetDataSkipHintFromScheduledResult(scheduled_result, data_name)) {
        return true;
      }
    }
  }
  return false;
}

af::AscGraph MakeMultiSuccessorGraph(const std::string &name, int64_t size) {
  auto graph = AscGraphBuilder(name)
                   .Loops({size})
                   .Data("data0", 0, af::DT_UINT8)
                   .Load("load0", "data0")
                   .Store("store0", "load0")
                   .Output("output0", "store0", 0, af::DT_UINT8)
                   .Load("load1", "data0")
                   .Store("store1", "load1")
                   .Output("output1", "store1", 1, af::DT_UINT8)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);
  return graph;
}

af::AscGraph MakeMultiDataSameIndexGraph(const std::string &name, int64_t size) {
  auto graph = AscGraphBuilder(name)
                   .Loops({size})
                   .Data("data0", 0, af::DT_UINT8)
                   .Load("load0", "data0")
                   .Store("store0", "load0")
                   .Output("output0", "store0", 0, af::DT_UINT8)
                   .Data("data1", 0, af::DT_UINT8)
                   .Load("load1", "data1")
                   .Store("store1", "load1")
                   .Output("output1", "store1", 1, af::DT_UINT8)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);
  return graph;
}
void SetTestPlatformL2Size(int64_t l2_size) {
  ge::PlatformInfo info;
  info.soc_ver = "2201";
  info.aiv_num = 8;
  info.ub_size = 256 * 1024;
  info.l2_size = l2_size;
  ge::PlatformContext::GetInstance().SetPlatformInfo(info);
}

// 单Broadcast插视图+expand图：Data为[2,1,rows]物理输入，Load视图[2,1,rows]广播成[2,2,rows]（fp32）
af::AscGraph MakeSingleBroadcastGraph(const std::string &name, const std::vector<int64_t> &brc_shape, int64_t rows) {
  const auto to_exprs = [](const std::vector<int64_t> &dims) {
    std::vector<Expression> exprs;
    for (const auto dim : dims) {
      exprs.emplace_back(af::Symbol(dim));
    }
    return exprs;
  };
  auto graph = AscGraphBuilder(name)
                   .Loops({2, 2, rows})
                   .Data("data0", 0, to_exprs({2, 1, rows}), to_exprs({rows, rows, 1}), af::DT_FLOAT)
                   .Load("load0", "data0", to_exprs({2, 1, rows}), to_exprs({rows, rows, 1}))
                   .Broadcast("brc0", "load0", to_exprs(brc_shape))
                   .Store("store0", "brc0")
                   .Output("output0", "store0", 0, af::DT_FLOAT)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);
  return graph;
}
}  // namespace

TEST_F(SetL2CtrlTest, CalcTensorSizesBasic) {
  constexpr int64_t kInputSize = 16;
  auto [hint_graph, fused_result] = BuildResult({MakeStaticLoadStoreGraph("g0", kInputSize, 0)});

  ascir::GmTensorSizes sizes;
  EXPECT_EQ(optimize::L2CacheHintManager::CalcTensorSizes(*ToComputeGraph(hint_graph), fused_result, sizes),
            af::SUCCESS);
  ASSERT_EQ(sizes.input_sizes.size(), 1UL);
  ASSERT_EQ(sizes.output_sizes.size(), 1UL);
  EXPECT_TRUE(sizes.total_size.IsValid());
  EXPECT_TRUE(sizes.input_sizes[0].IsValid());
  EXPECT_TRUE(sizes.output_sizes[0].IsValid());
  EXPECT_EQ(sizes.min_total_size, 2LL * kInputSize);
  EXPECT_TRUE(sizes.total_size.IsConstExpr());
  EXPECT_EQ(sizes.input_sizes[0].IsConstExpr(), true);
  EXPECT_EQ(sizes.output_sizes[0].IsConstExpr(), true);
}

TEST_F(SetL2CtrlTest, CalcTensorSizesWithSymbolicDim) {
  auto graph = AscGraphBuilder("sym_g")
                   .Loops({af::testing::Sym("sym_n")})
                   .Data("sym_data", 0, af::DT_UINT8)
                   .Load("sym_load", "sym_data")
                   .Store("sym_store", "sym_load")
                   .Output("sym_output", "sym_store", 0, af::DT_UINT8)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);

  af::AscGraph hint_graph = graph;
  auto fused_result = MakeFusedScheduledResultWithGraphs({graph});
  BufQueAllocator allocator;
  EXPECT_EQ(allocator.PrepareImplGraphMemoryPlan(fused_result), af::SUCCESS);
  EXPECT_EQ(allocator.CollectFusedIoNodes(fused_result), af::SUCCESS);

  ascir::GmTensorSizes sizes;
  EXPECT_EQ(optimize::L2CacheHintManager::CalcTensorSizes(*ToComputeGraph(hint_graph), fused_result, sizes),
            af::SUCCESS);
  ASSERT_EQ(sizes.input_sizes.size(), 1UL);
  ASSERT_EQ(sizes.output_sizes.size(), 1UL);
  EXPECT_FALSE(sizes.input_sizes[0].IsConstExpr());
  EXPECT_FALSE(sizes.output_sizes[0].IsConstExpr());
  EXPECT_FALSE(sizes.total_size.IsConstExpr());
  EXPECT_EQ(sizes.min_total_size, 2);
}

TEST_F(SetL2CtrlTest, MarkSkipHintMultiSuccessor) {
  auto fused_result = MakeFusedScheduledResultWithGraphs({MakeMultiSuccessorGraph("g0", 16)});
  EXPECT_EQ(optimize::L2CacheHintManager::MarkInputsNeedSkipL2CacheHint(fused_result), af::SUCCESS);
  EXPECT_TRUE(GetDataSkipHint(fused_result, "data0"));
}

TEST_F(SetL2CtrlTest, MarkSkipHintNormalNotSet) {
  auto fused_result = MakeFusedScheduledResultWithGraphs({MakeStaticLoadStoreGraph("g0", 3 * 1024 * 1024, 0)});
  EXPECT_EQ(optimize::L2CacheHintManager::MarkInputsNeedSkipL2CacheHint(fused_result), af::SUCCESS);
  EXPECT_FALSE(GetDataSkipHint(fused_result, "data0"));
}

TEST_F(SetL2CtrlTest, MarkSkipHintMultiDataSameIndex) {
  auto fused_result = MakeFusedScheduledResultWithGraphs({MakeMultiDataSameIndexGraph("g0", 16)});
  EXPECT_EQ(optimize::L2CacheHintManager::MarkInputsNeedSkipL2CacheHint(fused_result), af::SUCCESS);
  EXPECT_TRUE(GetDataSkipHint(fused_result, "data0"));
  EXPECT_TRUE(GetDataSkipHint(fused_result, "data1"));
}

// 单broadcast inline场景命中：输入[2,1,8388608]fp32=64MB恰为L2(128MB)的一半，
// 输出[2,2,8388608]fp32=128MB等于L2，单广播轴1->2——打skip保留L2
TEST_F(SetL2CtrlTest, MarkSingleBroadcastInlineSkipHit) {
  constexpr int64_t kRows = 8388608L;
  auto [hint_graph, fused_result] = BuildResult({MakeSingleBroadcastGraph("brc_g", {2, 2, kRows}, kRows)});
  EXPECT_EQ(optimize::L2CacheHintManager::ParseGraph(*ToComputeGraph(hint_graph), fused_result), af::SUCCESS);
  EXPECT_TRUE(GetDataSkipHint(fused_result, "data0"));
}

// inductor真实形态：外层ComputeGraph为Data→AscBackend(挂融合子图)→Output包装，
// Broadcast位于子图内——穿透包装后同样命中豁免
TEST_F(SetL2CtrlTest, MarkSingleBroadcastInlineSkipHitWithAscBackendWrapper) {
  constexpr int64_t kRows = 8388608L;
  auto [hint_graph, fused_result] = BuildResult({MakeSingleBroadcastGraph("inner_brc_g", {2, 2, kRows}, kRows)});

  auto wrapper = std::make_shared<af::ComputeGraph>("wrapper_brc_g");
  af::ascir_op::Data data0("data0");
  auto data0_ir_attr = data0.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  data0_ir_attr->SetIndex(0);
  auto data0_node = wrapper->AddNode(af::OpDescUtils::GetOpDescFromOperator(data0));

  af::OpDescBuilder ascbc_builder("ascbc0", "AscBackend");
  ascbc_builder.AddDynamicInput("x", 1);
  ascbc_builder.AddDynamicOutput("y", 1);
  auto ascbc_node = wrapper->AddNode(ascbc_builder.Build());
  ascbc_node->SetOwnerComputeGraph(wrapper);
  auto fuse_attrs = ascbc_node->GetOpDesc()->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
  fuse_attrs->SetAscGraph(std::make_shared<af::AscGraph>(hint_graph));

  af::ascir_op::Output output0("output0");
  auto output0_ir_attr = output0.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  output0_ir_attr->SetIndex(0);
  auto output0_node = wrapper->AddNode(af::OpDescUtils::GetOpDescFromOperator(output0));

  af::GraphUtils::AddEdge(data0_node->GetOutDataAnchor(0), ascbc_node->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc_node->GetOutDataAnchor(0), output0_node->GetInDataAnchor(0));
  wrapper->TopologicalSorting();

  EXPECT_EQ(optimize::L2CacheHintManager::ParseGraph(*wrapper, fused_result), af::SUCCESS);
  EXPECT_TRUE(GetDataSkipHint(fused_result, "data0"));
}

// 广播轴不是2（1->3）：不豁免，维持原逻辑
TEST_F(SetL2CtrlTest, MarkSingleBroadcastInlineSkipAxisNotTwo) {
  constexpr int64_t kRows = 8388608L;
  auto [hint_graph, fused_result] = BuildResult({MakeSingleBroadcastGraph("brc_g3", {2, 3, kRows}, kRows)});
  EXPECT_EQ(optimize::L2CacheHintManager::ParseGraph(*ToComputeGraph(hint_graph), fused_result), af::SUCCESS);
  EXPECT_FALSE(GetDataSkipHint(fused_result, "data0"));
}

// 输出大小不等于L2：不豁免，维持原逻辑（small rows使输出<L2，输入亦非L2一半）
TEST_F(SetL2CtrlTest, MarkSingleBroadcastInlineSkipOutputSizeMismatch) {
  constexpr int64_t kRows = 4194304L;
  auto [hint_graph, fused_result] = BuildResult({MakeSingleBroadcastGraph("brc_g4", {2, 2, kRows}, kRows)});
  SetTestPlatformL2Size(1024);
  EXPECT_EQ(optimize::L2CacheHintManager::ParseGraph(*ToComputeGraph(hint_graph), fused_result), af::SUCCESS);
  EXPECT_FALSE(GetDataSkipHint(fused_result, "data0"));
}

// 图中无Broadcast节点：不豁免，维持原逻辑（输入>2MB不触原有小张量规则，隔离验证新增分支）
TEST_F(SetL2CtrlTest, MarkSingleBroadcastInlineSkipNoBroadcast) {
  auto [hint_graph, fused_result] = BuildResult({MakeStaticLoadStoreGraph("g0", 3 * 1024 * 1024, 0)});
  SetTestPlatformL2Size(1024 * 1024);
  EXPECT_EQ(optimize::L2CacheHintManager::ParseGraph(*ToComputeGraph(hint_graph), fused_result), af::SUCCESS);
  EXPECT_FALSE(GetDataSkipHint(fused_result, "data0"));
}

// 插轴场景（unsqueeze+expand，2D输入→3D输出）：命中豁免，覆盖IsInsertedAxisTimesTwo分支
TEST_F(SetL2CtrlTest, MarkSingleBroadcastInlineSkipHitInsertedAxis) {
  constexpr int64_t kRows = 8388608L;
  // 图：Data[2,rows] → Load[2,rows] → Broadcast→[2,2,rows]（插轴扩2）
  const auto to_exprs = [](const std::vector<int64_t> &dims) {
    std::vector<Expression> exprs;
    for (const auto dim : dims) {
      exprs.emplace_back(af::Symbol(dim));
    }
    return exprs;
  };
  auto graph = AscGraphBuilder("brc_insert_g")
                   .Loops({2, 2, kRows})
                   .Data("data0", 0, to_exprs({2, kRows}), to_exprs({kRows, 1}), af::DT_FLOAT)
                   .Load("load0", "data0", to_exprs({2, kRows}), to_exprs({kRows, 1}))
                   .Broadcast("brc0", "load0", to_exprs({2, 2, kRows}))
                   .Store("store0", "brc0")
                   .Output("output0", "store0", 0, af::DT_FLOAT)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);
  auto [hint_graph, fused_result] = BuildResult({graph});
  EXPECT_EQ(optimize::L2CacheHintManager::ParseGraph(*ToComputeGraph(hint_graph), fused_result), af::SUCCESS);
  EXPECT_TRUE(GetDataSkipHint(fused_result, "data0"));
}

// 多Broadcast节点：不豁免，覆盖FindSingleBroadcastNode的多Broadcast拒绝分支
TEST_F(SetL2CtrlTest, MarkSingleBroadcastInlineSkipMultipleBroadcast) {
  constexpr int64_t kRows = 1024L;
  const auto to_exprs = [](const std::vector<int64_t> &dims) {
    std::vector<Expression> exprs;
    for (const auto dim : dims) {
      exprs.emplace_back(af::Symbol(dim));
    }
    return exprs;
  };
  auto graph = AscGraphBuilder("brc_multi_g")
                   .Loops({2, 2, kRows})
                   .Data("data0", 0, af::DT_FLOAT)
                   .Load("load0", "data0")
                   .Broadcast("brc0", "load0", to_exprs({2, 1, kRows}))
                   .Broadcast("brc1", "brc0", to_exprs({2, 2, kRows}))
                   .Store("store0", "brc1")
                   .Output("output0", "store0", 0, af::DT_FLOAT)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);
  auto [hint_graph, fused_result] = BuildResult({graph});
  EXPECT_EQ(optimize::L2CacheHintManager::ParseGraph(*ToComputeGraph(hint_graph), fused_result), af::SUCCESS);
  EXPECT_FALSE(GetDataSkipHint(fused_result, "data0"));
}

// 广播输入溯源失败（Load的上游不是Data而是另一个计算节点）：不豁免，覆盖ResolveBroadcastInputSource溯源失败分支
TEST_F(SetL2CtrlTest, MarkSingleBroadcastInlineSkipNoDataTrace) {
  constexpr int64_t kRows = 1024L;
  const auto to_exprs = [](const std::vector<int64_t> &dims) {
    std::vector<Expression> exprs;
    for (const auto dim : dims) {
      exprs.emplace_back(af::Symbol(dim));
    }
    return exprs;
  };
  // 图：Data → Load → Relu(非Data) → Broadcast → Store → Output
  // 溯源在Relu处有两个前驱或到达非Data节点 → 失败
  auto graph = AscGraphBuilder("brc_nodata_g")
                   .Loops({2, 1, kRows})
                   .Data("data0", 0, af::DT_FLOAT)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0", to_exprs({2, 2, kRows}))
                   .Store("store0", "brc0")
                   .Output("output0", "store0", 0, af::DT_FLOAT)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);
  auto [hint_graph, fused_result] = BuildResult({graph});
  EXPECT_EQ(optimize::L2CacheHintManager::ParseGraph(*ToComputeGraph(hint_graph), fused_result), af::SUCCESS);
  EXPECT_FALSE(GetDataSkipHint(fused_result, "data0"));
}

// 图中有非白名单节点（Relu在Broadcast同层）：不豁免，覆盖CollectAnalysisNodes白名单拒绝分支
TEST_F(SetL2CtrlTest, MarkSingleBroadcastInlineSkipNonPureInnerNode) {
  constexpr int64_t kRows = 1024L;
  const auto to_exprs = [](const std::vector<int64_t> &dims) {
    std::vector<Expression> exprs;
    for (const auto dim : dims) {
      exprs.emplace_back(af::Symbol(dim));
    }
    return exprs;
  };
  // 内层子图含Relu（非Data/Load/Store/Output/Broadcast）→ 白名单拒绝
  auto graph = AscGraphBuilder("brc_nonpure_g")
                   .Loops({2, 2, kRows})
                   .Data("data0", 0, to_exprs({2, 1, kRows}), to_exprs({kRows, kRows, 1}), af::DT_FLOAT)
                   .Load("load0", "data0", to_exprs({2, 1, kRows}), to_exprs({kRows, kRows, 1}))
                   .Broadcast("brc0", "load0", to_exprs({2, 2, kRows}))
                   .Abs("abs0", "brc0")
                   .Store("store0", "abs0")
                   .Output("output0", "store0", 0, af::DT_FLOAT)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);
  auto [hint_graph, fused_result] = BuildResult({graph});
  EXPECT_EQ(optimize::L2CacheHintManager::ParseGraph(*ToComputeGraph(hint_graph), fused_result), af::SUCCESS);
  EXPECT_FALSE(GetDataSkipHint(fused_result, "data0"));
}
