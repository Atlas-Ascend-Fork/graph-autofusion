/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <string>
#include <iostream>
#include "gtest/gtest.h"
#include "base/att_const_values.h"
#include "gen_model_info/api_perf_register/utils/vf_perf_utils.h"
#include "gen_model_info.h"
#include "test_fa_ascir_graph.h"
#include "parser/ascend_graph_parser.h"
#define private public
#include "expr_gen/pipe_perf_expr.h"
#undef private
#include "ascir_ops.h"
#include "ascir_node_param/ascir_param_builder.h"
#include "expr_gen/arg_list_manager.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph_construct_utils.h"

namespace af {
namespace ascir {
namespace cg {
Status BuildEqAscendGraphND(af::AscGraph &graph) {
  auto s0 = af::Symbol("S0");
  auto s2 = af::Symbol(2);
  auto s3 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0, z2, z3}, 0, FORMAT_ND);
  LOOP(*z0TB) {
    LOOP(*z0T) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto eq = Eq("eq", load1, load1);
      auto store1 = Store("store1", eq);
      GE_ASSERT_SUCCESS(
          att::GraphConstructUtils::UpdateOutputTensorAxes({*z0TB, *z0Tb, *z0T, *z0t, z2, z3}, {load1, eq, store1}, 2));
      auto output1 = Output("output1", store1);
    }
  }
  att::GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return af::SUCCESS;
}

// 构建VectorFunc子图
static Status BuildVectorFuncSubgraph(af::AscGraph &subgraph) {
  auto ND = af::Symbol("ND");
  auto nd = subgraph.CreateAxis("nd", ND);
  auto [ndB, ndb] = subgraph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = subgraph.TileSplit(ndb->id);
  auto data1 = subgraph.CreateContiguousData("input1", DT_FLOAT, {*ndbt}, 0);
  auto load1 = Load("load1", data1);
  auto abs1 = Abs("abs1", load1);
  auto sub1 = Sub("sub1", abs1, abs1);
  auto store1 = Store("store1", sub1);
  auto output1 = Output("output1", store1);
  return af::SUCCESS;
}

// 公共函数：创建S0轴
static af::Axis CreateS0Axis(af::AscGraph &graph) {
  auto S0 = af::Symbol("S0");
  return graph.CreateAxis("z0", S0);
}

// PipePerfExpr测试：添加VectorFunc节点到主图
static Status AddVectorFuncToMainGraph(af::AscGraph &graph) {
  auto z0 = CreateS0Axis(graph);
  auto [z0B, z0b] = graph.BlockSplit(z0.id);
  auto [z0bT, z0bt] = graph.TileSplit(z0b->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0}, 0);
  LOOP(*z0B) {
    LOOP(*z0bT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto vector_func1 = ascir_op::VectorFunc("vector_func");
      vector_func1.SetAttr("sub_graph_name", "vector_func");
      vector_func1.InstanceOutputy(1);
      vector_func1.x = {load1};
      *vector_func1.y[0].axis = {z0bT->id, z0bt->id};
      *(vector_func1.y[0].repeats) = {z0bT->size, z0bt->size};
      *(vector_func1.y[0].strides) = {z0bt->size, af::Symbol(1)};
      *vector_func1.y[0].vectorized_axis = {z0bt->id};
      auto store1 = Store("store1", vector_func1.y[0]);
      GE_ASSERT_SUCCESS(att::GraphConstructUtils::UpdateOutputTensorAxes({*z0B, *z0bT, *z0bt}, {load1, store1}, 1));
      auto output1 = Output("output1", store1);
    }
  }
  return af::SUCCESS;
}

// 为VectorFunc测试构建图
static Status BuildVectorFuncTestGraph(af::AscGraph &graph) {
  GE_ASSERT_SUCCESS(AddVectorFuncToMainGraph(graph));

  // 添加VectorFunc子图
  auto S0 = af::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0B, z0b] = graph.BlockSplit(z0.id);
  auto [z0bT, z0bt] = graph.TileSplit(z0b->id);

  constexpr char_t vector_func_node_name[] = "vector_func";
  AscGraph subgraph(vector_func_node_name);
  GE_ASSERT_SUCCESS(BuildVectorFuncSubgraph(subgraph));
  graph.AddSubGraph(subgraph);
  auto node = graph.FindNode(vector_func_node_name);
  GE_ASSERT_NOTNULL(node);
  node->attr.sched.axis = {z0bT->id};
  node->attr.sched.loop_axis = z0bT->id;
  af::AttrUtils::SetStr(node->GetOpDescBarePtr(), "sub_graph_name", vector_func_node_name);

  att::GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return af::SUCCESS;
}
}  // namespace cg
}  // namespace ascir
}  // namespace af
namespace att {
static TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
class TestPipePerfExpr : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestPipePerfExpr, case0) {
  af::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);

  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph), af::SUCCESS);
  EXPECT_EQ(ArgListManager::GetInstance().LoadArgList(tuning_space), af::SUCCESS);
  PipePerfExpr pipe_perf(tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> exe_times;
  Expr head_cost;
  EXPECT_EQ(pipe_perf.GetPerfExpr(pipe_costs, exe_times, head_cost), af::SUCCESS);
}

namespace {
TensorPtr CreateTensor(const std::string &name, const std::string &dtype, const Expr &repeat) {
  TensorPtr tensor = std::make_shared<Tensor>();
  tensor->name = name;
  tensor->data_type = dtype;
  tensor->data_type_size = dtype == kFloat16 ? 2U : 4U;
  tensor->repeat = {repeat};
  tensor->stride = {af::sym::kSymbolOne};
  return tensor;
}

NodeInfo CreateSubNodeInfo(const std::string &name, const std::string &type, const std::string &input_dtype,
                           const std::string &output_dtype, const Expr &output_repeat) {
  NodeInfo node_info;
  node_info.name = name;
  node_info.node_type = type;
  node_info.inputs.emplace_back(CreateTensor(name + "_input_0", input_dtype, output_repeat));
  node_info.outputs.emplace_back(CreateTensor(name + "_output_0", output_dtype, output_repeat));
  return node_info;
}

void SetAscirNodeParamsForTest(const af::AscNodePtr &node, const ascir_param::AscirNodeParamsPtr &params) {
  ASSERT_NE(node, nullptr);
  ASSERT_NE(params, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  ASSERT_TRUE(op_desc->SetExtAttr("AscirNodeParams", params));
}

NodePerfInfo CreateVectorFuncNodePerfInfo() {
  NodePerfInfo node_perf_info;
  node_perf_info.optype = kAbs;
  node_perf_info.input_dtype = kFloat32;
  node_perf_info.output_dtype = kFloat32;
  node_perf_info.dims = {CreateExpr("vf_dim")};
  return node_perf_info;
}

void ExpectValidPerfWithoutDynamicTernary(const ascir_param::VectorFuncNodeParams &vector_func_params) {
  std::vector<NodePerfInfo> node_perf_infos = {CreateVectorFuncNodePerfInfo()};
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  Expr perf;

  ASSERT_EQ(VfPerfUtils::GetVectorFunctionPerf(node_perf_infos, vector_func_params, ternary_ops, perf), af::SUCCESS);

  EXPECT_TRUE(perf.IsValid());
  EXPECT_TRUE(ternary_ops.empty());
}

// 辅助函数：验证 contrib 变量
void VerifyContribVar(const std::map<Expr, TernaryOp, ExprCmp> &exe_times) {
  bool found_aiv_vec_contrib = false;
  for (const auto &pair : exe_times) {
    if (Str(pair.first) == "eq_Eq_AIV_VEC_contrib") {
      found_aiv_vec_contrib = true;
      const std::string &contrib_expr = pair.second.GetTernaryOpStr();
      EXPECT_NE(contrib_expr.find("eq_Eq_compare_node"), std::string::npos)
          << "contrib expr should reference eq_Eq_compare_node: " << contrib_expr;
      EXPECT_NE(contrib_expr.find("eq_exe_time"), std::string::npos)
          << "contrib expr should reference eq_exe_time: " << contrib_expr;
    }
  }
  EXPECT_TRUE(found_aiv_vec_contrib) << "Should find eq_Eq_AIV_VEC_contrib in exe_times";
}

// 辅助函数：验证描述信息
void VerifyDescriptions(const std::map<Expr, TernaryOp, ExprCmp> &exe_times) {
  bool found_eq_desc = false;
  for (const auto &pair : exe_times) {
    std::string var_name = Str(pair.first);
    std::string desc = pair.second.GetDescription();
    if (!desc.empty()) {
      std::cout << "var_name=" << var_name << ", desc=" << desc << std::endl;
      if (var_name.find("compare_node") != std::string::npos) {
        found_eq_desc = true;
        EXPECT_TRUE(desc.find("in[") != std::string::npos || desc.find("out[") != std::string::npos)
            << "Description should contain shape info: " << desc;
      }
    }
  }
  EXPECT_TRUE(found_eq_desc) << "Should find Eq node with description";
}
}  // namespace

TEST_F(TestPipePerfExpr, VectorFuncParserUsesSubgraphNodeTensorInfo) {
  af::AscGraph graph("vf_graph");
  ASSERT_EQ(af::ascir::cg::BuildVectorFuncTestGraph(graph), af::SUCCESS);

  TuningSpacePtr ts = std::make_shared<TuningSpace>();
  ASSERT_NE(ts, nullptr);
  att::AscendGraphParser ascend_graph_parser(ts);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph), af::SUCCESS);

  const auto vector_func_iter = std::find_if(ts->node_infos.begin(), ts->node_infos.end(),
                                             [](const NodeInfo &node) { return node.node_type == kVectorFunc; });
  ASSERT_NE(vector_func_iter, ts->node_infos.end());
  const auto abs_iter = std::find_if(vector_func_iter->sub_nodes_infos.begin(), vector_func_iter->sub_nodes_infos.end(),
                                     [](const NodeInfo &node) { return node.name == "abs1"; });
  ASSERT_NE(abs_iter, vector_func_iter->sub_nodes_infos.end());
  ASSERT_FALSE(abs_iter->inputs.empty());
  ASSERT_FALSE(abs_iter->outputs.empty());
  EXPECT_EQ(abs_iter->inputs[0]->name, "abs1_input_0");
  EXPECT_EQ(abs_iter->outputs[0]->name, "abs1_output_0");
}

TEST_F(TestPipePerfExpr, VectorFuncParserUsesGraphAxisWhenVectorizedAxisNotInTensorAxis) {
  af::AscGraph graph("vf_graph");
  ASSERT_EQ(af::ascir::cg::BuildVectorFuncTestGraph(graph), af::SUCCESS);
  auto merged_axis = graph.CreateAxis("merged_axis", af::Axis::kAxisTypeMerged, CreateExpr("merged_size"), {}, -1);
  af::AscGraph subgraph("vector_func");
  ASSERT_EQ(graph.FindSubGraph("vector_func", subgraph), af::SUCCESS);
  auto abs_node = subgraph.FindNode("abs1");
  ASSERT_NE(abs_node, nullptr);
  abs_node->outputs[0].attr.vectorized_axis = {merged_axis.id};
  abs_node->outputs[0].attr.vectorized_strides = {af::sym::kSymbolOne};

  TuningSpacePtr ts = std::make_shared<TuningSpace>();
  ASSERT_NE(ts, nullptr);
  att::AscendGraphParser ascend_graph_parser(ts);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph), af::SUCCESS);

  const auto vector_func_iter = std::find_if(ts->node_infos.begin(), ts->node_infos.end(),
                                             [](const NodeInfo &node) { return node.node_type == kVectorFunc; });
  ASSERT_NE(vector_func_iter, ts->node_infos.end());
  const auto abs_iter = std::find_if(vector_func_iter->sub_nodes_infos.begin(), vector_func_iter->sub_nodes_infos.end(),
                                     [](const NodeInfo &node) { return node.name == "abs1"; });
  ASSERT_NE(abs_iter, vector_func_iter->sub_nodes_infos.end());
  ASSERT_FALSE(abs_iter->outputs.empty());
  ASSERT_EQ(abs_iter->outputs[0]->repeat.size(), 1UL);
  EXPECT_EQ(Str(abs_iter->outputs[0]->repeat[0]), "merged_size");
}

TEST_F(TestPipePerfExpr, ConvertToPerfInfoUsesEachSubNodeOutputDims) {
  TuningSpacePtr ts = std::make_shared<TuningSpace>();
  PipePerfExpr pipe_perf(ts);
  std::vector<NodeInfo> sub_nodes = {
      CreateSubNodeInfo("cast1", kCast, kFloat16, kFloat32, CreateExpr("cast_size")),
      CreateSubNodeInfo("relu1", kRelu, kFloat32, kFloat32, CreateExpr("relu_size")),
  };
  std::vector<NodePerfInfo> node_perf_infos;

  ASSERT_EQ(pipe_perf.ConvertToPerfInfo(sub_nodes, node_perf_infos), af::SUCCESS);

  ASSERT_EQ(node_perf_infos.size(), 2UL);
  ASSERT_EQ(node_perf_infos[0].dims.size(), 1UL);
  ASSERT_EQ(node_perf_infos[1].dims.size(), 1UL);
  EXPECT_EQ(Str(node_perf_infos[0].dims[0]), "cast_size");
  EXPECT_EQ(Str(node_perf_infos[1].dims[0]), "relu_size");
}

TEST_F(TestPipePerfExpr, ConvertToPerfInfoUsesParentGraphDims) {
  TuningSpacePtr ts = std::make_shared<TuningSpace>();
  PipePerfExpr pipe_perf(ts);
  std::vector<NodeInfo> sub_nodes = {
      CreateSubNodeInfo("relu1", kRelu, kFloat32, kFloat32, CreateExpr("ndbt_size")),
  };
  std::vector<NodePerfInfo> node_perf_infos;
  std::vector<Expr> vector_func_dims = {CreateExpr("z0bt_size")};

  ASSERT_EQ(pipe_perf.ConvertToPerfInfo(sub_nodes, node_perf_infos, vector_func_dims), af::SUCCESS);

  ASSERT_EQ(node_perf_infos.size(), 1UL);
  ASSERT_EQ(node_perf_infos[0].dims.size(), 1UL);
  EXPECT_EQ(Str(node_perf_infos[0].dims[0]), "z0bt_size");
}

TEST_F(TestPipePerfExpr, VectorFuncPerfUsesDefaultParamsWhenNoCodegenParams) {
  ascir_param::VectorFuncNodeParams vector_func_params;

  ExpectValidPerfWithoutDynamicTernary(vector_func_params);
}

TEST_F(TestPipePerfExpr, VectorFuncPerfUsesUnequalFormulaWhenDoubleLoopHasNoStride) {
  ascir_param::VectorFuncNodeParams vector_func_params;
  vector_func_params.is_double_loop = true;
  vector_func_params.output_dims = {CreateExpr("outer_dim"), CreateExpr("inner_dim")};

  ExpectValidPerfWithoutDynamicTernary(vector_func_params);
}

TEST_F(TestPipePerfExpr, VectorFuncPerfUsesEqualFormulaWhenDoubleLoopHasSingleStride) {
  ascir_param::VectorFuncNodeParams vector_func_params;
  vector_func_params.is_double_loop = true;
  vector_func_params.all_strides = {CreateExpr("only_stride")};
  vector_func_params.output_dims = {CreateExpr("outer_dim"), CreateExpr("inner_dim")};

  ExpectValidPerfWithoutDynamicTernary(vector_func_params);
}

TEST_F(TestPipePerfExpr, VectorFuncPerfCreatesDynamicStrideTernaryOp) {
  std::vector<NodePerfInfo> node_perf_infos = {CreateVectorFuncNodePerfInfo()};
  ascir_param::VectorFuncNodeParams vector_func_params;
  vector_func_params.is_double_loop = true;
  vector_func_params.all_strides = {CreateExpr("input_stride"), CreateExpr("output_stride")};
  vector_func_params.output_dims = {CreateExpr("outer_dim"), CreateExpr("inner_dim")};
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  Expr perf;

  ASSERT_EQ(VfPerfUtils::GetVectorFunctionPerf(node_perf_infos, vector_func_params, ternary_ops, perf), af::SUCCESS);

  EXPECT_EQ(Str(perf), "vf_dynamic_perf");
  auto iter = ternary_ops.find(perf);
  ASSERT_NE(iter, ternary_ops.end());
  EXPECT_EQ(iter->second.GetDescription(), "vf_dynamic_perf");
  const std::string ternary_expr = iter->second.GetTernaryOpStr();
  EXPECT_NE(ternary_expr.find("input_stride"), std::string::npos);
  EXPECT_NE(ternary_expr.find("output_stride"), std::string::npos);
}

TEST_F(TestPipePerfExpr, ParserCopiesVectorFuncNodeParamsFromAscirNodeParams) {
  af::AscGraph graph("vf_graph");
  ASSERT_EQ(af::ascir::cg::BuildVectorFuncTestGraph(graph), af::SUCCESS);
  auto vf_node = graph.FindNode("vector_func");
  ASSERT_NE(vf_node, nullptr);
  auto params = std::make_shared<ascir_param::AscirNodeParams>();
  ascir_param::VectorFuncNodeParams expected_params;
  expected_params.is_double_loop = true;
  expected_params.all_strides = {CreateExpr("input_stride"), CreateExpr("output_stride")};
  expected_params.output_dims = {CreateExpr("outer_dim"), CreateExpr("inner_dim")};
  params->api_name = kVectorFunc;
  params->status = ascir_param::ParamBuildStatus::kBuilt;
  params->specific_params = expected_params;
  SetAscirNodeParamsForTest(vf_node, params);

  TuningSpacePtr ts = std::make_shared<TuningSpace>();
  ASSERT_NE(ts, nullptr);
  att::AscendGraphParser ascend_graph_parser(ts);
  ASSERT_EQ(ascend_graph_parser.GraphParser(graph), af::SUCCESS);

  const auto vector_func_iter = std::find_if(ts->node_infos.begin(), ts->node_infos.end(),
                                             [](const NodeInfo &node) { return node.node_type == kVectorFunc; });
  ASSERT_NE(vector_func_iter, ts->node_infos.end());
  EXPECT_TRUE(vector_func_iter->vector_func_params.is_double_loop);
  ASSERT_EQ(vector_func_iter->vector_func_params.all_strides.size(), 2U);
  EXPECT_EQ(Str(vector_func_iter->vector_func_params.all_strides[0]), "input_stride");
  ASSERT_EQ(vector_func_iter->vector_func_params.output_dims.size(), 2U);
  EXPECT_EQ(Str(vector_func_iter->vector_func_params.output_dims[1]), "inner_dim");
}

TEST_F(TestPipePerfExpr, ParserKeepsDefaultVectorFuncNodeParamsWhenAscirNodeParamsMissing) {
  af::AscGraph graph("vf_graph");
  ASSERT_EQ(af::ascir::cg::BuildVectorFuncTestGraph(graph), af::SUCCESS);

  TuningSpacePtr ts = std::make_shared<TuningSpace>();
  ASSERT_NE(ts, nullptr);
  att::AscendGraphParser ascend_graph_parser(ts);
  ASSERT_EQ(ascend_graph_parser.GraphParser(graph), af::SUCCESS);

  const auto vector_func_iter = std::find_if(ts->node_infos.begin(), ts->node_infos.end(),
                                             [](const NodeInfo &node) { return node.node_type == kVectorFunc; });
  ASSERT_NE(vector_func_iter, ts->node_infos.end());
  EXPECT_FALSE(vector_func_iter->vector_func_params.is_double_loop);
  EXPECT_TRUE(vector_func_iter->vector_func_params.all_strides.empty());
  EXPECT_TRUE(vector_func_iter->vector_func_params.output_dims.empty());
}

TEST_F(TestPipePerfExpr, VectorFuncTailPerfUsesVectorFuncInternalPath) {
  af::AscGraph graph("vf_graph");
  ASSERT_EQ(af::ascir::cg::BuildVectorFuncTestGraph(graph), af::SUCCESS);
  auto vf_node = graph.FindNode("vector_func");
  ASSERT_NE(vf_node, nullptr);
  auto params = std::make_shared<ascir_param::AscirNodeParams>();
  ascir_param::VectorFuncNodeParams vector_func_params;
  vector_func_params.is_double_loop = true;
  vector_func_params.all_strides = {CreateExpr(1)};
  vector_func_params.output_dims = {CreateExpr(1), CreateExpr(64)};
  params->api_name = kVectorFunc;
  params->status = ascir_param::ParamBuildStatus::kBuilt;
  params->specific_params = vector_func_params;
  SetAscirNodeParamsForTest(vf_node, params);

  TuningSpacePtr ts = std::make_shared<TuningSpace>();
  ASSERT_NE(ts, nullptr);
  att::AscendGraphParser ascend_graph_parser(ts);
  ASSERT_EQ(ascend_graph_parser.GraphParser(graph), af::SUCCESS);
  const auto vector_func_iter = std::find_if(ts->node_infos.begin(), ts->node_infos.end(),
                                             [](const NodeInfo &node) { return node.node_type == kVectorFunc; });
  ASSERT_NE(vector_func_iter, ts->node_infos.end());

  PipePerfExpr pipe_perf(ts);
  std::map<PipeType, Expr> node_perf;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  std::vector<PerfBreakdownGroup> perf_breakdowns;

  ASSERT_EQ(pipe_perf.GetNodePerfInternal(*vector_func_iter, node_perf, ternary_ops, perf_breakdowns, true),
            af::SUCCESS);

  const auto iter = node_perf.find(PipeType::AIV_VEC);
  ASSERT_NE(iter, node_perf.end());
  const auto ternary_iter = ternary_ops.find(iter->second);
  ASSERT_NE(ternary_iter, ternary_ops.end());
  const std::string desc = ternary_iter->second.GetDescription();
  EXPECT_NE(desc.find("VectorFunc"), std::string::npos) << desc;
  EXPECT_NE(desc.find("AIV_VEC_perf"), std::string::npos) << desc;
}

TEST_F(TestPipePerfExpr, case_get_perf_for_loop) {
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::BuildEqAscendGraphND(graph), af::SUCCESS);
  graph.FindNode("eq")->outputs[0].attr.dtype = af::DT_UINT8;
  TuningSpacePtr ts = std::make_shared<TuningSpace>();
  ASSERT_NE(ts, nullptr);
  att::AscendGraphParser ascend_graph_parser(ts);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph), af::SUCCESS);
  EXPECT_EQ(ArgListManager::GetInstance().LoadArgList(ts), af::SUCCESS);
  PipePerfExpr pipe_perf(ts);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> exe_times;
  Expr head_cost;
  EXPECT_EQ(pipe_perf.GetPerfExpr(pipe_costs, exe_times, head_cost), af::SUCCESS);
  ASSERT_EQ(pipe_costs.size(), 3);

  // 验证 exe_time 格式
  for (const auto &pair : exe_times) {
    std::string var_name = Str(pair.first);
    if (var_name.rfind("exe_time") == (var_name.length() - std::string("exe_time").length())) {
      EXPECT_EQ(pair.second.GetTernaryOpStr(), "Ceiling((S0 / (z0t_size)))");
    }
  }

  // 验证 contrib 变量
  const auto pipe_vec_cost = std::string(pipe_costs[PipeType::AIV_VEC].Str().get());
  EXPECT_NE(pipe_vec_cost.find("eq_Eq_AIV_VEC_contrib"), std::string::npos);
  VerifyContribVar(exe_times);

  // 验证描述信息
  VerifyDescriptions(exe_times);

  // 调试输出
  for (const auto &pipe_cost : pipe_costs) {
    std::cout << "pipe_cost.first: " << static_cast<int32_t>(pipe_cost.first)
              << ", pipe_cost.second: " << pipe_cost.second << std::endl;
  }
}

TEST_F(TestPipePerfExpr, case1) {
  af::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);

  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph), af::SUCCESS);
  EXPECT_EQ(ArgListManager::GetInstance().LoadArgList(tuning_space), af::SUCCESS);
  PipePerfExpr pipe_perf(tuning_space);
  std::unordered_set<std::string> skip_node_types = {kData, kWorkspace, kOutput, kTbufData};
  std::map<PipeType, Expr> pipe_costs;
  bool match_input_from_l2 = false;
  for (const auto &node : tuning_space->node_infos) {
    // 跳过不算pipe性能的node
    if (skip_node_types.count(node.node_type) != 0U) {
      continue;
    }
    std::vector<TensorPtr> l2_inputs;         // 涉及L2的tensor
    std::map<uint32_t, uint32_t> tensor_ids;  // stride==0的index

    uint32_t idx = 0U;
    bool is_input_from_l2 = false;
    GELOGD("Check node[%s] input is l2.", node.name.c_str());
    for (size_t i = 0U; i < node.inputs.size(); i++) {
      auto input_tensor = node.inputs[i];
      GELOGD("Input tensor is [%s].", input_tensor->name.c_str());
      if (input_tensor->loc == HardwareDef::GM) {
        auto node_type = input_tensor->node_type;
        GELOGD("Owner node type is [%s].", node_type.c_str());
        if (node_type != kData) {
          continue;
        }
        for (auto &stride_info : input_tensor->ori_stride) {
          if (stride_info == 0) {
            tensor_ids[idx++] = i;
            l2_inputs.emplace_back(input_tensor);
            break;
          }
        }
      }
    }
    if (l2_inputs.size() != 0U) {
      is_input_from_l2 = true;
    }

    if (is_input_from_l2) {
      match_input_from_l2 = true;
      //       EXPECT_TRUE(pipe_perf.GetL2PerfExpr(pipe_costs, node, l2_inputs, tensor_ids) != SUCCESS);
    }
  }
  // 当前构图不涉及L2，后续适配
  EXPECT_FALSE(match_input_from_l2);
}

TEST_F(TestPipePerfExpr, TestTailExeTimeCase1) {
  SubAxis z1;
  auto z1_size = std::make_unique<SubAxis>(z1);
  SubAxis z1t;
  z1t.axis_type = AxisPosition::INNER;
  z1t.parent_axis = {z1_size.get()};
  auto z1t_size = std::make_unique<SubAxis>(z1t);
  SubAxis z1T;
  z1T.axis_type = AxisPosition::OUTER;
  z1T.parent_axis = {z1_size.get()};
  auto z1T_size = std::make_unique<SubAxis>(z1T);

  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->dim_info = {z1t_size.get()};

  NodeInfo node;
  node.inputs.emplace_back(tensor);
  node.loop_axes = {z1T_size.get()};

  Expr tail_exe_times;
  Expr node_exe_times = CreateExpr("node_exe_time");
  PipePerfExpr pipe_perf(tuning_space);
  pipe_perf.GetTailExeTime(node, node_exe_times, tail_exe_times);
  EXPECT_EQ(Str(tail_exe_times), "1");
}

TEST_F(TestPipePerfExpr, TestTailExeTimeCase2) {
  auto z1size = af::Symbol("z1_size");
  auto z1tsize = af::Symbol("z1t_size");
  auto z1Tsize = af::Symbol("z1T_size");
  auto z0size = af::Symbol("z0_size");
  auto z0z1Tsize = af::Symbol("z0z1T_size");
  auto z0z1Tbsize = af::Symbol("z0z1Tb_size");

  SubAxis z1;
  z1.repeat = z1size;
  auto z1_size = std::make_unique<SubAxis>(z1);
  SubAxis z1t;
  z1t.repeat = z1tsize;
  z1t.axis_type = AxisPosition::INNER;
  z1t.parent_axis = {z1_size.get()};
  auto z1t_size = std::make_unique<SubAxis>(z1t);
  SubAxis z1T;
  z1T.repeat = z1Tsize;
  z1T.axis_type = AxisPosition::OUTER;
  z1T.parent_axis = {z1_size.get()};
  auto z1T_size = std::make_unique<SubAxis>(z1T);
  SubAxis z0;
  z0.repeat = z0size;
  auto z0_size = std::make_unique<SubAxis>(z0);
  SubAxis z0z1T;
  z0z1T.repeat = z0z1Tsize;
  z0z1T.axis_type = AxisPosition::MERGED;
  z0z1T.parent_axis = {z0_size.get(), z1T_size.get()};
  auto z0z1T_size = std::make_unique<SubAxis>(z0z1T);
  SubAxis z0z1Tb;
  z0z1Tb.repeat = z0z1Tbsize;
  z0z1Tb.axis_type = AxisPosition::INNER;
  z0z1Tb.parent_axis = {z0z1T_size.get()};
  auto z0z1Tb_size = std::make_unique<SubAxis>(z0z1Tb);

  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->dim_info = {z1t_size.get()};

  NodeInfo node;
  node.inputs.emplace_back(tensor);
  node.loop_axes = {z0z1Tb_size.get()};

  Expr tail_exe_times;
  Expr node_exe_times = CreateExpr("node_exe_time");
  PipePerfExpr pipe_perf(tuning_space);
  pipe_perf.GetTailExeTime(node, node_exe_times, tail_exe_times);
  EXPECT_EQ(Str(tail_exe_times), "Ceiling((node_exe_time / (z1T_size)))");
}

TEST_F(TestPipePerfExpr, TestTailRepeatCase1) {
  auto z1size = af::Symbol("z1_size");
  auto z1tsize = af::Symbol("z1t_size");
  auto z0size = af::Symbol("z0_size");

  SubAxis z1;
  z1.repeat = z1size;
  auto z1_size = std::make_unique<SubAxis>(z1);
  SubAxis z1t;
  z1t.repeat = z1tsize;
  z1t.axis_type = AxisPosition::INNER;
  z1t.parent_axis = {z1_size.get()};
  auto z1t_size = std::make_unique<SubAxis>(z1t);
  SubAxis z0;
  z0.repeat = z0size;
  auto z0_size = std::make_unique<SubAxis>(z0);

  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->dim_info = {z0_size.get(), z1t_size.get()};

  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  auto ret = GetTensorTailRepeat(tensor, ternary_ops);
  EXPECT_TRUE(ret.size() == 2);
  EXPECT_EQ(Str(ret[0]), "z0_size");
  EXPECT_EQ(Str(ret[1]), "z1t_size_tail");
  auto iter = ternary_ops.find(ret[1]);
  EXPECT_TRUE(iter != ternary_ops.end());
  EXPECT_EQ(iter->second.GetTernaryOpStr(),
            "TernaryOp(IsEqual(Mod(z1_size, z1t_size), 0), z1t_size, Mod(z1_size, z1t_size))");
}

TEST_F(TestPipePerfExpr, TestTailRepeatCase2) {
  auto z1size = af::Symbol(9, "z1_size");
  auto z1tsize = af::Symbol(8, "z1t_size");
  auto z0size = af::Symbol("z0_size");

  SubAxis z1;
  z1.repeat = z1size;
  auto z1_size = std::make_unique<SubAxis>(z1);
  SubAxis z1t;
  z1t.repeat = z1tsize;
  z1t.axis_type = AxisPosition::INNER;
  z1t.parent_axis = {z1_size.get()};
  auto z1t_size = std::make_unique<SubAxis>(z1t);
  SubAxis z0;
  z0.repeat = z0size;
  auto z0_size = std::make_unique<SubAxis>(z0);

  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->dim_info = {z0_size.get(), z1t_size.get()};

  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  auto ret = GetTensorTailRepeat(tensor, ternary_ops);
  EXPECT_TRUE(ret.size() == 2);
  EXPECT_EQ(Str(ret[0]), "z0_size");
  EXPECT_EQ(Str(ret[1]), "1");
}

TEST_F(TestPipePerfExpr, TestUpdatePipeHead) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "Load";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr(50), CreateExpr(1000)};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  pipe_costs[PipeType::PIPE_NONE] = af::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, ternary_ops), af::SUCCESS);
  auto iter = pipe_costs.find(PipeType::PIPE_NONE);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second), "((32.7200012207031 * block_dim) + 1575.03002929688)");
}

TEST_F(TestPipePerfExpr, TestUpdatePipeHeadV1) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "Load";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr(50), CreateExpr(1000)};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  pipe_costs[PipeType::AIV_MTE2] = af::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, ternary_ops), af::SUCCESS);
  auto iter = pipe_costs.find(PipeType::AIV_MTE2);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second), "((32.7200012207031 * block_dim) + 1575.03002929688)");
}

TEST_F(TestPipePerfExpr, TestUpdatePipeHeadV2) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "Load";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr(32), CreateExpr(64)};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  pipe_costs[PipeType::AIV_MTE2] = af::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, ternary_ops), af::SUCCESS);
  auto iter = pipe_costs.find(PipeType::AIV_MTE2);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second), "((15.8900003433228 * block_dim) + 882.090026855469)");
}

TEST_F(TestPipePerfExpr, TestUpdatePipeHeadV3) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node1;
  node1.node_type = "Load";
  node1.inputs.push_back(std::make_shared<Tensor>());
  node1.inputs[0]->repeat = {CreateExpr(32), CreateExpr(64)};
  node1.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node1);
  NodeInfo node2;
  node2.node_type = "Load";
  node2.inputs.push_back(std::make_shared<Tensor>());
  node2.inputs[0]->repeat = {CreateExpr(25), CreateExpr(1000)};
  node2.inputs[0]->data_type_size = 4U;
  test_tuning_space->node_infos.emplace_back(node2);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  pipe_costs[PipeType::AIV_MTE2] = af::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, ternary_ops), af::SUCCESS);
  auto iter = pipe_costs.find(PipeType::AIV_MTE2);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second), "((32.7200012207031 * block_dim) + 1575.03002929688)");
}

TEST_F(TestPipePerfExpr, TestUpdatePipeHeadTernaryOp) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "Load";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr("z0t_size")};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  pipe_costs[PipeType::AIV_MTE2] = af::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, ternary_ops), af::SUCCESS);
  auto iter = pipe_costs.find(PipeType::AIV_MTE2);
  EXPECT_TRUE(iter != pipe_costs.end());
  auto iter2 = ternary_ops.find(iter->second);
  EXPECT_TRUE(iter2 != ternary_ops.end());
  EXPECT_EQ(iter2->second.GetTernaryOpStr(),
            "TernaryOp((2 * z0t_size) < 25000, ((15.8900003433228 * block_dim) + 882.090026855469), ((32.7200012207031 "
            "* block_dim) + 1575.03002929688))");
}

// 测试VectorFunc性能注释生成
TEST_F(TestPipePerfExpr, TestVectorFuncPerfAnnotation) {
  af::AscGraph graph("vf_graph");
  ASSERT_EQ(af::ascir::cg::BuildVectorFuncTestGraph(graph), af::SUCCESS);

  TuningSpacePtr ts = std::make_shared<TuningSpace>();
  ASSERT_NE(ts, nullptr);
  att::AscendGraphParser ascend_graph_parser(ts);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph), af::SUCCESS);
  EXPECT_EQ(ArgListManager::GetInstance().LoadArgList(ts), af::SUCCESS);

  PipePerfExpr pipe_perf(ts);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  Expr head_cost;
  EXPECT_EQ(pipe_perf.GetPerfExpr(pipe_costs, ternary_ops, head_cost), af::SUCCESS);

  // 验证ternary_ops中包含VectorFunc性能变量
  bool found_vector_func_perf = false;
  for (const auto &entry : ternary_ops) {
    const std::string &desc = entry.second.GetDescription();
    if (desc.find("vector_func_VectorFunc") != std::string::npos) {
      found_vector_func_perf = true;
      EXPECT_TRUE(desc.find("VectorFunc") != std::string::npos) << "Description should contain node type 'VectorFunc'";
      break;
    }
  }
  EXPECT_TRUE(found_vector_func_perf) << "VectorFunc performance variable not found in ternary_ops";

  // 验证pipe_costs包含AIV_VEC类型性能
  auto iter = pipe_costs.find(PipeType::AIV_VEC);
  EXPECT_TRUE(iter != pipe_costs.end()) << "pipe_costs should contain AIV_VEC performance";
}
}  // namespace att
