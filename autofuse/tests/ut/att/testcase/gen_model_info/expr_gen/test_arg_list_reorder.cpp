/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <limits>
#include "gtest/gtest.h"
#include "gen_model_info/api_perf_register/v1/perf_param_v1.h"
#include "v35/att/api_perf_register/perf_param_v2.h"
#define private public
#include "expr_gen/arg_list_reorder.h"

namespace att {
class TestArgListReorder : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }
};

namespace {
constexpr uint32_t kTensorDataTypeSize = 2U;
constexpr uint32_t kAxisDataTypeSize = 4U;

class TestTilingScheduleConfigTable : public TilingScheduleConfigTable {
 public:
  TestTilingScheduleConfigTable(uint32_t cache_line_size, uint32_t vector_len_size)
      : cache_line_size_(cache_line_size), vector_len_size_(vector_len_size) {}

  [[nodiscard]] bool IsEnableBlockLoopAutoTune() const override {
    return false;
  }
  [[nodiscard]] bool IsEnableCacheLineCheck() const override {
    return true;
  }
  [[nodiscard]] TradeOffConfig GetTradeOffConfig() const override {
    return {};
  }
  [[nodiscard]] double GetUbThresholdPerfValEffect() const override {
    return 0.0;
  }
  [[nodiscard]] TilingScheduleConfig GetModelTilingScheduleConfig() const override {
    TilingScheduleConfig config;
    config.cache_line_size = cache_line_size_;
    config.vector_len_size = vector_len_size_;
    return config;
  }
  [[nodiscard]] uint32_t GetCacheLineSize() const override {
    return cache_line_size_;
  }
  [[nodiscard]] uint32_t GetVectorLenSize() const override {
    return vector_len_size_;
  }
  [[nodiscard]] bool IsCoreNumThresholdPenaltyEnable() const override {
    return false;
  }

 private:
  uint32_t cache_line_size_;
  uint32_t vector_len_size_;
};

struct ReduceTailSortCase {
  std::shared_ptr<TuningSpace> tuning_space;
  ModelInfo model_info;
  std::shared_ptr<TilingScheduleConfigTable> tiling_schedule_config_table;
};

struct ReduceTailSubAxes {
  std::unique_ptr<SubAxis> origin_reduce;
  std::unique_ptr<SubAxis> reduce;
  std::unique_ptr<SubAxis> tail;
};

std::shared_ptr<AttAxis> MakeAttAxis(const std::string &name) {
  auto axis = std::make_shared<AttAxis>();
  axis->name = name;
  return axis;
}

size_t GetArgIndex(const std::vector<AttAxisPtr> &arg_list, const std::string &name) {
  for (size_t i = 0U; i < arg_list.size(); ++i) {
    if (arg_list[i]->name == name) {
      return i;
    }
  }
  return arg_list.size();
}

size_t GetArgOrder(const std::vector<AttAxisPtr> &arg_list, const std::string &name) {
  const size_t index = GetArgIndex(arg_list, name);
  return index < arg_list.size() ? arg_list[index]->order : SIZE_MAX;
}

SymInfoPtr MakeAxisSize(const Expr &expr) {
  if (expr.IsConstExpr()) {
    auto size = std::make_shared<SymConstInfo>(expr);
    uint64_t value = 0UL;
    if (expr.GetConstValue(value)) {
      size->const_value = static_cast<uint32_t>(value);
    }
    size->data_type_size = kAxisDataTypeSize;
    return size;
  }
  auto size = std::make_shared<SymVarInfo>(expr);
  size->data_type_size = kAxisDataTypeSize;
  return size;
}

void InitAttAxis(const AttAxisPtr &axis, const std::string &name, const Expr &size_expr, bool bind_multicore = false) {
  axis->name = name;
  axis->axis_pos = AxisPosition::INNER;
  axis->bind_multicore = bind_multicore;
  axis->size = MakeAxisSize(size_expr);
}

ReduceTailSubAxes MakeReduceTailSubAxes(const Expr &reduce_size, const Expr &tail_size, bool bind_multicore) {
  ReduceTailSubAxes sub_axes;
  sub_axes.origin_reduce = std::make_unique<SubAxis>();
  sub_axes.reduce = std::make_unique<SubAxis>();
  sub_axes.tail = std::make_unique<SubAxis>();
  sub_axes.origin_reduce->name = "origin_reduce";
  sub_axes.reduce->name = "reduce";
  sub_axes.tail->name = "tail";
  sub_axes.origin_reduce->repeat = reduce_size;
  sub_axes.reduce->repeat = reduce_size;
  sub_axes.tail->repeat = tail_size;
  sub_axes.reduce->axis_type = AxisPosition::INNER;
  sub_axes.tail->axis_type = AxisPosition::INNER;
  sub_axes.origin_reduce->data_type_size = kAxisDataTypeSize;
  sub_axes.reduce->data_type_size = kAxisDataTypeSize;
  sub_axes.tail->data_type_size = kAxisDataTypeSize;
  sub_axes.origin_reduce->is_bind_multi_core = bind_multicore;
  sub_axes.reduce->orig_axis_name = {sub_axes.origin_reduce->name};
  sub_axes.reduce->orig_axis.emplace_back(sub_axes.origin_reduce.get());
  sub_axes.tail->is_node_innerest_dim = true;
  return sub_axes;
}

ReduceTailSortCase BuildReduceTailSortCase(const Expr &reduce_size, const Expr &tail_size,
                                           uint32_t cache_line_size = 128U, uint32_t vector_len_size = 256U,
                                           bool bind_multicore = true) {
  auto sub_axes = MakeReduceTailSubAxes(reduce_size, tail_size, bind_multicore);
  auto input = std::make_shared<Tensor>();
  auto output = std::make_shared<Tensor>();
  input->name = "reduce_tail_input";
  input->data_type_size = kTensorDataTypeSize;
  input->dim_info = {sub_axes.reduce.get(), sub_axes.tail.get()};
  input->repeat = {sub_axes.reduce->repeat, sub_axes.tail->repeat};
  input->stride = {sub_axes.tail->repeat, af::Symbol(1)};
  output->name = "reduce_tail_output";
  output->data_type_size = kTensorDataTypeSize;
  output->dim_info = {sub_axes.reduce.get(), sub_axes.tail.get()};
  output->repeat = {af::Symbol(1), sub_axes.tail->repeat};
  output->stride = {af::Symbol(0), af::Symbol(1)};

  NodeInfo node;
  node.name = "ReduceTail";
  node.node_type = "ReduceTail";
  node.inputs = {input};
  node.outputs = {output};

  ReduceTailSortCase test_case;
  test_case.tuning_space = std::make_shared<TuningSpace>();
  test_case.tiling_schedule_config_table =
      std::make_shared<TestTilingScheduleConfigTable>(cache_line_size, vector_len_size);
  test_case.tuning_space->tiling_schedule_config_table = test_case.tiling_schedule_config_table.get();
  test_case.tuning_space->node_infos = {node};
  test_case.tuning_space->sub_axes.emplace_back(std::move(sub_axes.origin_reduce));
  test_case.tuning_space->sub_axes.emplace_back(std::move(sub_axes.reduce));
  test_case.tuning_space->sub_axes.emplace_back(std::move(sub_axes.tail));
  auto reduce_axis = MakeAttAxis("reduce");
  auto tail_axis = MakeAttAxis("tail");
  InitAttAxis(reduce_axis, "reduce", reduce_size);
  InitAttAxis(tail_axis, "tail", tail_size);
  test_case.model_info.arg_list = {reduce_axis, tail_axis};
  return test_case;
}

ReduceTailSortCase BuildReduceTailSortCaseWithOriginalAxes(const Expr &reduce_size, const Expr &tail_size,
                                                           const Expr &origin_reduce_size, const Expr &origin_tail_size,
                                                           bool bind_multicore = false) {
  auto test_case = BuildReduceTailSortCase(reduce_size, tail_size, 128U, 256U, bind_multicore);
  test_case.tuning_space->sub_axes[0]->repeat = origin_reduce_size;
  auto origin_tail = std::make_unique<SubAxis>();
  origin_tail->name = "origin_tail";
  origin_tail->repeat = origin_tail_size;
  origin_tail->data_type_size = kAxisDataTypeSize;
  auto *tail = test_case.tuning_space->sub_axes[2].get();
  tail->orig_axis_name = {origin_tail->name};
  tail->orig_axis.emplace_back(origin_tail.get());
  test_case.tuning_space->sub_axes.emplace_back(std::move(origin_tail));
  return test_case;
}

ReduceTailSortCase BuildReduceInnermostSortCase() {
  auto test_case = BuildReduceTailSortCase(af::Symbol(129800), af::Symbol(84));
  test_case.tuning_space->sub_axes[1]->is_node_innerest_dim = true;
  test_case.tuning_space->sub_axes[2]->is_node_innerest_dim = false;
  return test_case;
}
}  // namespace

TEST_F(TestArgListReorder, case0) {
  // Define TuningSpace
  // Create node: MatMul
  // input : [m, k][k, n]
  // repeat : [M, K][K, N]
  // stride : [MM, KK][KK, NN]
  // output : [m, n, k]
  // repeat : [M, N, ONE]
  // stride : [MM, NN, ZERO]
  NodeInfo MatMul;
  Tensor Tensor00;
  std::shared_ptr<Tensor> tensor00 = std::make_shared<Tensor>(Tensor00);
  Tensor Tensor01;
  std::shared_ptr<Tensor> tensor01 = std::make_shared<Tensor>(Tensor01);
  Tensor Tensor02;
  std::shared_ptr<Tensor> tensor02 = std::make_shared<Tensor>(Tensor02);
  SubAxis subaxis_m;
  auto m = std::make_unique<SubAxis>(subaxis_m);
  SubAxis subaxis_k;
  auto k = std::make_unique<SubAxis>(subaxis_k);
  SubAxis subaxis_n;
  auto n = std::make_unique<SubAxis>(subaxis_n);
  auto M = af::Symbol("M");
  auto K = af::Symbol("K");
  auto N = af::Symbol("N");
  auto MM = af::Symbol("MM");
  auto KK = af::Symbol("KK");
  auto NN = af::Symbol("NN");
  auto ONE = af::Symbol(1, "ONE");
  auto ZERO = af::Symbol(0, "ZERO");
  m->name = "m";
  k->name = "k";
  n->name = "n";
  n->is_node_innerest_dim = true;
  tensor00->name = "MatMul_input_0";
  tensor00->dim_info = {m.get(), k.get()};
  tensor00->repeat = {M, K};
  tensor00->stride = {MM, KK};
  tensor01->name = "MatMul_input_1";
  tensor01->dim_info = {k.get(), n.get()};
  tensor01->repeat = {K, N};
  tensor01->stride = {KK, NN};
  tensor02->name = "MatMul_output_0";
  tensor02->dim_info = {m.get(), n.get(), k.get()};
  tensor02->repeat = {M, N, ONE};
  tensor02->stride = {MM, NN, ZERO};
  MatMul.name = "MatMul";
  MatMul.node_type = "MatMul";
  MatMul.inputs = {tensor00, tensor01};
  MatMul.outputs = {tensor02};

  // Create node: Load
  // input : [a, b]
  // repeat : [A, B]
  // stride : [AA, BB]
  // output : [a, b]
  // repeat : [A, B]
  // stride : [AA, BB]
  NodeInfo Load;
  Tensor Tensor10;
  std::shared_ptr<Tensor> tensor10 = std::make_shared<Tensor>(Tensor10);
  Tensor Tensor11;
  std::shared_ptr<Tensor> tensor11 = std::make_shared<Tensor>(Tensor11);
  SubAxis subaxis_a;
  auto a = std::make_unique<SubAxis>(subaxis_a);
  SubAxis subaxis_b;
  auto b = std::make_unique<SubAxis>(subaxis_b);
  auto A = af::Symbol("A");
  auto B = af::Symbol("B");
  auto AA = af::Symbol("AA");
  auto BB = af::Symbol("BB");
  a->name = "a";
  b->name = "b";
  tensor10->name = "Load_input";
  tensor10->dim_info = {a.get(), b.get()};
  tensor10->repeat = {A, B};
  tensor10->stride = {AA, BB};
  tensor11->name = "Load_output";
  tensor11->dim_info = {a.get(), b.get()};
  tensor11->repeat = {A, B};
  tensor11->stride = {AA, BB};
  Load.name = "Load";
  Load.node_type = "Load";
  Load.inputs = {tensor10};
  Load.outputs = {tensor11};

  auto tuning_space_ = std::make_shared<TuningSpace>();
  tuning_space_->node_infos = {MatMul, Load};
  tuning_space_->sub_axes.emplace_back(std::move(m));
  tuning_space_->sub_axes.emplace_back(std::move(k));
  tuning_space_->sub_axes.emplace_back(std::move(n));
  tuning_space_->sub_axes.emplace_back(std::move(a));
  tuning_space_->sub_axes.emplace_back(std::move(b));
  // End Define

  // Define Modelinfo
  ModelInfo model_info;
  AttAxis att_m;
  std::shared_ptr<AttAxis> attaxis_m = std::make_shared<AttAxis>(att_m);
  AttAxis att_k;
  std::shared_ptr<AttAxis> attaxis_k = std::make_shared<AttAxis>(att_k);
  AttAxis att_n;
  std::shared_ptr<AttAxis> attaxis_n = std::make_shared<AttAxis>(att_n);
  AttAxis att_a;
  std::shared_ptr<AttAxis> attaxis_a = std::make_shared<AttAxis>(att_a);
  AttAxis att_b;
  std::shared_ptr<AttAxis> attaxis_b = std::make_shared<AttAxis>(att_b);
  attaxis_m->name = "m";
  attaxis_k->name = "k";
  attaxis_n->name = "n";
  attaxis_a->name = "a";
  attaxis_b->name = "b";
  model_info.arg_list = {attaxis_m, attaxis_k, attaxis_n, attaxis_a, attaxis_b};
  // End Define

  ArgListReorder arg_list_reorder(tuning_space_);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
  std::map<std::string, size_t> arg_id_map;
  for (size_t i = 0; i < model_info.arg_list.size(); i++) {
    auto arg = model_info.arg_list[i];
    arg_id_map[arg->name] = i;
  }
  EXPECT_EQ(arg_id_map["k"], 0);
  EXPECT_EQ(arg_id_map["n"], 1);
}

TEST_F(TestArgListReorder, case1) {
  // Define TuningSpace
  // Create node: MatMul
  // input : [m, k][k, n]
  // repeat : [M, K][K, N]
  // stride : [MM, KK][KK, NN]
  // output : [m, n, k]
  // repeat : [M, N, ONE]
  // stride : [MM, NN, ZERO]
  NodeInfo node1;
  std::shared_ptr<Tensor> tensor0 = std::make_shared<Tensor>();
  std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>();

  auto z0 = std::make_unique<SubAxis>();
  auto z1 = std::make_unique<SubAxis>();
  auto z2 = std::make_unique<SubAxis>();
  auto z0z1 = std::make_unique<SubAxis>();
  auto z0z1t = std::make_unique<SubAxis>();
  auto z2t = std::make_unique<SubAxis>();

  z0->name = "z0";
  z1->name = "z1";
  z2->name = "z2";
  z0z1->name = "z0z1";
  z0z1t->name = "z0z1t";
  z2t->name = "z2t";

  z0z1t->is_node_innerest_dim = true;
  tensor0->name = "node1_input0";
  tensor0->dim_info = {z0z1t.get(), z2t.get()};
  tensor0->repeat = {af::Symbol("z0z1t_size"), af::Symbol("z2t_size")};
  tensor0->stride = {af::Symbol("z2t_size"), af::Symbol(1, "ONE")};
  tensor1->name = "node1_output0";
  tensor1->dim_info = {z0z1t.get(), z2t.get()};
  tensor1->repeat = {af::Symbol(1, "ONE"), af::Symbol("z2t_size")};
  tensor1->stride = {af::Symbol(0, "ZERO"), af::Symbol(1, "ONE")};
  node1.name = "Node1";
  node1.node_type = "LayerNorm";
  node1.inputs = {tensor0};
  node1.outputs = {tensor1};

  auto tuning_space_ = std::make_shared<TuningSpace>();
  tuning_space_->node_infos = {node1};
  tuning_space_->sub_axes.emplace_back(std::move(z0));
  tuning_space_->sub_axes.emplace_back(std::move(z1));
  tuning_space_->sub_axes.emplace_back(std::move(z2));
  tuning_space_->sub_axes.emplace_back(std::move(z0z1));
  tuning_space_->sub_axes.emplace_back(std::move(z0z1t));
  tuning_space_->sub_axes.emplace_back(std::move(z2t));
  // End Define

  // Define Modelinfo
  ModelInfo model_info;
  std::shared_ptr<AttAxis> att_z0 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z1 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z2 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z0z1 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z0z1t = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z2t = std::make_shared<AttAxis>();

  att_z0->name = "z0";
  att_z1->name = "z1";
  att_z2->name = "z2";
  att_z0z1->name = "z0z1";
  att_z0z1t->name = "z0z1t";
  att_z2t->name = "z2t";

  att_z0z1->from_axis.emplace_back(att_z0.get());
  att_z0z1->from_axis.emplace_back(att_z1.get());
  att_z0z1t->from_axis.emplace_back(att_z0z1.get());
  att_z2t->from_axis.emplace_back(att_z2.get());

  model_info.arg_list = {att_z0, att_z1, att_z2, att_z0z1, att_z0z1t, att_z2t};
  // End Define

  ArgListReorder arg_list_reorder(tuning_space_);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
  std::map<std::string, size_t> arg_id_map;
  for (size_t i = 0; i < model_info.arg_list.size(); i++) {
    auto arg = model_info.arg_list[i];
    arg_id_map[arg->name] = i;
  }
  EXPECT_EQ(arg_id_map["z0z1t"] < arg_id_map["z2t"], true);
}

TEST_F(TestArgListReorder, case2) {
  NodeInfo node1;
  std::shared_ptr<Tensor> tensor0 = std::make_shared<Tensor>();
  std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>();

  auto z0 = std::make_unique<SubAxis>();
  auto z1 = std::make_unique<SubAxis>();
  auto z2 = std::make_unique<SubAxis>();
  auto z0z1 = std::make_unique<SubAxis>();
  auto z0z1t = std::make_unique<SubAxis>();
  auto z2t = std::make_unique<SubAxis>();

  z0->name = "z0";
  z1->name = "z1";
  z2->name = "z2";
  z0z1->name = "z0z1";
  z0z1t->name = "z0z1t";
  z2t->name = "z2t";

  z0z1t->is_node_innerest_dim = true;
  tensor0->name = "node1_output0";
  tensor0->dim_info = {z0z1t.get(), z2t.get()};
  tensor0->repeat = {af::Symbol("z0z1t_size"), af::Symbol("z2t_size")};
  tensor0->stride = {af::Symbol("z2t_size"), af::Symbol(1, "ONE")};
  tensor1->name = "node1_input0";
  tensor1->dim_info = {z0z1t.get(), z2t.get()};
  tensor1->repeat = {af::Symbol(1, "ONE"), af::Symbol("z2t_size")};
  tensor1->stride = {af::Symbol(0, "ZERO"), af::Symbol(1, "ONE")};
  node1.name = "Node1";
  node1.node_type = "LayerNorm";
  node1.inputs = {tensor1};
  node1.outputs = {tensor0};

  auto tuning_space_ = std::make_shared<TuningSpace>();
  tuning_space_->node_infos = {node1};
  tuning_space_->sub_axes.emplace_back(std::move(z0));
  tuning_space_->sub_axes.emplace_back(std::move(z1));
  tuning_space_->sub_axes.emplace_back(std::move(z2));
  tuning_space_->sub_axes.emplace_back(std::move(z0z1));
  tuning_space_->sub_axes.emplace_back(std::move(z0z1t));
  tuning_space_->sub_axes.emplace_back(std::move(z2t));
  // End Define

  // Define Modelinfo
  ModelInfo model_info;
  std::shared_ptr<AttAxis> att_z0 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z1 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z2 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z0z1 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z0z1t = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z2t = std::make_shared<AttAxis>();

  att_z0->name = "z0";
  att_z1->name = "z1";
  att_z2->name = "z2";
  att_z0z1->name = "z0z1";
  att_z0z1t->name = "z0z1t";
  att_z2t->name = "z2t";

  att_z0z1->from_axis.emplace_back(att_z0.get());
  att_z0z1->from_axis.emplace_back(att_z1.get());
  att_z0z1t->from_axis.emplace_back(att_z0z1.get());
  att_z2t->from_axis.emplace_back(att_z2.get());

  model_info.arg_list = {att_z0, att_z1, att_z2, att_z0z1, att_z0z1t, att_z2t};
  // End Define

  ArgListReorder arg_list_reorder(tuning_space_);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
  std::map<std::string, size_t> arg_id_map;
  for (size_t i = 0; i < model_info.arg_list.size(); i++) {
    auto arg = model_info.arg_list[i];
    arg_id_map[arg->name] = i;
  }
  EXPECT_EQ(arg_id_map["z0z1t"] < arg_id_map["z2t"], true);
}

TEST_F(TestArgListReorder, reorder_duplicate_tiling_r_with_non_reduce_first) {
  auto test_case = BuildReduceInnermostSortCase();
  ArgListReorder arg_list_reorder(test_case.tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  ASSERT_EQ(arg_list_reorder.SortArgList(test_case.model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
  ASSERT_EQ(tiling_R_arg_list.size(), test_case.model_info.arg_list.size());
  EXPECT_LT(GetArgIndex(test_case.model_info.arg_list, "reduce"), GetArgIndex(test_case.model_info.arg_list, "tail"));
  EXPECT_LT(GetArgIndex(tiling_R_arg_list, "tail"), GetArgIndex(tiling_R_arg_list, "reduce"));
}

TEST_F(TestArgListReorder, skip_duplicate_tiling_r_when_dependencies_prevent_reorder) {
  auto test_case = BuildReduceInnermostSortCase();
  test_case.model_info.arg_list[1]->from_axis.emplace_back(test_case.model_info.arg_list[0].get());
  ArgListReorder arg_list_reorder(test_case.tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  ASSERT_EQ(arg_list_reorder.SortArgList(test_case.model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
  EXPECT_TRUE(tiling_R_arg_list.empty());
  EXPECT_LT(GetArgIndex(test_case.model_info.arg_list, "reduce"), GetArgIndex(test_case.model_info.arg_list, "tail"));
}

TEST_F(TestArgListReorder, reorder_duplicate_tiling_r_when_only_origin_order_differs) {
  auto test_case = BuildReduceInnermostSortCase();
  test_case.tuning_space->sub_axes[1]->is_node_innerest_dim = false;
  auto origin_axis = MakeAttAxis("origin");
  InitAttAxis(origin_axis, "origin", CreateExpr("origin_size"));
  origin_axis->axis_pos = AxisPosition::ORIGIN;
  test_case.model_info.arg_list.insert(test_case.model_info.arg_list.begin(), origin_axis);
  ArgListReorder arg_list_reorder(test_case.tuning_space);
  arg_list_reorder.innermost_dim_map_["origin"] = true;
  std::vector<AttAxisPtr> tiling_R_arg_list;
  ASSERT_EQ(arg_list_reorder.SortArgList(test_case.model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
  ASSERT_EQ(tiling_R_arg_list.size(), test_case.model_info.arg_list.size());
  EXPECT_LT(GetArgIndex(tiling_R_arg_list, "tail"), GetArgIndex(tiling_R_arg_list, "reduce"));
}

TEST_F(TestArgListReorder, duplicate_tiling_r_preserves_parent_order) {
  auto test_case = BuildReduceInnermostSortCase();
  test_case.model_info.arg_list[1]->from_axis.emplace_back(test_case.model_info.arg_list[0].get());
  auto outer_axis = MakeAttAxis("outer");
  InitAttAxis(outer_axis, "outer", CreateExpr("outer_size"));
  test_case.model_info.arg_list.emplace_back(outer_axis);
  ArgListReorder arg_list_reorder(test_case.tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  ASSERT_EQ(arg_list_reorder.SortArgList(test_case.model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
  ASSERT_EQ(tiling_R_arg_list.size(), 3U);
  EXPECT_LT(GetArgIndex(tiling_R_arg_list, "outer"), GetArgIndex(tiling_R_arg_list, "reduce"));
  EXPECT_LT(GetArgIndex(tiling_R_arg_list, "reduce"), GetArgIndex(tiling_R_arg_list, "tail"));
}

TEST_F(TestArgListReorder, duplicate_tiling_r_reorders_multiple_non_reduce_axes) {
  auto test_case = BuildReduceInnermostSortCase();
  constexpr size_t kAxisCount = 32U;
  for (size_t i = 0U; i < kAxisCount; ++i) {
    const auto name = "outer" + std::to_string(i);
    auto axis = MakeAttAxis(name);
    InitAttAxis(axis, name, CreateExpr((name + "_size").c_str()));
    test_case.model_info.arg_list.emplace_back(axis);
  }
  ArgListReorder arg_list_reorder(test_case.tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  ASSERT_EQ(arg_list_reorder.SortArgList(test_case.model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
  ASSERT_EQ(tiling_R_arg_list.size(), kAxisCount + 2U);
  EXPECT_EQ(GetArgIndex(tiling_R_arg_list, "reduce"), tiling_R_arg_list.size() - 1U);
}

TEST_F(TestArgListReorder, keep_tiling_r_arg_list_when_reduce_block_split) {
  auto test_case = BuildReduceTailSortCase(af::Symbol(256), af::Symbol(32));
  ArgListReorder arg_list_reorder(test_case.tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(test_case.model_info.arg_list, tiling_R_arg_list,
                                         &test_case.model_info.runtime_reorder_rules),
            af::SUCCESS);
  ASSERT_FALSE(tiling_R_arg_list.empty());
  EXPECT_TRUE(test_case.model_info.runtime_reorder_rules.empty());
  EXPECT_LT(GetArgIndex(tiling_R_arg_list, "tail"), GetArgIndex(tiling_R_arg_list, "reduce"));
}

TEST_F(TestArgListReorder, keep_tiling_r_arg_list_when_reduce_block_split_without_threshold_match) {
  auto tail_not_small_case = BuildReduceTailSortCase(af::Symbol(256), af::Symbol(64));
  ArgListReorder tail_not_small_reorder(tail_not_small_case.tuning_space);
  std::vector<AttAxisPtr> tail_not_small_tiling_R_arg_list;
  EXPECT_EQ(
      tail_not_small_reorder.SortArgList(tail_not_small_case.model_info.arg_list, tail_not_small_tiling_R_arg_list),
      af::SUCCESS);
  EXPECT_FALSE(tail_not_small_tiling_R_arg_list.empty());

  auto reduce_not_large_case = BuildReduceTailSortCase(af::Symbol(128), af::Symbol(32));
  ArgListReorder reduce_not_large_reorder(reduce_not_large_case.tuning_space);
  std::vector<AttAxisPtr> reduce_not_large_tiling_R_arg_list;
  EXPECT_EQ(reduce_not_large_reorder.SortArgList(reduce_not_large_case.model_info.arg_list,
                                                 reduce_not_large_tiling_R_arg_list),
            af::SUCCESS);
  EXPECT_FALSE(reduce_not_large_tiling_R_arg_list.empty());
}

TEST_F(TestArgListReorder, reorder_single_template_for_reduce_tile_small_tail_large_reduce) {
  auto test_case = BuildReduceTailSortCase(af::Symbol(256), af::Symbol(32), 128U, 256U, false);
  ArgListReorder arg_list_reorder(test_case.tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(test_case.model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
  EXPECT_TRUE(tiling_R_arg_list.empty());
  EXPECT_LT(GetArgIndex(test_case.model_info.arg_list, "tail"), GetArgIndex(test_case.model_info.arg_list, "reduce"));
}

TEST_F(TestArgListReorder, set_same_order_for_reduce_tile_large_reduce_large_tail) {
  auto test_case = BuildReduceTailSortCase(af::Symbol(256), af::Symbol(80), 128U, 256U, false);
  ArgListReorder arg_list_reorder(test_case.tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(test_case.model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
  EXPECT_TRUE(tiling_R_arg_list.empty());
  EXPECT_EQ(GetArgOrder(test_case.model_info.arg_list, "reduce"), GetArgOrder(test_case.model_info.arg_list, "tail"));
  const auto &reduce_axis = test_case.model_info.arg_list[GetArgIndex(test_case.model_info.arg_list, "reduce")];
  const auto &tail_axis = test_case.model_info.arg_list[GetArgIndex(test_case.model_info.arg_list, "tail")];
}

TEST_F(TestArgListReorder, set_same_order_for_reduce_tile_small_reduce_small_tail) {
  auto test_case = BuildReduceTailSortCase(af::Symbol(64), af::Symbol(32), 128U, 256U, false);
  ArgListReorder arg_list_reorder(test_case.tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(test_case.model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
  EXPECT_TRUE(tiling_R_arg_list.empty());
  EXPECT_EQ(GetArgOrder(test_case.model_info.arg_list, "reduce"), GetArgOrder(test_case.model_info.arg_list, "tail"));
}

TEST_F(TestArgListReorder, keep_reduce_first_for_reduce_tile_small_reduce_large_tail) {
  auto test_case = BuildReduceTailSortCase(af::Symbol(64), af::Symbol(80), 128U, 256U, false);
  ArgListReorder arg_list_reorder(test_case.tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(test_case.model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
  EXPECT_TRUE(tiling_R_arg_list.empty());
  EXPECT_LT(GetArgIndex(test_case.model_info.arg_list, "reduce"), GetArgIndex(test_case.model_info.arg_list, "tail"));
  EXPECT_NE(GetArgOrder(test_case.model_info.arg_list, "reduce"), GetArgOrder(test_case.model_info.arg_list, "tail"));
}

TEST_F(TestArgListReorder, set_same_order_for_reduce_tile_threshold_boundaries) {
  const std::vector<std::pair<uint64_t, uint64_t>> shapes = {{128UL, 64UL}, {129UL, 64UL}, {128UL, 65UL}};
  for (const auto &[reduce_size, tail_size] : shapes) {
    auto test_case = BuildReduceTailSortCase(af::Symbol(reduce_size), af::Symbol(tail_size), 128U, 256U, false);
    ArgListReorder arg_list_reorder(test_case.tuning_space);
    std::vector<AttAxisPtr> tiling_R_arg_list;
    EXPECT_EQ(arg_list_reorder.SortArgList(test_case.model_info.arg_list, tiling_R_arg_list), af::SUCCESS);
    EXPECT_EQ(GetArgOrder(test_case.model_info.arg_list, "reduce"), GetArgOrder(test_case.model_info.arg_list, "tail"));
  }
}

TEST_F(TestArgListReorder, skip_reduce_balance_when_combined_equal_order_group_has_more_than_two_axes) {
  auto reduce_axis = MakeAttAxis("reduce");
  auto tail_axis = MakeAttAxis("tail");
  auto transpose_axis = MakeAttAxis("transpose");
  InitAttAxis(reduce_axis, "reduce", af::Symbol(256));
  InitAttAxis(tail_axis, "tail", af::Symbol(80));
  InitAttAxis(transpose_axis, "transpose", af::Symbol(64));
  reduce_axis->order = 3U;
  tail_axis->order = 4U;
  transpose_axis->order = 5U;
  std::vector<AttAxisPtr> arg_list{reduce_axis, tail_axis, transpose_axis};

  ArgListReorder arg_list_reorder(std::make_shared<TuningSpace>());
  arg_list_reorder.equal_order_reduce_tail_axes_ = {"reduce", "tail"};
  arg_list_reorder.load_store_inner_most_dims_ = {"tail", "transpose"};
  arg_list_reorder.MakeSureLoadStoreInnerestSameOrder(arg_list);

  EXPECT_EQ(tail_axis->order, transpose_axis->order);
  EXPECT_NE(reduce_axis->order, tail_axis->order);
}

TEST_F(TestArgListReorder, fallback_for_invalid_reduce_tail_balance_inputs) {
  auto test_case = BuildReduceTailSortCase(af::Symbol(128), af::Symbol(64), 128U, 256U, false);
  ArgListReorder arg_list_reorder(test_case.tuning_space);
  ArgListReorder::ReduceTailTileInfo tile_info;
  tile_info.reduce_repeat = CreateExpr("reduce_size");
  tile_info.tail_repeat = af::Symbol(64);
  tile_info.data_type_size = kTensorDataTypeSize;
  EXPECT_EQ(arg_list_reorder.ClassifyReduceTailTilePolicy(tile_info), ArgListReorder::ReduceTailTilePolicy::kFallback);

  tile_info.reduce_repeat = af::Symbol(std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(arg_list_reorder.ClassifyReduceTailTilePolicy(tile_info), ArgListReorder::ReduceTailTilePolicy::kFallback);

  tile_info.reduce_repeat = af::Symbol(128);
  tile_info.data_type_size = 0U;
  EXPECT_EQ(arg_list_reorder.ClassifyReduceTailTilePolicy(tile_info), ArgListReorder::ReduceTailTilePolicy::kFallback);
}

TEST_F(TestArgListReorder, keep_default_single_template_for_reduce_tile_without_threshold_match) {
  auto tail_not_small_case = BuildReduceTailSortCase(af::Symbol(512), af::Symbol(48), 64U, 512U, false);
  ArgListReorder tail_not_small_reorder(tail_not_small_case.tuning_space);
  std::vector<AttAxisPtr> tail_not_small_tiling_R_arg_list;
  EXPECT_EQ(
      tail_not_small_reorder.SortArgList(tail_not_small_case.model_info.arg_list, tail_not_small_tiling_R_arg_list),
      af::SUCCESS);
  EXPECT_TRUE(tail_not_small_tiling_R_arg_list.empty());
  EXPECT_EQ(GetArgOrder(tail_not_small_case.model_info.arg_list, "reduce"),
            GetArgOrder(tail_not_small_case.model_info.arg_list, "tail"));

  auto reduce_not_large_case = BuildReduceTailSortCase(af::Symbol(256), af::Symbol(16), 64U, 1024U, false);
  ArgListReorder reduce_not_large_reorder(reduce_not_large_case.tuning_space);
  std::vector<AttAxisPtr> reduce_not_large_tiling_R_arg_list;
  EXPECT_EQ(reduce_not_large_reorder.SortArgList(reduce_not_large_case.model_info.arg_list,
                                                 reduce_not_large_tiling_R_arg_list),
            af::SUCCESS);
  EXPECT_TRUE(reduce_not_large_tiling_R_arg_list.empty());
  EXPECT_EQ(GetArgOrder(reduce_not_large_case.model_info.arg_list, "reduce"),
            GetArgOrder(reduce_not_large_case.model_info.arg_list, "tail"));

  auto origin_threshold_not_match_case =
      BuildReduceTailSortCaseWithOriginalAxes(af::Symbol(512), af::Symbol(16), af::Symbol(64), af::Symbol(80));
  ArgListReorder origin_threshold_not_match_reorder(origin_threshold_not_match_case.tuning_space);
  std::vector<AttAxisPtr> origin_threshold_not_match_tiling_R_arg_list;
  EXPECT_EQ(origin_threshold_not_match_reorder.SortArgList(origin_threshold_not_match_case.model_info.arg_list,
                                                           origin_threshold_not_match_tiling_R_arg_list),
            af::SUCCESS);
  EXPECT_TRUE(origin_threshold_not_match_tiling_R_arg_list.empty());
  EXPECT_LT(GetArgIndex(origin_threshold_not_match_case.model_info.arg_list, "reduce"),
            GetArgIndex(origin_threshold_not_match_case.model_info.arg_list, "tail"));

  auto origin_threshold_match_case =
      BuildReduceTailSortCaseWithOriginalAxes(af::Symbol(64), af::Symbol(80), af::Symbol(256), af::Symbol(16));
  ArgListReorder origin_threshold_match_reorder(origin_threshold_match_case.tuning_space);
  std::vector<AttAxisPtr> origin_threshold_match_tiling_R_arg_list;
  EXPECT_EQ(origin_threshold_match_reorder.SortArgList(origin_threshold_match_case.model_info.arg_list,
                                                       origin_threshold_match_tiling_R_arg_list),
            af::SUCCESS);
  EXPECT_TRUE(origin_threshold_match_tiling_R_arg_list.empty());
  EXPECT_LT(GetArgIndex(origin_threshold_match_case.model_info.arg_list, "tail"),
            GetArgIndex(origin_threshold_match_case.model_info.arg_list, "reduce"));
}

TEST_F(TestArgListReorder, record_runtime_reorder_for_dynamic_reduce_tile) {
  auto test_case =
      BuildReduceTailSortCaseWithOriginalAxes(CreateExpr("reduce_size"), CreateExpr("tail_size"),
                                              CreateExpr("origin_reduce_size"), CreateExpr("origin_tail_size"));
  ArgListReorder arg_list_reorder(test_case.tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(test_case.model_info.arg_list, tiling_R_arg_list,
                                         &test_case.model_info.runtime_reorder_rules),
            af::SUCCESS);
  EXPECT_TRUE(tiling_R_arg_list.empty());
  EXPECT_EQ(test_case.model_info.runtime_reorder_rules.size(), 1U);
  const auto &rule = test_case.model_info.runtime_reorder_rules[0];
  ASSERT_EQ(rule.preferred_order.size(), 2U);
  EXPECT_EQ(Str(rule.preferred_order[0]), "tail_size");
  EXPECT_EQ(Str(rule.preferred_order[1]), "reduce_size");
  EXPECT_EQ(Str(rule.condition_axis), "origin_tail_size");
  EXPECT_EQ(Str(rule.compare_axis), "origin_reduce_size");
  EXPECT_EQ(rule.condition_threshold, 64U);
  EXPECT_EQ(rule.compare_threshold, 128U);
}

TEST_F(TestArgListReorder, keep_canonical_relative_order_inside_equal_order_group) {
  auto axis0 = MakeAttAxis("axis0");
  auto axis1 = MakeAttAxis("axis1");
  auto axis2 = MakeAttAxis("axis2");
  InitAttAxis(axis0, "axis0", CreateExpr("axis0"));
  InitAttAxis(axis1, "axis1", CreateExpr("axis1"));
  InitAttAxis(axis2, "axis2", CreateExpr("axis2"));
  axis0->order = 3U;
  axis1->order = 3U;
  axis2->order = 4U;
  const std::vector<AttAxisPtr> canonical_axes{axis0, axis1, axis2};

  ArgListReorder arg_list_reorder(std::make_shared<TuningSpace>());
  RuntimeReorderRule rule;
  EXPECT_TRUE(arg_list_reorder.SetRuntimePreferredOrder(canonical_axes, canonical_axes, {3U, 2U, 1U}, rule));
  ASSERT_EQ(rule.preferred_order.size(), 3U);
  EXPECT_EQ(Str(rule.preferred_order[0]), "axis2");
  EXPECT_EQ(Str(rule.preferred_order[1]), "axis0");
  EXPECT_EQ(Str(rule.preferred_order[2]), "axis1");
  EXPECT_EQ(axis0->order, 3U);
  EXPECT_EQ(axis1->order, 3U);
}

TEST_F(TestArgListReorder, v2_micro_api_len_equals_schedule_vector_len) {
  PerfParamTableV2 perf_param_table;
  TilingScheduleConfigTableV2 tiling_schedule_config_table;
  EXPECT_EQ(perf_param_table.GetMicroApiLen(), tiling_schedule_config_table.GetVectorLenSize());
  EXPECT_EQ(tiling_schedule_config_table.GetModelTilingScheduleConfig().vector_len_size,
            tiling_schedule_config_table.GetVectorLenSize());
}

TEST_F(TestArgListReorder, v1_micro_api_len_equals_schedule_vector_len) {
  PerfParamTableV1 perf_param_table;
  TilingScheduleConfigTableV1 tiling_schedule_config_table;
  EXPECT_EQ(perf_param_table.GetMicroApiLen(), tiling_schedule_config_table.GetVectorLenSize());
  EXPECT_EQ(tiling_schedule_config_table.GetModelTilingScheduleConfig().vector_len_size,
            tiling_schedule_config_table.GetVectorLenSize());
}
}  // namespace att
