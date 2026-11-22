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
#include "gtest/gtest.h"
#include "solver.h"

class TilingCaseUT1Solver : public GeneralSolver {
 public:
  TilingCaseUT1Solver(SolverConfig &config) : GeneralSolver(config) {}
  void DisplayVarVal(uint64_t vars[]) override;
  double GetObj(uint64_t vars[]) override;
  double GetSmoothObj(uint64_t vars[]) override;
  double GetBuffCost(uint64_t vars[]) override;
  double GetLeqCost(uint64_t vars[]) override;
  bool CheckLocalValid(double leqs[], int32_t idx) override;
  void UpdateLeqs(uint64_t vars[], int32_t idx, double leqs[]) override;
  double GetBuffDiff(uint64_t *vars, double *weight) override;
  double GetLeqDiff(uint64_t *vars, double *weight) override;
  int32_t left_edge;
  int32_t right_edge;
};

double TilingCaseUT1Solver::GetObj(uint64_t vars[]) {
  double d_x = static_cast<double>(vars[0]);
  if (d_x > 10) {
    return d_x;
  }
  return 10;
}

double TilingCaseUT1Solver::GetSmoothObj(uint64_t vars[]) {
  double d_x = static_cast<double>(vars[0]);
  if (d_x > 10) {
    return d_x;
  }
  return 10;
}

double TilingCaseUT1Solver::GetBuffCost(uint64_t vars[]) {
  double d_x = static_cast<double>(vars[0]);
  return 20 - d_x;
}

double TilingCaseUT1Solver::GetBuffDiff(uint64_t *vars, double *weight) {
  double d_x = static_cast<double>(vars[0]);
  double leq_cost = d_x - 20;
  leq_cost *= weight[0] < 0 ? weight[0] : 0;
  return leq_cost;
}

double TilingCaseUT1Solver::GetLeqCost(uint64_t vars[]) {
  double d_x = static_cast<double>(vars[0]);
  if (d_x < left_edge) {
    return -d_x;
  } else if (d_x > right_edge) {
    return d_x;
  }
  return 0;
}

double TilingCaseUT1Solver::GetLeqDiff(uint64_t *vars, double *weight) {
  double d_x = static_cast<double>(vars[0]);
  double leq_cost = 0;
  if (d_x < left_edge) {
    leq_cost = -d_x;
  } else if (d_x > right_edge) {
    leq_cost = d_x;
  }
  leq_cost *= weight[0] > 0 ? weight[0] : 0;
  return leq_cost;
}

bool TilingCaseUT1Solver::CheckLocalValid(double leqs[], int32_t idx) {
  if (idx == 0) {
    return leqs[0] <= 0;
  }
  return false;
}

void TilingCaseUT1Solver::UpdateLeqs(uint64_t vars[], int32_t idx, double leqs[]) {
  double d_x = static_cast<double>(vars[0]);
  if (idx == -1 || idx == 0) {
    if (d_x >= left_edge && d_x <= right_edge) {
      leqs[0] = -1;
    } else {
      leqs[0] = 1;
    }
  }
}

void TilingCaseUT1Solver::DisplayVarVal(uint64_t vars[]) {
  return;
}

class UTTEST_GENERAL_SOLVER_01 : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    uint64_t top_num = 1;
    uint64_t search_length = 1;
    uint64_t iterations = 500;
    bool simple_ver = false;
    bool get_log = false;
    double momentum_factor = 0.9;

    SolverConfig cfg = {top_num, search_length, iterations, simple_ver, get_log, momentum_factor};

    solver_ = new TilingCaseUT1Solver(cfg);

    uint64_t init_vars[1] = {20};
    uint64_t upper_bound[1] = {20};
    uint64_t lower_bound[1] = {1};
    bool update_last[1] = {false};
    SolverInput input = {1, 1, upper_bound, lower_bound, init_vars, update_last};
    solver_->Init(input);
    solver_->Initialize(1);
  }

  void TearDown() override {}
  bool TestCoarseLoc(int32_t left_edge, int32_t right_edge, uint64_t &value, uint64_t &step, Locality &cur_locality);
  bool TestFineLoc(int32_t left_edge, int32_t right_edge, uint64_t &value, uint64_t &step, Locality &cur_locality);
  bool TestPeerLoc(int32_t left_edge, int32_t right_edge, uint64_t &value, Locality &cur_locality);
  bool TestLocateLoc(int32_t left_edge, int32_t right_edge, int32_t searched_value, uint64_t &value, uint64_t &step,
                     Locality &cur_locality, Locality &best_locality);
  bool TestTryLocate(int32_t left_edge, int32_t right_edge, uint64_t &value, Locality &best_locality);

  bool TestSearchLoc(UpdateDirection update_direction, uint64_t &step, double &cur_obj, TunePriority &cur_priority);
  bool TestHarmlessLoc(bool simple_ver, uint64_t &step, double &cur_obj);
  bool TestDilatedLoc(bool simple_ver, uint64_t &step);
  bool TestTuneLoc(bool simple_ver, UpdateDirection update_direction, uint64_t &value, TunePriority &cur_priority,
                   TunePriority &best_priority);
  bool TestTryTune(bool simple_ver, UpdateDirection update_direction, uint64_t &value, TunePriority &best_priority);

  void TestInitialize(int iter, uint64_t value);
  bool TestRecordBestVarVal(uint64_t value);

  double TestGetFuncVal(std::vector<uint64_t> vars, std::vector<double> weight, FuncInfo func_info);
  UpdateDirection TestGetDescent(int32_t left_edge, int32_t right_edge, std::vector<uint64_t> vars);

  TilingCaseUT1Solver *solver_;
};

bool UTTEST_GENERAL_SOLVER_01::TestCoarseLoc(int32_t left_edge, int32_t right_edge, uint64_t &value, uint64_t &step,
                                             Locality &cur_locality) {
  bool ret;
  solver_->left_edge = left_edge;
  solver_->right_edge = right_edge;
  solver_->UpdateCurVarVal(value, 0);
  UpdateInfo *update_info = new UpdateInfo(0, 20, UpdateDirection::NEGATIVE, 0);

  ret = solver_->GetCoarseLoc(update_info, step, cur_locality);
  return ret;
}

bool UTTEST_GENERAL_SOLVER_01::TestFineLoc(int32_t left_edge, int32_t right_edge, uint64_t &value, uint64_t &step,
                                           Locality &cur_locality) {
  bool ret;
  solver_->left_edge = left_edge;
  solver_->right_edge = right_edge;
  solver_->UpdateCurVarVal(value, 0);
  UpdateInfo *update_info = new UpdateInfo(0, 20, UpdateDirection::NEGATIVE, 0);

  ret = solver_->GetFineLoc(update_info, step, cur_locality);
  value = solver_->var_info_->cur_vars[0];
  return ret;
}

bool UTTEST_GENERAL_SOLVER_01::TestPeerLoc(int32_t left_edge, int32_t right_edge, uint64_t &value,
                                           Locality &cur_locality) {
  bool ret;
  solver_->left_edge = left_edge;
  solver_->right_edge = right_edge;
  solver_->UpdateCurVarVal(value, 0);
  UpdateInfo *update_info = new UpdateInfo(0, 20, UpdateDirection::NEGATIVE, 0);

  ret = solver_->GetPeerLoc(update_info, cur_locality);
  value = solver_->var_info_->cur_vars[0];
  return ret;
}

bool UTTEST_GENERAL_SOLVER_01::TestLocateLoc(int32_t left_edge, int32_t right_edge, int32_t searched_value,
                                             uint64_t &value, uint64_t &step, Locality &cur_locality,
                                             Locality &best_locality) {
  bool ret;
  solver_->left_edge = left_edge;
  solver_->right_edge = right_edge;
  solver_->UpdateCurVarVal(searched_value, 0);
  solver_->visited_node_->GetVarVal(solver_->var_info_->cur_vars)->searched = true;
  solver_->UpdateCurVarVal(value, 0);
  UpdateInfo *update_info = new UpdateInfo(0, 20, UpdateDirection::NEGATIVE, 0);

  ret = solver_->LocateLoc(update_info, step, cur_locality, best_locality);
  value = solver_->var_info_->cur_vars[0];
  return ret;
}

bool UTTEST_GENERAL_SOLVER_01::TestTryLocate(int32_t left_edge, int32_t right_edge, uint64_t &value,
                                             Locality &best_locality) {
  bool ret;
  solver_->left_edge = left_edge;
  solver_->right_edge = right_edge;
  solver_->UpdateCurVarVal(value, 0);

  ret = solver_->TryLocate(0, 20, best_locality);
  value = solver_->var_info_->target_val[0];
  return ret;
}

bool UTTEST_GENERAL_SOLVER_01::TestSearchLoc(UpdateDirection update_direction, uint64_t &step, double &cur_obj,
                                             TunePriority &cur_priority) {
  bool ret;
  solver_->left_edge = 5;
  solver_->right_edge = 15;
  solver_->UpdateCurVarVal(10, 0);
  solver_->Initialize(1);
  UpdateInfo *update_info = new UpdateInfo(0, 10, update_direction, 10);

  ret = solver_->SearchLoc(update_info, step, cur_obj, cur_priority);
  return ret;
}

bool UTTEST_GENERAL_SOLVER_01::TestHarmlessLoc(bool simple_ver, uint64_t &step, double &cur_obj) {
  bool ret;
  solver_->left_edge = 5;
  solver_->right_edge = 15;
  solver_->UpdateCurVarVal(15, 0);
  cur_obj = 15;
  solver_->Initialize(1);
  solver_->solver_config_.simple_ver = simple_ver;
  UpdateInfo *update_info = new UpdateInfo(0, 15, UpdateDirection::NEGATIVE, 15);

  ret = solver_->GetHarmlessLoc(update_info, step, cur_obj);
  return ret;
}

bool UTTEST_GENERAL_SOLVER_01::TestDilatedLoc(bool simple_ver, uint64_t &step) {
  bool ret;
  solver_->left_edge = 5;
  solver_->right_edge = 15;
  solver_->UpdateCurVarVal(5, 0);
  solver_->Initialize(1);
  solver_->solver_config_.simple_ver = simple_ver;
  UpdateInfo *update_info = new UpdateInfo(0, 5, UpdateDirection::POSITIVE, 10, 15);

  ret = solver_->GetDilatedLoc(update_info, step);
  return ret;
}

bool UTTEST_GENERAL_SOLVER_01::TestTuneLoc(bool simple_ver, UpdateDirection update_direction, uint64_t &value,
                                           TunePriority &cur_priority, TunePriority &best_priority) {
  bool ret;
  uint64_t step = 0;
  uint64_t thres = 0;
  solver_->left_edge = 5;
  solver_->right_edge = 15;
  solver_->UpdateCurVarVal(value, 0);
  solver_->Initialize(1);
  solver_->solver_config_.simple_ver = simple_ver;
  double init_obj = solver_->GetObj(solver_->var_info_->cur_vars);
  double init_cons = solver_->GetBuffCost(solver_->var_info_->cur_vars);
  if (update_direction == UpdateDirection::POSITIVE) {
    thres = 20 - value;
  } else {
    thres = value;
  }
  UpdateInfo *update_info = new UpdateInfo(0, thres, update_direction, init_obj, init_cons);

  ret = solver_->TuneLoc(update_info, init_obj, step, cur_priority, best_priority);
  value = solver_->var_info_->cur_vars[0];
  return ret;
}

bool UTTEST_GENERAL_SOLVER_01::TestTryTune(bool simple_ver, UpdateDirection update_direction, uint64_t &value,
                                           TunePriority &best_priority) {
  bool ret;
  solver_->left_edge = 5;
  solver_->right_edge = 15;
  solver_->UpdateCurVarVal(value, 0);
  solver_->Initialize(1);
  double init_obj = solver_->GetObj(solver_->var_info_->cur_vars);
  double init_cons = solver_->GetBuffCost(solver_->var_info_->cur_vars);
  ret = solver_->TryTune(0, update_direction, init_obj, init_cons, best_priority);
  value = solver_->var_info_->target_val[0];
  return ret;
}

void UTTEST_GENERAL_SOLVER_01::TestInitialize(int iter, uint64_t value) {
  solver_->left_edge = 5;
  solver_->right_edge = 15;
  solver_->UpdateCurVarVal(value, 0);
  solver_->Initialize(iter);
}

bool UTTEST_GENERAL_SOLVER_01::TestRecordBestVarVal(uint64_t value) {
  solver_->left_edge = 5;
  solver_->right_edge = 15;
  solver_->UpdateCurVarVal(value, 0);
  solver_->Initialize(1);
  return solver_->RecordBestVarVal();
}

double UTTEST_GENERAL_SOLVER_01::TestGetFuncVal(std::vector<uint64_t> vars, std::vector<double> weight,
                                                FuncInfo func_info) {
  uint64_t var[1] = {vars[0]};
  double weights[1] = {weight[0]};
  solver_->left_edge = 5;
  solver_->right_edge = 15;
  return solver_->GetFuncVal(var, weights, func_info);
}

UpdateDirection UTTEST_GENERAL_SOLVER_01::TestGetDescent(int32_t left_edge, int32_t right_edge,
                                                         std::vector<uint64_t> vars) {
  uint64_t var[1] = {vars[0]};
  solver_->left_edge = left_edge;
  solver_->right_edge = right_edge;
  return solver_->GetDescent(var, 0, FuncInfo::LEQ);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_coarse_loc) {
  uint64_t step;
  uint64_t value;
  Locality cur_locality;

  step = 0;
  value = 20;
  TestCoarseLoc(0, 15, value, step, cur_locality);
  EXPECT_EQ(step, 8);
  EXPECT_EQ(cur_locality, Locality::GLOBALVALID);

  step = 0;
  value = 20;
  TestCoarseLoc(-2, -1, value, step, cur_locality);
  EXPECT_EQ(step, 32);
  EXPECT_EQ(cur_locality, Locality::INVALID);

  step = 0;
  value = 20;
  TestCoarseLoc(8, 9, value, step, cur_locality);
  EXPECT_EQ(step, 8);
  EXPECT_EQ(cur_locality, Locality::CROSSREGION);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_fine_loc) {
  uint64_t step;
  uint64_t value;
  Locality cur_locality;

  step = 8;
  value = 12;
  TestFineLoc(0, 15, value, step, cur_locality);
  ASSERT_TRUE(value == 15);

  step = 32;
  value = 1;
  TestFineLoc(-2, -1, value, step, cur_locality);
  ASSERT_TRUE(value == 1);

  step = 8;
  value = 12;
  TestFineLoc(8, 9, value, step, cur_locality);
  ASSERT_TRUE(value == 12);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_peer_loc) {
  uint64_t value;
  Locality cur_locality;

  value = 15;
  TestPeerLoc(0, 15, value, cur_locality);
  ASSERT_TRUE(value == 15);

  value = 9;
  TestPeerLoc(6, 9, value, cur_locality);
  EXPECT_EQ(value, 6);
  ASSERT_TRUE(cur_locality == Locality::ALTERNATIVE);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_locate_loc) {
  uint64_t step;
  uint64_t value;
  Locality cur_locality;
  Locality best_locality;

  value = 12;
  step = 8;
  cur_locality = Locality::GLOBALVALID;
  best_locality = Locality::REJECT;
  TestLocateLoc(0, 15, -1, value, step, cur_locality, best_locality);
  EXPECT_EQ(value, 15);
  EXPECT_EQ(best_locality, Locality::GLOBALVALID);

  value = 12;
  step = 8;
  cur_locality = Locality::GLOBALVALID;
  best_locality = Locality::REJECT;
  TestLocateLoc(6, 13, -1, value, step, cur_locality, best_locality);
  EXPECT_EQ(value, 13);

  value = 12;
  step = 8;
  cur_locality = Locality::GLOBALVALID;
  best_locality = Locality::REJECT;
  TestLocateLoc(6, 13, 13, value, step, cur_locality, best_locality);
  EXPECT_EQ(value, 6);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_try_locate) {
  uint64_t value;
  Locality cur_locality;
  Locality best_locality;

  value = 20;
  best_locality = Locality::REJECT;
  TestTryLocate(0, 15, value, best_locality);
  EXPECT_EQ(value, 15);
  EXPECT_EQ(best_locality, Locality::GLOBALVALID);

  value = 20;
  best_locality = Locality::GLOBALVALID;
  EXPECT_EQ(TestTryLocate(-2, -1, value, best_locality), false);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_search_loc) {
  uint64_t step;
  double cur_obj;
  TunePriority cur_priority;
  solver_->solver_config_.search_length = 2;

  step = 0;
  cur_priority = TunePriority::REFUSE;
  TestSearchLoc(UpdateDirection::POSITIVE, step, cur_obj, cur_priority);
  EXPECT_EQ(step, 1);
  EXPECT_EQ(cur_obj, 11);
  EXPECT_EQ(cur_priority, TunePriority::OTHER);

  step = 0;
  cur_priority = TunePriority::REFUSE;
  TestSearchLoc(UpdateDirection::NEGATIVE, step, cur_obj, cur_priority);
  EXPECT_EQ(step, 1);
  EXPECT_EQ(cur_obj, 10);
  EXPECT_EQ(cur_priority, TunePriority::DILATED);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_harmless_loc) {
  uint64_t step;
  double cur_obj;

  step = 0;
  TestHarmlessLoc(false, step, cur_obj);
  EXPECT_EQ(step, 5);
  EXPECT_EQ(cur_obj, 10);

  step = 0;
  TestHarmlessLoc(true, step, cur_obj);
  EXPECT_EQ(step, 8);
  EXPECT_EQ(cur_obj, 10);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_dilated_loc) {
  uint64_t step;

  step = 0;
  TestDilatedLoc(false, step);
  EXPECT_EQ(step, 5);

  step = 0;
  TestDilatedLoc(true, step);
  EXPECT_EQ(step, 4);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_tune_loc) {
  uint64_t value;
  bool ret;
  TunePriority cur_priority;
  TunePriority best_priority;

  value = 15;
  best_priority = TunePriority::REFUSE;
  cur_priority = TunePriority::HARMLESS;
  ret = TestTuneLoc(false, UpdateDirection::NEGATIVE, value, cur_priority, best_priority);
  EXPECT_EQ(value, 10);
  ASSERT_TRUE(best_priority == TunePriority::HARMLESS);
  ASSERT_TRUE(ret == true);

  value = 5;
  best_priority = TunePriority::HARMLESS;
  cur_priority = TunePriority::DILATED;
  ret = TestTuneLoc(false, UpdateDirection::POSITIVE, value, cur_priority, best_priority);
  ASSERT_TRUE(ret == false);

  value = 5;
  best_priority = TunePriority::REFUSE;
  cur_priority = TunePriority::DILATED;
  TestTuneLoc(false, UpdateDirection::POSITIVE, value, cur_priority, best_priority);
  ASSERT_TRUE(value == 10);

  value = 7;
  best_priority = TunePriority::REFUSE;
  cur_priority = TunePriority::DILATED;
  TestTuneLoc(false, UpdateDirection::NEGATIVE, value, cur_priority, best_priority);
  ASSERT_TRUE(best_priority == TunePriority::OTHER);

  value = 7;
  best_priority = TunePriority::REFUSE;
  cur_priority = TunePriority::DILATED;
  TestTuneLoc(true, UpdateDirection::NEGATIVE, value, cur_priority, best_priority);
  ASSERT_TRUE(best_priority == TunePriority::REFUSE);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_try_tune) {
  uint64_t value;
  bool ret;
  TunePriority best_priority;

  value = 15;
  best_priority = TunePriority::REFUSE;
  ret = TestTryTune(false, UpdateDirection::NEGATIVE, value, best_priority);
  EXPECT_EQ(value, 10);
  EXPECT_EQ(ret, false);
  EXPECT_EQ(best_priority, TunePriority::HARMLESS);

  value = 5;
  best_priority = TunePriority::REFUSE;
  ret = TestTryTune(false, UpdateDirection::POSITIVE, value, best_priority);
  EXPECT_EQ(value, 10);
  EXPECT_EQ(ret, false);
  EXPECT_EQ(best_priority, TunePriority::DILATED);

  value = 15;
  best_priority = TunePriority::REFUSE;
  solver_->var_info_->upper_bound[0] = 15;
  ret = TestTryTune(false, UpdateDirection::POSITIVE, value, best_priority);
  EXPECT_EQ(ret, true);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_initialize) {
  solver_->has_feasible_ = false;
  TestInitialize(1, 1);
  EXPECT_EQ(solver_->is_feasible_, false);
  EXPECT_EQ(solver_->has_feasible_, false);
  EXPECT_EQ(solver_->var_info_->history_vars[0], 1);
  EXPECT_EQ(solver_->var_info_->rec_vars[0], 1);

  TestInitialize(2, 10);
  EXPECT_EQ(solver_->is_feasible_, true);
  EXPECT_EQ(solver_->has_feasible_, true);
  EXPECT_EQ(solver_->var_info_->history_vars[0], 1);
  EXPECT_EQ(solver_->var_info_->rec_vars[0], 10);

  TestInitialize(3, 1);
  EXPECT_EQ(solver_->is_feasible_, false);
  EXPECT_EQ(solver_->has_feasible_, true);
  EXPECT_EQ(solver_->var_info_->history_vars[0], 10);
  EXPECT_EQ(solver_->var_info_->rec_vars[0], 1);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_record_best_varval) {
  bool ret;
  ret = TestRecordBestVarVal(1);
  EXPECT_EQ(ret, false);
  EXPECT_EQ(solver_->result_->solution_num_, 0);

  ret = TestRecordBestVarVal(10);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(solver_->result_->solution_num_, 1);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_get_func_val) {
  EXPECT_EQ(TestGetFuncVal({5}, {-1}, FuncInfo::BUFFER), 15);
  EXPECT_EQ(TestGetFuncVal({15}, {-1}, FuncInfo::BUFFER), 5);

  EXPECT_EQ(TestGetFuncVal({10}, {-3}, FuncInfo::LEQ), 0);
  EXPECT_EQ(TestGetFuncVal({5}, {1}, FuncInfo::LEQ), 0);
  EXPECT_EQ(TestGetFuncVal({10}, {1}, FuncInfo::LEQ), 0);
  EXPECT_EQ(TestGetFuncVal({15}, {1}, FuncInfo::LEQ), 0);
  EXPECT_EQ(TestGetFuncVal({18}, {1}, FuncInfo::LEQ), 18);
  EXPECT_EQ(TestGetFuncVal({2}, {1}, FuncInfo::LEQ), -2);
}

TEST_F(UTTEST_GENERAL_SOLVER_01, test_get_descent) {
  solver_->var_info_->upper_bound[0] = 20;
  EXPECT_EQ(TestGetDescent(5, 15, {10}), UpdateDirection::NONE);
  EXPECT_EQ(TestGetDescent(5, 15, {3}), UpdateDirection::POSITIVE);
  EXPECT_EQ(TestGetDescent(5, 15, {1}), UpdateDirection::POSITIVE);
  EXPECT_EQ(TestGetDescent(5, 15, {17}), UpdateDirection::NEGATIVE);
  EXPECT_EQ(TestGetDescent(5, 15, {20}), UpdateDirection::NEGATIVE);

  solver_->var_info_->upper_bound[0] = 1;
  EXPECT_EQ(TestGetDescent(5, 15, {1}), UpdateDirection::NONE);
}
