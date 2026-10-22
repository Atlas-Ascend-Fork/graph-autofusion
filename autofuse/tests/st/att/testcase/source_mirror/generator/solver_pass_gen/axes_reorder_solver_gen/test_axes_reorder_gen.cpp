/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <signal.h>
#include <map>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "gtest/gtest.h"
#include "base/base_types.h"
#include "generator/solver_pass/axes_reorder_solver_code.h"
#define private public
#include "generator/solver_pass_gen/axes_reorder_solver/axes_reorder_solver_gen.h"
#include "test_common_utils.h"
using namespace att;

class TestAxesReorderSolverGen : public ::testing::Test {
 public:
  static std::filesystem::path original_work_dir_;
  static std::filesystem::path test_work_dir_;

  static void SetUpTestCase() {
    original_work_dir_ = std::filesystem::current_path();
    const auto run_root = std::filesystem::path(TOP_DIR) / "build" / "test_runs";
    std::filesystem::create_directories(run_root);
    std::string template_path = (run_root / "equal_priority_ub_XXXXXX").string();
    if (mkdtemp(template_path.data()) == nullptr) {
      FAIL() << "Failed to create test run directory";
    }
    test_work_dir_ = std::filesystem::path(template_path) / "run" / "att";
    std::filesystem::create_directories(test_work_dir_);
    std::filesystem::current_path(test_work_dir_);
  }

  static void TearDownTestCase() {
    std::filesystem::current_path(original_work_dir_);
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
  }
};

std::filesystem::path TestAxesReorderSolverGen::original_work_dir_;
std::filesystem::path TestAxesReorderSolverGen::test_work_dir_;

namespace {
bool RunCommand(const std::vector<std::string> &args, uint32_t timeout_seconds) {
  std::vector<char *> exec_args;
  exec_args.reserve(args.size() + 1U);
  for (const auto &arg : args) {
    exec_args.push_back(const_cast<char *>(arg.c_str()));
  }
  exec_args.push_back(nullptr);

  const pid_t pid = fork();
  if (pid < 0) {
    return false;
  }
  if (pid == 0) {
    execvp(exec_args[0], exec_args.data());
    _exit(127);
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
  int status = 0;
  while (true) {
    const pid_t result = waitpid(pid, &status, WNOHANG);
    if (result == pid) {
      return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    if (result < 0) {
      return false;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

std::string MakeGeneratedSolverSource(bool enable_equal_order) {
  std::ostringstream source;
  source << R"cpp(#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

#include "op_log.h"

                  struct Variable;
                  struct TilingVariable;

                  namespace {
                  constexpr char OP_NAME[] = "EqualPriorityUb";
                  struct TestState {
                    const char *scenario = "search-pr";
                    int local_calls = 0;
                    int fail_local_on_call = 0;
                    int fail_local_from_call = 0;
                  };
                  TestState g_state;
                  TilingVariable *g_var_a = nullptr;
                  TilingVariable *g_var_b = nullptr;

                  inline int64_t CeilDiv(int64_t value, int64_t align) {
                    return (value + align - 1) / align;
                  }
                  inline int64_t Ceiling(double value) {
                    return static_cast<int64_t>(value) + (value > static_cast<int64_t>(value));
                  }
                  inline int64_t Mod(int64_t value, int64_t align) {
                    return value % align;
                  }

                  inline int64_t Pr1807Ub();
                  inline bool LocalSatisfied();
                  inline int64_t LocalEval(TilingVariable **, Variable **, int64_t);
                  inline int64_t MixedEval(TilingVariable **, Variable **, int64_t);
                  inline int64_t UpperBound17(Variable **);
                  }  // namespace

#define private public
  )cpp";
  source << GetAxesSolverSolverHead(enable_equal_order);
  source << R"cpp(#undef private

                  namespace {
                  inline int64_t Pr1807Ub() {
                    const int64_t a = g_var_a->value;
                    const int64_t b = g_var_b->value;
                    const int64_t c = CeilDiv(b, 8);
                    const int64_t q0 = 1088 * c;
                    const int64_t q1 = q0 * a;
                    const int64_t q2 = 32 * c;
                    return 4 * q1 + 2 * q2 + std::max<int64_t>(8192, q2) + std::max<int64_t>(8192, std::max<int64_t>(q0, q1)) + q0;
                  }

                  inline bool LocalSatisfied() {
                    ++g_state.local_calls;
                    if (g_state.local_calls == g_state.fail_local_on_call ||
                        (g_state.fail_local_from_call > 0 && g_state.local_calls >= g_state.fail_local_from_call)) {
                      return false;
                    }
                    if (std::string(g_state.scenario) == "search-no-solution") {
                      return false;
                    }
                    if (std::string(g_state.scenario) == "search-sum") {
                      return g_var_a->value + g_var_b->value <= 32;
                    }
                    return Pr1807Ub() <= 253696;
                  }

                  inline int64_t LocalEval(TilingVariable **, Variable **, int64_t) {
                    return LocalSatisfied() ? 0 : 1;
                  }
                  inline int64_t MixedEval(TilingVariable **, Variable **, int64_t) {
                    return (g_var_a->value <= 42 && g_var_b->value <= 17) ? 0 : 1;
                  }
                  inline int64_t UpperBound17(Variable **) {
                    return 17;
                  }
                  }  // namespace

                  class TestSolver : public AxesReorderSolver {
                   public:
                    explicit TestSolver(const AxesReorderSolverInput &input) : AxesReorderSolver(input) {}
                    bool CalUsedCoreNum(double &) override {
                      return true;
                    }
                    bool CalRealUsedCoreNum(int64_t &used_core_num) override {
                      used_core_num = 1;
                      return true;
                    }
                    bool SatisfyThresholdUBSize() override {
                      return true;
                    }
                    double GetPerf() override {
                      return 0.0;
                    }
                    bool SatisfyUBSizeCacheLine(uint32_t) override {
                      return true;
                    }
                  };
  )cpp";
  source << GetAxesSolverSolverFunc(enable_equal_order);
  source << R"cpp(
    int main(int argc, char **argv) {
      if (argc != 2) {
        return 2;
      }
      g_state.scenario = argv[1];

      TilingVariable var_a;
      TilingVariable var_b;
      var_a.align = 1;
      var_b.align = 1;
      var_a.order = 10;
      var_b.order = 10;
      var_a.upper_bound = UpperBound17;
      var_b.upper_bound = UpperBound17;
      g_var_a = &var_a;
      g_var_b = &var_b;

      Constraint local_constraint;
      Constraint mixed_constraint;
      local_constraint.type = ConstraintType::LOCAL_BUFFER;
      mixed_constraint.type = ConstraintType::LB_MIXED;
      local_constraint.eval = LocalEval;
      mixed_constraint.eval = MixedEval;
      TilingVariable *tiling_vars[] = {&var_a, &var_b};
      local_constraint.rel_tiling_vars = tiling_vars;
      mixed_constraint.rel_tiling_vars = tiling_vars;
      Constraint *var_constraints[] = {&local_constraint, &mixed_constraint};
      var_a.rel_cons = var_constraints;
      var_b.rel_cons = var_constraints;
      var_a.rel_cons_size = 2;
      var_b.rel_cons_size = 2;
      Constraint *all_constraints[] = {&local_constraint, &mixed_constraint};

      AxesReorderSolverInput input;
      input.core_num = 1;
      input.ub_size = 253696;
      input.corenum_threshold = 0.0;
      input.local_buffer_vars = tiling_vars;
      input.local_buffer_vars_size = 2;
      input.ordered_local_buffer_vars = tiling_vars;
      input.ordered_to_canonical = new uint32_t[2]{0, 1};
      input.all_cons = all_constraints;
      input.all_cons_size = 2;
      TestSolver solver(input);

      const std::string scenario(g_state.scenario);
#ifdef ENABLE_EQUAL_ORDER
      if (scenario == "search-fixed") {
        var_a.SetValue(17);
        var_b.SetValue(16);
        TilingVariable *search_vars[] = {&var_b};
        return solver.BinarySearchWithAlignment(search_vars, 1, 16, 17, 1, "B search") && var_a.value == 17 &&
                       var_b.value == 16
                   ? 0
                   : 1;
      }
      if (scenario == "search-sum") {
        int64_t upper_a = -1;
        int64_t upper_b = -1;
        return solver.BinarySearchEqualPriorityAxesWithDualThreshold(tiling_vars, 1, 17, upper_a, upper_b) &&
                       var_a.value == 16 && var_b.value == 16 && upper_a == 16 && upper_b == 16
                   ? 0
                   : 1;
      }
      if (scenario == "search-pr") {
        int64_t upper_a = -1;
        int64_t upper_b = -1;
        return solver.BinarySearchEqualPriorityAxesWithDualThreshold(tiling_vars, 1, 17, upper_a, upper_b) &&
                       var_a.value == 16 && var_b.value == 16 && Pr1807Ub() == 184576
                   ? 0
                   : 1;
      }
      if (scenario == "search-iterative") {
        uint32_t axis_idx[] = {0, 1};
        return solver.IterativeSolveEqualPriorityAxes(2, axis_idx, 1, 17) && var_a.value > 0 && var_b.value > 0 &&
                       Pr1807Ub() <= 253696
                   ? 0
                   : 1;
      }
      if (scenario == "search-single") {
        var_a.SetValue(16);
        var_b.SetValue(16);
        TilingVariable *search_vars[] = {&var_b};
        return solver.BinarySearchWithAlignment(search_vars, 1, 16, 16, 1, "single search") && var_b.value == 16 ? 0 : 1;
      }
      if (scenario == "search-align8") {
        var_a.SetValue(16);
        var_b.SetValue(8);
        TilingVariable *search_vars[] = {&var_b};
        return solver.BinarySearchWithAlignment(search_vars, 1, 8, 17, 8, "align8 search") && var_b.value == 16 ? 0 : 1;
      }
      if (scenario == "search-no-solution") {
        var_a.SetValue(17);
        var_b.SetValue(1);
        TilingVariable *search_vars[] = {&var_b};
        return !solver.BinarySearchWithAlignment(search_vars, 1, 1, 17, 1, "no solution search") ? 0 : 1;
      }
      if (scenario == "search-reverse") {
        var_a.SetValue(17);
        var_b.SetValue(16);
        TilingVariable *search_vars[] = {&var_b};
        return !solver.BinarySearchWithAlignment(search_vars, 1, 17, 16, 1, "reverse search") ? 0 : 1;
      }
      if (scenario == "full-naive") {
        return solver.NaiveLocalBufTiling(true) && var_a.value <= 42 && var_b.value <= 17 && Pr1807Ub() <= 253696 ? 0 : 1;
      }
      if (scenario == "full-binary") {
        return solver.BinaryLocalBufTiling(true) && var_a.value <= 42 && var_b.value <= 17 && Pr1807Ub() <= 253696 ? 0 : 1;
      }
      if (scenario == "fallback-naive") {
        g_state.fail_local_on_call = 9;
        const bool success = solver.NaiveLocalBufTiling(true);
        std::cout << "local_calls=" << g_state.local_calls << std::endl;
        return success && g_state.local_calls > g_state.fail_local_on_call && Pr1807Ub() <= 253696 ? 0 : 1;
      }
      if (scenario == "fallback-binary") {
        g_state.fail_local_on_call = 11;
        const bool success = solver.BinaryLocalBufTiling(true);
        std::cout << "local_calls=" << g_state.local_calls << std::endl;
        return success && g_state.local_calls > g_state.fail_local_on_call && Pr1807Ub() <= 253696 ? 0 : 1;
      }
      if (scenario == "fallback-failure-naive") {
        g_state.fail_local_on_call = 9;
        g_state.fail_local_from_call = 12;
        const bool success = solver.NaiveLocalBufTiling(true);
        return !success && g_state.local_calls > g_state.fail_local_on_call ? 0 : 1;
      }
      if (scenario == "fallback-failure-binary") {
        g_state.fail_local_on_call = 11;
        g_state.fail_local_from_call = 20;
        const bool success = solver.BinaryLocalBufTiling(true);
        return !success && g_state.local_calls > g_state.fail_local_on_call ? 0 : 1;
      }
#endif
#ifdef ENABLE_NORMAL_MODE
      if (scenario == "normal-naive") {
        return solver.NaiveLocalBufTiling() && var_a.value <= 42 && var_b.value <= 17 && Pr1807Ub() <= 253696 ? 0 : 1;
      }
      if (scenario == "normal-binary") {
        return solver.BinaryLocalBufTiling() && var_a.value <= 42 && var_b.value <= 17 && Pr1807Ub() <= 253696 ? 0 : 1;
      }
#endif
      return 3;
    }
  )cpp";
  return source.str();
}

std::filesystem::path CompileGeneratedSolver(bool enable_equal_order) {
  static std::map<bool, std::filesystem::path> binaries;
  const auto iter = binaries.find(enable_equal_order);
  if (iter != binaries.end()) {
    return iter->second;
  }

  const auto source_path = TestAxesReorderSolverGen::test_work_dir_ /
                           ("tiling_func_main_equal_priority_ub_" + std::to_string(enable_equal_order) + ".cpp");
  const auto binary_path = TestAxesReorderSolverGen::test_work_dir_ /
                           ("tiling_func_main_equal_priority_ub_" + std::to_string(enable_equal_order));
  std::ofstream source_file(source_path);
  source_file << MakeGeneratedSolverSource(enable_equal_order);
  source_file.close();
  if (!source_file.good()) {
    ADD_FAILURE() << "Failed to write generated solver source";
    return {};
  }

  const char *cxx = std::getenv("CXX");
  const std::string mode_define = enable_equal_order ? "-DENABLE_EQUAL_ORDER" : "-DENABLE_NORMAL_MODE";
  const std::vector<std::string> compile_args = {cxx == nullptr ? "g++" : cxx,
                                                 "-std=c++17",
                                                 "-DSTUB_LOG",
                                                 mode_define,
                                                 "-I" + std::string(ST_DIR) + "/testcase",
                                                 source_path.filename().string(),
                                                 "-o",
                                                 binary_path.filename().string()};
  if (!RunCommand(compile_args, 60)) {
    ADD_FAILURE() << "Generated solver compilation failed";
    return {};
  }
  return binaries.emplace(enable_equal_order, binary_path).first->second;
}

void RunGeneratedSolver(bool enable_equal_order, const std::string &scenario) {
  const auto &binary = CompileGeneratedSolver(enable_equal_order);
  ASSERT_FALSE(binary.empty());
  ASSERT_TRUE(RunCommand({binary.string(), scenario}, 30)) << "Generated solver scenario " << scenario;
}
}  // namespace

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbBSearchCommitsFeasibleValue) {
  RunGeneratedSolver(true, "search-fixed");
}

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbMainSearchRestoresBothAxes) {
  RunGeneratedSolver(true, "search-sum");
}

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbPr1807Search) {
  RunGeneratedSolver(true, "search-pr");
}

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbSearchBoundaries) {
  RunGeneratedSolver(true, "search-iterative");
  RunGeneratedSolver(true, "search-single");
  RunGeneratedSolver(true, "search-align8");
  RunGeneratedSolver(true, "search-no-solution");
  RunGeneratedSolver(true, "search-reverse");
}

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbNaiveFinalCheck) {
  RunGeneratedSolver(true, "full-naive");
}

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbBinaryFinalCheck) {
  RunGeneratedSolver(true, "full-binary");
}

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbNaiveFallbackOnce) {
  RunGeneratedSolver(true, "fallback-naive");
}

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbBinaryFallbackOnce) {
  RunGeneratedSolver(true, "fallback-binary");
}

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbFallbackFailure) {
  RunGeneratedSolver(true, "fallback-failure-naive");
  RunGeneratedSolver(true, "fallback-failure-binary");
}

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbNormalModeCompilesAndRuns) {
  RunGeneratedSolver(false, "normal-naive");
  RunGeneratedSolver(false, "normal-binary");
}

TEST_F(TestAxesReorderSolverGen, TEST_ARRANGE) {
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::map<HardwareDef, Expr> hardware_cons;
  std::vector<Expr> search_args;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = af::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x0 + x1;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 1;
  axis_priority[x1] = 2;
  search_args = {x0, x1};

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.Arrange();

  auto vars = solver_gen.local_buffer_tiling_vars_;
  EXPECT_NE(vars.size(), 0);
  EXPECT_EQ(Str(vars[0]), "x0");
}

TEST_F(TestAxesReorderSolverGen, TEST_ARRANGE_SIZE_VAR_AS_INPUT) {
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  Expr x2 = CreateExpr("x2");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::map<HardwareDef, Expr> hardware_cons;
  std::vector<Expr> search_args;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = af::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::UB] = x0 + x1 + x2;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 1;
  axis_priority[x1] = 2;
  search_args = {x0, x1};

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.Arrange();

  auto vars = solver_gen.input_args_;
  EXPECT_NE(vars.size(), 0);
  EXPECT_EQ(Str(vars[0]), "x2");
}

TEST_F(TestAxesReorderSolverGen, TEST_GEN_SOLVER) {
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::vector<Expr> search_args{x0, x1};
  std::map<HardwareDef, Expr> hardware_cons;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = af::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x0 + x1;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 2;
  axis_priority[x1] = 1;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.Arrange();

  auto vars = solver_gen.local_buffer_tiling_vars_;
  EXPECT_EQ(Str(vars[0]), "x1");

  std::string impl_code = solver_gen.GenSolverFuncImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}

TEST_F(TestAxesReorderSolverGen, TEST_GEN_SOLVER_case2) {
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("block_dim");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::vector<Expr> search_args{x0, x1};
  input_args.emplace_back(x1);
  std::map<HardwareDef, Expr> hardware_cons;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = af::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x0 + x1;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 2;
  axis_priority[x1] = 1;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.Arrange();

  auto vars = solver_gen.local_buffer_tiling_vars_;

  std::string impl_code = solver_gen.GenSolverClassImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}

TEST_F(TestAxesReorderSolverGen, GenUpperBoundFunc_ConstExpr) {
  Expr var = CreateExpr("var");
  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.from_axes_map_[var] = {CreateExpr(2), CreateExpr(3)};

  std::string result = solver_gen.GenUpperBoundFunc(var);

  std::string expected = R"(    GetUpperBoundFuncPtr var_upper_bound = [](Variable **parent_vars) {
      (void)parent_vars;
      int64_t upper_bound = 1;
      upper_bound *= 2;
      upper_bound *= 3;
      return upper_bound;
    };
    var.upper_bound = var_upper_bound;
)";
  EXPECT_EQ(result, expected);
}

TEST_F(TestAxesReorderSolverGen, GenUpperBoundInfo_ConstExpr) {
  Expr var = CreateExpr("var");
  AxesReorderSolverGen solver("case_test", "TilingData");
  solver.from_axes_map_[var] = {CreateExpr(5)};  // Constant expression
  EXPECT_EQ(solver.GenUpperBoundInfo(var), "");
}

TEST_F(TestAxesReorderSolverGen, InitiateArgs_MixedValidInvalidArgs) {
  // Setup test data with mixed valid and invalid args
  AxesReorderSolverGen solver_("case_test", "TilingData");
  Expr valid = CreateExpr("valid");
  Expr invalid = CreateExpr("invalid");
  solver_.input_args_ = {valid, invalid};
  solver_.input_align_ = {{valid, af::Symbol(4)}, {invalid, af::Symbol(1)}};  // One needs alignment
  solver_.mc_args_ = {};
  solver_.local_buffer_tiling_vars_ = {};

  // Expected output
  std::string expected =
      "    Variable valid;\n"
      "    valid.value = (tiling_data.get_valid() + std::max(1, 4) - 1) / std::max(1, 4) * std::max(1, 4);\n"
      "    Variable invalid;\n"
      "    invalid.value = tiling_data.get_invalid();\n";

  std::string result = solver_.InitiateArgs();
  EXPECT_EQ(result, expected);
}

TEST_F(TestAxesReorderSolverGen, NeitherInPriorityMap) {
  // Neither expression in priority map
  // Should use string comparison
  Expr expr1 = CreateExpr("var1");
  Expr expr2 = CreateExpr("var2");
  Expr expr3 = CreateExpr("var3");
  Expr expr4 = CreateExpr("var4");
  AxesReorderSolverGen solver("case_test", "TilingData");
  solver.VarCmp(expr1, expr2);
}

TEST_F(TestAxesReorderSolverGen, GenInputInfoTest) {
  std::vector<Expr> all_cons;
  std::vector<Expr> local_buffer_cons;
  std::vector<Expr> mc_mixed_cons;

  AxesReorderSolverGen solver("case_test", "TilingData");
  std::map<HardwareDef, Expr> hardware_use_map_;
  std::vector<Expr> total_cut_cons_;
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  ExprExprMap arg_align_map_;
  ExprUintMap arg_prompt_align_map_;
  ExprUintMap is_concat_outer_map_;
  std::vector<Expr> concat_inner_dims_;

  // Initialize test data
  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(1);
  hardware_use_map_[HardwareDef::GM] = CreateExpr(1024);
  hardware_use_map_[HardwareDef::UB] = CreateExpr(512);

  total_cut_cons_.push_back(CreateExpr(256));
  total_cut_cons_.push_back(CreateExpr(128));

  mc_args_.push_back(CreateExpr("mc_var1"));
  mc_args_.push_back(CreateExpr("mc_var2"));

  Expr var1 = CreateExpr("local_var1");
  Expr var2 = CreateExpr("local_var2");
  Expr var3 = af::Symbol(255, "local_var3");

  local_buffer_tiling_vars_.push_back(var1);
  local_buffer_tiling_vars_.push_back(var2);

  arg_align_map_[var1] = af::Symbol(64);
  arg_align_map_[var2] = af::Symbol(32);

  arg_prompt_align_map_[var1] = 128;
  arg_prompt_align_map_[var3] = 64;

  is_concat_outer_map_[var1] = 1;
  is_concat_outer_map_[var2] = 0;

  concat_inner_dims_.push_back(var3);

  // Set the test data to the solver
  solver.hardware_use_map_ = hardware_use_map_;
  solver.total_cut_cons_ = total_cut_cons_;
  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.arg_align_map_ = arg_align_map_;
  solver.arg_prompt_align_map_ = arg_prompt_align_map_;
  solver.is_concat_outer_map_ = is_concat_outer_map_;
  solver.concat_inner_dims_ = concat_inner_dims_;

  std::string result = solver.GenInputInfo(all_cons, local_buffer_cons, mc_mixed_cons);

  // 4. Check local buffer variables processing
  // Verify alignment strings are generated
  EXPECT_TRUE(result.find("local_var1.align = std::max(1, 64)") != std::string::npos);
  EXPECT_TRUE(result.find("local_var2.align = std::max(1, 32)") != std::string::npos);

  std::cout << result << std::endl;
  // Verify prompt alignment for concat outer case
  EXPECT_TRUE(result.find("local_var1.prompt_align = 128") != std::string::npos);

  // Verify no prompt alignment for non-concat outer case
  EXPECT_TRUE(result.find("local_var2.prompt_align") == std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenInputInfoCase2Test) {
  std::vector<Expr> all_cons;
  std::vector<Expr> local_buffer_cons;
  std::vector<Expr> mc_mixed_cons;

  AxesReorderSolverGen solver("case_test", "TilingData");
  std::map<HardwareDef, Expr> hardware_use_map_;
  std::vector<Expr> total_cut_cons_;
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  ExprExprMap arg_align_map_;
  ExprUintMap arg_prompt_align_map_;
  ExprUintMap is_concat_outer_map_;
  std::vector<Expr> concat_inner_dims_;

  // Initialize test data
  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(1);
  hardware_use_map_[HardwareDef::GM] = CreateExpr(1024);
  hardware_use_map_[HardwareDef::UB] = CreateExpr(512);

  total_cut_cons_.push_back(CreateExpr(256));
  total_cut_cons_.push_back(CreateExpr(128));

  mc_args_.push_back(CreateExpr("mc_var1"));
  mc_args_.push_back(CreateExpr("mc_var2"));

  Expr var1 = CreateExpr("local_var1");
  Expr var2 = CreateExpr("local_var2");
  Expr var3 = af::Symbol("local_var3");

  local_buffer_tiling_vars_.push_back(var1);
  local_buffer_tiling_vars_.push_back(var2);

  arg_align_map_[var1] = af::Symbol(64);
  arg_align_map_[var2] = af::Symbol(32);

  arg_prompt_align_map_[var1] = 128;
  arg_prompt_align_map_[var3] = 64;

  is_concat_outer_map_[var1] = 1;
  is_concat_outer_map_[var2] = 0;

  concat_inner_dims_.push_back(var3);

  // Set the test data to the solver
  solver.hardware_use_map_ = hardware_use_map_;
  solver.total_cut_cons_ = total_cut_cons_;
  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.arg_align_map_ = arg_align_map_;
  solver.arg_prompt_align_map_ = arg_prompt_align_map_;
  solver.is_concat_outer_map_ = is_concat_outer_map_;
  solver.concat_inner_dims_ = concat_inner_dims_;

  std::string result = solver.GenInputInfo(all_cons, local_buffer_cons, mc_mixed_cons);

  // 4. Check local buffer variables processing
  // Verify alignment strings are generated
  EXPECT_TRUE(result.find("local_var1.align = std::max(1, 64)") != std::string::npos);
  EXPECT_TRUE(result.find("local_var2.align = std::max(1, 32)") != std::string::npos);

  std::cout << result << std::endl;
  // Verify prompt alignment for concat outer case
  EXPECT_TRUE(result.find("local_var1.prompt_align = 128") != std::string::npos);

  // Verify no prompt alignment for non-concat outer case
  EXPECT_TRUE(result.find("local_var2.prompt_align") == std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenUBThresholdFunc_WithUBAndVariables) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.SetEnableMulticoreUBTradeoff(true);
  std::string actual = solver.GenUBThresholdFunc();
  EXPECT_TRUE(actual.find("return (ub_size - 0) > static_cast<uint32_t>(input_.ub_threshold * input_.ub_size);") !=
              std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenSolverFuncImpl_WithUBAndVariables) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.SetEnableMulticoreUBTradeoff(true);
  std::string actual = solver.GenSolverFuncImpl();
  EXPECT_TRUE(actual.find("solver.Run(true, ") != std::string::npos);
}
TEST_F(TestAxesReorderSolverGen, test_contain_heavy_op) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.SetEnableMulticoreUBTradeoff(true);
  std::string actual = solver.GenSolverFuncImpl();
  EXPECT_TRUE(actual.find("solver.Run(true, ") != std::string::npos);
}
