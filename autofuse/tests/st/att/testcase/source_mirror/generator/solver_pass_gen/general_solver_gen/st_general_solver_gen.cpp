/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "base/base_types.h"
#include "generator/solver_pass_gen/general_solver/general_solver_gen.h"
#include "test_common_utils.h"

using att::CreateExpr;
using att::Expr;
using att::ExprExprMap;
using att::ExprUintMap;
using att::GeneralSolverGen;
using att::HardwareDef;
using att::PipeType;

class ST_GENERAL_SOLVER_GEN : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    Expr x0 = CreateExpr("x0");
    Expr x1 = CreateExpr("x1");
    Expr x2 = CreateExpr("x2");
    Expr x3 = CreateExpr("x3");
    Expr a = CreateExpr("a");

    solver_ = new GeneralSolverGen("Case0", "TilingData");
    solver_->SetSearchArgs({x0, x1, x3});

    ExprExprMap expr_relation;
    ExprExprMap vars_relation;
    vars_relation[x0] = x0;
    vars_relation[x1] = x1;
    solver_->SetExprRelation(expr_relation, vars_relation);

    ExprUintMap const_args;
    const_args[a] = 0;
    solver_->SetConstArgs(const_args);
    solver_->SetSolvedArgs({x2});

    std::map<PipeType, Expr> obj;
    obj[PipeType::AIC_MTE1] = x0 + x1;
    solver_->SetObj(obj);

    std::map<HardwareDef, Expr> buffer_cons;
    buffer_cons[HardwareDef::GM] = x0 + x1;

    solver_->SetBufferCons(buffer_cons);
    solver_->SetCutCons({((x0 + x1) - a)});

    ExprExprMap max_value;
    ExprExprMap min_value;
    max_value[x0] = (CreateExpr(2) * a);
    max_value[x1] = (CreateExpr(2) * a);
    min_value[x0] = af::sym::kSymbolOne;
    min_value[x1] = af::sym::kSymbolOne;
    solver_->SetMaxValue(max_value);
    solver_->SetMinValue(min_value);

    solver_->SetInnestDim({x0});
    solver_->FixVar(2, 1);
    solver_->FixRange(0, 1, 5);
  }

  void TearDown() override {
    delete solver_;
  }
  GeneralSolverGen *solver_;
};

namespace {
void AppendSolverImplPart0(std::string &codes) {
  codes += "/*\n";
  codes += "Users can override the Run function in a derived class to construct a custom solving algorithm:\n";
  codes += "  void bool Run(int32_t &solution_num, uint64_t *solutions) override;\n";
  codes += "where:\n";
  codes += "  solution_num: an int32_t parameter for the actual number of solutions found\n";
  codes +=
      "  solutions: a uint64_t array pointing to num_var * top_num elements where the algorithm stores feasible "
      "solutions\n";
  codes += "The Run function can use the following helper functions:\n";
  codes += "  bool CheckValid()\n";
  codes += "    Checks whether the current solution is feasible\n";
  codes += "  bool UpdateCurVarVal(uint64_t value, int32_t idx)\n";
  codes += "    Sets the variable at index idx to value and updates cons_info_->leqs\n";
  codes += "  bool RecordBestVarVal()\n";
  codes += "    Optimizes the objective function for the current variable values\n";
  codes += "The Run function can use the following parameters:\n";
  codes +=
      "  cons_info_->leqs, a double array storing inequality constraint values; its indices are defined as follows:\n";
  codes += "    cons_info_->leqs[0] = (x0 + x1 - hbm_size)\n";
  codes += "    cons_info_->leqs[1] = (x0 + x1 - a)\n";
  codes +=
      "  var_info_->cur_vars, a uint64_t array storing current variable values; its indices are defined as follows:\n";
  codes += "    var_info_->cur_vars[0] = x3\n";
  codes += "  var_info_->upper_bound, a uint64_t array storing upper bounds for the variables\n";
  codes += "  var_info_->lower_bound, a uint64_t array storing lower bounds for the variables\n";
  codes += "*/\n";
  codes += "class GeneralSolverCase0 : public GeneralSolver<GeneralSolverCase0>\n";
  codes += "{\n";
  codes += "    public:\n";
  codes += "        explicit GeneralSolverCase0(SolverConfig& config, TilingData& tiling_data) {\n";
  codes += "            solver_config_ = config;\n";
  codes += "            hbm_size = tiling_data.get_hbm_size();\n";
  codes += "            x2 = tiling_data.get_x2();\n";
  codes += "        }\n\n";
  codes += "        double GetObj(uint64_t* vars);\n";
  codes += "        double GetSmoothObj(uint64_t* vars);\n";
  codes += "        double GetBuffCost(uint64_t* vars);\n";
  codes += "        bool CheckLocalValid(double* leqs, int32_t idx);\n";
  codes += "        void DisplayVarVal(uint64_t* vars);\n";
  codes += "        void UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs);\n";
  codes += "        double GetBuffDiff(uint64_t* vars, double* weight);\n";
  codes += "        double GetLeqDiff(uint64_t* vars, double* weight);\n";
  codes += "        double Gethbm_sizeCost(uint64_t* vars);\n";
  codes += "        double GetSmoothhbm_sizeCost(uint64_t* vars);\n";
  codes += "        void MapVarVal(uint64_t* vars, TilingData& tiling_data);\n";
}

void AppendSolverImplPart1(std::string &codes) {
  codes += "        void GetResult(int32_t solution_num, uint64_t* solution, TilingData& tiling_data);\n";
  codes += "    private:\n";
  codes += "        const int64_t x0_idx = 0;\n";
  codes += "        const int64_t x1_idx = 1;\n";
  codes += "        uint64_t x3{1};\n";
  codes += "        uint64_t a{0};\n";
  codes += "        uint64_t hbm_size;\n";
  codes += "        uint64_t x2;\n";
  codes += "};\n";

  codes += "/*\n";
  codes += "Function: Gethbm_sizeCost(important)\n";
  codes += "Description:\n";
  codes += "  Gets cache occupancy information (occupy-buff) from hbm_size\n";
  codes += "Input parameters:\n";
  codes += "  vars:an array of length num_var corresponding to the variables\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::Gethbm_sizeCost(uint64_t* vars)\n";
  codes += "{\n";
  codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  codes += "    return (x0 + x1 - hbm_size);\n";
  codes += "}\n";
  codes += "\n";

  codes += "/*\n";
  codes += "Function: GetSmoothhbm_sizeCost(important)\n";
  codes += "Description:\n";
  codes += "  Gets smoothed cache occupancy information from hbm_size\n";
  codes += "  Compared with Gethbm_sizeCost, integer division is replaced with floating-point division\n";
  codes += "Input parameters:\n";
  codes += "  vars:an array of length num_var corresponding to the variables\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::GetSmoothhbm_sizeCost(uint64_t* vars)\n";
  codes += "{\n";
  codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  codes += "    return (x0 + x1 - hbm_size);\n";
  codes += "}\n";
  codes += "\n";

  codes += "/*\n";
  codes += "Function: GetObj(important)\n";
  codes += "Description:\n";
  codes += "  Outputs the objective function for the variable values\n";
}

void AppendSolverImplPart2(std::string &codes) {
  codes += "Input parameters:\n";
  codes += "  vars:an array of length num_var corresponding to the variables\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::GetObj(uint64_t* vars)\n";
  codes += "{\n";
  codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  codes += "    double AIC_MTE1 = (x0 + x1);\n";
  codes += "    OP_LOGD(OP_NAME, \"AIC_MTE1 = %f\", AIC_MTE1);\n";
  codes += "    return AIC_MTE1;\n";
  codes += "}\n";

  codes += "/*\n";
  codes += "Function: GetSmoothObj(important)\n";
  codes += "Description:\n";
  codes += "  Outputs the smoothed objective function for the variable values\n";
  codes += "  Compared with GetObj, integer division is replaced with floating-point division\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::GetSmoothObj(uint64_t* vars)\n";
  codes += "{\n";
  codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  codes += "    double AIC_MTE1 = (x0 + x1);\n";
  codes += "    return AIC_MTE1;\n";
  codes += "}\n";

  codes += "/*\n";
  codes += "Function: GetBuffCost(important)\n";
  codes += "Description:\n";
  codes += "  Outputs the cache occupancy penalty function (sigma(min(0, occupy-buff)^2))\n";
  codes += "  Quantifies solution quality in terms of cache occupancy\n";
  codes += "Input parameters:\n";
  codes += "  vars:an array of length num_var corresponding to the variables\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::GetBuffCost(uint64_t* vars)\n";
  codes += "{\n";
  codes += "    double hbm_size_cost = Gethbm_sizeCost(vars);\n";
  codes += "    return (Min(0, hbm_size_cost) * Min(0, hbm_size_cost));\n";
  codes += "}\n";

  codes += "/*\n";
  codes += "Function: GetBuffDiff(important)\n";
  codes += "Description:\n";
  codes += "  Gets the weighted cache occupancy difference for smooth cache occupancy\n";
  codes += "  The formula is sigma_j(delta_{var_i}(g_j(var))) * g_j(var))\n";
}

void AppendSolverImplPart3(std::string &codes) {
  codes +=
      "  where g_j is the j-th cache occupancy inequality, and delta_{var_i}(g_j(var)) is the change in g_j(var) when "
      "var_i increases by one unit\n";
  codes += "  Determines the update direction that increases cache occupancy\n";
  codes += "Input parameters:\n";
  codes += "  vars:an array of length num_var corresponding to the variables\n";
  codes += "  weight:an array of length num_leq representing the weight of each cache occupancy\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::GetBuffDiff(uint64_t* vars, double* weight)\n";
  codes += "{\n";
  codes += "    double hbm_size_cost = GetSmoothhbm_sizeCost(vars);\n";
  codes += "    hbm_size_cost *= weight[0] < 0 ? weight[0] : 0;\n";
  codes += "    return hbm_size_cost;\n";
  codes += "}\n";

  codes += "/*\n";
  codes += "Function: GetLeqDiff(important)\n";
  codes += "Description:\n";
  codes +=
      "  Gets the weighted difference of inequality constraints; the weight is the actual inequality function value\n";
  codes += "  The formula is sigma_j(delta_{var_i}(f_j(var))) * f_j(var))\n";
  codes +=
      "  where f_j is the j-th inequality constraint, and delta_{var_i}(f_j(var)) is the change in f_j(var) when var_i "
      "increases by one unit\n";
  codes += "  Determines the update direction from outside the feasible region toward the inequality boundary\n";
  codes += "Input parameters:\n";
  codes += "  vars:an array of length num_var corresponding to the variables\n";
  codes += "  weight:an array of length num_leq representing the weight of each cache occupancy\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::GetLeqDiff(uint64_t* vars, double* weight)\n";
  codes += "{\n";
  codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  codes += "    double hbm_size_cost = GetSmoothhbm_sizeCost(vars);\n";
  codes += "    hbm_size_cost *= weight[0] > 0 ? weight[0] : 0;\n";
  codes += "    double leq1_cost = (x0 + x1 - a);\n";
  codes += "    leq1_cost *= weight[1] > 0 ? weight[1] : 0;\n";
  codes += "    return hbm_size_cost + leq1_cost;\n";
  codes += "}\n";

  codes += "inline bool GeneralSolverCase0::CheckLocalValid(double* leqs, int32_t idx)\n";
  codes += "{\n";
  codes += "    if (idx == x0_idx) {\n";
  codes += "        return leqs[0] <= 0 && leqs[1] <= 0;\n";
  codes += "    } else if (idx == x1_idx) {\n";
  codes += "        return leqs[0] <= 0 && leqs[1] <= 0;\n";
  codes += "    }\n";
  codes += "    return true;\n";
  codes += "}\n";
  codes += "\n";
}

void AppendSolverImplPart4(std::string &codes) {
  codes += "inline void GeneralSolverCase0::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)\n";
  codes += "{\n";
  codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  codes += "    if (idx == x0_idx) {\n";
  codes += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  codes += "        leqs[1] = (x0 + x1 - a);\n";
  codes += "    } else if (idx == x1_idx) {\n";
  codes += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  codes += "        leqs[1] = (x0 + x1 - a);\n";
  codes += "    } else if (idx == -1) {\n";
  codes += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  codes += "        leqs[1] = (x0 + x1 - a);\n";
  codes += "    }\n";
  codes += "}\n";
  codes += "\n";

  codes += "inline void GeneralSolverCase0::DisplayVarVal(uint64_t* vars)\n";
  codes += "{\n";
  codes += "    uint64_t x0 = vars[x0_idx];\n";
  codes += "    uint64_t x1 = vars[x1_idx];\n";
  codes += "    OP_LOGD(OP_NAME, \"x3 = %lu\", static_cast<uint64_t>(1));\n";
  codes += "}\n";
  codes += "\n";

  codes += "inline void GeneralSolverCase0::MapVarVal(uint64_t* vars, TilingData& tiling_data)\n";
  codes += "{\n";
  codes += "    uint64_t x0 = vars[x0_idx];\n";
  codes += "    uint64_t x1 = vars[x1_idx];\n";
  codes += "    OP_LOGD(OP_NAME, \"The output of the solver for tilingCaseId Case0 is:\");\n";
  codes += "    tiling_data.set_x3(static_cast<uint64_t>(1));\n";
  codes += "    OP_LOGD(OP_NAME, \"x3 = %llu\", static_cast<unsigned long long>(tiling_data.get_x3()));\n";
  codes += "}\n";
  codes += "\n";

  codes +=
      "inline void GeneralSolverCase0::GetResult(int32_t solution_num, uint64_t* solution, TilingData& "
      "tiling_data)\n{\n";
  codes += "    if (solution_num > 0) {\n";
  codes += "        OP_LOGD(OP_NAME, \"Filling tilingdata for Case0.\");\n";
  codes += "        OP_LOGD(OP_NAME, \"Estimate the occupy.\");\n";
  codes +=
      "        OP_LOGD(OP_NAME, \"hbm_size = %ld\", static_cast<uint64_t>(Gethbm_sizeCost(solution) + hbm_size));\n";
  codes += "        OP_LOGD(OP_NAME, \"Simulate the cost.\");\n";
  codes += "        OP_LOGD(OP_NAME, \"Objective value for Case0 is %f.\", GetObj(solution));\n";
}

void AppendSolverImplPart5(std::string &codes) {
  codes += "        MapVarVal(solution, tiling_data);\n";
  codes += "    }\n";
  codes += "}\n\n";

  codes += "bool ExecuteCase0GeneralSolver(TilingData& tiling_data)\n";
  codes += "{\n";

  codes += "    SolverConfig cfg;\n";
  codes += "    cfg.top_num = cfg_top_num;\n";
  codes += "    cfg.search_length = cfg_search_length;\n";
  codes += "    cfg.iterations = cfg_iterations;\n";
  codes += "    cfg.simple_ver = cfg_simple_ver;\n";
  codes +=
      "    cfg.momentum_factor = cfg_momentum_factor > 1 ? 1 : (cfg_momentum_factor < 0 ? 0 : cfg_momentum_factor);\n";
  codes += "    OP_LOGD(OP_NAME, \"Record a maximum of %lu solutions.\", cfg.top_num);\n";
  codes += "    OP_LOGD(OP_NAME, \"The searching range covers %lu unit(s).\", cfg.search_length);\n";
  codes += "    OP_LOGD(OP_NAME, \"The maximum number of iterations is %lu.\", cfg.iterations);\n";
  codes += "    if (cfg.simple_ver) {\n";
  codes += "        OP_LOGD(OP_NAME, \"Using high-efficiency version.\");\n";
  codes += "    } else {\n";
  codes += "        OP_LOGD(OP_NAME, \"Using high-performance version.\");\n";
  codes += "    }\n";
  codes += "    OP_LOGD(OP_NAME, \"The momentum factor is %f.\", cfg.momentum_factor);\n";
  codes += "\n";

  codes += "    // Do not modify parameters unless marked as configurable\n";
  codes += "    // Number of variables passed from modelinfo\n";
  codes += "    int32_t num_var = 2;\n";
  codes += "    // Number of inequality constraints passed from modelinfo\n";
  codes += "    int32_t num_leq = 2;\n";
  codes +=
      "    OP_LOGD(OP_NAME, \"The number of variable is %d(x0, x1), the number of constraints is %d.\", num_var, "
      "num_leq);\n";
  codes += "    // (Configurable) Initial variable values; the algorithm tends to find a local optimum near them\n";
  codes += "    uint64_t init_vars[num_var] = {static_cast<uint64_t>(5), static_cast<uint64_t>((2 * a))};\n";
  codes +=
      "    // (Configurable) "
      "Variable upper bounds; overly large bounds increase search range and time, while overly small bounds may "
      "produce a worse local optimum\n";
  codes += "    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(5), static_cast<uint64_t>((2 * a))};\n";
  codes +=
      "    // (Configurable) "
      "Variable lower bounds; overly small bounds increase search range and time, while overly large bounds may "
      "produce a worse local optimum\n";
  codes += "    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};\n";
  codes += "    // (Configurable) Last updated variables; variables set to true stay closer to their initial values\n";
  codes += "    bool update_last[num_var] = {true, false};\n";
  codes += "    // Initialize the number of solutions to 0\n";
}

void AppendSolverImplPart6(std::string &codes) {
  codes += "    int32_t solution_num = 0;\n";
  codes += "    // Allocate memory for solver output\n";
  codes += "    uint64_t* solution = new(std::nothrow) uint64_t[num_var * cfg.top_num];\n";
  codes += "    if (solution == nullptr)\n";
  codes += "    {\n";
  codes += "        OP_LOGW(OP_NAME, \"Create solution failed.\");\n";
  codes += "        return false;\n";
  codes += "    }\n";
  codes += "    // Generic solver input parameters\n";
  codes += "    SolverInput input;\n";
  codes += "    input.var_num = num_var;\n";
  codes += "    input.leq_num = num_leq;\n";
  codes += "    input.cur_vars = init_vars;\n";
  codes += "    input.upper_bound = upper_bound;\n";
  codes += "    input.lower_bound = lower_bound;\n";
  codes += "    input.update_last = update_last;\n";
  codes +=
      "    OP_LOGD(OP_NAME, \"x0->init value: %lu, range: [%lu, %lu].\", init_vars[0], lower_bound[0], "
      "upper_bound[0]);\n";
  codes +=
      "    OP_LOGD(OP_NAME, \"x1->init value: %lu, range: [%lu, %lu].\", init_vars[1], lower_bound[1], "
      "upper_bound[1]);\n";
  codes += "\n";

  codes += "    GeneralSolverCase0* solver = new(std::nothrow) GeneralSolverCase0(cfg, tiling_data);\n";
  codes += "    if (solver != nullptr) {\n";
  codes += "        // Import and initialize generic solver input parameters\n";
  codes += "        OP_LOGD(OP_NAME, \"Start initializing the input.\");\n";
  codes += "        if (solver -> Init(input)) {\n";
  codes += "            // Run the generic solver and obtain algorithm solutions\n";
  codes += "            OP_LOGD(OP_NAME, \"Initialization finished, start running the solver.\");\n";
  codes += "            if (solver -> Run(solution_num, solution)) {\n";
  codes += "                solver -> GetResult(solution_num, solution, tiling_data);\n";
  codes += "                delete solver;\n";
  codes += "                delete[] solution;\n";
  codes += "                OP_LOGD(OP_NAME, \"The solver executed successfully.\");\n";
  codes += "                return true;\n";
  codes += "            }\n";
  codes += "            OP_LOGW(OP_NAME, \"Failed to find any solution.\");\n";
  codes += "        }\n";
  codes += "    }\n";
  codes += "    if (solver != nullptr) {\n";
  codes += "        delete solver;\n";
  codes += "    }\n";
  codes += "    if (solution != nullptr) {\n";
}

void AppendSolverImplPart7(std::string &codes) {
  codes += "        delete[] solution;\n";
  codes += "    }\n";

  codes += "    OP_LOGW(OP_NAME, \"The solver execution failed.\");\n";
  codes += "    return false;\n";
  codes += "}\n";
  codes += "\n";
}

std::string GetExpectedSolverImplCodes() {
  std::string codes;
  AppendSolverImplPart0(codes);
  AppendSolverImplPart1(codes);
  AppendSolverImplPart2(codes);
  AppendSolverImplPart3(codes);
  AppendSolverImplPart4(codes);
  AppendSolverImplPart5(codes);
  AppendSolverImplPart6(codes);
  AppendSolverImplPart7(codes);
  return codes;
}

void AppendSolverInvokePart0(std::string &codes) {
  codes += "bool ExecuteCase0GeneralSolver(TilingData& tiling_data)\n";
  codes += "{\n";

  codes += "    SolverConfig cfg;\n";
  codes += "    cfg.top_num = cfg_top_num;\n";
  codes += "    cfg.search_length = cfg_search_length;\n";
  codes += "    cfg.iterations = cfg_iterations;\n";
  codes += "    cfg.simple_ver = cfg_simple_ver;\n";
  codes +=
      "    cfg.momentum_factor = cfg_momentum_factor > 1 ? 1 : (cfg_momentum_factor < 0 ? 0 : cfg_momentum_factor);\n";
  codes += "    OP_LOGD(OP_NAME, \"Record a maximum of %lu solutions.\", cfg.top_num);\n";
  codes += "    OP_LOGD(OP_NAME, \"The searching range covers %lu unit(s).\", cfg.search_length);\n";
  codes += "    OP_LOGD(OP_NAME, \"The maximum number of iterations is %lu.\", cfg.iterations);\n";
  codes += "    if (cfg.simple_ver) {\n";
  codes += "        OP_LOGD(OP_NAME, \"Using high-efficiency version.\");\n";
  codes += "    } else {\n";
  codes += "        OP_LOGD(OP_NAME, \"Using high-performance version.\");\n";
  codes += "    }\n";
  codes += "    OP_LOGD(OP_NAME, \"The momentum factor is %f.\", cfg.momentum_factor);\n";
  codes += "\n";

  codes += "    // Do not modify parameters unless marked as configurable\n";
  codes += "    // Number of variables passed from modelinfo\n";
  codes += "    int32_t num_var = 2;\n";
  codes += "    // Number of inequality constraints passed from modelinfo\n";
  codes += "    int32_t num_leq = 2;\n";
  codes +=
      "    OP_LOGD(OP_NAME, \"The number of variable is %d(x0, x1), the number of constraints is %d.\", num_var, "
      "num_leq);\n";
  codes += "    // (Configurable) Initial variable values; the algorithm tends to find a local optimum near them\n";
  codes += "    uint64_t init_vars[num_var] = {static_cast<uint64_t>(5), static_cast<uint64_t>((2 * a))};\n";
  codes +=
      "    // (Configurable) "
      "Variable upper bounds; overly large bounds increase search range and time, while overly small bounds may "
      "produce a worse local optimum\n";
  codes += "    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(5), static_cast<uint64_t>((2 * a))};\n";
  codes +=
      "    // (Configurable) "
      "Variable lower bounds; overly small bounds increase search range and time, while overly large bounds may "
      "produce a worse local optimum\n";
  codes += "    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};\n";
  codes += "    // (Configurable) Last updated variables; variables set to true stay closer to their initial values\n";
  codes += "    bool update_last[num_var] = {true, false};\n";
  codes += "    // Initialize the number of solutions to 0\n";
  codes += "    int32_t solution_num = 0;\n";
}

void AppendSolverInvokePart1(std::string &codes) {
  codes += "    // Allocate memory for solver output\n";
  codes += "    uint64_t* solution = new(std::nothrow) uint64_t[num_var * cfg.top_num];\n";
  codes += "    if (solution == nullptr)\n";
  codes += "    {\n";
  codes += "        OP_LOGW(OP_NAME, \"Create solution failed.\");\n";
  codes += "        return false;\n";
  codes += "    }\n";
  codes += "    // Generic solver input parameters\n";
  codes += "    SolverInput input;\n";
  codes += "    input.var_num = num_var;\n";
  codes += "    input.leq_num = num_leq;\n";
  codes += "    input.cur_vars = init_vars;\n";
  codes += "    input.upper_bound = upper_bound;\n";
  codes += "    input.lower_bound = lower_bound;\n";
  codes += "    input.update_last = update_last;\n";
  codes +=
      "    OP_LOGD(OP_NAME, \"x0->init value: %lu, range: [%lu, %lu].\", init_vars[0], lower_bound[0], "
      "upper_bound[0]);\n";
  codes +=
      "    OP_LOGD(OP_NAME, \"x1->init value: %lu, range: [%lu, %lu].\", init_vars[1], lower_bound[1], "
      "upper_bound[1]);\n";
  codes += "\n";

  codes += "    GeneralSolverCase0* solver = new(std::nothrow) GeneralSolverCase0(cfg, tiling_data);\n";
  codes += "    if (solver != nullptr) {\n";
  codes += "        // Import and initialize generic solver input parameters\n";
  codes += "        OP_LOGD(OP_NAME, \"Start initializing the input.\");\n";
  codes += "        if (solver -> Init(input)) {\n";
  codes += "            // Run the generic solver and obtain algorithm solutions\n";
  codes += "            OP_LOGD(OP_NAME, \"Initialization finished, start running the solver.\");\n";
  codes += "            if (solver -> Run(solution_num, solution)) {\n";
  codes += "                solver -> GetResult(solution_num, solution, tiling_data);\n";
  codes += "                delete solver;\n";
  codes += "                delete[] solution;\n";
  codes += "                OP_LOGD(OP_NAME, \"The solver executed successfully.\");\n";
  codes += "                return true;\n";
  codes += "            }\n";
  codes += "            OP_LOGW(OP_NAME, \"Failed to find any solution.\");\n";
  codes += "        }\n";
  codes += "    }\n";
  codes += "    if (solver != nullptr) {\n";
  codes += "        delete solver;\n";
  codes += "    }\n";
  codes += "    if (solution != nullptr) {\n";
  codes += "        delete[] solution;\n";
}

void AppendSolverInvokePart2(std::string &codes) {
  codes += "    }\n";

  codes += "    OP_LOGW(OP_NAME, \"The solver execution failed.\");\n";
  codes += "    return false;\n";
  codes += "}\n";
  codes += "\n";
}

std::string GetExpectedSolverInvokeCodes() {
  std::string codes;
  AppendSolverInvokePart0(codes);
  AppendSolverInvokePart1(codes);
  AppendSolverInvokePart2(codes);
  return codes;
}
}  // namespace

TEST_F(ST_GENERAL_SOLVER_GEN, test_gen_solver_impl) {
  std::string codes = solver_->GenSolverClassImpl();
  std::string expect_codes = GetExpectedSolverImplCodes();
  EXPECT_NE(codes, "");
}

TEST_F(ST_GENERAL_SOLVER_GEN, test_gen_solver_invoke) {
  std::string codes = solver_->GenSolverFuncInvoke();
  std::string expect_codes = GetExpectedSolverInvokeCodes();
  EXPECT_NE(codes, "");
}
