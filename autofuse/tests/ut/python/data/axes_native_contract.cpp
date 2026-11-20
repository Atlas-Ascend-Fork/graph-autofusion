/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <type_traits>
#include <vector>
#define OP_LOGD(...) ((void)0)
#define OP_LOGI(...) ((void)0)
#define OP_LOGW(...) ((void)0)
#define OP_LOGE(...) ((void)0)
#define OP_NAME "NativeEvaluationContract"
#include "axes_native_core.h"

template <typename Base>
class NativeSolver : public Base {
 public:
  explicit NativeSolver(const AxesReorderSolverInput &input) : Base(input) {}

 protected:
  bool CalUsedCoreNum(double &value) override {
    value = 1.0;
    return true;
  }
  bool CalRealUsedCoreNum(int64_t &value) override {
    value = 1;
    return true;
  }
  bool SatisfyThresholdUBSize() override {
    return true;
  }
  bool SatisfyUBSizeCacheLine(uint32_t) override {
    return true;
  }
  double GetPerf() override {
    return 1.0;
  }
};

class CoreCostSolver : public AxesReorderSolver {
 public:
  CoreCostSolver(const AxesReorderSolverInput &input, double core_cost, bool nonmonotone = false)
      : AxesReorderSolver(input), core_cost_(core_cost), nonmonotone_(nonmonotone) {}
  double Objective() {
    return GetPerf();
  }
  int64_t Blocks() {
    int64_t blocks = 0;
    CalRealUsedCoreNum(blocks);
    return blocks;
  }

 protected:
  bool CalUsedCoreNum(double &value) override {
    value = 64.0 / input_.local_buffer_vars[0]->value / input_.pure_mc_vars[0]->value;
    return true;
  }
  bool CalRealUsedCoreNum(int64_t &value) override {
    double fractional = 0;
    CalUsedCoreNum(fractional);
    value = static_cast<int64_t>(std::ceil(fractional));
    return true;
  }
  bool SatisfyThresholdUBSize() override {
    return input_.local_buffer_vars[0]->value >= 1;
  }
  bool SatisfyUBSizeCacheLine(uint32_t) override {
    return true;
  }
  double GetPerf() override {
    if (nonmonotone_) {
      const auto blocks = Blocks();
      // Descriptor/model branches can separate a better basin from the initial local minimum.
      if (blocks >= 12) {
        return 100.0 + (blocks - 16) * (blocks - 16);
      }
      if (blocks >= 3) {
        return 200.0 + (blocks - 8) * (blocks - 8);
      }
      return 10.0 + (blocks - 1) * (blocks - 1);
    }
    return core_cost_ * Blocks() + input_.local_buffer_vars[0]->value * input_.pure_mc_vars[0]->value;
  }

 private:
  double core_cost_;
  bool nonmonotone_;
};

void InitThresholdVariables(TilingVariable &tile, TilingVariable &loop, Variable **parents, bool nonmonotone) {
  tile.value = 1;
  tile.mc_related = true;
  tile.prompt_align = 1;
  tile.upper_bound = [](Variable **) -> int64_t { return 64; };
  loop.upper_bound_vars = parents;
  loop.upper_bound = [](Variable **vars) -> int64_t { return (64 + vars[0]->value - 1) / vars[0]->value; };
  if (nonmonotone) {
    loop.upper_bound = [](Variable **) -> int64_t { return 1; };
  }
}

bool CheckThresholdObjectiveRefinement(bool enable_refinement, double core_cost, int64_t expected_blocks,
                                       bool nonmonotone = false, bool preserve_penalty_seed = false) {
  TilingVariable tile;
  TilingVariable loop;
  Variable *parents[] = {&tile};
  InitThresholdVariables(tile, loop, parents, nonmonotone);
  TilingVariable *variables[] = {&tile};
  TilingVariable *all_variables[] = {&loop, &tile};
  TilingVariable *multicore[] = {&loop};
  Constraint ub;
  ub.type = ConstraintType::LOCAL_BUFFER;
  ub.rel_tiling_vars = variables;
  ub.rel_hw_spec = 64;
  ub.eval = [](TilingVariable **vars, Variable **, int64_t limit) { return vars[0]->value - limit; };
  Constraint *constraints[] = {&ub};
  tile.rel_cons = constraints;
  tile.rel_cons_size = 1;
  uint32_t order[] = {0};
  AxesReorderSolverInput input;
  input.core_num = nonmonotone ? 32 : 16;
  input.ub_size = 64;
  input.ub_threshold = 0.01;
  input.corenum_threshold = nonmonotone ? 0.5 : 1.0;
  input.tiling_vars = all_variables;
  input.tiling_vars_size = 2;
  input.pure_mc_vars = multicore;
  input.pure_mc_vars_size = 1;
  input.local_buffer_vars = variables;
  input.local_buffer_vars_size = 1;
  input.ordered_local_buffer_vars = variables;
  input.ordered_to_canonical = order;
  input.all_cons = constraints;
  input.all_cons_size = 1;
  input.preserve_reduce_store_penalty_seed = preserve_penalty_seed;
  CoreCostSolver solver(input, core_cost, nonmonotone);
  if (!solver.Run(true, false, enable_refinement, false)) {
    return false;
  }
  // Original threshold tiling uses 16 blocks and tile 4. Refinement must preserve legality and lower cost.
  const bool valid = solver.Blocks() == expected_blocks && tile.value > 0 && tile.value <= 64 &&
                     solver.Objective() <= (nonmonotone ? 100.0 : core_cost * 16 + 4);
  if (!valid) {
    std::fprintf(stderr, "refine=%d cost=%g nonmonotone=%d blocks=%ld expected=%ld tile=%ld loop=%ld objective=%g\n",
                 enable_refinement, core_cost, nonmonotone, solver.Blocks(), expected_blocks, tile.value, loop.value,
                 solver.Objective());
  }
  return valid;
}

class BalanceSeedSolver : public NativeSolver<AxesReorderSolver> {
 public:
  explicit BalanceSeedSolver(const AxesReorderSolverInput &input) : NativeSolver(input) {}

 protected:
  bool CalUsedCoreNum(double &value) override {
    value = 512.0 * 16.0 / input_.local_buffer_vars[0]->value / input_.pure_mc_vars[0]->value;
    return true;
  }

  bool CalRealUsedCoreNum(int64_t &value) override {
    const auto tiles = std::ceil(16.0 / input_.local_buffer_vars[0]->value);
    value = static_cast<int64_t>(std::ceil(512.0 * tiles / input_.pure_mc_vars[0]->value));
    return true;
  }

  double GetPerf() override {
    const auto tile = input_.local_buffer_vars[0]->value;
    const auto loop = input_.pure_mc_vars[0]->value;
    if (tile == 8 && loop == 19) {
      return 10.0;
    }
    double cores = 0;
    CalUsedCoreNum(cores);
    return 50.0 + std::abs(cores - 37.0);
  }
};

bool CheckBalancedThresholdSeed(bool enable_refinement, int64_t expected_tile, bool enable_core_search = true,
                                bool equal_order = false, bool preserve_penalty_seed = false) {
  TilingVariable tile;
  tile.value = 1;
  tile.mc_related = true;
  tile.upper_bound = [](Variable **) -> int64_t { return 16; };
  TilingVariable loop;
  Variable *parents[] = {&tile};
  loop.upper_bound_vars = parents;
  loop.upper_bound_vars_size = 1;
  loop.upper_bound = [](Variable **vars) -> int64_t { return 512 * ((16 + vars[0]->value - 1) / vars[0]->value); };
  TilingVariable *local[] = {&tile};
  TilingVariable *all[] = {&loop, &tile};
  TilingVariable *multicore[] = {&loop};
  Constraint ub;
  ub.type = ConstraintType::LOCAL_BUFFER;
  ub.rel_tiling_vars = local;
  ub.rel_hw_spec = 16;
  ub.eval = [](TilingVariable **vars, Variable **, int64_t limit) { return vars[0]->value - limit; };
  Constraint *constraints[] = {&ub};
  tile.rel_cons = constraints;
  tile.rel_cons_size = 1;
  uint32_t order[] = {0};
  AxesReorderSolverInput input;
  input.core_num = 56;
  input.ub_size = 16;
  input.ub_threshold = 0.01;
  input.corenum_threshold = 1.0;
  input.tiling_vars = all;
  input.tiling_vars_size = 2;
  input.pure_mc_vars = multicore;
  input.pure_mc_vars_size = 1;
  input.local_buffer_vars = local;
  input.local_buffer_vars_size = 1;
  input.ordered_local_buffer_vars = local;
  input.ordered_to_canonical = order;
  input.all_cons = constraints;
  input.all_cons_size = 1;
  input.preserve_reduce_store_penalty_seed = preserve_penalty_seed;
  BalanceSeedSolver solver(input);
  if (!solver.Run(true, false, enable_refinement, equal_order, enable_core_search) || tile.value != expected_tile ||
      loop.value != 19) {
    std::fprintf(stderr, "balance seed: refine=%d tile=%lld expected=%lld loop=%lld\n", enable_refinement,
                 static_cast<long long>(tile.value), static_cast<long long>(expected_tile),
                 static_cast<long long>(loop.value));
    return false;
  }
  return true;
}

int main() {
  if (!CheckBalancedThresholdSeed(false, 15) || !CheckBalancedThresholdSeed(true, 8) ||
      !CheckBalancedThresholdSeed(true, 8, false) || !CheckBalancedThresholdSeed(false, 15, false) ||
      !CheckBalancedThresholdSeed(true, 15, false, true) || !CheckBalancedThresholdSeed(true, 8, true, false, true)) {
    return 5;
  }
  if (!CheckThresholdObjectiveRefinement(false, 8.0, 16) || !CheckThresholdObjectiveRefinement(true, 8.0, 3) ||
      !CheckThresholdObjectiveRefinement(true, 0.0, 16) || !CheckThresholdObjectiveRefinement(false, 8.0, 16, true) ||
      !CheckThresholdObjectiveRefinement(true, 8.0, 3, false, true) ||
      !CheckThresholdObjectiveRefinement(true, 8.0, 1, true) ||
      !CheckThresholdObjectiveRefinement(true, 8.0, 16, true, true)) {
    return 4;
  }
  TilingVariable variable;
  variable.value = 1;
  variable.upper_bound = [](Variable **) -> int64_t { return 16; };
  TilingVariable *variables[] = {&variable};
  Constraint constraint;
  constraint.type = ConstraintType::LOCAL_BUFFER;
  constraint.rel_tiling_vars = variables;
  constraint.rel_hw_spec = 8;
  constraint.eval = [](TilingVariable **vars, Variable **, int64_t limit) { return vars[0]->value - limit; };
  Constraint *constraints[] = {&constraint};
  variable.rel_cons = constraints;
  variable.rel_cons_size = 1;
  uint32_t order[] = {0};
  AxesReorderSolverInput input;
  input.core_num = 1;
  input.ub_size = 16;
  input.tiling_vars = variables;
  input.tiling_vars_size = 1;
  input.local_buffer_vars = variables;
  input.local_buffer_vars_size = 1;
  input.ordered_local_buffer_vars = variables;
  input.ordered_to_canonical = order;
  input.all_cons = constraints;
  input.all_cons_size = 1;
  NativeSolver<AxesReorderSolver> solver(input);
  if (!solver.Run(false, false, false, AF_AXES_EQUAL_ORDER) || variable.value != 8) {
    return 1;
  }
  NativeSolver<AxesReorderPgoSolver> pgo(input);
  const std::vector<std::vector<uint32_t>> expected{{1}, {2}, {4}, {8}};
  if (!pgo.PgoSolverGenerateAllTilingData() || pgo.GetTilingDataList() != expected) {
    return 2;
  }
  variable.upper_bound = [](Variable **) -> int64_t { return 0; };
  NativeSolver<AxesReorderSolver> empty(input);
  return empty.Run(true, true, true, AF_AXES_EQUAL_ORDER) && empty.IsEmptyTensor() ? 0 : 3;
}
