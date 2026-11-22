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

int main() {
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
