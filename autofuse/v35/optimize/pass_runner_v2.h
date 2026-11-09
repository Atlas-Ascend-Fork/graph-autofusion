/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V2_PASS_RUNNER_V2_H
#define OPTIMIZE_PLATFORM_V2_PASS_RUNNER_V2_H

#include "optimize/platform/common/pass_runner.h"
#include "optimize/graph_pass/broadcast_backward_pass.h"
#include "optimize/graph_pass/redundant_broadcast_eliminate_pass.h"
#include "optimize/graph_pass/broadcast_const_to_store.h"
#include "optimize/graph_pass/scalar_to_1d_tensor.h"
#include "optimize/graph_pass/scalar_broadcast_optimization.h"
#include "optimize/graph_pass/expand_dims_for_all_reduce.h"
#include "optimize/graph_pass/pow_equiv_substitution_pass.h"
#include "optimize/graph_pass/masked_fill_input_reorder_pass.h"
#include "optimize/graph_pass/duplicate_elewise_cse_pass.h"
#include "v35/optimize/graph_pass/continues_broadcast_optimization.h"
#include "v35/optimize/graph_pass/same_source_broadcast_cse_pass.h"
#include "v35/optimize/graph_pass/gather_to_load.h"
#include "v35/optimize/graph_pass/split_concat_optimization_pass.h"
#include "v35/optimize/graph_pass/swap_cast_broadcast_pass.h"

namespace optimize {
class PassRunnerV2 final : public BasePassRunner {
 public:
  explicit PassRunnerV2() : BasePassRunner() {
    this->RegisterPass<PowEquivSubstitutionPass>();
    this->RegisterPass<BroadcastConstToStorePass>();
    this->RegisterPass<ScalarTo1DTensorPass>();
    this->RegisterPass<SameSourceBroadcastCsePass>();
    this->RegisterPass<DuplicateElewiseCsePass>();
    this->RegisterPass<RedundantBroadcastEliminatePass>();
    this->RegisterPass<BroadcastBackwardPass>();
    this->RegisterPass<SwapCastBroadcastPass>();
    this->RegisterPass<ScalarBroadcastOptimizationPass>();
    this->RegisterPass<MaskedFillInputReorderPass>();
    this->RegisterPass<ExpandDimsForAllReducePass>();
    this->RegisterPass<ContinuesBroadcastOptimizationPass>();
    this->RegisterPass<GatherToLoadPass>();
    this->RegisterPass<SplitConcatOptimizationPass>();
  }
};
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_V2_PASS_RUNNER_V2_H
