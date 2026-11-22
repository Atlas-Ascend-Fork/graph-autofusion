/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_time_pass.h"

namespace att {
namespace {
using Mode = ascir_param::CacheExecutionMode;
using Condition = ascir_param::ReduceCacheCondition;
using Execution = ascir_param::CacheExecutionParams;

bool MatchesLoop(const Execution &execution, const CacheGuardInfo &guard, int64_t id = -1) {
  size_t matches = 0U;
  for (const auto &loop : execution.loops) {
    if (loop.axis == guard.guard_axis && (id < 0 || loop.axis_id == id)) {
      ++matches;
      if (!loop.extent.IsValid() || loop.extent.Simplify() != guard.loop_extent.Simplify()) {
        return false;
      }
    }
  }
  return matches == 1U;
}

// Codegen publishes the condition it actually emitted on the call site; ATT must
// never rebuild it from scheduling axes.  Reconstructing the guard once produced
// an underestimate on a NDDMA small-input broadcast: the call ran 20 times but
// was costed as 1, the MTE2 estimate dropped by ~49k cycles, and the default
// tiling picked a ~93 us template over a ~45 us one.  Every uncertain path below therefore keeps the full base count:
// overestimating only makes a candidate look expensive, while underestimating
// silently promotes a slow template.  See docs/zh/autofuse/design/features/
// codegen.md section 3.2.1.
bool ReadOrdinaryGuard(const Execution &execution, CacheGuardInfo &guard) {
  if (execution.mode != Mode::kOriginBroadcast && execution.mode != Mode::kFusedBroadcast) {
    return false;
  }
  guard = execution.guard;
  const auto expected =
      execution.mode == Mode::kOriginBroadcast ? CacheGuardKind::kOriginBroadcast : CacheGuardKind::kFusedBroadcast;
  return guard.kind == expected && ValidateCacheGuardInfo(guard) &&
         MatchesLoop(execution, guard, execution.call_loop_axis_id);
}

// Lower bound on the number of iterations that skip the call, used when the
// guard is proven to be block dependent.  Every block restarts at i == 0 and can
// only add multiples of period for i > 0, so
//   hits <= 1 + Ceiling((loop_extent - 1) / period)
// holds for any phase, independent of block_inner_extent.  Feeding the smallest
// possible miss into the union complement below therefore keeps the estimate an
// upper bound on the real call count.
Expr MinSkippedCalls(const CacheGuardInfo &guard) {
  const Expr extra = af::sym::Div(af::sym::Sub(guard.loop_extent, CreateExpr(1U)), guard.period);
  return af::sym::Sub(guard.loop_extent, af::sym::Add(CreateExpr(1U), af::sym::Ceiling(extra))).Simplify();
}

// R predicates are (i < 1) || (i % period < 1), independent of block_dim, so the
// executed count is the i == 0 term plus one entry per period boundary.
Expr RGuardHits(const CacheGuardInfo &guard) {
  const Expr extra = af::sym::Floor(af::sym::Div(af::sym::Sub(guard.loop_extent, CreateExpr(1U)), guard.period));
  return af::sym::Add(CreateExpr(1U), extra).Simplify();
}

Expr CountReduceCalls(const Execution &execution, const Expr &base, const Expr &blocks) {
  if (execution.reduce_guards.empty()) {
    return base;
  }
  Expr loop_product = CreateExpr(1U);
  for (const auto &loop : execution.loops) {
    loop_product = loop_product * loop.extent;
  }
  if (loop_product.Simplify() != base.Simplify()) {
    return base;
  }
  std::map<Expr, Expr, ExprCmp> misses;
  Expr control_misses = CreateExpr(1U);
  for (const auto &[condition, guard] : execution.reduce_guards) {
    if (condition == Condition::kControl) {
      // The emitted control flag is false exactly when its loop size is one.
      int64_t value = 0;
      if (!guard.loop_extent.IsValid() ||
          (guard.loop_extent.IsConstExpr() && (!guard.loop_extent.GetConstValue(value) || value <= 0))) {
        return base;
      }
      control_misses = af::sym::Floor(af::sym::Div(CreateExpr(1U), guard.loop_extent));
      continue;
    }
    if (!MatchesLoop(execution, guard) || !ValidateCacheGuardInfo(guard) || misses.count(guard.guard_axis) != 0U) {
      return base;
    }
    if (condition == Condition::kR) {
      // R predicates never reference block_dim, so every block hits the same
      // count and the closed form is exact.
      misses.emplace(guard.guard_axis, guard.loop_extent - RGuardHits(guard));
      continue;
    }
    if (!IsBlockCountUniform(guard, blocks)) {
      // A depends on block_dim * block_inner_extent while the DoubleTile record
      // takes that extent and the period from two different axes, so the phase
      // differs per block and no exact closed form exists under a symbolic
      // block_dim.  Dropping the whole record (the pre-fix behaviour) also threw
      // away the already exact R and control terms; bounding the hits instead
      // keeps those and still stays an upper bound on the real call count.
      misses.emplace(guard.guard_axis, MinSkippedCalls(guard));
      continue;
    }
    misses.emplace(guard.guard_axis, guard.loop_extent - CountGuardHits(guard, CreateExpr(0U)));
  }
  Expr missed_calls = control_misses;
  for (const auto &loop : execution.loops) {
    const auto missed = misses.find(loop.axis);
    missed_calls = missed_calls * (missed == misses.end() ? loop.extent : missed->second);
  }
  // OR is the union of refresh points. Counting the complement avoids counting
  // the intersection of independent A/R refresh points twice.
  return (base - missed_calls).Simplify();
}

Expr CountCalls(const Execution &execution, const Expr &base, const Expr &blocks) {
  if (execution.mode == Mode::kReduceDoubleTile) {
    return CountReduceCalls(execution, base, blocks);
  }
  CacheGuardInfo guard;
  if (!ReadOrdinaryGuard(execution, guard) || !IsBlockCountUniform(guard, blocks)) {
    return base;
  }
  Expr others = CreateExpr(1U);
  for (const auto &loop : execution.loops) {
    if (loop.axis_id != execution.call_loop_axis_id) {
      others = others * loop.extent;
    }
  }
  if ((others * guard.loop_extent).Simplify() != base.Simplify()) {
    return base;
  }
  return (others * CountGuardHits(guard, CreateExpr(0U))).Simplify();
}
}  // namespace

Expr ExeTimePassManager::GetBlockDimExpr() const {
  if (tuning_space_ == nullptr || tuning_space_->block_dims.empty()) {
    return Expr();
  }
  Expr block_dim;
  bool has_block_dim = false;
  for (const auto &block_axes : tuning_space_->block_dims) {
    Expr group_dim = CreateExpr(1U);
    for (const auto *axis : block_axes) {
      if (axis == nullptr) {
        return Expr();
      }
      Expr axis_dim = axis->repeat;
      if (!axis_dim.IsValid()) {
        return Expr();
      }
      group_dim = af::sym::Mul(group_dim, axis_dim);
    }
    block_dim = has_block_dim ? af::sym::Max(block_dim, group_dim) : group_dim;
    has_block_dim = true;
  }
  return block_dim.Simplify();
}

TernaryOp ExeTimePassManager::UpdateNodeExeTime(const NodeInfo &node, const Expr &exe_time) const {
  if (node.cache_executions.empty()) {
    GELOGD("Node[%s] has no final Codegen cache record; preserve full execution count.", node.name.c_str());
    return TernaryOp(exe_time);
  }
  if (node.is_cv_ub_fusion) {
    return TernaryOp(exe_time);
  }
  Expr total = CreateExpr(0U);
  const auto blocks = GetBlockDimExpr();
  for (const auto &execution : node.cache_executions) {
    Expr base = CreateExpr(1U);
    for (const auto &loop : execution.loops) {
      if (!loop.extent.IsValid()) {
        return TernaryOp(exe_time);
      }
      base = base * loop.extent;
    }
    if (node.cache_executions.size() == 1U) {
      base = exe_time;
    }
    total = total + CountCalls(execution, base, blocks);
  }
  return TernaryOp(total.Simplify());
}
}  // namespace att
