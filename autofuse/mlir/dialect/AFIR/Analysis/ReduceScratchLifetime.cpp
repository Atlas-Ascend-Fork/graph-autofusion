/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Reduce scratch use scopes are separate from physical resource lifetimes.

#include "dialect/AFIR/Analysis/ReduceScratchLifetime.h"
#include "dialect/AFIR/Analysis/ReducePlan.h"
#include "dialect/AFIR/IR/ReduceScratch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
namespace mlir::afir {
namespace {
Block *parentBlock(Block *block) {
  auto *parent = block->getParentOp();
  return parent ? parent->getBlock() : nullptr;
}
Block *commonBlock(Block *lhs, Block *rhs) {
  for (Block *a = lhs; a; a = parentBlock(a)) {
    for (Block *b = rhs; b; b = parentBlock(b)) {
      if (a == b) {
        return a;
      }
    }
  }
  return nullptr;
}
FailureOr<int64_t> loopScope(Block *block, func::FuncOp func) {
  for (Operation *parent = block->getParentOp(); parent != func; parent = parent->getParentOp()) {
    if (auto loop = dyn_cast<scf::ForOp>(parent)) {
      auto binding = loop->getAttrOfType<AxisBindingAttr>(kAxisBindingAttrName);
      if (!binding || binding.getRole() != AxisValueRole::Loop) {
        loop.emitError() << kInvalidLifetimeDiagnostic << ": scratch scope requires a bound loop axis";
        return failure();
      }
      return binding.getAxis();
    }
  }
  return -1;
}
}  // namespace

LogicalResult verifyReduceScratchScope(Value scratch, Operation *consumer) {
  Operation *owner = scratch.getDefiningOp();
  auto scope = owner ? owner->getAttrOfType<IntegerAttr>(kReduceScratchScopeAttr) : IntegerAttr();
  if (!scope || scope.getInt() == -1) {
    return success();
  }
  for (Operation *parent = consumer->getParentOp(); parent; parent = parent->getParentOp()) {
    auto binding = parent->getAttrOfType<AxisBindingAttr>(kAxisBindingAttrName);
    if (isa<scf::ForOp>(parent) && binding && binding.getRole() == AxisValueRole::Loop &&
        binding.getAxis() == scope.getInt()) {
      return success();
    }
  }
  return consumer->emitError() << kInvalidLifetimeDiagnostic << ": scratch use escapes loop axis " << scope.getInt();
}

static LogicalResult collectActiveReduceScratch(func::FuncOp func, llvm::SetVector<Value> &active) {
  LogicalResult status = success();
  func.walk([&](Operation *op) {
    if (failed(status) || !op->hasAttr(kReduceScratchRolesAttr)) {
      return;
    }
    auto partition = getOperandPartition(op);
    if (failed(partition)) {
      status = failure();
      return;
    }
    auto scratch = op->getOperands().take_back(partition->tmpBufferCount);
    std::string reason;
    auto roles = getReduceScratchIndices(op, scratch.size(), &reason);
    if (failed(roles)) {
      status = op->emitError() << kInvalidLifetimeDiagnostic << ": " << reason;
      return;
    }
    auto accumulation = analyzeReduceAccumulation(op, &reason);
    if (failed(accumulation)) {
      status = op->emitError() << kInvalidLifetimeDiagnostic << ": " << reason;
      return;
    }
    for (int64_t axis : accumulation->accumulationAxes) {
      bool found = false;
      for (Operation *parent = op->getParentOp(); parent && parent != func; parent = parent->getParentOp()) {
        auto binding = parent->getAttrOfType<AxisBindingAttr>(kAxisBindingAttrName);
        found |=
            isa<scf::ForOp>(parent) && binding && binding.getRole() == AxisValueRole::Loop && binding.getAxis() == axis;
      }
      if (!found) {
        status = op->emitError() << kInvalidLifetimeDiagnostic << ": accumulation axis " << axis
                                 << " has no enclosing materialized loop";
        return;
      }
    }
    active.insert(scratch[roles->api]);
    if (accumulation->isAccumulating()) {
      if (!roles->roundResult) {
        status = op->emitError() << kInvalidLifetimeDiagnostic << ": accumulating reduce requires round_result";
        return;
      }
      auto api = getBufferDescAttr(scratch[roles->api]);
      auto round = getBufferDescAttr(scratch[*roles->roundResult]);
      if (api && round && api.getBufferId() == round.getBufferId()) {
        status = op->emitError() << kInvalidLifetimeDiagnostic << ": API scratch and round_result overlap";
        return;
      }
      active.insert(scratch[*roles->roundResult]);
    }
  });
  return status;
}

static LogicalResult placeReduceScratchAllocation(Value value, func::FuncOp func, DominanceInfo &dominance) {
  auto alloc = value.getDefiningOp<memref::AllocOp>();
  // Imported scratch arguments/aliases retain their declared scope. Only owned
  // allocations can be moved; no assumption about the lifetime of their contents.
  if (!alloc) {
    return success();
  }
  Block *block = nullptr;
  bool scratchOnly = true;
  for (OpOperand &use : value.getUses()) {
    auto partition = getOperandPartition(use.getOwner());
    if (!use.getOwner()->hasAttr(kReduceScratchRolesAttr) || failed(partition) ||
        use.getOperandNumber() < partition->inputCount + partition->destinationCount) {
      scratchOnly = false;
      break;
    }
    block = block ? commonBlock(block, use.getOwner()->getBlock()) : use.getOwner()->getBlock();
  }
  if (!scratchOnly || !block) {
    return success();
  }
  auto scope = loopScope(block, func);
  if (failed(scope)) {
    return failure();
  }
  Operation *anchor = nullptr;
  for (Operation *user : value.getUsers()) {
    Operation *site = block->findAncestorOpInBlock(*user);
    if (!anchor || site->isBeforeInBlock(anchor)) {
      anchor = site;
    }
  }
  for (Value operand : alloc->getOperands()) {
    if (!dominance.dominates(operand, anchor)) {
      return alloc.emitError() << kInvalidLifetimeDiagnostic << ": scratch size does not dominate its use scope";
    }
  }
  alloc->setAttr(kReduceScratchScopeAttr, IntegerAttr::get(IntegerType::get(func.getContext(), 64), *scope));
  alloc->moveBefore(anchor);
  return success();
}

LogicalResult scopeReduceScratchAllocations(func::FuncOp func) {
  llvm::SetVector<Value> active;
  LogicalResult status = collectActiveReduceScratch(func, active);
  if (failed(status)) {
    return failure();
  }
  // Explicit imported scopes are contracts, not hints to silently overwrite.
  func.walk([&](Operation *op) {
    if (failed(status) || !op->hasAttr(kReduceScratchScopeAttr)) {
      return;
    }
    if (op->getNumResults() != 1) {
      status = op->emitError() << kInvalidLifetimeDiagnostic << ": scratch scope requires one result";
      return;
    }
    for (Operation *user : op->getResult(0).getUsers()) {
      if (failed(verifyReduceScratchScope(op->getResult(0), user))) {
        status = failure();
        return;
      }
    }
  });
  if (failed(status)) {
    return failure();
  }
  DominanceInfo dominance(func);
  for (Value value : active) {
    if (failed(placeReduceScratchAllocation(value, func, dominance))) {
      return failure();
    }
  }
  return success();
}
}  // namespace mlir::afir
