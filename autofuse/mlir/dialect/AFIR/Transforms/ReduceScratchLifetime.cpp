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
#include "dialect/AFIR/Transforms/AFIRPasses.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace mlir::afir {
namespace {
struct ReduceScratchLifetimePass : PassWrapper<ReduceScratchLifetimePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReduceScratchLifetimePass)
  StringRef getArgument() const final {
    return "afir-scope-reduce-scratch";
  }
  StringRef getDescription() const final {
    return "Infer Reduce scratch use scopes without changing physical allocation";
  }
  void runOnOperation() override {
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      if (hasFunctionRole(func, FunctionRole::KernelBody) && failed(scopeReduceScratchAllocations(func))) {
        signalPassFailure();
        return;
      }
    }
  }
};
}  // namespace
std::unique_ptr<Pass> createAFIRScopeReduceScratchPass() {
  return std::make_unique<ReduceScratchLifetimePass>();
}
void registerAFIRScopeReduceScratchPass() {
  registerPass([] { return createAFIRScopeReduceScratchPass(); });
}
}  // namespace mlir::afir
