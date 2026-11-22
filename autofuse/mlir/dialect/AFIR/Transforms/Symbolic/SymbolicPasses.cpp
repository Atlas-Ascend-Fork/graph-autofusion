/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/Transforms/AFIRPasses.h"

#include <memory>
#include <utility>

#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Transforms/Symbolic/SymbolicTransformDetail.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::afir {
namespace {

struct AFIRSimplifySymbolicPass : public PassWrapper<AFIRSimplifySymbolicPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRSimplifySymbolicPass)
  StringRef getArgument() const final {
    return "afir-simplify-symbolic";
  }
  StringRef getDescription() const final {
    return "Simplify AFIR symbolic evaluation expressions";
  }

  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](SymbolicEvalOp op) {
      Result<Expression> expression = Expression::Parse(op.getExpr().getValue().getValue());
      Result<Expression> simplified =
          expression ? expression.value().Simplify() : Result<Expression>(expression.error());
      if (!simplified) {
        op.emitError("failed to simplify symbolic expression: ") << simplified.error().message;
        return WalkResult::interrupt();
      }
      FailureOr<ExprAttr> attr = makeExprAttr(op.getContext(), simplified.value().Serialize());
      if (failed(attr)) {
        return WalkResult::interrupt();
      }
      op.setExprAttr(*attr);
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

struct AFIRVerifySymbolicSafetyPass : public PassWrapper<AFIRVerifySymbolicSafetyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRVerifySymbolicSafetyPass)
  StringRef getArgument() const final {
    return "afir-verify-symbolic-safety";
  }
  StringRef getDescription() const final {
    return "Verify AFIR symbolic integer domain and divisor safety";
  }

  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](SymbolicEvalOp op) {
      if (failed(detail::verifySymbolicSafety(op))) {
        return WalkResult::interrupt();
      }
      op.setSafetyVerifiedAttr(UnitAttr::get(op.getContext()));
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

struct AFIRLowerSymbolicPass : public PassWrapper<AFIRLowerSymbolicPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRLowerSymbolicPass)
  StringRef getArgument() const final {
    return "afir-lower-symbolic";
  }
  StringRef getDescription() const final {
    return "Lower verified AFIR symbolic evaluations to index operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, index::IndexDialect, math::MathDialect>();
  }

  void runOnOperation() override {
    bool missingSafety = false;
    getOperation().walk([&](SymbolicEvalOp op) {
      if (!op.getSafetyVerifiedAttr()) {
        op.emitOpError("must be processed by afir-verify-symbolic-safety before lowering");
        missingSafety = true;
      }
    });
    if (missingSafety) {
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(&getContext());
    detail::populateSymbolicLoweringPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct AFIRVerifyNoSymbolicEvalPass : public PassWrapper<AFIRVerifyNoSymbolicEvalPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRVerifyNoSymbolicEvalPass)
  StringRef getArgument() const final {
    return "afir-verify-no-symbolic-eval";
  }
  StringRef getDescription() const final {
    return "Reject symbolic evaluations remaining before backend translation";
  }

  void runOnOperation() override {
    bool found = false;
    getOperation().walk([&](SymbolicEvalOp op) {
      op.emitOpError("remains before backend translation");
      found = true;
    });
    if (found) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRSimplifySymbolicPass() {
  return std::make_unique<AFIRSimplifySymbolicPass>();
}

std::unique_ptr<Pass> createAFIRVerifySymbolicSafetyPass() {
  return std::make_unique<AFIRVerifySymbolicSafetyPass>();
}

std::unique_ptr<Pass> createAFIRLowerSymbolicPass() {
  return std::make_unique<AFIRLowerSymbolicPass>();
}

std::unique_ptr<Pass> createAFIRVerifyNoSymbolicEvalPass() {
  return std::make_unique<AFIRVerifyNoSymbolicEvalPass>();
}

void registerAFIRSymbolicPasses() {
  PassRegistration<AFIRSimplifySymbolicPass>();
  PassRegistration<AFIRVerifySymbolicSafetyPass>();
  PassRegistration<AFIRLowerSymbolicPass>();
  PassRegistration<AFIRVerifyNoSymbolicEvalPass>();
}

}  // namespace mlir::afir
