/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- RaceCheckPass.cpp - af-opt entry point for the race checker --------===//
//
// 提供 RaceChecker 的 af-opt 入口，便于检查单个 kernel。
// 核心 happens-before 和竞争判定由 test/ut/dialect/AFIRRaceCheckTest.cpp 覆盖；
// LIT 负责检查命令行诊断文本（设计文档 5.3，步 8b）。
//
//===----------------------------------------------------------------------===//

#include "dialect/AFIR/Analysis/RaceCheck.h"
#include "dialect/AFIR/Transforms/AFIRPasses.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::afir {
namespace {

struct AFIRRaceCheckPass : public PassWrapper<AFIRRaceCheckPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRRaceCheckPass)

  StringRef getArgument() const final {
    return "afir-race-check";
  }

  StringRef getDescription() const final {
    return "Report AFIR buffer accesses that may race: aliasing accesses with no happens-before between them";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AFIRDialect, func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    bool failed = false;
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      if (func.isExternal()) {
        continue;
      }
      RaceChecker checker;
      llvm::SmallVector<RaceFinding> findings = checker.check(func);
      failed |= mlir::failed(reportRaceFindings(findings));
    }
    if (failed) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRRaceCheckPass() {
  return std::make_unique<AFIRRaceCheckPass>();
}

void registerAFIRRaceCheckPass() {
  PassRegistration<AFIRRaceCheckPass>();
}

}  // namespace mlir::afir
