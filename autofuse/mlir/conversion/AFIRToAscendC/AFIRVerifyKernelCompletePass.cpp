/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"

#include <memory>

#include "conversion/AFIRToAscendC/AFIRSessionScopedPass.h"
#include "dialect/AFIR/Transforms/VerifyKernelComplete.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir::ascend::kernel {
namespace {

// 设计文档 5.2：`af-opt --afir-verify-kernel-complete`，每条判据一组正负向 LIT，作为
// PR gate。
//
// **默认只跑 C3**，其余三条要显式点名。这不是把门槛放低，而是如实反映现状：C1 的所有权
// 一半、C2 的同步族 lowering、C4 的实现覆盖面都还没落地（5.2 的「当前」列分别是「部分」
// 「未满足」「覆盖面窄」）。把四条一起设成默认等于让这个 pass 在今天的每个生产 kernel
// 上都失败——那样它会立刻被加进忽略名单，然后再也没人看。
//
// 判据逐条可选，于是每条都能独立地从红转绿，而转绿的那一刻是可见的：把它加进默认集合
// 是一次显式的提交。
//
// `--target=` 是必需的：C4 要查实现注册表，注册表是按目标构建的。
struct AFIRVerifyKernelCompletePass : public AFIRSessionScopedPass<AFIRVerifyKernelCompletePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRVerifyKernelCompletePass)

  AFIRVerifyKernelCompletePass() = default;
  AFIRVerifyKernelCompletePass(const AFIRVerifyKernelCompletePass &other) : AFIRSessionScopedPass(other) {
    criteria = other.criteria;
  }

  StringRef getArgument() const final {
    return "afir-verify-kernel-complete";
  }

  StringRef getDescription() const final {
    return "Check the AFIR completeness criteria C1-C4 on every AFIR kernel function";
  }

  ListOption<std::string> criteria{
      *this, "criteria",
      llvm::cl::desc("Completeness criteria to check: c1 (buffer traceability), c2 (async completion), "
                     "c3 (AFIR-only representation), c4 (deterministic lowering). Defaults to c3."),
      llvm::cl::list_init<std::string>({std::string("c3")})};

  // 本 pass 自己不组子流水线：它要的只是 session 里的实现注册表。
  LogicalResult populatePipeline(OpPassManager &, const AFIRToAscendCPipelineSession &) {
    return success();
  }

  void getStaticDependentDialects(DialectRegistry &registry) const {
    registry.insert<afir::AFIRDialect, func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    llvm::SmallVector<afir::CompletenessCriterion> selected;
    for (const std::string &name : criteria) {
      std::optional<afir::CompletenessCriterion> criterion = parseCriterion(name);
      if (!criterion) {
        getOperation().emitError() << "unknown completeness criterion '" << name << "'; expected c1, c2, c3 or c4";
        return signalPassFailure();
      }
      selected.push_back(*criterion);
    }

    // C4 的查询在这里组装：注册表是目标层的东西，判据本身不认识它（VerifyKernelComplete.h
    // 的 ImplementationQuery）。目标名只进诊断。
    const instruction::ImplementationRegistry *registry = session_.implementationRegistry.get();
    auto contains = [registry](Operation *op) { return registry->containsSourceOp(op->getName()); };
    afir::ImplementationQuery implementations{"Ascend", {}};
    if (registry != nullptr && registry->isFrozen()) {
      implementations.contains = contains;
    }
    bool sawViolation = false;
    getOperation().walk([&](func::FuncOp func) {
      // 只检查 AFIR kernel 函数。模块里还有 host 侧的包装器与 tiling 函数，它们不在
      // 这些判据的适用范围内。
      if (!afir::getFunctionRoleAttr(func)) {
        return;
      }
      for (afir::CompletenessCriterion criterion : selected) {
        sawViolation |= failed(afir::verifyCompletenessCriterion(func, criterion, implementations));
      }
    });
    if (sawViolation) {
      signalPassFailure();
    }
  }

 private:
  static std::optional<afir::CompletenessCriterion> parseCriterion(llvm::StringRef name) {
    for (afir::CompletenessCriterion criterion :
         {afir::CompletenessCriterion::BufferTraceability, afir::CompletenessCriterion::AsyncCompletion,
          afir::CompletenessCriterion::AfirOnlyRepresentation, afir::CompletenessCriterion::DeterministicLowering}) {
      if (name == afir::getCompletenessCriterionName(criterion)) {
        return criterion;
      }
    }
    return std::nullopt;
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRVerifyKernelCompletePass() {
  return std::make_unique<AFIRVerifyKernelCompletePass>();
}

void registerAFIRVerifyKernelCompletePass() {
  PassRegistration<AFIRVerifyKernelCompletePass>();
}

}  // namespace mlir::ascend::kernel
