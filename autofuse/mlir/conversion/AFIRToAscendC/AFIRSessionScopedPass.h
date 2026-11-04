/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_AFIRSESSIONSCOPEDPASS_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_AFIRSESSIONSCOPEDPASS_H

#include "conversion/AFIRToAscendC/AFIRToAscendCPipeline.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend::kernel {

// Resolves the target profile selected for this process through the target
// layer's registry (target/Ascend/TargetProfileRegistry.h), which owns the
// command-line dependency and parses each profile once. Kept here so every
// session-scoped pass shares one implementation and one diagnostic path.
FailureOr<std::shared_ptr<const ascend::TargetProfile>> resolveSessionTargetProfile(MLIRContext &context);

// Shared base for the AFIR-to-AscendC passes whose pipeline can only be
// built once an AFIRToAscendCPipelineSession is available, i.e. once the
// pass has an MLIRContext to build the ImplementationRegistry/
// SelectionPolicyRegistry with (createAFIRToAscendCPipelineSession requires
// MLIRContext&). That rules out the standard PassPipelineRegistration, whose
// builder callback only ever receives an OpPassManager&, not a context.
//
// initialize(MLIRContext*) is the correct upstream hook for this: it runs
// once per pass instance, after dependent dialects are loaded, and is
// documented as the place to resolve context-dependent state that
// runOnOperation() then relies on (Pass.h's initialize() doc comment
// explicitly warns getContext()/getOperation() must not be used here, so the
// context has to come from the parameter, exactly as needed here).
// runOnOperation() then drives the resolved session through a nested
// OpPassManager via Pass::runPipeline(), the same pattern
// CompositeFixedPointPass and SparsificationAndBufferizationPass use
// upstream for a dynamically populated sub-pipeline.
template <typename ConcretePass>
class AFIRSessionScopedPass : public PassWrapper<ConcretePass, OperationPass<ModuleOp>> {
 public:
  LogicalResult initialize(MLIRContext *context) final {
    auto targetProfile = resolveSessionTargetProfile(*context);
    if (failed(targetProfile)) {
      return failure();
    }
    auto session = createAFIRToAscendCPipelineSession(*context, *targetProfile);
    if (failed(session)) {
      return failure();
    }
    session_ = std::move(*session);

    OpPassManager pipeline(ModuleOp::getOperationName());
    if (failed(static_cast<ConcretePass *>(this)->populatePipeline(pipeline, session_))) {
      // `add*Pipeline` 收的是 OpPassManager，拿不到 MLIRContext，所以它只能返回
      // failure。诊断在这里发：这一层有 context，而一个不说原因的 pass 初始化失败在
      // af-opt 上只会打出一句「failed to initialize」。
      emitError(UnknownLoc::get(context))
          << "failed to build the " << static_cast<ConcretePass *>(this)->getArgument()
          << " pipeline: the resolved session is incomplete (target profile, frozen implementation registry)";
      return failure();
    }
    pipeline_ = std::move(pipeline);
    return success();
  }

  // mlir::PassManager::run() collects and loads every pass's dependent
  // dialects into the MLIRContext BEFORE calling initialize() (Pass.cpp's
  // PassManager::run: "Register all dialects for the current pipeline" runs,
  // then only afterward "Initialize all of the passes..."). pipeline_ is only
  // populated inside initialize(), so deferring to pipeline_.
  // getDependentDialects() here would see it still empty and silently fail to
  // load dialects the nested stage passes need. Upstream's own precedent for
  // "pipeline built dynamically at runtime" (SparsificationAndBufferizationPass)
  // keeps its dependent-dialect list static and independent of the
  // OpPassManager instances it builds inside runOnOperation() for exactly this
  // reason; ConcretePass::getStaticDependentDialects() mirrors that.
  void getDependentDialects(DialectRegistry &registry) const final {
    static_cast<const ConcretePass *>(this)->getStaticDependentDialects(registry);
  }

  void runOnOperation() override {
    if (failed(this->runPipeline(pipeline_, this->getOperation()))) {
      this->signalPassFailure();
    }
  }

 protected:
  AFIRToAscendCPipelineSession session_;
  OpPassManager pipeline_{ModuleOp::getOperationName()};
};

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_AFIRSESSIONSCOPEDPASS_H
