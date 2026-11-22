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
#include "target/Ascend/Capability/ImplementationSelection.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::ascend::kernel {
namespace {

struct AFIRSelectImplementationPass : public AFIRSessionScopedPass<AFIRSelectImplementationPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRSelectImplementationPass)

  StringRef getArgument() const final {
    return "afir-select-implementation";
  }

  StringRef getDescription() const final {
    return "Select Ascend implementations for AFIR operations";
  }

  LogicalResult populatePipeline(OpPassManager &pipeline, const AFIRToAscendCPipelineSession &session) {
    pipeline.addPass(ascend::instruction::createSelectImplementationPass(
        session.implementationRegistry, session.targetProfile, session.selectionPolicies, nullptr));
    return success();
  }

  // The wrapped ascend::instruction::SelectImplementationPass declares no
  // dependent dialects of its own.
  void getStaticDependentDialects(DialectRegistry &registry) const {}
};

}  // namespace

std::unique_ptr<Pass> createAFIRSelectImplementationStandalonePass() {
  return std::make_unique<AFIRSelectImplementationPass>();
}

void registerAFIRSelectImplementationStandalonePass() {
  PassRegistration<AFIRSelectImplementationPass>();
}

}  // namespace mlir::ascend::kernel
