/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_AFIRTOASCENDCPIPELINE_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_AFIRTOASCENDCPIPELINE_H

#include <memory>

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class MLIRContext;
class OpPassManager;
}  // namespace mlir

namespace mlir::ascend {
class TargetProfile;
namespace instruction {
class ImplementationRegistry;
class SelectionPolicyRegistry;
}  // namespace instruction
}  // namespace mlir::ascend

namespace mlir::ascend::kernel {

struct AFIRToAscendCPipelineSession {
  std::shared_ptr<const ascend::TargetProfile> targetProfile;
  std::shared_ptr<const ascend::instruction::ImplementationRegistry> implementationRegistry;
  std::shared_ptr<const ascend::instruction::SelectionPolicyRegistry> selectionPolicies;
};

FailureOr<AFIRToAscendCPipelineSession> createAFIRToAscendCPipelineSession(
    MLIRContext &context, std::shared_ptr<const ascend::TargetProfile> targetProfile);

LogicalResult addAFIRToAscendCPipeline(OpPassManager &passManager, const AFIRToAscendCPipelineSession &session);

// Convenience pipeline for a single plan candidate module: prepares
// the kernel body signature, runs the shared AFIR-to-AscendC lowering
// pipeline while the canonical TilingData argument is still present, then
// generates the per-candidate tiling schema and entry dispatcher. Exposed so
// both the codegen orchestration path and the af-opt tool can drive the
// exact same per-candidate lowering sequence.
LogicalResult addPlanCandidateLoweringPipeline(OpPassManager &passManager, const AFIRToAscendCPipelineSession &session);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_AFIRTOASCENDCPIPELINE_H
