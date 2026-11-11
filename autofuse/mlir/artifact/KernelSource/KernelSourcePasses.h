/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_PASSES_H
#define AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}  // namespace mlir

namespace mlir::ascend::kernel {

// Aggregates the per-candidate plan modules nested in the pass root into one
// entry module. Only the IR half of the aggregate stage is a pass: the text
// artifacts that follow it are produced by a translation, behind
// runAggregateLaunchDriver (AggregateLaunchDriver.h).
std::unique_ptr<Pass> createAFIRAggregatePass();
void registerAFIRAggregatePass();

// Registers this library's passes separately from registerAFIRToAscendCPasses.
// This library consumes lowered modules; registering it from conversion would
// reverse the dependency and make conversion link against its consumer.
void registerKernelSourcePasses();

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_PASSES_H
