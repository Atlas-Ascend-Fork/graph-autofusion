/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_GENAGGREGATEKERNELENTRY_H
#define AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_GENAGGREGATEKERNELENTRY_H

#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::ascend::kernel {

// Aggregate main argument layout. arguments lists input/output/workspace parameters
// in order, excluding the trailing tiling parameter; tiling_arg_index identifies that parameter.
struct AggregateArgumentLayout {
  SmallVector<KernelArgumentIdentity> arguments;
  size_t tiling_arg_index = 0;
};

// Validate entryModule's kind, tiling schema and plan_entries against planModules.
// The entry count must match the plan count. Exposed separately so callers can
// report contract errors before constructing the aggregate entry.
LogicalResult validateAggregateContract(ArrayRef<ModuleOp> planModules, ModuleOp entryModule);

// The aggregate operation consumes the Entry module produced by the aggregate
// schema operation and never traverses Plan bodies.
FailureOr<AggregateArgumentLayout> runGenAggregateKernelEntry(ArrayRef<ModuleOp> planModules, ModuleOp entryModule);
}  // namespace mlir::ascend::kernel

#endif
