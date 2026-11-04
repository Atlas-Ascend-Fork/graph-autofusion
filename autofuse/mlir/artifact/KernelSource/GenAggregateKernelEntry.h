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

// The argument layout of the aggregate `main` produced by
// runGenAggregateKernelEntry: `arguments` mirrors the aggregate main's
// parameter list (excluding the trailing tiling struct parameter) in order,
// and `tiling_arg_index` is the position of that trailing tiling parameter.
struct AggregateArgumentLayout {
  SmallVector<KernelArgumentIdentity> arguments;
  size_t tiling_arg_index = 0;
};

// Validates that `entryModule` carries a well-formed aggregate contract for
// `planModules` (module kind, tiling schema, and a `plan_entries` list whose
// size matches `planModules`). Exposed independently of
// runGenAggregateKernelEntry so callers can surface contract violations
// before attempting the (more expensive) aggregation itself.
LogicalResult validateAggregateContract(ArrayRef<ModuleOp> planModules, ModuleOp entryModule);

// The aggregate operation consumes the Entry module produced by the aggregate
// schema operation and never traverses Plan bodies.
FailureOr<AggregateArgumentLayout> runGenAggregateKernelEntry(ArrayRef<ModuleOp> planModules, ModuleOp entryModule);
}  // namespace mlir::ascend::kernel

#endif
