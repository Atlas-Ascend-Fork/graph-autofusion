/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_GENAGGREGATEKERNELBODY_H
#define AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_GENAGGREGATEKERNELBODY_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::ascend::kernel {
// After runGenAggregateKernelEntry, clone each plan's <symbol>_main body into its
// private entryModule declaration. Promote aggregate main to KernelEntry and rename
// it using the legacy convention derived from fusedGraphName.
// Validate the aggregate contract, plan modules and main definitions before mutation;
// on mismatch, return failure with diagnostics only.
LogicalResult runGenAggregateKernelBody(ArrayRef<ModuleOp> planModules, ModuleOp entryModule, StringRef fusedGraphName);
}  // namespace mlir::ascend::kernel

#endif
