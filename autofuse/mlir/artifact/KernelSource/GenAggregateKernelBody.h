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
// Runs after runGenAggregateKernelEntry.  Clones each Plan's real "<symbol>_main"
// body into the matching body-less private declaration inside entryModule, then
// promotes entryModule's aggregate "main" to FunctionRole::KernelEntry and renames
// it to the legacy-convention name derived from fusedGraphName.  Fails without
// mutating entryModule further than diagnostics if the aggregate contract does not
// match planModules (missing plan module, missing "<symbol>_main" definition).
LogicalResult runGenAggregateKernelBody(ArrayRef<ModuleOp> planModules, ModuleOp entryModule, StringRef fusedGraphName);
}  // namespace mlir::ascend::kernel

#endif
