/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_GENAGGREGATETILINGDATA_H
#define AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_GENAGGREGATETILINGDATA_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <memory>

namespace mlir {
class Pass;
}  // namespace mlir

namespace mlir::ascend::kernel {
// The aggregate operation is intentionally an orchestration API because its input
// consists of independent root modules rather than one bundle operation.
FailureOr<OwningOpRef<ModuleOp>> runGenAggregateTilingData(ArrayRef<ModuleOp> planModules);
}  // namespace mlir::ascend::kernel

#endif
