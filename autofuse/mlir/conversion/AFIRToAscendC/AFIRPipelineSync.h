/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_AFIRPIPELINESYNC_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_AFIRPIPELINESYNC_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;
}  // namespace mlir

namespace mlir::ascend::kernel {

// The pass wrapper is declared in Passes.h. ConvertAFIRToAscendCPass also calls
// this function directly to satisfy a lowering precondition.
LogicalResult synchronizeAFIRPipeline(ModuleOp module);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_AFIRPIPELINESYNC_H
