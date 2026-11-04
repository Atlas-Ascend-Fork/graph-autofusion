/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_ASC_TRANSFORMS_PASSES_H
#define ASCIR_DIALECT_ASC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ascendc {

#define GEN_PASS_DECL
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createDeclarePyStructPass();
std::unique_ptr<Pass> createDefineCubeOnlyPass();
std::unique_ptr<Pass> createDetectEnableDebugPass();
std::unique_ptr<Pass> createDetectKernelTypePass();
std::unique_ptr<Pass> createEraseSyncPass();
std::unique_ptr<Pass> createGenerateBoilerplatePass();
std::unique_ptr<Pass> createHoistQueBindPass();
std::unique_ptr<Pass> createHoistTensorAllocationPass(bool excludeInOut = false);
std::unique_ptr<Pass> createInputOutputTensorPass();
std::unique_ptr<Pass> createInsertQueSyncPass();
std::unique_ptr<Pass> createLegalizeKernelArgsPass(bool setFftsAddr = false);
std::unique_ptr<Pass> createMaterializeTensorPass(bool alwaysBuf = false);
std::unique_ptr<Pass> createNoopPass();
std::unique_ptr<Pass> createPrivatizeFuncPass();
std::unique_ptr<Pass> createUnifyPipePass();
std::unique_ptr<Pass> createVerifySyncPass();

} // namespace ascendc

#define GEN_PASS_REGISTRATION
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"

} // end namespace mlir

#endif // ASCIR_DIALECT_ASC_TRANSFORMS_PASSES_H
