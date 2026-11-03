/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_UTILS_REGISTRATION_H
#define ASCIR_DIALECT_UTILS_REGISTRATION_H

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"

namespace mlir {
namespace ascir {

inline void registerDialects(DialectRegistry& registry)
{
    registerAllDialects(registry);
    registry.insert<ascendc::AscendCDialect, emitasc::EmitAscDialect>();
    ascendc::registerExternalModels(registry);
    emitasc::registerExternalModels(registry);
}

inline void registerExtensions(DialectRegistry& registry) { registerAllExtensions(registry); }

inline void registerPasses()
{
    registerAllPasses();
    registerascendcPasses();
}

} // namespace ascir
} // namespace mlir

#endif // ASCIR_DIALECT_UTILS_REGISTRATION_H
