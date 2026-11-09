/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_ARTIFACT_TILINGSOURCE_TILINGFUNCEMITTER_H
#define AUTOFUSE_MLIR_ARTIFACT_TILINGSOURCE_TILINGFUNCEMITTER_H

#include <string>

#include "bridge/AttSchemaAdapter/TilingCodeBundle.h"
#include "artifact/TilingSource/TilingProgramSpec.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::bridge {

FailureOr<std::string> assembleTilingSources(const TilingCodeBundle &bundle);

// Assembles ATT's split sources and emits the Inductor host tiling ABI from the
// same MLIR contract that produced TilingData.

FailureOr<std::string> emitInductorTilingFunc(const TilingProgramSpec &spec, const TilingCodeBundle &bundle);

}  // namespace mlir::bridge

#endif  // AUTOFUSE_MLIR_ARTIFACT_TILINGSOURCE_TILINGFUNCEMITTER_H
