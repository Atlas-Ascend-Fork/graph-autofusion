/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_EMIT_APIEXTENSIONINLINER_H
#define AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_EMIT_APIEXTENSIONINLINER_H

#include <string>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::ascend::kernel {

// Resolves the name of an Autofuse AscendC API extension header to the header
// text, or to an empty string for a name it does not provide. Header names are
// globally unique across the mem-based and reg-based families, so a provider is
// a flat name -> text table with no platform axis of its own.
using ApiExtensionProvider = llvm::function_ref<llvm::StringRef(llvm::StringRef)>;

// Replace resolved extension includes with their header text; leave all other lines unchanged.
// The kernel reaches bisheng as one translation unit without its own include path,
// so extension headers must be inlined (elementwise completion plan, ADR-E1).
// IR inspection paths such as af-opt and LIT skip this step and retain readable includes.
std::string inlineApiExtensionHeaders(llvm::StringRef kernelSource, ApiExtensionProvider provider);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_EMIT_APIEXTENSIONINLINER_H
