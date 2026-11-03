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

// Replaces every `#include "<name>"` line of `kernelSource` whose name the
// provider resolves with the header text itself, leaving every other line
// untouched. This is how the extension headers reach the compiler: the kernel
// is handed to bisheng as a single translation unit with no include path of its
// own, so the text has to be part of it -- see ADR-E1 in the elementwise op
// completion plan for why this was chosen over shipping the headers separately.
//
// Callers that produce IR for inspection rather than for compilation (af-opt,
// LIT) simply do not run this, and keep the readable `#include "abs.h"` form.
std::string inlineApiExtensionHeaders(llvm::StringRef kernelSource, ApiExtensionProvider provider);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_EMIT_APIEXTENSIONINLINER_H
