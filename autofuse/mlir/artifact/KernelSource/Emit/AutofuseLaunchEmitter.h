/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_EMIT_AUTOFUSELAUNCHEMITTER_H
#define AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_EMIT_AUTOFUSELAUNCHEMITTER_H

#include "artifact/KernelSource/GenAggregateKernelEntry.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::ascend::kernel {

// Emits an `extern "C" int64_t AutofuseLaunch(...)` host wrapper that invokes
// `graphName<<<blockDim, nullptr, stream>>>(...)` with the aggregate main's
// input/output/workspace/tiling arguments in the ABI order recorded by
// `layout`. Byte-for-byte compatible with the existing
// `Kernel::GenKernelFuncCallForInductor` (non-CV) launch text in
// codegen_kernel.cpp. Returns failure() when `layout` does not match exactly
// one workspace argument, which should already have been rejected by
// validateAggregateContract.
FailureOr<std::string> emitAutofuseLaunch(const AggregateArgumentLayout &layout, ArrayRef<Type> argumentTypes,
                                          llvm::StringRef graphName, llvm::StringRef tilingTypeName);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_EMIT_AUTOFUSELAUNCHEMITTER_H
