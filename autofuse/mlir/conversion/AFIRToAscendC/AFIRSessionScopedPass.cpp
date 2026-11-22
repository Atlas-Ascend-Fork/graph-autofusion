/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AFIRSessionScopedPass.h"

#include <memory>
#include <utility>

#include "target/Ascend/TargetProfileRegistry.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::ascend::kernel {

FailureOr<std::shared_ptr<const ascend::TargetProfile>> resolveSessionTargetProfile(MLIRContext &context) {
  std::string diagnostic;
  llvm::raw_string_ostream diagnosticStream(diagnostic);
  // Resolve through the target registry to keep CLI option access in the target layer
  // and reuse one parsed profile across session-scoped passes.
  auto profile = ascend::getSelectedTargetProfile(ascend::TargetSelectionRequirement::Required, diagnosticStream);
  if (failed(profile)) {
    emitError(UnknownLoc::get(&context)) << diagnosticStream.str();
    return failure();
  }
  return *profile;
}

}  // namespace mlir::ascend::kernel
