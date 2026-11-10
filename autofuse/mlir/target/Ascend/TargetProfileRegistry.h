/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- TargetProfileRegistry.h - Shared selected target profile -*- C++ -*-===//
//
// Resolve the selected target through this registry, keeping command-line option
// access in the target layer and parsing each selection's profile once.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_TARGET_ASCEND_TARGETPROFILEREGISTRY_H
#define AUTOFUSE_MLIR_TARGET_ASCEND_TARGETPROFILEREGISTRY_H

#include <memory>

#include "target/Ascend/AscendTargetProfile.h"
#include "target/Ascend/TargetSelection.h"

namespace llvm {
class raw_ostream;
}

namespace mlir::ascend {

// Return the selected target's profile, loading it once per distinct selection.
// TargetProfile is immutable and independent of MLIRContext, so contexts, passes
// and threads can share the same instance.
//
// `requirement` mirrors resolveTargetSelection: Required reports an error when
// no target was selected, Optional returns a null profile in that case.
FailureOr<std::shared_ptr<const TargetProfile>> getSelectedTargetProfile(TargetSelectionRequirement requirement,
                                                                         llvm::raw_ostream &diagnostic);

// Load (or return the cached) profile for an explicit selection, bypassing the
// command line. For embedders and tests that resolve the target themselves.
FailureOr<std::shared_ptr<const TargetProfile>> getTargetProfile(const TargetSelection &selection,
                                                                 llvm::raw_ostream &diagnostic);

}  // namespace mlir::ascend

#endif  // AUTOFUSE_MLIR_TARGET_ASCEND_TARGETPROFILEREGISTRY_H
