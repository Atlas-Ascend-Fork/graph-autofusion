/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_AFIR_REDUCE_SCRATCH_H
#define AUTOFUSE_MLIR_AFIR_REDUCE_SCRATCH_H

#include "dialect/AFIR/IR/AFIRAttributes.h"

#include <optional>
#include <string>
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::afir {
enum class ReduceScratchRole { ApiScratch, RoundResult };
inline llvm::StringRef stringifyReduceScratchRole(ReduceScratchRole role) {
  return role == ReduceScratchRole::ApiScratch ? "api_scratch" : "round_result";
}
struct ReduceScratchIndices {
  unsigned api;
  std::optional<unsigned> roundResult;
};
// Shared structural contract for the verifier and ReducePlan. No axis/layout analysis.
FailureOr<ReduceScratchIndices> getReduceScratchIndices(Operation *op, unsigned scratchCount,
                                                        std::string *diagnostic = nullptr);
}  // namespace mlir::afir
#endif
