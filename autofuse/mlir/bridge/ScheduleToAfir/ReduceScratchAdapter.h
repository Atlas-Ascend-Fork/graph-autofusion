/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_REDUCE_SCRATCH_ADAPTER_H
#define AUTOFUSE_MLIR_REDUCE_SCRATCH_ADAPTER_H
#include "bridge/AfirBuilder/AfirBuilder.h"

namespace mlir::afir::bridge {
// CalcReduceTmpSize uses -1/0 as role tags, not actual schedule axis ids.
// Both roles are function-owned TBuf storage; their contents are consumed within
// a reduce invocation/round. The destination, not this scratch, holds the accumulator.
inline std::optional<TmpBufferDTO> normalizeLegacyReduceScratch(int64_t id, int64_t legacyLifetime) {
  if (id < 0) {
    return std::nullopt;
  }
  if (legacyLifetime == -1) {
    return TmpBufferDTO{id, -1, ReduceScratchRole::ApiScratch};
  }
  if (legacyLifetime == 0) {
    return TmpBufferDTO{id, -1, ReduceScratchRole::RoundResult};
  }
  return std::nullopt;
}
// All bridge consumers, including shared-buffer lifetime collection, normalize
// these role tags before writing storage metadata.
inline std::optional<TmpBufferDTO> translateSchedulerTmpBuffer(llvm::StringRef nodeType, int64_t id,
                                                               int64_t legacyLifetime) {
  if (id < 0 || legacyLifetime < -1) {
    return std::nullopt;
  }
  if (isReduceNodeType(nodeType)) {
    return normalizeLegacyReduceScratch(id, legacyLifetime);
  }
  return TmpBufferDTO{id, legacyLifetime};
}
}  // namespace mlir::afir::bridge
#endif
