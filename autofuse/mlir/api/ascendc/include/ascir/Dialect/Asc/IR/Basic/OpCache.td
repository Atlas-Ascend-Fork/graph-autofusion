/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_CACHE_TD
#define ASC_BASIC_OP_CACHE_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Data Cache Clean And Invalid Ops
//===----------------------------------------------------------------------===//

def AscendC_DataCacheCleanAndInvalidGlobalOp
    : APIOp<"data_cache_clean_invalid_global", "DataCacheCleanAndInvalid", [AscFunc]> {
  let description = "Clean and invalidate data cache for GlobalTensor (with DcciDst).";
  let arguments = (ins
    AscendC_GlobalTensor:$dst,
    AscendC_CacheLineAttr:$entireType,
    AscendC_DcciDstAttr:$dcciDst
  );
  let paramTypeLists = [2, 3, 3];
}

def AscendC_DataCacheCleanAndInvalidGlobalNoDcciDstOp
    : APIOp<"data_cache_clean_invalid_global_nodccidst", "DataCacheCleanAndInvalid", [AscFunc]> {
  let description = "Clean and invalidate data cache for GlobalTensor.";
  let arguments = (ins
    AscendC_GlobalTensor:$dst,
    AscendC_CacheLineAttr:$entireType
  );
  let paramTypeLists = [2, 3];
}

def AscendC_GetICachePreloadStatusOp : APIOp<"get_icache_preload_status", "GetICachePreloadStatus", [AscFunc]> {
  let description = "Get the status of ICache preload: 0 indicates idle, 1 indicates busy.";
  let results = (outs I64:$status);
}

def AscendC_ICachePreLoadOp : APIOp<"icache_preload", "ICachePreLoad", [AscFunc]> {
  let description = "Preload instructions from DDR address to ICache.";
  let arguments = (ins
    I64:$preFetchLen
  );
  let paramTypeLists = [0];
}
#endif //ASC_BASIC_OP_CACHE_TD
