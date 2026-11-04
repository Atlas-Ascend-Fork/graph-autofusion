/*
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_ADV_SORT_TD
#define ASC_ADV_SORT_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Concat Op
//===----------------------------------------------------------------------===//

def AscendC_ConcatOp
    : APIOp<"concat", "Concat", [AscFunc]> {
  let description = "Preprocess source tensor data and merge it into the target tensor concat before sort operation.";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$tmp,
    AnyType:$repeatTime
  );
}

//===----------------------------------------------------------------------===//
// Extract Op
//===----------------------------------------------------------------------===//

def AscendC_ExtractOp
    : APIOp<"extract", "Extract", [AscFunc]> {
  let description = "Extract sorted values and indices from source tensor after sort operation.";
  let arguments = (ins
    AscendC_LocalTensor:$dstValue,
    AscendC_LocalTensor:$dstIndex,
    AscendC_LocalTensor:$src,
    AnyType:$repeatTime
  );
}

#endif // ASC_ADV_SORT_TD
