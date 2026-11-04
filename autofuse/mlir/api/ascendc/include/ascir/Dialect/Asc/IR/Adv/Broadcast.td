/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_ADV_BROADCAST_TD
#define ASC_ADV_BROADCAST_TD

include "Base.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/IR/OpBase.td"

def AscendC_BroadcastOp : VectorOp<"broadcast", "Broadcast", [
  AttrSizedOperandSegments, OpWithDstInterface, OpWithSrcInterface
]> {
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                   Variadic<I32>:$dstShape, Variadic<I32>:$srcShape);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

#endif // ASC_ADV_BROADCAST_TD
