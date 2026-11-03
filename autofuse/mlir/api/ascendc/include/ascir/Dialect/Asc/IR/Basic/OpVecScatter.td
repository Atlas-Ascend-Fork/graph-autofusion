/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_SCATTER_TD
#define ASC_BASIC_OP_SCATTER_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Other vector binary operations
//===----------------------------------------------------------------------===//

def AscendC_ScatterL0Op : VectorOp<"scatter_l0", "Scatter", [AscFunc, OpWithDstInterface]> {
  let description = "Scatter elements from source tensor to destination tensor (L0 API). ";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$dstOffset,
    AnyType:$dstBase,
    AnyType:$mask,
    AnyType:$repeatTimes,
    AnyType:$srcRepStride
  );
}

def AscendC_ScatterL1Op : VectorOp<"scatter_l1", "Scatter", [OpWithDstInterface]> {
  let description = "Scatter elements from source tensor to destination tensor (L1 API). ";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$dstOffset,
    AnyType:$dstBase,
    Variadic<AnyType>:$mask,
    AnyType:$repeatTimes,
    AnyType:$srcRepStride
  );
}

def AscendC_ScatterL2Op : VectorOp<"scatter_l2", "Scatter", [AscFunc, OpWithDstInterface]> {
  let description = "Scatter elements from source tensor to destination tensor (L2 API). ";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$dstOffset,
    AnyType:$dstBase,
    AnyType:$count
  );
}
#endif //ASC_BASIC_OP_SCATTER_TD