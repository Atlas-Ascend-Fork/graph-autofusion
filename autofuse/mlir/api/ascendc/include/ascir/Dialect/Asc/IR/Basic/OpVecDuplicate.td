/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_DUPLICATE_TD
#define ASC_BASIC_OP_DUPLICATE_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_DuplicateL0Op : VectorOp<"duplicate_l0", "Duplicate", [OpWithDstInterface]> {
    let description = "Duplicate elements from source tensor (L0 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$scalar, AnyType:$mask,
                         AnyType:$repeatTimes, AnyType:$dstBlockStride,
                         AnyType:$dstRepeatStride, UnitAttr:$isSetMask);
}

def AscendC_DuplicateL1Op : VectorOp<"duplicate_l1", "Duplicate", [OpWithDstInterface]> {
    let description = "Duplicate elements from source tensor (L1 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$scalar, Variadic<UI64>:$mask,
                         AnyType:$repeatTimes, AnyType:$dstBlockStride,
                         AnyType:$dstRepeatStride, UnitAttr:$isSetMask);
}

def AscendC_DuplicateL2Op : VectorOp<"duplicate_l2", "Duplicate", [OpWithDstInterface]> {
    let description = "Duplicate elements from source tensor (L2 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$scalar,
                         AnyType:$calCount);
}
#endif //ASC_BASIC_OP_DUPLICATE_TD
