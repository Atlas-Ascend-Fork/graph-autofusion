/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_GATHER_TD
#define ASC_BASIC_OP_GATHER_TD

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

def AscendC_GatherbL0Op : VectorOp<"gatherb_l0", "Gatherb", [OpWithDstInterface]> {
    let description = "Gatherb elements from source tensor (L0 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0,
                         AscendC_LocalTensor:$offset, AnyType:$repeatTimes,
                         AscendC_GatherRepeatParams:$repeatParams);
}

def AscendC_GatherL0Op : VectorOp<"gather_l0", "Gather", [OpWithDstInterface]> {
    let description = "Gather elements from source tensor (L0 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                         AscendC_LocalTensor:$srcOffset, AnyType:$srcBaseAddr,
                         AnyType:$mask, AnyType:$repeatTimes,
                         AnyType:$dstRepStride);
}

def AscendC_GatherL1Op : VectorOp<"gather_l1", "Gather", [OpWithDstInterface]> {
    let description = "Gather elements from source tensor (L1 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                         AscendC_LocalTensor:$dstOffset, AnyType:$dstBase,
                         Variadic<AnyType>:$mask, AnyType:$repeatTimes,
                         AnyType:$srcRepStride);
}

def AscendC_GatherL2Op : VectorOp<"gather_l2", "Gather", [OpWithDstInterface]> {
    let description = "Gather elements from source tensor (L2 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                         AscendC_LocalTensor:$srcOffset, AnyType:$srcBaseAddr,
                         AnyType:$count);
}
#endif //ASC_BASIC_OP_GATHER_TD
