/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_VEC_BILINEARINTERPOLATION_TD
#define ASC_BASIC_OP_VEC_BILINEARINTERPOLATION_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_BilinearInterpolationL0Op : VectorOp<"bilinearInterpolation_l0", "BilinearInterpolation", [OpWithDstInterface]> {
    let description = "AscendC::BilinearInterpolation is a vector binary operation (L0 API).\n";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0,
                         AscendC_LocalTensor:$src0Offset, AscendC_LocalTensor:$src1,
                         AnyType:$mask, AnyType:$hRepeat, AnyType:$repeatMode,
                         AnyType:$dstBlkStride, AnyType:$vROffset, AnyType:$vRepeat,
                         AscendC_LocalTensor:$sharedTmpBuffer);
}

def AscendC_BilinearInterpolationL1Op : VectorOp<"bilinearInterpolation_l1", "BilinearInterpolation", [OpWithDstInterface]> {
    let description = "AscendC::BilinearInterpolation is a vector binary operation (L1 API).\n";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0,
                         AscendC_LocalTensor:$src0Offset, AscendC_LocalTensor:$src1,
                         Variadic<UI64>:$mask, AnyType:$hRepeat, AnyType:$repeatMode,
                         AnyType:$dstBlkStride, AnyType:$vROffset, AnyType:$vRepeat,
                         AscendC_LocalTensor:$sharedTmpBuffer);
}
#endif //ASC_BASIC_OP_VEC_BILINEARINTERPOLATION_TD
