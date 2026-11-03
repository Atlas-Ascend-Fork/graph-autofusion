/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_VEC_VCONV_TD
#define ASC_BASIC_OP_VEC_VCONV_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_CastL0Op : VectorOp<"cast_l0", "Cast", [
  OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Convert the input tensor to the specified data type (L0 API)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_RoundModeAttr:$roundMode, AnyType:$mask,
                       AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                       UnitAttr:$isSetMask);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def AscendC_CastL1Op : VectorOp<"cast_l1", "Cast", [OpWithDstInterface]> {
  let description = "Convert the input tensor to the specified data type (L1 API)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_RoundModeAttr:$roundMode, Variadic<UI64>:$mask,
                       AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                       UnitAttr:$isSetMask);
}

def AscendC_CastL2Op : VectorOp<"cast_l2", "Cast", [
  OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Convert the input tensor to the specified data type (L2 API)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_RoundModeAttr:$roundMode, AnyType:$calCount);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def AscendC_CastDeqL0Op : VectorOp<"cast_deq_l0", "CastDeq"> {
  let description = "Quantize and convert the input tensor to the specified data type (L0 API)";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$mask,
                       AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                       UnitAttr:$isSetMask, UnitAttr:$isVecDeq, UnitAttr:$halfBlock);
}

def AscendC_CastDeqL1Op : VectorOp<"cast_deq_l1", "CastDeq"> {
  let description = "Quantize and convert the input tensor to the specified data type (L1 API)";
  let arguments = (ins AnyType:$dst, AnyType:$src, Variadic<UI64>:$mask,
                       AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                       UnitAttr:$isSetMask, UnitAttr:$isVecDeq, UnitAttr:$halfBlock);
}

def AscendC_CastDeqL2Op : VectorOp<"cast_deq_l2", "CastDeq"> {
  let description = "Quantize and convert the input tensor to the specified data type (L2 API)";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$calCount,
                       UnitAttr:$isVecDeq, UnitAttr:$halfBlock);
}

def AscendC_SetDeqScaleOp : APIOp<"set_deq_scale", "SetDeqScale", [AttrSizedOperandSegments, AscFunc]> {
  let arguments = (ins AnyType:$scale, Optional<I16>:$offset, Optional<AnyType>:$signMode);
  let assemblyFormat = [{
    $scale attr-dict (`,` $offset^)? (`,` $signMode^)? `:`
    qualified(type($scale)) (`,` type($offset)^)? (`,` type($signMode)^)?
  }];
}

def AscendC_SetDeqScaleL4Op : VectorOp<"set_deq_scale_l4", "SetDeqScale"> {
    let description = "Set dequantization scale, offset and sign mode information";
    let arguments = (ins AscendC_LocalTensor:$vdeq, AscendC_VdeqInfo:$vdeqInfo
    );
}

defm AddReluCast : BinaryCastL012Op<"add_relu_cast", "AddReluCast">;
defm SubReluCast : BinaryCastL012Op<"sub_relu_cast", "SubReluCast">;

#endif // ASC_BASIC_OP_VEC_VCONV_TD
