/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_ADV_ACTIVATION_TD
#define ASC_ADV_ACTIVATION_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_SimpleSoftMaxOp : VectorOp<"simple_softmax", "SimpleSoftMax", [
  AttrSizedOperandSegments, OpWithDstInterface, OpWithSrcInterface
]> {
  let arguments = (ins UnitAttr:$reuseSource, UnitAttr:$basicBlock,
                   UnitAttr:$dataFormatNZ, AscendC_LocalTensor:$dst,
                   AscendC_LocalTensor:$sumTensor,
                   AscendC_LocalTensor:$maxTensor,
                   AscendC_LocalTensor:$src,
                   Optional<AscendC_LocalTensor>:$sharedTmpBuffer,
                   AscendC_SoftMaxTiling:$tiling,
                   Optional<AnyType>:$softmaxShapeInfo);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors()
    {
        SmallVector<Value> dstTensors{getDst()};
        if (auto tensor = getSharedTmpBuffer())
            dstTensors.emplace_back(tensor);
        return dstTensors;
    }
    SmallVector<Value> getSrcTensors()
    {
        return {getSumTensor(), getMaxTensor(), getSrc()};
    }
  }];
}

def AscendC_SoftMaxOp : VectorOp<"softmax", "SoftMax", [
  AttrSizedOperandSegments, OpWithDstInterface, OpWithSrcInterface
]> {
  let arguments = (ins UnitAttr:$reuseSource, UnitAttr:$basicBlock,
                   UnitAttr:$dataFormatNZ, AscendC_LocalTensor:$dst,
                   Optional<AscendC_LocalTensor>:$sumTensor,
                   Optional<AscendC_LocalTensor>:$maxTensor,
                   AscendC_LocalTensor:$src,
                   Optional<AscendC_LocalTensor>:$sharedTmpBuffer,
                   AscendC_SoftMaxTiling:$tiling,
                   Optional<AnyType>:$softmaxShapeInfo);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors()
    {
        SmallVector<Value> dstTensors{getDst()};
        if (auto tensor = getSumTensor())
            dstTensors.emplace_back(tensor);
        if (auto tensor = getMaxTensor())
            dstTensors.emplace_back(tensor);
        if (auto tensor = getSharedTmpBuffer())
            dstTensors.emplace_back(tensor);
        return dstTensors;
    }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def AscendC_SoftmaxFlashV2Op
    : APIOp<"softmax_flash_v2", "SoftmaxFlashV2", [AttrSizedOperandSegments, OpWithDstInterface]> {
  let arguments = (ins UnitAttr:$isUpdate, UnitAttr:$reuseSource,
                   UnitAttr:$basicBlock, UnitAttr:$dataFormatNZ,
                   AscendC_LocalTensor:$dst,
                   Optional<AscendC_LocalTensor>:$outReduceMax,
                   AscendC_LocalTensor:$expSumTensor,
                   AscendC_LocalTensor:$maxTensor,
                   AscendC_LocalTensor:$src,
                   AscendC_LocalTensor:$expMaxTensor,
                   AscendC_LocalTensor:$inExpSumTensor,
                   AscendC_LocalTensor:$inMaxTensor,
                   Optional<AscendC_LocalTensor>:$sharedTmpBuffer,
                   AscendC_SoftMaxTiling:$tiling,
                   Optional<AscendC_SoftMaxShapeInfo>:$softmaxShapeInfo,
                   Optional<AscendC_SoftmaxConfig>:$softmaxConfig);
}

def AscendC_SwiGLUOp : APIOp<"swiglu", "SwiGLU", [AttrSizedOperandSegments, OpWithDstInterface]> {
  let arguments = (ins AscendC_LocalTensor:$dst,
                   AscendC_LocalTensor:$srcTensor0,
                   AscendC_LocalTensor:$srcTensor1,
                   AnyType:$scalarValue,
                   Optional<AscendC_LocalTensor>:$sharedTmpBuffer,
                   Optional<AnyType>:$calCount);
}
#endif // ASC_ADV_ACTIVATION_TD
