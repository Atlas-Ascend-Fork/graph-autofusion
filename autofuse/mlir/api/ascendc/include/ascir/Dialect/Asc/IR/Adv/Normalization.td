/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_ADV_NORMALIZATION_TD
#define ASC_ADV_NORMALIZATION_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_RmsNormOp : VectorOp<"rms_norm", "RmsNorm", [
  OpWithDstInterface, OpWithSrcInterface
]> {
  let arguments = (ins UnitAttr:$basicBlock, AscendC_LocalTensor:$dst,
                   AscendC_LocalTensor:$src, AscendC_LocalTensor:$gamma,
                   AnyType:$epsilon, AscendC_RmsNormTiling:$tiling,
                   Optional<AscendC_LocalTensor>:$sharedTmpBuffer);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors()
    {
        SmallVector<Value> dstTensors{getDst()};
        if (auto tensor = getSharedTmpBuffer())
            dstTensors.emplace_back(tensor);
        return dstTensors;
    }
    SmallVector<Value> getSrcTensors() { return {getSrc(), getGamma()}; }
  }];
}

def AscendC_LayerNormOp : VectorOp<"layer_norm", "LayerNorm", [
  OpWithDstInterface, OpWithSrcInterface
]> {
  let arguments = (ins AscendC_LocalTensor:$dst,
                   AscendC_LocalTensor:$dstMean, AscendC_LocalTensor:$dstVarRstd,
                   AscendC_LocalTensor:$src, AscendC_LocalTensor:$gamma,
                   AscendC_LocalTensor:$beta, AnyType:$epsilon,
                   AscendC_LayerNormSeparateTiling:$separateTiling,
                   AscendC_LayerNormPara:$para,
                   AscendC_LocalTensor:$sharedTmpBuffer,
                   UnitAttr:$outputRstd);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getSrcTensors() { return {getSrc(), getGamma(), getBeta()}; }
    SmallVector<Value> getDstTensors() {
      return {getDst(), getDstMean(), getDstVarRstd(), getSharedTmpBuffer()};
    }
  }];
}
#endif //ASC_ADV_NORMALIZATION_TD
