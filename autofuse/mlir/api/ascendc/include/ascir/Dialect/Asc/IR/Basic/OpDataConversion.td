/*
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_DATA_CONVERSION_TD
#define ASC_BASIC_OP_DATA_CONVERSION_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_TransposeOp             : TransposeSimpleOp<"transpose", "Transpose">;
def AscendC_TransposeExtOp          : TransposeExtOp<"transpose_ext", "Transpose">;
def AscendC_TransDataTo5HDTensorListOp : TransDataTo5HDTensorListOp<"trans_data_to_5hd_tensor_list", "TransDataTo5HD">;
def AscendC_TransDataTo5HDUintListOp : TransDataTo5HDUintListOp<"trans_data_to_5hd_uint_list", "TransDataTo5HD">;
def AscendC_TransDataTo5HDOp        : TransDataTo5HDSingleOp<"trans_data_to_5hd", "TransDataTo5HD">;

def AscendC_TransDataTo5HDTensorOp: VectorOp<"trans_data_to_5hd_tensor","TransDataTo5HD", 
    [OpWithDstInterface, OpWithSrcInterface]> {
  let summary = "Call `AscendC::" # apiName # "` with offsets in src and dst tensor";
  let arguments = (ins AscendC_LocalTensor:$dst,
                       AscendC_LocalTensor:$src,
                       DenseI32ArrayAttr:$dstOffsets,
                       DenseI32ArrayAttr:$srcOffsets,
                       AscendC_TransDataTo5HDParams:$params);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

#endif // ASC_BASIC_OP_DATA_CONVERSION_TD