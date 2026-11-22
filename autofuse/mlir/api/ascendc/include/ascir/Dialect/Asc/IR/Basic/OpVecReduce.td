/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_VEC_REDUCE_TD
#define ASC_BASIC_OP_VEC_REDUCE_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Block Reduce operations
//===----------------------------------------------------------------------===//

def AscendC_BlockReduceSumL0Op : VectorOp<"block_reduce_sum_l0", "BlockReduceSum", [AscFunc]> {
    let description = "Sum all elements in each block (single mask version)";
    let arguments = (ins
      AscendC_LocalTensor:$dst,
      AscendC_LocalTensor:$src,
      AnyType:$repeatTime,
      AnyType:$mask,
      AnyType:$dstRepStride,
      AnyType:$srcBlkStride,
      AnyType:$srcRepStride
  );
  let paramTypeLists = [0, 0, 0, 0, 0, 0, 0];
}

def AscendC_BlockReduceSumL1Op : VectorOp<"block_reduce_sum_l1", "BlockReduceSum", [OpWithDstInterface]> {
    let description = "Sum all elements in each block (mask array version)";
    let arguments = (ins
      AscendC_LocalTensor:$dst,
      AscendC_LocalTensor:$src,
      AnyType:$repeatTime,
      Variadic<UI64>:$mask,
      AnyType:$dstRepStride,
      AnyType:$srcBlkStride,
      AnyType:$srcRepStride
  );
}

def AscendC_BlockReduceMaxL0Op : VectorOp<"block_reduce_max_l0", "BlockReduceMax", [AscFunc]> {
    let description = "Find maximum value in each block (single mask version)";
    let arguments = (ins
      AscendC_LocalTensor:$dst,
      AscendC_LocalTensor:$src,
      AnyType:$repeatTime,
      AnyType:$mask,
      AnyType:$dstRepStride,
      AnyType:$srcBlkStride,
      AnyType:$srcRepStride
  );
  let paramTypeLists = [0, 0, 0, 0, 0, 0, 0];
}

def AscendC_BlockReduceMaxL1Op : VectorOp<"block_reduce_max_l1", "BlockReduceMax", [OpWithDstInterface]> {
    let description = "Find maximum value in each block (mask array version)";
    let arguments = (ins
      AscendC_LocalTensor:$dst,
      AscendC_LocalTensor:$src,
      AnyType:$repeatTime,
      Variadic<UI64>:$mask,
      AnyType:$dstRepStride,
      AnyType:$srcBlkStride,
      AnyType:$srcRepStride
  );
}

def AscendC_BlockReduceMinL0Op : VectorOp<"block_reduce_min_l0", "BlockReduceMin", [AscFunc]> {
    let description = "Find minimum value in each block (single mask version)";
    let arguments = (ins
      AscendC_LocalTensor:$dst,
      AscendC_LocalTensor:$src,
      AnyType:$repeatTime,
      AnyType:$mask,
      AnyType:$dstRepStride,
      AnyType:$srcBlkStride,
      AnyType:$srcRepStride
  );
  let paramTypeLists = [0, 0, 0, 0, 0, 0, 0];
}

def AscendC_BlockReduceMinL1Op : VectorOp<"block_reduce_min_l1", "BlockReduceMin", [OpWithDstInterface]> {
    let description = "Find minimum value in each block (mask array version)";
    let arguments = (ins
      AscendC_LocalTensor:$dst,
      AscendC_LocalTensor:$src,
      AnyType:$repeatTime,
      Variadic<UI64>:$mask,
      AnyType:$dstRepStride,
      AnyType:$srcBlkStride,
      AnyType:$srcRepStride
  );
}

def AscendC_PairReduceSumL0Op : VectorOp<"pair_reduce_sum_l0", "PairReduceSum", [AscFunc]> {
    let description = "Pair reduce sum operation with continuous mask (L0 API)";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        AnyType:$repeatTime,
        AnyType:$mask,
        AnyType:$dstRepStride,
        AnyType:$srcBlkStride,
        AnyType:$srcRepStride,
        UnitAttr:$isSetMask
    );
    let paramTypeLists = [0, 0, 0, 0, 0, 0, 0];
}

def AscendC_PairReduceSumL1Op : VectorOp<"pair_reduce_sum_l1", "PairReduceSum", [OpWithDstInterface]> {
    let description = "Pair reduce sum operation with bit mask (L1 API)";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        AnyType:$repeatTime,
        Variadic<UI64>:$mask,
        AnyType:$dstRepStride,
        AnyType:$srcBlkStride,
        AnyType:$srcRepStride,
        UnitAttr:$isSetMask
    );
}

def AscendC_RepeatReduceSumL0Op : VectorOp<"repeat_reduce_sum_l0", "RepeatReduceSum", [AscFunc]> {
    let description = "Repeat reduce sum operation (L0 API)";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        AnyType:$repeatTime,
        AnyType:$mask,
        AnyType:$dstBlkStride,
        AnyType:$srcBlkStride,
        AnyType:$dstRepStride,
        AnyType:$srcRepStride,
        UnitAttr:$isSetMask
    );
    let paramTypeLists = [0, 0, 0, 0, 0, 0, 0, 0];
}

def AscendC_WholeReduceMaxL0Op : VectorOp<"whole_reduce_max_l0", "WholeReduceMax", [AscFunc]> {
    let description = "Whole reduce max operation with continuous mask (L0 API)";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        AnyType:$mask,
        AnyType:$repeatTime,
        AnyType:$dstRepStride,
        AnyType:$srcBlkStride,
        AnyType:$srcRepStride,
        AscendC_ReduceOrderAttr:$order,
        UnitAttr:$isSetMask
    );
    let paramTypeLists = [0, 0, 0, 0, 0, 0, 0, -1];
}

def AscendC_WholeReduceMaxL1Op : VectorOp<"whole_reduce_max_l1", "WholeReduceMax", [OpWithDstInterface]> {
    let description = "Whole reduce max operation with bit mask (L1 API)";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        Variadic<UI64>:$mask,
        AnyType:$repeatTime,
        AnyType:$dstRepStride,
        AnyType:$srcBlkStride,
        AnyType:$srcRepStride,
        AscendC_ReduceOrderAttr:$order,
        UnitAttr:$isSetMask
    );
}

def AscendC_WholeReduceMinL0Op : VectorOp<"whole_reduce_min_l0", "WholeReduceMin", [AscFunc]> {
    let description = "Whole reduce min operation with continuous mask (L0 API)";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        AnyType:$mask,
        AnyType:$repeatTime,
        AnyType:$dstRepStride,
        AnyType:$srcBlkStride,
        AnyType:$srcRepStride,
        AscendC_ReduceOrderAttr:$order,
        UnitAttr:$isSetMask
    );
    let paramTypeLists = [0, 0, 0, 0, 0, 0, 0, -1];
}

def AscendC_WholeReduceMinL1Op : VectorOp<"whole_reduce_min_l1", "WholeReduceMin", [OpWithDstInterface]> {
    let description = "Whole reduce min operation with bit mask (L1 API)";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        Variadic<UI64>:$mask,
        AnyType:$repeatTime,
        AnyType:$dstRepStride,
        AnyType:$srcBlkStride,
        AnyType:$srcRepStride,
        AscendC_ReduceOrderAttr:$order,
        UnitAttr:$isSetMask
    );
}

def AscendC_WholeReduceSumL0Op : VectorOp<"whole_reduce_sum_l0", "WholeReduceSum", [AscFunc]> {
    let description = "Whole reduce sum operation with continuous mask (L0 API)";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        AnyType:$mask,
        AnyType:$repeatTime,
        AnyType:$dstRepStride,
        AnyType:$srcBlkStride,
        AnyType:$srcRepStride,
        UnitAttr:$isSetMask
    );
    let paramTypeLists = [0, 0, 0, 0, 0, 0, 0];
}

def AscendC_WholeReduceSumL1Op : VectorOp<"whole_reduce_sum_l1", "WholeReduceSum", [OpWithDstInterface]> {
    let description = "Whole reduce sum operation with bit mask (L1 API)";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        Variadic<UI64>:$mask,
        AnyType:$repeatTime,
        AnyType:$dstRepStride,
        AnyType:$srcBlkStride,
        AnyType:$srcRepStride,
        UnitAttr:$isSetMask
    );
}

//===----------------------------------------------------------------------===//
// ReduceMax operations
//===----------------------------------------------------------------------===//

def AscendC_ReduceMaxL0Op : VectorOp<"reduce_max_l0", "ReduceMax", [
  AscFunc, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Reduce maximum elements (L0 API). ";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$sharedTmpBuffer,
    AnyType:$mask,
    AnyType:$repeatTime,
    AnyType:$srcRepStride,
    AnyType:$calIndex
  );
  let paramTypeLists = [0, 0, 0, 0, 0, 0, 0];
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors() { return {getDst(), getSharedTmpBuffer()}; }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def AscendC_ReduceMaxL1Op : VectorOp<"reduce_max_l1", "ReduceMax", [OpWithDstInterface]> {
  let description = "Reduce maximum elements (L1 API). ";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$sharedTmpBuffer,
    Variadic<AnyType>:$mask,
    AnyType:$repeatTime,
    AnyType:$srcRepStride,
    AnyType:$calIndex
  );
}

def AscendC_ReduceMaxL2Op : VectorOp<"reduce_max_l2", "ReduceMax", [
  AscFunc, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Reduce maximum elements (L2 API). ";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$sharedTmpBuffer,
    AnyType:$count,
    AnyType:$calIndex
  );
  let paramTypeLists = [0, 0, 0, 0, 0];
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors() { return {getDst(), getSharedTmpBuffer()}; }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

//===----------------------------------------------------------------------===//
// ReduceMin operations
//===----------------------------------------------------------------------===//

def AscendC_ReduceMinL0Op : VectorOp<"reduce_min_l0", "ReduceMin", [
  AscFunc, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Reduce minimum elements (L0 API). ";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$sharedTmpBuffer,
    AnyType:$mask,
    AnyType:$repeatTime,
    AnyType:$srcRepStride,
    AnyType:$calIndex
  );
  let paramTypeLists = [0, 0, 0, 0, 0, 0, 0];
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors() { return {getDst(), getSharedTmpBuffer()}; }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def AscendC_ReduceMinL1Op : VectorOp<"reduce_min_l1", "ReduceMin", [OpWithDstInterface]> {
  let description = "Reduce minimum elements (L1 API). ";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$sharedTmpBuffer,
    Variadic<AnyType>:$mask,
    AnyType:$repeatTime,
    AnyType:$srcRepStride,
    AnyType:$calIndex
  );
}

def AscendC_ReduceMinL2Op : VectorOp<"reduce_min_l2", "ReduceMin", [
  AscFunc, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Reduce minimum elements (L2 API). ";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$sharedTmpBuffer,
    AnyType:$count,
    AnyType:$calIndex
  );
  let paramTypeLists = [0, 0, 0, 0, 0];
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors() { return {getDst(), getSharedTmpBuffer()}; }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

//===----------------------------------------------------------------------===//
// ReduceSum operations
//===----------------------------------------------------------------------===//

def AscendC_ReduceSumL0Op : VectorOp<"reduce_sum_l0", "ReduceSum", [
  AscFunc, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Reduce sum elements (L0 API, int32_t mask). ";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$sharedTmpBuffer,
    AnyType:$mask,
    AnyType:$repeatTime,
    AnyType:$srcRepStride
  );
  let paramTypeLists = [0, 0, 0, 0, 0, 0];
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors() { return {getDst(), getSharedTmpBuffer()}; }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def AscendC_ReduceSumL1Op : VectorOp<"reduce_sum_l1", "ReduceSum", [OpWithDstInterface]> {
  let description = "Reduce sum elements (L1 API, uint64_t[] mask). ";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$sharedTmpBuffer,
    Variadic<AnyType>:$mask,
    AnyType:$repeatTime,
    AnyType:$srcRepStride
  );
}

def AscendC_ReduceSumL2Op : VectorOp<"reduce_sum_l2", "ReduceSum", [
  AscFunc, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Reduce sum elements (L2 API, int32_t count). ";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$sharedTmpBuffer,
    AnyType:$count
  );
  let paramTypeLists = [0, 0, 0, 0];
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors() { return {getDst(), getSharedTmpBuffer()}; }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

#endif // ASC_BASIC_OP_VEC_REDUCE_TD
