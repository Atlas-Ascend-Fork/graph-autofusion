/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_REG_TD
#define ASC_BASIC_OP_REG_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Binary register API operations
//===----------------------------------------------------------------------===//

def AddRegOp : BinaryRegOp<"add", "Add">;
def AndRegOp : BinaryRegOp<"and", "And">;
def DivRegOp : BinaryRegOp<"div", "Div">;
def FusedAbsSubRegOp : BinaryRegOp<"fused_abs_sub", "FusedAbsSub">;
def FusedExpSubRegOp : BinaryRegOp<"fused_exp_sub", "FusedExpSub">;
def FusedMulDstAddRegOp : BinaryRegOp<"fused_mul_dst_add", "FusedMulDstAdd">;
def SubRegOp : BinaryRegOp<"sub", "Sub">;
def MaxRegOp : BinaryRegOp<"max", "Max">;
def MinRegOp : BinaryRegOp<"min", "Min">;
def MulRegOp : BinaryRegOp<"mul", "Mul">;
def MulAddDstRegOp : BinaryRegOp<"mul_add_dst", "MulAddDst">;
def OrRegOp : BinaryRegOp<"or", "Or">;
def PreluRegOp : BinaryRegOp<"prelu", "Prelu">;
def XorRegOp : BinaryRegOp<"xor", "Xor">;

//===----------------------------------------------------------------------===//
// Unary register API operations
//===----------------------------------------------------------------------===//

def AbsRegOp : UnaryRegOp<"abs", "Abs">;
def ExpRegOp : UnaryRegOp<"exp", "Exp">;
def LnRegOp : UnaryRegOp<"ln", "Ln">;
def LogRegOp : UnaryRegOp<"log", "Log">;
def Log10RegOp : UnaryRegOp<"log10", "Log10">;
def MaskNotRegOp : UnaryRegOp<"mask_not", "MaskNot">;
def NegRegOp : UnaryRegOp<"neg", "Neg">;
def NotRegOp : UnaryRegOp<"not", "Not">;
def ReluRegOp : UnaryRegOp<"relu", "Relu">;
def SqrtRegOp : UnaryRegOp<"sqrt", "Sqrt">;

//===----------------------------------------------------------------------===//
// VecScalar register API operations
//===----------------------------------------------------------------------===//

def AddsRegOp : VecScalarRegOp<"adds", "Adds">;
def LeakyReluRegOp : VecScalarRegOp<"leaky_relu", "LeakyRelu">;
def MulsRegOp : VecScalarRegOp<"muls", "Muls">;
def MaxsRegOp : VecScalarRegOp<"maxs", "Maxs">;
def MinsRegOp : VecScalarRegOp<"mins", "Mins">;
def ShiftLeftsRegOp : VecScalarRegOp<"shift_lefts", "ShiftLefts">;
def ShiftRightsRegOp : VecScalarRegOp<"shift_rights", "ShiftRights">;

//===----------------------------------------------------------------------===//
// Reduce register API operations
//===----------------------------------------------------------------------===//

def ReduceMaxRegOp : RegOp<"reduce_max_reg", "ReduceMax"> {
  let arguments = (ins AscendC_RegTensor:$dstReg, AscendC_RegTensor:$srcReg, AscendC_MaskReg:$maskReg);
}

def ReduceSumRegOp : RegOp<"reduce_sum_reg", "ReduceSum"> {
  let arguments = (ins AscendC_RegTensor:$dstReg, AscendC_RegTensor:$srcReg, AscendC_MaskReg:$maskReg);
}

def ReduceMinRegOp : RegOp<"reduce_min_reg", "ReduceMin"> {
  let arguments = (ins AscendC_RegTensor:$dstReg, AscendC_RegTensor:$srcReg, AscendC_MaskReg:$maskReg);
}

def DuplicateRegOp : RegOp<"duplicate_reg", "Duplicate"> {
  let arguments = (ins AscendC_RegTensor:$dstReg, AnyType:$srcReg, AscendC_MaskReg:$maskReg);
}

def DuplicateScalarRegOp : AscendC_Op<"duplicate", []> {
  let arguments = (ins AscendC_RegTensor:$dstReg, AnyType:$scalar);
  let assemblyFormat = "$dstReg `,` $scalar attr-dict `:` type($dstReg) `,` type($scalar)";
}

//===----------------------------------------------------------------------===//
// Other register API operations
//===----------------------------------------------------------------------===//

def SelectRegOp : RegOp<"select_reg", "Select"> {
  let arguments = (ins AscendC_RegTensor:$dstReg, AscendC_RegTensor:$src0Reg,
                       AscendC_RegTensor:$src1Reg, AscendC_MaskReg:$maskReg);
}

def DataCopyLoadOp : RegOp<"data_copy_vld_reg", "DataCopy"> {
  let description = "Perform copying between tensors (UB to RegTensor)";
  let arguments = (ins AscendC_RegTensor:$dstReg,
                       AnyRankedOrUnrankedMemRef:$src, AscendC_LoadDist:$dist);
}

def DataCopyStoreOp : RegOp<"data_copy_vst_reg", "DataCopy"> {
  let description = "Perform copying between tensors (RegTensor to UB)";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$dst,
                       AscendC_RegTensor:$srcReg,
                       AscendC_MaskReg:$maskReg, AscendC_StoreDist:$dist);
}

def UpdateMaskOp : RegOp<"update_mask", "UpdateMask"> {
  let summary = "Update reg mask";
  let comment = "The mask is updated on every iteration to match the total count";
  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "counter", [MemWrite, MemRead]>:$count, TypeAttr:$type);
  let results = (outs AscendC_MaskReg:$maskReg);
  let assemblyFormat = [{
    $type `,` $count attr-dict `:` type($count)
  }];
}

def RegTensorOp : RegOp<"reg_tensor", "RegTensor"> {
  let summary = "Create RegTensor";
  let results = (outs AscendC_RegTensor:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
  let hasCanonicalizeMethod = 1;
}

def LocalMemBarOp : RegOp<"local_mem_bar", "LocalMemBar"> {
  let arguments = (ins AscendC_MemType:$src, AscendC_MemType:$dst);
  let assemblyFormat = "$src `,` $dst attr-dict";
}

def MaskRegOp : RegOp<"mask_reg", "MaskReg"> {
  let summary = "Create RegTensor";
  let results = (outs AscendC_MaskReg:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def CreateMaskOp : RegOp<"create_mask", "CreateMask", [Pure]> {
  let summary = "Create RegTensor";
  let arguments = (ins TypeAttr:$dtype, AscendC_MaskPattern:$mask);
  let results = (outs AscendC_MaskReg:$result);
  let assemblyFormat = [{
    $dtype `,` $mask attr-dict `:` type($result)
  }];
}

#endif // ASC_BASIC_OP_REG_TD
