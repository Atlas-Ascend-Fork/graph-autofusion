/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OP_ACTIVATION_TD
#define ASC_AUTOFUSE_OP_ACTIVATION_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "Autofuse/Base.td"
include "Autofuse/Interfaces.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// `AscendC::Gelu(dst, src, tmp, count)`。与 `Sigmoid` 同形的 stock 调用——要一块调用方
// 给的 tmp buffer，但不引任何扩展头。
//
// **两张支持表给的是同一个调用**（legacy 两侧的 impl 类都是 `UnaryApiTmpCall` +
// apiName `Gelu`，只在 `IncludeApiHeaderFiles` 里带一份 CANN 自己的
// `adv_api/activation/gelu.h`），所以只有这一个 def，没有 `_reg_base` 的那一半。
// 两侧 `CalcTmpBufSize` 给的尺寸不同（`GetInputDataSizeTmpBuffer` vs
// `CalcVoidTmpSizeV2` 的 0），但 AFIR 只带 id 不带尺寸，这个差别落在 ATT 那一侧。
def AscendC_GeluOp : UnaryOp<"gelu", "Gelu", [AttrSizedOperandSegments]> {
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                   Optional<AscendC_LocalTensor>:$sharedTmpBuffer, Optional<AnyType>:$calCount);
}

// `AscendC::Sigmoid(dst, src, tmp, count)`. Unlike most stock vector calls it
// wants a caller-provided tmp buffer, which is why the AFIR side keeps "needs a tmp
// buffer" separate from "uses the Autofuse wrapper" -- here both are true on one
// API family and only the second is false on the other.
def AscendC_SigmoidOp : UnaryOp<"sigmoid", "Sigmoid", [AttrSizedOperandSegments]> {
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                   Optional<AscendC_LocalTensor>:$sharedTmpBuffer, Optional<AnyType>:$calCount);
}

// `SigmoidExtend(dst, src, tmp_buf, cal_cnt)` from autofuse/ascendc/api/sigmoid.h
// -- an Autofuse wrapper in the global namespace, not `AscendC::SigmoidExtend`.
// The operand order is the header's, and the tmp buffer is mandatory: the wrapper
// duplicates a ones-tensor into it before dividing.
def AscendC_SigmoidExtendOp : UnaryOp<"sigmoid_extend", "SigmoidExtend", [ApiExtensionOpInterface]> {
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                   AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["sigmoid.h"], ["utils.h"]>.decl # SrcTensorsFromSrc<>.decl;
}

#endif // ASC_AUTOFUSE_OP_ACTIVATION_TD
