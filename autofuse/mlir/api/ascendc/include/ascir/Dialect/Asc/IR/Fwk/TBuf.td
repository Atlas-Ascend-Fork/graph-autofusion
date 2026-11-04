/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_FWK_TBUF_TD
#define ASC_FWK_TBUF_TD
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

//===----------------------------------------------------------------------===//
// Buffer operations (TBuf)
//===----------------------------------------------------------------------===//

def AscendC_TBufOp : AscendC_Op<"tbuf", [AscConstructor]> {
  let summary = "Instantiate buffer";
  let results = (outs AscendC_TBuf:$buffer);
  let assemblyFormat = "attr-dict `:` type($buffer)";
}

def AscendC_TBufGetTensorOp : APIOp<"tbuf.get_tensor", "Get"> {
  let summary = "Obtain tensor from initialized buffer";
  let arguments = (ins AscendC_TBuf:$buffer, Optional<AnyType>:$len);
  let results = (outs AscendC_LocalTensor:$tensor);
  let assemblyFormat = [{
    $buffer attr-dict (`,` $len^)? `:` qualified(type($buffer)) `,` qualified(type($tensor)) (`,` type($len)^)?
  }];
}

def AscendC_TBufGetWithOffsetOp : APIOp<"tbuf.get_with_offset", "GetWithOffset"> {
  let summary = "Obtain tensor from initialized buffer with offset";
  let arguments = (ins AscendC_TBuf:$buffer, AnyType:$size, AnyType:$bufOffset);
  let results = (outs AscendC_LocalTensor:$tensor);
  let assemblyFormat = [{
    $buffer `,` $size `,` $bufOffset attr-dict `:`
    qualified(type($buffer)) `,` qualified(type($size)) `,` qualified(type($bufOffset)) `,`
    qualified(type($tensor))
  }];
}

#endif
