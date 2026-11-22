/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_GATHER_MASK_TD
#define ASC_BASIC_OP_GATHER_MASK_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// GatherMask/GetGatherMaskRemainCount operation definition
//===----------------------------------------------------------------------===//

def AscendC_GatherMaskOp : APIOp<"gather_mask", "GatherMask"> {
  let summary = "Call `AscendC::GatherMask` function";
  let description = "Gather elements from source tensor using mask pattern";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0,
                  AnyType:$src1Pattern,
                  AnyType:$reduceMode, AnyType:$mask,
                  AscendC_GatherMaskParams:$params,
                  AscendC_GatherMaskModeAttr:$gatherMaskMode);
  let results = (outs AnyType:$rsvdCnt);
  let assemblyFormat = [{
    $dst `,` $src0 `,` $src1Pattern `,` $reduceMode `,` $mask `,` $params `,` $gatherMaskMode
    attr-dict `:` qualified(type($dst)) `,` qualified(type($src0)) `,` qualified(type($src1Pattern)) `,`
    type($reduceMode) `,` type($mask) `,` type($params)  `->` type($rsvdCnt)
  }];
}

def AscendC_GetGatherMaskRemainCountOp : APIOp<"get_gather_mask_remain_count", "GetGatherMaskRemainCount",[AscFunc]> {
  let summary = "Call `AscendC::GetGatherMaskRemainCount` function";
  let description = "Get remaining count from last GatherMask operation";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}


#endif //ASC_BASIC_OP_GATHER_MASK_TD
