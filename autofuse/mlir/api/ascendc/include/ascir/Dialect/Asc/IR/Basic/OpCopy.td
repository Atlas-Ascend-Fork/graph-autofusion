/*
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_COPY_TD
#define ASC_BASIC_OP_COPY_TD

include "Base.td"

//===----------------------------------------------------------------------===//
// Copy operations
//===----------------------------------------------------------------------===//

class CopyBaseOp<string mnemonic, list<Trait> traits = []>
    : APIOp<mnemonic, "Copy", traits> {
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AnyInteger:$repeatTime,
    AscendC_CopyRepeatParams:$repeatParams,
    AnyInteger:$isSetMask
  );
}

def AscendC_CopyL0Op
    : CopyBaseOp<"copy_l0"> {
  let summary = "Copy with bitwise mask";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    Variadic<AnyInteger>:$mask,
    AnyInteger:$repeatTime,
    AscendC_CopyRepeatParams:$repeatParams,
    AnyInteger:$isSetMask
  );
}

def AscendC_CopyL1Op : CopyBaseOp<"copy_l1"> {
  let summary = "Copy with continuous mask";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AnyInteger:$mask,
    AnyInteger:$repeatTime,
    AscendC_CopyRepeatParams:$repeatParams,
    AnyInteger:$isSetMask
  );
}
#endif // ASC_BASIC_OP_COPY_TD