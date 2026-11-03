/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_HCCL_CONTEXT_TD
#define ASC_BASIC_OP_HCCL_CONTEXT_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_SetHcclContextOp : APIOp<"set_hccl_context", "SetHcclContext", [AscFunc]> {
  let arguments = (ins AnyType:$index, AnyType:$context);
  let paramTypeLists = [4, 0];
  let assemblyFormat = "$index `,` $context attr-dict `:` type($index) `,` type($context)";
}

def AscendC_GetHcclContextOp : APIOp<"get_hccl_context", "GetHcclContext", [AscFunc]> {
  let arguments = (ins AnyType:$index);
  let results = (outs AnyType:$context);
  let paramTypeLists = [4];
  let assemblyFormat = "$index attr-dict `:` type($index) `,` type($context)";
}

#endif // ASC_BASIC_OP_HCCL_CONTEXT_TD
