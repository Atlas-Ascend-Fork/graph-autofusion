/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_COMMON_TD
#define ASC_BASIC_OP_COMMON_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_InitSocStateOp : APIOp<"init_soc_state", "InitSocState", [AscFunc]> {
  let summary = "Initialize AI Core global state";
  let description = "Initialize global state registers like atomic accumulation state and mask mode";
  let assemblyFormat = "attr-dict";
}

def AscendC_SetStoreAtomicConfigOp : APIOp<"set_store_atomic_config", "SetStoreAtomicConfig", [AscFunc]> {
  let summary = "Set atomic operation enable bit and atomic operation type";
  let description = "Configure atomic operation enable bit and atomic operation type";
  let arguments = (ins AscendC_AtomicDtypeAttr:$atomicType, AscendC_AtomicOpAttr:$atomicOp);
  let assemblyFormat = "$atomicType `,` $atomicOp attr-dict";
  let paramTypeLists = [3, 3];
}

def AscendC_GetStoreAtomicConfigOp : APIOp<"get_store_atomic_config", "GetStoreAtomicConfig"> {
  let summary = "Get atomic operation enable bit and atomic operation type";
  let description = "Get the values of atomic operation enable bit and atomic operation type";
  let results = (outs AnyType:$atomicType, AnyType:$atomicOp);
  let assemblyFormat = "attr-dict `:` type($atomicType) `,` type($atomicOp)";
}

def AscendC_CheckLocalMemoryIAOp : APIOp<"check_local_memory_ia", "CheckLocalMemoryIA", [AscFunc]> {
  let description = "Get the values of atomic operation enable bit and atomic operation type";
  let arguments = (ins AscendC_CheckLocalMemoryIAParam:$param);
  let paramTypeLists = [0];
}

#endif // ASC_BASIC_OP_COMMON_TD
