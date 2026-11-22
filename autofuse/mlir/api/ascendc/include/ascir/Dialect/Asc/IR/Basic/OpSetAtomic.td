/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_SET_ATOMIC_TD
#define ASC_BASIC_OP_SET_ATOMIC_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_SetAtomicTypeOp : APIOp<"set_atomic_type", "SetAtomicType", [AscFunc]> {
    let summary = "Set atomic operation data type by template type parameter";
    let arguments = (ins TypeAttr:$dtype);
    let paramTypeLists = [5];
}

def AscendC_SetAtomicAddOp : APIOp<"set_atomic_add", "SetAtomicAdd", [AscFunc]> {
    let summary = "Enable atomic addition for subsequent data transfers using the specified data type";
    let arguments = (ins TypeAttr:$dtype);
    let paramTypeLists = [5];
}

def AscendC_SetAtomicNoneOp : APIOp<"set_atomic_none", "SetAtomicNone", [AscFunc]> {
    let summary = "Disable all atomic operations for subsequent data transfers";
}

def AscendC_SetAtomicMaxOp : APIOp<"set_atomic_max", "SetAtomicMax", [AscFunc]> {
    let summary = "Enable atomic maximum comparison using the specified data type";
    let arguments = (ins TypeAttr:$dtype);
    let paramTypeLists = [5];
}

def AscendC_SetAtomicMinOp : APIOp<"set_atomic_min", "SetAtomicMin", [AscFunc]> {
    let summary = "Enable atomic minimum comparison using the specified data type";
    let arguments = (ins TypeAttr:$dtype);
    let paramTypeLists = [5];
}

#endif //ASC_BASIC_OP_SET_ATOMIC_TD
