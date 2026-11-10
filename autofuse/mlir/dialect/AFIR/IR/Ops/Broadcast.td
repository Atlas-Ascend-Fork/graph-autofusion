// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Broadcast.td - AFIR broadcast operation -------------*- tablegen -*-===//

#ifndef AFIR_BROADCAST_OPS
#define AFIR_BROADCAST_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

// BF16 属于 Broadcast 的目标实现支持并集，因此 ODS 接受该类型。
// reg 分段 BroadcastExtend 的 SupportBytes 接受 2 字节类型，支持 bf16；
// mem 分段 broadcast.h 的类型分发不包含 bf16，由能力层拒绝，不能在 ODS 中提前拒绝所有目标。
//
// 两侧 stock Duplicate 头均列出 bfloat16_t，但当前 Duplicate implementation 的类型集合
// 还取决于 legacy 未限定 Duplicate 的绑定，需单独核对 kDuplicateElementTypes 后再开放。
// ODS 比 legacy 注册表更宽的情况由 ElementTypeMatrixTest 的 kOdsWiderThanLegacy 记录，
// 判定依据是实际头文件的能力，而不是仅比较 REG_ASC_IR 的声明。
defvar AFIR_BroadcastElementTypes = [I8, UI8, I16, UI16, I32, UI32, UI64, F16, F32, BF16];

// Keep the logical and buffer forms on the same ODS element-type constraint.
// TensorOrMemRef permits the two storage forms while preserving the canonical
// element-type list in TableGen.
def AFIR_BroadcastOp : AFIR_BufferizableUnaryOpBase<"broadcast",
    [AFIRBroadcastInterface, AFIRDestinationStyleInterface,
     DeclareOpInterfaceMethods<AFIRBroadcastInterface>,
     DeclareOpInterfaceMethods<AFIRDestinationStyleInterface>],
    1, AFIR_BufferizableShapedType<AFIR_BroadcastElementTypes>>,
    AFIR_OdsElementTypes<AFIR_BroadcastElementTypes> {
  let extraClassDeclaration = baseMethodDecl # odsElementTypesDecl;
}

#endif // AFIR_BROADCAST_OPS
