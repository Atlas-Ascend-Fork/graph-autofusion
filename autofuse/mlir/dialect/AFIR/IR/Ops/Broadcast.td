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
defvar AFIR_BroadcastElementTypes = [I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32, BF16];

// Keep the logical and buffer forms on the same ODS element-type constraint.
// TensorOrMemRef permits the two storage forms while preserving the canonical
// element-type list in TableGen.
//
// 源这一位收裸标量：bridge 对 ASCIR `Scalar` / `IndexExpr` 节点「不建 op 只建值」，
// 交过来的就是裸标量，而 legacy 的 `BroadcastScalar` 本来就有一支不需要 tile
// （三支的对照见 `Implementations/Broadcast.cpp` 文件头）。写法沿用二元 elementwise 的
// `scalarOperand`：`Elementwise.td` 的 `AnyTypeOf<[shaped] # elementTypes>`。
// 目的地与结果两位不动，它们恒是一块 tile。
//
// 这里放开的只是「表达得出」。哪一种标量源真能生成，由能力层逐 implementation 判——于是
// 拒收落在能力层、整图干净回落 legacy，而不是在 verifier 上变成硬失败。
// logical（tensor）形态不受影响：`BroadcastOp::verify()` 的非 buffer 分支要求输入与结果
// 都是 `TensorType`。
def AFIR_BroadcastOp : AFIR_BufferizableUnaryOpBase<"broadcast",
    [AFIRBroadcastInterface, AFIRDestinationStyleInterface,
     DeclareOpInterfaceMethods<AFIRBroadcastInterface>,
     DeclareOpInterfaceMethods<AFIRDestinationStyleInterface>],
    1, AnyTypeOf<[AFIR_BufferizableShapedType<AFIR_BroadcastElementTypes>] # AFIR_BroadcastElementTypes>,
    AFIR_BufferizableShapedType<AFIR_BroadcastElementTypes>>,
    AFIR_OdsElementTypes<AFIR_BroadcastElementTypes> {
  let extraClassDeclaration = baseMethodDecl # odsElementTypesDecl;
}

#endif // AFIR_BROADCAST_OPS
