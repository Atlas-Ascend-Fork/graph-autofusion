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

// **BF16 在这里，而它比 legacy 的声明宽，是对的。** 2026-10-26 之前这张表上没有 BF16，
// 理由写的是「legacy 的 Broadcast 两列都不供 bf16」——那句话读的是 `REG_ASC_IR` 的声明，
// 不是 AscendC 头。按功能需求 3 的口径（**以头文件为准，不以 legacy 注册表为准**）逐档核
// 过这个 op 的四个 implementation，三档供得起 bf16：
//
//   - `broadcast.tensor.extend_reg_base`（reg 分段，`BroadcastExtend`）：**供**。
//     `v35/ascendc/api_regbase/broadcast_extend_impl.h:260` 的
//     `static_assert(SupportBytes<T, 1, 2, 4, 8>())` 只看字节宽，bf16 是 b16。
//   - `broadcast.tensor.duplicate` / `_reg_base`（常量源，`AscendC::Duplicate`）：**供**。
//     `dav_c220` 与 `dav_3510` 两份 `kernel_operator_vec_duplicate_impl.h` 的
//     `SupportType` 都列着 `bfloat16_t`。**但这一档本次不开位**——它收哪几种的判据是
//     「legacy 那句不带限定的 `Duplicate` 绑到谁」（见 `Implementations/Broadcast.cpp`
//     的 `kDuplicateElementTypes`），那条判据要单独核一遍，入册。
//   - `broadcast.tensor.segmented`（mem 分段，`broadcast.h` 的包装）：**不供**。
//     `ascendc/api/broadcast.h:511` 的分发链只收 i64/ui64、i8/ui8、
//     i16/ui16/half/float/i32/ui32 三支，bf16 落在 `else` 的
//     `ASSERT(false && "Broadcast type is not supported.")` 上。
//
// 所以这一位开在 ODS（= 两侧「供不供得起」的并集）与 reg 分段那一档的注册处，mem 分段那一
// 档不开——mem 侧的 bf16 广播于是在**能力层**答不支持、整图干净回落，而不是像开位之前那样
// 失败在 verifier 上。实测基线：探针 mem 侧 126 格硬失败里 120 格、reg 侧 118 格里 43 格
// 带的是这一条诊断（reg 另有 70 格 bf16 硬失败**一条诊断都没有**，开位前后都是 70 格，
// 是另一个缺口，见设计文档功能需求 6 的 F-1 那一节）。
//
// 这一位让 ODS 比 `ascir_op_lowering_matrix.csv` 的两列都宽，`ElementTypeMatrixTest` 的
// `kOdsWiderThanLegacy` 快照钉着它，理由与证据也记在那里。
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
