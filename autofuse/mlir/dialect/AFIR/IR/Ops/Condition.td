// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Condition.td - AFIR condition operations ------------*- tablegen -*-===//

#ifndef AFIR_CONDITION_OPS
#define AFIR_CONDITION_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
// 三元选择族用的是 Ops/Elementwise.td 里的 AFIR_BufferizableTernaryElementwiseOp。
include "Ops/Elementwise.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

// 选择条件为 ui8，两个候选值与结果同 dtype。mem 侧使用 Where，reg 侧使用 WhereExtend，
// 具体实现区分 scratch 需求和标量位置。
// select 与 where 的类型集合分开：mem 侧相同，但 reg 侧 Where 额外接受 ui8 / ui16。
// 不能因共享包装而同时放宽 Select。
// 两侧 legacy 可用实现均未声明 bf16，因此不在 ODS 中开放；ElementTypeMatrixTest
// 检查 ODS、注册实现与 legacy 的一致性。计算族的 bf16 支持不能直接推广到选择族。
defvar AFIR_SelectElementTypes = [I16, I32, I64, F16, F32];
defvar AFIR_WhereElementTypes = [UI8, UI16, I16, I32, I64, F16, F32, BF16];

// `scalarInputs = 1`：`then` / `else` 各自可以是标量（`where.h` 的场景 1/2/3），`mask` 不行
// ——三个标量重载的 `mask` 都是 `LocalTensor<uint8_t>`。
class AFIR_TernaryConditionOp<string mnemonic, list<Type> conditionTypes, list<Type> types> :
    AFIR_BufferizableTernaryElementwiseOp<mnemonic, conditionTypes, types, types,
                                          /*sameElementType=*/1, /*scalarInputs=*/1>;

def AFIR_SelectOp : AFIR_TernaryConditionOp<"select", [UI8], AFIR_SelectElementTypes>;
def AFIR_WhereOp : AFIR_TernaryConditionOp<"where", [UI8], AFIR_WhereElementTypes>;

// 这一类没有 `masked_fill`，是有意的。 `MaskedFillInputReorderPass`
// （`optimize/graph_pass/masked_fill_input_reorder_pass.cpp`）无条件把每个 `MaskedFill`
// 节点改写成 `Select`，而它挂在 `Optimizer::GraphPass` 上、跑在调度之前；bridge 消费的是
// 调度结果，所以这条路上永远看不到 `MaskedFill`。
//
// 曾经有过一个 `AFIR_MaskedFillOp` 加一份 mem 侧 implementation，两者都没有任何用例走得
// 到，已在 2026-09-23 删除。改写变成有条件的那天，要一起加回来的是 ODS、bridge 那一行和
// 两侧的 implementation——不是只补这一个 def。

#endif // AFIR_CONDITION_OPS
