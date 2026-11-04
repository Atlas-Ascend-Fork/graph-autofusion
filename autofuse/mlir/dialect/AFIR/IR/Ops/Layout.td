// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Layout.td - AFIR layout transform operations ---------*- tablegen -*-===//
//
// 布局变换族（设计文档 2026-09-17 划的第 7 族）：同一批元素换一种片上摆法，逻辑形状与
// 元素类型都不变，变的只是 descriptor 的 `vectorized_strides`。`Pad` / `Transpose` /
// `Ub2ub` 后续落在本文件。
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_LAYOUT_OPS
#define AFIR_LAYOUT_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
include "Ops/Elementwise.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

// 元素类型取 legacy 两侧供给表的那一份，两侧逐字相同
// （`ascir_builtin_ops_v1.cpp:163` 与 `ascir_builtin_ops_v2.cpp:558` 共用同一个 `TensorType`）。
// **`ui32` 与 `bf16` 在 lowering 那一层的 `ElementTypes` 里没有位**，所以 implementation 收不到
// 它们；ODS 比 implementation 宽是对的——这里说的是 op 收什么，不是某个 implementation 发什么。
defvar AFIR_RemovePadElementTypes = [I16, UI16, I32, UI32, F16, F32, BF16];

// **它在 `AFIR_BufferizableUnaryElementwiseOp` 上，但它不是 elementwise。**
//
// 复用的是那一族的**形状契约**——一元、两端同 dtype、两端同逻辑形状——以及它的 verifier、
// 类型推导与 destination-style 那一套。这组契约与 elementwise 的逐字相同，不是近似：
// `AddRemovePadAfter`（`optimize/schedule_utils.cpp:947`）把生产者的输出属性整份拷给新节点，
// 所以两端的 `axis` / `repeats` 完全一致。
//
// 它与 elementwise 的差别全部落在 lowering，而那一层本来就是 per-implementation 的：
// 发的不是逐元素 API 而是一条 `GatherMask` 包装，读的是**两侧不同**的
// `vectorized_strides`（输入带填充、输出紧凑）——这个 op 存在的全部理由就是那个差。
def AFIR_RemovePadOp : AFIR_BufferizableUnaryElementwiseOp<"remove_pad", AFIR_RemovePadElementTypes> {
  let afirStability = AFIR_Experimental;
  let summary = "Compact a padded on-chip tile by dropping the tail-axis padding";
  let description = [{
    把片上 tile 里按 32B 对齐补出来的尾轴填充挤掉：逻辑形状、元素类型都不变，输入的行与行
    之间相距一个对齐后的行距，输出连续无间隙。

    它解决的是**非对齐小尾轴下 MTE3 带宽利用率低**的问题——尾轴补齐之后搬出要按行发，
    行短则每行一次；先在 UB 内挤掉填充，搬出就变成一次连续搬运。

    调度在两处插入它，**两处的条件不同**：主对齐策略在「尾向量化轴 stride ≠ 1」的 Load 之后插
    （`optimize/platform/common/base_alignment_strategy.cpp:302`，只有 mem 侧），非对齐模板在
    非 scalar 的 Broadcast 与非连续的 Load 之后插（`optimize/platform/v1/template/unaligned_template.cpp:69`）。

    两侧走同一个扩展包装 `RemovePadExtend`（`removepad.h`，矩阵 tier D1），所以只有一个
    implementation。本 Op 在 `compute` pipe 上执行。
  }];
}

#endif // AFIR_LAYOUT_OPS
