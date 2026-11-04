/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OP_VEC_REMOVE_PAD_TD
#define ASC_AUTOFUSE_OP_VEC_REMOVE_PAD_TD

include "Base.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "Autofuse/Base.td"
include "Autofuse/Interfaces.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// `RemovePadExtend(dst, src, last_axis_actual_size, last_axis_size, non_last_axis_merge)`
// ——Autofuse 自己的包装（`autofuse/ascendc/api/removepad.h:14`），legacy 两侧的
// `RemovePadApiCall` 逐字发的就是它（`codegen/api_call/elewise/remove_pad_api_call.cpp:86`）。
// **两侧共用这一个包装**（矩阵 tier D1，`api_regbase/` 下没有对应文件），所以这里只有一个 def，
// 由 `ApiFamily::Shared` 在注册处说出「两侧都用它」。
//
// 包装内部是一条 `AscendC::GatherMask`，参数的实际含义按它反推：
//
//   从 src 取 `non_last_axis_merge` 行（进 `repeatTimes`），行距 `last_axis_size` 个元素
//   （只经 `last_axis_size * sizeof(T) / 32` 进 `src0RepeatStride`，单位是 32B 块），
//   每行取前 `last_axis_actual_size` 个元素（counter 模式的 `mask`），连续写到 dst。
//
// **dst 侧一个跨度参数都没有**：`GatherMask` 的写是连续的。目的地必须在整串向量化段上稠密，
// 这一条由下降侧的判据保证（`Implementations/RemovePad.cpp`）。
//
// 三个尺寸都留着、按 C++ 形参表的次序发，不在 emitter 里补算式：`last_axis_size` 在
// legacy 那边是 `SizeAlign(行长, 32/sizeof(T))` 现算的，这里发的是 descriptor 里那条真实行距
// ——**两者在对齐的调度下相等，而发真实行距的那一份同时把「布局真的是这样」查了一遍**。
// 那条整除前提也因此变成一条可判的判据，而不是一个假设。
//
// 扩展头只写 `["removepad.h"]`，**无前置头**：包装里的名字是 `AscendC::GatherMask` 与
// `AscendC::GatherMaskParams`，都是限定名，不靠 `utils.h` 的 `KernelUtils` / `MAX_REPEAT_TIME`。
// 与 legacy 两侧的 `LoadApiHeaderFiles` 逐字一致（各返回 `{"removepad.h"}`）。
//
// 不带 `AscFunc`：自动发射器会把它打成 `AscendC::RemovePadExtend`，而包装住在全局作用域。
def AscendC_RemovePadExtendOp
    : VectorOp<"remove_pad_extend", "RemovePadExtend",
               [ApiExtensionOpInterface, OpWithDstInterface, OpWithSrcInterface]> {
    let description = "Call the Autofuse RemovePadExtend wrapper (shared by both API supply tables)";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        AnyType:$lastAxisActualSize,
        AnyType:$lastAxisSize,
        AnyType:$nonLastAxisMerge
    );
    let extraClassDeclaration = extraClassDeclarationBase
        # ApiExtensionHeader<["removepad.h"]>.decl
        # SrcTensorsFromSrc<>.decl;
}

#endif // ASC_AUTOFUSE_OP_VEC_REMOVE_PAD_TD
