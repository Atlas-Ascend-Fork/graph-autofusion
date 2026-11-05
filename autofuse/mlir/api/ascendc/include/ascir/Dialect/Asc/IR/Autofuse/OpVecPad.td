/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OP_VEC_PAD_TD
#define ASC_AUTOFUSE_OP_VEC_PAD_TD

include "Base.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "Autofuse/Base.td"
include "Autofuse/Interfaces.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// `AscendC::Pad(dst, src, PadParams&, const LocalTensor<uint8_t>&, PadTiling&)`——CANN 官方
// API（`lib/pad/pad.h:53`），legacy 两侧的 `PadApiCall` 逐字发的就是它
// （`codegen/api_call/elewise/pad_api_call.cpp:71`）。两侧供给表共用同一调用（tier D1），
// 与 `RemovePadExtend` 一样只有一个 implementation，但它**不是包装**：函数体住在官方头里，
// 所以此处走 `ApiExtensionHeader` 的第三段（官方头，`-I` 解析、不内联文本）。
//
// **官方头的名字写 `lib/pad/pad.h`，不是 legacy 的 `adv_api/pad/pad.h`**。两个名字指向
// 同一份声明，差在解析它的 `-I` 视角：`adv_api/...` 以设备编译器的
// `aarch64-linux/asc/include` 为根；MLIR kernel TU（CPU e2e 与设备编译 alike）的 `-I` 集
// 里解析得到的是 `lib/pad/pad.h`（`compiler/tikcpp/tikcfw/lib/pad/pad.h`），与
// `kMatmul` 的 `lib/matmul_intf.h` 同一条约定。写 legacy 那个名字在 e2e 编译里直接
// `file not found`——不是风格差异，是解析得到与否。
//
// **要手写 printer，不走 AscFunc 的自动发射**：官方形参表里的 `PadParams&` 是非常量
// 引用，braced 临时量绑不上去，必须像 legacy 那样先落一个局部 `PadParams` 再传
// （`pad_api_call.cpp:66`）。printer 在 `lib/Target/AscendC/Autofuse/Pad.cpp`。
//
// `padNum` 是**右侧**补位数：`PadParams{leftPad, rightPad, padValue}` 的第二个成员
// （`kernel_utils_struct_confusion_pad.h:380` 的声明序），legacy 发 `{0, padNum, 0}`。
// `apiTilingData` 是 `emitasc.member` 的一份 `PadTiling` 局部拷贝（`Transpose` 同款），
// 非常量引用绑上去，包装内部改 tiling 不写穿 kernel 的 tiling 数据。
def AscendC_PadOp
    : VectorOp<"pad", "Pad", [ApiExtensionOpInterface, OpWithDstInterface, OpWithSrcInterface]> {
    let description = "Call the official AscendC::Pad API with a right-pad-only PadParams and host tiling";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        AnySignlessIntegerOrIndex:$padNum,
        AscendC_LocalTensor:$tmpBuffer,
        AnyType:$apiTilingData
    );
    let extraClassDeclaration = extraClassDeclarationBase
        # ApiExtensionHeader<[], [], ["lib/pad/pad.h"]>.decl
        # SrcTensorsFromSrc<>.decl;
}

// `AscendC::DataCopy(dst, src, KernelUtils::BlkAlign<T>(count))`——`afir.pad` 没补位分支
// 的退化形态（`pad_api_call.cpp:75`）。count 是各向量化轴实际元素数的连乘，`BlkAlign<T>`
// 把它抬到 32B 对齐。**`utils.h` 走前导（第二段）**，与本族其余声明它的 op 同一条约定
// （`floor_div.h` / `sigmoid.h` 那批）：`KernelUtils` 是 autofuse 自己的命名空间，前导
// 保证它排在所有扩展头之前，而不是靠 `utils.h` 的字典序运气。
//
// 与 `DataCopyPadExtendOp` 不同的两处：那个是 GM↔UB 的三段搬运包装（`datacopy.h`），
// 本 op 是 UB→UB 的**标准调用 + 一个模板抬齐**，count 的对齐算术不在 MLIR 侧重算——
// legacy 逐字发 `KernelUtils::BlkAlign<dtype>(product)`（`codegen_kernel.cpp:1109`），
// 这里照抄形态，不把 `(n + k - 1) / k * k` 展开成 arith。
def AscendC_DataCopyBlkAlignedOp
    : VectorOp<"data_copy_blk_aligned", "DataCopy",
               [ApiExtensionOpInterface, OpWithDstInterface, OpWithSrcInterface]> {
    let description = "AscendC::DataCopy between local tensors with a KernelUtils::BlkAlign-wrapped count";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        AnySignlessIntegerOrIndex:$count
    );
    let extraClassDeclaration = extraClassDeclarationBase
        # ApiExtensionHeader<[], ["utils.h"]>.decl
        # SrcTensorsFromSrc<>.decl;
}

#endif // ASC_AUTOFUSE_OP_VEC_PAD_TD
