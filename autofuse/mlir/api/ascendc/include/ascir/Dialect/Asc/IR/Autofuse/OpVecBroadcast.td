/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OP_VEC_BROADCAST_TD
#define ASC_AUTOFUSE_OP_VEC_BROADCAST_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "Autofuse/Base.td"
include "Autofuse/Interfaces.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// NOTE: no `AscFunc` trait — the auto-generated emitter cannot express this op's
// C++ form (`AscendC::Broadcast<T, dim, axis>(dst, src, dstShape[], srcShape[])`):
// it needs two non-type template params (dim AND axis), the two LocalTensor
// operands share one element type, and the shape variadics must become stack
// `uint32_t[]` arrays rather than a `reinterpret_cast`. A hand-written printer in
// lib/Target/AscendC/Basic/VecBroadcast.cpp handles all of that instead.
def AscendC_BroadcastL2Op : VectorOp<"broadcast_l2", "Broadcast", [AttrSizedOperandSegments]> {
    let description = "Broadcast elements from source tensor to destination tensor (L2 API)";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        Variadic<I32>:$dstShape,
        Variadic<I32>:$srcShape,
        I32Attr:$constRank
    );
}


// `Broadcast(dst, src, src_m, src_k, src_z, dst_m, dst_k, dst_z, tmp_buf)`——Autofuse 自己的
// 包装（`autofuse/ascendc/api/broadcast.h:512`），legacy 的 `BroadcastOneAxis` 逐字发的就是它
// （`codegen/api_call/broadcast/broadcast_api_call.cpp:189`）。
//
// **它不是上面那个 `BroadcastL2Op`。** 那个发的是原生 `AscendC::Broadcast<T, dim, axis>`：
// 吃 shape 数组、不吃 tmp buffer，而且它的手写 emitter 从**静态**张量 shape 反推 `axis`，
// 这条路上的 tile 一律是 `!ascendc.local_tensor<?xf32>`，反推恒为 0。名字沾边，函数不是同一个。
//
// 包装按 dtype 分流到 `BroadcastInt64` / `BroadcastWithCast` / `BroadcastCommon`，再由
// `BroadcastCommon` 按 `src_m` / `src_k` / `src_z` 的形态选 `BroadcastFirstDim`、原生
// `Broadcast<T,2,1>` 或 `BroadcastMiddleDim`。**发什么形态是调用方的事**，op 只如实带上六个
// 尺寸；`Or`-free 那条不变量（`src_m == 1` 或 `src_z == 0`）由下降侧保证、由 lit 钉住。
//
// 六个尺寸都留着而不是把恒为 0 的 `src_z` / `dst_z` 省掉：其余扩展 op 一律照抄 C++ 形参表，
// 省掉一个就得在 emitter 里补一个字面量 0，把「发的是什么」拆到两个文件里去读。
//
// 扩展头写 `["broadcast.h", "duplicate.h"]`：包装的 `(1,1) -> (A,B)` 那一支调 `Duplicate`。
// 前导 `utils.h`：`BroadcastFirstDim` 用了 `KernelUtils::BlkSize` 与 `MAX_REPEAT_TIME`，
// 两个都只有它声明。两个头的次序与字典序一致，满足 AddKernelIncludes 那一层的约束。
def AscendC_BroadcastDim3Op
    : VectorOp<"broadcast_dim3", "Broadcast", [ApiExtensionOpInterface, OpWithDstInterface, OpWithSrcInterface]> {
    let description = "Call the Autofuse Broadcast wrapper in its three-size (m, k, z) form";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        AnyType:$srcM, AnyType:$srcK, AnyType:$srcZ,
        AnyType:$dstM, AnyType:$dstK, AnyType:$dstZ,
        AscendC_LocalTensor:$sharedTmpBuffer
    );
    let extraClassDeclaration = extraClassDeclarationBase
        # ApiExtensionHeader<["broadcast.h", "duplicate.h"], ["utils.h"]>.decl
        # SrcTensorsFromSrc<>.decl;
}

// `BroadcastExtend<T, rank>(dst, src, dst_shape, src_shape)`——reg 侧那张供给表上的广播
// （`v35/ascendc/api_regbase/broadcast.h:17`），legacy 的 `BroadcastRegApiCall` 逐字发的
// 就是它（`v35/codegen/reg_api_call/reg_broadcast_api_call.cpp:120`）。
//
// **它与上面两个都不是同一个函数**：`BroadcastL2Op` 是原生 `AscendC::Broadcast<T, dim, axis>`，
// `BroadcastDim3Op` 是 mem 侧那张表的包装。这一个吃**整串**向量化轴的 shape 数组、不吃
// tmp buffer，合轴与循环分解由它自己在 tiling 里做（`broadcast_extend_impl.h`），所以
// mem 侧「最内两段窗口 + 外层 `scf.for`」那套模型在这一侧既不需要也对不上。
//
// `rank` 是 shape 数组的长度，同时也是那个非类型模板实参：包装把它转给
// `GetBroadcastTilingInfoExtendImpl<T, constRank>`，而 `broadcast_extend_impl.h:258` 的
// `static_assert` 把它钉在 1..9。两串 shape 等长由下降侧保证。
//
// 两串 shape 是 `Variadic`，因为 rank 由算子的分段结果决定；手写 emitter 把它们落成两个
// 局部 `const uint32_t[]`，与 `BroadcastL2Op` 发数组的写法同一个（`VecBroadcast.cpp`）。
//
// 扩展头只写 `["broadcast_reg_base.h"]`。legacy 的 `BroadcastAscIrCodegenImplV2` 那一行还带
// 一个 `duplicate.h`，那是给它同一个 node impl 里的 `BroadcastScalar` 分支用的——那条路在
// 这里是另一个 implementation（`broadcast.tensor.duplicate_reg_base`），发标准
// `AscendC::Duplicate`、一个扩展头都不引。这条包装自己够不着 `Duplicate`：它往下只调
// `AscendC::BroadcastInternal::*`，那些是限定名，出自 CANN 自己的头。
//
// 没有前导头：包装与 `broadcast_extend_impl.h` 里的名字都带 `AscendC::` 限定，不靠
// `utils.h` 的 `KernelUtils` / `MAX_REPEAT_TIME`。
def AscendC_BroadcastExtendRegBaseOp
    : VectorOp<"broadcast_extend_reg_base", "BroadcastExtend",
               [AttrSizedOperandSegments, ApiExtensionOpInterface, OpWithDstInterface, OpWithSrcInterface]> {
    let description = "Call the reg-base BroadcastExtend wrapper over the whole vectorized-axis shape";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        Variadic<AnyType>:$dstShape,
        Variadic<AnyType>:$srcShape
    );
    let extraClassDeclaration = extraClassDeclarationBase
        # ApiExtensionHeader<["broadcast_reg_base.h"]>.decl
        # SrcTensorsFromSrc<>.decl;
}

#endif //ASC_AUTOFUSE_OP_VEC_BROADCAST_TD
