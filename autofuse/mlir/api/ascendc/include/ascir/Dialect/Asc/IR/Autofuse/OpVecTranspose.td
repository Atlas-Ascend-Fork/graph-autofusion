/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You should refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OP_VEC_TRANSPOSE_TD
#define ASC_AUTOFUSE_OP_VEC_TRANSPOSE_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "Autofuse/Attributes.td"
include "Autofuse/Base.td"
include "Autofuse/Interfaces.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// `TransposeExtend<inner_dim, dim>(dst, src, tmp_buf, dst_dims[dim], src_strides[dim],
// dst_strides[dim])`——reg 侧供给表上的转置包装（`v35/ascendc/api_regbase/transpose.h:299`），
// legacy 的 `TransposeRegApiCall` 逐字发的就是它（`v35/codegen/reg_api_call/reg_transpose_api_call.cpp`）。
//
// **T 不发**：包装的模板表是 `<inner_dim, dim, T>`，legacy 只发前两个（`api_name_` +
// `template_params`），T 从 `LocalTensor` 形参推导——这里照抄，第三个实参写出来反而与
// legacy 的文本不同。
//
// 三组尺寸按**段**给（`Variadic`），段数 = `totalDim`。legacy 的发射形态是调用内联的
// 花括号列表（`GenTransposeDimParam`：`{static_cast<int16_t>(d0), ...}, {...}, {...}`），
// 不是栈数组；每个元素 cast 到 `int16_t`/`int32_t`（`RangeType<T>` 的宽度，按元素
// sizeof 选，`GenDimensionParam` 同一条）。`>4` 维的外层循环由下降侧拆
// （`VectorizedLoops`，与 legacy 的 `outer_loop_axes` 同语义）。
//
// 扩展头 `transpose_reg_base.h` 是 CMake 生成的 raw-string 头名（源文件
// `api_regbase/transpose.h`，`ascendc_reg_base_api_register.cpp` 按生成名注册）；
// 官方头 `basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h` 声明 MicroAPI
// `Gather` 等原语，legacy 由 `IncludeApiHeaderFiles` 发进 kernel（
// `v2_ascir_codegen_impl.h:3745-3749`），这里走 `ApiExtensionHeader` 的第三段列表。
//
// 没有 `AscFunc`：自动 emitter 发不出「双模板实参 + 三个花括号列表」的形态，手写
// printer 在 `lib/Target/AscendC/Autofuse/Transpose.cpp`。
def AscendC_TransposeExtendRegBaseOp
    : VectorOp<"transpose_extend_reg_base", "TransposeExtend",
               [AttrSizedOperandSegments, ApiExtensionOpInterface, OpWithDstInterface,
                OpWithSrcInterface]> {
    let description = "Call the reg-base TransposeExtend wrapper over the collapsed vectorized-axis segments";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        AscendC_LocalTensor:$tmpBuffer,
        Variadic<AnyType>:$dstDims,
        Variadic<AnyType>:$srcStrides,
        Variadic<AnyType>:$dstStrides,
        I8Attr:$innerDim,
        I8Attr:$totalDim
    );
    let extraClassDeclaration = extraClassDeclarationBase
        # ApiExtensionHeader<["transpose_reg_base.h"], [],
                             ["basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h"]>.decl
        # SrcTensorsFromSrc<>.decl;
}

// `codegen::ConfusionTranspose<T>(dst, src, tmp_buf, transposeType, tiling)`——mem 侧供给表
// 上的转置包装（`autofuse/ascendc/api/transpose.h:1325`），legacy 的 `TransposeApiCall`
// 逐字发的就是它（`codegen/api_call/transpose/transpose_api_call.cpp:146-158`）。
// **带 `codegen::` 限定**：包装住在 `namespace codegen` 里，是全局名，emitter 手写前缀。
//
// 形参表最后一个吃 `ConfusionTransposeTiling &`（**非 const 引用**）：tiling 值操作数来自
// `emitasc::MemberOp`（`auto v = tiling.<field>;`，一份局部拷贝），legacy 的
// `auto apiTilingData = t-><field>;` 同一形态——引用绑到拷贝上，包装内部改 tiling 不会
// 写穿 kernel 的 tiling 数据。
//
// 枚举属性打印 `AutoFuseTransposeType::<symbol>`，7 个合法值与
// `transpose_base_type.h` 的枚举同名；legacy 发的局部变量
// `AutoFuseTransposeType transposeType = ...;` 不复刻（语义等价、少一行，见设计文档
// 「不做的事 ④」）。**尾随 `AscendC::PipeBarrier<PIPE_ALL>()` 是调用的一部分**：legacy
// 每条 ConfusionTranspose 后面都跟着它（`transpose_api_call.cpp:158`），V 管线写完 UB
// 才允许后续读，emitter 一并发出，不让下降侧记得。
//
// 扩展头两个：`transpose_base_type.h`（枚举）在前、`transpose.h`（包装）在后，字典序即
// 依赖序，与 legacy 两侧 `LoadApiHeaderFiles` 的返回一致。无官方头——mem 侧包装只调
// `TransDataTo5HD` / `DataCopy`（`kernel_operator.h` 已含）。
def AscendC_ConfusionTransposeOp
    : VectorOp<"confusion_transpose", "ConfusionTranspose",
               [ApiExtensionOpInterface, OpWithDstInterface, OpWithSrcInterface]> {
    let description = "Call the mem-based codegen::ConfusionTranspose wrapper with folded permutation and host tiling";
    let arguments = (ins
        AscendC_LocalTensor:$dst,
        AscendC_LocalTensor:$src,
        AscendC_LocalTensor:$tmpBuffer,
        AscendC_AutoFuseTransposeTypeAttr:$transposeType,
        AnyType:$apiTilingData
    );
    let extraClassDeclaration = extraClassDeclarationBase
        # ApiExtensionHeader<["transpose_base_type.h", "transpose.h"]>.decl
        # SrcTensorsFromSrc<>.decl;
}

#endif // ASC_AUTOFUSE_OP_VEC_TRANSPOSE_TD
