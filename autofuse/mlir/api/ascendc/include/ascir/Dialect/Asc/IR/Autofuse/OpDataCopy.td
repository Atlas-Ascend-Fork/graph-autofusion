/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OP_DATACOPY_TD
#define ASC_AUTOFUSE_OP_DATACOPY_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "Autofuse/Base.td"
include "Autofuse/Interfaces.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// v35 reg-base wrapper operation. It deliberately does not carry AscFunc:
// its target emitter prints the wrapper from api_regbase/datacopy.h rather
// than the underlying AscendC::DataCopyPad call. The wrapper has the same
// scalar argument form in both directions; dst/src tensor types determine
// which C++ overload is selected.
def AscendC_DataCopyPadExtendOp
    : DataCopyOp<"data_copy_pad_extend", "DataCopyPadExtend", [ApiExtensionOpInterface]> {
  let description = "Call the v35 DataCopyPadExtend wrapper";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AnySignlessIntegerOrIndex:$blockCount,
                       AnySignlessIntegerOrIndex:$blockLen,
                       AnySignlessIntegerOrIndex:$srcStride,
                       AnySignlessIntegerOrIndex:$dstStride);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["datacopy.h"]>.decl;
}

// `DataCopyNddma<T, dim>(dst, src, output_dims[dim], output_stride[dim], input_stride[dim])`
// ——reg 侧那张供给表上的 ND-DMA 包装（`v35/ascendc/api_regbase/datacopy_nddma.h`），
// legacy 的 `NddmaApiCall` 逐字发的就是它（`v35/codegen/reg_api_call/reg_nddma_api_call.cpp`）。
//
// **它与 `AscendC_DataCopyNdDmaOp` 不是同一个函数。** 那个发的是原生
// `AscendC::DataCopy(dst, src, NdDmaParams, dims)`，吃一个带 padValue / padLeft / padRight
// 的参数结构；这个包装把三个 `int64_t` 数组翻成 `MultiCopyLoopInfo` 再调
// `AscendC::DataCopy<T, dim, config>`，没有 padding 一说。名字沾边，形参表不一样。
//
// 三组尺寸都是 `Variadic`，由下降侧按**段**给：段数就是 `dim`，而 legacy 恒发 5、左边用 1
// 补齐。这里**不补齐**——`dim` 是模板实参，由数组长度推出来，补出来的那些 1 只是让包装多
// 转几圈空循环。段数超过包装能收的上限时由下降侧拆外层循环，与 legacy 的 `CreateNddmaCall`
// 同一个办法。
//
// 没有 `AscFunc`：自动 emitter 发不出「三个栈上数组 + 两个模板实参」这种形态，
// 手写 printer 在 `lib/Target/AscendC/Autofuse/DataCopy.cpp` 里。
//
// **扩展头名是 `datacopy_nddma_reg_base.h`，不是源文件名 `datacopy_nddma.h`。** 源文件在
// `v35/ascendc/api_regbase/` 下，`api_regbase/CMakeLists.txt` 把每个 `x.h` 包成一个字符串
// 字面量头 `x_reg_base.h`，而 `ascendc_reg_base_api_register.cpp:314` 的 `api_to_file` 用的
// 就是那个生成名作键。写成源文件名的后果是 `AddKernelIncludes` 发出一条
// `#include "datacopy_nddma.h"`、内联那一层查不到这个键，于是 kernel 里只剩一条解析不了
// 的 include——不是在这一层报错，是设备编译器报「找不到头文件」。
def AscendC_DataCopyNddmaRegBaseOp
    : DataCopyOp<"data_copy_nddma_reg_base", "DataCopyNddma",
                 [AttrSizedOperandSegments, ApiExtensionOpInterface, OpWithSrcInterface]> {
  let description = "Call the reg-base DataCopyNddma wrapper over the collapsed vectorized-axis segments";
  let arguments = (ins AscendC_LocalTensor:$dst,
                       AscendC_GlobalTensor:$src,
                       Variadic<AnyType>:$outputDims,
                       Variadic<AnyType>:$outputStride,
                       Variadic<AnyType>:$inputStride);
  let extraClassDeclaration = extraClassDeclarationBase
      # ApiExtensionHeader<["datacopy_nddma_reg_base.h"]>.decl
      # SrcTensorsFromSrc<>.decl;
}

#endif // ASC_AUTOFUSE_OP_DATACOPY_TD
