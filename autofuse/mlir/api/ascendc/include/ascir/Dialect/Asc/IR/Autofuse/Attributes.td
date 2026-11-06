/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_ATTRIBUTES_TD
#define ASC_AUTOFUSE_ATTRIBUTES_TD

include "Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def AscendC_ReduceLayoutAttr : I32EnumAttr<"ReduceLayout", "", [
  I32EnumAttrCase<"AR", 0>,
  I32EnumAttrCase<"RA", 1>
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Reduce layout: AR (A,R) or RA (R,A)";
  let underlyingType = "uint8_t";
}

// `AutoFuseTransposeType`（`autofuse/ascendc/api/transpose_base_type.h:19`）逐字镜像：7 个
// 合法 perm 档位 + `TRANSPOSE_INVALID`。存储走 i32、打印是裸整数（`transposeType = 4 : i32`），
// 枚举名由 C++ 发射侧的 `stringifyAutoFuseTransposeType` 还原——`codegen::ConfusionTranspose`
// 的形参要的是 `AutoFuseTransposeType::TRANSPOSE_ND2ND_021` 这个带限定的名字。值与声明序
// 都必须与 legacy 头一致：下降侧的 21 项 perm 折叠表按这些值发码。
def AscendC_AutoFuseTransposeTypeAttr : I32EnumAttr<"AutoFuseTransposeType", "", [
  I32EnumAttrCase<"TRANSPOSE_ND2ND_ONLY", 0>,
  I32EnumAttrCase<"TRANSPOSE_ND2ND_102", 1>,
  I32EnumAttrCase<"TRANSPOSE_ND2ND_0213", 2>,
  I32EnumAttrCase<"TRANSPOSE_ND2ND_2103", 3>,
  I32EnumAttrCase<"TRANSPOSE_ND2ND_021", 4>,
  I32EnumAttrCase<"TRANSPOSE_ND2ND_210", 5>,
  I32EnumAttrCase<"TRANSPOSE_ND2ND_0321", 6>,
  I32EnumAttrCase<"TRANSPOSE_INVALID", 7>
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Folded transpose permutation tier for the mem-based ConfusionTranspose wrapper";
  let underlyingType = "uint8_t";
}

// The `reduce_type` template parameter of `ReduceInit` (`ascendc/api/reduce_init.h`).
// The values are that header's `kReduceOp*` constants and must stay equal to them —
// the emitter prints the constant's name, but the header selects the padding value
// with `if constexpr (reduce_type == kReduceOpSum)`, so a renumbering here that is
// not mirrored there silently pads with the wrong identity element.
def AscendC_ReduceInitTypeAttr : I32EnumAttr<"ReduceInitType", "", [
  I32EnumAttrCase<"Min", 0>,
  I32EnumAttrCase<"Max", 1>,
  I32EnumAttrCase<"Sum", 2>,
  I32EnumAttrCase<"Prod", 3>,
  I32EnumAttrCase<"Any", 4>,
  I32EnumAttrCase<"All", 5>,
  I32EnumAttrCase<"Mean", 6>
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Which reduction ReduceInit pads for: min / max / sum / prod / any / all / mean";
  let underlyingType = "uint8_t";
}

#endif // ASC_AUTOFUSE_ATTRIBUTES_TD
