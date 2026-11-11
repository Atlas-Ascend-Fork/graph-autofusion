// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Cast.td - AFIR cast operations ----------------------*- tablegen -*-===//

#ifndef AFIR_CAST_OPS
#define AFIR_CAST_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
include "Ops/Elementwise.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

// Cast 的结果 dtype 由调用方指定，不能仅从操作数推导，故不继承 InferTypeOpInterface。
// 此处只复用 destination-style 契约。若使用只比较形状的 isCompatibleReturnTypes，
// 可能错误接受推导出的元素类型；谓词和 truediv 的结果可由输入确定，与 Cast 不同。
class AFIR_BufferizableCastOp<string mnemonic, list<Type> inputTypes, list<Type> resultTypes> :
    AFIR_BufferizableUnaryOpBase<mnemonic,
                                 AFIR_BufferizableDestinationStyleTraits,
                                 1,
                                 AFIR_BufferizableShapedType<inputTypes>,
                                 AFIR_BufferizableShapedType<resultTypes>>,
    AFIR_ElementwiseShapeMethods,
    AFIR_OdsElementTypes<inputTypes>,
    AFIR_BufferizableElementwiseMethods<[{ {getInput()} }], [{ {adaptor.getInput()} }]> {
  let extraClassDeclaration = baseMethodDecl # elementwiseShapeMethodDecl # odsElementTypesDecl;
  let extraClassDefinition = baseMethods # elementwiseShapeMethodDef #
      bufferizableElementwiseCommonMethodDef # [{
    ::mlir::LogicalResult $cppClass::verify() {
      const ::mlir::LogicalResult form = isBufferizedForm()
          ? ::mlir::afir::elementwise::verifyBufferForm(getOperation(), {getInput()}, getDestination())
          : ::mlir::afir::elementwise::verifyLogicalForm(getOperation());
      if (::mlir::failed(form)) {
        return form;
      }
      return ::mlir::afir::elementwise::verifyCastPair(getOperation(), getInput().getType(), getResult().getType());
    }
  }];
}

// 这两个列表不是判据，只是「并集里出现过哪些 dtype」——真正的判据是
// `Ops/Cast.cpp` 里那两张 (输入, 输出) 对表。合法的 46 对在 12x12 的笛卡尔积里只占
// 三分之一，ODS 的类型列表说不出这件事，所以两边都写成同一个全集，剩下的交给
// `verify()`。**改这里之前先改那两张表**，反过来会放进没有可用实现的组合。
//
// 原先输入列表少一个 `UI16`、输出列表少一个 `UI64`，而 mem 侧的对表里
// `ui16 -> i16` 与 reg 侧的 `i64 -> ui64` 都在——两处都是漏写，这一轮补上。
// 没被发现是因为这个 op 当时还没有任何 implementation，与 `floor_div` 那一轮同一种。
//
// 两侧都不收标量（对照表两格都是 `no`），所以没有标量形态。
defvar AFIR_CastElementTypes = [I4, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32, BF16];

def AFIR_CastOp : AFIR_BufferizableCastOp<"cast", AFIR_CastElementTypes, AFIR_CastElementTypes> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise dtype conversion of a tile";
  let description = [{
    逐元素改元素类型，形状不变。

    **这是本切片里唯一结果元素类型既不是常量、也不是输入的函数的 op**，所以它不声明
    `InferTypeOpInterface`，也不在 `AFIR_BufferizableUnaryElementwiseOp` 上——理由写在
    上面那个基类头上。

    两侧走的都是 Autofuse 的 `CastExtend` 包装，不是标准的 `AscendC::Cast`（包装自己按
    (输入, 输出) 推 `RoundMode`，还替中间要转两道的那些 dtype 对开循环）。两侧同名而
    **签名完全不同**：mem 侧 `CastExtend(dst, src, tmp, size)`（`cast.h`，要一块 tmp
    buffer），reg 侧 `CastExtend(dst, src, {dims}, {out_stride}, {in_stride})`
    （`cast_reg_base.h`，不要 tmp）。

    合法的 (输入, 输出) 对不是元素类型列表的笛卡尔积——`f16 -> i4` 有供给而
    `f16 -> ui32` 没有，144 格里只有 46 格成立。逐行那两张表在 `Ops/Cast.cpp`，
    mem 侧 36 对、reg 侧 43 对、并集 46 对。本 Op 在 `compute` pipe 上执行。
  }];
}

#endif // AFIR_CAST_OPS
