/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OP_VEC_REDUCE_ND_TD
#define ASC_AUTOFUSE_OP_VEC_REDUCE_ND_TD

include "Base.td"
include "Core/Attributes.td"
include "Autofuse/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "Autofuse/Base.td"
include "Autofuse/Interfaces.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"


// 两维归约的高阶 API 族。`adv_api/reduce/reduce.h` 里 ReduceSum / ReduceMax / ReduceMin /
// ReduceProd / ReduceAny / ReduceAll 六个函数的签名逐字相同，只差函数名，所以它们共用这一个
// 类，各自只填 mnemonic 与 apiName。
//
// 没有 `AscFunc` trait、也没有 `paramTypeLists`：自动生成的 emitter 表达不了这个 C++ 形态
//
//   AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, false>(
//       dst, src, sharedTmpBuffer, srcShape, srcInnerPad)
//
// 三处超出它的能力：`pattern` 是**类型**模板参数（`AscendC::Pattern::Reduce::AR`，
// `reduce_common.h:34`）而不是值；`isReuseSource` 是第二个非类型模板参数；`srcShape` 是
// `const uint32_t[]`，调用前必须先在栈上物化成数组。这三件由
// `lib/Target/AscendC/Autofuse/VecReduceND.cpp` 里手写的 printer 处理，与 BroadcastL2Op
// 出于同样的理由走同一条路。printer 取的是 `getAPIName()`，所以这一族共用同一份实现。
//
// 没有 ReduceMean：它在跨 tile 累加下不能按 tile 求均值再合并（那是「均值的均值」）。与
// legacy 一致，Mean 发 ReduceSum，最后一次性 `Muls` 乘上归约元素数的倒数
// （`codegen/api_call/reduce/reduce_api_call.cpp:80`）。
class BaseReduce2DL2Op<string mnemonic, string apiName, list<Trait> traits = []>
    : VectorOp<mnemonic, apiName, !listconcat([OpWithDstInterface], traits)> {
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$sharedTmpBuffer,
    AscendC_ReduceLayoutAttr:$layout,
    Variadic<I32>:$srcShape,
    DefaultValuedAttr<BoolAttr, "false">:$srcInnerPad,
    DefaultValuedAttr<BoolAttr, "false">:$isReuseSource
  );
  let description = [{
    Two-dimensional reduction over the high-level AscendC API.

    `layout` selects the pattern: `AR` reduces the last dimension of `[A, R]`,
    `RA` the first dimension of `[R, A]`. `srcShape` carries those two extents in
    the same order.

    `srcInnerPad` tells the API that the last axis of `src` is padded up to a
    32-byte boundary; the padding itself must already hold the reduction's identity
    element, which is what `ascendc.reduce_init` is for.

    `sharedTmpBuffer` is the API's scratch space. It is required here: the overload
    without it pops a stack buffer at runtime, which this pipeline does not model.
  }];
  code reduceClassDeclaration = extraClassDeclarationBase # [{
    bool isARLayout() {
      return getLayout() == ReduceLayout::AR;
    }

    bool isRALayout() {
      return getLayout() == ReduceLayout::RA;
    }
  }];
  let extraClassDeclaration = reduceClassDeclaration;
  // `srcShape` is variadic in ODS but the API reads it at fixed indices 0 and 1
  // (`reduce.h:233` takes `const uint32_t srcShape[]` and indexes both). The hand
  // written printer expands whatever is there into a stack array, so a one-element
  // operand list compiles into a one-element array that the API still reads twice:
  // an out-of-bounds read at run time, with nothing wrong to see at build time. The
  // other three irregularities of this op (the type template parameter, `isReuseSource`
  // and the stack array) are all carried by that printer; arity is the one it cannot.
  let hasVerifier = 1;
}

def AscendC_ReduceSum2DL2Op : BaseReduce2DL2Op<"reduce_sum_2d_l2", "ReduceSum">;
// Exact integer sum is an Autofuse extension, not AscendC::ReduceSum.
def AscendC_ReduceSumInt32Op : BaseReduce2DL2Op<"reduce_sum_int32", "ReduceSumInt32", [ApiExtensionOpInterface]> {
  let extraClassDeclaration = reduceClassDeclaration # ApiExtensionHeader<["reduce.h"], ["utils.h"]>.decl;
}
def AscendC_ReduceMax2DL2Op : BaseReduce2DL2Op<"reduce_max_2d_l2", "ReduceMax">;
def AscendC_ReduceMin2DL2Op : BaseReduce2DL2Op<"reduce_min_2d_l2", "ReduceMin">;
def AscendC_ReduceProd2DL2Op : BaseReduce2DL2Op<"reduce_prod_2d_l2", "ReduceProd">;
def AscendC_ReduceAny2DL2Op : BaseReduce2DL2Op<"reduce_any_2d_l2", "ReduceAny">;
def AscendC_ReduceAll2DL2Op : BaseReduce2DL2Op<"reduce_all_2d_l2", "ReduceAll">;

// Autofuse's own helper, not an AscendC API: `ascendc/api/reduce_init.h`. It fills
// the alignment padding with the reduction identity. This is a lowering-only
// description: convert-afir-to-ascendc expands it to native Duplicate operations
// and structured control flow. It must not survive to C++ emission.
def AscendC_ReduceInitOp : VectorOp<"reduce_init", "ReduceInit", [OpWithDstInterface]> {
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AnySignlessIntegerOrIndex:$dimA,
    AnySignlessIntegerOrIndex:$dimR,
    AnySignlessIntegerOrIndex:$dimRCurrent,
    AnySignlessIntegerOrIndex:$innerR,
    AnySignlessIntegerOrIndex:$innerRStride,
    AscendC_ReduceInitTypeAttr:$reduceType,
    DefaultValuedAttr<BoolAttr, "false">:$isTailLast
  );
  let description = [{
    Fill the reduce axis' alignment padding with the reduction's identity element.

    `dimA` / `dimR` are the padded two-dimensional extents the reduce API will see,
    `dimRCurrent` the real extent of the current R segment, and `innerR` the extent
    of the innermost reduce axis. `isTailLast` says whether the reduce axis is the
    last one, which changes how padding addresses are computed. `innerRStride` is the
    32-byte-aligned physical pitch of the innermost R axis, in elements. It comes
    from the nominal extent and does not shrink with `innerR` on tail tiles.
  }];
}
#endif // ASC_AUTOFUSE_OP_VEC_REDUCE_ND_TD
