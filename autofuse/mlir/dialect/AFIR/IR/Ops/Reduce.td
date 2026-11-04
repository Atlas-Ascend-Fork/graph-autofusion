// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Reduce.td - AFIR reduce operations ------------------*- tablegen -*-===//

#ifndef AFIR_REDUCE_OPS
#define AFIR_REDUCE_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"

// Tensor 形态按 axis 归约并降秩；buffer 形态写入 destination，归约维由 descriptor 描述。
// scratch 为调度器规划的 ui8 临时缓冲，各自携带 buffer_desc，不属于结果。
// Buffer 形态读写 destination/scratch，不能声明 Pure；类型与形状由 verifier 校验。
// 算子在 compute pipe 上执行。
class AFIR_ReduceOp<string mnemonic, list<Type> types> : AFIR_BaseOp<mnemonic,
    [AttrSizedOperandSegments, AFIRRoleInterface, AFIRDestinationStyleInterface,
     DeclareOpInterfaceMethods<AFIRRoleInterface>,
     DeclareOpInterfaceMethods<AFIRDestinationStyleInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<ConditionallySpeculatable>]>,
    AFIR_CommonAttrs,
    AFIR_OdsElementTypes<types> {
  let afirStability = AFIR_Experimental;
  let summary = "Reduce a tile over its reduced dimensions";
  let description = [{
    Buffer-form scratch operands carry an `afir.scratch_roles` string array.
    Each role is unique: `api_scratch` is required; `round_result` is additionally
    required by an accumulating implementation. The legacy bridge also preserves
    round_result for non-accumulating reductions; their lowering does not consume
    it. Roles describe uses, not storage
    lifetimes. A non-negative buffer life_time_axis_id always denotes a real axis.
  }];
  let arguments = !con(
    (ins AFIR_BufferizableShapedType<types>:$input,
         Optional<AFIR_BufferizableShapedType<types>>:$destination,
         Variadic<AFIR_BufferizableShapedType<[UI8]>>:$scratch,
         OptionalAttr<I32Attr>:$axis),
    commonArgs
  );
  let results = (outs AFIR_BufferizableShapedType<types>:$result);
  let assemblyFormat = [{
    $input (`outs` `(` $destination^ `)`)? (`scratch` `(` $scratch^ `)`)? attr-dict `:`
    qualified(type($input)) (`,` qualified(type($destination))^)?
    (`,` `scratch` `(` qualified(type($scratch))^ `)`)? `->` qualified(type($result))
  }];
  let extraClassDeclaration = odsElementTypesDecl # [{
    bool isBufferizedForm() { return static_cast<bool>(getDestination()); }
  }];
}

// ODS 保留各 op 的元素类型白名单，具体实现支持由 lowering 能力查询判断。
def AFIR_MaxOp : AFIR_ReduceOp<"max", [F16, F32, BF16]>;
def AFIR_MinOp : AFIR_ReduceOp<"min", [F16, F32, BF16]>;
def AFIR_SumOp : AFIR_ReduceOp<"sum", [I32, F32, BF16]>;
def AFIR_MeanOp : AFIR_ReduceOp<"mean", [F32, BF16]>;
def AFIR_ProdOp : AFIR_ReduceOp<"prod", [F32, BF16]>;
def AFIR_AnyOp : AFIR_ReduceOp<"any", [F32, BF16]>;
def AFIR_AllOp : AFIR_ReduceOp<"all", [F32, BF16]>;

#endif // AFIR_REDUCE_OPS
