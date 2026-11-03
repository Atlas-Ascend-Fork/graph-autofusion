/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EMITASC_OPS_TD
#define EMITASC_OPS_TD

include "Dialect.td"
include "Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class EmitAsc_Op<string mnemonic, list<Trait> traits = []>
    : Op<EmitAsc_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::emitasc";
}

def EmitAsc_CallOpaqueOp : EmitAsc_Op<"call_opaque"> {
  let summary = "Call C++ function by its name";
  let description = [{
    Callee name must be demangled. It may include namespaces and template
    arguments. Constructor should be represented as its parent class name.
  }];
  let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$callee_operands);
  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = [{
    $callee `(` $callee_operands `)` attr-dict `:`
    functional-type($callee_operands, $result)
  }];
}

def EmitAsc_CopyStructOp : EmitAsc_Op<"copy_struct"> {
  let summary = "Create local structure and perform memcpy from another one";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$base attr-dict `:` type($base) `,` type($result)";
}

def EmitAsc_DeclarePyStructOp : EmitAsc_Op<"declare_py_struct", [Pure]> {
  let summary = "Declare custom structure";
  let arguments = (ins TypeAttr:$pystruct);
  let assemblyFormat = "$pystruct attr-dict";
}

def EmitAsc_DereferenceOp : EmitAsc_Op<"dereference", [Pure]> {
  let summary = "Dereference pointer to underlying object (unary `operator*`)";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$base attr-dict `:` type($base) `,` type($result)";
}

def EmitAsc_InitStructOp : EmitAsc_Op<"init_struct", [Pure]> {
  let summary = "Create a struct instance and initialize its fields";
  let arguments = (ins StrArrayAttr:$fieldNames, Variadic<AnyType>:$fieldValues);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "Type":$result), [{
      build($_builder, $_state, result, $_builder.getArrayAttr(std::nullopt),
            ValueRange{});
    }]>,
  ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    size_t getNumFields();
    bool hasField(::llvm::StringRef name);
    Value getField(::llvm::StringRef name);
    std::optional<size_t> getFieldOperandIndex(::llvm::StringRef name);
    void setField(::llvm::StringRef name, Value value);
    void addField(::llvm::StringRef name, Value value);
  }];
}

def EmitAsc_MaskOp : EmitAsc_Op<"mask", [Pure]> {
  let summary = "Creates a mask in bit-by-bit mode (pair of 64-bit integers)";
  let arguments = (ins I64 : $maskH, I64 : $maskL);
  let results = (outs EmitAsc_Mask : $mask);
  let assemblyFormat = "$maskH `,` $maskL attr-dict";
  let builders = [
    OpBuilder<(ins "Value":$maskH, "Value":$maskL), [{
      build($_builder, $_state, MaskType::get($_builder.getContext()), maskH, maskL);
    }]>,
  ];
}

def EmitAsc_MemberOp : EmitAsc_Op<"member"> {
  let arguments = (ins AnyType:$base, StrAttr:$field);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $base $field attr-dict `:` type($base) `,` type($result)
  }];
}

def EmitAsc_MemberPtrOp : EmitAsc_Op<"member_ptr", [Pure]> {
  let summary = "Access pointer-to-struct member (`operator->`)";
  let description = [{
    This operation represents getting an address of a struct member at the given
    pointer, and it should be emitted as the following:

    ```cpp
    auto *result = (ResultT *)&base->member;
    ```
  }];
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base, IndexAttr:$index, OptionalAttr<StrAttr>:$field);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);
  let assemblyFormat = [{
    $base `[` $index `]` ($field^)? attr-dict `:` type($base) `,` type($result)
  }];
}

def EmitAsc_MemberRefOp : EmitAsc_Op<"member_ref", [Pure]> {
  let summary = "Access pointer-to-struct member (emit result as reference)";
  let description = [{
    This operation represents obtaining a struct member at the given pointer as
    a reference, and it should be emitted as the following:

    ```cpp
    ResultT &result = base->member;
    ```
  }];
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base, IndexAttr:$index, OptionalAttr<StrAttr>:$field);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $base `[` $index `]` ($field^)? attr-dict `:` type($base) `,` type($result)
  }];
}

def EmitAsc_SetMemberOp : EmitAsc_Op<"set_member"> {
  let arguments = (ins AnyType:$base, StrAttr:$field, AnyType:$value);
  let assemblyFormat = [{
    $base $field `,` $value attr-dict `:` type($base) `,` type($value)
  }];
}

def EmitAsc_PtrOffsetOp : EmitAsc_Op<"ptr_offset", [
  Pure, DeclareOpInterfaceMethods<ViewLikeOpInterface>
]> {
  let summary = "Apply `operator+` between pointer and offset";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base, OptionalAttr<IndexAttr>:$staticOffset,
                       Optional<Index>:$dynamicOffset);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);
  let assemblyFormat = [{
    $base `[` ($dynamicOffset^):($staticOffset ` `)? `]` attr-dict `:`
    type($base) `,` type($result)
  }];
  let hasFolder = 1;
}

def EmitAsc_ReinterpretCastOp : EmitAsc_Op<"reinterpret_cast", [
  Pure, DeclareOpInterfaceMethods<CastOpInterface>
]> {
  let summary = "Convert between types by reinterpreting the underlying data";
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

def EmitAsc_VariableOp : EmitAsc_Op<"variable"> {
  let summary = "Define a mutable variable";
  let description = [{
    This operation combines an allocation and storing a value to `memref<1x...>`.
    Its element type must be the same as _init_ type (either static or dynamic).
  }];
  let arguments = (ins OptionalAttr<TypedAttrInterface>:$staticInit,
                       Optional<AnyType>:$dynamicInit);
  let results = (outs AnyNon0RankedMemRef:$result);
  let assemblyFormat = [{
    attr-dict ($dynamicInit^ `:` type($dynamicInit)):($staticInit)? `,` type($result)
  }];
  let builders = [
    OpBuilder<(ins "OpFoldResult":$init), [{
      TypedAttr staticInit;
      Value dynamicInit;
      Type type;
      if (auto di = dyn_cast<Value>(init)) {
        dynamicInit = di;
        type = di.getType();
      } else {
        auto si = cast<TypedAttr>(cast<Attribute>(init));
        staticInit = si;
        type = si.getType();
      }
      build($_builder, $_state, MemRefType::get(1, type), staticInit, dynamicInit);
    }]>,
  ];
  let extraClassDeclaration = [{
    bool isStatic();
    OpFoldResult getInit(bool fold = true);
  }];
  let hasCanonicalizeMethod = 1;
}

def EmitAsc_VerbatimOp : EmitAsc_Op<"verbatim"> {
  let arguments = (ins StrAttr:$value, Variadic<AnyType>:$args);
  let assemblyFormat = "$value ($args^ `:` type($args))?  attr-dict";
}

def VFForOp : EmitAsc_Op<"vf_for", [SingleBlockImplicitTerminator<"::mlir::emitasc::YieldOp">, AutomaticAllocationScope,
  RecursivelySpeculatable, RecursiveMemoryEffects, DeclareOpInterfaceMethods<LoopLikeOpInterface>
]> {
  let arguments = (ins AnySignlessIntegerOrIndex:$upperBound);
  let regions = (region SizedRegion<1>:$region);

  let extraClassDeclaration = [{
    BlockArgument getInductionVar() { return getBody()->getArgument(0); }
    int64_t getLowerBoundAsInt() { return 0; }
    int64_t getStepAsInt() { return 1; }
    void setUpperBound(Value bound) { getOperation()->setOperand(0, bound); }
    Type getUnderlyingType() { return IntegerType::get(getContext(), 16, IntegerType::Unsigned); }
  }];

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$upperBound)>];

  let hasCanonicalizeMethod = 1;
  let assemblyFormat = "operands `:` qualified(type(operands)) $region attr-dict";
  let hasVerifier = 1;
}

def VecScopeOp : EmitAsc_Op<"vec_scope", [SingleBlockImplicitTerminator<"::mlir::emitasc::YieldOp">]> {
  let regions = (region SizedRegion<1> : $region);
  let assemblyFormat = "$region attr-dict";
}

def YieldOp : EmitAsc_Op<"yield", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

#endif // EMITASC_OPS_TD
