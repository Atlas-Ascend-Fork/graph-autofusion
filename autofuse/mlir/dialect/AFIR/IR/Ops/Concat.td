// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Concat.td - AFIR concat operations ------------------*- tablegen -*-===//

#ifndef AFIR_CONCAT_OPS
#define AFIR_CONCAT_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"

def AFIR_ConcatOp : AFIR_VariadicOp<"concat", [Pure, InferTensorTypeAdaptor], [I8, UI8, I16, UI16, I32, UI32, I64, F16, F32, BF16], 1> {
  let arguments = !con(
    (ins
      Variadic<TensorOf<[I8, UI8, I16, UI16, I32, UI32, I64, F16, F32, BF16]>>:$inputs,
      I32Attr:$concat_axis
    ),
    commonArgs
  );
  let extraClassDeclaration = baseMethodDecl # [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
  let extraClassDefinition = baseMethods # [{
    bool $cppClass::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      if (l.size() != r.size() || l.size() != 1) return false;
      return succeeded(verifyCompatibleShape(l[0], r[0]));
    }
  }];
}

#endif // AFIR_CONCAT_OPS
