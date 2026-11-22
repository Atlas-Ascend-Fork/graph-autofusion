// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#ifndef AFIR_IR_OPS_SYMBOLICEVALOP
#define AFIR_IR_OPS_SYMBOLICEVALOP

def AFIR_SymbolicEvalOp : AFIR_Op<"symbolic_eval", [NoMemoryEffect]> {
  let summary = "Evaluate an operand-normalized symbolic expression";
  let description = [{
    The expression may reference only the reserved placeholders __arg0 through
    __argN-1, where N is the number of operands. The operation is the SSA
    boundary for symbolic metadata that has acquired a real consumer. The
    safety pass consumes `positive_args` and `overflow_checked` proof facts
    frozen by the producer before lowering dangerous arithmetic.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                       ExprAttr:$expr,
                       OptionalAttr<DenseI64ArrayAttr>:$positive_args,
                       OptionalAttr<UnitAttr>:$overflow_checked,
                       OptionalAttr<UnitAttr>:$safety_verified);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $result)";
  let hasVerifier = 1;
}

#endif  // AFIR_IR_OPS_SYMBOLICEVALOP
