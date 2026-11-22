/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EMITASC_TYPES_TD
#define EMITASC_TYPES_TD

include "Dialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class EmitAsc_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<EmitAsc_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def EmitAsc_Mask : EmitAsc_Type<"Mask", "mask"> {
  let description = "Represents 2-element array with high and low mask";
  let extraClassDeclaration = [{
    IntegerType getValueType() { return IntegerType::get(getContext(), 64U); }
  }];
}

def PyStruct : EmitAsc_Type<"PyStruct", "py_struct", [MemRefElementTypeInterface]> {
  let parameters = (ins "StringAttr":$nameAttr, "ArrayAttr":$typesAttr, "ArrayAttr":$namesAttr);
  let assemblyFormat = [{  `<` $nameAttr `,` $typesAttr `,` $namesAttr `>` }];
}

#endif // EMITASC_TYPES_TD
