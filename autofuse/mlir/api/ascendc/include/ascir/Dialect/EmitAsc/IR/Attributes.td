/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EMITASC_ATTRIBUTES_TD
#define EMITASC_ATTRIBUTES_TD

include "Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class EmitAsc_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<EmitAsc_Dialect, name, traits> {
  let cppNamespace = "::mlir::emitasc";
  let mnemonic = attrMnemonic;
}

def EmitAsc_KernelArgument : I32EnumAttr<"KernelArgument", "kernel argument", [
  I32EnumAttrCase<"Explicit", 0, "explicit">,
  I32EnumAttrCase<"FftsAddr", 1, "ffts_addr">,
  I32EnumAttrCase<"DumpAddr", 2, "dump_addr">,
]> {
  let cppNamespace = "::mlir::emitasc";
  let genSpecializedAttr = 0;
}

def EmitAsc_KernelArgumentAttr
    : EnumAttr<EmitAsc_Dialect, EmitAsc_KernelArgument, "kernel_arg"> {
  let cppNamespace = "::mlir::emitasc";
  let summary = "kind of kernel argument for the launcher";
  let description = [{
    Describes kind of kernel argument: explicit argument, target specific
    argument (workspace, ffts address, etc.)
  }];
}

#endif // EMITASC_ATTRIBUTES_TD
