// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- AFIRAttrTypeBase.td - AFIR attr/type base definitions -*- tablegen -*-===//
//
// Base type/attribute classes for the AFIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CORE_BASE
#define AFIR_CORE_BASE

include "AFIRDialect.td"

//===----------------------------------------------------------------------===//
// Base AFIR Type Definition
//===----------------------------------------------------------------------===//

class AFIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AFIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Base AFIR Attribute Definition
//===----------------------------------------------------------------------===//

class AFIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AFIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Extra integer types used by the ported ops
//===----------------------------------------------------------------------===//
def I4 : I<4>;
def SI4 : SI<4>;
def UI4 : UI<4>;

def F16Attr : FloatAttrBase<F16, "16-bit float attribute">;

#endif // AFIR_CORE_BASE
