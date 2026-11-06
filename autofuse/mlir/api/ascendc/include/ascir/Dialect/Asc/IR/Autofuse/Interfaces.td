/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_INTERFACES_TD
#define ASC_AUTOFUSE_INTERFACES_TD

include "Interfaces.td"

// `AscendC_OpInterface` carries its method list as fields of the class it mixes
// in (`AscendC_InterfaceMethods`), and a `def` body names them bare. TableGen
// cannot reopen that class, so an interface declared here brings its own methods
// along in a derived class rather than appending to the upstream list.
class Autofuse_OpInterface<string name, list<Interface> baseInterfaces = []>
    : AscendC_OpInterface<name, baseInterfaces> {
  InterfaceMethod getExtensionHeaders = InterfaceMethod<
    "Obtain the Autofuse API extension headers declaring this operation's API, in inlining order",
    "::llvm::ArrayRef<::llvm::StringRef>", "getExtensionHeaders">;
  InterfaceMethod getExtensionPreludeHeaders = InterfaceMethod<
    "Obtain the headers this operation's extension headers must all be preceded by, possibly none",
    "::llvm::ArrayRef<::llvm::StringRef>", "getExtensionPreludeHeaders">;
  InterfaceMethod getExtensionOfficialIncludes = InterfaceMethod<
    "Obtain the CANN official headers this operation's kernel must include, possibly none",
    "::llvm::ArrayRef<::llvm::StringRef>", "getExtensionOfficialIncludes">;
}

// Autofuse ships two families of AscendC API extension headers that are not
// part of the stock AscendC library: mem-based `autofuse/ascendc/api/` and
// reg-based `autofuse/v35/ascendc/api_regbase/`. An operation lowered onto one
// of those APIs declares which header provides it; the kernel-source layer
// collects the declarations and inlines the header text. Header names are
// globally unique across both families (reg-based ones end in `_reg_base.h`),
// so the name alone identifies the provider and carries no platform axis.
def ApiExtensionOpInterface : Autofuse_OpInterface<"ApiExtensionOp", [APIOpInterface]> {
  let description = "Operation whose API comes from an Autofuse API extension header";
  let methods = [getExtensionHeaders, getExtensionPreludeHeaders, getExtensionOfficialIncludes];
}

#endif // ASC_AUTOFUSE_INTERFACES_TD
