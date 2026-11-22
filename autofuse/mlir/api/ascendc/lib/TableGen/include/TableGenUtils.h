/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOOLS_ASCIR_TBLGEN_UTILS_H
#define TOOLS_ASCIR_TBLGEN_UTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"

#include <string>
#include <vector>

namespace mlir {
namespace asc {
/*
VirtualArg represents an argument of the binding function and should be emitted:

    .def("...", [](OpBuilder &self, ..., const <cppType> &<name>, ...) {
        self.create<...>(..., <substitution>, ...);
    }, ..., "<name>"_a = <defaultValue>)

*/
struct VirtualArg {
    std::string cppType;
    std::string name;
    std::string substitution;
    std::string defaultValue;
    bool optional = false;
};

llvm::StringRef fetchOpClass(llvm::StringRef defName);

void fetchResults(const llvm::DagInit* resultsDag, std::vector<VirtualArg>& dest);

void fetchArguments(const llvm::DagInit* argsDag, std::vector<VirtualArg>& dest);

llvm::StringRef removeDialectPrefix(llvm::StringRef fullName, llvm::StringRef dialectName);
llvm::StringRef removeAscDialectNameSpace(llvm::StringRef fullName, llvm::StringRef ascCppNamespace);

} // namespace asc
} // namespace mlir
#endif
