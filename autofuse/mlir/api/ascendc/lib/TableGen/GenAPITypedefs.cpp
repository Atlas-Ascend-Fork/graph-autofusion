/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using mlir::raw_indented_ostream;

namespace {

class GenAPITypedefs {
    const RecordKeeper& records;

public:
    explicit GenAPITypedefs(const RecordKeeper& records) : records(records) {}

    void run(raw_ostream& os);
};

void GenAPITypedefs::run(raw_ostream& os)
{
    raw_indented_ostream ios(os);
    for (const auto* def : records.getAllDerivedDefinitions("APIType")) {
        if (!def->getValueAsBit("genTypedef")) {
            continue;
        }
        auto defName = def->getValueAsString("typeName");
        ios << "def AscendC_" << defName << " : AscendC_Type<\"" << defName << "\", \""
            << def->getValueAsString("mnemonic") << "\"> {\n";
        ios.indent() << "let description = \"Represents " << def->getValueAsString("apiName") << "\";\n";
        ios.unindent() << "}\n";
    }
}

TableGen::Emitter::OptClass<GenAPITypedefs> registration(
    "gen-api-typedefs", "Generate MLIR typedefs from API type declarations");

} // namespace
