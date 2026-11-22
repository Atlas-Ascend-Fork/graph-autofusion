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

#include <string_view>

using namespace llvm;
using mlir::raw_indented_ostream;

namespace {

class GenAPITypes {
    const RecordKeeper& records;

public:
    explicit GenAPITypes(const RecordKeeper& records) : records(records) {}

    void run(raw_ostream& os);
};

void GenAPITypes::run(raw_ostream& os)
{
    raw_indented_ostream ios(os);
    auto typeDefs = records.getAllDerivedDefinitions("APIType");
    ios << "#ifdef GEN_EMITTER\n";
    for (const auto* def : typeDefs) {
        if (!def->getValueAsBit("genEmitter")) {
            continue;
        }
        ios << "if (auto concrete = dyn_cast<::mlir::ascendc::" << def->getValueAsString("typeName")
            << "Type>(type)) {\n";
        ios.indent() << "os << \"" << def->getValueAsString("apiName") << "\";\n";
        ios << "return success();\n";
        ios.unindent() << "}\n";
    }
    ios << "#undef GEN_EMITTER\n#endif // GEN_EMITTER\n";
}

TableGen::Emitter::OptClass<GenAPITypes> registration("gen-api-types", "Generate C++ from API type declarations");

} // namespace
