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
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <string>

using namespace llvm;
using mlir::raw_indented_ostream;

namespace {

using TypeNamePair = std::pair<std::string, std::string>;

class GenPybindDefsTypes {
    const RecordKeeper& records;

public:
    explicit GenPybindDefsTypes(const RecordKeeper& records) : records(records) {}

    void run(raw_ostream& os);
};

void GenPybindDefsTypes::run(raw_ostream& os)
{
    raw_indented_ostream ios(os);
    for (const auto* def : records.getAllDerivedDefinitions("APIType")) {
        if (!def->getValueAsBit("genTypedef")) {
            continue;
        }
        auto defName = def->getValueAsString("typeName");
        ios << ".def(\"get_asc_" << defName << "Type\", [](PyOpBuilder &self) -> ::mlir::Type {\n";
        ios.indent() << "return self->getType<::mlir::ascendc::" << defName << "Type>();\n";
        ios.unindent() << "})\n";
    }
}

TableGen::Emitter::OptClass<GenPybindDefsTypes> registration(
    "gen-pybind-defs-types", "Generate PyOpBuilder methods from API Types defs");

} // namespace
