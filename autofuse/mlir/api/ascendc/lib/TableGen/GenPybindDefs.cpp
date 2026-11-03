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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>

#include "include/TableGenUtils.h"

using namespace llvm;
using mlir::raw_indented_ostream;

namespace {

using TypeNamePair = std::pair<std::string, std::string>;

class GenPybindDefs {
    const RecordKeeper& records;

public:
    explicit GenPybindDefs(const RecordKeeper& records) : records(records) {}

    void run(raw_ostream& os);
};

void printMethod(raw_indented_ostream& os, const Record* def)
{
    if (def->getValueAsBit("skipDefaultBuilders")) {
        return;
    }
    auto name = mlir::asc::fetchOpClass(def->getName());
    std::vector<mlir::asc::VirtualArg> args;
    mlir::asc::fetchResults(def->getValueAsDag("results"), args);
    bool hasSingleResult = args.size() == 1 && args[0].cppType == "::mlir::Type";
    mlir::asc::fetchArguments(def->getValueAsDag("arguments"), args);
    os << ".def(\"create_";
    auto dialectName = def->getValueAsDef("opDialect")->getValueAsString("name");
    if (dialectName == "ascendc") {
        os << "asc";
    } else {
        os << dialectName;
    }
    os << '_' << name << "\", [](PyOpBuilder &self";
    for (const auto& arg : args) {
        os << ", const " << arg.cppType << " &" << arg.name;
    }
    os << ") ";
    if (hasSingleResult) {
        os << "-> Value ";
    } else {
        os << "-> Operation* ";
    }
    os << "{\n";
    os.indent();
    os << "return ";
    os << "self.create<" << def->getValueAsString("cppNamespace") << "::" << name << ">(";
    interleaveComma(args, os, [&os](const auto& arg) { os << arg.substitution; });
    os << ");\n";
    os.unindent() << "}";
    if (!hasSingleResult) {
        os << ", py::return_value_policy::reference";
    }
    auto lastRequired =
        std::find_if(args.rbegin(), args.rend(), [](const mlir::asc::VirtualArg& arg) { return !arg.optional; });
    std::for_each(lastRequired, args.rend(), [](auto& a) { a.optional = false; });
    for (const auto& arg : args) {
        os << ", \"" << arg.name << "\"_a";
        if (arg.optional) {
            os << " = " << arg.defaultValue;
        }
    }
    os << ")\n";
}

void GenPybindDefs::run(raw_ostream& os)
{
    raw_indented_ostream ios(os);
    for (const auto* def : records.getAllDerivedDefinitions("Op")) {
        printMethod(ios, def);
    }
}

TableGen::Emitter::OptClass<GenPybindDefs> registration(
    "gen-pybind-defs", "Generate PyOpBuilder methods from MLIR operation defs");

} // namespace
