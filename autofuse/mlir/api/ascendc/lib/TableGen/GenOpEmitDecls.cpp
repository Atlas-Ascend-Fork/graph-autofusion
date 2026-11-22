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

#include "include/Constant.h"
#include "include/TableGenUtils.h"

using namespace llvm;
using mlir::raw_indented_ostream;

namespace {
class GenOpEmitsDecls {
public:
    explicit GenOpEmitsDecls(const RecordKeeper& records) : records(records) {}
    void run(raw_ostream& os);

private:
    const RecordKeeper& records;
};

void printOpType(raw_indented_ostream& os, const Record* def)
{
    const auto opTypeFullName = def->getName();
    const auto opType = mlir::asc::removeDialectPrefix(opTypeFullName, mlir::asc::kAscDialectName);
    os << mlir::asc::kAscDialectNameSpace << opType << " ,";
}

void GenOpEmitsDecls::run(raw_ostream& os)
{
    raw_indented_ostream ios(os);
    for (const auto* def : records.getAllDerivedDefinitions("Op")) {
        if (!def->getValueAsBit(mlir::asc::kAutoEmitAttr)) {
            continue;
        }
        printOpType(ios, def);
    }
}

TableGen::Emitter::OptClass<GenOpEmitsDecls> registration(
    "gen-opemit-decls", "Generate op emit methods from MLIR operation decls");
} // namespace
