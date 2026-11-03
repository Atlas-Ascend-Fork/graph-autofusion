/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

namespace {

class PrintDecls {
    const RecordKeeper& records;

public:
    explicit PrintDecls(const RecordKeeper& records) : records(records) {}

    void run(raw_ostream& os)
    {
        emitSourceFileHeader("Classes and Defs", os);
        os << records;
    }
};

TableGen::Emitter::OptClass<PrintDecls> registration("print-decls", "Print all declared Classes and Defs");

} // namespace
