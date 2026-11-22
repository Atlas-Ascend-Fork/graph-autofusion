// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- AFIRDialect.td - AFIR dialect definition ------------*- tablegen -*-===//
//
// AFIR dialect definition.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_DIALECT
#define AFIR_DIALECT

include "mlir/IR/OpBase.td"

def AFIR_Dialect : Dialect {
  let name = "afir";
  let summary = "Ascend Frontend IR Dialect";
  let description = [{
    AFIR (Ascend Frontend IR) is a high-level dialect for representing
    computational operations targeting Ascend hardware. In this demo it is the
    entry carrier that the schedule-result bridge builds into, and it is lowered
    to the `ascendc` dialect for Ascend C emission.
  }];

  let cppNamespace = "::mlir::afir";

  // We provide custom attribute parsers/printers (types stay standard MLIR).
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;

  // Keep attributes as attributes (not properties) to match the ported subset.
  let usePropertiesForAttributes = 0;

  let extraClassDeclaration = [{
    // Type parsing and printing
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &printer) const override;

    // Attribute parsing and printing
    Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
    void printAttribute(Attribute attr, DialectAsmPrinter &printer) const override;

    LogicalResult verifyRegionArgAttribute(Operation *op, unsigned regionIndex,
                                           unsigned argIndex, NamedAttribute attribute) override;
    LogicalResult verifyRegionResultAttribute(Operation *op, unsigned regionIndex,
                                              unsigned resultIndex, NamedAttribute attribute) override;
    LogicalResult verifyOperationAttribute(Operation *op, NamedAttribute attribute) override;
  }];
}

#endif // AFIR_DIALECT
