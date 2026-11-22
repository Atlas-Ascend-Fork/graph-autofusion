/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_ASC_IR_ASC_H
#define ASCIR_DIALECT_ASC_IR_ASC_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ascir/Dialect/Asc/IR/AscFuncTrait.h"

#include "ascir/Dialect/Asc/IR/AscendCDialect.h.inc"

#include "ascir/Dialect/Asc/IR/AscendCEnums.h.inc"

#include "ascir/Dialect/Asc/IR/AscendCTypeInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ascir/Dialect/Asc/IR/AscendCAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ascir/Dialect/Asc/IR/AscendCTypes.h.inc"

#include "ascir/Dialect/Asc/IR/AscendCOpInterfaces.h.inc"
#include "ascir/Dialect/Asc/IR/AscendCAutofuseOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "ascir/Dialect/Asc/IR/AscendCOps.h.inc"

namespace mlir {
namespace ascendc {

ParseResult parsePrettyTPosition(AsmParser& odsParser, TPositionAttr& attr);
void printPrettyTPosition(AsmPrinter& odsPrinter, const TPositionAttr& attr);

ParseResult parsePrettyCubeFormat(AsmParser& odsParser, CubeFormatAttr& attr);
void printPrettyCubeFormat(AsmPrinter& odsPrinter, const CubeFormatAttr& attr);

ParseResult parsePrettyLayoutMode(AsmParser& odsParser, LayoutModeAttr& attr);
void printPrettyLayoutMode(AsmPrinter& odsPrinter, const LayoutModeAttr& attr);

ParseResult parsePrettyCO2Layout(AsmParser& odsParser, CO2LayoutAttr& attr);
void printPrettyCO2Layout(AsmPrinter& odsPrinter, const CO2LayoutAttr& attr);

void registerExternalModels(DialectRegistry& registry);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_DIALECT_ASC_IR_ASC_H
