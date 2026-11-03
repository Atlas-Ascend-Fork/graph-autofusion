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
#include <cctype>
#include <iterator>
#include <sstream>
#include <string>

#include "include/Constant.h"
#include "include/TableGenUtils.h"

using namespace llvm;
using mlir::raw_indented_ostream;

namespace {
class GenOpEmitsDefs {
public:
    explicit GenOpEmitsDefs(const RecordKeeper& records) : records(records) {}
    void run(raw_ostream& os);

private:
    const RecordKeeper& records;
};

void printFuncDefine(raw_indented_ostream& os, StringRef opType)
{
    os << mlir::asc::kRetType << mlir::asc::kSpaceSeparated << mlir::asc::kPrintFuncName;
    os << "(CodeEmitter &emitter, ";
    os << mlir::asc::kAscDialectNameSpace << opType << mlir::asc::kSpaceSeparated << "op) {\n";
}

std::string capitalizeFirstLetter(const std::string& str)
{
    std::string capStr(str);
    if (!str.empty()) {
        capStr[0] = static_cast<char>(std::toupper(str[0]));
    }
    return capStr;
}

std::string genGetterName(const std::vector<mlir::asc::VirtualArg>& args, size_t i)
{
    if (i >= args.size()) {
        return {};
    }
    return mlir::asc::kGetFunPrefix.str() + capitalizeFirstLetter(args[i].name);
}

bool hasTrait(const Record* def, StringRef traitName)
{
    auto traits = def->getValueAsListOfDefs(mlir::asc::kTraitName);
    return llvm::any_of(traits, [&](const Record* trait) { return trait->getName() == traitName; });
}

void indentedNewLine(raw_indented_ostream& os) { os << mlir::asc::kLineBreak << mlir::asc::kIndentationSpace; }

void printInferOperandType(raw_indented_ostream& os, StringRef operand, const std::string& varName)
{
    os << "auto " << varName << " = op." << operand << "().getType();";
    indentedNewLine(os);
    os << R"(FAIL_OR(emitter.emitType(op.getLoc(), )" << varName << "));";
    indentedNewLine(os);
}

void printTemplateType(raw_indented_ostream& os, StringRef operand, const std::string& varName)
{
    os << "auto " << varName << " = op." << operand << "();";
    indentedNewLine(os);
    os << R"(FAIL_OR(emitter.emitType(op.getLoc(), )" << varName << "));";
    indentedNewLine(os);
}

void printInferElementType(raw_indented_ostream& os, StringRef operand, StringRef typeName, StringRef elementName)
{
    os << "  auto " << typeName << " = op." << operand << "().getType();";
    indentedNewLine(os);
    os << "auto " << elementName << " = " << typeName << ".getElementType();";
    indentedNewLine(os);
    os << R"(FAIL_OR(emitter.emitType(op.getLoc(),)" << elementName << R"());)";
    indentedNewLine(os);
}

void printInferEnumType(raw_indented_ostream& os, StringRef operand, StringRef attrType, const std::string& typeName)
{
    os << "  auto " << typeName << "= op." << operand << "();";
    indentedNewLine(os);
    os << R"(os << ascNamespace << ")" << attrType << R"(::")";
    os << R"( << ascendc::stringifyEnum()" << typeName << R"().upper();)";
    indentedNewLine(os);
}

void printFuncOutputDeclaration(raw_indented_ostream& os)
{
    os << R"(  auto resNum = op.getOperation()->getNumResults();
  auto& os = emitter.ostream();
  if (resNum == 1) {
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    os << " = ";
  })";
    indentedNewLine(os);
}

void printOperand(raw_indented_ostream& os, StringRef operand, const mlir::asc::VirtualArg& arg)
{
    if (arg.optional) {
        os << "EXEC_IF_TRUE(op." << operand << "(), ";
        os << R"(os << ", " << emitter.getOrCreateName(op.)" << operand << R"(());)";
        os << ")";
    } else {
        os << R"(os << emitter.getOrCreateName(op.)" << operand << R"(());)";
    }
    indentedNewLine(os);
}

void printOperandValue(raw_indented_ostream& os, StringRef operand)
{
    os << R"(os << emitter.getOrCreateName(op.)" << operand << "());";
    indentedNewLine(os);
}

void printPointerOperandValue(raw_indented_ostream& os, StringRef operand)
{
    os << "os << \"&\" << emitter.getOrCreateName(op." << operand << "());";
    indentedNewLine(os);
}

void printPointerToIntOperandValue(raw_indented_ostream& os, StringRef operand)
{
    os << "os << \"reinterpret_cast<uint64_t>(\" << emitter.getOrCreateName(op." << operand << "()) << \")\";";
    indentedNewLine(os);
}

bool hasTemplateParams(const std::vector<int64_t>& paramTypes)
{
    return std::any_of(
        paramTypes.cbegin(), paramTypes.cend(), [](const auto& type) { return type > mlir::asc::kNormalType; });
}

void printTemplateCallStart(raw_indented_ostream& os, const std::vector<int64_t>& paramTypes, bool isMemberFunc)
{
    if (isMemberFunc) {
        if (paramTypes[0] != mlir::asc::kNormalType) {
            os << mlir::asc::kOutTemplate;
            indentedNewLine(os);
        }
        os << mlir::asc::kOutFuncName << mlir::asc::kOutTemplateCallStart;
        indentedNewLine(os);
    } else {
        os << mlir::asc::kOutAscNamespace << mlir::asc::kOutFuncName << mlir::asc::kOutTemplateCallStart;
        indentedNewLine(os);
    }
}

bool printTemplateParam(
    raw_indented_ostream& os, const Record* def, const std::vector<int64_t>& paramTypes,
    const std::vector<mlir::asc::VirtualArg>& args, bool isMemberFunc = false)
{
    if (!hasTemplateParams(paramTypes)) {
        return false;
    }
    printTemplateCallStart(os, paramTypes, isMemberFunc);
    bool first = true;
    for (size_t i = 0; i < paramTypes.size(); ++i) {
        if (paramTypes[i] <= mlir::asc::kNormalType) {
            continue;
        }
        if (!first) {
            os << mlir::asc::kOutSeparate;
            indentedNewLine(os);
        }
        first = false;
        std::string templateTypeVar = "templateType" + std::to_string(i);
        std::string elementTypeVar = "elType" + std::to_string(i);
        std::string attrTypeVar = "iAttr" + std::to_string(i);
        switch (paramTypes[i]) {
            case mlir::asc::kInferType: // infer operand type
                printInferOperandType(os, genGetterName(args, i), templateTypeVar);
                break;
            // infer operand element type, such as get T form LocalTensor<T>
            case mlir::asc::kInferElementType:
                printInferElementType(os, genGetterName(args, i), templateTypeVar, elementTypeVar);
                break;
            case mlir::asc::kInferEnumType: // pass by attr
                printInferEnumType(
                    os, genGetterName(args, i),
                    mlir::asc::removeAscDialectNameSpace(args[i].cppType, def->getValueAsString("cppNamespace")),
                    attrTypeVar);
                break;
            case mlir::asc::kInferValue: // pass by value
                printOperandValue(os, genGetterName(args, i));
                break;
            case mlir::asc::kTemplateType:  // pass by template type
            case mlir::asc::kInferTypeAttr: // type attribute, use directly as template type
                printTemplateType(os, genGetterName(args, i), templateTypeVar);
                break;
        }
    }
    os << mlir::asc::kOutTemplateCallEnd;
    indentedNewLine(os);
    return true;
}

void printFunctionParam(
    raw_indented_ostream& os, const Record* def, const std::vector<int64_t>& paramTypes,
    const std::vector<mlir::asc::VirtualArg>& args, bool hasTemplate = false, bool isMemberFunc = false)
{
    size_t i = isMemberFunc ? 1 : 0;
    if (!hasTemplate) {
        if (!isMemberFunc)
            os << mlir::asc::kOutAscNamespace;
        os << mlir::asc::kOutFuncName << mlir::asc::kOutFuncCallStart;
        indentedNewLine(os);
    } else {
        os << mlir::asc::kOutFuncCallStart;
        indentedNewLine(os);
    }
    bool first = true;
    for (; i < paramTypes.size(); ++i) {
        if (paramTypes[i] > mlir::asc::kInferElementType) {
            continue;
        }
        if (!first && !args[i].optional) {
            os << mlir::asc::kOutSeparate;
            indentedNewLine(os);
        }
        first = false;
        std::string operandVar = "var" + std::to_string(i);
        if (paramTypes[i] == -1) { // -1 means attr argument
            printInferEnumType(
                os, genGetterName(args, i),
                mlir::asc::removeAscDialectNameSpace(args[i].cppType, def->getValueAsString("cppNamespace")),
                operandVar);
        } else if (paramTypes[i] == mlir::asc::kPointerValue) {
            printPointerOperandValue(os, genGetterName(args, i));
        } else if (paramTypes[i] == mlir::asc::kPointerToIntValue) {
            printPointerToIntOperandValue(os, genGetterName(args, i));
        } else {
            printOperand(os, genGetterName(args, i), args[i]);
        }
    }
    os << "os << \")\";";
    indentedNewLine(os);
    os << R"(return success();)"
       << "\n}\n";
}

void printOp(raw_indented_ostream& os, const Record* def)
{
    const auto opTypeFullName = def->getName();
    const auto opType = mlir::asc::removeDialectPrefix(opTypeFullName, "AscendC");
    auto templatePos = def->getValueAsListOfInts("paramTypeLists");
    std::vector<mlir::asc::VirtualArg> args;
    mlir::asc::fetchArguments(def->getValueAsDag("arguments"), args);
    bool isMemberFunc = hasTrait(def, "AscMemberFunc");
    printFuncDefine(os, opType);
    if (!templatePos.empty()) {
        printFuncOutputDeclaration(os);
        if (isMemberFunc) {
            os << R"(os << emitter.getOrCreateName(op.)" << genGetterName(args, 0) << R"(()) << ".";)";
            indentedNewLine(os);
        }
        bool hasTemplate = printTemplateParam(os, def, templatePos, args, isMemberFunc);
        printFunctionParam(os, def, templatePos, args, hasTemplate, isMemberFunc);
    } else {
        os << mlir::asc::kIndentationSpace << "return autoPrintOp<ascendc::" << opType << ">(emitter, op);\n}\n";
    }
}

void GenOpEmitsDefs::run(raw_ostream& os)
{
    raw_indented_ostream ios(os);
    for (const auto* def : records.getAllDerivedDefinitions("Op")) {
        if (!def->getValueAsBit(mlir::asc::kAutoEmitAttr)) {
            continue;
        }
        printOp(ios, def);
    }
}

TableGen::Emitter::OptClass<GenOpEmitsDefs> registration(
    "gen-opemit-defs", "Generate op emit methods from MLIR operation defs");

} // namespace
