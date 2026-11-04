/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "include/TableGenUtils.h"

#include <sstream>

using namespace llvm;
namespace mlir {
namespace asc {
StringRef fetchOpClass(StringRef defName)
{
    auto split = defName.rsplit('_');
    if (split.second.empty()) {
        return defName;
    }
    return split.second;
}

void fetchResults(const DagInit* resultsDag, std::vector<VirtualArg>& dest)
{
    const auto* outsOp = dyn_cast<DefInit>(resultsDag->getOperator());
    assert(outsOp && outsOp->getDef()->getName() == "outs");
    for (unsigned i = 0, e = resultsDag->getNumArgs(); i < e; ++i) {
        VirtualArg result;
        auto name = resultsDag->getArgNameStr(i);
        if (name.empty()) {
            result.name = "arg" + std::to_string(i);
        } else {
            result.name = name;
        }
        result.substitution = result.name;
        const auto* init = dyn_cast<DefInit>(resultsDag->getArg(i));
        assert(init && "argument must have defined types");
        const auto* resultDef = init->getDef();
        if (resultDef->isSubClassOf("Res"))
            resultDef = resultDef->getValueAsDef("constraint");
        if (resultDef->isSubClassOf("Variadic")) {
            result.cppType = "::std::vector< ::mlir::Type >";
        } else {
            result.cppType = "::mlir::Type";
        }
        dest.push_back(result);
    }
}

void fetchArguments(const DagInit* argsDag, std::vector<VirtualArg>& dest)
{
    const auto* insOp = dyn_cast<DefInit>(argsDag->getOperator());
    assert(insOp && insOp->getDef()->getName() == "ins");
    for (unsigned i = 0, e = argsDag->getNumArgs(); i < e; ++i) {
        VirtualArg arg;
        auto name = argsDag->getArgNameStr(i);
        if (name.empty()) {
            arg.name = "arg" + std::to_string(i);
        } else {
            arg.name = name;
        }
        arg.substitution = arg.name;
        const auto* init = dyn_cast<DefInit>(argsDag->getArg(i));
        assert(init && "argument must have defined types");
        const auto* argDef = init->getDef();
        if (argDef->isSubClassOf("Arg"))
            argDef = argDef->getValueAsDef("constraint");
        if (argDef->isSubClassOf("TypeConstraint")) {
            if (argDef->isSubClassOf("Variadic")) {
                arg.cppType = "::std::vector< ::mlir::Value >";
            } else {
                arg.cppType = "::mlir::Value";
            }
            if (argDef->isSubClassOf("Optional")) {
                arg.optional = true;
                std::stringstream str;
                str << arg.name << ".value_or(" << arg.cppType << "{})";
                arg.substitution = str.str();
                arg.cppType = "::std::optional< " + arg.cppType + " >";
                arg.defaultValue = "py::none()";
            }
        } else if (argDef->isSubClassOf("AttrConstraint")) {
            arg.cppType = argDef->getValueAsString("returnType");
            if (argDef->getName() == "UnitAttr") {
                arg.optional = true;
                arg.substitution = arg.name;
                arg.defaultValue = "false";
            } else if (argDef->isSubClassOf("OptionalAttr")) {
                arg.optional = true;
                std::stringstream str;
                str << arg.name << ".value_or(" << argDef->getValueAsString("storageType").str() << "{})";
                arg.substitution = str.str();
                arg.defaultValue = "py::none()";
            }
        }
        dest.push_back(arg);
    }
}

StringRef removeDialectPrefix(StringRef fullName, StringRef dialectName)
{
    std::string prefix = (dialectName + "_").str();
    if (fullName.starts_with(prefix)) {
        return fullName.substr(dialectName.size() + 1);
    }
    return fullName;
}

StringRef removeAscDialectNameSpace(StringRef fullName, StringRef ascCppNamespace)
{
    std::string prefix = ascCppNamespace.str();
    if (fullName.starts_with(prefix)) {
        return fullName.substr(ascCppNamespace.size());
    }
    return fullName;
}

} // namespace asc
} // namespace mlir
