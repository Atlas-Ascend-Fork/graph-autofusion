/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "artifact/KernelSource/GenAggregateKernelBody.h"
#include "dialect/AFIR/IR/AFIRAttributes.h"
#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>

using namespace mlir::ascend::kernel;

using namespace mlir;
using namespace mlir::afir;

namespace {

// 候选身份是一个 uniqued 的属性，匹配即一次相等比较。
ModuleOp findPlanModuleByCandidate(ArrayRef<ModuleOp> planModules, afplan::CandidateAttr wanted) {
  for (ModuleOp module : planModules) {
    if (afplan::getCandidate(module) == wanted) {
      return module;
    }
  }
  return {};
}

// Local reimplementation of ascgen_utils::CamelToLowerSneak (autofuse/common/common_utils.cpp).
// Kept local instead of linked so AFIRToAscendC does not gain a reverse dependency on autofuse/common.
std::string camelToLowerSnake(const std::string &str) {
  std::string s1 = std::regex_replace(str, std::regex("(.)([A-Z][a-z]+)"), "$1_$2");
  std::string s2 = std::regex_replace(s1, std::regex("([a-z0-9])([A-Z])"), "$1_$2");
  std::transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  return s2;
}

// Local reimplementation of ascgen_utils::GenValidName (autofuse/common/common_utils.cpp).
std::string genValidName(const std::string &name) {
  std::string result;
  bool lastWasUnderscore = false;
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
      result += c;
      lastWasUnderscore = false;
    } else if (!lastWasUnderscore) {
      result += '_';
      lastWasUnderscore = true;
    }
  }
  if (!result.empty() && result.front() == '_') {
    result = result.substr(1);
  }
  if (!result.empty() && std::isdigit(static_cast<unsigned char>(result.front())) != 0) {
    result = "t_" + result;
  }
  return result;
}

// Plan _main calls its stage/template functions rather than inlining their bodies.
// Clone these callees with _main to avoid undefined symbols in entryModule.
// Different plans may reuse names such as stage0_template0, so qualify names within
// each source plan first. SymbolTable::rename updates _main calls before cloning.
LogicalResult relocateTemplateFunctions(ModuleOp planModule, ModuleOp entryModule, StringRef mainName) {
  if (planModule->hasAttr(mlir::ascend::kernel::kAscendCKernelBodyTemplatesRelocatedAttrName)) {
    return success();
  }
  SymbolTable planSymbols(planModule);
  SmallVector<func::FuncOp> templateFuncs;
  for (func::FuncOp candidate : planModule.getOps<func::FuncOp>()) {
    if (candidate.getName() != mainName) {
      templateFuncs.push_back(candidate);
    }
  }
  for (func::FuncOp templateFunc : templateFuncs) {
    std::string qualifiedName = planModule.getSymName()->str() + "_" + templateFunc.getName().str();
    if (failed(planSymbols.rename(templateFunc, qualifiedName))) {
      return templateFunc.emitError("GenAggregateKernelBody: failed to relocate template function ")
             << templateFunc.getName();
    }
  }
  func::FuncOp declaration = entryModule.lookupSymbol<func::FuncOp>(mainName);
  if (!declaration) {
    return entryModule.emitError("GenAggregateKernelBody: missing aggregate declaration for ") << mainName;
  }
  OpBuilder builder(entryModule.getBodyRegion());
  builder.setInsertionPoint(declaration);
  for (func::FuncOp templateFunc : templateFuncs) {
    builder.clone(*templateFunc.getOperation());
  }
  planModule->setAttr(mlir::ascend::kernel::kAscendCKernelBodyTemplatesRelocatedAttrName,
                      UnitAttr::get(planModule.getContext()));
  return success();
}

LogicalResult cloneDeclarationBody(ArrayRef<ModuleOp> planModules, ModuleOp entryModule, DictionaryAttr entry) {
  afplan::CandidateAttr candidate = afplan::getCandidate(entry);
  if (!candidate) {
    return entryModule.emitError("GenAggregateKernelBody: malformed ")
           << mlir::ascend::kernel::kAscendCPlanEntriesAttrName << " entry";
  }
  ModuleOp planModule = findPlanModuleByCandidate(planModules, candidate);
  auto symbol = planModule ? planModule.getSymName() : std::optional<StringRef>();
  if (!planModule || !symbol) {
    return entryModule.emitError("GenAggregateKernelBody: no Plan module for ") << candidate;
  }

  std::string mainName = symbol->str() + "_main";
  func::FuncOp declaration = entryModule.lookupSymbol<func::FuncOp>(mainName);
  if (!declaration || !declaration.isExternal()) {
    return entryModule.emitError("GenAggregateKernelBody: missing aggregate declaration for ") << mainName;
  }
  func::FuncOp definition = planModule.lookupSymbol<func::FuncOp>(mainName);
  if (!definition || definition.isExternal()) {
    return planModule.emitError("GenAggregateKernelBody: Plan module has no definition for ") << mainName;
  }
  if (failed(relocateTemplateFunctions(planModule, entryModule, mainName))) {
    return failure();
  }

  IRMapping mapping;
  definition.getFunctionBody().cloneInto(&declaration.getFunctionBody(), mapping);
  return success();
}

// Count input identities after GenAggregateKernelEntry has built the aggregate
// main. The aggregate arguments are already canonicalized, but counting the
// explicit identities keeps this logic independent of argument positions.
unsigned computeAggregateInputCount(func::FuncOp main) {
  unsigned count = 0;
  for (unsigned index = 0; index < main.getNumArguments(); ++index) {
    afir::KernelArgAttr argument = afir::getKernelArgAttr(main, index);
    // 按值标量也是 kernel 输入，一并计入——`ascendc.num_inputs` 数的是数据参数的个数，
    // 不是 GM 指针的个数。
    if (argument &&
        (argument.getRole() == afir::KernelArgRole::Input || argument.getRole() == afir::KernelArgRole::ScalarInput)) {
      ++count;
    }
  }
  return count;
}

LogicalResult promoteAggregateMain(ModuleOp entryModule, StringRef fusedGraphName) {
  func::FuncOp main = entryModule.lookupSymbol<func::FuncOp>("main");
  if (!main) {
    return entryModule.emitError("GenAggregateKernelBody: entry module has no aggregate main");
  }
  setAscendCKernelAttributes(main, afir::FunctionRole::KernelEntry, computeAggregateInputCount(main));
  afir::setFunctionRole(main, afir::FunctionRole::KernelEntry);

  std::string graphName = camelToLowerSnake(genValidName(fusedGraphName.str()));
  if (graphName.empty()) {
    return entryModule.emitError("GenAggregateKernelBody: fused graph name produced an empty symbol");
  }
  main.setName(graphName);
  return success();
}

}  // namespace

LogicalResult mlir::ascend::kernel::runGenAggregateKernelBody(ArrayRef<ModuleOp> planModules, ModuleOp entryModule,
                                                              StringRef fusedGraphName) {
  auto planEntries = entryModule->getAttrOfType<ArrayAttr>(mlir::ascend::kernel::kAscendCPlanEntriesAttrName);
  if (!planEntries) {
    return entryModule.emitError("GenAggregateKernelBody: entry module has no ")
           << mlir::ascend::kernel::kAscendCPlanEntriesAttrName << " contract";
  }
  for (Attribute attr : planEntries) {
    auto entry = dyn_cast<DictionaryAttr>(attr);
    if (!entry || failed(cloneDeclarationBody(planModules, entryModule, entry))) {
      return entryModule.emitError("GenAggregateKernelBody: invalid ")
             << mlir::ascend::kernel::kAscendCPlanEntriesAttrName << " entry";
    }
  }
  return promoteAggregateMain(entryModule, fusedGraphName);
}
