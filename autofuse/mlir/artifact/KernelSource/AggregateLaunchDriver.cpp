/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "artifact/KernelSource/AggregateLaunchDriver.h"
#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "artifact/KernelSource/AddKernelIncludes.h"

#include <utility>

#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "artifact/KernelSource/Emit/AutofuseLaunchEmitter.h"
#include "artifact/KernelSource/GenAggregateKernelBody.h"
#include "artifact/KernelSource/GenAggregateKernelEntry.h"
#include "artifact/KernelSource/GenAggregateTilingData.h"
#include "artifact/KernelSource/Emit/TilingDataHeaderEmitter.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "ascir/Target/Asc/Translation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::afir;

namespace mlir::ascend::kernel {

// Aggregate planModules in order: tiling data, contract validation, kernel entry,
// then kernel body. This is the aggregation stage formerly in Codegen::CodegenViaMlir.
FailureOr<std::pair<OwningOpRef<ModuleOp>, AggregateArgumentLayout>> buildAggregateEntryModule(
    ArrayRef<ModuleOp> planModules, StringRef fusedGraphName, std::string *diagnostic) {
  auto entry = runGenAggregateTilingData(planModules);
  if (failed(entry)) {
    if (diagnostic) {
      *diagnostic = "aggregate tiling data generation failed";
    }
    return failure();
  }
  ModuleOp entryModule = **entry;
  if (failed(validateAggregateContract(planModules, entryModule))) {
    if (diagnostic) {
      *diagnostic = "aggregate contract validation failed";
    }
    return failure();
  }
  auto layout = runGenAggregateKernelEntry(planModules, entryModule);
  if (failed(layout)) {
    if (diagnostic) {
      *diagnostic = "aggregate kernel entry generation failed";
    }
    return failure();
  }
  if (failed(runGenAggregateKernelBody(planModules, entryModule, fusedGraphName))) {
    if (diagnostic) {
      *diagnostic = "aggregate kernel body generation failed";
    }
    return failure();
  }
  if (failed(runAddKernelIncludes(entryModule))) {
    if (diagnostic) {
      *diagnostic = "kernel include generation failed";
    }
    return failure();
  }
  return std::make_pair(std::move(*entry), *layout);
}

namespace {

// Translate entryModule to AscendC and emit its tiling header and AutofuseLaunch wrapper.
// This is the emission stage formerly in Codegen::CodegenViaMlir.
void setDiagnostic(std::string *diagnostic, const llvm::Twine &message) {
  if (diagnostic) {
    *diagnostic = message.str();
  }
}

FailureOr<func::FuncOp> findKernelEntry(ModuleOp entryModule, std::string *diagnostic) {
  for (func::FuncOp func : entryModule.getOps<func::FuncOp>()) {
    if (hasFunctionRole(func, FunctionRole::KernelEntry)) {
      return func;
    }
  }
  setDiagnostic(diagnostic, "no KernelEntry function found in aggregate module");
  return failure();
}

LogicalResult emitAggregateTilingData(ModuleOp entryModule, std::string &out, std::string *diagnostic) {
  auto schemaAttr = entryModule->getAttrOfType<TypeAttr>(mlir::ascend::kernel::kAscendCTilingSchemaAttrName);
  if (!schemaAttr) {
    setDiagnostic(diagnostic, "aggregate module has no " + kAscendCTilingSchemaAttrName.str() + " attribute");
    return failure();
  }
  auto pyStruct = getTilingSchema(schemaAttr.getValue());
  if (!pyStruct) {
    setDiagnostic(diagnostic, kAscendCTilingSchemaAttrName.str() + " attribute is not a valid tiling schema");
    return failure();
  }
  llvm::raw_string_ostream tilingDataOs(out);
  if (failed(emitTilingDataHeader({pyStruct}, tilingDataOs))) {
    setDiagnostic(diagnostic, "emitTilingDataHeader failed");
    return failure();
  }
  tilingDataOs.flush();
  return success();
}

FailureOr<AggregateLaunchArtifacts> EmitArtifactsFromEntryModule(ModuleOp entryModule,
                                                                 const AggregateArgumentLayout &layout,
                                                                 std::string *diagnostic) {
  PassManager cleanup(entryModule.getContext());
  cleanup.addPass(createCSEPass());
  if (failed(cleanup.run(entryModule))) {
    setDiagnostic(diagnostic, "aggregate module cleanup failed");
    return failure();
  }

  FailureOr<func::FuncOp> kernelEntryFunc = findKernelEntry(entryModule, diagnostic);
  if (failed(kernelEntryFunc)) {
    return failure();
  }
  const std::string graphName = kernelEntryFunc->getName().str();
  ArrayRef<Type> kernelArgumentTypes = kernelEntryFunc->getArgumentTypes();
  // compile_adapter's tiling_repr rewrite removes "AutofuseTilingData t" and
  // inserts a constexpr t. All uses must carry this same ABI parameter name.
  kernelEntryFunc->setArgAttr(kernelEntryFunc->getNumArguments() - 1, "ascendc.parameter_name",
                              StringAttr::get(entryModule.getContext(), "t"));

  AggregateLaunchArtifacts artifacts;
  llvm::raw_string_ostream kernelBodyOs(artifacts.kernel_body);
  if (failed(translateToAscendC(entryModule, kernelBodyOs))) {
    setDiagnostic(diagnostic, "translateToAscendC failed");
    return failure();
  }
  kernelBodyOs.flush();

  if (failed(emitAggregateTilingData(entryModule, artifacts.tiling_data, diagnostic))) {
    return failure();
  }
  auto schemaAttr = entryModule->getAttrOfType<TypeAttr>(kAscendCTilingSchemaAttrName);
  llvm::raw_string_ostream reprOs(artifacts.tiling_data_repr);
  if (!schemaAttr || failed(emitTilingDataRepr(getTilingSchema(schemaAttr.getValue()), reprOs))) {
    setDiagnostic(diagnostic, "emitTilingDataRepr failed");
    return failure();
  }
  reprOs.flush();

  constexpr llvm::StringLiteral kTilingTypeName = "AutofuseTilingData";
  auto launchText = emitAutofuseLaunch(layout, kernelArgumentTypes.drop_back(), graphName, kTilingTypeName);
  if (failed(launchText)) {
    setDiagnostic(diagnostic, "emitAutofuseLaunch failed");
    return failure();
  }
  artifacts.autofuse_launch = *launchText;
  return artifacts;
}

}  // namespace

FailureOr<AggregateLaunchArtifacts> runAggregateLaunchDriver(ArrayRef<ModuleOp> planModules, StringRef fusedGraphName,
                                                             std::string *diagnostic) {
  auto entry = buildAggregateEntryModule(planModules, fusedGraphName, diagnostic);
  if (failed(entry)) {
    return failure();
  }
  return EmitArtifactsFromEntryModule(entry->first.get(), entry->second, diagnostic);
}

}  // namespace mlir::ascend::kernel
