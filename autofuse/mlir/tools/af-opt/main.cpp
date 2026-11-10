/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/AscTransformsPasses.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include "artifact/KernelSource/AggregateLaunchDriver.h"
#include "artifact/KernelSource/KernelSourcePasses.h"
#include "dialect/AFIR/Transforms/AFIRPasses.h"
#include "target/Ascend/TargetProfileRegistry.h"
#include "target/Ascend/TargetSelection.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

namespace {
constexpr const char *kAutofuseMlirVersion = "af-mlir-phase1";

llvm::cl::opt<bool> afirAggregateLaunchOption(
    "afir-aggregate-launch",
    llvm::cl::desc("Aggregate per-candidate plan modules already lowered via "
                   "--afir-plan-candidate-lowering-pipeline into the kernel body, tiling-data header, "
                   "and AutofuseLaunch host wrapper text artifacts"));
llvm::cl::opt<std::string> afirAggregateLaunchGraphNameOption(
    "afir-aggregate-launch-graph-name", llvm::cl::desc("Fused kernel entry name for --afir-aggregate-launch"),
    llvm::cl::value_desc("name"), llvm::cl::init("fused_graph"));

void PrintAutofuseMlirVersion(llvm::raw_ostream &os) {
  os << "  Autofuse MLIR version " << kAutofuseMlirVersion << "\n";
}

// Validate --target/--target-profile before parsing input, even when no pipeline is requested.
// Report conflicting options, unknown or invalid platform names, and unreadable profiles here.
// Session-scoped passes obtain the actual TargetProfile during initialization through
// resolveSessionTargetProfile; this early check also covers bare af-opt invocations.
mlir::LogicalResult ValidateTargetSelectionOptions() {
  std::string diagnostic;
  llvm::raw_string_ostream diagnosticStream(diagnostic);
  if (failed(mlir::ascend::getSelectedTargetProfile(mlir::ascend::TargetSelectionRequirement::Optional,
                                                    diagnosticStream))) {
    llvm::errs() << diagnosticStream.str();
    return mlir::failure();
  }
  return mlir::success();
}

void RegisterDialectsAndPasses(mlir::DialectRegistry &registry) {
  registry
      .insert<mlir::afir::AFIRDialect, mlir::afplan::AFPlanDialect, mlir::arith::ArithDialect,
              mlir::ascendc::AscendCDialect, mlir::bufferization::BufferizationDialect, mlir::cf::ControlFlowDialect,
              mlir::emitasc::EmitAscDialect, mlir::emitc::EmitCDialect, mlir::func::FuncDialect,
              mlir::index::IndexDialect, mlir::math::MathDialect, mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
              mlir::tensor::TensorDialect, mlir::transform::TransformDialect>();
  mlir::ascendc::registerExternalModels(registry);
  mlir::ascendc::registerInlinerInterfaces(registry);
  mlir::emitasc::registerExternalModels(registry);
  mlir::registerascendcPasses();

  mlir::ascend::kernel::registerAFIRToAscendCPasses();
  mlir::ascend::kernel::registerKernelSourcePasses();
  mlir::afir::registerAFIRSymbolicPasses();
  mlir::afir::registerAFIRExecConditionPass();
  mlir::afir::registerAFIRRaceCheckPass();
  mlir::afir::registerAFIRScopeReduceScratchPass();
  mlir::afir::registerAFIRMaterializeTilingParamsPass();

  // 注册上游 transform 方言与 interpreter，用 transform_dialect_baseline.mlir
  // 验证当前 LLVM 基线可用且可扩展，为设计文档 3.4 的 Schedule 迁移提供基线证据。
  // 当前不注册自定义 transform.afir.* op。
  mlir::transform::registerTransformPasses();
}

// Parse the input as a ModuleOp container and return its direct child modules.
// parseSourceFile wraps sibling module @planN fixtures in one outer module, so the
// children match the per-candidate inputs expected by runAggregateLaunchDriver.
mlir::FailureOr<llvm::SmallVector<mlir::ModuleOp>> ParsePlanModules(llvm::StringRef inputFilename,
                                                                    mlir::MLIRContext &context,
                                                                    mlir::OwningOpRef<mlir::ModuleOp> &outerModule) {
  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << '\n';
    return mlir::failure();
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  mlir::ParserConfig parserConfig(&context);
  outerModule = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, parserConfig);
  if (!outerModule) {
    return mlir::failure();
  }
  llvm::SmallVector<mlir::ModuleOp> planModules;
  for (mlir::ModuleOp planModule : outerModule->getOps<mlir::ModuleOp>()) {
    planModules.push_back(planModule);
  }
  if (planModules.empty()) {
    planModules.push_back(*outerModule);
  }
  return planModules;
}

// Print each aggregate-launch artifact under a distinct marker so FileCheck can
// match one section without consuming text from another.
void PrintAggregateLaunchArtifacts(const mlir::ascend::kernel::AggregateLaunchArtifacts &artifacts,
                                   llvm::raw_ostream &os) {
  os << "// -----KERNEL_BODY-----\n" << artifacts.kernel_body;
  os << "// -----TILING_DATA-----\n" << artifacts.tiling_data;
  os << "// -----AUTOFUSE_LAUNCH-----\n" << artifacts.autofuse_launch;
}

mlir::LogicalResult RunAFIRAggregateLaunch(llvm::StringRef inputFilename, llvm::StringRef outputFilename,
                                           llvm::StringRef fusedGraphName, mlir::DialectRegistry &registry) {
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> outerModule;
  auto planModules = ParsePlanModules(inputFilename, context, outerModule);
  if (mlir::failed(planModules)) {
    return mlir::failure();
  }
  std::string diagnostic;
  auto artifacts = mlir::ascend::kernel::runAggregateLaunchDriver(*planModules, fusedGraphName, &diagnostic);
  if (mlir::failed(artifacts)) {
    llvm::errs() << "afir-aggregate-launch failed: " << diagnostic << '\n';
    return mlir::failure();
  }
  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << '\n';
    return mlir::failure();
  }
  PrintAggregateLaunchArtifacts(*artifacts, output->os());
  output->keep();
  return mlir::success();
}
}  // namespace

int main(int argc, char **argv) {
  llvm::cl::AddExtraVersionPrinter(PrintAutofuseMlirVersion);

  mlir::DialectRegistry registry;
  RegisterDialectsAndPasses(registry);

  auto [inputFilename, outputFilename] =
      mlir::registerAndParseCLIOptions(argc, argv, "Autofuse MLIR migration optimizer\n", registry);

  if (failed(ValidateTargetSelectionOptions())) {
    return EXIT_FAILURE;
  }

  if (afirAggregateLaunchOption) {
    return mlir::asMainReturnCode(
        RunAFIRAggregateLaunch(inputFilename, outputFilename, afirAggregateLaunchGraphNameOption, registry));
  }

  mlir::LogicalResult result = mlir::MlirOptMain(argc, argv, inputFilename, outputFilename, registry);
  return mlir::asMainReturnCode(result);
}
