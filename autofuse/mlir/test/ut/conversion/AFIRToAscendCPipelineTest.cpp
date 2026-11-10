/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include "conversion/AFIRToAscendC/AFIRToAscendCPipeline.h"
#include "conversion/AFIRToAscendC/RegisterAscendCImplementations.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"
#include "target/Ascend/AscendTargetProfile.h"
#include "target/Ascend/TargetProfileLoader.h"

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/StringSet.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace mlir::afir {
using namespace mlir::ascend::kernel;
namespace {

std::shared_ptr<const ascend::TargetProfile> CreateTargetProfile() {
  ascend::TargetProfile target({{"AscendSynthetic", "AscendTest", ascend::NpuArch::Unknown},
                                {8, 4, 4, ascend::CubeVectorTopology::Split},
                                {},
                                {},
                                {},
                                {},
                                {}},
                               "fingerprint");
  return std::make_shared<const ascend::TargetProfile>(std::move(target));
}

std::shared_ptr<const ascend::TargetProfile> CreateAscend910B1Profile() {
  llvm::raw_null_ostream diagnostic;
  auto profile = ascend::TargetProfileLoader::loadBuiltin("Ascend910B1", diagnostic);
  if (failed(profile)) {
    return nullptr;
  }
  return std::make_shared<const ascend::TargetProfile>(std::move(*profile));
}

void LoadDialects(MLIRContext &context) {
  context.getOrLoadDialect<afir::AFIRDialect>();
  context.getOrLoadDialect<afplan::AFPlanDialect>();
  context.getOrLoadDialect<ascendc::AscendCDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
}

void LoadPlanCandidateDialects(MLIRContext &context) {
  LoadDialects(context);
  context.getOrLoadDialect<bufferization::BufferizationDialect>();
  context.getOrLoadDialect<emitasc::EmitAscDialect>();
  context.getOrLoadDialect<emitc::EmitCDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
}

OwningOpRef<ModuleOp> ParsePlanCandidateModule(MLIRContext &context) {
  return parseSourceString<ModuleOp>(R"MLIR(
#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#output_desc = #afir.tensor_desc<axis = [0], role = <output>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>
#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>
#load_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>
#output_queue = #afir.buffer_desc<space = <ub>, buffer_id = 3, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %input: memref<16xf16> {afir.buffer_desc = #gm_buffer, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>},
      %output: memref<16xf16> {afir.buffer_desc = #gm_buffer, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>}) attributes {
    afir.arg_descs = [#input_desc, #output_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 10,
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %load_tile = memref.alloc() {afir.buffer_desc = #load_buffer} : memref<16xf16>
    %compute_tile = memref.alloc() {afir.buffer_desc = #output_queue} : memref<16xf16>
    %loaded = afir.load %input outs(%load_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#input_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed = afir.add %loaded, %loaded outs(%compute_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#compute_desc, #compute_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %computed outs(%output) {
      afir.destination_descs = [#output_desc],
      afir.operand_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}
)MLIR",
                                     &context);
}

OwningOpRef<ModuleOp> ParseBufferizedAdd(MLIRContext &context, bool hasSelection) {
  const char *selection = hasSelection ? R"(,
      "ascend.implementation" = {kind = "ascendc_api", id = "add.tensor_tensor"})"
                                       : "";
  std::string ir = R"(module {
  func.func @test(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>,
                  %destination: !ascendc.local_tensor<?xf16>) {
    %result = afir.add %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#afir.tensor_desc<role = <input>>, #afir.tensor_desc<role = <input>>],
      afir.destination_descs = [#afir.tensor_desc<role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<role = <compute>>])";
  ir += selection;
  ir += R"(}
      )";
  ir += R"( : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }
})";
  return parseSourceString<ModuleOp>(ir, &context);
}

OwningOpRef<ModuleOp> ParseTensorAdd(MLIRContext &context) {
  return parseSourceString<ModuleOp>(R"(module {
  func.func @test(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>) -> tensor<4xf16> {
    %result = afir.add %lhs, %rhs : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
})",
                                     &context);
}

std::shared_ptr<ascend::instruction::ImplementationRegistry> CreateFrozenRegistry(MLIRContext &context) {
  auto registry = std::make_shared<ascend::instruction::ImplementationRegistry>(context);
  if (failed(registerAscendCImplementations(*registry)) || failed(registry->freeze())) {
    return nullptr;
  }
  return registry;
}

LogicalResult RunConversionPass(ModuleOp module,
                                std::shared_ptr<const ascend::instruction::ImplementationRegistry> registry,
                                std::shared_ptr<const ascend::TargetProfile> target) {
  PassManager passManager(module.getContext());
  passManager.addPass(createConvertAFIRToAscendCPass(std::move(registry), std::move(target)));
  return passManager.run(module);
}

TEST(AFIRToAscendCPipelineTest, RejectsNullTargetWhenCreatingSession) {
  MLIRContext context;
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });

  EXPECT_TRUE(failed(createAFIRToAscendCPipelineSession(context, nullptr)));
}

TEST(AFIRToAscendCPipelineTest, CreatesSessionWithFrozenProductRegistry) {
  MLIRContext context;
  LoadDialects(context);
  auto target = CreateTargetProfile();

  auto session = createAFIRToAscendCPipelineSession(context, target);

  ASSERT_TRUE(succeeded(session));
  EXPECT_EQ(session->targetProfile, target);
  ASSERT_TRUE(session->implementationRegistry);
  EXPECT_TRUE(session->implementationRegistry->isFrozen());
  EXPECT_TRUE(session->implementationRegistry->containsSourceOp(OperationName(AddOp::getOperationName(), &context)));
  ASSERT_TRUE(session->selectionPolicies);
  EXPECT_TRUE(session->selectionPolicies->isFrozen());
}

TEST(AFIRToAscendCPipelineTest, RejectsIncompleteOrUnfrozenSession) {
  MLIRContext context;
  auto target = CreateTargetProfile();
  auto unfrozenRegistry = std::make_shared<ascend::instruction::ImplementationRegistry>(context);

  PassManager emptySessionPipeline(&context);
  EXPECT_TRUE(failed(addAFIRToAscendCPipeline(emptySessionPipeline, {})));

  PassManager missingRegistryPipeline(&context);
  EXPECT_TRUE(failed(addAFIRToAscendCPipeline(missingRegistryPipeline, AFIRToAscendCPipelineSession{target, nullptr})));

  PassManager missingTargetPipeline(&context);
  EXPECT_TRUE(
      failed(addAFIRToAscendCPipeline(missingTargetPipeline, AFIRToAscendCPipelineSession{nullptr, unfrozenRegistry})));

  PassManager unfrozenRegistryPipeline(&context);
  EXPECT_TRUE(failed(
      addAFIRToAscendCPipeline(unfrozenRegistryPipeline, AFIRToAscendCPipelineSession{target, unfrozenRegistry})));
}

TEST(AFIRToAscendCPipelineTest, SelectionAndConversionPipelineLowersBufferizedAdd) {
  MLIRContext context;
  LoadDialects(context);
  OwningOpRef<ModuleOp> module = ParseBufferizedAdd(context, /*hasSelection=*/false);
  ASSERT_TRUE(module);
  auto session = createAFIRToAscendCPipelineSession(context, CreateTargetProfile());
  ASSERT_TRUE(succeeded(session));
  PassManager passManager(&context);
  ASSERT_TRUE(succeeded(addAFIRToAscendCPipeline(passManager, *session)));

  ASSERT_TRUE(succeeded(passManager.run(*module)));

  size_t sourceCount = 0;
  size_t getSizeCount = 0;
  size_t targetCount = 0;
  module->walk([&](AddOp) { ++sourceCount; });
  module->walk([&](ascendc::LocalTensorGetSizeOp) { ++getSizeCount; });
  module->walk([&](ascendc::AddL2Op op) {
    ++targetCount;
    EXPECT_EQ(op->getAttrOfType<StringAttr>("ascendc.unit").getValue(), "AiCore.Vector");
  });
  EXPECT_EQ(sourceCount, 0U);
  EXPECT_EQ(getSizeCount, 1U);
  EXPECT_EQ(targetCount, 1U);
}

TEST(AFIRToAscendCPipelineTest, PlanCandidateLoweringPipelineProducesDispatcherAndLowersKernelBody) {
  MLIRContext context;
  LoadPlanCandidateDialects(context);
  OwningOpRef<ModuleOp> module = ParsePlanCandidateModule(context);
  ASSERT_TRUE(module);
  auto target = CreateAscend910B1Profile();
  ASSERT_TRUE(target);
  auto session = createAFIRToAscendCPipelineSession(context, target);
  ASSERT_TRUE(succeeded(session));
  PassManager passManager(&context);
  ASSERT_TRUE(succeeded(addPlanCandidateLoweringPipeline(passManager, *session)));

  ASSERT_TRUE(succeeded(passManager.run(*module)));

  EXPECT_TRUE(module->getOperation()->hasAttr("ascendc.tiling_schema"));

  size_t loadCount = 0;
  size_t addCount = 0;
  size_t storeCount = 0;
  size_t addL2Count = 0;
  size_t switchCount = 0;
  module->walk([&](LoadOp) { ++loadCount; });
  module->walk([&](AddOp) { ++addCount; });
  module->walk([&](StoreOp) { ++storeCount; });
  module->walk([&](ascendc::AddL2Op) { ++addL2Count; });
  module->walk([&](scf::IndexSwitchOp) { ++switchCount; });
  EXPECT_EQ(loadCount, 0U);
  EXPECT_EQ(addCount, 0U);
  EXPECT_EQ(storeCount, 0U);
  EXPECT_EQ(addL2Count, 1U);
  EXPECT_EQ(switchCount, 1U);

  func::FuncOp mainFunc = module->lookupSymbol<func::FuncOp>("plan0_main");
  EXPECT_TRUE(mainFunc);
}

TEST(AFIRToAscendCConversionPassTest, RejectsMissingOrUnfrozenDependencies) {
  MLIRContext context;
  LoadDialects(context);
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  auto target = CreateTargetProfile();
  auto frozenRegistry = CreateFrozenRegistry(context);
  ASSERT_TRUE(frozenRegistry);
  auto unfrozenRegistry = std::make_shared<ascend::instruction::ImplementationRegistry>(context);

  OwningOpRef<ModuleOp> missingRegistryModule = ParseBufferizedAdd(context, /*hasSelection=*/true);
  OwningOpRef<ModuleOp> missingTargetModule = ParseBufferizedAdd(context, /*hasSelection=*/true);
  OwningOpRef<ModuleOp> unfrozenRegistryModule = ParseBufferizedAdd(context, /*hasSelection=*/true);
  ASSERT_TRUE(missingRegistryModule);
  ASSERT_TRUE(missingTargetModule);
  ASSERT_TRUE(unfrozenRegistryModule);

  EXPECT_TRUE(failed(RunConversionPass(*missingRegistryModule, nullptr, target)));
  EXPECT_TRUE(failed(RunConversionPass(*missingTargetModule, frozenRegistry, nullptr)));
  EXPECT_TRUE(failed(RunConversionPass(*unfrozenRegistryModule, unfrozenRegistry, target)));
}

TEST(AFIRToAscendCConversionPassTest, RequiresSelectionForBufferizedAdd) {
  MLIRContext context;
  LoadDialects(context);
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  OwningOpRef<ModuleOp> module = ParseBufferizedAdd(context, /*hasSelection=*/false);
  ASSERT_TRUE(module);
  auto registry = CreateFrozenRegistry(context);
  ASSERT_TRUE(registry);

  EXPECT_TRUE(failed(RunConversionPass(*module, registry, CreateTargetProfile())));
}

TEST(AFIRToAscendCConversionPassTest, RewritesSelectedBufferizedAdd) {
  MLIRContext context;
  LoadDialects(context);
  OwningOpRef<ModuleOp> module = ParseBufferizedAdd(context, /*hasSelection=*/true);
  ASSERT_TRUE(module);
  auto registry = CreateFrozenRegistry(context);
  ASSERT_TRUE(registry);

  ASSERT_TRUE(succeeded(RunConversionPass(*module, registry, CreateTargetProfile())));

  size_t sourceCount = 0;
  size_t getSizeCount = 0;
  size_t targetCount = 0;
  module->walk([&](AddOp) { ++sourceCount; });
  module->walk([&](ascendc::LocalTensorGetSizeOp) { ++getSizeCount; });
  module->walk([&](ascendc::AddL2Op) { ++targetCount; });
  EXPECT_EQ(sourceCount, 0U);
  EXPECT_EQ(getSizeCount, 1U);
  EXPECT_EQ(targetCount, 1U);
}

TEST(AFIRToAscendCConversionPassTest, RejectsTensorFormAddResidual) {
  MLIRContext context;
  LoadDialects(context);
  OwningOpRef<ModuleOp> module = ParseTensorAdd(context);
  ASSERT_TRUE(module);
  auto registry = CreateFrozenRegistry(context);
  ASSERT_TRUE(registry);

  EXPECT_TRUE(failed(RunConversionPass(*module, registry, CreateTargetProfile())));
}

// 两个 pipeline pass 的 `getDependentDialects` 是按阶段拼出来的（AFIRToAscendCPipeline.cpp
// 的 `*Dialects`），不是手写的并集——但每个阶段的那一行仍是人写的。这条用例是防它漂：
// 拿真实流水线自己报的依赖方言，与 pass 声明的那份比对。
//
// 为什么非有不可：`PassManager::run()` 在 `initialize()` 之前就把依赖方言加载进
// context 了，而这两个 pass 的子流水线只在 `initialize()` 里才建出来，所以声明漏了一个
// 方言不会在这里失败——它会在某条恰好用到那个方言的路径上，以一句「dialect not
// registered」的形式出现在很远的地方。手写的那份并集已经漂过一次。
void ExpectDeclaredDialectsCoverPipeline(std::unique_ptr<Pass> pass, OpPassManager &pipeline) {
  DialectRegistry declared;
  pass->getDependentDialects(declared);
  llvm::StringSet<> declaredNames;
  for (llvm::StringRef name : declared.getDialectNames()) {
    declaredNames.insert(name);
  }
  DialectRegistry actual;
  pipeline.getDependentDialects(actual);
  for (llvm::StringRef name : actual.getDialectNames()) {
    EXPECT_TRUE(declaredNames.contains(name))
        << "pipeline depends on dialect '" << name.str()
        << "' but the pass does not declare it; add it to the phase's *Dialects in AFIRToAscendCPipeline.cpp";
  }
}

TEST(AFIRToAscendCPipelineTest, DeclaredDependentDialectsCoverTheRealPipeline) {
  MLIRContext context;
  LoadPlanCandidateDialects(context);
  auto target = CreateAscend910B1Profile();
  ASSERT_TRUE(target);
  auto session = createAFIRToAscendCPipelineSession(context, target);
  ASSERT_TRUE(succeeded(session));

  OpPassManager inner(ModuleOp::getOperationName());
  ASSERT_TRUE(succeeded(addAFIRToAscendCPipeline(inner, *session)));
  ExpectDeclaredDialectsCoverPipeline(createAFIRToAscendCPipelineStandalonePass(), inner);

  OpPassManager full(ModuleOp::getOperationName());
  ASSERT_TRUE(succeeded(addPlanCandidateLoweringPipeline(full, *session)));
  ExpectDeclaredDialectsCoverPipeline(createAFIRPlanCandidateLoweringPipelineStandalonePass(), full);
}

}  // namespace
}  // namespace mlir::afir
