/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include "conversion/AFIRToAscendC/AFIRToAscendCPipeline.h"

#include "conversion/AFIRToAscendC/AFIRSessionScopedPass.h"
#include "conversion/AFIRToAscendC/RegisterAscendCImplementations.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "dialect/AFIR/Transforms/AFIRPasses.h"

#include <memory>
#include <utility>

#include "ascir/Dialect/Asc/IR/Asc.h"
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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir::afir;

namespace mlir::ascend::kernel {

FailureOr<AFIRToAscendCPipelineSession> createAFIRToAscendCPipelineSession(
    MLIRContext &context, std::shared_ptr<const ascend::TargetProfile> targetProfile) {
  if (!targetProfile) {
    emitError(UnknownLoc::get(&context), "AFIR to AscendC pipeline requires a target profile");
    return failure();
  }

  auto implementationRegistry = std::make_shared<ascend::instruction::ImplementationRegistry>(context);
  auto selectionPolicies = std::make_shared<ascend::instruction::SelectionPolicyRegistry>(context);
  if (failed(registerAscendCImplementations(*implementationRegistry, selectionPolicies.get())) ||
      failed(implementationRegistry->freeze()) || failed(selectionPolicies->freeze())) {
    emitError(UnknownLoc::get(&context), "failed to build the frozen AscendC implementation registry");
    return failure();
  }
  return AFIRToAscendCPipelineSession{std::move(targetProfile), std::move(implementationRegistry),
                                      std::move(selectionPolicies)};
}

namespace {

// 流水线按 IR 形态分为四个阶段，每个阶段的 add* 与 *Dialects 放在一起维护。
// AFIRPipelineDependentDialectsTest 对比方言清单与实际流水线的 getDependentDialects()，
// 检查遗漏的方言依赖。
//
//   A 准备      补齐 AscendC 参数表，将运行期尺寸符号物化为 SSA 值
//   B AFIR 物化  轴 -> scf.for、缓冲 -> 资源、顺序 -> 同步边
//   C 下降      选择实现，转换为 AscendC
//   D host 产物  生成每候选的 tiling schema 与入口分派
//
// B 的循环生成依赖 A 补齐的 TilingData 参数；C 需要先为 B 产生的同步算子选择实现
// （ADR-19）；D 的改名必须在最后一个 TilingData 消费者之后。

// 阶段 A。
void addKernelABIPreparation(OpPassManager &passManager) {
  passManager.addPass(createAFIRKernelBodySignaturePass());
  // 提前物化尺寸 SSA 值，固定尺寸序言的顺序，并集中报告无效尺寸表达式。
  // 这不是后续 pass 的必需前置：循环生成和资源物化均可通过 getOrCreateTilingParam
  // 补建缺失参数，LIT 也覆盖了省略此 pass 的路径。
  passManager.addPass(afir::createAFIRMaterializeTilingParamsPass());
}

void kernelABIPreparationDialects(DialectRegistry &registry) {
  registry
      .insert<AFIRDialect, arith::ArithDialect, emitasc::EmitAscDialect, func::FuncDialect, memref::MemRefDialect>();
}

// 阶段 B：不依赖 session，将 AFIR 转换为带已物化 AscendC 资源的 IR。
void addAFIRMaterialization(OpPassManager &passManager) {
  passManager.addPass(createAscendCLoopGeneratePass());
  // Scratch uses are scoped only after the actual loop tree is known. Physical IDs stay unchanged.
  passManager.addPass(afir::createAFIRScopeReduceScratchPass());
  passManager.addPass(createAFIRMemoryAllocatePass());
  // 同步必须先于实现选择：此 pass 产生 AFIR 同步算子（V -> V 屏障），需先选实现再下降。
  // 旧路径在 conversion 内、实现选择后运行，当时只产生 AscendC 算子。
  passManager.addPass(createAFIRPipelineSyncPass());
  // 缓存守卫只包围算子计算，资源生命周期操作和同步边仍须每轮执行。
  // 因此在资源物化和同步插入之后添加守卫，见 ExecConditionPass.cpp。
  passManager.addPass(afir::createAFIRExecConditionPass());
}

void afirMaterializationDialects(DialectRegistry &registry) {
  // workspace 偏移参数为 ui64，转换为 index 需要 emitc.cast；arith.index_cast
  // 不接受带符号性标记的整数。见 CanonicalMaterialization.cpp 的 materializeWorkspaceBase。
  registry.insert<AFIRDialect, arith::ArithDialect, ascendc::AscendCDialect, bufferization::BufferizationDialect,
                  cf::ControlFlowDialect, emitasc::EmitAscDialect, emitc::EmitCDialect, func::FuncDialect,
                  index::IndexDialect, math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
}

// 阶段 C：从 session 读取实现注册表和目标 profile。其他阶段不依赖 session。
void addAscendCLowering(OpPassManager &passManager, const AFIRToAscendCPipelineSession &session) {
  // 先将二次融合 region 转为寄存器形态，再选择 region 外算子的实现。
  // region 内算子仍为 tensor 形态，没有 destination，也没有 tile 级实现，直接参与
  // 选择会误报 "'afir.add' op has no supported Ascend implementation"。
  // 此步骤依赖资源物化产生的 !ascendc.local_tensor 边界张量，不能放在阶段 B 之前；
  // 它产生的 emitasc.vec_scope 需要后续生成，因此又必须先于 host 产物阶段。
  passManager.addPass(createAFIRVectorFuncToRegBasePass());
  passManager.addPass(ascend::instruction::createSelectImplementationPass(
      session.implementationRegistry, session.targetProfile, session.selectionPolicies, nullptr));
  passManager.addPass(createConvertAFIRToAscendCPass(session.implementationRegistry, session.targetProfile));
}

void ascendCLoweringDialects(DialectRegistry &registry) {
  registry.insert<AFIRDialect, arith::ArithDialect, ascendc::AscendCDialect, emitasc::EmitAscDialect, func::FuncDialect,
                  index::IndexDialect, math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
}

// 阶段 D。
void addHostArtifacts(OpPassManager &passManager) {
  passManager.addPass(createGenTilingDataPass());
  passManager.addPass(createGenKernelEntryPass());
}

void hostArtifactsDialects(DialectRegistry &registry) {
  // GenKernelEntry 在 template 调用点插入 memref.cast：_main 的 GM 参数按 ABI
  // 声明为 unranked，各 template 保留各自的调度视图。见 appendMainArgument。
  registry
      .insert<AFIRDialect, afplan::AFPlanDialect, arith::ArithDialect, ascendc::AscendCDialect, emitasc::EmitAscDialect,
              emitc::EmitCDialect, func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
}

// 检查手工构造的 session 是否完整。OpPassManager 没有 MLIRContext，两个 add* 入口
// 也不接收 context，因此这里只返回状态；正常创建路径由 createAFIRToAscendCPipelineSession
// 使用 context 报告诊断。
bool isUsableSession(const AFIRToAscendCPipelineSession &session) {
  return session.targetProfile && session.implementationRegistry && session.implementationRegistry->isFrozen() &&
         (!session.selectionPolicies || session.selectionPolicies->isFrozen());
}

}  // namespace

// 组合阶段 B 和 C，完成 AFIR 到 AscendC 的转换。
// 阶段 A 只准备 ABI 参数表，与后端 IR 形态无关，由调用方单独添加。
LogicalResult addAFIRToAscendCPipeline(OpPassManager &passManager, const AFIRToAscendCPipelineSession &session) {
  if (!isUsableSession(session)) {
    return failure();
  }
  addAFIRMaterialization(passManager);
  addAscendCLowering(passManager, session);
  return success();
}

LogicalResult addPlanCandidateLoweringPipeline(OpPassManager &passManager,
                                               const AFIRToAscendCPipelineSession &session) {
  if (!isUsableSession(session)) {
    return failure();
  }
  addKernelABIPreparation(passManager);
  addAFIRMaterialization(passManager);
  addAscendCLowering(passManager, session);
  addHostArtifacts(passManager);
  return success();
}

namespace {

struct AFIRToAscendCPipelinePass : public AFIRSessionScopedPass<AFIRToAscendCPipelinePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRToAscendCPipelinePass)

  StringRef getArgument() const final {
    return "afir-to-ascendc-pipeline";
  }

  StringRef getDescription() const final {
    return "Run the formal AFIR implementation selection and AscendC conversion pipeline";
  }

  LogicalResult populatePipeline(OpPassManager &pipeline, const AFIRToAscendCPipelineSession &session) {
    return addAFIRToAscendCPipeline(pipeline, session);
  }

  // 复用各阶段的方言清单，避免新增 pass 时遗漏独立维护的方言并集。
  void getStaticDependentDialects(DialectRegistry &registry) const {
    afirMaterializationDialects(registry);
    ascendCLoweringDialects(registry);
  }
};

struct AFIRPlanCandidateLoweringPipelinePass : public AFIRSessionScopedPass<AFIRPlanCandidateLoweringPipelinePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRPlanCandidateLoweringPipelinePass)

  StringRef getArgument() const final {
    return "afir-plan-candidate-lowering-pipeline";
  }

  StringRef getDescription() const final {
    return "Run the full per-candidate lowering pipeline (kernel body signature, tiling data, "
           "kernel entry, then the AFIR-to-AscendC pipeline)";
  }

  LogicalResult populatePipeline(OpPassManager &pipeline, const AFIRToAscendCPipelineSession &session) {
    return addPlanCandidateLoweringPipeline(pipeline, session);
  }

  void getStaticDependentDialects(DialectRegistry &registry) const {
    kernelABIPreparationDialects(registry);
    afirMaterializationDialects(registry);
    ascendCLoweringDialects(registry);
    hostArtifactsDialects(registry);
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRToAscendCPipelineStandalonePass() {
  return std::make_unique<AFIRToAscendCPipelinePass>();
}

void registerAFIRToAscendCPipelineStandalonePass() {
  PassRegistration<AFIRToAscendCPipelinePass>();
}

std::unique_ptr<Pass> createAFIRPlanCandidateLoweringPipelineStandalonePass() {
  return std::make_unique<AFIRPlanCandidateLoweringPipelinePass>();
}

void registerAFIRPlanCandidateLoweringPipelineStandalonePass() {
  PassRegistration<AFIRPlanCandidateLoweringPipelinePass>();
}

}  // namespace mlir::ascend::kernel
