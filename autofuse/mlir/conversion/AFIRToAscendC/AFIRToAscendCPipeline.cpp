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

// 流水线的四个相位。每个相位有一个 `add*` 与一个 `*Dialects`，写在一起，往相位里加一个
// pass 时两处都在眼前。方言清单最终由 `AFIRPipelineDependentDialectsTest` 与真实流水线
// 的 `OpPassManager::getDependentDialects()` 对账，漏写会变红。
//
// 相位划分的依据是 IR 的形态，不是调用方：
//
//   A 准备      补齐 AscendC 的参数表，把全部运行期尺寸符号变成 SSA 值
//   B AFIR 物化  轴 -> scf.for、缓冲 -> 资源、顺序 -> 同步边
//   C 下降      选实现，转成 AscendC
//   D host 产物  每候选的 tiling schema 与入口分派
//
// 相邻相位之间是真实约束，不是约定：B 的第一步要用 A 补出的 TilingData 参数，C 要 B 产出
// 的同步算子先被选出实现（ADR-19），D 的改名必须在最后一个 TilingData 消费者之后。

// 相位 A。
void addKernelABIPreparation(OpPassManager &passManager) {
  passManager.addPass(createAFIRKernelBodySignaturePass());
  // 尺寸的 SSA 化。这一步**不是**后面几步的前置——`getOrCreateTilingParam` 是
  // get-or-create，循环生成与资源物化都会在缺的时候自己建（LIT 里有不跑这一步的 RUN 行）。
  // 放在这里是为了两件事：尺寸序言的顺序与来源无关地固定下来，以及坏的尺寸表达式在一处
  // 一次性报出来，而不是等到某个消费者恰好第一个碰到它。
  passManager.addPass(afir::createAFIRMaterializeTilingParamsPass());
}

void kernelABIPreparationDialects(DialectRegistry &registry) {
  registry
      .insert<AFIRDialect, arith::ArithDialect, emitasc::EmitAscDialect, func::FuncDialect, memref::MemRefDialect>();
}

// 相位 B。这三步都不需要 session：它们读 AFIR，产物是 AFIR 加已物化的 AscendC 资源。
void addAFIRMaterialization(OpPassManager &passManager) {
  passManager.addPass(createAscendCLoopGeneratePass());
  // Scratch uses are scoped only after the actual loop tree is known. Physical IDs stay unchanged.
  passManager.addPass(afir::createAFIRScopeReduceScratchPass());
  passManager.addPass(createAFIRMemoryAllocatePass());
  // 同步排在实现选择**之前**：它现在产出 AFIR 的同步算子（V -> V 屏障），而 AFIR 算子要
  // 先被选出实现才能下降。此前它跑在 conversion 内部、选择之后，那时产物全是 AscendC
  // 算子，不经过选择这条路。
  passManager.addPass(createAFIRPipelineSyncPass());
  // 缓存守卫排在最后：被它包起来的只有算子本身，生命周期与同步边每趟照发，所以它必须
  // 看到已经物化完的资源与已经插好的同步（ExecConditionPass.cpp 顶部那段）。
  passManager.addPass(afir::createAFIRExecConditionPass());
}

void afirMaterializationDialects(DialectRegistry &registry) {
  // `emitc`：workspace 张量的偏移形参是 `ui64`，转 `index` 只能走 `emitc.cast`
  // （`arith.index_cast` 不吃带符号性标记的整数），见 `CanonicalMaterialization.cpp` 的
  // `materializeWorkspaceBase`。
  registry.insert<AFIRDialect, arith::ArithDialect, ascendc::AscendCDialect, bufferization::BufferizationDialect,
                  cf::ControlFlowDialect, emitasc::EmitAscDialect, emitc::EmitCDialect, func::FuncDialect,
                  index::IndexDialect, math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
}

// 相位 C。唯一需要 session 的相位——实现注册表与目标 profile 都在里面。
void addAscendCLowering(OpPassManager &passManager, const AFIRToAscendCPipelineSession &session) {
  // 二次融合的 region **排在实现选择之前**。region 里是 tensor 形态的算子，它们没有
  // destination、也没有 tile 级的实现可选——选型走进去的结果是
  // 「'afir.add' op has no supported Ascend implementation」，一条看起来像「这个 op 没接」
  // 的诊断，而真正的事实是「它不该被问」。这一步把整段 region 换成寄存器形态之后，
  // 后面两步看到的就只剩 region 外的算子了。
  //
  // 它放在这一相位而不是相位 B，是因为它要的边界张量（`!ascendc.local_tensor`）由
  // 资源物化产出；放在 host 产物那一相位之前，因为它产出的 `emitasc.vec_scope` 要被发射。
  passManager.addPass(createAFIRVectorFuncToRegBasePass());
  passManager.addPass(ascend::instruction::createSelectImplementationPass(
      session.implementationRegistry, session.targetProfile, session.selectionPolicies, nullptr));
  passManager.addPass(createConvertAFIRToAscendCPass(session.implementationRegistry, session.targetProfile));
}

void ascendCLoweringDialects(DialectRegistry &registry) {
  registry.insert<AFIRDialect, arith::ArithDialect, ascendc::AscendCDialect, emitasc::EmitAscDialect, func::FuncDialect,
                  index::IndexDialect, math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
}

// 相位 D。
void addHostArtifacts(OpPassManager &passManager) {
  passManager.addPass(createGenTilingDataPass());
  passManager.addPass(createGenKernelEntryPass());
}

void hostArtifactsDialects(DialectRegistry &registry) {
  // `memref`：`GenKernelEntry` 在调用 template 的点上插 `memref.cast`——`_main` 的 GM 形参按 ABI
  // 形态声明（无条件退到 unranked），各 template 保留自己的调度视图，理由见 `appendMainArgument`。
  registry
      .insert<AFIRDialect, afplan::AFPlanDialect, arith::ArithDialect, ascendc::AscendCDialect, emitasc::EmitAscDialect,
              emitc::EmitCDialect, func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
}

// session 的完整性。诊断发不出来——`OpPassManager` 上没有 MLIRContext，这两个 `add*`
// 入口也不收一个。发诊断的是调用方：session 由 `createAFIRToAscendCPipelineSession` 建，
// 那里有 context；本函数只挡住手工拼出来的 session。
bool isUsableSession(const AFIRToAscendCPipelineSession &session) {
  return session.targetProfile && session.implementationRegistry && session.implementationRegistry->isFrozen() &&
         (!session.selectionPolicies || session.selectionPolicies->isFrozen());
}

}  // namespace

// 相位 B + C。名字里的「AFIR 到 AscendC」说的正是这两步合起来的效果：进来是 AFIR，出去是
// AscendC。相位 A 不在里面——它补的是 ABI 参数表，与后端形态无关的准备工作。
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

  // 按相位拼，不再手写并集。手写的那份已经漂过一次：注释声称是四个 pass 的并集，而
  // `MaterializeTilingParams` 与 `PipelineSync` 后来加进流水线时没人回来改它。
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
