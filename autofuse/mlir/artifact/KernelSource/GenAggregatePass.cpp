/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "artifact/KernelSource/KernelSourcePasses.h"

#include <string>

#include "artifact/KernelSource/AggregateLaunchDriver.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/PlanMetadata.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir::afir;

namespace mlir::ascend::kernel {
namespace {

// Aggregate independently lowered plan modules through an IR pass. Inputs are
// ModuleOp children of the pass root; the aggregate replaces the root body,
// matching af-opt's existing multi-module input convention.
//
// AscendC translation, tiling-header emission and host-wrapper emission produce
// text rather than IR, so they remain in runAggregateLaunchDriver.
struct AFIRAggregatePass : public PassWrapper<AFIRAggregatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRAggregatePass)

  AFIRAggregatePass() = default;
  AFIRAggregatePass(const AFIRAggregatePass &other) : PassWrapper(other) {
    graphName = other.graphName;
  }

  Pass::Option<std::string> graphName{*this, "graph-name",
                                      llvm::cl::desc("Fused kernel entry name for the aggregate module"),
                                      llvm::cl::init("fused_graph")};

  StringRef getArgument() const final {
    return "afir-aggregate";
  }

  StringRef getDescription() const final {
    return "Aggregate the per-candidate plan modules nested in this module into one entry module";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // attachPlanContracts 构造 afplan.candidate，因此本 pass 必须声明 afplan 依赖。
    // memref：Plan 调用 template 时恢复 ranked 视图；聚合调用点也可能需要适配独立提供的
    // ranked Plan 声明。两层入口使用 ABI 类型，template 签名保持不变。
    registry.insert<afir::AFIRDialect, afplan::AFPlanDialect, ascendc::AscendCDialect, emitasc::EmitAscDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp root = getOperation();
    // Several plan candidates arrive as sibling top-level modules, which the
    // parser wraps in one outer module; a single candidate is parsed as the
    // root itself. Both shapes are accepted, matching how af-opt already reads
    // a multi-module fixture.
    SmallVector<ModuleOp> planModules(root.getOps<ModuleOp>());
    if (planModules.empty()) {
      if (!afir::isPlanModule(root)) {
        root.emitError() << "afir-aggregate expects plan candidate modules, either nested in this module or as "
                            "this module itself";
        return signalPassFailure();
      }
      planModules.push_back(root);
    }

    std::string diagnostic;
    auto aggregate = buildAggregateEntryModule(planModules, graphName, &diagnostic);
    if (failed(aggregate)) {
      root.emitError() << "afir-aggregate failed: " << diagnostic;
      return signalPassFailure();
    }

    // The aggregate owns clones of the plan bodies, so the plan modules can be
    // dropped before its contents move in.
    ModuleOp entry = aggregate->first.get();
    root.getBody()->clear();
    root.getBody()->getOperations().splice(root.getBody()->end(), entry.getBody()->getOperations());

    // The aggregate contract lives on the module: tiling schema, module kind
    // and plan entries all have to travel with the body.
    NamedAttrList attributes(root->getAttrDictionary());
    for (NamedAttribute attribute : entry->getAttrs()) {
      attributes.set(attribute.getName(), attribute.getValue());
    }
    root->setAttrs(attributes.getDictionary(root.getContext()));
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRAggregatePass() {
  return std::make_unique<AFIRAggregatePass>();
}

void registerAFIRAggregatePass() {
  PassRegistration<AFIRAggregatePass>();
}

}  // namespace mlir::ascend::kernel
