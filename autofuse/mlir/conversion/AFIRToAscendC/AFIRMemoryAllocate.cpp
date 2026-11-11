/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AFIRToAscendCLoweringStage.h"
#include "conversion/AFIRToAscendC/MemoryAllocate/CanonicalMemoryAllocator.h"

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Transforms/LoopGenerate.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"
#include "dialect/AFIR/Transforms/Symbolic/SymbolicTransformDetail.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::afir;
using namespace mlir::ascend::kernel;

namespace mlir::ascend::kernel::detail {
namespace {

// 函数里现存的全部 (轴, 角色)。
llvm::DenseSet<std::pair<int64_t, int64_t>> collectLiveBindings(func::FuncOp func) {
  llvm::DenseSet<std::pair<int64_t, int64_t>> live;
  func.walk([&](Operation *operation) {
    for (AxisBindingAttr binding : getAxisBindings(operation)) {
      live.insert({binding.getAxis(), static_cast<int64_t>(binding.getRole())});
    }
  });
  return live;
}

LogicalResult verifyFunctionAxisBindings(func::FuncOp func) {
  auto manifest = func->getAttrOfType<ArrayAttr>(kAxisBindingsAttrName);
  if (!manifest) {
    return success();
  }
  const llvm::DenseSet<std::pair<int64_t, int64_t>> live = collectLiveBindings(func);
  LogicalResult result = success();
  for (Attribute entry : manifest) {
    auto binding = dyn_cast<AxisBindingAttr>(entry);
    if (!binding) {
      result = func.emitError() << kAxisBindingsAttrName << " must contain only #afir.axis_binding entries";
      continue;
    }
    if (live.contains({binding.getAxis(), static_cast<int64_t>(binding.getRole())})) {
      continue;
    }
    // 报告缺失的轴与角色，便于定位删除绑定值的改写。
    result = func.emitError() << "axis binding was dropped after loop generation: axis " << binding.getAxis()
                              << ", role " << stringifyAxisValueRole(binding.getRole())
                              << ". Check whether a rewrite removed the definition or overwrote its bindings";
  }
  return result;
}

}  // namespace

LogicalResult CanonicalFunctionMemoryAllocator::run() {
  // 根据 IR 结构判断阶段状态，见 LoweringStage.h。
  // 已有 AscendC 资源时拒绝重跑，防止重复插入资源。无片上资源的函数没有物化产物，
  // 也没有待物化的 destination，重复执行为空操作，因此允许通过。
  // 这与旧 memory_allocated 标记由 commit() 无条件写入的行为不同。
  if (hasMaterializedAscendCResources(func)) {
    return func.emitError(
        "canonical AFIR resources are already materialized: the function already contains "
        "AscendC operations");
  }
  if (!afir::hasGeneratedLoops(func)) {
    return func.emitError(
               "canonical AFIR memory allocation requires materialized loops: no value in the function "
               "carries '")
           << afir::kAxisBindingAttrName << "'";
  }
  if (failed(getCanonicalTilingArgument(func))) {
    return failure();
  }
  // AFIR 层按 descriptor 契约收集和校验绑定，见 Analysis/CanonicalBindings.h。
  // 此处消费绑定表并生成 AscendC 算子，将后端无关分析与后端物化分开。
  FailureOr<afir::CanonicalBindings> bindings = afir::collectCanonicalBindings(func);
  if (failed(bindings)) {
    return failure();
  }
  values = std::move(bindings->values);
  valueOrder = std::move(bindings->valueOrder);
  if (failed(validateCanonicalBindings()) || failed(collectSymbolicBindings()) || failed(createCanonicalResources()) ||
      failed(materializeCanonicalValues()) || failed(rewriteCanonicalOperations())) {
    rollback();
    return failure();
  }
  if (failed(lowerSymbolicExpressions())) {
    rollback();
    return failure();
  }
  // 属性绑定不是 SSA use；在最后一次全函数重写之后验证它们仍然存活。
  return verifyFunctionAxisBindings(func);
}

void CanonicalFunctionMemoryAllocator::rollback() {
  for (Operation *operation : llvm::reverse(newOperations)) {
    if (operation->use_empty()) {
      operation->erase();
    }
  }
  for (Operation *operation : llvm::reverse(createdOps)) {
    if (operation->use_empty()) {
      operation->erase();
    }
  }
}

}  // namespace mlir::ascend::kernel::detail

namespace {

struct AFIRMemoryAllocatePass : public PassWrapper<AFIRMemoryAllocatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRMemoryAllocatePass)

  StringRef getArgument() const final {
    return "afir-memory-allocate";
  }

  StringRef getDescription() const final {
    return "Allocate TQue/TBuf and explicit GlobalTensor/LocalTensor buffers for scheduled AFIR";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<afir::AFIRDialect, arith::ArithDialect, ascendc::AscendCDialect, index::IndexDialect,
                    bufferization::BufferizationDialect, cf::ControlFlowDialect, func::FuncDialect, scf::SCFDialect>();
    // workspace 张量的基址由 `emitasc.ptr_offset` / `emitasc.reinterpret_cast` 物化，偏移
    // 形参的 `ui32 -> index` 转换由 `emitc.cast` 物化（`CanonicalMaterialization.cpp` 的
    // `materializeWorkspaceBase`）。整条流水线里这两个方言早被别的 pass 拉起来了，但单独
    // 跑本 pass 时没有，缺声明就是建 op 时直接 `report_fatal_error`。
    registry.insert<math::MathDialect, emitasc::EmitAscDialect, emitc::EmitCDialect>();
  }

  LogicalResult runFunction(func::FuncOp func) {
    if (!afir::hasFunctionRole(func, afir::FunctionRole::KernelBody)) {
      return success();
    }
    // 「这个函数走 canonical 路径吗」的判据是它是否携带 canonical descriptor 契约。
    // 迁移前看的是 `afir.arg_buffer_descs` 这个数组属性存在与否——空数组也算。
    // descriptor 归位到定义点之后，零参数、零局部 buffer 的函数在定义点上什么都没有，
    // 因此判据回到同样恒存在的 `afir.arg_descs`（设计文档 5.1 步 8）。
    if (func->hasAttr(afir::kArgDescsAttrName) || afir::hasBufferDescriptors(func.getOperation())) {
      return mlir::ascend::kernel::detail::CanonicalFunctionMemoryAllocator(func).run();
    }
    return func.emitError() << afir::kInvalidDescriptorDiagnostic
                            << ": AFIR memory allocation requires canonical buffer descriptors";
  }

  void runOnOperation() override {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      if (failed(runFunction(func))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

namespace mlir::ascend::kernel {

std::unique_ptr<Pass> createAFIRMemoryAllocatePass() {
  return std::make_unique<AFIRMemoryAllocatePass>();
}

void registerAFIRMemoryAllocatePass() {
  PassRegistration<AFIRMemoryAllocatePass>();
}

}  // namespace mlir::ascend::kernel
