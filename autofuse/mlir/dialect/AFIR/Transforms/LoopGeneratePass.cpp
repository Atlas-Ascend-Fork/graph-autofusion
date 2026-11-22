/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for the specific language governing permissions and limitations under the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- LoopGeneratePass.cpp - Generic AFIR loop-generation pass -------------===//

#include "dialect/AFIR/Transforms/LoopGenerate.h"

#include <utility>

#include "dialect/AFIR/IR/AFIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::afir {
namespace {

struct AFIRLoopGeneratePass : public PassWrapper<AFIRLoopGeneratePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRLoopGeneratePass)

  explicit AFIRLoopGeneratePass(SymbolBindingProvider provider) : provider(std::move(provider)) {}

  StringRef getArgument() const final {
    return "afir-loop-generate";
  }

  StringRef getDescription() const final {
    return "Generate legacy-compatible axis values and scf loop trees for scheduled AFIR operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AFIRDialect, arith::ArithDialect, bufferization::BufferizationDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
    if (provider.registerDialects) {
      provider.registerDialects(registry);
    }
  }

  void runOnOperation() override {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      if (!hasFunctionRole(func, FunctionRole::KernelBody)) {
        continue;
      }
      if (hasGeneratedLoops(func)) {
        continue;
      }
      if (func.isExternal() || !llvm::hasSingleElement(func.getBody())) {
        func.emitError("AFIR loop generation expects a single-block function");
        signalPassFailure();
        return;
      }
      auto graph = func->getAttrOfType<AscGraphAttrGroupsAttr>(kAscGraphAttrName);
      if (!graph) {
        func.emitError("AFIR loop generation requires ") << kAscGraphAttrName;
        signalPassFailure();
        return;
      }
      Operation *firstAfir = nullptr;
      for (Operation &operation : func.front()) {
        if (operation.getDialect() && operation.getDialect()->getNamespace() == AFIRDialect::getDialectNamespace()) {
          firstAfir = &operation;
          break;
        }
      }
      if (!firstAfir) {
        continue;
      }
      OpBuilder builder(&getContext());
      builder.setInsertionPoint(firstAfir);
      if (!provider.materialize) {
        func.emitError("AFIR loop generation requires a symbol binding provider");
        signalPassFailure();
        return;
      }
      FailureOr<SymbolBindingMap> bindings = provider.materialize(builder, func, graph.getSizeVar());
      if (failed(bindings) || failed(generateAFIRLoops(func, std::move(*bindings)))) {
        signalPassFailure();
        return;
      }
    }
  }

 private:
  SymbolBindingProvider provider;
};

}  // namespace

std::unique_ptr<Pass> createAFIRLoopGeneratePass(SymbolBindingProvider provider) {
  return std::make_unique<AFIRLoopGeneratePass>(std::move(provider));
}

}  // namespace mlir::afir
