/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- TilingParams.cpp - Materialize afir.tiling_param values -------------===//
//
// 尺寸的 SSA 化（设计文档 3.3 / ADR-11）。队列与缓冲的尺寸是符号表达式，而
// `memref.alloc(%n)` 要的是 index 操作数；这一步把符号变成 SSA 值，AFIR 层从此
// 能表达「一块运行期尺寸的缓冲」。
//
// 本 pass 排在 `AFIRKernelBodySignature` 之后（那一步才把 tiling 参数追加到函数签名
// 上）、循环生成之前（循环生成要用这些值物化轴尺寸）。它不碰 ABI：产出的
// `afir.tiling_param` 只携带符号名，名字到 TilingData 字段的对应关系留给下降阶段。
//
//===----------------------------------------------------------------------===//

#include "dialect/AFIR/Transforms/TilingParams.h"
#include "dialect/AFIR/Analysis/ResourceIds.h"

#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"
#include "dialect/AFIR/Transforms/AFIRPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

using namespace mlir;

namespace mlir::afir {
namespace {

// 收集一个尺寸表达式里的自由符号。常量表达式（`"16"`）没有自由符号，直接返回空集。
LogicalResult collectExpressionSymbols(func::FuncOp func, ExprAttr expression,
                                       llvm::SetVector<llvm::StringRef> &symbols) {
  if (!expression) {
    return success();
  }
  auto parsed = Expression::Parse(expression.getValue().getValue());
  if (!parsed) {
    return func.emitError() << kInvalidBindingDiagnostic
                            << ": invalid resource size expression: " << parsed.error().message;
  }
  auto free = detail::ExpressionAccess::GetFreeSymbols(parsed.value());
  if (!free) {
    return func.emitError() << kInvalidBindingDiagnostic
                            << ": failed to resolve resource size expression symbols: " << free.error().message;
  }
  MLIRContext *context = func.getContext();
  for (const std::string &symbol : free.value()) {
    // StringRef 不能指向这个临时 std::string，交给 context 的字符串池持有。
    symbols.insert(StringAttr::get(context, symbol).getValue());
  }
  return success();
}

// 与资源物化阶段收集的是同一个集合，只是提前到这里做一次：轴尺寸来自
// `#afir.asc_graph` 的 `size_var`，缓冲字节数来自每一个 `#afir.buffer_desc`——要么是它的
// `size_bytes_expr`，要么（片上资源）是 `size_from_resource` 指向的那个 canonical 尺寸
// 字段。descriptor 挂在值的定义点上（设计文档步 8），所以函数参数与函数体里的定义 Op
// 两处都要走一遍。
LogicalResult collectDescriptorSymbols(func::FuncOp func, BufferDescAttr descriptor,
                                       llvm::SetVector<llvm::StringRef> &symbols) {
  if (descriptor.getSizeFromResource()) {
    const std::string name =
        makeCanonicalResourceSizeName(isCanonicalQueueResource(descriptor), descriptor.getBufferId());
    // StringRef 不能指向这个临时 std::string，交给 context 的字符串池持有。
    symbols.insert(StringAttr::get(func.getContext(), name).getValue());
    return success();
  }
  return collectExpressionSymbols(func, descriptor.getSizeBytesExpr(), symbols);
}

// 一份 tensor descriptor 里的尺寸符号：`repeats` / `strides` / `vectorized_strides`。
//
// 这一份不能省。下降侧把 descriptor 里非字面量的 extent 与 stride 物化成 `index` 值
// （`materializeExtentValue` / `materializeStrideValue` → `materializeSizeAt`），自由符号绑到
// `getOrCreateTilingParam`；本 pass 不收它们的话，第一个碰到的消费者就地建一条，而那时
// 实现选择已经跑过、诊断也早该在这里一次性报完了。
//
// 只走 Op 上的三份（`afir.operand_descs` / `afir.destination_descs` / `afir.result_descs`），
// **不额外走函数的 `afir.arg_descs`**：一个 SSA 值上的 canonical descriptor 只能有一份
// （`CanonicalBindings` 把块参数的 arg_desc 与用到它的 Op 那一份绑成同一条，冲突即
// `InvalidBinding`），所以 `arg_descs` 里不可能出现 Op 那边没有的符号。写过那一遍，变异
// 验证显示它一条用例都挡不住。
LogicalResult collectTensorDescSymbols(func::FuncOp func, Attribute attr, llvm::SetVector<llvm::StringRef> &symbols) {
  auto descriptors = dyn_cast_if_present<ArrayAttr>(attr);
  if (!descriptors) {
    return success();
  }
  for (Attribute entry : descriptors) {
    auto descriptor = dyn_cast<TensorDescAttr>(entry);
    if (!descriptor) {
      continue;
    }
    for (ArrayRef<ExprAttr> expressions :
         {descriptor.getRepeats(), descriptor.getStrides(), descriptor.getVectorizedStrides()}) {
      for (ExprAttr expression : expressions) {
        if (failed(collectExpressionSymbols(func, expression, symbols))) {
          return failure();
        }
      }
    }
  }
  return success();
}

LogicalResult collectRequiredSymbols(func::FuncOp func, llvm::SetVector<llvm::StringRef> &symbols) {
  if (auto graph = func->getAttrOfType<AscGraphAttrGroupsAttr>(kAscGraphAttrName)) {
    for (ExprAttr sizeVariable : graph.getSizeVar()) {
      if (failed(collectExpressionSymbols(func, sizeVariable, symbols))) {
        return failure();
      }
    }
  }
  for (BlockArgument argument : func.getArguments()) {
    if (auto descriptor = getBufferDescAttr(argument)) {
      if (failed(collectDescriptorSymbols(func, descriptor, symbols))) {
        return failure();
      }
    }
  }
  LogicalResult result = success();
  func.walk([&](Operation *op) {
    if (failed(result)) {
      return;
    }
    if (auto descriptor = op->getAttrOfType<BufferDescAttr>(kBufferDescAttrName)) {
      result = collectDescriptorSymbols(func, descriptor, symbols);
      if (failed(result)) {
        return;
      }
    }
    for (llvm::StringLiteral name : {kOperandDescsAttrName, kDestinationDescsAttrName, kResultDescsAttrName}) {
      result = collectTensorDescSymbols(func, op->getAttr(name), symbols);
      if (failed(result)) {
        return;
      }
    }
  });
  return result;
}

struct AFIRMaterializeTilingParamsPass : public PassWrapper<AFIRMaterializeTilingParamsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRMaterializeTilingParamsPass)

  StringRef getArgument() const final {
    return "afir-materialize-tiling-params";
  }

  StringRef getDescription() const final {
    return "Hoist every runtime size symbol of an AFIR kernel body into the size prologue as an "
           "afir.tiling_param value (optional: consumers create missing ones on demand)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AFIRDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      if (func.isExternal()) {
        continue;
      }
      if (failed(materializeTilingParams(func))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

LogicalResult materializeTilingParams(func::FuncOp func) {
  if (func.isExternal() || func.getBody().empty()) {
    return success();
  }
  llvm::SetVector<llvm::StringRef> symbols;
  if (failed(collectRequiredSymbols(func, symbols))) {
    return failure();
  }
  // 反向建，好让入口块里的顺序与收集顺序一致：每一个都插在块首。
  for (llvm::StringRef symbol : llvm::reverse(symbols)) {
    (void)getOrCreateTilingParam(func.front(), symbol);
  }
  return success();
}

std::unique_ptr<Pass> createAFIRMaterializeTilingParamsPass() {
  return std::make_unique<AFIRMaterializeTilingParamsPass>();
}

void registerAFIRMaterializeTilingParamsPass() {
  PassRegistration<AFIRMaterializeTilingParamsPass>();
}

}  // namespace mlir::afir
