/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_LOOPGENERATE_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_LOOPGENERATE_H

#include <functional>
#include <memory>

#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class Pass;
namespace afir {

struct SymbolBindingProvider {
  using MaterializeFn = std::function<FailureOr<SymbolBindingMap>(OpBuilder &, func::FuncOp, ArrayRef<ExprAttr>)>;
  using RegisterDialectsFn = std::function<void(DialectRegistry &)>;

  MaterializeFn materialize;
  RegisterDialectsFn registerDialects;
};

// Generate loops using bindings supplied by an external provider. The pass
// does not know how a target ABI provides those values.
std::unique_ptr<Pass> createAFIRLoopGeneratePass(SymbolBindingProvider provider);

// Generate loops using already-materialized symbolic bindings. The core does
// not know how a target ABI provides those values.
LogicalResult generateAFIRLoops(func::FuncOp func, SymbolBindingMap bindings);

// 循环是否已经物化。
//
// 这个问题此前由 `afir.loop_generated` 回答——一个记录「哪个 pass 跑过」的 IR 属性
// （设计文档 D5）。属性是同一事实的第二份编码：循环生成的产物就摆在 IR 里，判据应当
// 直接问 IR。对公开面尤其如此——一份形态完全正确、但没挂这个内部属性的 IR 会被判成
// 「没跑过」，而属性名不在任何对外文档里。
//
// 判据取 `#afir.axis_binding`：它是循环生成的产物，也是步 3b 为「从物化后的 IR 认回
// 轴」建立的那条替代通路——`scf.for` 上是 `role = loop`，物化出的标量上是
// `role = axis_index | axis_size | loop_size | tail_size | actual_size`。轴在生成前
// 只以 `#afir.asc_graph` 的轴表形式存在，不会出现这个标记；生成后每根轴都有。
//
// 比「函数体里有没有 `scf.for`」准确：只调度在 BlockOuter 上的算子不产生 `scf.for`，
// 但它的块索引同样带 `#afir.axis_binding`。
bool hasGeneratedLoops(func::FuncOp func);

}  // namespace afir
}  // namespace mlir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_LOOPGENERATE_H
