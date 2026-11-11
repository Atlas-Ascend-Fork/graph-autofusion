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
// 根据循环生成的 IR 结构判断，不使用 afir.loop_generated 执行标记（设计文档 D5）。
// 结构正确的手写 IR 也应被识别，不应依赖未公开的 pass 状态属性。
//
// 检查 axis_binding：scf.for 使用 loop 角色，物化标量使用 axis_index、axis_size、
// loop_size、tail_size 或 actual_size。循环生成前轴仅存在于 asc_graph 轴表中；
// 生成后通过这些绑定追踪轴与值的对应关系。
//
// 不能仅检查 scf.for：仅调度到 BlockOuter 的算子不生成循环，但块索引仍带 axis_binding。
bool hasGeneratedLoops(func::FuncOp func);

}  // namespace afir
}  // namespace mlir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_LOOPGENERATE_H
