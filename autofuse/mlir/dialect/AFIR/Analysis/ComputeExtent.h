/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_COMPUTEEXTENT_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_COMPUTEEXTENT_H

#include "dialect/AFIR/Analysis/VectorSegmentation.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::afir {
LogicalResult collectFreeSymbols(MLIRContext *context, ExprAttr expression, llvm::SetVector<StringRef> &symbols);
enum class ExtentSource { Nominal, Actual };
Value findAxisValue(func::FuncOp func, Operation *user, int64_t axisId, AxisValueRole role);
SymbolBindingMap collectTilingParamBindings(func::FuncOp func);
FailureOr<int64_t> getLiteralStride(ExprAttr expression);
FailureOr<SymbolBindingMap> collectExtentBindings(func::FuncOp func, Operation *op, AscGraphAttrGroupsAttr graph,
                                                  ArrayRef<ExprAttr> extents);
FailureOr<Value> materializeExpression(OpBuilder &builder, Location location, ExprAttr expression,
                                       const SymbolBindingMap &bindings);
// ActualSize bindings must dominate the computation. Missing tail bindings fail.
FailureOr<Value> materializeActualSegmentExtent(Operation *op, OpBuilder &builder, const VectorSegment &segment);
FailureOr<Value> materializeSegmentExtent(Operation *op, OpBuilder &builder, const VectorSegment &segment,
                                          int64_t scheduledAxis, Value tailAwareSize);
}  // namespace mlir::afir
#endif
