/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ReduceGeometry.h - 归约下降共用的二维几何 ----------------*- C++ -*-===//
//
// 普通归约的布局证明、物理形状物化、内存空间判定，以及取整和相乘工具。
// 算子类型、scratch 角色和具体指令选择由 Reduce.cpp 处理。
//
#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_REDUCEGEOMETRY_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_REDUCEGEOMETRY_H

#include <cstdint>
#include <string>

#include "dialect/AFIR/Analysis/ReducePlan.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "conversion/AFIRToAscendC/AscendCAttributes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend::kernel {

// 一次高阶归约用到的 UB 资源的对齐宽度：32 字节一块。
constexpr int64_t kBlockBytes = 32;

// 区分位于 UB、缺少 descriptor 和非 UB；缺少信息时不可默认成 UB。
enum class ReduceResidency { Ok, Unknown, NotUb };

ReduceResidency getReduceResidency(Value input, Value destination);

// 物理形状：AR 为 [A…, R…]，RA 为 [R…, A…]。
// 归约 API 按物理次序取形状；累加与 Mean 缩放按语义 A 槽位跨度取 count。
struct PhysicalReduceShape {
  Value outerActual;           // 物理外维的行数：恒取 actual，尾块下必须少走几行
  Value innerFootprint;        // 物理内维的行距：align(最内 nominal) × 其余 nominal
  Value innerActualFootprint;  // 同上但取 actual，只有 AR 的 `ReduceInit` 用得到，RA 下为空
  Value innermostInnerActual;  // 物理最内轴的 actual，同上，RA 下为空
  Value innermostInnerStride;  // 物理最内轴的标称对齐行距，不随尾块变化，RA 下为空
  // 累加与缩放覆盖的 A 槽位跨度，可能包含不写回 GM 的 padding。
  // AR 为连续结果数 outerActual；RA 按 1 + Σ (actual - 1) × stride 计算，
  // stride 使用最内轴对齐后的 nominal 布局。
  Value resultSpan;
};

bool hasCanonicalDescriptors(Operation *op);

bool isLiteralExpr(afir::ExprAttr expr, llvm::StringRef literal);

afir::ExprAttr graphExtentOf(const llvm::DenseMap<int64_t, afir::AxisAttr> &axesById, int64_t axisId);

afir::ExprAttr vectorizedStrideOf(afir::TensorDescAttr desc, int64_t axisId);

LogicalResult failLayout(std::string *diagnostic, llvm::StringRef reason);

// 按 plan.pattern 的物理次序证明布局。
LogicalResult proveReduceLayout(MLIRContext *context, afir::TensorDescAttr input, afir::TensorDescAttr destination,
                                const llvm::DenseMap<int64_t, afir::AxisAttr> &axesById, const afir::ReducePlan &plan,
                                int64_t alignNum, std::string *diagnostic);

LogicalResult checkReduceLayout(Operation *op, const afir::ReducePlan &plan, int64_t alignNum,
                                std::string *diagnostic = nullptr);

// 累加轴对应的那层 `scf.for`。
scf::ForOp findAccumulationLoop(Operation *op, int64_t axisId);

Value alignUp(OpBuilder &builder, Location location, Value value, int64_t alignment);

Value multiplyChecked(OpBuilder &builder, Location location, Value lhs, Value rhs);

Value multiplyAll(OpBuilder &builder, Location location, ArrayRef<Value> values);

Value toI32(OpBuilder &builder, Location location, Value value);

FailureOr<PhysicalReduceShape> materializePhysicalReduceShape(Operation *op, OpBuilder &builder,
                                                              const afir::ReducePlan &plan, int64_t alignNum);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_REDUCEGEOMETRY_H
