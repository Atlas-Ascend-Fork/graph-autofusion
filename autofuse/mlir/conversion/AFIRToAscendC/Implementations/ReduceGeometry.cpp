/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ReduceGeometry.cpp - 归约下降共用的二维几何 --------------*- C++ -*-===//

#include "conversion/AFIRToAscendC/Implementations/ReduceGeometry.h"

#include <cstdint>
#include <string>

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "dialect/AFIR/Analysis/TransferPlan.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

using namespace mlir::afir;

namespace mlir::ascend::kernel {

bool hasCanonicalDescriptors(Operation *op) {
  return op->hasAttr(afir::kOperandDescsAttrName) && op->hasAttr(afir::kDestinationDescsAttrName) &&
         op->hasAttr(afir::kResultDescsAttrName);
}

// 按规范表达式文本识别常量 0/1；无法确认时不视为常量。
bool isLiteralExpr(afir::ExprAttr expr, llvm::StringRef literal) {
  return expr && expr.getValue().getValue().trim() == literal;
}

// 标称 extent 取自 asc_graph，与形状物化同源；stride 取自 descriptor。
// descriptor 的 repeats 未保证与轴尺寸相等，不能替代此处的 extent。
afir::ExprAttr graphExtentOf(const llvm::DenseMap<int64_t, afir::AxisAttr> &axesById, int64_t axisId) {
  auto found = axesById.find(axisId);
  if (found == axesById.end()) {
    return {};
  }
  return found->second.getSize();
}

afir::ExprAttr vectorizedStrideOf(afir::TensorDescAttr desc, int64_t axisId) {
  ArrayRef<int64_t> axes = desc.getVectorizedAxis();
  auto strides = desc.getVectorizedStrides();
  if (strides.size() != axes.size()) {
    return {};
  }
  const auto *found = llvm::find(axes, axisId);
  if (found == axes.end()) {
    return {};
  }
  return strides[std::distance(axes.begin(), found)];
}

LogicalResult failLayout(std::string *diagnostic, llvm::StringRef reason) {
  if (diagnostic != nullptr) {
    diagnostic->assign(reason.begin(), reason.end());
  }
  return failure();
}

// 证明物理布局连续：外侧 stride = 内侧 stride × 内侧标称长度，最内长度须对齐。
// 跳过可证为 1 的轴，但保留物理最内轴用于检查行内连续性。
LogicalResult proveReduceDestinationLayout(MLIRContext *context, afir::TensorDescAttr destination,
                                           const llvm::DenseMap<int64_t, afir::AxisAttr> &axesById,
                                           const afir::ReducePlan &plan, std::string *diagnostic) {
  // The API writes compact results. Prove the whole kept-axis stride chain,
  // not unit stride on every axis (e.g. [4, 1] is a valid compact A group).
  for (int64_t axisId : plan.last.axes) {
    if (!isLiteralExpr(vectorizedStrideOf(destination, axisId), "0")) {
      return failLayout(diagnostic, "destination vectorized stride on a reduced axis is not provably 0");
    }
  }
  afir::ExprAttr expected = afir::ExprAttr::get(context, StringAttr::get(context, "1"));
  for (int64_t axisId : llvm::reverse(plan.first.axes)) {
    auto extent = graphExtentOf(axesById, axisId);
    if (isLiteralExpr(extent, "1")) {
      continue;
    }
    auto stride = vectorizedStrideOf(destination, axisId);
    if (afir::proveEqual(stride, expected) != afir::detail::TruthValue::True) {
      return failLayout(diagnostic, "destination kept axes are not provably compact");
    }
    auto product = afir::MultiplyExpressions(expected.getValue().getValue().str(), extent.getValue().getValue().str());
    if (!product) {
      return failLayout(diagnostic, "cannot form the compact destination stride");
    }
    expected = afir::ExprAttr::get(context, StringAttr::get(context, product.value()));
  }
  return success();
}

LogicalResult proveReduceLayout(MLIRContext *context, afir::TensorDescAttr input, afir::TensorDescAttr destination,
                                const llvm::DenseMap<int64_t, afir::AxisAttr> &axesById, const afir::ReducePlan &plan,
                                int64_t alignNum, std::string *diagnostic) {
  // 按物理顺序检查：AR 为 [A…, R…]，RA 为 [R…, A…]。
  // AR 还须证明输出 A 轴组紧凑。
  SmallVector<int64_t, 4> physical;
  const bool isAR = plan.pattern == afir::ReducePattern::AR;
  for (int64_t axisId : isAR ? ArrayRef<int64_t>(plan.first.axes) : ArrayRef<int64_t>(plan.last.axes)) {
    physical.push_back(axisId);
  }
  for (int64_t axisId : isAR ? ArrayRef<int64_t>(plan.last.axes) : ArrayRef<int64_t>(plan.first.axes)) {
    physical.push_back(axisId);
  }

  SmallVector<int64_t, 4> effective;
  for (auto [position, axisId] : llvm::enumerate(physical)) {
    afir::ExprAttr nominal = graphExtentOf(axesById, axisId);
    if (!nominal) {
      return failLayout(diagnostic,
                        "asc_graph has no size for a vectorized axis, so the API row pitch cannot be "
                        "proven against the tile layout");
    }
    // 最内轴即使长度为 1 也保留，以校验行内 stride。
    if (isLiteralExpr(nominal, "1") && position + 1 < physical.size()) {
      continue;
    }
    effective.push_back(axisId);
  }
  if (effective.empty()) {
    return failLayout(diagnostic, "every vectorized axis has a degenerate extent, leaving no layout to prove");
  }

  // 最内一根的 stride 必须是 1：高阶 API 按元素连续读每一行。
  const int64_t innermost = effective.back();
  afir::ExprAttr innermostStride = vectorizedStrideOf(input, innermost);
  afir::ExprAttr innermostNominal = graphExtentOf(axesById, innermost);
  // A statically unit axis may have zero stride, but still reserves an aligned row.
  // Use unit stride for its footprint and retain the outer pitch check.
  if (isLiteralExpr(innermostNominal, "1") && isLiteralExpr(innermostStride, "0")) {
    innermostStride = afir::ExprAttr::get(context, StringAttr::get(context, "1"));
  }
  if (!isLiteralExpr(innermostStride, "1")) {
    return failLayout(diagnostic, "innermost vectorized input stride is not provably 1");
  }
  FailureOr<afir::ExprAttr> innermostAligned = afir::alignUpExpr(context, innermostNominal, alignNum);
  if (failed(innermostAligned)) {
    return failLayout(diagnostic, "cannot form the 32B-aligned footprint of the innermost vectorized axis");
  }

  // 由内向外逐对查连续性。内侧长度：最内一根取对齐后的值，其余取标称值。
  for (size_t position = effective.size(); position-- > 1;) {
    const int64_t outer = effective[position - 1];
    const int64_t inner = effective[position];
    afir::ExprAttr innerExtent = inner == innermost ? *innermostAligned : graphExtentOf(axesById, inner);
    afir::ExprAttr innerStride = inner == innermost ? innermostStride : vectorizedStrideOf(input, inner);
    if (!afir::provablyContiguous(context, vectorizedStrideOf(input, outer), innerStride, innerExtent)) {
      return failLayout(diagnostic,
                        "input vectorized strides are not provably contiguous across the A/R axis group, so the "
                        "flat row pitch handed to the reduce API may not match the tile layout");
    }
  }

  return proveReduceDestinationLayout(context, destination, axesById, plan, diagnostic);
}

// 能力查询和改写共用布局判据，失败原因写入 diagnostic。
// RA 限单根 R 轴；AR 限至多两根 R 轴，以满足 ReduceInit 的后缀填充方式。
// AR 多 A 轴须连续，且仅最外有效 A 轴可有尾块。
LogicalResult checkKeptAxisTails(Operation *op, func::FuncOp func, afir::AscGraphAttrGroupsAttr graph,
                                 const llvm::DenseMap<int64_t, afir::AxisAttr> &axesById, const afir::ReducePlan &plan,
                                 std::string *diagnostic) {
  // Once an outer A can contain multiple rows, each inner A must retain its
  // nominal extent so the flattened rows remain contiguous.
  bool hasOuterRows = false;
  for (int64_t axisId : plan.first.axes) {
    auto extent = graphExtentOf(axesById, axisId);
    if (!extent) {
      return failLayout(diagnostic, "missing kept-axis extent");
    }
    if (hasOuterRows) {
      auto expression = afir::Expression::Parse(extent.getValue().getValue().str());
      if (!expression) {
        return failLayout(diagnostic, "cannot prove inner kept-axis extent");
      }
      auto symbols = afir::detail::ExpressionAccess::GetFreeSymbols(expression.value());
      if (!symbols) {
        return failLayout(diagnostic, "cannot resolve inner kept-axis symbols");
      }
      for (auto axis : graph.getAxes()) {
        if (!axis.getSize() || !llvm::is_contained(symbols.value(), axis.getSize().getValue().getValue().str())) {
          continue;
        }
        const bool splitInner =
            (axis.getAxisType() == afir::AxisType::TileInner || axis.getAxisType() == afir::AxisType::BlockInner) &&
            axis.getSplitPairOtherId() >= 0;
        if (splitInner || afir::findAxisValue(func, op, axis.getId(), afir::AxisValueRole::ActualSize)) {
          return failLayout(diagnostic, "inner kept-axis tail may leave gaps between valid rows");
        }
      }
    }
    hasOuterRows |= !isLiteralExpr(extent, "1");
  }
  return success();
}

LogicalResult checkReduceLayout(Operation *op, const afir::ReducePlan &plan, int64_t alignNum,
                                std::string *diagnostic) {
  if (diagnostic != nullptr) {
    diagnostic->clear();
  }
  if (plan.pattern != afir::ReducePattern::AR) {
    // RA 的外侧 R 组限单轴，避免扁平遍历跨过尾块空行。
    if (plan.last.axes.size() != 1) {
      return failLayout(diagnostic, "AFIR reduce RA layout currently requires a single reduced axis");
    }
    return success();
  }
  if (plan.last.axes.size() > 2) {
    return failLayout(diagnostic, "AFIR reduce AR layout currently supports at most two reduced axes");
  }
  auto operandDescs = op->getAttrOfType<ArrayAttr>(afir::kOperandDescsAttrName);
  auto destinationDescs = op->getAttrOfType<ArrayAttr>(afir::kDestinationDescsAttrName);
  if (!operandDescs || operandDescs.empty() || !destinationDescs || destinationDescs.empty()) {
    return failLayout(diagnostic, "missing operand or destination tensor descriptors");
  }
  auto input = dyn_cast<afir::TensorDescAttr>(operandDescs[0]);
  auto destination = dyn_cast<afir::TensorDescAttr>(destinationDescs[0]);
  if (!input || !destination) {
    return failLayout(diagnostic, "operand or destination descriptor is not a tensor descriptor");
  }
  // extent 必须与 `materializeScheduledAxisExtents` 同源：同一份 `#afir.asc_graph`。
  auto func = op->getParentOfType<func::FuncOp>();
  auto graph = func ? func->getAttrOfType<afir::AscGraphAttrGroupsAttr>(afir::kAscGraphAttrName)
                    : afir::AscGraphAttrGroupsAttr();
  if (!graph) {
    return failLayout(diagnostic, "enclosing function has no asc_graph, so axis extents cannot be resolved");
  }
  llvm::DenseMap<int64_t, afir::AxisAttr> axesById;
  for (afir::AxisAttr axis : graph.getAxes()) {
    axesById.try_emplace(axis.getId(), axis);
  }
  if (failed(checkKeptAxisTails(op, func, graph, axesById, plan, diagnostic))) {
    return failure();
  }
  return proveReduceLayout(op->getContext(), input, destination, axesById, plan, alignNum, diagnostic);
}

// 通过 loop 角色的 axis_binding 找到累加轴对应的 scf.for。
scf::ForOp findAccumulationLoop(Operation *op, int64_t axisId) {
  for (Operation *parent = op->getParentOp(); parent != nullptr; parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop) {
      continue;
    }
    if (afir::hasAxisBinding(loop, axisId, afir::AxisValueRole::Loop)) {
      return loop;
    }
  }
  return nullptr;
}

// 向上对齐到 alignment 的整数倍；乘法使用溢出检查，防止 tiling 长度回绕。
Value alignUp(OpBuilder &builder, Location location, Value value, int64_t alignment) {
  Value alignmentValue = builder.create<arith::ConstantIndexOp>(location, alignment);
  Value blocks = builder.create<arith::CeilDivSIOp>(location, value, alignmentValue);
  auto product = builder.create<arith::MulSIExtendedOp>(location, blocks, alignmentValue);
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value lowNonNegative = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::sge, product.getLow(), zero);
  Value highZero = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, product.getHigh(), zero);
  Value noOverflow = builder.create<arith::AndIOp>(location, lowNonNegative, highZero);
  builder.create<cf::AssertOp>(location, noOverflow, "reduce aligned extent must not overflow");
  return product.getLow();
}

Value multiplyChecked(OpBuilder &builder, Location location, Value lhs, Value rhs) {
  auto product = builder.create<arith::MulSIExtendedOp>(location, lhs, rhs);
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value lowNonNegative = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::sge, product.getLow(), zero);
  Value highZero = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, product.getHigh(), zero);
  Value noOverflow = builder.create<arith::AndIOp>(location, lowNonNegative, highZero);
  builder.create<cf::AssertOp>(location, noOverflow, "reduce merged extent must not overflow");
  return product.getLow();
}

Value multiplyAll(OpBuilder &builder, Location location, ArrayRef<Value> values) {
  if (values.size() == 1) {
    return values.front();
  }
  Value product = builder.create<arith::ConstantIndexOp>(location, 1);
  for (Value value : values) {
    product = multiplyChecked(builder, location, product, value);
  }
  return product;
}

Value toI32(OpBuilder &builder, Location location, Value value) {
  if (value.getType().isInteger(32)) {
    return value;
  }
  return builder.create<arith::IndexCastOp>(location, builder.getI32Type(), value);
}

// 将语义 A/R 轴组合成 UB 的二维物理形状。
// 搬运视图受 GM/UB 双侧连续性约束，不能直接作为归约的二维形状。
FailureOr<PhysicalReduceShape> materializePhysicalReduceShape(Operation *op, OpBuilder &builder,
                                                              const afir::ReducePlan &plan, int64_t alignNum) {
  const bool isAR = plan.pattern == afir::ReducePattern::AR;
  const ArrayRef<int64_t> outerAxes = isAR ? ArrayRef<int64_t>(plan.first.axes) : ArrayRef<int64_t>(plan.last.axes);
  const ArrayRef<int64_t> innerAxes = isAR ? ArrayRef<int64_t>(plan.last.axes) : ArrayRef<int64_t>(plan.first.axes);

  // 按物理外、内维物化：AR 为 A/R，RA 为 R/A。
  FailureOr<SmallVector<afir::ScheduledAxisExtent, 4>> outer =
      afir::materializeScheduledAxisExtents(op, builder, outerAxes);
  FailureOr<SmallVector<afir::ScheduledAxisExtent, 4>> inner =
      afir::materializeScheduledAxisExtents(op, builder, innerAxes);
  if (failed(outer) || failed(inner) || outer->empty() || inner->empty()) {
    return failure();
  }

  const Location location = op->getLoc();
  PhysicalReduceShape shape;

  SmallVector<Value, 4> outerActuals;
  for (const afir::ScheduledAxisExtent &extent : *outer) {
    outerActuals.push_back(extent.actual);
  }
  shape.outerActual = multiplyAll(builder, location, outerActuals);

  // 仅最内轴对齐：例如 [3, 5] 的 footprint 为 3 * align(5)，不是 align(15)。
  // 行距按 nominal 计算，不随尾块改变。
  Value innerNominalAligned = alignUp(builder, location, inner->back().nominal, alignNum);
  shape.innerFootprint = innerNominalAligned;
  // 仅 AR 的 ReduceInit 需要这些 actual 量，RA 不物化对应算术与断言。
  if (isAR) {
    shape.innermostInnerActual = inner->back().actual;
    shape.innermostInnerStride = innerNominalAligned;
    shape.innerActualFootprint =
        inner->size() == 1 ? alignUp(builder, location, shape.innermostInnerActual, alignNum) : innerNominalAligned;
  }
  for (size_t index = 0; index + 1 < inner->size(); ++index) {
    shape.innerFootprint = multiplyChecked(builder, location, shape.innerFootprint, (*inner)[index].nominal);
    if (isAR) {
      shape.innerActualFootprint =
          multiplyChecked(builder, location, shape.innerActualFootprint, (*inner)[index].actual);
    }
  }

  // AR 的 A 槽位连续，跨度直接复用 outerActual。
  shape.resultSpan = shape.outerActual;
  if (!isAR) {
    // RA 的 A 组在物理内侧；按 nominal stride 累计槽位跨度，最内轴长度先对齐。
    shape.resultSpan = inner->back().actual;
    Value stride = innerNominalAligned;
    for (size_t index = inner->size() - 1; index > 0; --index) {
      const afir::ScheduledAxisExtent &axis = (*inner)[index - 1];
      Value reduced =
          builder.create<arith::SubIOp>(location, axis.actual, builder.create<arith::ConstantIndexOp>(location, 1));
      shape.resultSpan = builder.create<arith::AddIOp>(location, shape.resultSpan,
                                                       builder.create<arith::MulIOp>(location, reduced, stride));
      if (index > 1) {
        stride = multiplyChecked(builder, location, stride, axis.nominal);
      }
    }
  }
  return shape;
}
ReduceResidency getReduceResidency(Value input, Value destination) {
  if (!input || !destination) {
    return ReduceResidency::Unknown;
  }
  FailureOr<ascend::MemorySpace> inputSpace = getValueMemorySpace(input);
  FailureOr<ascend::MemorySpace> destinationSpace = getValueMemorySpace(destination);
  if (failed(inputSpace) || failed(destinationSpace)) {
    return ReduceResidency::Unknown;
  }
  return *inputSpace == ascend::MemorySpace::UB && *destinationSpace == ascend::MemorySpace::UB
             ? ReduceResidency::Ok
             : ReduceResidency::NotUb;
}

}  // namespace mlir::ascend::kernel
