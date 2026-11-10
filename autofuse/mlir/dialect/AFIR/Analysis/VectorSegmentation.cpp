/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/Analysis/VectorSegmentation.h"

#include <string>

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::afir {

namespace {

constexpr llvm::StringLiteral kOne = "1";

StringRef exprText(ExprAttr expression) {
  return expression.getValue().getValue();
}

}  // namespace

// `lhs` 与 `rhs` 可证相等吗。证不了（含 `Unknown` 与解析失败）返回 false。
//
// 文本相同先短路：合轴里绝大多数比较是字面量对字面量，短路省掉一次 SymEngine 往返。
// 它只能答「相等」，答不了「不等」，所以不影响保守方向。
// 供本文件逐对合并与 `Implementations/` 侧的形态判据（`Transpose.cpp` / `Pad.cpp`）
// 共用——两侧都是「判不了就拒」的保守方向，短路与 SymEngine 语义必须同一份。
bool provablyEqual(ExprAttr lhs, ExprAttr rhs) {
  if (!lhs || !rhs) {
    return false;
  }
  if (exprText(lhs) == exprText(rhs)) {
    return true;
  }
  return ProveExpressionsEqual(exprText(lhs), exprText(rhs));
}

bool provablyOne(ExprAttr expression) {
  if (exprText(expression) == kOne) {
    return true;
  }
  return ProveExpressionsEqual(exprText(expression), kOne);
}

namespace {

// `lhs * rhs` 的规范形式。搬运偏移的强度削减问的是同一件事，实现因此收在
// `SymbolicBuilder.h`（`multiplyExprs`），这里只留名字。
FailureOr<ExprAttr> multiply(MLIRContext *context, ExprAttr lhs, ExprAttr rhs) {
  return multiplyExprs(context, lhs, rhs);
}

// `op` 上参与判据的张量：两侧 descriptor 数组里角色为 `compute` 的那些。
SmallVector<VectorTensor> collectTensors(Operation *op) {
  SmallVector<VectorTensor> tensors;
  const std::pair<llvm::StringLiteral, VectorTensor::Side> sides[] = {
      {kOperandDescsAttrName, VectorTensor::Side::Operand},
      {kDestinationDescsAttrName, VectorTensor::Side::Destination}};
  for (const auto &[name, side] : sides) {
    auto descriptors = op->getAttrOfType<ArrayAttr>(name);
    if (!descriptors) {
      continue;
    }
    for (auto [index, attribute] : llvm::enumerate(descriptors)) {
      auto descriptor = dyn_cast<TensorDescAttr>(attribute);
      if (!descriptor || !descriptor.getRole() || descriptor.getRole().getValue() != Role::Compute) {
        continue;
      }
      // 没有向量化轴的 compute 张量不参与：标量就是这个形态。调度给标量的
      // descriptor 是 `axis = [0, 1], repeats = [1, 1], strides = [0, 0], role = compute`，
      // 一个 `vectorized_axis` 都没有——它在整条循环里就是同一个值，既约束不了合轴，
      // 也不需要偏移。跳过它，下降侧「标量不是 tile、不生成 subindex」那条判断就不必再写一遍。
      if (descriptor.getVectorizedAxis().empty()) {
        continue;
      }
      tensors.push_back(VectorTensor{side, static_cast<unsigned>(index), descriptor});
    }
  }
  return tensors;
}

// descriptor 自身的长度约束。`TensorDescAttr::verify` 只在两个数组都非空时才比长度
// （`AFIRDialect.cpp:727` / `:737`），所以「`axis` 非空而 `repeats` 为空」是合法属性——
// 而下面要按 `axis` 的下标去 `repeats` 里取，那样会越界读。这道检查因此不是冗余的。
bool hasConsistentLengths(TensorDescAttr descriptor) {
  return descriptor.getAxis().size() == descriptor.getRepeats().size() &&
         descriptor.getVectorizedAxis().size() == descriptor.getVectorizedStrides().size();
}

}  // namespace

// `axisId` 在 `descriptor.getAxis()` 里的下标。
FailureOr<size_t> axisPosition(TensorDescAttr descriptor, int64_t axisId) {
  ArrayRef<int64_t> axis = descriptor.getAxis();
  const auto *position = llvm::find(axis, axisId);
  if (position == axis.end()) {
    return failure();
  }
  return static_cast<size_t>(position - axis.begin());
}

namespace {

// 一根轴的公共 extent：各张量在该轴上的 repeat 中，凡不可证为 1 的必须互相可证相等。
// 全为 1 时返回字面量 1，调用方据此丢掉这根轴。
FailureOr<ExprAttr> commonExtent(MLIRContext *context, ArrayRef<VectorTensor> tensors, int64_t axisId) {
  ExprAttr extent;
  for (const VectorTensor &tensor : tensors) {
    FailureOr<size_t> position = axisPosition(tensor.descriptor, axisId);
    if (failed(position)) {
      return failure();
    }
    ExprAttr repeat = tensor.descriptor.getRepeats()[*position];
    if (provablyOne(repeat)) {
      continue;
    }
    if (!extent) {
      extent = repeat;
      continue;
    }
    if (!provablyEqual(extent, repeat)) {
      // 同一根轴上两个不同的非 1 extent：这不是广播（广播那一侧是 1），是矛盾的 descriptor。
      // 采用其中一个会让后面所有 stride 算式都建立在猜测上，所以报失败。
      return failure();
    }
  }
  if (extent) {
    return extent;
  }
  return makeExprAttr(context, kOne);
}

}  // namespace

static FailureOr<VectorPlan> mergeVectorSegments(Operation *op, VectorPlan plan, ArrayRef<int64_t> axes,
                                                 ArrayRef<ExprAttr> extents,
                                                 const SmallVector<SmallVector<ExprAttr>> &strides, bool compactTail) {
  MLIRContext *context = op->getContext();
  auto func = op->getParentOfType<func::FuncOp>();
  auto graph = func ? func->getAttrOfType<AscGraphAttrGroupsAttr>(kAscGraphAttrName) : AscGraphAttrGroupsAttr();
  auto canShrink = [&](int64_t axisId) {
    if (!graph) {
      return false;
    }
    for (AxisAttr axis : graph.getAxes()) {
      if (axis.getId() == axisId) {
        return axis.getSplitPairOtherId() >= 0 &&
               (axis.getAxisType() == AxisType::TileInner || axis.getAxisType() == AxisType::BlockInner);
      }
    }
    return false;
  };

  // 第二步：从最内往外逐对合并。段内→外收集，最后整体反转。
  const auto axisCount = static_cast<int64_t>(axes.size());
  SmallVector<VectorSegment> reversedSegments;
  int64_t inner = axisCount - 1;
  while (inner >= 0) {
    // 开一段：extent 与 stride 都先取最内轴自己的。
    VectorSegment segment;
    segment.axes.push_back(axes[inner]);
    segment.extent = extents[inner];
    segment.axisExtents.push_back(extents[inner]);
    segment.strides.assign(strides[inner].begin(), strides[inner].end());

    // 往外吞：候选轴 `outer` 与当前段最内侧已吞入的那根轴 `previous` 逐对比较。
    int64_t previous = inner;
    int64_t outer = inner - 1;
    while (outer >= 0) {
      // 标称布局保留尾块边界；紧凑的单位 stride 后缀随实际范围连续存储。
      const bool compactSuffix = compactTail && llvm::all_of(segment.strides, provablyOne);
      if (canShrink(axes[previous]) && !compactSuffix) {
        break;
      }
      bool mergeable = true;
      for (auto [tensorIndex, tensor] : llvm::enumerate(plan.tensors)) {
        FailureOr<ExprAttr> expected = multiply(context, strides[previous][tensorIndex], extents[previous]);
        if (failed(expected) || !provablyEqual(strides[outer][tensorIndex], *expected)) {
          mergeable = false;
          break;
        }
      }
      if (!mergeable) {
        break;
      }
      FailureOr<ExprAttr> extent = multiply(context, segment.extent, extents[outer]);
      if (failed(extent)) {
        return failure();
      }
      segment.extent = *extent;
      // 段内轴列表要外→内，而这里是由内往外吞，所以插到最前面。
      segment.axes.insert(segment.axes.begin(), axes[outer]);
      segment.axisExtents.insert(segment.axisExtents.begin(), extents[outer]);
      previous = outer;
      --outer;
    }

    reversedSegments.push_back(std::move(segment));
    inner = outer;
  }

  plan.segments.assign(std::make_move_iterator(reversedSegments.rbegin()),
                       std::make_move_iterator(reversedSegments.rend()));
  return plan;
}

FailureOr<VectorPlan> collapseVectorizedAxes(Operation *op, bool compactTail) {
  if (op == nullptr) {
    return failure();
  }
  MLIRContext *context = op->getContext();

  VectorPlan plan;
  plan.tensors = collectTensors(op);
  if (plan.tensors.empty()) {
    return failure();
  }

  // 所有张量必须谈论同一串向量化轴。legacy 的 `DimensionCollapse` 在这一条不成立时直接
  // 报错（`broadcast_api_call.cpp:41`），这里同样不猜——两串不同的轴之间没有「相邻」可言。
  ArrayRef<int64_t> vectorizedAxes = plan.tensors.front().descriptor.getVectorizedAxis();
  for (const VectorTensor &tensor : plan.tensors) {
    if (!hasConsistentLengths(tensor.descriptor) || tensor.descriptor.getVectorizedAxis() != vectorizedAxes) {
      return failure();
    }
  }

  // 第一步：丢掉 extent 可证为 1 的轴，并记下每根留下来的轴的 extent。
  // 丢轴必须在逐对比较之前做，否则一根 extent 为 1 的轴会把它两侧的相邻关系割断。
  SmallVector<int64_t> axes;
  SmallVector<ExprAttr> extents;
  for (int64_t axisId : vectorizedAxes) {
    FailureOr<ExprAttr> extent = commonExtent(context, plan.tensors, axisId);
    if (failed(extent)) {
      return failure();
    }
    if (provablyOne(*extent)) {
      continue;
    }
    axes.push_back(axisId);
    extents.push_back(*extent);
  }
  if (axes.empty()) {
    // 每根向量化轴的 extent 都是 1：这个 op 在向量化轴上没有工作量。调用方（常量源广播的
    // `Duplicate` 一档）自己认这个形态，本函数只如实报「没有段」。
    return plan;
  }

  // 每根留下来的轴上，各张量的 stride。下标与 `axes` 对齐。
  SmallVector<SmallVector<ExprAttr>> strides(axes.size());
  for (auto [axisIndex, axisId] : llvm::enumerate(axes)) {
    const auto *position = llvm::find(vectorizedAxes, axisId);
    const auto vectorizedIndex = static_cast<size_t>(position - vectorizedAxes.begin());
    for (const VectorTensor &tensor : plan.tensors) {
      strides[axisIndex].push_back(tensor.descriptor.getVectorizedStrides()[vectorizedIndex]);
    }
  }

  return mergeVectorSegments(op, std::move(plan), axes, extents, strides, compactTail);
}

}  // namespace mlir::afir
