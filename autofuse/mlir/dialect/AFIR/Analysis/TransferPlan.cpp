/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/Analysis/TransferPlan.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <string>

namespace mlir::afir {
namespace {
constexpr unsigned kBitsPerByte = 8;
constexpr size_t kTwoDimensions = 2;

bool isDirectProjection(AffineMap map) {
  if (map.getNumSymbols() != 0) {
    return false;
  }
  llvm::SmallDenseSet<unsigned, 8> dimensions;
  for (AffineExpr result : map.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(result);
    if (!dim || !dimensions.insert(dim.getPosition()).second) {
      return false;
    }
  }
  return true;
}

FailureOr<SmallVector<AffineMap>> getMaps(Operation *op) {
  auto maps = op->getAttrOfType<ArrayAttr>(afir::kIndexingMapsAttrName);
  if (!maps) {
    return SmallVector<AffineMap>{};
  }
  if (maps.size() != 2) {
    return failure();
  }
  SmallVector<AffineMap> result;
  result.reserve(2);
  for (Attribute attr : maps) {
    auto map = dyn_cast<AffineMapAttr>(attr);
    if (!map) {
      return failure();
    }
    result.push_back(map.getValue());
  }
  if (result[0].getNumDims() != result[1].getNumDims() || result[0].getNumSymbols() != result[1].getNumSymbols()) {
    return failure();
  }
  return result;
}

// 按使用点的 tensor descriptor 判断 canonical 契约；buffer_desc 在实际取用处校验。
bool hasCanonicalDescriptors(Operation *op) {
  return op != nullptr && op->hasAttr(afir::kOperandDescsAttrName) && op->hasAttr(afir::kDestinationDescsAttrName);
}

}  // namespace

// 局部视图取自 TensorDescAttr：Load 取结果，Store 取操作数。
// `Implementations/` 侧（`Transpose.cpp` / `Pad.cpp`）的形态判据取同一份事实，导出共用。
FailureOr<afir::TensorDescAttr> getLocalDescriptor(Operation *op, TransferPlanSide side) {
  // 显式 Operand 请求取输入侧；否则按操作结果选择局部视图，不跨侧回落。
  StringRef name = side == TransferPlanSide::Operand || op->getNumResults() == 0 ? afir::kOperandDescsAttrName
                                                                                 : afir::kResultDescsAttrName;
  auto descriptors = op->getAttrOfType<ArrayAttr>(name);
  if (!descriptors || descriptors.empty()) {
    return failure();
  }
  auto descriptor = dyn_cast<afir::TensorDescAttr>(descriptors[0]);
  if (!descriptor) {
    return failure();
  }
  return descriptor;
}

// 图上的轴表（`afir.asc_graph` 的 axes），按轴 id 索引。取不到（无 func 或无属性）失败。
// `Implementations/` 侧（`Pad.cpp` 的 `isTileInnerAxis`）与本文件共用同一份轴表事实。
FailureOr<llvm::DenseMap<int64_t, AxisAttr>> collectGraphAxes(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  auto graph = func ? func->getAttrOfType<AscGraphAttrGroupsAttr>(afir::kAscGraphAttrName) : AscGraphAttrGroupsAttr();
  if (!graph) {
    return failure();
  }
  llvm::DenseMap<int64_t, AxisAttr> axesById;
  for (AxisAttr axis : graph.getAxes()) {
    axesById[axis.getId()] = axis;
  }
  return axesById;
}

namespace {
FailureOr<ExprAttr> resolveGlobalStride(MLIRContext *context, const llvm::DenseMap<int64_t, AxisAttr> &axes,
                                        const llvm::DenseMap<int64_t, ExprAttr> &strides, int64_t id,
                                        llvm::SmallDenseSet<int64_t> &visiting);

// GM 偏移使用落点的 stride，不能用归约后 UB 的零 stride 替代。
// Load 沿用局部 descriptor 的 strides；Store 优先使用带 vectorized_axis 的调度目的视图。
// 图输出使用原始轴时，将目的 GM stride 投影到局部调度轴；不能沿用 NDDMA 输入的 stride。
// 显式 Operand 请求保持输入侧视图。
FailureOr<afir::TensorDescAttr> getGlobalDescriptor(Operation *op, TransferPlanSide side,
                                                    afir::TensorDescAttr localDescriptor) {
  if (side == TransferPlanSide::Operand || op->getNumResults() != 0) {
    return localDescriptor;
  }
  auto descriptors = op->getAttrOfType<ArrayAttr>(afir::kDestinationDescsAttrName);
  if (!descriptors || descriptors.empty()) {
    return failure();
  }
  auto descriptor = dyn_cast<afir::TensorDescAttr>(descriptors[0]);
  if (!descriptor) {
    return failure();
  }
  if (!descriptor.getVectorizedAxis().empty()) {
    return descriptor;
  }
  // Some plain (unscheduled) destination descriptors carry only `axis`, with
  // no `strides` at all (e.g. a bare kernel-argument `tensor_desc`). There is
  // then nothing to compare against or project from, so this axis/stride
  // cross-check — and the projection fallback below, which needs the same
  // pairing — are both skipped; only the vectorized self-consistency check
  // decides the outcome, same as before this axis/stride table existed.
  const bool destinationHasStrideTable = descriptor.getAxis().size() == descriptor.getStrides().size();
  llvm::DenseMap<int64_t, ExprAttr> destinationStrides;
  if (destinationHasStrideTable) {
    for (auto [axis, stride] : llvm::zip_equal(descriptor.getAxis(), descriptor.getStrides())) {
      destinationStrides.try_emplace(axis, stride);
    }
  }
  // A scheduled layout is already usable when its GM and UB strides agree
  // on every vectorized axis (including identity views with no such axes)
  // AND every other axis it shares (by id) with the real destination
  // descriptor agrees with that descriptor's stride. The second half guards
  // a producer's own non-vectorized axis: its `strides` column there can
  // describe a broadcast/reduction relationship relative to the UB
  // computation (e.g. 0 because the value doesn't vary along that axis)
  // rather than the true GM address delta for that axis on this op's own
  // destination. Blindly trusting it drops that axis from the GM offset.
  // NDDMA breaks the first half: its strides still describe the input GM
  // layout while vectorized_strides describe the broadcast output tile.
  bool hasScheduledLayout = true;
  for (auto [axis, localStride] :
       llvm::zip_equal(localDescriptor.getVectorizedAxis(), localDescriptor.getVectorizedStrides())) {
    auto found = llvm::find(localDescriptor.getAxis(), axis);
    if (found == localDescriptor.getAxis().end()) {
      return failure();
    }
    const size_t index = std::distance(localDescriptor.getAxis().begin(), found);
    if (index >= localDescriptor.getStrides().size()) {
      return failure();
    }
    auto stride = localDescriptor.getStrides()[index];
    if (!ProveExpressionsEqual(stride.getValue().getValue(), localStride.getValue().getValue())) {
      hasScheduledLayout = false;
      break;
    }
  }
  if (hasScheduledLayout && destinationHasStrideTable &&
      localDescriptor.getAxis().size() == localDescriptor.getStrides().size()) {
    for (auto [axis, localStride] : llvm::zip_equal(localDescriptor.getAxis(), localDescriptor.getStrides())) {
      auto destinationStride = destinationStrides.find(axis);
      if (destinationStride != destinationStrides.end() &&
          !ProveExpressionsEqual(localStride.getValue().getValue(), destinationStride->second.getValue().getValue())) {
        hasScheduledLayout = false;
        break;
      }
    }
  }
  if (hasScheduledLayout) {
    return localDescriptor;
  }
  if (!destinationHasStrideTable) {
    return failure();
  }
  auto func = op->getParentOfType<func::FuncOp>();
  auto graph = func ? func->getAttrOfType<AscGraphAttrGroupsAttr>(kAscGraphAttrName) : AscGraphAttrGroupsAttr();
  if (!graph) {
    return failure();
  }
  llvm::DenseMap<int64_t, AxisAttr> axes;
  for (AxisAttr axis : graph.getAxes()) {
    axes.try_emplace(axis.getId(), axis);
  }
  SmallVector<ExprAttr> scheduledStrides;
  for (int64_t axis : localDescriptor.getAxis()) {
    llvm::SmallDenseSet<int64_t> visiting;
    auto stride = resolveGlobalStride(op->getContext(), axes, destinationStrides, axis, visiting);
    if (failed(stride)) {
      // Graph-level axis merging may erase the relation to the original
      // output axes. In that representation the scheduled view is the only
      // available address layout; retain the existing scheduled-view contract.
      return localDescriptor;
    }
    scheduledStrides.push_back(*stride);
  }
  return TensorDescAttr::get(op->getContext(), localDescriptor.getAxis(), localDescriptor.getRepeats(),
                             scheduledStrides, localDescriptor.getVectorizedAxis(),
                             localDescriptor.getVectorizedStrides(), localDescriptor.getRole());
}

// 收集尺寸/步长表达式中的自由符号。
LogicalResult collectFreeSymbols(MLIRContext *context, ExprAttr expression, llvm::SetVector<StringRef> &symbols) {
  if (!expression) {
    return success();
  }
  auto parsed = Expression::Parse(expression.getValue().getValue());
  if (!parsed) {
    return failure();
  }
  auto free = detail::ExpressionAccess::GetFreeSymbols(parsed.value());
  if (!free) {
    return failure();
  }
  for (const std::string &symbol : free.value()) {
    // StringRef 不能指向这个临时 std::string，交给 context 的字符串池持有。
    symbols.insert(StringAttr::get(context, symbol).getValue());
  }
  return success();
}

// 步长符号必须属于 asc_graph 的 size_var 闭包，确保能绑定到有效 tiling 参数。
bool stridesAreSymbolClosed(MLIRContext *context, AscGraphAttrGroupsAttr graph, afir::TensorDescAttr descriptor) {
  llvm::SetVector<StringRef> available;
  for (ExprAttr sizeVariable : graph.getSizeVar()) {
    if (failed(collectFreeSymbols(context, sizeVariable, available))) {
      return false;
    }
  }
  llvm::SetVector<StringRef> used;
  for (ExprAttr stride : descriptor.getStrides()) {
    if (failed(collectFreeSymbols(context, stride, used))) {
      return false;
    }
  }
  for (ExprAttr stride : descriptor.getVectorizedStrides()) {
    if (failed(collectFreeSymbols(context, stride, used))) {
      return false;
    }
  }
  return llvm::all_of(used, [&](StringRef symbol) { return available.contains(symbol); });
}

// 仅在符号层可证明相等时合并；Unknown 不视为相等，不使用字符串比较。
bool provenEqualTo(ExprAttr expression, StringRef literal) {
  ExprAttr constant = ExprAttr::get(expression.getContext(), StringAttr::get(expression.getContext(), literal));
  return proveEqual(expression, constant) == detail::TruthValue::True;
}

bool isProvenZero(ExprAttr expression) {
  return provenEqualTo(expression, "0");
}

bool isProvenOne(ExprAttr expression) {
  return provenEqualTo(expression, "1");
}

// 一段布尔谓词文本交给符号层判定。证不了（含解析、化简失败）一律 Unknown。
// `proveEqual`（文件尾）与下方两个防御判据共用这一条管线，避免 ExpectEq/Lt/Mod
// 各自复制一份 Parse-Simplify-Prove。
detail::TruthValue provePredicateText(StringRef text) {
  Result<Expression> parsed = Expression::ParseRelaxed(text);
  if (!parsed) {
    return detail::TruthValue::Unknown;
  }
  Result<Expression> simplified = parsed.value().Simplify();
  if (!simplified) {
    return detail::TruthValue::Unknown;
  }
  Result<detail::TruthValue> proof = detail::ExpressionAccess::Prove(simplified.value());
  return proof ? proof.value() : detail::TruthValue::Unknown;
}

// `lhs < rhs` 可证成立吗。与 `provenEqualTo` 同方向：只有 True 才算数。
bool isProvenLessThan(ExprAttr lhs, ExprAttr rhs) {
  if (!lhs || !rhs) {
    return false;
  }
  const std::string text = ("ExpectLt((" + lhs.getValue().getValue() + "), (" + rhs.getValue().getValue() + "))").str();
  return provePredicateText(text) == detail::TruthValue::True;
}

// `expression` 可证**不是** `factor` 的整数倍吗。与 RemovePad 档的 `isProvablyMultipleOf`
// 方向相反：那侧「非可证倍数即拒」（行距恒为对齐形态，答不了就是破契约）；本侧
// 「可证非倍数才拒」——这里的行距大量是符号形态（`16 * Ceiling(...)` 一类），倍数性
// 答不了是常态，方向反了会把合法调度产物整片误杀。
bool isProvenNotMultipleOf(ExprAttr expression, int64_t factor) {
  if (!expression || factor <= 0) {
    return false;
  }
  const std::string text =
      ("ExpectEq(Mod((" + expression.getValue().getValue() + "), " + std::to_string(factor) + "), 0)").str();
  return provePredicateText(text) == detail::TruthValue::False;
}

// 两个表达式的乘积，交给符号层化简。
FailureOr<ExprAttr> multiplyExpressions(MLIRContext *context, ExprAttr lhs, ExprAttr rhs) {
  if (!lhs || !rhs) {
    return failure();
  }
  // 生成的乘积可能需要重排因子，使用宽松解析后规范化。
  Result<Expression> parsed =
      Expression::ParseRelaxed(("(" + lhs.getValue().getValue() + ") * (" + rhs.getValue().getValue() + ")").str());
  if (!parsed) {
    return failure();
  }
  Result<Expression> simplified = parsed.value().Simplify();
  if (!simplified) {
    return failure();
  }
  return makeExprAttr(context, simplified.value().Serialize());
}

// 沿用 legacy ShouldIgnoreDataCopyZeroAxis：两侧零 stride 轴仍可能需保留 DMA 重复维。
bool shouldIgnoreZeroAxis(bool hasNonZeroAxis, size_t position, ArrayRef<ExprAttr> localStrides) {
  return hasNonZeroAxis || position == 0 || isProvenOne(localStrides[position - 1]) ||
         isProvenZero(localStrides[position - 1]);
}

afir::AxisType axisTypeOf(AscGraphAttrGroupsAttr graph, int64_t axisId) {
  for (AxisAttr axis : graph.getAxes()) {
    if (axis.getId() == axisId) {
      return axis.getAxisType();
    }
  }
  return afir::AxisType::Invalid;
}

// 按 legacy CalculateDmaParams 从内向外归约单张量的有效视图。
// 全图合轴不能替代这里针对各张量 stride 的判断。
bool transferStridesAreContiguous(MLIRContext *context, ExprAttr previousGlobalStride, ExprAttr previousLocalStride,
                                  ExprAttr previousExtent, ExprAttr globalStride, ExprAttr localStride,
                                  bool hasInnerDimension) {
  FailureOr<ExprAttr> expectedGlobal = hasInnerDimension
                                           ? multiplyExpressions(context, previousGlobalStride, previousExtent)
                                           : FailureOr<ExprAttr>(failure());
  FailureOr<ExprAttr> expectedLocal = hasInnerDimension
                                          ? multiplyExpressions(context, previousLocalStride, previousExtent)
                                          : FailureOr<ExprAttr>(failure());
  return succeeded(expectedGlobal) && succeeded(expectedLocal) &&
         proveEqual(*expectedGlobal, globalStride) == detail::TruthValue::True &&
         proveEqual(*expectedLocal, localStride) == detail::TruthValue::True;
}

LogicalResult appendDegenerateTransferDim(MLIRContext *, afir::TensorDescAttr descriptor,
                                          SmallVectorImpl<TransferDim> &dims) {
  auto axes = descriptor.getAxis();
  auto vectorizedAxes = descriptor.getVectorizedAxis();
  auto repeats = descriptor.getRepeats();
  auto globalStrides = descriptor.getStrides();
  auto localStrides = descriptor.getVectorizedStrides();
  // 全部轴被剔除时仍保留最内轴，表达常量源的一元素搬运；没有向量化轴则失败。
  if (vectorizedAxes.empty()) {
    return failure();
  }
  const auto *innermost = llvm::find(axes, vectorizedAxes.back());
  if (innermost == axes.end()) {
    return failure();
  }
  const size_t innermostPosition = std::distance(axes.begin(), innermost);
  if (!repeats[innermostPosition] || !globalStrides[innermostPosition] || !localStrides.back()) {
    return failure();
  }
  dims.push_back(TransferDim{repeats[innermostPosition], globalStrides[innermostPosition], localStrides.back()});
  return success();
}

bool canMergeTransferAxis(MLIRContext *context, AscGraphAttrGroupsAttr graph, ArrayRef<int64_t> vectorizedAxes,
                          size_t position, ExprAttr previousGlobalStride, ExprAttr previousLocalStride,
                          ExprAttr previousExtent, ExprAttr globalStride, ExprAttr localStride, bool hasInner) {
  const bool innerIsTileInner = position + 1 < vectorizedAxes.size() &&
                                axisTypeOf(graph, vectorizedAxes[position + 1]) == afir::AxisType::TileInner;
  const bool contiguous = transferStridesAreContiguous(context, previousGlobalStride, previousLocalStride,
                                                       previousExtent, globalStride, localStride, hasInner);
  return hasInner && !innerIsTileInner && contiguous;
}

LogicalResult reduceEffectiveView(MLIRContext *context, AscGraphAttrGroupsAttr graph, afir::TensorDescAttr descriptor,
                                  SmallVectorImpl<TransferDim> &dims) {
  ArrayRef<int64_t> axes = descriptor.getAxis();
  ArrayRef<ExprAttr> repeats = descriptor.getRepeats();
  ArrayRef<ExprAttr> globalStrides = descriptor.getStrides();
  ArrayRef<int64_t> vectorizedAxes = descriptor.getVectorizedAxis();
  ArrayRef<ExprAttr> localStrides = descriptor.getVectorizedStrides();
  // `vectorized_axis` 与 `vectorized_strides` 是外部输入（调度器写的）的一对对齐表。
  if (vectorizedAxes.size() != localStrides.size() || axes.size() != repeats.size() ||
      axes.size() != globalStrides.size()) {
    return failure();
  }

  // 上一项（更内层）的三个量，用来判断当前轴是否与它连续。
  ExprAttr previousGlobalStride;
  ExprAttr previousLocalStride;
  ExprAttr previousExtent;
  bool hasNonZeroAxis = false;
  bool droppedInnermostUnit = false;

  for (size_t position = vectorizedAxes.size(); position-- > 0;) {
    const int64_t axisId = vectorizedAxes[position];
    const auto *found = llvm::find(axes, axisId);
    // 向量化轴缺失于完整轴表属于契约错误。
    if (found == axes.end()) {
      return failure();
    }
    const size_t axisPosition = std::distance(axes.begin(), found);
    ExprAttr extent = repeats[axisPosition];
    ExprAttr globalStride = globalStrides[axisPosition];
    ExprAttr localStride = localStrides[position];
    if (!extent || !globalStride || !localStride) {
      return failure();
    }

    // 规则 1：范围为 1 的轴不占一维。
    if (isProvenOne(extent)) {
      droppedInnermostUnit |= dims.empty();
      continue;
    }
    // 规则 2：两侧 stride 都为 0 且满足 legacy 的保留条件时跳过。
    if (isProvenZero(globalStride) && isProvenZero(localStride) &&
        shouldIgnoreZeroAxis(hasNonZeroAxis, position, localStrides)) {
      continue;
    }
    hasNonZeroAxis = true;

    // 规则 3：GM 和 UB 两侧 stride 均等于内层 stride × extent 时才合并。
    // 规则 4：不跨相邻的 TileInner 边界合并，保留内层尾块的独立范围。
    const bool mergeable =
        canMergeTransferAxis(context, graph, vectorizedAxes, position, previousGlobalStride, previousLocalStride,
                             previousExtent, globalStride, localStride, !dims.empty());
    if (!mergeable) {
      dims.push_back(TransferDim{extent, globalStride, localStride});
      previousGlobalStride = globalStride;
      previousLocalStride = localStride;
      previousExtent = extent;
      continue;
    }
    FailureOr<ExprAttr> merged = multiplyExpressions(context, extent, dims.back().extent);
    if (failed(merged)) {
      return failure();
    }
    // 合并只改 extent：这一项的两侧 stride 仍然是最内那根轴的，合并出来的范围是乘积。
    dims.back().extent = *merged;
    previousExtent = *merged;
  }

  std::reverse(dims.begin(), dims.end());
  if (!dims.empty()) {
    // Removing a unit inner axis must not turn a strided row into blockLen
    // contiguous elements. E.g. [R, 1] with GM pitch 1 and UB pitch 8 still
    // needs R one-element DMA blocks, each padded to the UB row boundary.
    const TransferDim &inner = dims.back();
    if (droppedInnermostUnit && !isProvenOne(inner.localStride) && !isProvenZero(inner.localStride)) {
      ExprAttr one = ExprAttr::get(context, StringAttr::get(context, "1"));
      dims.push_back(TransferDim{one, one, one});
    }
    return success();
  }

  return appendDegenerateTransferDim(context, descriptor, dims);
}

// 一根轴的**原始轴集合**：沿 `from` 链上溯，收集所有 Original 轴。合轴（Merged）有多个来源，
// 于是集合多于一项——这正是需要集合而不是单根轴的原因：先合轴再分块的调度里，块轴同时含着
// 好几维。
//
// 工作表的长度以轴表大小为界。轴表由 bridge 构造，`from` 链有限且无环；这个界只是防止非法
// 轴表把这里变成死循环。链上出现表里没有的 id 时返回空集，调用方按「解析不出」处理。
llvm::SmallDenseSet<int64_t> collectOriginAxes(const llvm::DenseMap<int64_t, AxisAttr> &axes, int64_t id) {
  llvm::SmallDenseSet<int64_t> origins;
  llvm::SmallVector<int64_t> pending{id};
  for (size_t step = 0; step <= axes.size() && !pending.empty(); ++step) {
    const int64_t current = pending.pop_back_val();
    auto found = axes.find(current);
    if (found == axes.end()) {
      return {};
    }
    AxisAttr axis = found->second;
    if (axis.getAxisType() == AxisType::Original) {
      origins.insert(current);
      continue;
    }
    ArrayRef<int64_t> from = axis.getFrom();
    if (from.empty()) {
      return {};
    }
    pending.append(from.begin(), from.end());
  }
  return pending.empty() ? origins : llvm::SmallDenseSet<int64_t>{};
}

bool coversAll(const llvm::SmallDenseSet<int64_t> &outer, const llvm::SmallDenseSet<int64_t> &inner) {
  return llvm::all_of(inner, [&outer](int64_t id) { return outer.contains(id); });
}

bool disjoint(const llvm::SmallDenseSet<int64_t> &lhs, const llvm::SmallDenseSet<int64_t> &rhs) {
  return llvm::none_of(rhs, [&lhs](int64_t id) { return lhs.contains(id); });
}

// 缓存守卫要用的那几根轴与维度关系（`materializeExecConditionGuard`）。
//
// **块轴按「原始轴集合覆盖切分轴那一维」取，不按「与 `loop_axis` 配对」取。** 后者只在一维
// 调度上成立：那时 `loop_axis` 恰好是块内轴。多维调度里 `loop_axis` 往往是另一维的 TileOuter，
// 于是块轴一根都解析不出来，fused 取值整类回落（遗留问题清单 BR-1）。周期长度由切分轴那一维
// 决定，块轴要取的也是含着同一维的那一对，与循环跑在哪根轴上无关。
//
// 用集合而不是单根原始轴，是因为先合轴再分块是常见形态：`z0z1T` 这种合轴同时含 z0 与 z1，
// 它上面的块轴照样推进切分轴那一维。按单根原始轴判断会把这类调度判成「解析不出」，而它们
// 本来是能生成的。
void collectGuardAxes(AscGraphAttrGroupsAttr graph, TransferPlan &plan) {
  if (plan.tileOuterAxis < 0) {
    return;
  }
  llvm::DenseMap<int64_t, AxisAttr> axes;
  for (AxisAttr axis : graph.getAxes()) {
    axes.try_emplace(axis.getId(), axis);
  }
  const llvm::SmallDenseSet<int64_t> tileOrigins = collectOriginAxes(axes, plan.tileOuterAxis);
  if (tileOrigins.empty()) {
    return;
  }
  for (AxisAttr axis : graph.getAxes()) {
    if (axis.getAxisType() != AxisType::BlockOuter || !coversAll(collectOriginAxes(axes, axis.getId()), tileOrigins)) {
      continue;
    }
    plan.blockOuterAxis = axis.getId();
    plan.blockInnerAxis = axis.getSplitPairOtherId();
    break;
  }
  const llvm::SmallDenseSet<int64_t> loopOrigins = collectOriginAxes(axes, plan.loopAxis);
  if (loopOrigins.empty()) {
    return;
  }
  if (coversAll(loopOrigins, tileOrigins)) {
    plan.guardLoopRelation = GuardLoopRelation::SameDimension;
  } else if (disjoint(loopOrigins, tileOrigins)) {
    plan.guardLoopRelation = GuardLoopRelation::OtherDimension;
  }
  // 部分重叠留在 `Unresolved`：循环推进时切分轴那一维只在某些迭代上前进，相位与「只算首趟」
  // 都说不对，交给守卫报失败。
}

LogicalResult collectTransferOuterAxes(AscGraphAttrGroupsAttr graph, ArrayRef<int64_t> vectorizedAxes,
                                       ArrayRef<int64_t> globalAxes, ArrayRef<ExprAttr> globalStrides,
                                       TransferPlan &plan) {
  for (AxisAttr axis : graph.getAxes()) {
    if (axis.getId() == vectorizedAxes.front() && axis.getAxisType() == AxisType::TileInner) {
      plan.tileInnerAxis = axis.getId();
      plan.tileOuterAxis = axis.getSplitPairOtherId();
    }
  }
  collectGuardAxes(graph, plan);
  for (auto [axisId, globalStride] : llvm::zip_equal(globalAxes, globalStrides)) {
    if (llvm::is_contained(vectorizedAxes, axisId)) {
      continue;
    }
    if (!globalStride) {
      return failure();
    }
    plan.outerAxes.push_back(TransferOuterAxis{axisId, globalStride});
  }
  return success();
}

bool hasProjectedTransferDescriptors(ArrayRef<AffineMap> maps, TensorDescAttr localDescriptor,
                                     TensorDescAttr globalDescriptor) {
  return localDescriptor && globalDescriptor &&
         (maps.empty() ||
          (maps.size() == kTwoDimensions && isDirectProjection(maps[0]) && isDirectProjection(maps[1])));
}

// failure：契约错误，禁止回落 identity。
// Identity：调度路径不适用，调用方仍须校验 identity 条件。
// Scheduled：调度计划构造成功。
FailureOr<TransferPlan> buildScheduledPlan(Operation *op, ArrayRef<AffineMap> maps,
                                           afir::TensorDescAttr localDescriptor,
                                           afir::TensorDescAttr globalDescriptor) {
  auto sched = op->getAttrOfType<afir::ScheduleAttr>(afir::kSchedAttrName);
  if (!hasCanonicalDescriptors(op) || !sched || sched.getLoopAxis() < 0) {
    return TransferPlan{};
  }
  if (!hasProjectedTransferDescriptors(maps, localDescriptor, globalDescriptor)) {
    return TransferPlan{};
  }

  ArrayRef<int64_t> vectorizedAxes = localDescriptor.getVectorizedAxis();
  // 无向量化轴时交由 identity 路径判断；多维搬运不在这里限制轴数。
  if (vectorizedAxes.empty()) {
    return TransferPlan{};
  }

  auto func = op->getParentOfType<func::FuncOp>();
  if (!func) {
    return TransferPlan{};
  }
  auto graph = func->getAttrOfType<AscGraphAttrGroupsAttr>(afir::kAscGraphAttrName);
  if (!graph) {
    return TransferPlan{};
  }
  if (!stridesAreSymbolClosed(op->getContext(), graph, localDescriptor)) {
    return failure();
  }
  // GM stride 同样需要符号闭合，以便运行期物化。
  if (globalDescriptor != localDescriptor && !stridesAreSymbolClosed(op->getContext(), graph, globalDescriptor)) {
    return failure();
  }

  // 从 GM 轴表中排除局部 vectorized_axis，剩余轴及其 GM stride 用于 tile 起点偏移。
  // 必须遍历 GM 侧，保留 workspace Store 目的独有的块索引轴。
  ArrayRef<int64_t> axes = localDescriptor.getAxis();
  ArrayRef<ExprAttr> strides = localDescriptor.getStrides();
  // GM 轴与 stride 按下标对应，先校验长度。
  if (axes.size() != strides.size()) {
    return failure();
  }
  ArrayRef<int64_t> globalAxes = globalDescriptor.getAxis();
  ArrayRef<ExprAttr> globalStrides = globalDescriptor.getStrides();
  if (globalAxes.size() != globalStrides.size()) {
    return failure();
  }
  TransferPlan plan;
  plan.kind = TransferPlanKind::Scheduled;
  plan.loopAxis = sched.getLoopAxis();
  if (failed(collectTransferOuterAxes(graph, vectorizedAxes, globalAxes, globalStrides, plan))) {
    return failure();
  }
  // 没有 tile 外轴时交由 identity 检查，保留整块搬运路径。
  if (plan.outerAxes.empty()) {
    return TransferPlan{};
  }

  // 有效视图构造失败属于契约错误。
  // When the GM view is projected onto the local axes, use its strides for
  // both the tile origin and DMA rows. UB strides remain those of the local view.
  auto transferDescriptor =
      globalDescriptor.getAxis() == localDescriptor.getAxis()
          ? TensorDescAttr::get(op->getContext(), localDescriptor.getAxis(), localDescriptor.getRepeats(),
                                globalDescriptor.getStrides(), localDescriptor.getVectorizedAxis(),
                                localDescriptor.getVectorizedStrides(), localDescriptor.getRole())
          : localDescriptor;
  if (failed(reduceEffectiveView(op->getContext(), graph, transferDescriptor, plan.dims))) {
    return failure();
  }
  // 最内两项供 DMA，其余项生成外循环；计算算子的布局限制由各自能力查询负责。
  return plan;
}

// 通过 axis_binding 恢复生成后 SSA 值的轴身份和角色。
Value findAxisValueImpl(func::FuncOp func, Operation *user, int64_t axisId, afir::AxisValueRole role) {
  DominanceInfo dominance(func);
  Value result;
  func.walk([&](Operation *candidate) {
    if (result || candidate == user || !dominance.dominates(candidate, user)) {
      return;
    }
    if (!afir::hasAxisBinding(candidate, axisId, role)) {
      return;
    }
    if (candidate->getNumResults() != 0) {
      result = candidate->getResult(0);
    }
  });
  return result;
}

scf::ForOp findLoop(Operation *op, int64_t axisId) {
  for (Operation *parent = op->getParentOp(); parent; parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop) {
      continue;
    }
    if (afir::hasAxisBinding(loop, axisId, afir::AxisValueRole::Loop)) {
      return loop;
    }
  }
  return {};
}

// 外层轴优先取 scf.for 归纳变量；无循环的 BlockOuter 取 AxisIndex 绑定。
// 不可颠倒查找顺序，源轴的 AxisIndex 可能包含额外块偏移。
Value findOuterAxisIndex(func::FuncOp func, Operation *op, int64_t axisId) {
  if (scf::ForOp loop = findLoop(op, axisId)) {
    return loop.getInductionVar();
  }
  return findAxisValueImpl(func, op, axisId, afir::AxisValueRole::AxisIndex);
}

// 先按名字收集函数入口块已有的 tiling 参数。descriptor 里首次出现的符号由
// `materializeExpression` 用调用方的 builder 补建，以便 dialect conversion 跟踪并下降。
SymbolBindingMap collectTilingParamBindings(func::FuncOp func) {
  SymbolBindingMap bindings;
  for (Operation &operation : func.getBody().front()) {
    auto param = dyn_cast<TilingParamOp>(operation);
    if (!param) {
      continue;
    }
    FailureOr<ExprAttr> symbol = makeExprAttr(func.getContext(), param.getSymbol());
    if (succeeded(symbol)) {
      bindings.insert({*symbol, param.getResult()});
    }
  }
  return bindings;
}

// 字面量 stride 直接物化为常量。
FailureOr<int64_t> getLiteralStride(ExprAttr expression) {
  if (!expression) {
    return failure();
  }
  int64_t value = 0;
  if (expression.getValue().getValue().trim().getAsInteger(10, value)) {
    return failure();
  }
  return value;
}

// 仅切分 inner 轴可能有尾块；Original、Merged 和切分 outer 使用标称范围。
bool axisCanHaveTail(AxisAttr axis) {
  const bool isSplitInner =
      axis.getAxisType() == afir::AxisType::TileInner || axis.getAxisType() == afir::AxisType::BlockInner;
  return isSplitInner && axis.getSplitPairOtherId() >= 0;
}

// 将 extent 中的轴尺寸符号替换为 ActualSize，等价于 legacy ActualSize(expr)。
// 可能有尾块的轴缺少绑定时必须失败，不得用标称范围代替。
FailureOr<SymbolBindingMap> collectExtentBindings(func::FuncOp func, Operation *op, AscGraphAttrGroupsAttr graph,
                                                  ArrayRef<TransferDim> dims) {
  SymbolBindingMap bindings = collectTilingParamBindings(func);
  llvm::SetVector<StringRef> boundToActual;
  llvm::SetVector<StringRef> needsActual;
  for (AxisAttr axis : graph.getAxes()) {
    // 仅替换裸尺寸符号；无尾块的派生 outer 表达式不参与替换。
    if (!axis.getSize()) {
      continue;
    }
    const StringRef symbol = axis.getSize().getValue().getValue();
    if (axisCanHaveTail(axis)) {
      needsActual.insert(symbol);
    }
    if (Value actual = findAxisValueImpl(func, op, axis.getId(), afir::AxisValueRole::ActualSize)) {
      bindings[axis.getSize()] = actual;
      boundToActual.insert(symbol);
    }
  }

  llvm::SetVector<StringRef> used;
  for (const TransferDim &dim : dims) {
    if (failed(collectFreeSymbols(op->getContext(), dim.extent, used))) {
      return failure();
    }
  }
  for (StringRef symbol : used) {
    if (needsActual.contains(symbol) && !boundToActual.contains(symbol)) {
      return failure();
    }
  }
  return bindings;
}

// 表达式 -> index 值。字面量直接建常量，符号走 `materializeSymbolicExpr`。
FailureOr<Value> materializeExpression(OpBuilder &builder, Location location, ExprAttr expression,
                                       const SymbolBindingMap &bindings) {
  FailureOr<int64_t> literal = getLiteralStride(expression);
  if (succeeded(literal)) {
    return builder.create<arith::ConstantIndexOp>(location, *literal).getResult();
  }
  llvm::SetVector<StringRef> symbols;
  if (failed(collectFreeSymbols(builder.getContext(), expression, symbols))) {
    return failure();
  }
  SymbolBindingMap resolved = bindings;
  for (StringRef symbol : symbols) {
    FailureOr<ExprAttr> key = makeExprAttr(builder.getContext(), symbol);
    if (failed(key)) {
      return failure();
    }
    if (resolved.count(*key)) {
      continue;
    }
    Operation *parent = builder.getInsertionBlock()->getParentOp();
    auto func = dyn_cast<func::FuncOp>(parent);
    if (!func) {
      func = parent->getParentOfType<func::FuncOp>();
    }
    if (!func) {
      return failure();
    }
    resolved[*key] = getOrCreateTilingParam(builder, func.getBody().front(), symbol).getResult();
  }
  return materializeSymbolicExpr(builder, location, expression, resolved, builder.getIndexType());
}

// 外循环的不变量；空 stride 表示该空间偏移恒为 0。
struct OuterLoopTerm {
  Value bound;
  Value globalStride;
  Value localStride;
};

// 物化一个步长，字面量 0 时把 `result` 留空表示「这一项不生成」。
LogicalResult materializeStrideOrSkip(OpBuilder &builder, Location location, ExprAttr stride,
                                      const SymbolBindingMap &bindings, Value &result) {
  FailureOr<int64_t> literal = getLiteralStride(stride);
  if (succeeded(literal) && *literal == 0) {
    result = Value();
    return success();
  }
  FailureOr<Value> value = materializeExpression(builder, location, stride, bindings);
  if (failed(value)) {
    return failure();
  }
  result = *value;
  return success();
}

}  // namespace

Value findAxisValue(func::FuncOp func, Operation *user, int64_t axisId, afir::AxisValueRole role) {
  return findAxisValueImpl(func, user, axisId, role);
}

detail::TruthValue proveEqual(ExprAttr lhs, ExprAttr rhs) {
  if (!lhs || !rhs) {
    return detail::TruthValue::Unknown;
  }
  const std::string text = ("ExpectEq(" + lhs.getValue().getValue() + ", " + rhs.getValue().getValue() + ")").str();
  // ExpectEq(a, a) 会规范化为 True，需使用宽松解析；其余管线见 provePredicateText。
  return provePredicateText(text);
}

namespace {

// 两个入口共用的计划构造；无 sched 时允许空 descriptor 并跳过调度路径。
FailureOr<TransferPlan> buildPlanFromLocalDescriptor(Operation *op, afir::TensorDescAttr localDescriptor,
                                                     afir::TensorDescAttr globalDescriptor) {
  FailureOr<SmallVector<AffineMap>> maps = getMaps(op);
  if (failed(maps)) {
    return failure();
  }
  if (op->hasAttr(afir::kSchedAttrName)) {
    FailureOr<TransferPlan> scheduled = buildScheduledPlan(op, *maps, localDescriptor, globalDescriptor);
    // 调度路径失败必须传播，不能因 indexing_maps 为 identity 而改成整块搬运。
    if (failed(scheduled)) {
      return failure();
    }
    if (scheduled->kind == TransferPlanKind::Scheduled) {
      return scheduled;
    }
    // 调度路径不适用时，仍须通过下方 indexing_maps 的 identity 检查。
  }
  if (maps->size() == 2 && (*maps)[0].isIdentity() && (*maps)[1].isIdentity()) {
    return TransferPlan{};
  }
  if (maps->empty()) {
    return TransferPlan{};
  }
  return failure();
}

}  // namespace

FailureOr<TransferPlan> buildTransferPlan(Operation *op, TransferPlanSide side) {
  afir::TensorDescAttr local;
  if (op->hasAttr(afir::kSchedAttrName)) {
    FailureOr<afir::TensorDescAttr> resolved = getLocalDescriptor(op, side);
    if (failed(resolved)) {
      return failure();
    }
    local = *resolved;
  }
  // 局部侧为空（没有 `afir.sched`）时调度分支整个跳过，GM 侧也就无从谈起，一并留空。
  afir::TensorDescAttr global = local;
  if (local) {
    FailureOr<afir::TensorDescAttr> resolved = getGlobalDescriptor(op, side, local);
    if (failed(resolved)) {
      return failure();
    }
    global = *resolved;
  }
  return buildPlanFromLocalDescriptor(op, local, global);
}

bool blockDmaSegmentGeometryIsSound(Operation *op, const TransferPlan &plan) {
  // 只看喂给 DMA 的那一对维（行维 + 最内维，与 `emitScheduledTransfer` 的
  // `inner = dims.take_back(2)` 同一口径）；更外层的维只生成循环偏移，不进 DMA 参数。
  if (plan.kind != TransferPlanKind::Scheduled || plan.dims.size() < kTwoDimensions) {
    return true;
  }
  const TransferDim &row = plan.dims[plan.dims.size() - 2];
  const TransferDim &inner = plan.dims.back();
  // UB 行不能重叠；GM 的跨步转置读写不受此约束。
  if (isProvenLessThan(row.localStride, inner.extent)) {
    return false;
  }
  // DMA 的 UB 步距按 32B 量化。只拒绝可证不整除的行距；未知符号保守放行。
  // Store 的零间隙另行放行，因为物理布局由生产者决定。推进公式与实测依据见
  // 2026-10-19-afir-transpose-e2e-design.md 的“已补齐的覆盖欠账”。
  if (op == nullptr || op->getNumOperands() < 1) {
    return true;
  }
  auto shaped = dyn_cast<ShapedType>(op->getOperand(0).getType());
  if (!shaped || !shaped.getElementType().isIntOrFloat()) {
    return true;
  }
  const unsigned bits = shaped.getElementType().getIntOrFloatBitWidth();
  constexpr int64_t kUbAccessGranularityBytes = 32;
  if (bits % kBitsPerByte != 0 || kUbAccessGranularityBytes % (bits / kBitsPerByte) != 0) {
    return true;
  }
  const int64_t granularity = kUbAccessGranularityBytes / (bits / 8);
  if (isa<afir::StoreOp>(op) && proveEqual(row.localStride, inner.extent) == detail::TruthValue::True) {
    return true;
  }
  return !isProvenNotMultipleOf(row.localStride, granularity);
}

FailureOr<TransferPlan> buildTransferPlanForDescriptor(Operation *op, afir::TensorDescAttr localDescriptor) {
  if (!op || !localDescriptor) {
    return failure();
  }
  // 调用方直接给的是一份局部视图，没有「另一侧」可言：两个位置传同一份。
  return buildPlanFromLocalDescriptor(op, localDescriptor, localDescriptor);
}

FailureOr<Value> materializeScheduledGlobalOffset(Operation *op, OpBuilder &builder, const TransferPlan &plan) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func || plan.kind != TransferPlanKind::Scheduled) {
    return failure();
  }
  const Location location = op->getLoc();
  // GM 元素偏移 = Σ（tile 外轴索引 × 该轴 GM stride），不假设轴间连续嵌套。
  const SymbolBindingMap symbols = collectTilingParamBindings(func);
  Value globalOffset;
  for (const TransferOuterAxis &outer : plan.outerAxes) {
    FailureOr<int64_t> literal = getLiteralStride(outer.globalStride);
    // 零 stride 对偏移无贡献，省略该项。
    if (succeeded(literal) && *literal == 0) {
      continue;
    }
    Value index = findOuterAxisIndex(func, op, outer.axis);
    if (!index) {
      return failure();
    }
    Value term = index;
    if (!succeeded(literal) || *literal != 1) {
      FailureOr<Value> stride =
          succeeded(literal) ? FailureOr<Value>(builder.create<arith::ConstantIndexOp>(location, *literal).getResult())
                             : materializeExpression(builder, location, outer.globalStride, symbols);
      if (failed(stride)) {
        return failure();
      }
      term = builder.create<arith::MulIOp>(location, index, *stride);
    }
    globalOffset = globalOffset ? builder.create<arith::AddIOp>(location, globalOffset, term).getResult() : term;
  }
  if (!globalOffset) {
    globalOffset = builder.create<arith::ConstantIndexOp>(location, 0);
  }

  return globalOffset;
}

static LogicalResult emitOuterTransferLoops(OpBuilder &builder, Location location, ArrayRef<TransferDim> outer,
                                            const SymbolBindingMap &actualSizes, const SymbolBindingMap &symbols,
                                            Value globalOffset, Value blockCount, Value blockLen, Value globalStride,
                                            Value localStride, ScheduledTransferBody emitBody,
                                            ArrayRef<bool> actualLocalStrides) {
  // 循环外物化上界和 stride：上界取实际范围，GM stride 用实际尺寸，UB stride 用标称布局。
  SmallVector<OuterLoopTerm, 2> outerTerms;
  outerTerms.reserve(outer.size());
  for (auto [index, dim] : llvm::enumerate(outer)) {
    FailureOr<Value> bound = materializeExpression(builder, location, dim.extent, actualSizes);
    if (failed(bound)) {
      return failure();
    }
    OuterLoopTerm term{*bound, Value(), Value()};
    // 零 stride 的一侧不生成偏移项。
    if (failed(materializeStrideOrSkip(builder, location, dim.globalStride, actualSizes, term.globalStride)) ||
        failed(materializeStrideOrSkip(builder, location, dim.localStride,
                                       actualLocalStrides[index] ? actualSizes : symbols, term.localStride))) {
      return failure();
    }
    outerTerms.push_back(term);
  }

  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Value, 2> inductionVars;
  if (!outerTerms.empty()) {
    Value lowerBound = builder.create<arith::ConstantIndexOp>(location, 0);
    Value step = builder.create<arith::ConstantIndexOp>(location, 1);
    // 按 dims 从外到内生成循环；合并维无单一轴身份，不附加 axis_binding。
    for (const OuterLoopTerm &term : outerTerms) {
      auto loop = builder.create<scf::ForOp>(location, lowerBound, term.bound, step);
      inductionVars.push_back(loop.getInductionVar());
      builder.setInsertionPointToStart(loop.getBody());
    }
  }

  // GM 行内偏移叠加 tile 起点；UB 从基址 0 起算，仅累加 tile 内部行偏移。
  Value localOffset;
  for (auto [inductionVar, term] : llvm::zip_equal(inductionVars, outerTerms)) {
    if (term.globalStride) {
      Value scaled = builder.create<arith::MulIOp>(location, inductionVar, term.globalStride);
      globalOffset = builder.create<arith::AddIOp>(location, globalOffset, scaled);
    }
    if (term.localStride) {
      Value scaled = builder.create<arith::MulIOp>(location, inductionVar, term.localStride);
      localOffset = localOffset ? builder.create<arith::AddIOp>(location, localOffset, scaled).getResult() : scaled;
    }
  }
  if (!localOffset) {
    localOffset = builder.create<arith::ConstantIndexOp>(location, 0);
  }
  return emitBody(builder,
                  ScheduledTileIndices{globalOffset, localOffset, blockCount, blockLen, globalStride, localStride});
}

LogicalResult emitScheduledTransfer(Operation *op, OpBuilder &builder, const TransferPlan &plan,
                                    ScheduledTransferBody emitBody, bool compact) {
  if (plan.kind != TransferPlanKind::Scheduled || plan.dims.empty()) {
    return failure();
  }
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func) {
    return failure();
  }
  auto graph = func->getAttrOfType<AscGraphAttrGroupsAttr>(afir::kAscGraphAttrName);
  if (!graph) {
    return failure();
  }
  const Location location = op->getLoc();

  const SymbolBindingMap symbols = collectTilingParamBindings(func);
  FailureOr<Value> offset = materializeScheduledGlobalOffset(op, builder, plan);
  if (failed(offset)) {
    return failure();
  }
  Value globalOffset = *offset;

  // extent 可能是多个轴尺寸的乘积，逐符号替换为 ActualSize 以处理尾块。
  FailureOr<SymbolBindingMap> actualSizes = collectExtentBindings(func, op, graph, plan.dims);
  if (failed(actualSizes)) {
    return failure();
  }

  // 最内两项进 DMA，其余每项一层 `scf.for`。一维是 `inner` 只有一项的退化情形，走同一
  // 条路径。
  ArrayRef<TransferDim> dims = plan.dims;
  ArrayRef<TransferDim> inner = dims.take_back(std::min<size_t>(dims.size(), 2));
  ArrayRef<TransferDim> outer = dims.drop_back(inner.size());

  FailureOr<Value> blockLen = materializeExpression(builder, location, inner.back().extent, *actualSizes);
  if (failed(blockLen)) {
    return failure();
  }

  // 一维：blockCount 恒 1、两侧缝隙恒 0，与二维共用同一条生成路径。
  Value blockCount = builder.create<arith::ConstantIndexOp>(location, 1);
  Value globalStride = builder.create<arith::ConstantIndexOp>(location, 0);
  Value localStride = globalStride;
  if (inner.size() == 2) {
    const TransferDim &row = inner.front();
    FailureOr<Value> count = materializeExpression(builder, location, row.extent, *actualSizes);
    // 行间距 = 外层 stride − 实际 blockLen。GM stride 使用实际尺寸绑定；
    // UB stride 使用标称布局，尾块缩小范围但不改变对齐后的行距。
    FailureOr<Value> outerGlobalStride = materializeExpression(builder, location, row.globalStride, *actualSizes);
    FailureOr<Value> outerLocalStride =
        materializeExpression(builder, location, row.localStride, compact ? *actualSizes : symbols);
    if (failed(count) || failed(outerGlobalStride) || failed(outerLocalStride)) {
      return failure();
    }
    blockCount = *count;
    globalStride = builder.create<arith::SubIOp>(location, *outerGlobalStride, *blockLen);
    localStride = builder.create<arith::SubIOp>(location, *outerLocalStride, *blockLen);
  }

  // Only the contiguous suffix is repacked on a tail tile; outer gaps retain
  // their nominal physical strides, just as in the transpose consumer.
  SmallVector<bool> actualLocalStrides(dims.size(), false);
  actualLocalStrides.back() = compact;
  for (size_t i = dims.size() - 1; compact && i > 0; --i) {
    auto expected = multiplyExpressions(op->getContext(), dims[i].extent, dims[i].localStride);
    compact = succeeded(expected) && proveEqual(dims[i - 1].localStride, *expected) == detail::TruthValue::True;
    actualLocalStrides[i - 1] = compact;
  }
  return emitOuterTransferLoops(builder, location, outer, *actualSizes, symbols, globalOffset, blockCount, *blockLen,
                                globalStride, localStride, emitBody,
                                ArrayRef<bool>(actualLocalStrides).take_front(outer.size()));
}

FailureOr<Value> materializeNominalInnermostExtent(Operation *op, OpBuilder &builder, const TransferPlan &plan) {
  if (plan.kind != TransferPlanKind::Scheduled || plan.dims.empty()) {
    return failure();
  }
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func) {
    return failure();
  }
  return materializeExpression(builder, op->getLoc(), plan.dims.back().extent, collectTilingParamBindings(func));
}

FailureOr<SmallVector<ScheduledAxisExtent, 4>> materializeScheduledAxisExtents(Operation *op, OpBuilder &builder,
                                                                               ArrayRef<int64_t> axes) {
  if (axes.empty()) {
    return failure();
  }
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func) {
    return failure();
  }
  auto graph = func->getAttrOfType<AscGraphAttrGroupsAttr>(afir::kAscGraphAttrName);
  if (!graph) {
    return failure();
  }

  llvm::DenseMap<int64_t, AxisAttr> axesById;
  for (AxisAttr axis : graph.getAxes()) {
    axesById.try_emplace(axis.getId(), axis);
  }
  SmallVector<TransferDim, 4> requested;
  requested.reserve(axes.size());
  for (int64_t axisId : axes) {
    auto found = axesById.find(axisId);
    if (found == axesById.end() || !found->second.getSize()) {
      return failure();
    }
    requested.push_back(TransferDim{found->second.getSize(), {}, {}});
  }

  const SymbolBindingMap nominalBindings = collectTilingParamBindings(func);
  FailureOr<SymbolBindingMap> actualBindings = collectExtentBindings(func, op, graph, requested);
  if (failed(actualBindings)) {
    return failure();
  }
  SmallVector<ScheduledAxisExtent, 4> values;
  values.reserve(requested.size());
  for (const TransferDim &dim : requested) {
    FailureOr<Value> nominal = materializeExpression(builder, op->getLoc(), dim.extent, nominalBindings);
    FailureOr<Value> actual = materializeExpression(builder, op->getLoc(), dim.extent, *actualBindings);
    if (failed(nominal) || failed(actual)) {
      return failure();
    }
    values.push_back(ScheduledAxisExtent{*nominal, *actual});
  }
  return values;
}

FailureOr<SmallVector<Value, 2>> materializeScheduledTransferDims(Operation *op, OpBuilder &builder,
                                                                  const TransferPlan &plan) {
  if (plan.kind != TransferPlanKind::Scheduled || plan.dims.empty() || plan.dims.size() > 2) {
    return failure();
  }
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func) {
    return failure();
  }
  auto graph = func->getAttrOfType<AscGraphAttrGroupsAttr>(afir::kAscGraphAttrName);
  if (!graph) {
    return failure();
  }
  FailureOr<SymbolBindingMap> actualSizes = collectExtentBindings(func, op, graph, plan.dims);
  if (failed(actualSizes)) {
    return failure();
  }
  SmallVector<Value, 2> values;
  values.reserve(plan.dims.size());
  for (const TransferDim &dim : plan.dims) {
    FailureOr<Value> value = materializeExpression(builder, op->getLoc(), dim.extent, *actualSizes);
    if (failed(value)) {
      return failure();
    }
    values.push_back(*value);
  }
  return values;
}

FailureOr<Value> materializeScheduledLocalSpan(Operation *op, OpBuilder &builder, const TransferPlan &plan) {
  if (plan.kind != TransferPlanKind::Scheduled || plan.dims.empty()) {
    return failure();
  }
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func) {
    return failure();
  }
  auto graph = func->getAttrOfType<AscGraphAttrGroupsAttr>(afir::kAscGraphAttrName);
  if (!graph) {
    return failure();
  }

  FailureOr<SymbolBindingMap> actualSizes = collectExtentBindings(func, op, graph, plan.dims);
  if (failed(actualSizes)) {
    return failure();
  }
  const SymbolBindingMap nominalSizes = collectTilingParamBindings(func);
  const Location location = op->getLoc();
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value span = one;
  for (const TransferDim &dim : plan.dims) {
    FailureOr<Value> actual = materializeExpression(builder, location, dim.extent, *actualSizes);
    FailureOr<Value> stride = materializeExpression(builder, location, dim.localStride, nominalSizes);
    if (failed(actual) || failed(stride)) {
      return failure();
    }
    Value reduced = builder.create<arith::SubIOp>(location, *actual, one);
    Value distance = builder.create<arith::MulIOp>(location, reduced, *stride);
    span = builder.create<arith::AddIOp>(location, span, distance);
  }
  return span;
}

FailureOr<Value> materializeSegmentExtent(Operation *op, OpBuilder &builder, const VectorSegment &segment,
                                          int64_t scheduledAxis, Value tailAwareSize) {
  if (op == nullptr || segment.axes.size() != segment.axisExtents.size() || segment.axes.empty()) {
    return failure();
  }
  Value extent;
  for (auto [axisId, axisExtent] : llvm::zip(segment.axes, segment.axisExtents)) {
    Value factor;
    if (tailAwareSize && axisId == scheduledAxis) {
      factor = tailAwareSize;
    } else {
      FailureOr<Value> materialized = materializeExtentValue(op, builder, axisExtent);
      if (failed(materialized)) {
        return failure();
      }
      factor = *materialized;
    }
    extent = extent ? builder.create<arith::MulIOp>(op->getLoc(), extent, factor).getResult() : factor;
  }
  return extent;
}

FailureOr<Value> materializeExecConditionGuard(Operation *op, OpBuilder &builder, const TransferPlan &plan,
                                               ExecCondition condition) {
  if (condition == ExecCondition::NoCache || plan.kind != TransferPlanKind::Scheduled) {
    return failure();
  }
  auto func = op->getParentOfType<func::FuncOp>();
  scf::ForOp loop = findLoop(op, plan.loopAxis);
  if (!func || !loop) {
    return failure();
  }
  // 两支共用的那一项：本核第一趟。
  Location loc = op->getLoc();
  Value index = loop.getInductionVar();
  // `origin`：TILE_IN 及以外的有效轴全是广播轴，缓存值在整个分块循环里不变。
  if (condition == ExecCondition::CacheBlockSplitOriginBroadcastAxis) {
    return builder
        .create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, index, builder.create<arith::ConstantIndexOp>(loc, 0))
        .getResult();
  }
  // 维度关系没解析出来就报失败。**这与「解析得出且是另一维」不是一件事**：那一支下面要发
  // 「首趟」那个谓词，而这里连缓存值沿哪一维变化都不知道，发任何谓词都是猜。
  if (plan.guardLoopRelation == GuardLoopRelation::Unresolved) {
    return failure();
  }
  // `fused` 且循环跑的那根轴在**另一维**上：缓存值沿切分轴那一维变化，而那一维的 tile 序号
  // 在本循环内不变，于是本循环内也不变，只在首趟算。相位在这里无从谈起——`index` 走的是另一
  // 维；外层循环每推进一步，本循环重新从 0 开始，首趟那一项正好重算一次。
  //
  // legacy 在这种形态下发的是跨维取模（`codegen_kernel_loop.cpp` 取轴表里第一个 BlockInner
  // 与第一个 TileOuter，循环变量却是当前这根轴的），它只会比需要的更频繁地重算、不会漏算；
  // 这里按维度关系分支，重算次数不多于它。
  if (plan.guardLoopRelation == GuardLoopRelation::OtherDimension) {
    return builder
        .create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, index, builder.create<arith::ConstantIndexOp>(loc, 0))
        .getResult();
  }
  // 先校验所有依赖，避免失败时留下部分算术操作。
  Value blockIndex = findAxisValue(func, op, plan.blockOuterAxis, afir::AxisValueRole::AxisIndex);
  Value blockSize = findAxisValue(func, op, plan.blockInnerAxis, afir::AxisValueRole::AxisSize);
  Value tileLoopSize = findAxisValue(func, op, plan.tileOuterAxis, afir::AxisValueRole::LoopSize);
  if (!blockIndex || !blockSize || !tileLoopSize) {
    return failure();
  }
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value isFirstIteration = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, index, zero);
  Value linear = builder.create<arith::AddIOp>(loc, builder.create<arith::MulIOp>(loc, blockIndex, blockSize), index);
  Value phase = builder.create<arith::RemSIOp>(loc, linear, tileLoopSize);
  Value isCycleStart = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, phase, zero);
  return builder.create<arith::OrIOp>(loc, isFirstIteration, isCycleStart).getResult();
}

FailureOr<Value> materializeActualSegmentExtent(Operation *op, OpBuilder &builder, const VectorSegment &segment) {
  auto func = op->getParentOfType<func::FuncOp>();
  auto graph = func ? func->getAttrOfType<AscGraphAttrGroupsAttr>(kAscGraphAttrName) : AscGraphAttrGroupsAttr();
  if (!graph || segment.axisExtents.empty()) {
    return failure();
  }
  SmallVector<TransferDim> dims;
  for (ExprAttr extent : segment.axisExtents) {
    dims.push_back({extent, {}, {}});
  }
  auto bindings = collectExtentBindings(func, op, graph, dims);
  if (failed(bindings)) {
    return failure();
  }
  Value result;
  for (const TransferDim &dim : dims) {
    auto value = materializeExpression(builder, op->getLoc(), dim.extent, *bindings);
    if (failed(value)) {
      return failure();
    }
    result = result ? builder.create<arith::MulIOp>(op->getLoc(), result, *value).getResult() : *value;
  }
  return result;
}

FailureOr<Value> materializeTailAwareTileSize(Operation *op, OpBuilder &builder, const TransferPlan &plan) {
  if (plan.kind != TransferPlanKind::Scheduled || plan.tileInnerAxis < 0) {
    return failure();
  }
  FailureOr<SmallVector<ScheduledAxisExtent, 4>> values =
      materializeScheduledAxisExtents(op, builder, {plan.tileInnerAxis});
  if (failed(values)) {
    return failure();
  }
  return values->front().actual;
}

namespace {
FailureOr<ExprAttr> resolveGlobalStride(MLIRContext *context, const llvm::DenseMap<int64_t, AxisAttr> &axes,
                                        const llvm::DenseMap<int64_t, ExprAttr> &strides, int64_t id,
                                        llvm::SmallDenseSet<int64_t> &visiting) {
  auto stride = strides.find(id);
  if (stride != strides.end()) {
    return stride->second;
  }
  auto found = axes.find(id);
  if (found == axes.end() || !visiting.insert(id).second) {
    return failure();
  }
  AxisAttr axis = found->second;
  auto resolve = [&](int64_t source) { return resolveGlobalStride(context, axes, strides, source, visiting); };
  FailureOr<ExprAttr> result = failure();
  if (axis.getAxisType() == AxisType::Merged && !axis.getFrom().empty()) {
    SmallVector<ExprAttr> units;
    for (int64_t source : axis.getFrom()) {
      FailureOr<ExprAttr> unit = resolve(source);
      if (failed(unit)) {
        return failure();
      }
      units.push_back(*unit);
    }
    for (size_t i = 1; i < units.size(); ++i) {
      auto inner = axes.find(axis.getFrom()[i]);
      if (inner == axes.end() || !provablyContiguous(context, units[i - 1], units[i], inner->second.getSize())) {
        return failure();
      }
    }
    result = units.back();
  } else if (axis.getFrom().size() == 1) {
    result = resolve(axis.getFrom().front());
    if (succeeded(result)) {
      if (axis.getAxisType() == AxisType::TileOuter || axis.getAxisType() == AxisType::BlockOuter) {
        auto inner = axes.find(axis.getSplitPairOtherId());
        if (inner == axes.end()) {
          return failure();
        }
        result = multiplyExprs(context, *result, inner->second.getSize());
      } else if (axis.getAxisType() != AxisType::TileInner && axis.getAxisType() != AxisType::BlockInner) {
        return failure();
      }
    }
  }
  visiting.erase(id);
  return result;
}
}  // namespace

FailureOr<ExprAttr> globalAxisUnitStride(Operation *op, int64_t axisId) {
  auto func = op->getParentOfType<func::FuncOp>();
  auto graph = func ? func->getAttrOfType<AscGraphAttrGroupsAttr>(afir::kAscGraphAttrName) : AscGraphAttrGroupsAttr();
  StringRef name = op->getNumResults() ? afir::kOperandDescsAttrName : afir::kDestinationDescsAttrName;
  auto descriptors = op->getAttrOfType<ArrayAttr>(name);
  if (!graph || !descriptors || descriptors.empty()) {
    return failure();
  }
  auto global = dyn_cast<TensorDescAttr>(descriptors[0]);
  if (!global || global.getAxis().size() != global.getStrides().size()) {
    return failure();
  }
  llvm::DenseMap<int64_t, AxisAttr> axes;
  for (AxisAttr axis : graph.getAxes()) {
    axes.try_emplace(axis.getId(), axis);
  }
  llvm::DenseMap<int64_t, ExprAttr> strides;
  for (auto [id, stride] : llvm::zip_equal(global.getAxis(), global.getStrides())) {
    strides.try_emplace(id, stride);
  }
  llvm::SmallDenseSet<int64_t> visiting;
  return resolveGlobalStride(op->getContext(), axes, strides, axisId, visiting);
}

LogicalResult localTileIsSingleSegment(Operation *op) {
  FailureOr<VectorPlan> plan = collapseVectorizedAxes(op);
  auto local = getLocalDescriptor(op, TransferPlanSide::Automatic);
  if (failed(plan) || failed(local) || plan->segments.size() > 1) {
    return failure();
  }
  for (auto [index, tensor] : llvm::enumerate(plan->tensors)) {
    if (tensor.descriptor == *local) {
      // A single segment can still have a non-unit stride. Flattening its
      // physical span into one Duplicate would also write the intervening slots.
      return success(plan->segments.empty() || provenEqualTo(plan->segments.front().strides[index], "1"));
    }
  }
  return failure();
}

}  // namespace mlir::afir
