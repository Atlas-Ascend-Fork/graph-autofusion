/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- BufferAlias.cpp - Base-storage resolution and alias queries --------===//

#include "dialect/AFIR/Analysis/BufferAlias.h"

#include <algorithm>
#include <numeric>

#include "dialect/AFIR/IR/AFIR.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace mlir::afir {
namespace {

constexpr int64_t kUnknownSize = -1;

// 仅识别步长为 1 的 scf.for 归纳变量形成的 (iv + phase) mod modulus，
// 允许省略加法，取模支持 arith.remsi / remui。
// 非单位步长需要求解 d*step ≡ Δ (mod S)，不适用此处相位算法。
// 无法识别时返回 Unknown，使别名判断保守处理，避免误判不重叠。
std::optional<std::pair<Value, int64_t>> matchInductionVariablePlusConstant(Value value) {
  int64_t phase = 0;
  Value current = value;
  if (auto add = current.getDefiningOp<arith::AddIOp>()) {
    IntegerAttr constant;
    if (matchPattern(add.getRhs(), m_Constant(&constant))) {
      phase = constant.getInt();
      current = add.getLhs();
    } else if (matchPattern(add.getLhs(), m_Constant(&constant))) {
      phase = constant.getInt();
      current = add.getRhs();
    } else {
      return std::nullopt;
    }
  }
  auto blockArgument = dyn_cast<BlockArgument>(current);
  if (!blockArgument) {
    return std::nullopt;
  }
  auto forOp = dyn_cast_or_null<scf::ForOp>(blockArgument.getOwner()->getParentOp());
  if (!forOp || forOp.getInductionVar() != blockArgument) {
    return std::nullopt;
  }
  IntegerAttr step;
  if (!matchPattern(forOp.getStep(), m_Constant(&step)) || step.getInt() != 1) {
    return std::nullopt;
  }
  return std::make_pair(current, phase);
}

DimensionOffset classifyOffset(OpFoldResult offset) {
  if (auto attribute = dyn_cast<Attribute>(offset)) {
    auto integer = dyn_cast<IntegerAttr>(attribute);
    return integer ? DimensionOffset::getStatic(integer.getInt()) : DimensionOffset::getUnknown();
  }
  auto value = cast<Value>(offset);
  IntegerAttr constant;
  if (matchPattern(value, m_Constant(&constant))) {
    return DimensionOffset::getStatic(constant.getInt());
  }
  Value remainder;
  int64_t modulus = 0;
  if (auto remsi = value.getDefiningOp<arith::RemSIOp>()) {
    remainder = remsi.getLhs();
    IntegerAttr divisor;
    if (!matchPattern(remsi.getRhs(), m_Constant(&divisor))) {
      return DimensionOffset::getUnknown();
    }
    modulus = divisor.getInt();
  } else if (auto remui = value.getDefiningOp<arith::RemUIOp>()) {
    remainder = remui.getLhs();
    IntegerAttr divisor;
    if (!matchPattern(remui.getRhs(), m_Constant(&divisor))) {
      return DimensionOffset::getUnknown();
    }
    modulus = divisor.getInt();
  } else {
    return DimensionOffset::getUnknown();
  }
  if (modulus <= 0) {
    return DimensionOffset::getUnknown();
  }
  std::optional<std::pair<Value, int64_t>> induction = matchInductionVariablePlusConstant(remainder);
  if (!induction) {
    return DimensionOffset::getUnknown();
  }
  return DimensionOffset::getRotating(induction->first, modulus, induction->second);
}

int64_t classifySize(OpFoldResult size) {
  if (auto attribute = dyn_cast<Attribute>(size)) {
    auto integer = dyn_cast<IntegerAttr>(attribute);
    return integer ? integer.getInt() : kUnknownSize;
  }
  IntegerAttr constant;
  if (matchPattern(cast<Value>(size), m_Constant(&constant))) {
    return constant.getInt();
  }
  return kUnknownSize;
}

bool hasUnitStrides(memref::SubViewOp subView) {
  return llvm::all_of(subView.getMixedStrides(), [](OpFoldResult stride) {
    if (auto attribute = dyn_cast<Attribute>(stride)) {
      auto integer = dyn_cast<IntegerAttr>(attribute);
      return integer && integer.getInt() == 1;
    }
    IntegerAttr constant;
    return matchPattern(cast<Value>(stride), m_Constant(&constant)) && constant.getInt() == 1;
  });
}

void initializeWholeBase(BufferSlice &slice) {
  slice.wholeBase = true;
  slice.offsets.clear();
  slice.sizes.clear();
}

// 两段一维区间是否可证明不重叠。任何一端未知就放弃证明。
bool isDisjointStaticRange(int64_t lhsOffset, int64_t lhsSize, int64_t rhsOffset, int64_t rhsSize) {
  if (lhsSize == kUnknownSize || rhsSize == kUnknownSize) {
    return false;
  }
  return lhsOffset + lhsSize <= rhsOffset || rhsOffset + rhsSize <= lhsOffset;
}

// 判断 lhs 在第 i 轮、rhs 在第 i+d 轮是否可能使用同一槽位：
// (i + lhsPhase) mod S == (i + d + rhsPhase) mod S
// 等价于 d ≡ lhsPhase - rhsPhase (mod S)。
// 仅当两侧级数相同，且由距离 d 所属循环的同一归纳变量驱动时，才可使用此相位关系。
// 否则两侧 i 不在同一坐标系中，保守地按可能重叠处理。
bool mayRotatingOffsetsCollide(const DimensionOffset &lhs, const DimensionOffset &rhs, int64_t iterationDistance,
                               Value loopInductionVariable) {
  if (lhs.modulus != rhs.modulus) {
    return true;
  }
  if (!lhs.inductionVariable || lhs.inductionVariable != rhs.inductionVariable) {
    return true;
  }
  if (loopInductionVariable && lhs.inductionVariable != loopInductionVariable) {
    return true;
  }
  int64_t modulus = lhs.modulus;
  int64_t delta = ((lhs.value - rhs.value - iterationDistance) % modulus + modulus) % modulus;
  return delta == 0;
}

// 单个维度上能否证明两个访问不重叠。`iterationDistance` 只对轮转下标有意义：常量下标
// 在每次迭代上都一样，跨迭代与同迭代的结论相同。
// 一个维度上的一段：起点（静态或轮转）与长度。
struct DimensionSpan {
  const DimensionOffset &offset;
  int64_t size;
};

bool isDisjointDimension(DimensionSpan lhs, DimensionSpan rhs, int64_t iterationDistance, Value loopInductionVariable) {
  using Kind = DimensionOffset::Kind;
  if (lhs.offset.kind == Kind::Static && rhs.offset.kind == Kind::Static) {
    return isDisjointStaticRange(lhs.offset.value, lhs.size, rhs.offset.value, rhs.size);
  }
  if (lhs.offset.kind == Kind::Rotating && rhs.offset.kind == Kind::Rotating) {
    if (mayRotatingOffsetsCollide(lhs.offset, rhs.offset, iterationDistance, loopInductionVariable)) {
      return false;
    }
    // 撞不上同一个槽，但只有在每个槽本身不跨界时这个结论才成立：长度必须是 1 个槽。
    return lhs.size == 1 && rhs.size == 1;
  }
  return false;
}

}  // namespace

DimensionOffset DimensionOffset::getStatic(int64_t value) {
  DimensionOffset offset;
  offset.kind = Kind::Static;
  offset.value = value;
  return offset;
}

DimensionOffset DimensionOffset::getRotating(Value inductionVariable, int64_t modulus, int64_t phase) {
  DimensionOffset offset;
  offset.kind = Kind::Rotating;
  offset.modulus = modulus;
  offset.value = ((phase % modulus) + modulus) % modulus;
  offset.inductionVariable = inductionVariable;
  return offset;
}

DimensionOffset DimensionOffset::getUnknown() {
  return DimensionOffset{};
}

BufferSlice resolveBufferSlice(Value value) {
  BufferSlice slice;
  if (!value) {
    return slice;
  }

  llvm::DenseSet<Value> visited;
  Value current = value;
  bool haveSegment = false;   // 已经记下了一层段信息
  bool segmentUsable = true;  // 段信息仍然可信；置假之后只负责找根
  while (current) {
    if (!visited.insert(current).second) {
      segmentUsable = false;  // 环只可能来自畸形 IR；按整块处理而不是转圈。
      break;
    }
    if (Value tied = afir::getTiedDestination(current)) {
      current = tied;
      continue;
    }
    Operation *definingOp = current.getDefiningOp();
    if (!definingOp) {
      break;  // 块参数：根就是它自己。
    }
    if (auto castOp = dyn_cast<memref::CastOp>(definingOp)) {
      current = castOp.getSource();
      continue;
    }
    auto subView = dyn_cast<memref::SubViewOp>(definingOp);
    if (!subView) {
      break;
    }
    // 只接受单位 stride 的一层 subview，不叠加第二层——嵌套 subview 的下标换算要按
    // 外层的 stride 做，算错的后果是漏报，不值得为当前不存在的形态承担。
    if (haveSegment || !hasUnitStrides(subView)) {
      segmentUsable = false;
      current = subView.getSource();
      continue;
    }
    for (OpFoldResult offset : subView.getMixedOffsets()) {
      slice.offsets.push_back(classifyOffset(offset));
    }
    for (OpFoldResult size : subView.getMixedSizes()) {
      slice.sizes.push_back(classifySize(size));
    }
    haveSegment = true;
    current = subView.getSource();
  }

  slice.base = current;
  if (!haveSegment || !segmentUsable) {
    initializeWholeBase(slice);
  }
  return slice;
}

AliasResult aliasSlices(const BufferSlice &lhs, const BufferSlice &rhs, int64_t iterationDistance,
                        Value loopInductionVariable) {
  if (!lhs.isValid() || !rhs.isValid()) {
    return AliasResult::MayAlias;  // 追不到根就证明不了任何事。
  }
  if (lhs.base != rhs.base) {
    // 不同的根即不同的存储：memref 的别名只能来自同一个定义点，跨定义点的重叠在
    // AFIR 里没有表达形式（没有指针算术，也没有 `memref.reinterpret_cast` 的来源）。
    return AliasResult::NoAlias;
  }
  if (lhs.wholeBase || rhs.wholeBase) {
    return AliasResult::MayAlias;
  }
  if (lhs.offsets.size() != rhs.offsets.size() || lhs.offsets.size() != lhs.sizes.size() ||
      rhs.offsets.size() != rhs.sizes.size()) {
    return AliasResult::MayAlias;
  }
  for (size_t dimension = 0; dimension < lhs.offsets.size(); ++dimension) {
    if (isDisjointDimension({lhs.offsets[dimension], lhs.sizes[dimension]},
                            {rhs.offsets[dimension], rhs.sizes[dimension]}, iterationDistance, loopInductionVariable)) {
      return AliasResult::NoAlias;  // 任何一维不重叠，整体就不重叠。
    }
  }
  return AliasResult::MayAlias;
}

std::optional<int64_t> firstAliasingIterationDistance(const BufferSlice &lhs, const BufferSlice &rhs,
                                                      Value loopInductionVariable) {
  // 同一次迭代内不重叠不代表跨迭代不重叠——轮转下标正是这种情况，所以从距离 1 起逐个
  // 试到一个周期为止。
  //
  // 周期是各维级数的最小公倍数，不是最大值：两维分别 3 路和 4 路轮转时，两个访问
  // 要同时在两维上撞一次需要 d 同时满足两个同余式，最早可能出现在 d = 12。按最大值
  // 截断会在 d ≤ 4 里找不到解就收工，把真实的竞争报成「永不重叠」。
  int64_t period = 1;
  auto accumulate = [&period](const BufferSlice &slice) {
    for (const DimensionOffset &offset : slice.offsets) {
      if (offset.kind == DimensionOffset::Kind::Rotating && offset.modulus > 0) {
        period = std::lcm(period, offset.modulus);
      }
    }
  };
  accumulate(lhs);
  accumulate(rhs);
  // 级数互质时 lcm 会爆炸。与其枚举到天荒地老，不如按最保守的答案收工：相邻迭代就撞。
  // 现实里级数是缓冲级数（2 / 3），这条上限碰不到。
  constexpr int64_t kMaxSearchedPeriod = 1024;
  if (period <= 0 || period > kMaxSearchedPeriod) {
    return 1;
  }
  for (int64_t distance = 1; distance <= period; ++distance) {
    if (aliasSlices(lhs, rhs, distance, loopInductionVariable) == AliasResult::MayAlias) {
      return distance;
    }
  }
  return std::nullopt;
}

}  // namespace mlir::afir
