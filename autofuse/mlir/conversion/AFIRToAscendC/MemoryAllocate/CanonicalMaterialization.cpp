/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/MemoryAllocate/CanonicalMemoryAllocator.h"
#include "dialect/AFIR/Analysis/ReduceScratchLifetime.h"

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"
#include "dialect/AFIR/Transforms/Symbolic/SymbolicTransformDetail.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::afir;
using namespace mlir::ascend::kernel;

namespace mlir::ascend::kernel::detail {

Operation *CanonicalFunctionMemoryAllocator::valueAnchor(Value value) {
  if (Operation *def = value.getDefiningOp()) {
    return def;
  }
  for (Operation *user : value.getUsers()) {
    return user;
  }
  return firstAfirOperation();
}

bool CanonicalFunctionMemoryAllocator::isInLoopScope(Operation *operation, int64_t axisId) {
  for (Operation *parent = operation; parent != nullptr; parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (loop && afir::hasAxisBinding(loop, axisId, afir::AxisValueRole::Loop)) {
      return true;
    }
  }
  return false;
}

Value CanonicalFunctionMemoryAllocator::resolveAlias(Value value) const {
  auto it = values.find(value);
  if (it == values.end() || !it->second.alias) {
    return value;
  }
  return resolveAlias(it->second.alias);
}

bool CanonicalFunctionMemoryAllocator::isContiguousView(MemRefType type) {
  if (!type.getLayout().isIdentity()) {
    return false;
  }
  SmallVector<int64_t> strides;
  int64_t offset = ShapedType::kDynamic;
  if (failed(type.getStridesAndOffset(strides, offset)) || offset != 0) {
    return false;
  }
  return true;
}

// 以 buffer_id 是否在 module 的 workspace_tensor_ids 中判定 workspace 身份。
bool CanonicalFunctionMemoryAllocator::isWorkspaceValue(const CanonicalValueInfo &info) const {
  const int64_t bufferId = info.buffer.getBufferId();
  if (bufferId < 0) {
    return false;
  }
  auto module = func->getParentOfType<ModuleOp>();
  auto declared =
      module ? module->getAttrOfType<DenseI64ArrayAttr>(afir::kWorkspaceTensorIdsAttrName) : DenseI64ArrayAttr();
  return declared && llvm::is_contained(declared.asArrayRef(), bufferId);
}

// workspace 基址加字节偏移后转换为元素指针。
// 偏移形参由 workspace_offset 角色和 buffer id 匹配，缺失时报错，不默认成 0。
FailureOr<Value> CanonicalFunctionMemoryAllocator::materializeWorkspaceBase(OpBuilder &builder,
                                                                            const CanonicalValueInfo &info,
                                                                            MemRefType type) {
  Location loc = info.value.getLoc();
  const int64_t bufferId = info.buffer.getBufferId();
  std::optional<unsigned> workspaceArg = afir::findKernelArgument(func, afir::KernelArgRole::Workspace);
  if (!workspaceArg) {
    return emitError(loc) << afir::kInvalidBindingDiagnostic
                          << ": workspace tensor requires a workspace kernel argument";
  }
  std::optional<unsigned> offsetArg;
  for (unsigned index = 0; index < func.getNumArguments(); ++index) {
    afir::KernelArgAttr argument = afir::getKernelArgAttr(func, index);
    if (argument && argument.getRole() == afir::KernelArgRole::WorkspaceOffset && argument.getIndex() == bufferId) {
      offsetArg = index;
      break;
    }
  }
  if (!offsetArg) {
    return emitError(loc) << afir::kInvalidBindingDiagnostic << ": workspace " << bufferId
                          << " has no workspace_offset kernel argument, so its offset would never be assigned";
  }
  // 偏移为 ui64；arith.index_cast 只接受 signless 整数，因此使用 emitc.cast。
  Value rawOffset = func.getArgument(*offsetArg);
  Value offset = cast<IntegerType>(rawOffset.getType()).isSignless()
                     ? builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), rawOffset).getResult()
                     : builder.create<emitc::CastOp>(loc, builder.getIndexType(), rawOffset).getResult();
  Value base = func.getArgument(*workspaceArg);
  Value offsetBase = builder.create<emitasc::PtrOffsetOp>(loc, base.getType(), base, IntegerAttr(), offset);
  // 结果使用 AscendC 的整数地址空间属性；C++ 类型生成不接受 afir.space 方言属性。
  MemRefType emittedType = MemRefType::get(type.getShape(), type.getElementType(), type.getLayout(),
                                           getAscendCGlobalMemorySpaceAttr(builder.getContext()));
  return builder.create<emitasc::ReinterpretCastOp>(loc, emittedType, offsetBase).getResult();
}

LogicalResult CanonicalFunctionMemoryAllocator::materializeGlobalValue(CanonicalValueInfo &info, MemRefType type) {
  OpBuilder builder(materializationAnchor ? materializationAnchor : firstAfirOperation());
  auto globalType = ascendc::GlobalTensorType::get(type.getShape(), type.getElementType());
  Value global = builder.create<ascendc::GlobalTensorOp>(info.value.getLoc(), globalType);
  createdOps.push_back(global.getDefiningOp());
  // 普通 GM 绑定函数参数，workspace 绑定基址加偏移。
  // 不能绑定函数中段的 alloc，否则无法支配物化锚点处的 SetGlobalBuffer。
  Value buffer = info.value;
  if (isWorkspaceValue(info)) {
    FailureOr<Value> workspaceBase = materializeWorkspaceBase(builder, info, type);
    if (failed(workspaceBase)) {
      return failure();
    }
    buffer = *workspaceBase;
    createdOps.push_back(buffer.getDefiningOp());
  }
  OperationState state(info.value.getLoc(), ascendc::GlobalTensorSetGlobalBufferOp::getOperationName());
  state.addOperands({global, buffer});
  createdOps.push_back(builder.create(state));
  // 将存储描述随承载类型迁移到新的定义点，供后续 lowering 使用。
  afir::setBufferDescAttr(global.getDefiningOp(), info.buffer);
  info.materialized = global;
  return success();
}

namespace {
// 通过 AFIR 操作数分段识别 tmp buffer；size_from_resource 也用于普通 UB 和队列。
bool isTmpBufferOperand(Value value) {
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (owner->getDialect() != value.getContext()->getLoadedDialect<afir::AFIRDialect>()) {
      continue;
    }
    FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(owner);
    if (mlir::failed(partition) || partition->tmpBufferCount == 0) {
      continue;
    }
    const unsigned tmpBufferBegin = partition->inputCount + partition->destinationCount;
    if (use.getOperandNumber() >= tmpBufferBegin) {
      return true;
    }
  }
  return false;
}
}  // namespace

LogicalResult CanonicalFunctionMemoryAllocator::collectMaterializationGroup(
    CanonicalValueInfo &info, std::map<CanonicalGroupKey, SmallVector<CanonicalValueInfo *>> &groups) {
  if (resolveAlias(info.value) != info.value) {
    return success();
  }
  auto type = cast<MemRefType>(info.value.getType());
  if (!isContiguousView(type)) {
    return emitError(info.value.getLoc())
           << afir::kUnsupportedOperationDiagnostic
           << ": canonical resource materialization currently requires a contiguous identity StorageView";
  }
  if (info.view.resource.space == afir::Space::GM) {
    return success();
  }
  bool queue = isQueueBinding(info.view, info.buffer);
  CanonicalResourceKey resourceKey{info.view.resource.space, queue, info.buffer.getBufferId()};
  // tmp buffer 独占资源且必须从偏移 0 开始。
  const bool tmpBuffer = !queue && isTmpBufferOperand(info.value);
  if (tmpBuffer && info.view.offset != 0) {
    return emitError(info.value.getLoc())
           << afir::kInvalidBindingDiagnostic << ": tmp buffer occupies its whole resource and must start at offset 0";
  }
  CanonicalGroupKey groupKey{resourceKey, info.buffer.getReuseId(), info.buffer.getLifeTimeAxisId(),
                             tmpBuffer ? info.ordinal : -1};
  groups[groupKey].push_back(&info);
  return success();
}

FailureOr<Value> CanonicalFunctionMemoryAllocator::createLocalRaw(OpBuilder &builder, const CanonicalGroupKey &key,
                                                                  CanonicalResourceInfo &resource, Operation *anchor,
                                                                  ascendc::LocalTensorType rawType,
                                                                  const SmallVector<CanonicalValueInfo *> &members) {
  if (!key.resource.queue) {
    if (!resource.raw) {
      resource.raw = resourceBuilder->create<ascendc::TBufGetTensorOp>(anchor->getLoc(), rawType, resource.resource);
      createdOps.push_back(resource.raw.getDefiningOp());
    }
    return resource.raw;
  }
  // 队列组按程序序复用：上一组为纯计算成员时，继续使用同一 slot。
  QueueEpoch &epoch = queueEpochs[key.resource];
  if (!epoch.raw || epoch.transferred) {
    if (epoch.raw && failed(closeQueueEpoch(key.resource, epoch))) {
      return failure();
    }
    epoch = QueueEpoch{};
    epoch.raw = builder.create<ascendc::TQueBindAllocTensorOp>(anchor->getLoc(), rawType, resource.resource);
    epoch.anchor = anchor;
    createdOps.push_back(epoch.raw.getDefiningOp());
  }
  epoch.members.append(members.begin(), members.end());
  epoch.transferred = isTransferGroup(members);
  return epoch.raw;
}

void CanonicalFunctionMemoryAllocator::forEachGroupValue(
    const SmallVector<CanonicalValueInfo *> &members,
    llvm::function_ref<void(const CanonicalValueInfo &)> callback) const {
  llvm::SmallPtrSet<Value, 4> roots;
  for (const CanonicalValueInfo *member : members) {
    roots.insert(member->value);
  }
  for (Value value : valueOrder) {
    auto it = values.find(value);
    if (it == values.end() || !roots.contains(resolveAlias(value))) {
      continue;
    }
    callback(it->second);
  }
}

bool CanonicalFunctionMemoryAllocator::isTransferGroup(const SmallVector<CanonicalValueInfo *> &members) const {
  bool transfer = false;
  forEachGroupValue(members, [&](const CanonicalValueInfo &info) {
    transfer |= afir::getTransferDirection(info.view.sourceOwner) != afir::TransferDirection::None;
  });
  return transfer;
}

LogicalResult CanonicalFunctionMemoryAllocator::closeQueueEpoch(const CanonicalResourceKey &key, QueueEpoch &epoch) {
  if (!epoch.raw || epoch.anchor == nullptr) {
    return success();
  }
  auto resource = resources.find(key);
  if (resource == resources.end()) {
    return failure();
  }
  // 在整条复用链的末次使用后归还，避免相邻链同时占用队列。
  Block *block = epoch.anchor->getBlock();
  Operation *lastUse = lastQueueGroupUse(epoch.members, block);
  OpBuilder freeBuilder(block->getTerminator());
  if (lastUse != nullptr && lastUse != block->getTerminator()) {
    freeBuilder.setInsertionPointAfter(lastUse);
  }
  auto free =
      freeBuilder.create<ascendc::TQueBindFreeTensorOp>(epoch.anchor->getLoc(), resource->second.resource, epoch.raw);
  createdOps.push_back(free.getOperation());
  epoch = QueueEpoch{};
  return success();
}

FailureOr<Value> CanonicalFunctionMemoryAllocator::materializeCheckedAdd(Value lhs, Value rhs, StringRef message,
                                                                         Location loc) {
  auto add = resourceBuilder->create<arith::AddUIExtendedOp>(loc, lhs, rhs);
  createdOps.push_back(add.getOperation());
  Value falseValue = resourceBuilder->create<arith::ConstantIntOp>(loc, 0, 1);
  Value noOverflow =
      resourceBuilder->create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, add.getOverflow(), falseValue);
  OperationState assertState(loc, "cf.assert");
  assertState.addOperands(noOverflow);
  assertState.addAttribute("msg", resourceBuilder->getStringAttr(message));
  createdOps.push_back(falseValue.getDefiningOp());
  createdOps.push_back(noOverflow.getDefiningOp());
  createdOps.push_back(resourceBuilder->create(assertState));
  return add.getSum();
}

FailureOr<Value> CanonicalFunctionMemoryAllocator::materializeCheckedMul(Value lhs, Value rhs, StringRef message,
                                                                         Location loc) {
  auto multiply = resourceBuilder->create<arith::MulUIExtendedOp>(loc, lhs, rhs);
  createdOps.push_back(multiply.getOperation());
  Value zero = resourceBuilder->create<arith::ConstantIndexOp>(loc, 0);
  Value noOverflow = resourceBuilder->create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, multiply.getHigh(), zero);
  OperationState assertState(loc, "cf.assert");
  assertState.addOperands(noOverflow);
  assertState.addAttribute("msg", resourceBuilder->getStringAttr(message));
  createdOps.push_back(zero.getDefiningOp());
  createdOps.push_back(noOverflow.getDefiningOp());
  createdOps.push_back(resourceBuilder->create(assertState));
  return multiply.getLow();
}

FailureOr<int64_t> CanonicalFunctionMemoryAllocator::queueMemberElementBytes(const CanonicalValueInfo &member,
                                                                             Location loc) {
  auto type = cast<MemRefType>(member.value.getType());
  Type elementType = type.getElementType();
  if (!isa<IntegerType, FloatType>(elementType)) {
    emitError(loc) << afir::kUnsupportedOperationDiagnostic
                   << ": queue member size requires an integer or floating-point element type";
    return failure();
  }
  int64_t elementBits = elementType.getIntOrFloatBitWidth();
  if (elementBits <= 0 || elementBits % 8 != 0) {
    emitError(loc) << afir::kInvalidBindingDiagnostic
                   << ": queue member element type must have a positive byte-aligned width";
    return failure();
  }
  return elementBits / 8;
}

FailureOr<Value> CanonicalFunctionMemoryAllocator::materializeVectorizedAxisSpan(const CanonicalValueInfo &member,
                                                                                 size_t index, Value one,
                                                                                 Location loc) {
  auto axis = member.tensor.getAxis();
  auto repeats = member.tensor.getRepeats();
  int64_t vectorAxis = member.tensor.getVectorizedAxis()[index];
  auto strideAttr = member.tensor.getVectorizedStrides()[index];

  int64_t strideLiteral = 0;
  if (!strideAttr.getValue().getValue().getAsInteger(10, strideLiteral)) {
    if (strideLiteral == 0) {
      return Value();
    }
    if (strideLiteral < 0) {
      emitError(loc) << afir::kInvalidDescriptorDiagnostic << ": queue member vectorized stride must be non-negative";
      return failure();
    }
  }

  auto axisIt = std::find(axis.begin(), axis.end(), vectorAxis);
  if (axisIt == axis.end()) {
    emitError(loc) << afir::kInvalidDescriptorDiagnostic << ": vectorized axis " << vectorAxis
                   << " is not present in the queue member axis list";
    return failure();
  }
  size_t axisIndex = static_cast<size_t>(std::distance(axis.begin(), axisIt));
  FailureOr<Value> repeat = materializeExpression(repeats[axisIndex], loc);
  if (failed(repeat)) {
    return failure();
  }
  FailureOr<Value> stride = materializeExpression(strideAttr, loc);
  if (failed(stride)) {
    return failure();
  }
  Value repeatMinusOne = resourceBuilder->create<arith::SubIOp>(loc, *repeat, one).getResult();
  return materializeCheckedMul(repeatMinusOne, *stride, "queue member vectorized size must not overflow", loc);
}

// 按各非零 stride 向量化轴累计元素间隔，再加 1 得到包含末元素的跨度。
// 所有 stride 均为 0 时，跨度为单个元素。
FailureOr<Value> CanonicalFunctionMemoryAllocator::materializeQueueMemberElementCount(const CanonicalValueInfo &member,
                                                                                      Location loc) {
  constexpr StringLiteral kOverflow = "queue member vectorized size must not overflow";
  Value one = resourceBuilder->create<arith::ConstantIndexOp>(loc, 1).getResult();
  Value elementCount = resourceBuilder->create<arith::ConstantIndexOp>(loc, 0).getResult();
  bool hasNonZeroStride = false;
  for (size_t index = 0; index < member.tensor.getVectorizedAxis().size(); ++index) {
    FailureOr<Value> span = materializeVectorizedAxisSpan(member, index, one, loc);
    if (failed(span)) {
      return failure();
    }
    if (!*span) {
      continue;
    }
    if (hasNonZeroStride) {
      FailureOr<Value> newCount = materializeCheckedAdd(elementCount, *span, kOverflow, loc);
      if (failed(newCount)) {
        return failure();
      }
      elementCount = *newCount;
    } else {
      elementCount = *span;
    }
    hasNonZeroStride = true;
  }
  if (!hasNonZeroStride) {
    return one;
  }
  return materializeCheckedAdd(elementCount, one, kOverflow, loc);
}

FailureOr<Value> CanonicalFunctionMemoryAllocator::materializeLocalMemberSize(const CanonicalValueInfo &member,
                                                                              Location loc, bool queue) {
  if (!queue) {
    // 普通 UB 的成员大小等于资源大小，使用同一尺寸表达式或 canonical 尺寸字段。
    return materializeResourceSize(member.buffer, loc);
  }

  FailureOr<int64_t> elementBytesLiteral = queueMemberElementBytes(member, loc);
  if (failed(elementBytesLiteral)) {
    return failure();
  }

  auto vectorizedAxis = member.tensor.getVectorizedAxis();
  if (vectorizedAxis.empty()) {
    Value elementBytes = resourceBuilder->create<arith::ConstantIndexOp>(loc, *elementBytesLiteral).getResult();
    return alignResourceSize(elementBytes, 32, loc);
  }
  if (member.tensor.getAxis().size() != member.tensor.getRepeats().size() ||
      vectorizedAxis.size() != member.tensor.getVectorizedStrides().size()) {
    emitError(loc) << afir::kDescriptorCountMismatchDiagnostic
                   << ": queue member vectorized size requires axis/repeats and vectorized_axis/vectorized_strides "
                      "to be aligned";
    return failure();
  }

  FailureOr<Value> elementCount = materializeQueueMemberElementCount(member, loc);
  if (failed(elementCount)) {
    return failure();
  }
  Value elementBytes = resourceBuilder->create<arith::ConstantIndexOp>(loc, *elementBytesLiteral).getResult();
  FailureOr<Value> byteSize =
      materializeCheckedMul(*elementCount, elementBytes, "queue member byte size must not overflow", loc);
  if (failed(byteSize)) {
    return failure();
  }
  return alignResourceSize(*byteSize, 32, loc);
}

LogicalResult CanonicalFunctionMemoryAllocator::materializeLocalMember(OpBuilder &builder, LocalSlab &slab,
                                                                       CanonicalValueInfo &member) {
  Location loc = member.value.getLoc();
  auto type = cast<MemRefType>(member.value.getType());
  auto localType = ascendc::LocalTensorType::get(type.getShape(), type.getElementType());
  // 非队列成员复用存储，各自使用 view.offset，不累加前一成员大小。
  // 仅队列成员按大小推进偏移。
  Value offset = slab.offset;
  if (!slab.queue) {
    offset = builder.create<arith::ConstantIndexOp>(loc, member.view.offset);
    createdOps.push_back(offset.getDefiningOp());
  }
  Value byteView = builder.create<ascendc::LocalTensorSubIndexOp>(loc, slab.rawType, slab.raw, offset);
  createdOps.push_back(byteView.getDefiningOp());
  member.materialized = builder.create<ascendc::LocalTensorReinterpretCastOp>(loc, localType, byteView);
  createdOps.push_back(member.materialized.getDefiningOp());
  afir::setBufferDescAttr(member.materialized.getDefiningOp(), member.buffer);
  if (auto *owner = member.value.getDefiningOp()) {
    if (auto scope = owner->getAttr(afir::kReduceScratchScopeAttr)) {
      member.materialized.getDefiningOp()->setAttr(afir::kReduceScratchScopeAttr, scope);
    }
  }
  // 不推进偏移时，无需计算或检查下一成员的起点。
  if (!slab.queue) {
    return success();
  }
  FailureOr<Value> size = materializeLocalMemberSize(member, loc, slab.queue);
  if (failed(size)) {
    return failure();
  }
  auto add = builder.create<arith::AddUIExtendedOp>(loc, slab.offset, *size);
  createdOps.push_back(add.getOperation());
  Value highZero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value lowNonNegative = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, add.getSum(), highZero);
  Value falseValue = builder.create<arith::ConstantIntOp>(loc, 0, 1);
  Value noOverflow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, add.getOverflow(), falseValue);
  noOverflow = builder.create<arith::AndIOp>(loc, lowNonNegative, noOverflow);
  OperationState assertState(loc, "cf.assert");
  assertState.addOperands(noOverflow);
  assertState.addAttribute("msg", builder.getStringAttr("resource member offset must not overflow"));
  createdOps.push_back(highZero.getDefiningOp());
  createdOps.push_back(falseValue.getDefiningOp());
  createdOps.push_back(lowNonNegative.getDefiningOp());
  createdOps.push_back(noOverflow.getDefiningOp());
  createdOps.push_back(builder.create(assertState));
  slab.offset = add.getSum();
  return success();
}

Operation *CanonicalFunctionMemoryAllocator::lastQueueGroupUse(const SmallVector<CanonicalValueInfo *> &members,
                                                               Block *block) const {
  Operation *last = nullptr;
  forEachGroupValue(members, [&](const CanonicalValueInfo &info) {
    for (Operation *user : info.value.getUsers()) {
      Operation *ancestor = block->findAncestorOpInBlock(*user);
      if (ancestor == nullptr) {
        continue;
      }
      if (last == nullptr || last->isBeforeInBlock(ancestor)) {
        last = ancestor;
      }
    }
  });
  return last;
}

static FailureOr<Operation *> findReuseGroupAnchor(ArrayRef<Operation *> sites, Location loc) {
  auto parentBlock = [](Block *block) -> Block * {
    Operation *parent = block ? block->getParentOp() : nullptr;
    return parent ? parent->getBlock() : nullptr;
  };
  auto commonAncestor = [&](Block *lhs, Block *rhs) -> Block * {
    SmallVector<Block *, 8> lhsChain;
    for (Block *block = lhs; block != nullptr; block = parentBlock(block)) {
      lhsChain.push_back(block);
    }
    for (Block *block = rhs; block != nullptr; block = parentBlock(block)) {
      if (llvm::is_contained(lhsChain, block)) {
        return block;
      }
    }
    return nullptr;
  };
  Block *commonBlock = sites.front()->getBlock();
  for (Operation *site : sites) {
    commonBlock = commonAncestor(commonBlock, site->getBlock());
    if (commonBlock == nullptr) {
      return emitError(loc) << afir::kInvalidLifetimeDiagnostic
                            << ": one reuse group has uses without a common ancestor block";
    }
  }
  Operation *anchor = nullptr;
  for (Operation *site : sites) {
    Operation *representative = commonBlock->findAncestorOpInBlock(*site);
    if (representative == nullptr) {
      return emitError(loc) << afir::kInvalidLifetimeDiagnostic
                            << ": one reuse group has uses without a common ancestor block";
    }
    if (anchor == nullptr || representative->isBeforeInBlock(anchor)) {
      anchor = representative;
    }
  }
  return anchor;
}

LogicalResult CanonicalFunctionMemoryAllocator::materializeLocalGroup(const CanonicalGroupKey &key,
                                                                      SmallVector<CanonicalValueInfo *> &members) {
  auto resource = resources.find(key.resource);
  if (resource == resources.end()) {
    return failure();
  }
  llvm::sort(members,
             [](const CanonicalValueInfo *lhs, const CanonicalValueInfo *rhs) { return lhs->ordinal < rhs->ordinal; });
  auto anchorOf = [&](CanonicalValueInfo *member) {
    return key.resource.queue && member->view.sourceOwner ? member->view.sourceOwner : valueAnchor(member->value);
  };
  SmallVector<Operation *> sites;
  for (CanonicalValueInfo *member : members) {
    sites.push_back(anchorOf(member));
    for (Operation *user : member->value.getUsers()) {
      sites.push_back(user);
    }
  }
  auto groupAnchor = findReuseGroupAnchor(sites, members.front()->value.getLoc());
  if (failed(groupAnchor)) {
    return failure();
  }
  Operation *anchor = *groupAnchor;
  for (CanonicalValueInfo *member : members) {
    for (Operation *user : member->value.getUsers()) {
      if (failed(afir::verifyReduceScratchScope(member->value, user))) {
        return failure();
      }
    }
    if (key.lifeTimeAxisId >= 0 && !isInLoopScope(anchorOf(member), key.lifeTimeAxisId)) {
      return emitError(member->value.getLoc())
             << afir::kInvalidLifetimeDiagnostic << ": loop lifetime resource is outside its loop scope";
    }
  }
  OpBuilder builder(anchor);
  auto rawType =
      ascendc::LocalTensorType::get({ShapedType::kDynamic}, IntegerType::get(context, 8, IntegerType::Unsigned));
  FailureOr<Value> raw = createLocalRaw(builder, key, resource->second, anchor, rawType, members);
  if (failed(raw)) {
    return failure();
  }
  Value offset = builder.create<arith::ConstantIndexOp>(anchor->getLoc(), members.front()->view.offset);
  LocalSlab slab{*raw, rawType, offset, key.resource.queue};
  for (CanonicalValueInfo *member : members) {
    if (failed(materializeLocalMember(builder, slab, *member))) {
      return failure();
    }
  }
  // 队列的归还由 `closeQueueEpoch` 在整条复用链结束时生成一次，不在这里逐组生成。
  return success();
}

LogicalResult CanonicalFunctionMemoryAllocator::materializeCanonicalValues() {
  std::map<CanonicalGroupKey, SmallVector<CanonicalValueInfo *>> groups;
  for (Value value : valueOrder) {
    auto it = values.find(value);
    if (it == values.end() || failed(collectMaterializationGroup(it->second, groups))) {
      return failure();
    }
  }

  // Use DenseMap for SSA lookup only; materialize GlobalTensor operations in an
  // explicit order independent of map traversal.
  for (Value value : valueOrder) {
    auto it = values.find(value);
    if (it == values.end() || resolveAlias(value) != value || it->second.view.resource.space != afir::Space::GM) {
      continue;
    }
    if (failed(materializeGlobalValue(it->second, cast<MemRefType>(value.getType())))) {
      return failure();
    }
  }

  // 队列复用链按绑定收集的程序序排列，不能依赖 groups 的 reuse_id 键序。
  SmallVector<std::pair<const CanonicalGroupKey *, SmallVector<CanonicalValueInfo *> *>> ordered;
  for (auto &entry : groups) {
    ordered.emplace_back(&entry.first, &entry.second);
  }
  auto firstOrdinal = [](const SmallVector<CanonicalValueInfo *> &members) {
    unsigned first = std::numeric_limits<unsigned>::max();
    for (const CanonicalValueInfo *member : members) {
      first = std::min(first, member->ordinal);
    }
    return first;
  };
  llvm::stable_sort(ordered, [&](const auto &lhs, const auto &rhs) {
    if (lhs.first->resource < rhs.first->resource) {
      return true;
    }
    if (rhs.first->resource < lhs.first->resource) {
      return false;
    }
    return firstOrdinal(*lhs.second) < firstOrdinal(*rhs.second);
  });
  for (auto &entry : ordered) {
    if (failed(materializeLocalGroup(*entry.first, *entry.second))) {
      return failure();
    }
  }
  for (auto &entry : queueEpochs) {
    if (failed(closeQueueEpoch(entry.first, entry.second))) {
      return failure();
    }
  }
  return success();
}

}  // namespace mlir::ascend::kernel::detail
