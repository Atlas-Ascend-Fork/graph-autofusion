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
#include "dialect/AFIR/Analysis/ResourceIds.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"
#include "dialect/AFIR/Transforms/Symbolic/SymbolicTransformDetail.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
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
namespace {
constexpr int kResourcePasses = 2;
}

LogicalResult CanonicalFunctionMemoryAllocator::validateResourceProperties(const CanonicalValueInfo &info, bool queue) {
  if (queue) {
    if (info.buffer.getSlotCount() < 1 || info.buffer.getQueueDepth() < 1) {
      return emitError(info.value.getLoc())
             << afir::kInvalidBindingDiagnostic << ": queue descriptor requires slot_count and queue_depth >= 1";
    }
    return success();
  }
  if (info.buffer.getSlotCount() != -1 || info.buffer.getQueueDepth() != -1) {
    return emitError(info.value.getLoc())
           << afir::kInvalidBindingDiagnostic << ": compute buffer cannot carry queue fields";
  }
  return success();
}

LogicalResult CanonicalFunctionMemoryAllocator::registerResource(const CanonicalValueInfo &info,
                                                                 CanonicalResourceKey key, bool queue) {
  auto existing = resources.find(key);
  if (existing == resources.end()) {
    CanonicalResourceInfo resource{key,         info.buffer, {}, {}, {}, getTransferDirection(info.view.sourceOwner),
                                   info.ordinal};
    resources.emplace(key, std::move(resource));
  } else {
    TransferDirection direction = getTransferDirection(info.view.sourceOwner);
    if (queue && existing->second.queueDirection == TransferDirection::None) {
      existing->second.queueDirection = direction;
    } else if (queue && direction != TransferDirection::None && existing->second.queueDirection != direction) {
      return emitError(info.value.getLoc())
             << afir::kInvalidBindingDiagnostic << ": one queue resource cannot be both TransferIn and TransferOut";
    }
    auto &descriptor = existing->second.descriptor;
    // 字段清单在 `afir::canonicalResourceFactsAgree`（唯一一处）——bridge 的
    // `CheckResourceCompatibility` 查的是同一份。哪些字段不参与以及为什么，写在那个
    // 声明上；这里不再重抄一遍。
    if (!afir::canonicalResourceFactsAgree(afir::getCanonicalResourceFacts(descriptor),
                                           afir::getCanonicalResourceFacts(info.buffer))) {
      return emitError(info.value.getLoc())
             << afir::kInvalidBindingDiagnostic << ": conflicting descriptors for one canonical resource";
    }
    // 聚合 `slot_count` 取 max、`alignment_bytes` 取最小公倍数（理由见上面那份清单；
    // -1 表示没有要求，按 1 参与）。`reuse_id` 不参与相等比较，于是聚合后的这份
    // descriptor 上它已经没有意义——清成 -1 而不是留着第一个看到的值，免得下游把它
    // 当成「这条队列的复用组」。
    const int64_t slotCount = std::max(descriptor.getSlotCount(), info.buffer.getSlotCount());
    const int64_t alignmentBytes =
        afir::aggregateAlignmentBytes(descriptor.getAlignmentBytes(), info.buffer.getAlignmentBytes());
    if (slotCount != descriptor.getSlotCount() || alignmentBytes != descriptor.getAlignmentBytes() ||
        descriptor.getReuseId() != -1) {
      descriptor = afir::BufferDescAttr::get(descriptor.getContext(), descriptor.getSpace(), descriptor.getBufferId(),
                                             /*reuse_id=*/-1, slotCount, descriptor.getQueueDepth(),
                                             descriptor.getSizeBytesExpr(), descriptor.getSizeFromResource(),
                                             alignmentBytes, descriptor.getLifeTimeAxisId());
    }
  }
  return success();
}

LogicalResult CanonicalFunctionMemoryAllocator::validateResource(const CanonicalValueInfo &info) {
  afir::Space space = info.view.resource.space;
  if (space == afir::Space::GM) {
    if (info.buffer.getSlotCount() != -1 || info.buffer.getQueueDepth() != -1) {
      return emitError(info.value.getLoc())
             << afir::kInvalidBindingDiagnostic << ": GM descriptor cannot carry queue fields";
    }
    return success();
  }
  if (space != afir::Space::UB) {
    return emitError(info.value.getLoc())
           << afir::kUnsupportedOperationDiagnostic << ": canonical materialization supports only GM and UB resources";
  }
  if (info.buffer.getBufferId() < 0) {
    return emitError(info.value.getLoc()) << afir::kInvalidBindingDiagnostic << ": UB descriptor requires buffer_id";
  }
  if (info.buffer.getLifeTimeAxisId() == std::numeric_limits<int64_t>::min()) {
    return emitError(info.value.getLoc())
           << afir::kInvalidLifetimeDiagnostic << ": temporary UB descriptor requires life_time_axis_id";
  }
  // 两个来源二选一——都缺就没人说得出这块存储多大。「不能都有」由 `#afir.buffer_desc`
  // 的 verifier 拦，这里只补「不能都没有」。
  if (!info.buffer.getSizeBytesExpr() && !info.buffer.getSizeFromResource()) {
    return emitError(info.value.getLoc()) << afir::kInvalidBindingDiagnostic
                                          << ": canonical UB descriptor requires size_bytes_expr or size_from_resource";
  }
  bool queue = isQueueBinding(info.view, info.buffer);
  if (failed(validateResourceProperties(info, queue))) {
    return failure();
  }
  CanonicalResourceKey key{space, queue, info.buffer.getBufferId()};
  return registerResource(info, key, queue);
}

LogicalResult CanonicalFunctionMemoryAllocator::validateCanonicalBindings() {
  for (const auto &entry : values) {
    const StorageView &view = entry.second.view;
    if ((view.resource.space != afir::Space::GM && view.resource.bufferId < 0) ||
        (view.offset < 0 && view.offset != ShapedType::kDynamic)) {
      return emitError(entry.first.getLoc())
             << afir::kInvalidBindingDiagnostic << ": StorageView has an invalid resource or offset";
    }
    if (view.offset == ShapedType::kDynamic) {
      return emitError(entry.first.getLoc())
             << afir::kInvalidBindingDiagnostic << ": dynamic StorageView offset requires an explicit view binding";
    }
    if (view.sizes.size() != view.strides.size()) {
      return emitError(entry.first.getLoc()) << afir::kInvalidBindingDiagnostic << ": StorageView rank/stride mismatch";
    }
    if (llvm::any_of(view.strides, [](int64_t stride) { return stride < 0 && stride != ShapedType::kDynamic; })) {
      return emitError(entry.first.getLoc())
             << afir::kInvalidBindingDiagnostic << ": StorageView contains a negative stride";
    }
    if (failed(validateResource(entry.second))) {
      return failure();
    }
  }
  return success();
}

LogicalResult CanonicalFunctionMemoryAllocator::collectExpressionSymbols(afir::ExprAttr expression,
                                                                         llvm::StringSet<> &symbols) {
  if (!expression) {
    return success();
  }
  auto parsed = afir::Expression::Parse(expression.getValue().getValue());
  if (!parsed) {
    return emitError(func.getLoc()) << afir::kInvalidBindingDiagnostic
                                    << ": invalid resource size expression: " << parsed.error().message;
  }
  auto freeSymbols = afir::detail::ExpressionAccess::GetFreeSymbols(parsed.value());
  if (!freeSymbols) {
    return emitError(func.getLoc()) << afir::kInvalidBindingDiagnostic
                                    << ": failed to resolve resource size expression symbols: "
                                    << freeSymbols.error().message;
  }
  for (const std::string &symbol : freeSymbols.value()) {
    symbols.insert(symbol);
  }
  return success();
}

LogicalResult CanonicalFunctionMemoryAllocator::collectRequiredSizeSymbols(llvm::StringSet<> &requiredSymbols) {
  if (auto graph = func->getAttrOfType<afir::AscGraphAttrGroupsAttr>(afir::kAscGraphAttrName)) {
    for (afir::ExprAttr expression : graph.getSizeVar()) {
      if (failed(collectExpressionSymbols(expression, requiredSymbols))) {
        return failure();
      }
    }
  }
  for (Value value : valueOrder) {
    auto it = values.find(value);
    if (it == values.end()) {
      continue;
    }
    if (failed(collectExpressionSymbols(it->second.buffer.getSizeBytesExpr(), requiredSymbols))) {
      return failure();
    }
    // `size_from_resource` 引用的字段名不在 IR 里，按 descriptor 自己的 id 与队列身份
    // 派生——与 `AFIRKernelBodySignature` 往 TilingData 里追加字段用的是同一处拼法。
    if (it->second.buffer.getSizeFromResource()) {
      requiredSymbols.insert(afir::makeCanonicalResourceSizeName(afir::isCanonicalQueueResource(it->second.buffer),
                                                                 it->second.buffer.getBufferId()));
    }
  }
  return success();
}

LogicalResult CanonicalFunctionMemoryAllocator::bindSizeSymbolsToTiling(const llvm::StringSet<> &requiredSymbols) {
  FailureOr<Value> tiling = getCanonicalTilingArgument(func);
  if (failed(tiling)) {
    return failure();
  }
  auto tilingType = getTilingSchema((*tiling).getType());
  for (Attribute nameAttr : tilingType.getNamesAttr()) {
    auto name = cast<StringAttr>(nameAttr);
    if (!requiredSymbols.contains(name.getValue())) {
      continue;
    }
    FailureOr<afir::ExprAttr> expression = afir::makeExprAttr(context, name.getValue());
    if (failed(expression)) {
      return failure();
    }
    // 绑定的是 `afir.tiling_param`，不是当场建一个 `emitasc.member`——后者是后端算子，
    // 建在这里就让 `emitasc` 提前一个阶段进入函数体（判据 C3）。循环生成那一步已经把
    // 同一批符号物化过，这里取到的多半就是那一份；`getOrCreateTilingParam` 保证一个
    // 名字一份，两条路径不会各建一个。
    //
    // 不进 `createdOps`：那是本 pass 失败时的回滚清单，而 `afir.tiling_param` 可能是
    // 上一个 pass 的产物，回滚它会把别人的东西一起删掉。真新建了又回滚时留下的是一个
    // `Pure` 的无用值，既不影响正确性，也会被后续的 CSE/DCE 收走。
    symbolicBindings[*expression] = afir::getOrCreateTilingParam(func.front(), name.getValue()).getResult();
  }
  for (const auto &entry : requiredSymbols) {
    FailureOr<afir::ExprAttr> expression = afir::makeExprAttr(context, entry.getKey());
    if (failed(expression) || symbolicBindings.find(*expression) == symbolicBindings.end()) {
      return func.emitError("TilingData is missing canonical size expression field '") << entry.getKey() << "'";
    }
  }
  return success();
}

LogicalResult CanonicalFunctionMemoryAllocator::collectSymbolicBindings() {
  llvm::StringSet<> requiredSymbols;
  if (failed(collectRequiredSizeSymbols(requiredSymbols))) {
    return failure();
  }
  if (requiredSymbols.empty()) {
    return success();
  }
  if (!materializationAnchor && !func.front().empty()) {
    materializationAnchor = firstMaterializationAnchor();
  }
  return bindSizeSymbolsToTiling(requiredSymbols);
}

FailureOr<Value> CanonicalFunctionMemoryAllocator::materializeExpression(afir::ExprAttr expression, Location loc) {
  int64_t literal = 0;
  if (!expression.getValue().getValue().getAsInteger(10, literal)) {
    if (literal <= 0) {
      emitError(loc) << afir::kInvalidBindingDiagnostic
                     << ": canonical size expression must evaluate to a positive value";
      return failure();
    }
    return resourceBuilder->create<arith::ConstantIndexOp>(loc, literal).getResult();
  }
  FailureOr<Value> value = afir::materializeSymbolicExpr(*resourceBuilder, loc, expression, symbolicBindings,
                                                         resourceBuilder->getIndexType());
  if (failed(value)) {
    return failure();
  }
  if (symbolicBindings.empty()) {
    emitError(loc) << afir::kInvalidBindingDiagnostic << ": symbolic size expression has no bound runtime dimension";
    return failure();
  }
  Value zero = resourceBuilder->create<arith::ConstantIndexOp>(loc, 0);
  createdOps.push_back(zero.getDefiningOp());
  Value positive = resourceBuilder->create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, *value, zero);
  createdOps.push_back(positive.getDefiningOp());
  OperationState assertState(loc, "cf.assert");
  assertState.addOperands(positive);
  assertState.addAttribute("msg", resourceBuilder->getStringAttr("canonical resource size must be positive"));
  createdOps.push_back(resourceBuilder->create(assertState));
  return value;
}

FailureOr<Value> CanonicalFunctionMemoryAllocator::materializeResourceSize(const afir::BufferDescAttr &descriptor,
                                                                           Location loc) {
  // 片上资源的 slot 字节数是 ATT 侧按 `max(同 id 的每个张量) ⊔ Σ(同 reuse 组)` 聚合出来
  // 的，MLIR 这一层只能引用它。descriptor 说「大小取自本资源」，字段名在这里派生。
  if (descriptor.getSizeFromResource()) {
    FailureOr<afir::ExprAttr> reference = afir::makeExprAttr(
        context,
        afir::makeCanonicalResourceSizeName(afir::isCanonicalQueueResource(descriptor), descriptor.getBufferId()));
    if (failed(reference)) {
      return failure();
    }
    return materializeExpression(*reference, loc);
  }
  auto expression = descriptor.getSizeBytesExpr();
  if (!expression) {
    emitError(loc) << afir::kInvalidBindingDiagnostic
                   << ": canonical resource requires size_bytes_expr or size_from_resource";
    return failure();
  }
  return materializeExpression(expression, loc);
}

FailureOr<Value> CanonicalFunctionMemoryAllocator::alignResourceSize(Value size, int64_t alignment, Location loc) {
  Value zero = resourceBuilder->create<arith::ConstantIndexOp>(loc, 0);
  Value positive = resourceBuilder->create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, size, zero);
  createdOps.push_back(zero.getDefiningOp());
  createdOps.push_back(positive.getDefiningOp());
  OperationState positiveAssert(loc, "cf.assert");
  positiveAssert.addOperands(positive);
  positiveAssert.addAttribute("msg", resourceBuilder->getStringAttr("resource size must be positive"));
  createdOps.push_back(resourceBuilder->create(positiveAssert));
  if (alignment == -1) {
    return size;
  }
  if (alignment < 1) {
    return emitError(loc) << afir::kInvalidBindingDiagnostic << ": alignment_bytes must be positive or -1";
  }
  Value alignmentValue = resourceBuilder->create<arith::ConstantIndexOp>(loc, alignment);
  Value blocks = resourceBuilder->create<arith::CeilDivSIOp>(loc, size, alignmentValue);
  createdOps.push_back(alignmentValue.getDefiningOp());
  createdOps.push_back(blocks.getDefiningOp());
  auto multiply = resourceBuilder->create<arith::MulSIExtendedOp>(loc, blocks, alignmentValue);
  createdOps.push_back(multiply.getOperation());
  Value highZero = resourceBuilder->create<arith::ConstantIndexOp>(loc, 0);
  Value lowNonNegative =
      resourceBuilder->create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, multiply.getLow(), highZero);
  Value highZeroCheck =
      resourceBuilder->create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, multiply.getHigh(), highZero);
  Value noOverflow = resourceBuilder->create<arith::AndIOp>(loc, lowNonNegative, highZeroCheck);
  createdOps.push_back(highZero.getDefiningOp());
  createdOps.push_back(lowNonNegative.getDefiningOp());
  createdOps.push_back(highZeroCheck.getDefiningOp());
  createdOps.push_back(noOverflow.getDefiningOp());
  OperationState assertState(loc, "cf.assert");
  assertState.addOperands(noOverflow);
  assertState.addAttribute("msg", resourceBuilder->getStringAttr("aligned resource size must not overflow"));
  createdOps.push_back(resourceBuilder->create(assertState));
  return multiply.getLow();
}

Operation *CanonicalFunctionMemoryAllocator::firstAfirOperation() {
  for (Operation &operation : func.front()) {
    // `afir.tiling_param` 不算：它是函数里所有运行期尺寸的根，物化出来的东西全都要
    // 排在它之后。把它当锚点等于让资源插在自己依赖的值之前。
    if (isAfirOperation(&operation) && !isa<afir::TilingParamOp>(operation)) {
      return &operation;
    }
  }
  return func.front().getTerminator();
}

// 资源物化的插入点：入口块的第一个 Op，但要跳过开头那一段 `afir.tiling_param`。
// 队列与缓冲的字节数就是从这些值算出来的（`materializeExpression`），锚点排在它们
// 之前会让 `arith.muli` 引用一个还没定义的值——是支配关系错误，不是顺序不好看。
Operation *CanonicalFunctionMemoryAllocator::firstMaterializationAnchor() {
  for (Operation &operation : func.front()) {
    if (!isa<afir::TilingParamOp>(operation)) {
      return &operation;
    }
  }
  return func.front().getTerminator();
}

ascendc::TPosition CanonicalFunctionMemoryAllocator::getPosition(const CanonicalResourceInfo &resource) const {
  if (resource.key.queue) {
    return resource.queueDirection == TransferDirection::Out ? ascendc::TPosition::VECOUT : ascendc::TPosition::VECIN;
  }
  return ascendc::TPosition::VECCALC;
}

LogicalResult CanonicalFunctionMemoryAllocator::createQueueResource(Value pipe, CanonicalResourceInfo &resource) {
  auto queueType = ascendc::QueueType::get(context, getPosition(resource), resource.descriptor.getQueueDepth());
  resource.resource = resourceBuilder->create<ascendc::QueueOp>(func.getLoc(), queueType);
  auto *queue = resource.resource.getDefiningOp();
  createdOps.push_back(queue);
  FailureOr<Value> size = materializeResourceSize(resource.descriptor, func.getLoc());
  if (failed(size)) {
    return failure();
  }
  size = alignResourceSize(*size, resource.descriptor.getAlignmentBytes(), func.getLoc());
  if (failed(size)) {
    return failure();
  }
  Value count = resourceBuilder->create<arith::ConstantIntOp>(func.getLoc(), resource.descriptor.getSlotCount(), 32);
  auto init = resourceBuilder->create<ascendc::TPipeInitQueueOp>(func.getLoc(), pipe, resource.resource, count, *size);
  createdOps.push_back(init.getOperation());
  return success();
}

LogicalResult CanonicalFunctionMemoryAllocator::createBufferResource(Value pipe, CanonicalResourceInfo &resource) {
  auto bufferType = ascendc::TBufType::get(context, ascendc::TPosition::VECCALC);
  resource.resource = resourceBuilder->create<ascendc::TBufOp>(func.getLoc(), bufferType);
  auto *buffer = resource.resource.getDefiningOp();
  createdOps.push_back(buffer);
  FailureOr<Value> size = materializeResourceSize(resource.descriptor, func.getLoc());
  if (failed(size)) {
    return failure();
  }
  size = alignResourceSize(*size, resource.descriptor.getAlignmentBytes(), func.getLoc());
  if (failed(size)) {
    return failure();
  }
  auto init = resourceBuilder->create<ascendc::TPipeInitBufferOp>(func.getLoc(), pipe, resource.resource, *size);
  createdOps.push_back(init.getOperation());
  return success();
}

LogicalResult CanonicalFunctionMemoryAllocator::createCanonicalResources() {
  Operation *anchor = materializationAnchor;
  if (!anchor) {
    anchor = func.front().empty() ? func.front().getTerminator() : firstMaterializationAnchor();
    materializationAnchor = anchor;
  }
  resourceBuilder = std::make_unique<OpBuilder>(anchor);
  if (resources.empty()) {
    return success();
  }
  Value pipe = resourceBuilder->create<ascendc::PipeOp>(func.getLoc());
  createdOps.push_back(pipe.getDefiningOp());
  // 队列先于 TBuf 发 `InitBuffer`：TPipe 按调用序线性分配 UB，这决定了缓冲在 UB 里的
  // 布局。ConfusionTranspose 一类 scatter 型 API（`TransDataTo5HD` 地址表形态）把尾块
  // 的目的地址表混排「目标张量（队列槽）+ 共享 tmp（TBuf）」两块区域——CPU 模拟器
  // （`libcpudebug` 的 `scatter_vnchwconv_b32`）在 tmp 区域位于目标区域**之前**时会丢写
  // 尾块（输出尾行全 0，e2e `load_transpose_store_2x37x23` 实测）。队列在前、TBuf 在后
  // 让 tmp 恒在队列槽之上，与 legacy 的布局一致。只改发射顺序，不动 `resources` 的键序
  // ——那张表还有别的消费方。
  for (auto pass = 0; pass < kResourcePasses; ++pass) {
    for (auto &entry : resources) {
      auto &resource = entry.second;
      if ((pass == 0) != resource.key.queue) {
        continue;
      }
      LogicalResult created =
          resource.key.queue ? createQueueResource(pipe, resource) : createBufferResource(pipe, resource);
      if (failed(created)) {
        return failure();
      }
    }
  }
  resourceBuilder->setInsertionPoint(materializationAnchor);
  return success();
}

}  // namespace mlir::ascend::kernel::detail
