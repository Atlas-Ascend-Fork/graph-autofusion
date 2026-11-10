/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/Implementations/Synchronization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "target/Ascend/Capability/CapabilityProvider.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <optional>
#include <utility>

using namespace mlir::afir;

namespace mlir::ascend::kernel {
namespace {

using ascend::instruction::CapabilityReason;
using ascend::instruction::CapabilitySupport;
using ascend::instruction::ImplementationKey;
using ascend::instruction::ImplementationRegistry;
using ascend::instruction::PatternBuildContext;

// AFIR 的 pipe 词表是抽象引擎（ADR-20），到物理 pipe 的映射在这里，且只在这里。
ascendc::Pipe toAscendCPipe(afir::Pipe pipe) {
  switch (pipe) {
    case afir::Pipe::TransferIn:
      return ascendc::Pipe::PIPE_MTE2;
    case afir::Pipe::TransferOut:
      return ascendc::Pipe::PIPE_MTE3;
    case afir::Pipe::TransferLocal:
      return ascendc::Pipe::PIPE_MTE1;
    case afir::Pipe::Cube:
      return ascendc::Pipe::PIPE_M;
    case afir::Pipe::Fixpipe:
      return ascendc::Pipe::PIPE_FIX;
    case afir::Pipe::Scalar:
      return ascendc::Pipe::PIPE_S;
    case afir::Pipe::Vector:
      break;
  }
  return ascendc::Pipe::PIPE_V;
}

// `HardEvent` 本身就是一个 pipe 对枚举（`mte2_v` / `v_mte3` / ...），这正是 ADR-19 让
// `!afir.token` 携带 pipe 对的兑现点：映射是查表，不需要在下降期重新推断方向。
//
// 写成一张按 `(from, to)` 索引的表而不是一串 `if`：七台引擎有 42 个有序对，`HardEvent`
// 里存在的是其中 33 个，剩下 9 个硬件没有对应事件。用 `if` 链表达这件事，读的人无法
// 一眼看出哪些缺、缺的是不是写漏了。表里的 `kNone` 是硬件没有，不是没写。
//
// `from == to` 不在表里——同 pipe 内程序序即执行序，那条边由 `TokenType` 的 verifier
// 在解析期就拒掉了，对角线永远取不到。
constexpr ascendc::HardEvent kNone = ascendc::HardEvent::MAX;

constexpr unsigned kPipeCount = 7;

constexpr unsigned pipeIndex(afir::Pipe pipe) {
  return static_cast<unsigned>(pipe);
}

// 行是 `from`，列是 `to`，下标即 `afir::Pipe` 的枚举值。
constexpr ascendc::HardEvent kHardEventTable[kPipeCount][kPipeCount] = {
    // from = TransferIn (MTE2)
    {kNone, ascendc::HardEvent::MTE2_MTE3, ascendc::HardEvent::MTE2_V, ascendc::HardEvent::MTE2_MTE1,
     ascendc::HardEvent::MTE2_M, ascendc::HardEvent::MTE2_FIX, ascendc::HardEvent::MTE2_S},
    // from = TransferOut (MTE3)
    {ascendc::HardEvent::MTE3_MTE2, kNone, ascendc::HardEvent::MTE3_V, ascendc::HardEvent::MTE3_MTE1, kNone, kNone,
     ascendc::HardEvent::MTE3_S},
    // from = Vector (V)
    {ascendc::HardEvent::V_MTE2, ascendc::HardEvent::V_MTE3, kNone, ascendc::HardEvent::V_MTE1, ascendc::HardEvent::V_M,
     kNone, ascendc::HardEvent::V_S},
    // from = TransferLocal (MTE1)
    {ascendc::HardEvent::MTE1_MTE2, ascendc::HardEvent::MTE1_MTE3, ascendc::HardEvent::MTE1_V, kNone,
     ascendc::HardEvent::MTE1_M, ascendc::HardEvent::MTE1_FIX, kNone},
    // from = Cube (M)
    {ascendc::HardEvent::M_MTE2, kNone, ascendc::HardEvent::M_V, ascendc::HardEvent::M_MTE1, kNone,
     ascendc::HardEvent::M_FIX, ascendc::HardEvent::M_S},
    // from = Fixpipe (FIX)
    {ascendc::HardEvent::FIX_MTE2, ascendc::HardEvent::FIX_MTE3, kNone, ascendc::HardEvent::FIX_MTE1,
     ascendc::HardEvent::FIX_M, kNone, ascendc::HardEvent::FIX_S},
    // from = Scalar (S)
    {ascendc::HardEvent::S_MTE2, ascendc::HardEvent::S_MTE3, ascendc::HardEvent::S_V, kNone, kNone, kNone, kNone},
};

std::optional<ascendc::HardEvent> toHardEvent(afir::Pipe from, afir::Pipe to) {
  ascendc::HardEvent event = kHardEventTable[pipeIndex(from)][pipeIndex(to)];
  if (event == kNone) {
    return std::nullopt;
  }
  return event;
}

std::optional<ascendc::HardEvent> tokenHardEvent(Type type) {
  auto token = dyn_cast<afir::TokenType>(type);
  if (!token) {
    return std::nullopt;
  }
  return toHardEvent(token.getFrom(), token.getTo());
}

struct SynchronizationMapping {
  // FetchEventID 返回的事件号宽度。i8 与 `api/ascendc` 既有用法一致
  // （`Transforms/InsertSync.cpp` 的 `createSetGetValueSync`）。
  unsigned eventIdWidth = 8;
};

template <typename OpT>
class PipePairCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit PipePairCapabilityProvider(std::shared_ptr<const SynchronizationMapping> mapping)
      : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation, const ascend::instruction::QueryContext &) const override {
    auto op = dyn_cast<OpT>(operation);
    if (!op) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedTarget);
    }
    if constexpr (std::is_same_v<OpT, PipeBarrierOp> || std::is_same_v<OpT, BarrierAllOp> ||
                  std::is_same_v<OpT, SyncAllOp>) {
      // 无 token 的原语：pipe_barrier 的 pipe 固定可映射；barrier_all / sync_all 的落点
      // （PIPE_ALL / SyncAll）不依赖任何操作数事实。
      return CapabilitySupport::supported();
    } else {
      return tokenHardEvent(op.getToken().getType())
                 ? CapabilitySupport::supported()
                 : CapabilitySupport::unsupported(CapabilityReason::UnsupportedTarget);
    }
  }

 private:
  std::shared_ptr<const SynchronizationMapping> mapping_;
};

class LowerEventRecordOp final : public OpConversionPattern<EventRecordOp> {
 public:
  LowerEventRecordOp(TypeConverter &converter, MLIRContext *context, ImplementationKey key,
                     std::shared_ptr<const SynchronizationMapping> mapping)
      : OpConversionPattern(converter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(EventRecordOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    std::optional<ascendc::HardEvent> event = tokenHardEvent(op.getToken().getType());
    if (!event) {
      return rewriter.notifyMatchFailure(op, "pipe pair has no AscendC HardEvent");
    }
    Value pipe = findFunctionPipe(op);
    if (!pipe) {
      return rewriter.notifyMatchFailure(
          op, "AscendC event slots are owned by the function's TPipe, which resource materialization did not create");
    }
    // 槽位按 legacy 的纪律管理：record 侧 `AllocEventID` 占位，wait 侧 `WaitFlag` 后
    // `ReleaseEventID` 归还。`FetchEventID` 只取第一个空闲位不占位，交叠的同型对会共享
    // 同一个事件号（硬件上要么挂死要么丢失排序），不能用。
    Value eventId = ascendc::TPipeAllocEventIDOp::create(rewriter, op.getLoc(),
                                                         rewriter.getIntegerType(mapping_->eventIdWidth), pipe, *event);
    ascendc::SetFlagOp::create(rewriter, op.getLoc(), *event, eventId);
    rewriter.replaceOp(op, eventId);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const SynchronizationMapping> mapping_;
};

class LowerEventWaitOp final : public OpConversionPattern<EventWaitOp> {
 public:
  LowerEventWaitOp(TypeConverter &converter, MLIRContext *context, ImplementationKey key,
                   std::shared_ptr<const SynchronizationMapping> mapping)
      : OpConversionPattern(converter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(EventWaitOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    // pipe 对读的是原 Op 上的 token 类型：转换后的操作数已经是事件号，方向信息只在
    // 原类型上。这也是 ADR-19 把 pipe 对放进类型、而不是放进一个旁路属性的原因。
    std::optional<ascendc::HardEvent> event = tokenHardEvent(op.getToken().getType());
    if (!event) {
      return rewriter.notifyMatchFailure(op, "pipe pair has no AscendC HardEvent");
    }
    Value pipe = findFunctionPipe(op);
    if (!pipe) {
      return rewriter.notifyMatchFailure(
          op, "AscendC event slots are owned by the function's TPipe, which resource materialization did not create");
    }
    auto wait = rewriter.replaceOpWithNewOp<ascendc::WaitFlagOp>(op, *event, adaptor.getToken());
    // 归还 record 侧占下的事件槽。交叠的对在 wait 前各自持有不同的槽位，顺序化完成后
    // 这里释放，后续同型 record 可以复用（legacy `SyncMte3ToMte2` 的 Alloc/Release 纪律）。
    rewriter.setInsertionPointAfter(wait);
    ascendc::TPipeReleaseEventIDOp::create(rewriter, op.getLoc(), pipe, adaptor.getToken(), *event);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const SynchronizationMapping> mapping_;
};

class LowerPipeBarrierOp final : public OpConversionPattern<PipeBarrierOp> {
 public:
  LowerPipeBarrierOp(TypeConverter &converter, MLIRContext *context, ImplementationKey key,
                     std::shared_ptr<const SynchronizationMapping> mapping)
      : OpConversionPattern(converter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(PipeBarrierOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    rewriter.replaceOpWithNewOp<ascendc::PipeBarrierOp>(op, toAscendCPipe(op.getPipe()));
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const SynchronizationMapping> mapping_;
};

// 全引擎屏障：`afir.Pipe` 不设「全部」取值（单引擎纪律），跨 pipe 全序独立成 Op，
// 下降直接取 PIPE_ALL。
class LowerBarrierAllOp final : public OpConversionPattern<BarrierAllOp> {
 public:
  LowerBarrierAllOp(TypeConverter &converter, MLIRContext *context, ImplementationKey key,
                    std::shared_ptr<const SynchronizationMapping> mapping)
      : OpConversionPattern(converter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(BarrierAllOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    rewriter.replaceOpWithNewOp<ascendc::PipeBarrierOp>(op, ascendc::Pipe::PIPE_ALL);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const SynchronizationMapping> mapping_;
};

// hard 跨核栅栏显式设置 isAIVOnly，与 GenKernelEntry 一致。
// 当前 MLIR codegen 排除 Cube 融合，只有 AIV 核参与，应使用 AIV-only SyncAll 协议。
// false 对应 AIC+AIV 混模，会等待仅由 AIC 设置的 SYNC_AIC_AIV_FLAG；纯 AIV 路径会死锁。
// 将来支持混模时，需按生成模式选择并打印对应模板实参，不能依赖缺省 UnitAttr。
// soft 变体带 workspace / 核数参数，用于 superkernel feed-sync，暂不注册。
class LowerSyncAllOp final : public OpConversionPattern<SyncAllOp> {
 public:
  LowerSyncAllOp(TypeConverter &converter, MLIRContext *context, ImplementationKey key,
                 std::shared_ptr<const SynchronizationMapping> mapping)
      : OpConversionPattern(converter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(SyncAllOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    rewriter.replaceOpWithNewOp<ascendc::SyncAllHardOp>(op, rewriter.getUnitAttr());
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const SynchronizationMapping> mapping_;
};

}  // namespace

// 事件槽的持有者是 `TPipe`。资源物化阶段已经建了唯一的一个（`CanonicalResources.cpp`
// 的 `ascendc.pipe`），复用它而不是再建一个——理由见头文件。
// Query the last writer at the use, not the queue position: vector operations may
// reuse VECIN queue storage. Keep member subindices distinct when matching views.
namespace {

Value getStorageRoot(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (auto cast = dyn_cast<ascendc::LocalTensorReinterpretCastOp>(def)) {
      value = cast.getIn();
    } else if (isa<afir::AFIRDestinationStyleInterface>(def)) {
      auto partition = afir::getOperandPartition(def);
      if (failed(partition) || partition->destinationCount != 1) {
        break;
      }
      value = partition->destinations(def)[0];
    } else {
      break;
    }
  }
  return value;
}

struct WriterEventLookup {
  bool matched = false;
  std::optional<ascendc::HardEvent> event;
};

WriterEventLookup findWriterEvent(Operation *anchor, Value target) {
  for (Operation *writer = anchor->getPrevNode(); writer; writer = writer->getPrevNode()) {
    if (!isa<afir::AFIRDestinationStyleInterface>(writer)) {
      continue;
    }
    auto partition = afir::getOperandPartition(writer);
    if (failed(partition)) {
      continue;
    }
    for (Value destination : partition->destinations(writer)) {
      if (getStorageRoot(destination) != target) {
        continue;
      }
      if (writer->hasTrait<OpTrait::AFIRTransferIn>()) {
        return {true, ascendc::HardEvent::MTE2_S};
      }
      if (writer->hasTrait<OpTrait::AFIRTransferOut>()) {
        return {true, std::nullopt};
      }
      return {true, ascendc::HardEvent::V_S};
    }
  }
  return {};
}

std::optional<ascendc::HardEvent> findProducerEvent(Value source) {
  Value current = source;
  while (Operation *producer = current.getDefiningOp()) {
    if (auto cast = dyn_cast<ascendc::LocalTensorReinterpretCastOp>(producer)) {
      current = cast.getIn();
    } else if (auto view = dyn_cast<ascendc::LocalTensorSubIndexOp>(producer)) {
      current = view.getTensor();
    } else if (producer->getName().getDialectNamespace() == "afir") {
      auto partition = afir::getOperandPartition(producer);
      if (failed(partition) || partition->destinationCount != 1) {
        return std::nullopt;
      }
      current = partition->destinations(producer)[0];
    } else {
      break;
    }
  }
  if (auto deque = current.getDefiningOp<ascendc::TQueBindDequeTensorOp>()) {
    if (auto queue = dyn_cast<ascendc::QueueType>(deque.getQueue().getType())) {
      if (queue.getPosition() == ascendc::TPosition::VECIN) {
        return ascendc::HardEvent::MTE2_S;
      }
      return std::nullopt;
    }
    if (auto queBind = dyn_cast<ascendc::QueBindType>(deque.getQueue().getType())) {
      if (queBind.getSrcPosition() == ascendc::TPosition::VECIN) {
        return ascendc::HardEvent::MTE2_S;
      }
    }
    return std::nullopt;
  }
  if (current.getDefiningOp<ascendc::TBufGetTensorOp>()) {
    return ascendc::HardEvent::V_S;
  }
  return std::nullopt;
}

}  // namespace

std::optional<ascendc::HardEvent> scalarReadEvent(Operation *anchor, Value source) {
  Value target = getStorageRoot(source);
  WriterEventLookup writer = findWriterEvent(anchor, target);
  if (writer.matched) {
    return writer.event;
  }
  return findProducerEvent(source);
}

FailureOr<Value> materializeScalarRead(Operation *anchor, Value source, std::optional<ascendc::HardEvent> event,
                                       RewriterBase &rewriter) {
  auto type = dyn_cast<ascendc::LocalTensorType>(source.getType());
  Value pipe = findFunctionPipe(anchor);
  if (!type || !event || !pipe) {
    return failure();
  }
  Value id = rewriter.create<ascendc::TPipeFetchEventIDOp>(anchor->getLoc(), rewriter.getIntegerType(8), pipe, *event);
  rewriter.create<ascendc::SetFlagOp>(anchor->getLoc(), *event, id);
  rewriter.create<ascendc::WaitFlagOp>(anchor->getLoc(), *event, id);
  Value zero = rewriter.create<arith::ConstantIndexOp>(anchor->getLoc(), 0);
  return rewriter.create<ascendc::LocalTensorGetValueOp>(anchor->getLoc(), type.getElementType(), source, zero)
      .getResult();
}

Value findFunctionPipe(Operation *anchor) {
  auto func = anchor->getParentOfType<func::FuncOp>();
  if (!func) {
    return {};
  }
  Value pipe;
  func.walk([&](ascendc::PipeOp op) {
    pipe = op.getResult();
    return WalkResult::interrupt();
  });
  return pipe;
}

LogicalResult registerSynchronizationImplementations(ImplementationRegistry &registry,
                                                     ascend::instruction::SelectionPolicyRegistry *) {
  if (failed(registry.registerImplementation<EventRecordOp, SynchronizationMapping,
                                             PipePairCapabilityProvider<EventRecordOp>>(
          ascend::instruction::ImplementationKind::AscendC, {"event_record"}, SynchronizationMapping{},
          [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
             std::shared_ptr<const SynchronizationMapping> mapping) {
            patterns.add<LowerEventRecordOp>(context.typeConverter, patterns.getContext(), key, std::move(mapping));
          }))) {
    return failure();
  }
  if (failed(
          registry.registerImplementation<EventWaitOp, SynchronizationMapping, PipePairCapabilityProvider<EventWaitOp>>(
              ascend::instruction::ImplementationKind::AscendC, {"event_wait"}, SynchronizationMapping{},
              [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
                 std::shared_ptr<const SynchronizationMapping> mapping) {
                patterns.add<LowerEventWaitOp>(context.typeConverter, patterns.getContext(), key, std::move(mapping));
              }))) {
    return failure();
  }
  if (failed(registry.registerImplementation<BarrierAllOp, SynchronizationMapping,
                                             PipePairCapabilityProvider<BarrierAllOp>>(
          ascend::instruction::ImplementationKind::AscendC, {"barrier_all"}, SynchronizationMapping{},
          [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
             std::shared_ptr<const SynchronizationMapping> mapping) {
            patterns.add<LowerBarrierAllOp>(context.typeConverter, patterns.getContext(), key, std::move(mapping));
          }))) {
    return failure();
  }
  if (failed(registry.registerImplementation<SyncAllOp, SynchronizationMapping, PipePairCapabilityProvider<SyncAllOp>>(
          ascend::instruction::ImplementationKind::AscendC, {"sync_all"}, SynchronizationMapping{},
          [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
             std::shared_ptr<const SynchronizationMapping> mapping) {
            patterns.add<LowerSyncAllOp>(context.typeConverter, patterns.getContext(), key, std::move(mapping));
          }))) {
    return failure();
  }
  return registry
      .registerImplementation<PipeBarrierOp, SynchronizationMapping, PipePairCapabilityProvider<PipeBarrierOp>>(
          ascend::instruction::ImplementationKind::AscendC, {"pipe_barrier"}, SynchronizationMapping{},
          [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
             std::shared_ptr<const SynchronizationMapping> mapping) {
            patterns.add<LowerPipeBarrierOp>(context.typeConverter, patterns.getContext(), key, std::move(mapping));
          });
}

}  // namespace mlir::ascend::kernel
