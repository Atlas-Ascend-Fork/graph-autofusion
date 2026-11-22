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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

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
// 按存储查找标量写入者，再根据写入者确定同步事件。VECIN 存储可能被 Vector 算子复用，
// 仅按队列位置选择 MTE2_S 会遗漏矢量写入的同步。子视图同样回溯到容器，保守收集共享存储的写入流水线。
namespace {

// 沿张量来源回溯到存储容器，以队列或 TBuf 的 SSA 值作为存储标识。
// 写入者和读取者分别通过 alloc_tensor / deque_tensor 或 get_tensor 取得张量，得到的 SSA 值
// 不同，不能直接用张量值匹配。回溯到容器后，才能找到写入者并根据其 trait 确定流水线。
//
// 同一队列的多个成员会归为同一存储，因此可能额外等待其他成员的写入。这种保守处理会增加
// 同步开销，但可以避免遗漏必要的同步。
// handedOver 表示张量是否通过 deque_tensor 或 tbuf.get_tensor 获取，仅用于 containerPipe
// 的兜底判断。alloc_tensor 只申请张量，不表示其中已有有效数据。
struct StorageRoot {
  Value value;
  bool handedOver = false;
};

StorageRoot getStorageRootInfo(Value value) {
  StorageRoot root;
  while (Operation *def = value.getDefiningOp()) {
    if (auto cast = dyn_cast<ascendc::LocalTensorReinterpretCastOp>(def)) {
      value = cast.getIn();
    } else if (auto view = dyn_cast<ascendc::LocalTensorSubIndexOp>(def)) {
      // alloc/deque 两侧的子视图 SSA 不同，但仍属于同一容器。偏移不改变写入流水线；
      // 与其他共享成员一样，按容器保守同步，不能在子视图处丢失生产者和交接状态。
      value = view.getTensor();
    } else if (auto allocated = dyn_cast<ascendc::TQueBindAllocTensorOp>(def)) {
      value = allocated.getQueue();
    } else if (auto dequeued = dyn_cast<ascendc::TQueBindDequeTensorOp>(def)) {
      value = dequeued.getQueue();
      root.handedOver = true;
    } else if (auto dequeuedPos = dyn_cast<ascendc::TQueBindDequeTensorPosOp>(def)) {
      value = dequeuedPos.getQueue();
      root.handedOver = true;
    } else if (auto buffer = dyn_cast<ascendc::TBufGetTensorOp>(def)) {
      value = buffer.getBuffer();
      root.handedOver = true;
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
  root.value = value;
  return root;
}

Value getStorageRoot(Value value) {
  return getStorageRootInfo(value).value;
}

// 根据写入者的 trait 确定流水线。TransferOut 由调用方判为不支持，沿用既有行为。
afir::Pipe writerPipe(Operation *writer) {
  if (writer->hasTrait<OpTrait::AFIRTransferIn>()) {
    return afir::Pipe::TransferIn;
  }
  if (writer->hasTrait<OpTrait::AFIRTransferOut>()) {
    return afir::Pipe::TransferOut;
  }
  return afir::Pipe::Vector;
}

// 仅在函数内找不到写入者时，按容器类型和位置推断流水线，兼容外部写入或手工构造的 IR 片段。
// VECIN 对应搬入流水线，TBuf 对应矢量流水线，与 legacy 的 position 判据一致。
//
// 不能优先使用此规则：VECIN 存储也可能被 Vector 算子复用。此时必须根据实际写入者选择 V_S，
// 按位置选择 MTE2_S 会遗漏矢量写入的同步。
//
// handedOver 还要求张量来自已有数据的获取操作。刚由 alloc_tensor 申请且没有写入者的存储
// 继续判为不支持，避免把未初始化的数据当作可读取的标量。
// afir-singleton-queue-producers.mlir 的 UNKNOWN 用例覆盖这一限制。
std::optional<afir::Pipe> containerPipe(const StorageRoot &root) {
  Value storage = root.value;
  if (!root.handedOver) {
    return std::nullopt;
  }
  if (auto queue = dyn_cast<ascendc::QueueType>(storage.getType())) {
    if (queue.getPosition() == ascendc::TPosition::VECIN) {
      return afir::Pipe::TransferIn;
    }
    return std::nullopt;
  }
  if (auto queBind = dyn_cast<ascendc::QueBindType>(storage.getType())) {
    if (queBind.getSrcPosition() == ascendc::TPosition::VECIN) {
      return afir::Pipe::TransferIn;
    }
    return std::nullopt;
  }
  if (isa<ascendc::TBufType>(storage.getType())) {
    return afir::Pipe::Vector;
  }
  return std::nullopt;
}

std::optional<ascendc::HardEvent> toScalarReadEvent(afir::Pipe pipe) {
  if (pipe == afir::Pipe::TransferOut) {
    return std::nullopt;
  }
  return toHardEvent(pipe, afir::Pipe::Scalar);
}

SmallVector<ascendc::HardEvent> toScalarReadEvents(ArrayRef<afir::Pipe> pipes) {
  SmallVector<ascendc::HardEvent> events;
  for (afir::Pipe pipe : pipes) {
    std::optional<ascendc::HardEvent> event = toScalarReadEvent(pipe);
    if (!event) {
      // 任一流水线无法映射时，整体判为不支持，避免只生成部分必要的同步。
      return {};
    }
    events.push_back(*event);
  }
  return events;
}

}  // namespace

FailureOr<SmallVector<afir::Pipe>> analyzeScalarWriterPipes(Operation *anchor, unsigned inputIndex) {
  FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(anchor);
  if (failed(partition) || inputIndex >= partition->inputCount) {
    return failure();
  }
  Value source = partition->inputs(anchor)[inputIndex];
  if (!isa<ascendc::LocalTensorType>(source.getType())) {
    return failure();
  }
  auto func = anchor->getParentOfType<func::FuncOp>();
  if (!func) {
    return failure();
  }
  StorageRoot root = getStorageRootInfo(source);
  Value target = root.value;
  SmallVector<afir::Pipe> pipes;
  bool transferOut = false;
  // 遍历整个函数，不按写入者与读取者的相对位置过滤，原因见头文件中的排空语义说明。
  // 兄弟 scf.if、循环体和程序序上位于 anchor 之后的写入者都纳入分析。
  func.walk([&](Operation *writer) {
    if (writer == anchor || !isa<afir::AFIRDestinationStyleInterface>(writer)) {
      return;
    }
    FailureOr<afir::OperandPartition> writerPartition = afir::getOperandPartition(writer);
    if (failed(writerPartition)) {
      return;
    }
    for (Value destination : writerPartition->destinations(writer)) {
      if (getStorageRoot(destination) != target) {
        continue;
      }
      afir::Pipe pipe = writerPipe(writer);
      transferOut |= pipe == afir::Pipe::TransferOut;
      if (!llvm::is_contained(pipes, pipe)) {
        pipes.push_back(pipe);
      }
      break;
    }
  });
  if (transferOut) {
    return failure();
  }
  if (pipes.empty()) {
    if (std::optional<afir::Pipe> container = containerPipe(root)) {
      pipes.push_back(*container);
    }
  }
  // 按流水线枚举值排序，使属性中的顺序不受算子排列影响，便于稳定比较和 lit 检查。
  llvm::sort(pipes,
             [](afir::Pipe lhs, afir::Pipe rhs) { return static_cast<uint32_t>(lhs) < static_cast<uint32_t>(rhs); });
  return pipes;
}

SmallVector<ascendc::HardEvent> recordedScalarReadEvents(Operation *anchor, unsigned inputIndex) {
  auto recorded = anchor->getAttrOfType<ArrayAttr>(afir::kScalarReadSyncAttrName);
  if (!recorded || inputIndex >= recorded.size()) {
    return {};
  }
  auto entry = dyn_cast<ArrayAttr>(recorded[inputIndex]);
  if (!entry) {
    return {};
  }
  SmallVector<afir::Pipe> pipes;
  for (Attribute pipe : entry) {
    auto attribute = dyn_cast<afir::PipeAttr>(pipe);
    if (!attribute) {
      return {};
    }
    pipes.push_back(attribute.getValue());
  }
  return toScalarReadEvents(pipes);
}

SmallVector<ascendc::HardEvent> scalarReadEvents(Operation *anchor, unsigned inputIndex) {
  if (anchor->hasAttr(afir::kScalarReadSyncAttrName)) {
    return recordedScalarReadEvents(anchor, inputIndex);
  }
  FailureOr<SmallVector<afir::Pipe>> pipes = analyzeScalarWriterPipes(anchor, inputIndex);
  if (failed(pipes)) {
    return {};
  }
  return toScalarReadEvents(*pipes);
}

FailureOr<Value> materializeScalarRead(Operation *anchor, Value source, ArrayRef<ascendc::HardEvent> events,
                                       RewriterBase &rewriter) {
  auto type = dyn_cast<ascendc::LocalTensorType>(source.getType());
  Value pipe = findFunctionPipe(anchor);
  if (!type || events.empty() || !pipe) {
    return failure();
  }
  // 每条流水线依次生成 FetchEventID、SetFlag、WaitFlag，不交叠事件的等待区间。
  // FetchEventID 不占用事件槽；同类型事件若交叠，可能复用同一事件号，导致死锁或同步失效。
  for (ascendc::HardEvent event : events) {
    Value id = rewriter.create<ascendc::TPipeFetchEventIDOp>(anchor->getLoc(), rewriter.getIntegerType(8), pipe, event);
    rewriter.create<ascendc::SetFlagOp>(anchor->getLoc(), event, id);
    rewriter.create<ascendc::WaitFlagOp>(anchor->getLoc(), event, id);
  }
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
