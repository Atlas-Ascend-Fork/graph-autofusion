/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRPipelineSync.cpp - Insert AFIR pipeline synchronization --------===//

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include "conversion/AFIRToAscendC/AFIRPipelineSync.h"
#include "conversion/AFIRToAscendC/AFIRToAscendCLoweringStage.h"
#include "dialect/AFIR/Transforms/AFIRLoweringStage.h"

#include <cstdint>
#include <limits>
#include <utility>

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/Analysis/BufferAlias.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir::ascend::kernel;

using namespace mlir;
using namespace mlir::afir;

namespace {

enum class Pipeline : uint8_t {
  MTE2,
  MTE3,
  Vector,
};

// 队列消费者所在 pipe 的聚合位（释放屏障按读者 pipe 各插一条）。
constexpr unsigned kConsumerPipeMte3 = 1;
constexpr unsigned kConsumerPipeVector = 2;

struct BufferOrigin {
  Value queue;
  Value tensor;
  Value rawTensor;
  Value offset;
};

bool isAscendCBufferType(Type type) {
  return isa<ascendc::GlobalTensorType, ascendc::LocalTensorType>(type);
}

bool isBufferizedOperation(Operation *operation) {
  if (operation == nullptr || operation->getDialect() == nullptr ||
      operation->getDialect()->getNamespace() != afir::AFIRDialect::getDialectNamespace()) {
    return false;
  }
  if (!isa<afir::AFIRDestinationStyleInterface>(operation) || !operation->hasAttr(afir::kDestinationDescsAttrName)) {
    return false;
  }
  FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(operation);
  if (failed(partition) || partition->destinationCount == 0) {
    return false;
  }
  return llvm::all_of(partition->destinations(operation),
                      [](Value operand) { return isAscendCBufferType(operand.getType()); });
}

Pipeline getPipeline(Operation *operation) {
  // The buffer form currently carries no independent pipeline attribute. Transfers
  // retain their MTE semantics; the remaining supported AFIR operations use the vector pipeline.
  //
  // 判据是**方向 trait**（`AFIR_TransferIn` / `AFIR_TransferOut`），不是 op 名单。那两个
  // trait 存在的全部理由就是在 ODS 上标出搬运方向，而「按名单列」会在加第三个搬运 op 时
  // 静默把它当成 vector 算子：`afir.nddma` 落到 Vector 上的后果是 MTE2 与 vector 之间
  // 少一对 event，读到的是上一趟还没搬完的那块 UB。
  if (operation->hasTrait<OpTrait::AFIRTransferIn>()) {
    return Pipeline::MTE2;
  }
  if (operation->hasTrait<OpTrait::AFIRTransferOut>()) {
    return Pipeline::MTE3;
  }
  return Pipeline::Vector;
}

StringRef getPipelineName(Pipeline pipeline) {
  switch (pipeline) {
    case Pipeline::MTE2:
      return "MTE2";
    case Pipeline::MTE3:
      return "MTE3";
    case Pipeline::Vector:
      return "Vector";
  }
  return "Unknown";
}

afir::Pipe toAfirPipe(Pipeline pipeline) {
  switch (pipeline) {
    case Pipeline::MTE2:
      return afir::Pipe::TransferIn;
    case Pipeline::MTE3:
      return afir::Pipe::TransferOut;
    case Pipeline::Vector:
      break;
  }
  return afir::Pipe::Vector;
}

class FunctionPipelineSynchronizer {
 public:
  explicit FunctionPipelineSynchronizer(func::FuncOp func) : func(func), context(func.getContext()) {}

  LogicalResult run() {
    // 两个阶段判据都问 IR 自身，不问「哪个 pass 跑过」的标记属性（见 LoweringStage.h）。
    if (hasPipelineSynchronization(func)) {
      return success();
    }
    // 前置问的是「资源物化是否还欠这个函数的债」，而不是「有没有 ascendc 算子」。
    // 两者在一个没有任何片上资源的函数上会分道扬镳：那种函数物化跑过、但什么都没产出
    // （CanonicalResources 的 `resources.empty()` 分支），按后者会被误判成「没跑过」，
    // 而它既没有待物化的落点、也就没有可同步的算子。边界的完整说明见 LoweringStage.h。
    if (afir::hasPendingResourceMaterialization(func)) {
      return func.emitError(
          "AFIR pipeline synchronization requires materialized resources: the function still has AFIR "
          "operations writing into memref destinations");
    }
    if (failed(collectOperations()) || failed(collectTensorDependencies())) {
      return failure();
    }
    if (bufferizedOperations.empty()) {
      return success();
    }
    // 此前这里写入 `afir.pipeline_synchronized`。同步的产物（enque / deque /
    // pipe_barrier）就是这件事的结构证据，不必再存一份；没有可同步算子时本来就什么都
    // 没插入，重跑一次仍然什么都不插入，幂等性不依赖那个标记。
    return insertPipelineSynchronization();
  }

 private:
  LogicalResult collectOperations() {
    func.walk([&](Operation *operation) {
      if (!isBufferizedOperation(operation)) {
        return;
      }
      operationOrder[operation] = bufferizedOperations.size();
      bufferizedOperations.push_back(operation);
    });
    return success();
  }

  BufferOrigin findAllocationBuffer(Value value) const {
    // Trace through the AscendC view operations to the queue allocation.
    Value tensor = value;
    Value offset;
    llvm::DenseSet<Value> visited;
    while (value) {
      if (!visited.insert(value).second) {
        return {};
      }
      Operation *definingOp = value.getDefiningOp();
      if (definingOp == nullptr) {
        return {};
      }
      if (auto reinterpretCastOp = dyn_cast<ascendc::LocalTensorReinterpretCastOp>(definingOp)) {
        value = reinterpretCastOp.getIn();
        continue;
      }
      if (auto subIndexOp = dyn_cast<ascendc::LocalTensorSubIndexOp>(definingOp)) {
        if (!offset) {
          offset = subIndexOp.getIndex();
        }
        value = subIndexOp.getTensor();
        continue;
      }
      if (auto allocTensorOp = dyn_cast<ascendc::TQueBindAllocTensorOp>(definingOp)) {
        return {allocTensorOp.getQueue(), tensor, allocTensorOp.getResult(), offset};
      }
      return {};
    }
    return {};
  }

  BufferOrigin findBufferOrigin(Value value) const {
    // A bufferized result is a new SSA value, not an AscendC view of its output buffer.
    auto origin = bufferizedResultOrigins.find(value);
    if (origin != bufferizedResultOrigins.end()) {
      return origin->second;
    }
    return findAllocationBuffer(value);
  }

  // 记下一个片上 destination 的生产者，并核对它的每个消费者都在同一个块内、且拓扑序在后。
  // GM 的 destination 不参与流水同步，直接跳过（返回 success 但不记录）。
  LogicalResult collectDestinationDependency(Operation *operation, afir::AFIRDestinationStyleInterface destinationStyle,
                                             unsigned outputOperandStart, unsigned index) {
    Value destination = operation->getOperand(outputOperandStart + index);
    // 空间从 destination 值自己的定义点取：物化之后它挂在新的 AscendC 定义 Op 上
    // （`CanonicalMaterialization` 一并带过来），一个值仍然只有一份。
    afir::BufferDescAttr buffer = afir::getBufferDescAttr(destination);
    if (!buffer) {
      return operation->emitError("canonical destination descriptor has invalid type");
    }
    if (buffer.getSpace().getValue() == afir::Space::GM) {
      return success();
    }
    FailureOr<unsigned> tied = destinationStyle.getTiedResultIndex(index);
    if (failed(tied) || *tied >= operation->getNumResults()) {
      return operation->emitError("canonical local destination has no tied result");
    }
    Value bufferizedResult = operation->getResult(*tied);
    BufferOrigin origin = findAllocationBuffer(destination);
    destinationProducers[destination] = operation;
    for (Operation *user : bufferizedResult.getUsers()) {
      if (!isBufferizedOperation(user)) {
        continue;
      }
      auto producerOrder = operationOrder.find(operation);
      auto consumerOrder = operationOrder.find(user);
      if (producerOrder == operationOrder.end() || consumerOrder == operationOrder.end() ||
          producerOrder->second >= consumerOrder->second) {
        return operation->emitError("bufferized tensor consumer is not in topological order");
      }
      if (user->getBlock() != operation->getBlock()) {
        return operation->emitError("pipeline synchronization across blocks is not supported");
      }
    }
    bufferizedResultOrigins[bufferizedResult] = origin;
    return success();
  }

  // 每个 (queue, tensor) 上最后一个消费者——释放点要插在它之后。同一 (queue, tensor)
  // 的消费者可能分属不同引擎（例如同一个 load 的结果既被矢量算子读、又被 store 写出），
  // 释放屏障要覆盖全部读者所在的 pipe，按位聚合在 `queueConsumerPipes` 里。
  LogicalResult collectLastQueueConsumers() {
    for (Operation *operation : bufferizedOperations) {
      unsigned inputOperandCount = getOutputOperandStart(operation);
      for (OpOperand &operand : operation->getOpOperands().take_front(inputOperandCount)) {
        BufferOrigin origin = findBufferOrigin(operand.get());
        if (origin.queue && origin.rawTensor) {
          // 释放屏障只覆盖 MTE3 / V 两种读者（见 insertReleaseBarriers）。今天读队列
          // 成员 tile 的只有 store 和矢量算子；MTE2 op（load / nddma）只写队列、不读，
          // 源端永远在 GM。所以 MTE2 消费者目前到不了这里。但若将来出现 UB→UB 搬运
          // op，它会作为 MTE2 读者消费队列 tile，而它没有可用的释放屏障——静默按 0
          // 聚合等于漏插。故在此直接报错，不发没有依据的同步。
          if (getPipeline(operation) == Pipeline::MTE2) {
            return operation->emitError("MTE2 queue consumer has no release barrier to order against");
          }
          auto key = std::make_pair(origin.queue, origin.rawTensor);
          lastQueueConsumers[key] = operation;
          queueConsumerPipes[key] |= consumerPipeBit(getPipeline(operation));
        }
      }
    }
    return success();
  }

  static unsigned consumerPipeBit(Pipeline pipeline) {
    switch (pipeline) {
      case Pipeline::MTE3:
        return kConsumerPipeMte3;
      case Pipeline::Vector:
        return kConsumerPipeVector;
      case Pipeline::MTE2:
        break;
    }
    return 0;
  }

  LogicalResult collectTensorDependencies() {
    for (Operation *operation : bufferizedOperations) {
      if (!operation->hasAttr(afir::kDestinationDescsAttrName)) {
        continue;
      }
      auto destinationStyle = dyn_cast<afir::AFIRDestinationStyleInterface>(operation);
      auto descriptors = operation->getAttrOfType<ArrayAttr>(afir::kDestinationDescsAttrName);
      if (!destinationStyle || !descriptors || descriptors.size() != destinationStyle.getNumDestinations()) {
        return operation->emitError("canonical bufferized operation has invalid destination descriptors");
      }
      FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(operation);
      if (failed(partition)) {
        return operation->emitError("canonical bufferized operation has invalid destination descriptors");
      }
      unsigned outputOperandStart = partition->inputCount;
      for (unsigned index = 0; index < destinationStyle.getNumDestinations(); ++index) {
        if (failed(collectDestinationDependency(operation, destinationStyle, outputOperandStart, index))) {
          return failure();
        }
      }
    }
    if (failed(collectLastQueueConsumers())) {
      return failure();
    }
    collectProducersWithNonStoreConsumers();
    return success();
  }

  // 「load 除了被 store 直连写出，是否还有别的消费者（比如矢量算子）」这个判据必须在
  // 任何改写之前定案：队列同步会替换消费者操作数，等扫描走到直连 store 时，矢量消费者
  // 已经从 destination 视图的 user 列表里消失了——中途再问 IR 就会误判。这里对原始 IR
  // 逐个生产者算一遍并缓存结论。
  void collectProducersWithNonStoreConsumers() {
    for (Operation *operation : bufferizedOperations) {
      if (hasNonStoreConsumer(operation)) {
        producersWithNonStoreConsumers.insert(operation);
      }
    }
  }

  void insertEnqueue(OpBuilder &builder, Value queue, Value tensor) const {
    ascendc::TQueBindEnqueTensorOp::create(builder, builder.getUnknownLoc(), queue, tensor);
  }

  Value insertDequeue(OpBuilder &builder, Value queue, Type tensorType) const {
    return ascendc::TQueBindDequeTensorOp::create(builder, builder.getUnknownLoc(), tensorType, queue).getTensor();
  }

  // 同步视图与它替换掉的那个值是同一块存储，规划结果跟着新的定义点走；否则
  // lowering 阶段在这些视图上就取不到 descriptor（设计文档 5.1 步 8）。
  Value materializeSyncedTensorView(OpBuilder &builder, const BufferOrigin &origin, Value dequeuedRaw) const {
    afir::BufferDescAttr descriptor = afir::getBufferDescAttr(origin.tensor);
    Value view = dequeuedRaw;
    if (origin.offset) {
      view = builder
                 .create<ascendc::LocalTensorSubIndexOp>(builder.getUnknownLoc(),
                                                         cast<ascendc::LocalTensorType>(origin.rawTensor.getType()),
                                                         dequeuedRaw, origin.offset)
                 .getResult();
      afir::setBufferDescAttr(view.getDefiningOp(), descriptor);
    }
    if (view.getType() == origin.tensor.getType()) {
      return view;
    }
    Value cast = builder
                     .create<ascendc::LocalTensorReinterpretCastOp>(
                         builder.getUnknownLoc(), mlir::cast<ascendc::LocalTensorType>(origin.tensor.getType()), view)
                     .getResult();
    afir::setBufferDescAttr(cast.getDefiningOp(), descriptor);
    return cast;
  }

  void materializeQueueFree(Operation *consumer, Value queue, Value rawTensor, Value dequeuedRaw,
                            unsigned consumerPipes) const {
    SmallVector<ascendc::TQueBindFreeTensorOp> frees;
    for (Operation *user : rawTensor.getUsers()) {
      auto free = dyn_cast<ascendc::TQueBindFreeTensorOp>(user);
      if (free && free.getQueue() == queue) {
        frees.push_back(free);
      }
    }
    for (ascendc::TQueBindFreeTensorOp free : frees) {
      if (free->getBlock() == consumer->getBlock()) {
        free->moveAfter(consumer);
        free->setOperand(1, dequeuedRaw);
        OpBuilder builder(free);
        insertReleaseBarriers(builder, queue, consumerPipes);
        return;
      }
      free.erase();
    }

    OpBuilder builder(consumer);
    builder.setInsertionPointAfter(consumer);
    insertReleaseBarriers(builder, queue, consumerPipes);
    ascendc::TQueBindFreeTensorOp::create(builder, builder.getUnknownLoc(), queue, dequeuedRaw);
  }

  // 释放屏障按读者 pipe 逐条插入：一类读者一条。此前只插末位消费者所在 pipe 的那一条，
  // 只有一个引擎的读者时行为不变；矢量读者与 store 读者并存时两条都插，缺一条就是漏掉
  // 一侧的顺序保证。顺序按 pipe 位序固定，产物确定。
  static void insertReleaseBarriers(OpBuilder &builder, Value queue, unsigned consumerPipes) {
    // VECIN ownership orders Vector readers before the next MTE2 write, but a
    // direct Store also reads this slot on MTE3. A same-pipe barrier cannot stop
    // MTE2 from overwriting it. Complete that reader before returning the slot.
    auto queueType = dyn_cast<ascendc::QueueType>(queue.getType());
    if (queueType && queueType.getPosition() == ascendc::TPosition::VECIN &&
        (consumerPipes & (kConsumerPipeMte3 | kConsumerPipeVector)) == (kConsumerPipeMte3 | kConsumerPipeVector)) {
      auto tokenType = afir::TokenType::get(builder.getContext(), afir::Pipe::TransferOut, afir::Pipe::TransferIn);
      Value token = afir::EventRecordOp::create(builder, builder.getUnknownLoc(), tokenType).getToken();
      afir::EventWaitOp::create(builder, builder.getUnknownLoc(), token);
    }
    if (consumerPipes & kConsumerPipeMte3) {
      ascendc::PipeBarrierOp::create(builder, builder.getUnknownLoc(), ascendc::Pipe::PIPE_MTE3);
    }
    if (consumerPipes & kConsumerPipeVector) {
      ascendc::PipeBarrierOp::create(builder, builder.getUnknownLoc(), ascendc::Pipe::PIPE_V);
    }
  }

  // Insert queue synchronization for MTE2 -> Vector, Vector -> MTE3 and the store-only
  // MTE2 -> MTE3 transitions (load written straight out by store, no vector consumer).
  // The enqueue/dequeue logic is direction agnostic; the direction only decides whether
  // this path is taken at all.
  LogicalResult insertQueueSync(Operation *consumer, Value operandValue, Pipeline producerPipeline,
                                llvm::DenseMap<std::pair<Value, Value>, Value> &syncedQueues) {
    // Resolve the allocation origin for either a bufferized result or a raw local tensor view.
    BufferOrigin origin = findBufferOrigin(operandValue);
    if (!origin.queue || !origin.tensor || !origin.rawTensor) {
      return success();  // No queue sync needed
    }

    // Match the legacy AscendC queue ownership API: shared queue members are
    // views into one raw uint8 queue tensor. Enqueue/dequeue the raw allocation
    // exactly once, then rebuild each typed offset view for the consumer.
    OpBuilder builder(consumer);
    auto queueKey = std::make_pair(origin.queue, origin.rawTensor);
    auto existing = syncedQueues.find(queueKey);
    Value dequeuedRaw;
    if (existing == syncedQueues.end()) {
      if (producerPipeline == Pipeline::MTE2) {
        ascendc::PipeBarrierOp::create(builder, builder.getUnknownLoc(), ascendc::Pipe::PIPE_MTE2);
      }
      insertEnqueue(builder, origin.queue, origin.rawTensor);
      dequeuedRaw = insertDequeue(builder, origin.queue, origin.rawTensor.getType());
      syncedQueues.try_emplace(queueKey, dequeuedRaw);
    } else {
      dequeuedRaw = existing->second;
    }
    Value syncedView = materializeSyncedTensorView(builder, origin, dequeuedRaw);
    operandValue.replaceUsesWithIf(syncedView, [&](OpOperand &use) { return use.getOwner() == consumer; });
    releaseQueueIfLastConsumer(consumer, origin, dequeuedRaw);

    return success();
  }

  // 末位消费者触发队列释放：释放屏障覆盖该 (queue, tensor) 的全部读者 pipe。调用方
  // 保证 `slotTensor` 是当前要归还给队列的那块张量值（队列同步路径传刚出队的值；
  // 显式同步路径传既有出队值或原始分配）。
  void releaseQueueIfLastConsumer(Operation *consumer, const BufferOrigin &origin, Value slotTensor) {
    auto queueKey = std::make_pair(origin.queue, origin.rawTensor);
    auto lastConsumer = lastQueueConsumers.find(queueKey);
    if (lastConsumer == lastQueueConsumers.end() || lastConsumer->second != consumer ||
        !freedQueues.insert(queueKey).second) {
      return;
    }
    auto pipes = queueConsumerPipes.find(queueKey);
    materializeQueueFree(consumer, origin.queue, origin.rawTensor, slotTensor,
                         pipes == queueConsumerPipes.end() ? 0 : pipes->second);
  }

  // Insert barrier synchronization for Vector -> Vector transitions
  LogicalResult insertVectorBarrierSync(Operation *consumer, llvm::DenseSet<Operation *> &barrierInserted) const {
    // Only insert once per operation
    if (barrierInserted.contains(consumer)) {
      return success();
    }

    // 产出的是 **AFIR** 的 pipe_barrier，不是 AscendC 的。V -> V 这条路径上没有队列，
    // 顺序此前直接写成后端算子，AFIR 层看不见也就无从校验（3.2：放弃 TQue 自动同步后
    // 正确性归 AFIR 自己）。现在它先是一条 AFIR 的顺序事实，再由同步族的 lowering 落到
    // `ascendc.pipe_barrier`——文本产物不变，但竞争检查器从此看得见这条边。
    //
    // 队列路径（enque / deque 与 materializeQueueFree 的释放屏障）仍产 AscendC 算子：
    // TQue 的入队/出队自带隐式同步，叠一层显式 token 就是双重同步，那一段要和退出 TQue
    // （ADR-11/12 的 pool + memref.view）一起改。
    OpBuilder builder(consumer);
    afir::PipeBarrierOp::create(builder, builder.getUnknownLoc(), afir::Pipe::Vector);
    barrierInserted.insert(consumer);

    return success();
  }

  // Get the starting index of destination operands in a canonical operation,
  // which is also the number of input operands. Trailing tmp buffer operands are
  // neither: they are read from and written to by the callee alone, so they
  // carry no dependency and stay out of both halves.
  unsigned getOutputOperandStart(Operation *op) const {
    if (!op->hasAttr(afir::kDestinationDescsAttrName)) {
      return op->getNumOperands();
    }
    FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(op);
    return failed(partition) ? op->getNumOperands() : partition->inputCount;
  }

  // 一次流水线交接：谁产、谁消费、经由哪个值，以及两端各在哪条流水上。
  struct PipelineTransition {
    Operation *producer;
    Operation *consumer;
    Value operandValue;
    Pipeline producerPipeline;
    Pipeline consumerPipeline;
  };

  // 整趟扫描里「已经插过哪些同步」的账本，防止同一处交接被重复插同步：
  // - syncedQueues / barrierInserted 是既有两类（队列同步按 (queue, tensor) 记出队值，
  //   屏障按消费者记）；
  // - dualEndQueues / explicitPairs / explicitMte3PairedProducers 是 MTE2→MTE3 新增
  //   （待物化为双端 TQueBind 的队列、已建过的显式对 (record 位置, wait 位置)、已建过
  //   显式对的 load）。
  // dualEndQueues 的类型替换推迟到扫描结束统一做，原因见 materializeDualEndQueues。
  struct SyncState {
    llvm::DenseMap<std::pair<Value, Value>, Value> syncedQueues;
    llvm::DenseSet<Operation *> barrierInserted;
    llvm::DenseSet<Value> dualEndQueues;
    llvm::DenseSet<std::pair<Operation *, Operation *>> explicitPairs;
    // 已为某个 load 建过 transfer_in→transfer_out 显式对（load 另有矢量读者时用）。
    // 一个 load 的直连 store 全在 MTE3，wait 挡住首个 store 即按程序序覆盖后续 store，
    // 所以每个 load 只与首个 store 建对——需要的并发事件数恒为 1（MTE2_MTE3 可用的并发
    // 事件上限是 QUE_MAX_EVENT，逐 store 建对会超限）。
    llvm::DenseSet<Operation *> explicitMte3PairedProducers;
  };

  // load→store 直连同步的分派判据：load 的消费者里有没有非 store 的算子。
  // - 全是搬运出（store，MTE3）：走双端队列（enque/deque）；
  // - 另有矢量消费者：队列保持单端 VECIN 给矢量侧，store 侧改走显式同步对。
  // 只看 load 的 destination 视图与 bufferized result 的 SSA 可达消费者。
  // 该判据必须在任何改写之前定案：队列同步会替换消费者操作数，扫到后面矢量消费者就从
  // user 列表里消失了（结论由 collectProducersWithNonStoreConsumers 在收集阶段缓存）。
  // 「store 侧」按方向 trait（`AFIR_TransferOut`）认，不按 op 名单：将来加第二个搬运出
  // op 时按名单列会把它误当矢量消费者，无谓地放弃双端队列路径。
  bool hasNonStoreConsumer(Operation *producer) const {
    auto hasVectorUser = [&](Value value) {
      for (Operation *user : value.getUsers()) {
        if (user == producer || !isBufferizedOperation(user)) {
          continue;
        }
        if (!user->hasTrait<OpTrait::AFIRTransferOut>()) {
          return true;
        }
      }
      return false;
    };
    FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(producer);
    if (succeeded(partition)) {
      for (Value destination : partition->destinations(producer)) {
        if (hasVectorUser(destination)) {
          return true;
        }
      }
    }
    return llvm::any_of(producer->getResults(), hasVectorUser);
  }

  // 显式同步对：producer 之后插 `event_record`、consumer 之前插 `event_wait`，两者由 token
  // 串联，方向写在 token 类型上；同一 (producer, consumer) 只建一对。
  // 要求 producer 与 consumer 同块：这一对是按「共用同一块 buffer」找出来的，两者不一定有
  // 先后依赖，跨 block 时不保证 record 先于 wait 执行，wait 可能等到一个在该分支里没生成的
  // token（非法 IR）。
  LogicalResult insertExplicitPairSync(Operation *producer, Operation *consumer, afir::Pipe from, afir::Pipe to,
                                       SyncState &state) {
    if (producer->getBlock() != consumer->getBlock()) {
      return producer->emitError("pipeline synchronization across blocks is not supported");
    }
    if (!state.explicitPairs.insert(std::make_pair(producer, consumer)).second) {
      return success();
    }
    auto tokenType = afir::TokenType::get(context, from, to);
    OpBuilder recordBuilder(producer);
    recordBuilder.setInsertionPointAfter(producer);
    Value token = afir::EventRecordOp::create(recordBuilder, producer->getLoc(), tokenType).getToken();
    OpBuilder waitBuilder(consumer);
    afir::EventWaitOp::create(waitBuilder, consumer->getLoc(), token);
    return success();
  }

  // MTE2 → MTE3（load 直连 store）的分派。
  LogicalResult synchronizeTransferInToTransferOut(const PipelineTransition &transition, SyncState &state) {
    Operation *producer = transition.producer;
    Operation *consumer = transition.consumer;
    BufferOrigin origin = findBufferOrigin(transition.operandValue);
    if (producersWithNonStoreConsumers.contains(producer)) {
      // load 另有矢量消费者：队列归矢量侧的 MTE2→Vector 流程所有；store 侧只补一条
      // transfer_in→transfer_out 的显式同步边，读原始视图，不经队列出队。多个直连 store 消费
      // 同一 load 时，只与**首个** store 建对：store 全在 MTE3，wait 挡在首 store 前即按程序序
      // 覆盖后续 store，需要的并发事件数收敛为 1（MTE2_MTE3 可用的并发事件上限 QUE_MAX_EVENT）。
      if (state.explicitMte3PairedProducers.insert(producer).second) {
        if (failed(
                insertExplicitPairSync(producer, consumer, afir::Pipe::TransferIn, afir::Pipe::TransferOut, state))) {
          return failure();
        }
      }
      if (origin.queue && origin.rawTensor) {
        // load 另有矢量读者时，store 直接读原始缓冲、不走队列出队；这里释放队列时，要归还
        // 的队列张量值优先借用矢量侧的出队结果，没有就退回 alloc 出的原始值。只有本 store
        // 是末位消费者才真正释放。
        auto queueKey = std::make_pair(origin.queue, origin.rawTensor);
        auto existing = state.syncedQueues.find(queueKey);
        releaseQueueIfLastConsumer(consumer, origin,
                                   existing == state.syncedQueues.end() ? origin.rawTensor : existing->second);
      }
      return success();
    }
    // load 只被 store 消费：走双端队列。入队/出队/释放复用既有队列路径（方向无关）；队列类型改为
    // TQueBind<VECIN, VECOUT> 的动作推迟到扫描结束统一做。
    //
    // 顺序只能由队列的 enque / deque 给：`PipeBarrier` 只在同一条流水内排序，MTE2 与 MTE3 之间
    // 没有 barrier 可用。所以拿不到队列就报错，不静默放过（放过去就是 Store 可能读到还没搬完的 UB）。
    if (!origin.queue || !origin.rawTensor) {
      return producer->emitError(
          "MTE2 to MTE3 synchronization requires a queue-backed buffer; TBuf has no barrier that orders the two pipes");
    }
    if (failed(insertQueueSync(consumer, transition.operandValue, transition.producerPipeline, state.syncedQueues))) {
      return failure();
    }
    state.dualEndQueues.insert(origin.queue);
    return success();
  }

  LogicalResult synchronizeTransition(const PipelineTransition &transition, SyncState &state) {
    Pipeline producerPipeline = transition.producerPipeline;
    Pipeline consumerPipeline = transition.consumerPipeline;
    Operation *consumer = transition.consumer;
    Value operandValue = transition.operandValue;
    if (producerPipeline == Pipeline::MTE2 && consumerPipeline == Pipeline::Vector) {
      return insertQueueSync(consumer, operandValue, producerPipeline, state.syncedQueues);
    }
    if (producerPipeline == Pipeline::Vector && consumerPipeline == Pipeline::MTE3) {
      BufferOrigin origin = findBufferOrigin(operandValue);
      if (origin.queue && origin.tensor) {
        return insertQueueSync(consumer, operandValue, producerPipeline, state.syncedQueues);
      }
      return insertVectorBarrierSync(consumer, state.barrierInserted);
    }
    if (producerPipeline == Pipeline::Vector && consumerPipeline == Pipeline::Vector) {
      return insertVectorBarrierSync(consumer, state.barrierInserted);
    }
    if (producerPipeline == Pipeline::MTE2 && consumerPipeline == Pipeline::MTE3) {
      return synchronizeTransferInToTransferOut(transition, state);
    }
    return transition.producer->emitError("unsupported pipeline synchronization from ")
           << getPipelineName(producerPipeline) << " to " << getPipelineName(consumerPipeline);
  }

  LogicalResult insertPipelineSynchronization() {
    SyncState state;

    for (Operation *op : bufferizedOperations) {
      Pipeline consumerPipeline = getPipeline(op);
      unsigned outputOperandStart = getOutputOperandStart(op);
      for (OpOperand &operand : op->getOpOperands()) {
        unsigned operandIndex = operand.getOperandNumber();
        if (operandIndex >= outputOperandStart) {
          continue;
        }
        Value operandValue = operand.get();
        Operation *producer = operandValue.getDefiningOp();
        if (!producer || !isBufferizedOperation(producer)) {
          auto destinationProducer = destinationProducers.find(operandValue);
          producer = destinationProducer == destinationProducers.end() ? nullptr : destinationProducer->second;
        }
        if (!producer || !isBufferizedOperation(producer)) {
          continue;
        }
        Pipeline producerPipeline = getPipeline(producer);
        if (failed(synchronizeTransition({producer, op, operandValue, producerPipeline, consumerPipeline}, state))) {
          return failure();
        }
      }
    }
    if (failed(synchronizeWorkspaceReadback(state))) {
      return failure();
    }
    if (failed(synchronizeBufferReuse(state))) {
      return failure();
    }
    return materializeDualEndQueues(state.dualEndQueues);
  }

  // GM 视图 → memref。物化后的 GM 操作数是 `ascendc.global_tensor`，其 memref 事实在
  // `set_global_buffer` 上（一个视图一份）。非 GM 视图返回空。
  Value resolveGlobalTensorBuffer(Value value) const {
    if (!isa<ascendc::GlobalTensorType>(value.getType())) {
      return {};
    }
    for (Operation *user : value.getUsers()) {
      if (auto setBuffer = dyn_cast<ascendc::GlobalTensorSetGlobalBufferOp>(user)) {
        return setBuffer.getBuffer();
      }
    }
    return {};
  }

  // store 把数据写进 GM workspace、之后搬运入 op（load / nddma）再从同一块读回
  // （方向 MTE3 → MTE2）。做法：搬运入的 GM 输入命中程序序在前的搬运出写集时，在写者
  // 后 record、读者前 wait 一条 transfer_out→transfer_in 的显式同步对。读写按根 memref
  // 归并（SSA 同值，或经 resolveBufferSlice 追到同一根，别名即命中）。
  // 写者与读者都按方向 trait（`AFIR_TransferOut` / `AFIR_TransferIn`）认，不按 op 名单：
  // 名单漏掉 nddma 这类搬运入 op 时它读的是上一趟还没写完的 workspace，静默错数据。
  // 只与末位写者建对：GM 写者全是搬运出 op（MTE3），同 pipe 程序序保证先写的先完成，
  // 末写者的 record 即覆盖全部前序 store——需要的并发事件数从「写者个数」收敛到 1
  // （MTE3_MTE2 可用的并发事件上限是 QUE_MAX_EVENT，逐写者建对会在多 stage 各写子区再
  // 回读的形态下超限，AllocEventID 返回越界事件号）。
  // 每个读者一对：token 只能被等待一次，两个读者共用一个 record 会被 verifier 拒掉。
  LogicalResult synchronizeWorkspaceReadback(SyncState &state) {
    llvm::DenseMap<Value, Operation *> workspaceLastWriter;
    for (Operation *operation : bufferizedOperations) {
      FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(operation);
      if (failed(partition)) {
        continue;
      }
      if (operation->hasTrait<OpTrait::AFIRTransferOut>()) {
        for (Value destination : partition->destinations(operation)) {
          Value buffer = resolveGlobalTensorBuffer(destination);
          if (!buffer) {
            continue;
          }
          afir::BufferSlice slice = afir::resolveBufferSlice(buffer);
          if (slice.isValid()) {
            // 程序序推进，后写的覆盖先写的——load 只需要知道此刻的最后一位。
            workspaceLastWriter[slice.base] = operation;
          }
        }
        continue;
      }
      if (!operation->hasTrait<OpTrait::AFIRTransferIn>()) {
        continue;
      }
      for (Value input : partition->inputs(operation)) {
        Value buffer = resolveGlobalTensorBuffer(input);
        if (!buffer) {
          continue;
        }
        afir::BufferSlice slice = afir::resolveBufferSlice(buffer);
        if (!slice.isValid()) {
          continue;
        }
        auto known = workspaceLastWriter.find(slice.base);
        if (known == workspaceLastWriter.end()) {
          continue;
        }
        if (failed(insertExplicitPairSync(known->second, operation, afir::Pipe::TransferOut, afir::Pipe::TransferIn,
                                          state))) {
          return failure();
        }
      }
    }
    return success();
  }

  // UB buffer 复用（先读后写，WAR）——某块（非队列）UB 存储被读之后，又被复用写入。
  // 做法：在「每个 pipe 的末位读者」与「程序序在后的复用写者」之间插一条
  // 读者 pipe → 写者 pipe 的显式同步对。
  // 只对每个 pipe 的末读者建边：同 pipe 内程序序保证先执行的读者先完成，末读者挡住
  // 写者即覆盖该 pipe 的全部前序读者；同 pipe 的写者也由同一公理覆盖，不再建边。
  // 范围：只看无队列的 TBuf（队列成员的释放归 materializeQueueFree 的释放屏障）；
  // 存储身份按 TBuf 归并（resolveTBufStorage），不依赖「物化给每个成员恰好一份视图」
  // 的偶然事实；跨迭代多缓冲轮转的 WAR 不在本期范围。
  LogicalResult synchronizeBufferReuse(SyncState &state) {
    // key: (存储, 读者 pipe) → 该 pipe 上程序序最后一个读者。
    llvm::DenseMap<std::pair<Value, Pipeline>, Operation *> lastReaderPerPipe;
    for (Operation *operation : bufferizedOperations) {
      FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(operation);
      if (failed(partition)) {
        continue;
      }
      Pipeline writerPipeline = getPipeline(operation);
      for (Value destination : partition->destinations(operation)) {
        if (!isNonQueueLocalBuffer(destination)) {
          continue;
        }
        Value storage = resolveTBufStorage(destination);
        for (Pipeline pipe : {Pipeline::MTE2, Pipeline::Vector, Pipeline::MTE3}) {
          if (pipe == writerPipeline) {
            continue;
          }
          auto known = lastReaderPerPipe.find(std::make_pair(storage, pipe));
          if (known == lastReaderPerPipe.end()) {
            continue;
          }
          if (failed(insertExplicitPairSync(known->second, operation, toAfirPipe(pipe), toAfirPipe(writerPipeline),
                                            state))) {
            return failure();
          }
        }
      }
      for (Value input : partition->inputs(operation)) {
        if (isNonQueueLocalBuffer(input)) {
          lastReaderPerPipe[{resolveTBufStorage(input), writerPipeline}] = operation;
        }
      }
    }
    return success();
  }

  bool isNonQueueLocalBuffer(Value value) const {
    if (!isa<ascendc::LocalTensorType>(value.getType())) {
      return false;
    }
    return !findBufferOrigin(value).queue;
  }

  // 局部张量视图 → UB 复用同步（synchronizeBufferReuse）的存储归并键，粒度是成员字节视图（SubIndex）。
  // 物化的规范链是 ReinterpretCast(SubIndex(TBufGetTensor(tbuf), offset))，本函数追过
  // ReinterpretCast 后停在 SubIndex 值上。归并效果：
  // - 同一成员（同一 SubIndex 值）命中，WAR 边正确建立；
  // - 同 TBuf 不同 offset 的不相交成员不误并（本就无需 WAR 边）；
  // - offset 相同的复用组（同 buffer_id 不同 reuseId）落在同一视图上，同样命中。
  // 无 SubIndex 的直连整块视图（raw 视图不经 SubIndex 直达 op 操作数的形态）走
  // TBufGetTensorOp / TBufGetWithOffsetOp 分支返回 buffer；追不到已知视图 Op 的按值本身归并。
  Value resolveTBufStorage(Value value) const {
    while (Operation *definingOp = value.getDefiningOp()) {
      if (auto get = dyn_cast<ascendc::TBufGetTensorOp>(definingOp)) {
        return get.getBuffer();
      }
      if (auto getWithOffset = dyn_cast<ascendc::TBufGetWithOffsetOp>(definingOp)) {
        return getWithOffset.getBuffer();
      }
      if (auto cast = dyn_cast<ascendc::LocalTensorReinterpretCastOp>(definingOp)) {
        value = cast.getIn();
        continue;
      }
      return value;
    }
    return value;
  }

  // 双端队列物化：`ascendc.queue`（TQue）替换为 `ascendc.que_bind`（TQueBind<VECIN,
  // VECOUT>，深度模板与 legacy 对齐取 1）。放在扫描结束统一做，是因为替换会换掉队列
  // Value 的身份——lastQueueConsumers / syncedQueues 等映射都按旧值记键，边扫边换会让
  // 后续查找全部落空。成员操作（alloc / enque / deque / free）的队列操作数是
  // BaseQueueTypeInterface，对具体类型不敏感，RAUW 即完成迁移。
  LogicalResult materializeDualEndQueues(const llvm::DenseSet<Value> &queues) {
    for (Value queue : queues) {
      auto queueOp = dyn_cast_if_present<ascendc::QueueOp>(queue.getDefiningOp());
      if (!queueOp) {
        // queue 可能是函数入参（block argument），没有定义 op，不能当报错锚点；挂到
        // func 上（必然存在），顺带打印实际类型。
        return func->emitError("dual-end queue materialization requires an ascendc.queue operand, but found ")
               << queue.getType();
      }
      OpBuilder builder(queueOp);
      auto queBind = ascendc::QueBindOp::create(
          builder, queueOp.getLoc(),
          ascendc::QueBindType::get(context, ascendc::TPosition::VECIN, ascendc::TPosition::VECOUT, 1));
      queueOp.getResult().replaceAllUsesWith(queBind.getQueBind());
      queueOp.erase();
    }
    return success();
  }

  func::FuncOp func;
  MLIRContext *context;
  SmallVector<Operation *> bufferizedOperations;
  llvm::DenseMap<Operation *, size_t> operationOrder;
  llvm::DenseMap<Value, BufferOrigin> bufferizedResultOrigins;
  llvm::DenseMap<Value, Operation *> destinationProducers;
  llvm::DenseMap<std::pair<Value, Value>, Operation *> lastQueueConsumers;
  llvm::DenseMap<std::pair<Value, Value>, unsigned> queueConsumerPipes;
  llvm::DenseSet<std::pair<Value, Value>> freedQueues;
  llvm::DenseSet<Operation *> producersWithNonStoreConsumers;
};

struct AFIRPipelineSyncPass : public PassWrapper<AFIRPipelineSyncPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRPipelineSyncPass)

  StringRef getArgument() const final {
    return "afir-pipeline-sync";
  }

  StringRef getDescription() const final {
    return "Insert AFIR queue and pipeline synchronization after memory allocation";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<afir::AFIRDialect, ascendc::AscendCDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    if (failed(synchronizeAFIRPipeline(getOperation()))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mlir::ascend::kernel {

LogicalResult synchronizeAFIRPipeline(ModuleOp module) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!afir::hasFunctionRole(func, afir::FunctionRole::KernelBody)) {
      continue;
    }
    if (failed(FunctionPipelineSynchronizer(func).run())) {
      return failure();
    }
  }
  return success();
}

std::unique_ptr<Pass> createAFIRPipelineSyncPass() {
  return std::make_unique<AFIRPipelineSyncPass>();
}

void registerAFIRPipelineSyncPass() {
  PassRegistration<AFIRPipelineSyncPass>();
}

}  // namespace mlir::ascend::kernel
