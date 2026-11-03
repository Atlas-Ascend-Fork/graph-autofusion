/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- RaceCheck.cpp - Data race detection over AFIR buffer form ----------===//

#include "dialect/AFIR/Analysis/RaceCheck.h"

#include "dialect/AFIR/IR/AFIR.h"
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

using namespace mlir;

namespace mlir::afir {
namespace {

// 两个访问最近的共同循环。只有同在一个循环里才需要考虑跨迭代方向——循环之外的访问
// 各自只执行一次。
Operation *getCommonLoop(Operation *lhs, Operation *rhs) {
  llvm::SmallVector<Operation *> lhsLoops;
  for (Operation *parent = lhs->getParentOp(); parent; parent = parent->getParentOp()) {
    if (isa<LoopLikeOpInterface>(parent)) {
      lhsLoops.push_back(parent);
    }
  }
  for (Operation *parent = rhs->getParentOp(); parent; parent = parent->getParentOp()) {
    if (isa<LoopLikeOpInterface>(parent) && llvm::is_contained(lhsLoops, parent)) {
      return parent;  // 从内往外走，第一个共同祖先就是最近的那个。
    }
  }
  return nullptr;
}

bool isBufferValue(Value value) {
  return isa<MemRefType>(value.getType());
}

}  // namespace

Pipe getPipe(Operation *op) {
  if (op == nullptr) {
    return Pipe::Vector;
  }
  // 同步原语不跑在 trait 决定的 pipe 上：record 记在 `from` 一侧，wait 停在 `to` 一侧。
  // 这是它们把两条 pipe 的链接起来的方式。
  if (auto record = dyn_cast<EventRecordOp>(op)) {
    return record.getToken().getType().getFrom();
  }
  if (auto wait = dyn_cast<EventWaitOp>(op)) {
    return wait.getToken().getType().getTo();
  }
  if (op->hasTrait<OpTrait::AFIRTransferIn>()) {
    return Pipe::TransferIn;
  }
  if (op->hasTrait<OpTrait::AFIRTransferOut>()) {
    return Pipe::TransferOut;
  }
  // 兜底是矢量，不是「某台计算引擎」：今天有注册实现的计算 Op 全在矢量上。Cube 与标量
  // 各是一台独立引擎（`afir::Pipe` 里各有取值），它们的 Op 落地时要像搬运族那样带上
  // 自己的 trait——否则会被这里当成矢量，而竞争检查器给出的就是一份漏了 Cube 那几条边
  // 的清单。
  return Pipe::Vector;
}

llvm::StringRef getHazardKindName(HazardKind kind) {
  switch (kind) {
    case HazardKind::ReadAfterWrite:
      return "RAW";
    case HazardKind::WriteAfterRead:
      return "WAR";
    case HazardKind::WriteAfterWrite:
      return "WAW";
  }
  return "unknown";
}

// 同流相邻边。只连相邻的一对，传递性交给 `happensBefore` 的闭包——连全部 O(n^2) 条边
// 除了变慢没有任何区别。全序点（barrier_all / sync_all）不属任何一条 pipe 流，跳过。
void RaceChecker::buildStreamOrder() {
  nextInStream.clear();
  llvm::DenseMap<Pipe, Operation *> lastInStream;
  for (Operation *op : orderedOperations) {
    if (isa<BarrierAllOp, SyncAllOp>(op)) {
      continue;
    }
    Pipe stream = getPipe(op);
    auto last = lastInStream.find(stream);
    if (last != lastInStream.end()) {
      nextInStream[last->second] = op;
    }
    lastInStream[stream] = op;
  }
}

void RaceChecker::collectAccesses(func::FuncOp func) {
  accesses.clear();
  orderedOperations.clear();
  waits.clear();
  barrierOps.clear();
  operationOrderIndex.clear();
  reachableCache.clear();
  func.walk([&](Operation *op) {
    // 同步原语不访问存储，但它们必须进入这张图：record 接在 `from` pipe 的链上、wait
    // 接在 `to` pipe 的链上，两者之间由 token 的 use-def 相连，于是
    // 「producer → record → wait → consumer」这条路径在 `getReachableSet` 里自然连通，
    // 不需要为显式同步单开一套判定。全序点（barrier_all / sync_all）也进来占位——
    // 它们的效果在 `happensBefore` 里按程序序位置判定。
    if (isa<EventRecordOp>(op)) {
      orderedOperations.push_back(op);
      return;
    }
    if (auto wait = dyn_cast<EventWaitOp>(op)) {
      orderedOperations.push_back(op);
      waits.push_back(wait);
      return;
    }
    if (isa<BarrierAllOp, SyncAllOp>(op)) {
      orderedOperations.push_back(op);
      barrierOps.push_back(op);
      return;
    }
    // `afir.pipe_barrier` **故意**不进这张图，不是漏了。它是单 pipe 内的全序点，而本模型
    // 已经假定同 pipe 程序序即执行序——这正是 `!afir.token<from = X, to = X>` 被 verifier
    // 拒掉的同一条理由：那是一条什么都挡不住的空边。把 barrier 加进来只会得到一条已经
    // 成立的边。它在下降之后仍然必要（硬件在单 pipe 内会重叠执行），但那是 AscendC 层
    // 的事实，不是本层的 happens-before。
    if (!isa<AFIRDestinationStyleInterface>(op)) {
      return;
    }
    FailureOr<OperandPartition> partition = getOperandPartition(op);
    if (failed(partition)) {
      return;
    }
    orderedOperations.push_back(op);
    for (Value operand : partition->inputs(op)) {
      if (!isBufferValue(operand)) {
        continue;
      }
      accesses.push_back(Access{op, operand, /*isWrite=*/false, resolveBufferSlice(operand)});
    }
    // tmp buffer 跟 destination 一样按写记：被调 API 在里面读写自己的中间量，两个共享同一块
    // tmp buffer 的 Op 之间是真实的竞争。把它漏掉才是危险的那一侧——漏报没人会发现。
    for (Value written : op->getOperands().drop_front(partition->inputCount)) {
      if (!isBufferValue(written)) {
        continue;
      }
      accesses.push_back(Access{op, written, /*isWrite=*/true, resolveBufferSlice(written)});
    }
  });
  for (auto [index, op] : llvm::enumerate(orderedOperations)) {
    operationOrderIndex[op] = index;
  }
}

// happens-before 的传递闭包。两类边：
//   - SSA 值依赖：destination-style 的写把「这块存储被写过」表达成一个新结果，顺着
//     它往下走能到的 Op 就是被这次写覆盖到的消费者；
//   - 同流的相邻关系：同一条执行流内程序序即执行序，只连相邻的一对，传递性由闭包给出。
// 两类边必须放在一张图里一起走：`A` 与 `C` 既不同流也没有直接值依赖，但如果存在与 `A`
// 同流的 `B` 且 `B` 的结果流到 `C`，`A` 仍然先于 `C`。分开判会把这种情况误报成竞争。
const llvm::DenseSet<Operation *> &RaceChecker::getReachableSet(Operation *from) const {
  auto cached = reachableCache.find(from);
  if (cached != reachableCache.end()) {
    return cached->second;
  }
  llvm::DenseSet<Operation *> reachable;
  llvm::SmallVector<Operation *> worklist{from};
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    if (!reachable.insert(current).second) {
      continue;
    }
    for (Value result : current->getResults()) {
      for (Operation *user : result.getUsers()) {
        worklist.push_back(user);
      }
    }
    auto next = nextInStream.find(current);
    if (next != nextInStream.end()) {
      worklist.push_back(next->second);
    }
  }
  return reachableCache.try_emplace(from, std::move(reachable)).first->second;
}

bool RaceChecker::happensBefore(Operation *earlier, Operation *later) const {
  if (earlier == later || getReachableSet(earlier).contains(later)) {
    return true;
  }
  // 数据依赖与同流都连不通时，问最后一句：earlier 与 later 中间是否隔着屏障
  // （barrier_all / sync_all）。屏障的语义是「所有 pipe 此前的一切完成后，才开始
  // 此后的一切」，所以只要屏障的序号夹在两者之间，earlier 就必先于 later——两者
  // 之间不需要有任何数据流。
  auto earlierIndex = operationOrderIndex.find(earlier);
  auto laterIndex = operationOrderIndex.find(later);
  if (earlierIndex == operationOrderIndex.end() || laterIndex == operationOrderIndex.end()) {
    return false;
  }
  for (Operation *barrier : barrierOps) {
    auto barrierIndex = operationOrderIndex.find(barrier);
    if (barrierIndex != operationOrderIndex.end() && earlierIndex->second < barrierIndex->second &&
        barrierIndex->second < laterIndex->second) {
      return true;
    }
  }
  return false;
}

namespace {
bool isInside(Operation *op, Operation *ancestor) {
  return ancestor->isProperAncestor(op);
}
}  // namespace

// 跨迭代的顺序边只有一种形态：token 由循环的 iter_arg 携带。
//
//     %t0 = afir.event_record : !afir.token<from = transfer_out, to = transfer_in>
//     scf.for %i = ... iter_args(%w = %t0) -> (!afir.token<...>) {
//       afir.event_wait %w        // 等的是**上一次迭代**记下的那个事件
//       ... 写 ...                //   ← 被这条边挡住
//       ... 读 ...
//       %next = afir.event_record // 本次迭代记下，给下一次迭代等
//       scf.yield %next
//     }
//
// 这不是「把同一块 IR 换个写法」：SSA 里 wait 消费的是块参数而不是 record 的结果，
// 两者之间**没有**同迭代的 use-def 边——边跨的正好是一次迭代，由 iter_arg 的定义
// 给出，不需要靠文本先后去猜。AscendC 用事件 ID 复用达到同一效果，代价是配对关系
// 在 IR 上看不见。
//
// 于是判据是：存在这样一条 iter_arg token，`from` 能走到产出它的 record，wait 能
// 走到 `to`。
bool RaceChecker::happensBeforeAcrossIterations(Operation *from, Operation *to, Operation *loop) const {
  auto loopLike = dyn_cast_or_null<LoopLikeOpInterface>(loop);
  if (!loopLike) {
    return false;
  }
  for (EventWaitOp wait : waits) {
    auto carried = dyn_cast<BlockArgument>(wait.getToken());
    if (!carried || !isInside(wait, loop) || carried.getOwner()->getParentOp() != loop) {
      continue;  // token 不由这个循环携带：那对 record/wait 在同一次迭代内。
    }
    OpOperand *yielded = loopLike.getTiedLoopYieldedValue(carried);
    if (!yielded) {
      continue;
    }
    auto record = yielded->get().getDefiningOp<EventRecordOp>();
    if (!record || !isInside(record, loop)) {
      continue;
    }
    if (happensBefore(from, record) && happensBefore(wait, to)) {
      return true;
    }
  }
  return false;
}

void RaceChecker::checkPair(const Access &earlier, const Access &later,
                            llvm::SmallVectorImpl<RaceFinding> &findings) const {
  if (!earlier.isWrite && !later.isWrite) {
    return;  // 两个读永远不冲突。
  }
  if (earlier.op == later.op) {
    // 同一个 Op 的两个操作数：读写同一块存储是 in-place，由 Op 自身语义决定，不是竞争。
    return;
  }

  auto hazardKind = [](const Access &first, const Access &second) {
    if (first.isWrite) {
      return second.isWrite ? HazardKind::WriteAfterWrite : HazardKind::ReadAfterWrite;
    }
    return HazardKind::WriteAfterRead;
  };

  bool sameStream = getPipe(earlier.op) == getPipe(later.op);
  bool ordered = sameStream || happensBefore(earlier.op, later.op);

  // 同一次迭代（或根本不在循环里）。
  if (!ordered && aliasSlices(earlier.slice, later.slice, /*iterationDistance=*/0) == AliasResult::MayAlias) {
    findings.push_back(
        RaceFinding{hazardKind(earlier, later), earlier.op, later.op, earlier.value, later.value, std::nullopt});
    return;
  }

  Operation *commonLoop = getCommonLoop(earlier.op, later.op);
  if (!commonLoop) {
    return;
  }
  // 距离是**这个**循环的迭代数。轮转下标必须由它的归纳变量驱动，模运算才对得上；
  // 不是 `scf.for` 时拿不到归纳变量，别名判定随之退回保守侧。
  auto forOp = dyn_cast<scf::ForOp>(commonLoop);
  Value inductionVariable = forOp ? forOp.getInductionVar() : Value{};

  // 跨迭代方向有两条，方向不同结论也不同。
  //
  //   1. `earlier`@i 对 `later`@(i+d)。它被同迭代的顺序顺带覆盖：`earlier`@i 先于
  //      `earlier`@(i+d)（同一个 Op 的相继两次执行必然有序），后者又先于同迭代的
  //      `later`。所以只要同迭代有序，这条方向就不用报。
  //   2. `later`@i 对 `earlier`@(i+d)。**这条没有任何东西覆盖**——值依赖不跨迭代，
  //      唯一能覆盖它的是「两者同流」。这就是双缓冲 WAR 冒险的形式：消费者还在读第 i
  //      次的数据，生产者已经在写第 i+d 次的同一块存储。多缓冲把 d 推大，推不掉。
  if (!ordered) {
    if (std::optional<int64_t> distance =
            firstAliasingIterationDistance(earlier.slice, later.slice, inductionVariable)) {
      findings.push_back(
          RaceFinding{hazardKind(earlier, later), earlier.op, later.op, earlier.value, later.value, distance});
    }
  }
  if (!sameStream && !happensBeforeAcrossIterations(later.op, earlier.op, commonLoop)) {
    if (std::optional<int64_t> distance =
            firstAliasingIterationDistance(later.slice, earlier.slice, inductionVariable)) {
      findings.push_back(
          RaceFinding{hazardKind(later, earlier), later.op, earlier.op, later.value, earlier.value, distance});
    }
  }
}

llvm::SmallVector<RaceFinding> RaceChecker::check(func::FuncOp func) {
  collectAccesses(func);
  buildStreamOrder();
  llvm::SmallVector<RaceFinding> findings;
  for (size_t i = 0; i < accesses.size(); ++i) {
    for (size_t j = i + 1; j < accesses.size(); ++j) {
      checkPair(accesses[i], accesses[j], findings);
    }
  }
  // 同一对 Op 之间可能有多条操作数各自命中（`afir.add %t, %t` 读同一块存储两次）。
  // 报告的对象是「这两个 Op 之间缺一条 happens-before」，一条就够。
  llvm::SmallVector<RaceFinding> unique;
  for (const RaceFinding &finding : findings) {
    bool duplicate = llvm::any_of(unique, [&](const RaceFinding &kept) {
      return kept.first == finding.first && kept.second == finding.second && kept.kind == finding.kind &&
             kept.iterationDistance == finding.iterationDistance;
    });
    if (!duplicate) {
      unique.push_back(finding);
    }
  }
  return unique;
}

LogicalResult reportRaceFindings(llvm::ArrayRef<RaceFinding> findings) {
  for (const RaceFinding &finding : findings) {
    InFlightDiagnostic diagnostic = finding.first->emitError();
    diagnostic << kDataRaceDiagnostic << ": " << getHazardKindName(finding.kind) << " on the same storage between "
               << stringifyPipe(getPipe(finding.first)) << " and " << stringifyPipe(getPipe(finding.second))
               << " with no ordering between them";
    if (finding.iterationDistance) {
      diagnostic << " at loop-carried distance " << *finding.iterationDistance;
    }
    diagnostic.attachNote(finding.second->getLoc()) << "the other access is here";
  }
  return findings.empty() ? success() : failure();
}

}  // namespace mlir::afir
