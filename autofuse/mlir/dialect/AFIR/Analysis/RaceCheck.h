/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_RACECHECK_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_RACECHECK_H

#include <cstdint>
#include <optional>

#include "dialect/AFIR/Analysis/BufferAlias.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace func {
class FuncOp;
}  // namespace func
}  // namespace mlir

namespace mlir::afir {

// 一个 Op 跑在哪条 pipe 上。词表是方言里的 `afir::Pipe`（AFIREnums.td）——竞争检查器
// 与同步原语谈的必须是同一件事，否则 `afir.event_record` 给出的那条边挂不到这里的
// 依赖图上。AFIR 自己就能分类：`AFIRTransferIn` / `AFIRTransferOut` 是方言里已有的
// trait，其余算计算流；`afir.event_record` / `afir.event_wait` 则由所携带 token 的
// `from` / `to` 决定。
Pipe getPipe(Operation *op);

enum class HazardKind : uint8_t {
  ReadAfterWrite,
  WriteAfterRead,
  WriteAfterWrite,
};

llvm::StringRef getHazardKindName(HazardKind kind);

// 一条竞争。`first` 在程序序上先于 `second`；`iterationDistance` 为空表示同一次迭代，
// 否则是「`second` 比 `first` 晚这么多次迭代时才撞上」——单缓冲是 1，S 路轮转是 S。
struct RaceFinding {
  HazardKind kind = HazardKind::ReadAfterWrite;
  Operation *first = nullptr;
  Operation *second = nullptr;
  Value firstAccess;
  Value secondAccess;
  std::optional<int64_t> iterationDistance;
};

// 竞争检查：依赖图（SSA 值依赖 + 同流程序序）+ 别名分析（`BufferAlias.h`）。
//
// 判定规则只有一条：两个可能别名的访问，至少一个是写，且谁都不 happens-before 谁，
// 就是竞争。AFIR 层的 happens-before 有三个来源：
//
//   1. SSA 值依赖。destination-style Op 把「写了这块存储」表达成一个新结果，消费者
//      必须用那个结果而不是绕回去用 destination——绕过去就是「未覆盖的 producer」。
//   2. 同一条 pipe 内的程序序。
//   3. 显式同步：一对 `afir.event_record` / `afir.event_wait`（由同一个
//      `!afir.token` 相连）。record 落在 `from` pipe 的链上、wait 落在 `to` pipe 的
//      链上，中间由 token 的 use-def 串起来——于是这条边和前两类走的是同一张图，
//      不需要单独一套判定。
//
// 第 3 条是这套模型与 `TQue` 自动同步的分界：放弃 `TQue` 之后正确性归 AFIR 自己
// （设计文档 3.2 / ADR-12），而「归 AFIR 自己」的具体含义就是那条边必须在这张图里
// 看得见。`TQue` 的 enque/deque 在 lowering 之后才出现，这里看不到，也不应该看到。
//
// 跨迭代方向上第 1 条不成立（AFIR 的 buffer 形态不用 iter_args 串依赖），能覆盖它的
// 只有同 pipe 程序序、以及循环体内 wait 在 record 之前的那种配对——后者正是双缓冲
// WAR 冒险的防护形式：第 i+1 次迭代的 wait 等的是第 i 次迭代的 record。两者都没有，
// 跨 pipe 的循环携带依赖就是竞争，多缓冲只把距离推远，推不掉。
class RaceChecker {
 public:
  llvm::SmallVector<RaceFinding> check(func::FuncOp func);

 private:
  struct Access {
    Operation *op = nullptr;
    Value value;
    bool isWrite = false;
    BufferSlice slice;
  };

  void collectAccesses(func::FuncOp func);
  void buildStreamOrder();
  bool happensBefore(Operation *earlier, Operation *later) const;
  // `from` 在第 i 次迭代的执行，是否 happens-before `to` 在第 i+1 次及以后迭代的执行。
  // 唯一的来源是循环体内一对 record/wait，其中 wait 在程序序上位于 record 之前——那条
  // 边跨的正好是一次迭代。
  bool happensBeforeAcrossIterations(Operation *from, Operation *to, Operation *loop) const;
  const llvm::DenseSet<Operation *> &getReachableSet(Operation *from) const;
  void checkPair(const Access &earlier, const Access &later, llvm::SmallVectorImpl<RaceFinding> &findings) const;

  llvm::SmallVector<Access> accesses;
  llvm::SmallVector<Operation *> orderedOperations;
  llvm::SmallVector<EventWaitOp> waits;
  llvm::DenseMap<Operation *, Operation *> nextInStream;
  // 屏障（`afir.barrier_all` / `afir.sync_all`）的账本，只装这两种 op，由
  // collectAccesses 填写：
  //   barrierOps          —— 屏障清单；
  //   operationOrderIndex —— 每个 op 的程序序号。
  // 屏障对所有 pipe 统一生效（此前的一切 happens-before 此后的一切），不属于某一条
  // 流，nextInStream 装不下。happensBefore 在数据依赖与同流都连不通时，用这两本账
  // 问最后一句：earlier 与 later 的序号中间是否夹着屏障。
  llvm::SmallVector<Operation *> barrierOps;
  llvm::DenseMap<Operation *, size_t> operationOrderIndex;
  // happens-before 的可达集在一次 check() 里不会变，但要被 O(n^2) 对访问反复问到。
  mutable llvm::DenseMap<Operation *, llvm::DenseSet<Operation *>> reachableCache;
};

// 把发现报成诊断挂在相关 Op 上。返回 failure 当且仅当有发现。
LogicalResult reportRaceFindings(llvm::ArrayRef<RaceFinding> findings);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_RACECHECK_H
