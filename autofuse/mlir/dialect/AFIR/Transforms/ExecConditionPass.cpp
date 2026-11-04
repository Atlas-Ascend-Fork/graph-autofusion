/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ExecConditionPass.cpp - Turn sched.exec_condition into scf.if ------===//
//
// `afir.sched` 的 `exec_condition` 说的是一个算子的缓存复用周期：值在周期内不变，周期
// 起点算一次、其余迭代直接读上一次留在 UB 里的结果。这个 pass 把那句话变成循环体里的一个
// `scf.if`，谓词由 `materializeExecConditionGuard` 按轴关系建。
//
// **为什么不在循环生成里一起做**：那时算子之间还是 SSA 相连的（`afir.broadcast` 直接吃
// `afir.load` 的结果），把生产者挪进一个 region 就要靠 `scf.if` 的返回值把它带出来，而
// 目的地形态下那个值本来就等于 `outs` 操作数——多一层 yield 是纯噪声。资源物化
// （`afir-memory-allocate`）之后，队列成员之间已经由 `EnQue` / `DeQue` 断开，剩下的结果
// 要么没人用、要么可以直接换成它 tie 的那个 destination。
//
// **为什么排在同步之后**：被跳过的只有算子本身，`AllocTensor` / `EnQue` / `DeQue` /
// `FreeTensor` 与流水屏障每趟照发——缓冲是单 slot（可缓存的张量上游给的是 `buf_num = 1`），
// 地址每趟相同，所以跳过一趟写入之后读到的正是上一趟的结果。这与 legacy 的
// `Loop::GenerateBody` 逐字一致：那边的 `if` 也只包住 `call->Generate`，生命周期与同步都在
// 外面。把守卫加在同步之前会让同步算法看到一条有条件的边，那是另一件事。
//
//===----------------------------------------------------------------------===//

#include "dialect/AFIR/Transforms/AFIRPasses.h"

#include "dialect/AFIR/Analysis/TransferPlan.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "support/CapabilityShortfall.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::afir {
namespace {

ExecCondition getExecCondition(Operation *op) {
  auto sched = op->getAttrOfType<ScheduleAttr>(kSchedAttrName);
  return sched ? sched.getExecCondition() : ExecCondition::NoCache;
}

// 守卫谓词发在循环体开头，一个 (循环, 取值) 只发一份——legacy 那边也是每个循环一个
// `bool`，而这个 pass 之后没有 CSE，逐个算子各发一份就会在生成的 kernel 里留下重复的标量
// 算术。
class GuardCache {
 public:
  FailureOr<Value> get(Operation *op, const TransferPlan &plan, ExecCondition condition, scf::ForOp loop) {
    Key key{loop.getOperation(), static_cast<unsigned>(condition)};
    auto existing = cached.find(key);
    if (existing != cached.end()) {
      return existing->second;
    }
    Block &body = *loop.getBody();
    OpBuilder builder(&body, body.begin());
    FailureOr<Value> guard = materializeExecConditionGuard(op, builder, plan, condition);
    if (failed(guard)) {
      return failure();
    }
    cached.try_emplace(key, *guard);
    return guard;
  }

 private:
  using Key = std::pair<Operation *, unsigned>;
  llvm::DenseMap<Key, Value> cached;
};

// 结果换成它 tie 的 destination。目的地形态下两者是同一块存储（`getTiedDestination` 的
// 注释），所以这不是近似：换完之后算子的结果无人使用，可以整条挪进一个无返回值的
// `scf.if`。tie 不上又确实有使用者的形态**报错而不是放过**——放过的表现是 region 外用到
// region 内定义的值，那是 IR 校验失败，位置离这里已经很远。
LogicalResult detachResults(Operation *op) {
  for (Value result : op->getResults()) {
    if (result.use_empty()) {
      continue;
    }
    Value destination = getTiedDestination(result);
    if (!destination) {
      // 这一条**不是能力不足**：目的地形态下结果与 destination 是同一块存储，tie 不上又
      // 有使用者说明两者已经漂移，回落会把它藏起来。标签按此订正（原先写的是
      // `UnsupportedOperation`，那会让人以为换张图就行）。
      return op->emitError(
          "InvalidBinding: cached operation result is still in use but not tied to a destination operand; "
          "exec_condition cannot be lowered to a predicated region");
    }
    result.replaceAllUsesWith(destination);
  }
  return success();
}

LogicalResult guardOperation(Operation *op, ExecCondition condition, GuardCache &guards) {
  FailureOr<TransferPlan> plan = buildTransferPlan(op);
  if (failed(plan) || plan->kind != TransferPlanKind::Scheduled) {
    return emitCapabilityShortfall(op) << "UnsupportedOperation: exec_condition " << stringifyExecCondition(condition)
                                       << " needs a block/tile split schedule, which this operation lacks";
  }
  scf::ForOp loop;
  for (Operation *parent = op->getParentOp(); parent != nullptr; parent = parent->getParentOp()) {
    if (auto candidate = dyn_cast<scf::ForOp>(parent)) {
      if (hasAxisBinding(candidate, plan->loopAxis, AxisValueRole::Loop)) {
        loop = candidate;
        break;
      }
    }
  }
  if (!loop) {
    return emitCapabilityShortfall(op) << "UnsupportedOperation: exec_condition " << stringifyExecCondition(condition)
                                       << " requires an enclosing loop on axis " << plan->loopAxis
                                       << ", which loop generation did not materialize";
  }
  FailureOr<Value> guard = guards.get(op, *plan, condition, loop);
  if (failed(guard)) {
    return emitCapabilityShortfall(op) << "UnsupportedOperation: cannot build the cache guard for exec_condition "
                                       << stringifyExecCondition(condition);
  }
  if (failed(detachResults(op))) {
    return failure();
  }
  OpBuilder builder(op);
  auto ifOp = builder.create<scf::IfOp>(op->getLoc(), *guard, /*withElseRegion=*/false);
  Block *thenBlock = ifOp.thenBlock();
  op->moveBefore(thenBlock, thenBlock->begin());
  return success();
}

LogicalResult runOnFunction(func::FuncOp func) {
  SmallVector<std::pair<Operation *, ExecCondition>> pending;
  func.walk([&](Operation *op) {
    if (!isAfirOperation(op)) {
      return;
    }
    // **region 内不插缓存守卫。** `afir.vector_func` 的 region 是一段在寄存器里算完的
    // 数据流，守卫要包的是整段（算子自己带着它从子图首节点继承来的 `exec_condition`），
    // 往里插等于在 `__VEC_SCOPE__` 里发 tile 级的守卫算子。
    if (isInsideVectorFunc(op)) {
      return;
    }
    ExecCondition condition = getExecCondition(op);
    if (condition != ExecCondition::NoCache) {
      pending.emplace_back(op, condition);
    }
  });
  GuardCache guards;
  for (auto [op, condition] : pending) {
    if (failed(guardOperation(op, condition, guards))) {
      return failure();
    }
  }
  return success();
}

struct AFIRExecConditionPass : public PassWrapper<AFIRExecConditionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRExecConditionPass)

  StringRef getArgument() const final {
    return "afir-exec-condition";
  }

  StringRef getDescription() const final {
    return "Wrap operations carrying a cache exec_condition in the scf.if that gates their reuse cycle";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AFIRDialect, arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      if (func.isExternal()) {
        continue;
      }
      if (failed(runOnFunction(func))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRExecConditionPass() {
  return std::make_unique<AFIRExecConditionPass>();
}

void registerAFIRExecConditionPass() {
  PassRegistration<AFIRExecConditionPass>();
}

}  // namespace mlir::afir
