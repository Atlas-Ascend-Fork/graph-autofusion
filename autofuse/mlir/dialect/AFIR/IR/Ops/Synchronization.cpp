/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/IR/AFIR.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::afir {

// 一个 token 代表一个具体的事件槽，只能被等待一次。等第二次时该事件早已被第一次消耗，
// 硬件上不会阻塞——写的人以为挡住了依赖，实际什么都没挡住，失败方式是静默的数据竞争。
// 这正是 lowering 阶段的事件槽分配（DAG coloring）所依赖的前提：一个 token 一条边。
LogicalResult EventWaitOp::verify() {
  Value token = getToken();
  auto waits = llvm::make_filter_range(token.getUsers(), [](Operation *user) { return isa<EventWaitOp>(user); });
  if (std::distance(waits.begin(), waits.end()) > 1) {
    return emitOpError() << kInvalidDescriptorDiagnostic << ": a token can only be awaited once";
  }
  return success();
}

// 与「token 只能等一次」对称的另一半（弱化版）：record 产生的 token 至少要被消费一次。
// 没人消费的 record 在下降层 AllocEventID 占槽后无人 ReleaseEventID——槽泄漏，失败
// 形态是后续同型 record 拿到越界事件号。
// 只查「至少一次」、不查「恰好一次」：跨迭代携带（token 经 scf.for 的 iter_args /
// yield 传给下一迭代的 wait）是合法习语，其末迭代的 record 按构造不被等、必然泄漏
// 一个槽，强校验会拒掉它。pass 生成的对总是成对，这条拦的是手写 IR 里悬空的 record。
LogicalResult EventRecordOp::verify() {
  if (getToken().getUsers().empty()) {
    return emitOpError() << kInvalidDescriptorDiagnostic
                         << ": a recorded token must be consumed by an event_wait, directly or carried across "
                            "iterations; a dead record leaks its event slot";
  }
  return success();
}

// 兜底同步原语的驻地约束：它们语义上是 kernel 体内的全序/跨核事实，出现在别的函数
// 角色里没有意义（调度契约只对 KernelBody 成立）。
static LogicalResult verifyKernelBodyOnly(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func || !hasFunctionRole(func, FunctionRole::KernelBody)) {
    return op->emitOpError() << kInvalidDescriptorDiagnostic << ": only allowed inside a kernel body function";
  }
  return success();
}

LogicalResult BarrierAllOp::verify() {
  return verifyKernelBodyOnly(getOperation());
}

LogicalResult SyncAllOp::verify() {
  return verifyKernelBodyOnly(getOperation());
}

}  // namespace mlir::afir
