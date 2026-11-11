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

// 每个 token 对应一个事件槽，只能等待一次。事件被首次等待消耗后，第二次等待
// 不会阻塞，不能保护后续依赖。lowering 的槽位分配依赖这一约束：一个 token 对应一条边。
LogicalResult EventWaitOp::verify() {
  Value token = getToken();
  auto waits = llvm::make_filter_range(token.getUsers(), [](Operation *user) { return isa<EventWaitOp>(user); });
  if (std::distance(waits.begin(), waits.end()) > 1) {
    return emitOpError() << kInvalidDescriptorDiagnostic << ": a token can only be awaited once";
  }
  return success();
}

// record 产生的 token 必须有使用者，否则 AllocEventID 占用的槽位无法释放，
// 后续同类 record 可能取得越界事件号。
// 这里只检查至少一个使用者，不强制直接等待：token 可经 scf.for 的 iter_args / yield
// 传给下一轮，末轮 record 也可能不再被等待。严格检查直接消费次数会拒绝这种循环形式。
// 此检查主要拒绝手写 IR 中完全悬空的 record。
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
