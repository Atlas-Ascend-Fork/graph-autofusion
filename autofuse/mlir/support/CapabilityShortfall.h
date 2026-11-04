/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_SUPPORT_CAPABILITYSHORTFALL_H
#define AUTOFUSE_MLIR_SUPPORT_CAPABILITYSHORTFALL_H

#include <atomic>

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

namespace mlir::afir {

//===----------------------------------------------------------------------===//
// 「能力不足」与「坏了」的区分
//===----------------------------------------------------------------------===//
//
// 这条路上的失败有两种，调用方（`RunMlirCodegen`）要用它们做完全不同的事：
//
//   **能力不足**：这张图 MLIR 这一侧还收不了——某个节点类型没建模、某个形状的判据不满足、
//   某颗芯片没有 profile。legacy 收得了，所以应当回落给它，整条编译继续。
//
//   **坏了**：不变量被破坏、descriptor 自相矛盾、pass 之间语义漂移。回落会把它藏起来，
//   所以必须大声失败。
//
// **缺省是「坏了」。** 没有被显式标记的失败一律按坏了处置——漏标的后果是多报一次失败
// （吵，但看得见），反过来漏判的后果是一个真 bug 被静默回落、再也没人发现。
//
// 判据**不是**诊断文本的前缀。`UnsupportedOperation` 那个前缀今天两种都在用
// （「节点类型没注册」是能力不足，「Scalar 节点没有 value」是输入坏了），按它分类就是
// 拿一个不可靠的标签当判据，而且那等于把分类信息编码进要给人读的那句话里，再解析回来。
// 这里走 `Diagnostic` 的 metadata：产生方和消费方共用下面这一个常量，拼不错、不用解析。
//
// 用法：
//
//   能力不足的那一侧   `emitCapabilityShortfall(op) << "..."`
//   调用方           `CapabilityShortfallObserver observer(context);`
//                    失败时 `observer.sawShortfall() ? kNotSupported : kFailed`

// `Diagnostic::getMetadata()` 里标记这一条的整数。取值本身没有含义，只要全仓唯一——
// 这个仓库里没有别的 metadata 使用者（grep `getMetadata` 零命中）。
inline constexpr int64_t kCapabilityShortfallMetadata = 0x4146'0001;

// 发一条「能力不足」的错误诊断。文本由调用点续写，与普通 `emitError` 完全一样，所以
// 既有的 LIT 断言不受影响——标记只走 metadata，不进文本。
inline InFlightDiagnostic markCapabilityShortfall(InFlightDiagnostic diagnostic) {
  // 诊断引擎没有接收方时 `InFlightDiagnostic` 是空的（`getUnderlyingDiagnostic()` 返回
  // nullptr）。那时标记无处可放，也无人会读。
  if (Diagnostic *underlying = diagnostic.getUnderlyingDiagnostic()) {
    underlying->getMetadata().push_back(DiagnosticArgument(kCapabilityShortfallMetadata));
  }
  return diagnostic;
}

inline InFlightDiagnostic emitCapabilityShortfall(Operation *op) {
  return markCapabilityShortfall(op->emitError());
}

inline InFlightDiagnostic emitCapabilityShortfall(Location loc) {
  return markCapabilityShortfall(emitError(loc));
}

// 本作用域内有没有发生过「能力不足」。
//
// **不吞诊断**：handler 返回 `failure()` 表示「没处理」，诊断继续传给下一个 handler，
// 该打印的照常打印。这个观察者只是在旁边记一个标志。
//
// 标志是 atomic：MLIRContext 默认开多线程，pass 可能在工作线程上发诊断（只有在
// 开 dump 时才 `disableMultithreading`），而 MLIR 不保证 handler 串行调用。
class CapabilityShortfallObserver {
 public:
  explicit CapabilityShortfallObserver(MLIRContext &context)
      : handler_(&context, [this](Diagnostic &diagnostic) {
          for (const DiagnosticArgument &argument : diagnostic.getMetadata()) {
            if (argument.getKind() == DiagnosticArgument::DiagnosticArgumentKind::Integer &&
                argument.getAsInteger() == kCapabilityShortfallMetadata) {
              shortfall_.store(true, std::memory_order_relaxed);
              break;
            }
          }
          return failure();
        }) {}

  bool sawShortfall() const {
    return shortfall_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<bool> shortfall_{false};
  ScopedDiagnosticHandler handler_;
};

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_SUPPORT_CAPABILITYSHORTFALL_H
