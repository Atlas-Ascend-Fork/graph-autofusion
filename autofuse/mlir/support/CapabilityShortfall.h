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
// 区分能力不足与编译错误
//===----------------------------------------------------------------------===//
//
// RunMlirCodegen 根据失败类型决定是否回落 legacy：
//   能力不足：MLIR 尚未支持节点类型、形状或目标 profile，允许尝试 legacy 并继续编译。
//   编译错误：不变量被破坏、descriptor 矛盾或 pass 间语义不一致，必须报告失败。
//
// 未显式标记的失败按编译错误处理，避免把实现缺陷隐藏在回落路径中。
// 诊断文本的 UnsupportedOperation 前缀不能用于分类：节点类型未注册属于能力不足，
// Scalar 节点缺少 value 则是输入错误，两者都可能使用该前缀。
// 分类标记放在 Diagnostic metadata 中，产生方和消费方共用下面的常量，不解析诊断文本。
//
// 用法：
//
//   能力不足的那一侧   `emitCapabilityShortfall(op) << "..."`
//   调用方           `CapabilityShortfallObserver observer(context);`
//                    失败时 `observer.sawShortfall() ? kNotSupported : kFailed`

// Diagnostic metadata 中的能力不足标记。取值无业务含义，须避免与其他 metadata 标记冲突。
inline constexpr int64_t kCapabilityShortfallMetadata = 0x4146'0001;

// 给诊断添加能力不足标记，文本仍由调用点续写。标记不进入文本，不影响既有 LIT 断言。
inline InFlightDiagnostic markCapabilityShortfall(InFlightDiagnostic diagnostic) {
  // 没有诊断接收方时 getUnderlyingDiagnostic() 返回 nullptr，无需添加标记。
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

// 记录作用域内是否出现能力不足诊断，不拦截诊断传播。
// handler 返回 failure()，诊断仍交给后续 handler 处理和输出。
// MLIRContext 默认启用多线程（dump 时才禁用），pass 可在工作线程中输出诊断，
// 且 handler 不保证串行调用，因此标志使用 atomic。
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
