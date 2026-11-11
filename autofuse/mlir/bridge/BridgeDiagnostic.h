/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_BRIDGE_BRIDGE_DIAGNOSTIC_H
#define AUTOFUSE_MLIR_BRIDGE_BRIDGE_DIAGNOSTIC_H

#include <string>
#include <string_view>

namespace mlir::bridge {

inline void SetDiagnostic(std::string *diagnostic, std::string_view message) {
  if (diagnostic != nullptr) {
    *diagnostic = message;
  }
}

// 只设置能力不足标记，不清除下游已设置的值。调用方负责初始化为 false。
// unsupported 的语义见 PlanSelection.h。
inline void SetUnsupported(bool *unsupported) {
  if (unsupported != nullptr) {
    *unsupported = true;
  }
}

}  // namespace mlir::bridge

#endif  // AUTOFUSE_MLIR_BRIDGE_BRIDGE_DIAGNOSTIC_H
