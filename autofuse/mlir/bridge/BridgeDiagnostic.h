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

// 只置真、不置假：出口把这一位标成「能力不足」之后，调用链上游不该再把它抹掉。缺省的 `false`
// 由调用方初始化，语义见 `PlanSelection.h` 上 `unsupported` 那一段。
inline void SetUnsupported(bool *unsupported) {
  if (unsupported != nullptr) {
    *unsupported = true;
  }
}

}  // namespace mlir::bridge

#endif  // AUTOFUSE_MLIR_BRIDGE_BRIDGE_DIAGNOSTIC_H
