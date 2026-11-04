// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


// RUN: af-opt --split-input-file --verify-diagnostics %s
module {
  func.func @empty() {
    // expected-error@+1 {{afir.axis_binding must be an axis binding or a non-empty array of axis bindings}}
    %v = arith.constant {afir.axis_binding = []} 1 : index
    return
  }
}
// -----
module {
  func.func @wrong_element() {
    // expected-error@+1 {{afir.axis_binding must be an axis binding or a non-empty array of axis bindings}}
    %v = arith.constant {afir.axis_binding = [1 : i64]} 1 : index
    return
  }
}
