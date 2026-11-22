// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.tiling_param` 的 negative 用例。
//
// 参数名同时用于 expr 符号和目标 ABI 字段，必须是标识符。
// 非法名称应在本层拒绝，不能等到 emitasc.member 生成 C++ 时才报错。

// RUN: af-opt --split-input-file --verify-diagnostics %s

module {
  func.func @empty_name() {
    // expected-error @below {{tiling parameter name must not be empty}}
    %n = afir.tiling_param "" : index
    return
  }
}

// -----

module {
  func.func @leading_digit() {
    // expected-error @below {{tiling parameter name must be an identifier, got '0q_size'}}
    %n = afir.tiling_param "0q_size" : index
    return
  }
}

// -----

// 表达式片段不是名字。写成这样多半是想让 AFIR 去算一个表达式，而表达式该由
// `#afir.expr` 表达、由各自的符号绑定出来，不是塞进一个参数名里。
module {
  func.func @expression_as_name() {
    // expected-error @below {{tiling parameter name must be an identifier, got '2 * s0'}}
    %n = afir.tiling_param "2 * s0" : index
    return
  }
}
