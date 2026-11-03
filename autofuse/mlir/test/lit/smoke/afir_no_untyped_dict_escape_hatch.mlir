// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --split-input-file --verify-diagnostics %s

// AFIR Core 不提供无类型字典逃生舱口：`ir_attr_def` 已移出 `AFIR_CommonAttrs`
// （设计文档 I1 / 3.7 / ADR-7）。这里锁住"移出"这件事本身——限定名与非限定名
// 两种写法都必须被 AFIRRejectUnknownAttributes 拦截，而不是被静默接受成
// discardable 属性。若将来有人把字典属性加回 Core，本用例立即失败。

func.func @qualified_dict_rejected(%arg: memref<4xf16>) {
  // expected-error @+1 {{UnsupportedOperation: unknown AFIR operation attribute '"afir.ir_attr_def"'}}
  %result = afir.add %arg, %arg {afir.ir_attr_def = {stride = 2 : i64}} : (memref<4xf16>, memref<4xf16>) -> memref<4xf16>
  return
}

// -----

func.func @unqualified_dict_rejected(%arg: memref<4xf16>) {
  // expected-error @+1 {{UnsupportedOperation: unknown AFIR operation attribute '"ir_attr_def"'}}
  %result = afir.add %arg, %arg {ir_attr_def = {stride = 2 : i64}} : (memref<4xf16>, memref<4xf16>) -> memref<4xf16>
  return
}
