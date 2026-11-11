// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --split-input-file --verify-diagnostics %s

// ir_attr_def 不属于 AFIR_CommonAttrs（设计文档 I1/3.7/ADR-7）。
// 限定名和非限定名均应由 AFIRRejectUnknownAttributes 拒绝，不能作为 discardable 字典接收。

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
