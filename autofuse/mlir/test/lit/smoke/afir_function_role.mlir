// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt %s | FileCheck %s --check-prefix=ROLE
// RUN: af-opt --afir-kernel-body-signature %s | FileCheck %s --check-prefix=ABI
// RUN: af-opt --afir-kernel-body-signature --afir-kernel-body-signature %s | \
// RUN: FileCheck %s --check-prefix=ABI

func.func private @kernel_body(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>) -> tensor<4xf16> attributes {
  afir.function_role = #afir.function_role<kernel_body>
} {
  %0 = afir.add %lhs, %rhs : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

func.func @kernel_entry(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>) -> tensor<4xf16> attributes {
  afir.function_role = #afir.function_role<kernel_entry>
} {
  %0 = afir.add %lhs, %rhs : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// ROLE-LABEL: func.func private @kernel_body
// ROLE-SAME: afir.function_role = #afir.function_role<kernel_body>
// ROLE-LABEL: func.func @kernel_entry
// ROLE-SAME: afir.function_role = #afir.function_role<kernel_entry>

// ABI-LABEL: func.func private @kernel_body
// ABI-SAME: memref<4xf16, 22 : i32>
// ABI 追加出来的两个参数自己声明身份，不再靠「倒数第二个且类型相符」「结构体名字叫
// TilingData」反推（设计文档 3.6）。
// ABI-SAME: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}
// ABI-SAME: !emitasc.py_struct<"TilingData", [i64, i64], ["tile_len", "total"]> {afir.kernel_arg = #afir.kernel_arg<role = tiling>}
// ABI-SAME: ascendc.aicore
// ABI-SAME: ascendc.num_inputs = 2 : i32
// ABI-NOT: afir.num_inputs
// ABI-NOT: ascendc.global
// ABI-LABEL: func.func @kernel_entry
// ABI-SAME: ascendc.aicore
// ABI-SAME: ascendc.global
// ABI-SAME: ascendc.num_inputs = 2 : i32
