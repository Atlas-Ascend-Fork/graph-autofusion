// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `#afir.kernel_arg` 的 round-trip：四种角色、带与不带 index 的每一种形态都必须
// 打印得出、读得回（设计文档 5.3 的字段迁移纪律）。
//
// kernel_arg 统一表达 input/output、workspace 和 tiling 身份，替代整数 input/output 索引
// 及按类型或名称识别 ABI 参数的规则。round-trip 提供不依赖 pass 的序列化基线。

// RUN: af-opt %s | FileCheck %s
// RUN: af-opt %s | af-opt | FileCheck %s

module {
  // 数据参数：input / output 各自带角色内序号。序号不是参数列表下标——buffer 形态
  // 下 output 的参数下标在 input 之后，但它的 index 从 0 重新起算。
  // CHECK-LABEL: func.func private @data_arguments
  // CHECK-SAME: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>}
  // CHECK-SAME: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 1>}
  // CHECK-SAME: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>}
  func.func private @data_arguments(
      %in0: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>},
      %in1: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 1>},
      %out: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>})

  // ABI 参数：workspace 与 tiling 全 kernel 至多一个，没有序号，因此 `index` 整组缺省。
  // CHECK-LABEL: func.func private @abi_arguments
  // CHECK-SAME: memref<?xi8> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}
  // CHECK-SAME: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling>}
  func.func private @abi_arguments(
      %ws: memref<?xi8> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
      %tiling: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling>})

  // 完整的 kernel ABI 参数表：数据参数在前，workspace 与 tiling 在后。位置约定本身
  // 仍在（AscendC 侧要求），但「谁是什么」不再由位置回答。
  // CHECK-LABEL: func.func private @full_abi
  // CHECK-SAME: #afir.kernel_arg<role = input, index = 0>
  // CHECK-SAME: #afir.kernel_arg<role = output, index = 0>
  // CHECK-SAME: #afir.kernel_arg<role = workspace>
  // CHECK-SAME: #afir.kernel_arg<role = tiling>
  func.func private @full_abi(
      %in: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>},
      %out: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>},
      %ws: memref<?xi8> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
      %tiling: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling>})
}
