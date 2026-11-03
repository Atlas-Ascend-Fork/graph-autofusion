// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `#afir.kernel_arg` 的 negative 用例。
//
// 旧机制的失效方式全是静默的：`afir.input_index` 与 `afir.output_index` 同时挂在
// 一个参数上是合法 IR，两个都不挂则退化成按类型猜；序号为负只在个别消费端被查。
// 这些形态现在必须在属性构造时就失败，否则强类型化没有换来任何东西。

// RUN: af-opt --split-input-file --verify-diagnostics %s

// 数据参数缺序号：`arg_descs` 与主机侧实参表都按序号对齐，缺号的参数送不进 kernel。
module {
  // expected-error @below {{kernel_arg role 'input' requires a non-negative index}}
  func.func private @input_without_index(%in: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input>})
}

// -----

// 数据参数序号为负。-1 是「没写」的哨兵，不是一个可用的序号。
module {
  // expected-error @below {{kernel_arg role 'output' requires a non-negative index}}
  func.func private @negative_index(%out: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = -1>})
}

// -----

// workspace 带序号：全 kernel 至多一个 workspace，允许写序号等于默许了「第 2 个
// workspace」这种当前 ABI 表达不了的形态。
module {
  // expected-error @below {{kernel_arg role 'workspace' cannot carry an index}}
  func.func private @workspace_with_index(%ws: memref<?xi8> {afir.kernel_arg = #afir.kernel_arg<role = workspace, index = 0>})
}

// -----

// tiling 的序号是**可选**的，带号时号是 stage：Plan 的 `_main` 一 stage 收一个 tiling 形参
// （ATT 的最外层 TilingData 里没有 plan 那一层，见
// `common/tiling_schema/schedule_group_names.h`）。所以带号本身合法，负号才非法。
module {
  func.func private @tiling_with_stage(%t0: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling, index = 0>},
                                       %t1: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling, index = 1>})
}

// -----

// 但 -1 以下的序号不是 stage。
module {
  // expected-error @below {{kernel_arg role 'tiling' index must be a stage id or absent}}
  func.func private @tiling_negative_stage(%t: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling, index = -2>})
}

// -----

// 两个都不带号的 tiling 仍然互斥：不带号者共用 -1 这个键，撞在同一条判重路径上。
module {
  // expected-error @below {{duplicate kernel_arg role 'tiling' at argument 1; argument 0 already declares it}}
  func.func private @two_unnumbered_tilings(%a: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling>},
                                            %b: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling>})
}

// -----

// 两个 stage 抢同一个号也是重复。
module {
  // expected-error @below {{duplicate kernel_arg role 'tiling' at argument 1; argument 0 already declares it}}
  func.func private @two_same_stage_tilings(%a: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling, index = 1>},
                                            %b: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling, index = 1>})
}

// -----

// 两个 workspace。全 kernel 至多一个——「至多一个」此前只在 memory-allocate 跑过时
// 才被查，也就是说这份 IR 能原样 round-trip 过 af-opt，与旧机制「两个都挂是合法 IR」
// 是同一个形态。跨参数的约束必须在解析期就生效。
module {
  // expected-error @below {{duplicate kernel_arg role 'workspace' at argument 1; argument 0 already declares it}}
  func.func private @two_workspaces(%a: memref<?xi8> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
                                    %b: memref<?xi8> {afir.kernel_arg = #afir.kernel_arg<role = workspace>})
}

// -----

// 两个 tiling，同上。
module {
  // expected-error @below {{duplicate kernel_arg role 'tiling' at argument 1}}
  func.func private @two_tilings(%a: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling>},
                                 %b: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling>})
}

// -----

// 同一个角色内序号重复。两个参数都自称「第 0 个输入」，主机侧的实参表对不上，而
// 两个声明单看各自都合法——这条只能跨参数查。
module {
  // expected-error @below {{duplicate kernel_arg role 'input' at argument 1; argument 0 already declares it}}
  func.func private @same_input_index(%a: memref<4xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>},
                                      %b: memref<4xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>})
}

// -----

// 反例侧：input 与 output 各自的序号从 0 起算，同为 0 不是冲突——序号是**角色内**的
// 序号，不是参数列表下标。放过这条才说明判重查的是 (角色, 序号) 而不是序号本身。
module {
  func.func private @distinct_roles(%a: memref<4xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>},
                                    %b: memref<4xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>},
                                    %w: memref<?xi8> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
                                    %t: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling>})
}

// -----

// `workspace_offset` 缺序号。它和 input/output 一样按序号对齐——序号是这块 workspace
// 张量的 buffer id，下降侧靠它把偏移形参和 GM 值对上，缺号就对不上任何一块。
module {
  func.func private @workspace_offset_without_index(
      // expected-error @below {{kernel_arg role 'workspace_offset' requires a non-negative index}}
      %o: i32 {afir.kernel_arg = #afir.kernel_arg<role = workspace_offset>})
}

// -----

// 反例侧：两块 workspace 各带自己的 buffer id，同为 `workspace_offset` 不是冲突——
// 判重查的是 (角色, 序号)，而这两条的序号不同。
module {
  func.func private @distinct_workspace_offsets(
      %w: memref<?xi8> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
      %o2: i32 {afir.kernel_arg = #afir.kernel_arg<role = workspace_offset, index = 2>},
      %o5: i32 {afir.kernel_arg = #afir.kernel_arg<role = workspace_offset, index = 5>},
      %t: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling>})
}

// -----

// 角色写错。枚举化换来的正是这条——旧的 `afir.input_index` / `afir.output_index`
// 是属性名，名字写错只会被当成「没挂」。
module {
  // expected-error @below {{expected one of [input, output, workspace, tiling, scalar_input, workspace_offset]}}
  // expected-error @below {{failed to parse KernelArgAttr parameter 'role'}}
  func.func private @unknown_role(%in: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = inputs, index = 0>})
}
