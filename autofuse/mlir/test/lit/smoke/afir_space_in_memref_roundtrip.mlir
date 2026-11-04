// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 步 3（space 进 memref memory space）的等价性基线。
//
// 迁移前 space 只存在于 `#afir.buffer_desc` 的 `space` 字段里，memref 自己的 memory
// space 完全空着（问题 D8）。本文件锁住迁移后的编码：**同一个 SSA 值的地址空间写在
// 它的类型上**，`#afir.space<*>` 直接作为 memref 的 memory space 属性。
//
// 按 5.3 的硬纪律，这个文件先于生产代码落地：它在 `AfirBuilder` 还没开始产出带 space
// 的 memref 时就已经把目标编码钉死，之后生产代码的每一步只能向它靠拢，不能改它。
//
// 锁三件事：
//   1. 七个 space 取值都能作为 memory space 解析并原样打印；
//   2. `memref.subview` 自动保持 memory space——这正是 Op 属性编码做不到、
//      需要 `CanonicalBindings` 维护侧表才能补上的那条性质（问题 1.2-3）；
//   3. 打印形式是解析的不动点（第二条 RUN）。

// RUN: af-opt %s | FileCheck %s
// RUN: af-opt %s | af-opt | FileCheck %s

module {
  // 七个 space 各出现一次，与 afir_descriptor_roundtrip.mlir 的 @desc_all_spaces
  // 构成新旧编码的对照：那边 space 在 Op 属性里，这边在值的类型里。
  func.func private @space_all_values(
      %gm: memref<8xf16, #afir.space<gm>>,
      %ub: memref<8xf16, #afir.space<ub>>,
      %l1: memref<8xf16, #afir.space<l1>>,
      %l2: memref<8xf16, #afir.space<l2>>,
      %l0a: memref<8xf16, #afir.space<l0a>>,
      %l0b: memref<8xf16, #afir.space<l0b>>,
      %l0c: memref<8xf16, #afir.space<l0c>>) {
    return
  }

  // 无 memory space 的 memref 仍然合法：迁移期 bridge 产出的旧形态要能继续解析，
  // 否则步 3 就不是可回退的一步。
  func.func private @space_absent(%arg: memref<8xf16>) {
    return
  }

  // subview 保持 memory space。这条是步 3 的收益本身：切片后的值不需要任何侧表
  // 就带着正确的地址空间，而 Op 属性编码下每个使用点都得重新挂一份 descriptor。
  func.func private @space_survives_subview(%arg: memref<32x8xf16, #afir.space<ub>>)
      -> memref<4x8xf16, strided<[8, 1], offset: 64>, #afir.space<ub>> {
    %view = memref.subview %arg[8, 0] [4, 8] [1, 1]
        : memref<32x8xf16, #afir.space<ub>> to memref<4x8xf16, strided<[8, 1], offset: 64>, #afir.space<ub>>
    return %view : memref<4x8xf16, strided<[8, 1], offset: 64>, #afir.space<ub>>
  }

  // space 与 layout 正交：带 strided layout 的 memref 同时携带 memory space。
  // 步 6 引入 `#afir.layout` 时，layout 位置换人、space 位置不变。
  func.func private @space_with_layout(%arg: memref<4x8xf16, strided<[8, 1], offset: 16>, #afir.space<l1>>) {
    return
  }

  // 迁移期容忍：非 `#afir.space` 的 memory space 仍可解析。
  //
  // 这不是目标形态，而是 I4 待清理的现状：`AFIRKernelBodySignature` 会把 kernel 参数
  // 的地址空间直接换成 AscendC 的 CCE-C 魔数（22 = `ascendc::AddressSpace::gm`），而它
  // 在生产流水线 `addPlanCandidateLoweringPipeline` 里排在 AFIR op 全部消失**之前**，
  // 所以这种形态会真实出现在还带着 descriptor 的 AFIR op 上。
  //
  // I4 把该映射收为单点、AFIR 侧全程用 `#afir.space` 之后，这条用例应改为 negative。
  func.func private @foreign_memory_space_tolerated(%arg: memref<8xf16, 22 : i32>) {
    return
  }
}

// CHECK-LABEL: func.func private @space_all_values
// CHECK-SAME: memref<8xf16, #afir.space<gm>>
// CHECK-SAME: memref<8xf16, #afir.space<ub>>
// CHECK-SAME: memref<8xf16, #afir.space<l1>>
// CHECK-SAME: memref<8xf16, #afir.space<l2>>
// CHECK-SAME: memref<8xf16, #afir.space<l0a>>
// CHECK-SAME: memref<8xf16, #afir.space<l0b>>
// CHECK-SAME: memref<8xf16, #afir.space<l0c>>

// CHECK-LABEL: func.func private @space_absent
// CHECK-SAME: memref<8xf16>

// CHECK-LABEL: func.func private @space_survives_subview
// CHECK: memref.subview
// CHECK-SAME: to memref<4x8xf16, strided<[8, 1], offset: 64>, #afir.space<ub>>

// CHECK-LABEL: func.func private @space_with_layout
// CHECK-SAME: memref<4x8xf16, strided<[8, 1], offset: 16>, #afir.space<l1>>

// CHECK-LABEL: func.func private @foreign_memory_space_tolerated
// CHECK-SAME: memref<8xf16, 22 : i32>
