// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 步 3b（轴扁平属性合一）的等价性基线。
//
// 迁移前，轴信息散在 5 个无类型属性里：
//   afir.sched_axis      DenseI64Array，算子被调度在哪些轴上
//   afir.loop_axis       Integer，最内层循环轴
//   afir.vectorized_axis DenseI64Array，结果的向量化轴
//   afir.axis_id         Integer，某个 SSA 值 / scf.for 对应哪个轴
//   afir.axis_role       自由字符串，该值是这个轴的什么
//
// 其中 `axis_id` + `axis_role` 承载的是一条 IR 结构上不存在的反向引用：
// `LoopGenerate` 把轴物化成标量值之后，`TransferPlan` 只能靠遍历全函数、比对这对
// 属性把值找回来。角色是自由字符串，拼错既不报错也查不到值——失败方式是静默回退。
//
// 本文件先钉死替代通路的编码，`LoopGenerate` / `TransferPlan` 随后才改到它上面。

// RUN: af-opt %s | FileCheck %s
// RUN: af-opt %s | af-opt | FileCheck %s

module {
  // #afir.axis_binding 的六个角色各出现一次，锁住全枚举。
  // 角色进枚举之后，拼错在解析期就失败，不再是运行时查不到。
  // 属性挂在产生该值的 op 上，与 `LoopGenerate::tagAxisValue` 的落点一致。
  func.func private @axis_binding_all_roles() {
    %index = arith.constant {afir.axis_binding = #afir.axis_binding<axis = 1, role = axis_index>} 0 : index
    %size = arith.constant {afir.axis_binding = #afir.axis_binding<axis = 2, role = axis_size>} 1 : index
    %loop_size = arith.constant {afir.axis_binding = #afir.axis_binding<axis = 3, role = loop_size>} 2 : index
    %tail = arith.constant {afir.axis_binding = #afir.axis_binding<axis = 4, role = tail_size>} 3 : index
    %actual = arith.constant {afir.axis_binding = #afir.axis_binding<axis = 5, role = actual_size>} 4 : index
    return
  }

  // 替代通路的目标形态：`scf.for` 自己说它迭代的是哪个轴。
  // 这是 `CanonicalMaterialization` 从循环反查回 `AxisAttr` 所依赖的那条引用。
  func.func private @loop_carries_its_axis(%n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      scf.yield
    } {afir.axis_binding = #afir.axis_binding<axis = 7, role = loop>}
    return
  }

  // #afir.sched：两个字段合一后，「loop_axis 必须在 axis 里」由 verifier 保证，
  // 不再是每个消费端各查一遍。
  func.func private @sched_full(%arg: tensor<8xf16>) -> tensor<8xf16> attributes {
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %r = afir.load %arg {
      afir.sched = #afir.sched<axis = [0, 1, 2], loop_axis = 1>,
      afir.operand_descs = [#afir.tensor_desc<role = #afir.role<input>>],
      afir.result_descs = [#afir.tensor_desc<role = #afir.role<compute>>]
    } : tensor<8xf16> -> tensor<8xf16>
    return %r : tensor<8xf16>
  }

  // 最小形态：只有轴集合，没有最内层循环轴（未进入循环生成的算子）。
  // -1 是「未指定」的哨兵，按 ODS 默认值消隐。
  func.func private @sched_without_loop_axis(%arg: tensor<8xf16>) -> tensor<8xf16> attributes {
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %r = afir.load %arg {
      afir.sched = #afir.sched<axis = [3], loop_axis = -1>,
      afir.operand_descs = [#afir.tensor_desc<role = #afir.role<input>>],
      afir.result_descs = [#afir.tensor_desc<role = #afir.role<compute>>]
    } : tensor<8xf16> -> tensor<8xf16>
    return %r : tensor<8xf16>
  }
}

// CHECK-LABEL: func.func private @axis_binding_all_roles
// CHECK: #afir.axis_binding<axis = 1, role = axis_index>
// CHECK: #afir.axis_binding<axis = 2, role = axis_size>
// CHECK: #afir.axis_binding<axis = 3, role = loop_size>
// CHECK: #afir.axis_binding<axis = 4, role = tail_size>
// CHECK: #afir.axis_binding<axis = 5, role = actual_size>

// CHECK-LABEL: func.func private @loop_carries_its_axis
// CHECK: scf.for
// CHECK: afir.axis_binding = #afir.axis_binding<axis = 7, role = loop>

// CHECK-LABEL: func.func private @sched_full
// CHECK: afir.sched = #afir.sched<axis = [0, 1, 2], loop_axis = 1>

// 默认值消隐：显式写的 -1 在打印时消失。
// CHECK-LABEL: func.func private @sched_without_loop_axis
// CHECK: afir.sched = #afir.sched<axis = [3]>
