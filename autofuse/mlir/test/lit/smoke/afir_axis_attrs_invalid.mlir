// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 检查统一轴属性对角色拼写、轴号和调度轴集合的校验。
//
// 角色拼写错误、负轴号，以及 loop_axis 不在 sched_axis 中均应提前拒绝，
// 不能等到消费者查询不到绑定后回退或使用无效值。

// RUN: af-opt --split-input-file --verify-diagnostics %s

// 角色拼错：枚举不认识 `axis_idx`，解析期即失败。
func.func private @axis_role_typo_rejected() {
  // expected-error @+2 {{expected one of [loop, axis_index, axis_size, loop_size, tail_size, actual_size]}}
  // expected-error @+1 {{failed to parse AxisBindingAttr parameter 'role'}}
  %v = arith.constant {afir.axis_binding = #afir.axis_binding<axis = 0, role = axis_idx>} 0 : index
  return
}

// -----

// 负数轴号引用不到任何 AxisAttr。
func.func private @axis_binding_negative_axis_rejected() {
  // expected-error @+1 {{InvalidDescriptor: axis_binding axis must be non-negative}}
  %v = arith.constant {afir.axis_binding = #afir.axis_binding<axis = -1, role = axis_size>} 0 : index
  return
}

// -----

// 最内层循环轴必须出现在被调度的轴集合里，否则这条调度信息自相矛盾。
// 合一之前这只在 `LoopGenerate` 里查一次，其余读点直接信任。
func.func private @sched_loop_axis_outside_axis_rejected(%arg: tensor<8xf16>) -> tensor<8xf16> attributes {
  afir.function_role = #afir.function_role<kernel_body>
} {
  %r = afir.load %arg {
    // expected-error @+1 {{InvalidDescriptor: sched loop_axis 5 is not present in axis}}
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 5>,
    afir.operand_descs = [#afir.tensor_desc<role = #afir.role<input>>],
    afir.result_descs = [#afir.tensor_desc<role = #afir.role<compute>>]
  } : tensor<8xf16> -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----

// 负数轴号同样不合法。
func.func private @sched_negative_axis_rejected(%arg: tensor<8xf16>) -> tensor<8xf16> attributes {
  afir.function_role = #afir.function_role<kernel_body>
} {
  %r = afir.load %arg {
    // expected-error @+1 {{InvalidDescriptor: sched axis entries must be non-negative}}
    afir.sched = #afir.sched<axis = [0, -2]>,
    afir.operand_descs = [#afir.tensor_desc<role = #afir.role<input>>],
    afir.result_descs = [#afir.tensor_desc<role = #afir.role<compute>>]
  } : tensor<8xf16> -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----

// 旧扁平属性不再属于已知 AFIR 属性，重新写入时应由未知属性检查拒绝。
func.func private @old_flat_axis_attrs_rejected(%arg: tensor<8xf16>) -> tensor<8xf16> attributes {
  afir.function_role = #afir.function_role<kernel_body>
} {
  // expected-error @+1 {{UnsupportedOperation: unknown AFIR operation attribute '"afir.sched_axis"'}}
  %r = afir.load %arg {
    afir.sched_axis = array<i64: 0>,
    afir.operand_descs = [#afir.tensor_desc<role = #afir.role<input>>],
    afir.result_descs = [#afir.tensor_desc<role = #afir.role<compute>>]
  } : tensor<8xf16> -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----

func.func private @old_loop_axis_attr_rejected(%arg: tensor<8xf16>) -> tensor<8xf16> attributes {
  afir.function_role = #afir.function_role<kernel_body>
} {
  // expected-error @+1 {{UnsupportedOperation: unknown AFIR operation attribute '"afir.loop_axis"'}}
  %r = afir.load %arg {
    afir.loop_axis = 0 : i64,
    afir.operand_descs = [#afir.tensor_desc<role = #afir.role<input>>],
    afir.result_descs = [#afir.tensor_desc<role = #afir.role<compute>>]
  } : tensor<8xf16> -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----

func.func private @old_vectorized_axis_attr_rejected(%arg: tensor<8xf16>) -> tensor<8xf16> attributes {
  afir.function_role = #afir.function_role<kernel_body>
} {
  // expected-error @+1 {{UnsupportedOperation: unknown AFIR operation attribute '"afir.vectorized_axis"'}}
  %r = afir.load %arg {
    afir.vectorized_axis = array<i64: 0>,
    afir.operand_descs = [#afir.tensor_desc<role = #afir.role<input>>],
    afir.result_descs = [#afir.tensor_desc<role = #afir.role<compute>>]
  } : tensor<8xf16> -> tensor<8xf16>
  return %r : tensor<8xf16>
}
