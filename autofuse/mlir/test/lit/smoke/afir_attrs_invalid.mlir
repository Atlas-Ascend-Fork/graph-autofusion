// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --verify-diagnostics --split-input-file %s

module {
  func.func @invalid_tensor() attributes {
    // expected-error @+1 {{DescriptorCountMismatch: axis and repeats must have the same length}}
    afir.result_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"2">], strides = [#afir.expr<"1">], role = #afir.role<compute>>]
  } {
    return
  }
}

// -----

module {
  func.func @result_descriptor_count_mismatch(%input: tensor<4xf16>) -> tensor<4xf16> {
    // expected-error @+1 {{DescriptorCountMismatch: 'afir.result_descs' expected 1 element but found 0}}
    %result = afir.load %input {afir.result_descs = []} : tensor<4xf16> -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
}

// -----

module {
  func.func @operand_descriptor_count_mismatch(%input: tensor<4xf16>) -> tensor<4xf16> {
    // expected-error @+1 {{DescriptorCountMismatch: 'afir.operand_descs' expected 1 element but found 2}}
    %result = afir.load %input {
      afir.operand_descs = [#afir.tensor_desc<role = #afir.role<input>>,
                            #afir.tensor_desc<role = #afir.role<input>>]
    } : tensor<4xf16> -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
}

// -----

module {
  // expected-error @+1 {{DescriptorCountMismatch: 'afir.arg_descs' expected 1 element but found 0}}
  func.func @argument_descriptor_count_mismatch(%input: tensor<4xf16>) attributes {
    afir.arg_descs = []
  } {
    return
  }
}

// -----

// 规划结果挂在值的定义点上，kernel 参数的定义点就是 arg attr（设计文档 5.1 步 8）。
module {
  // expected-error @+1 {{DescriptorTypeMismatch: 'afir.buffer_desc' at argument 0 must be a #afir.buffer_desc}}
  func.func @argument_buffer_descriptor_type_mismatch(%input: tensor<4xf16> {afir.buffer_desc = #afir.tensor_desc<role = #afir.role<input>>}) {
    return
  }
}

// -----

module {
  func.func @invalid_repeat_stride_count() attributes {
    // expected-error @+1 {{DescriptorCountMismatch: repeats and strides must have the same length}}
    afir.result_descs = [#afir.tensor_desc<repeats = [#afir.expr<"2">],
      strides = [#afir.expr<"1">, #afir.expr<"1">], role = #afir.role<compute>>]
  } {
    return
  }
}

// -----

module {
  func.func @invalid_vectorized_count() attributes {
    // expected-error @+1 {{DescriptorCountMismatch: vectorized_axis and vectorized_strides must have the same length}}
    afir.result_descs = [#afir.tensor_desc<vectorized_axis = [0, 1],
      vectorized_strides = [#afir.expr<"1">], role = #afir.role<compute>>]
  } {
    return
  }
}

// -----

module {
  func.func @invalid_vectorized_strides_without_axis() attributes {
    // expected-error @+1 {{DescriptorCountMismatch: vectorized_strides requires vectorized_axis}}
    afir.result_descs = [#afir.tensor_desc<vectorized_strides = [#afir.expr<"1">],
      role = #afir.role<compute>>]
  } {
    return
  }
}

// -----

module {
  func.func @invalid_negative_axis() attributes {
    // expected-error @+1 {{InvalidDescriptor: axis entries must be non-negative}}
    afir.result_descs = [#afir.tensor_desc<axis = [-1], role = #afir.role<compute>>]
  } {
    return
  }
}

// -----

module {
  func.func @invalid_negative_vectorized_axis() attributes {
    // expected-error @+1 {{InvalidDescriptor: vectorized_axis entries must be non-negative}}
    afir.result_descs = [#afir.tensor_desc<vectorized_axis = [-1], role = #afir.role<compute>>]
  } {
    return
  }
}

// -----

module {
  // expected-error @+1 {{DescriptorTypeMismatch: 'afir.result_descs' must be an ArrayAttr}}
  func.func @tensor_desc_not_array() attributes {
    afir.result_descs = #afir.tensor_desc<role = #afir.role<compute>>
  } {
    return
  }
}

// -----

module {
  // expected-error @+1 {{DescriptorTypeMismatch: 'afir.result_descs' element 0 has an unexpected attribute type}}
  func.func @tensor_desc_element_type_mismatch() attributes {
    afir.result_descs = [#afir.buffer_desc<space = #afir.space<gm>,>]
  } {
    return
  }
}

// -----

module {
  // expected-error @+1 {{DescriptorTypeMismatch: 'afir.buffer_desc' at result 0 must be a #afir.buffer_desc}}
  func.func @result_buffer_descriptor_type_mismatch(%input: tensor<4xf16>) -> (tensor<4xf16> {afir.buffer_desc = #afir.tensor_desc<role = #afir.role<output>>}) {
    return %input : tensor<4xf16>
  }
}

// -----

module {
  func.func @operation_buffer_descriptor_type_mismatch(%input: tensor<4xf16>) -> tensor<4xf16> {
    // expected-error @+1 {{DescriptorTypeMismatch: 'afir.buffer_desc' must be a #afir.buffer_desc}}
    %result = afir.load %input {afir.buffer_desc = #afir.tensor_desc<role = #afir.role<compute>>} : tensor<4xf16> -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
}

// -----

// 生命周期轴必须指向 asc_graph 里真实存在的轴，三种载体上都查。
module {
  // expected-error @+1 {{InvalidLifetime: 'afir.buffer_desc' at argument 0 references unknown life_time_axis_id 999}}
  func.func @argument_buffer_descriptor_unknown_lifetime(
      %input: tensor<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<gm>, life_time_axis_id = 999>}) attributes {
    afir.asc_graph = #afir.asc_graph<axes = [
      <id = 0, name = "tile", axis_type = TileInner, size = #afir.expr<"4">>
    ], type = Compute>
  } {
    return
  }
}

// -----

module {
  // expected-error @+1 {{InvalidLifetime: 'afir.buffer_desc' at result 0 references unknown life_time_axis_id 999}}
  func.func @result_buffer_descriptor_unknown_lifetime(%input: tensor<4xf16>)
      -> (tensor<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<gm>, life_time_axis_id = 999>}) attributes {
    afir.asc_graph = #afir.asc_graph<axes = [
      <id = 0, name = "tile", axis_type = TileInner, size = #afir.expr<"4">>
    ], type = Compute>
  } {
    return %input : tensor<4xf16>
  }
}

// -----

module {
  func.func @operation_buffer_descriptor_unknown_lifetime(%input: tensor<4xf16>) -> tensor<4xf16> attributes {
    afir.asc_graph = #afir.asc_graph<axes = [
      <id = 0, name = "tile", axis_type = TileInner, size = #afir.expr<"4">>
    ], type = Compute>
  } {
    // expected-error @+1 {{InvalidLifetime: 'afir.buffer_desc' references unknown life_time_axis_id 999}}
    %result = afir.load %input {
      afir.buffer_desc = #afir.buffer_desc<space = #afir.space<ub>, life_time_axis_id = 999>
    } : tensor<4xf16> -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
}

// -----

// 4 个按下标对齐的数组属性已被删除（设计文档 5.1 步 8）；写回旧名字即拒。
module {
  // expected-error @+1 {{UnsupportedOperation: unknown AFIR function attribute '"afir.result_buffer_descs"'}}
  func.func @legacy_result_buffer_descs() attributes {
    afir.result_buffer_descs = [#afir.buffer_desc<space = #afir.space<gm>,>]
  } {
    return
  }
}

// -----

module {
  // expected-error @+1 {{UnsupportedOperation: unknown AFIR function attribute '"afir.arg_buffer_descs"'}}
  func.func @legacy_arg_buffer_descs(%input: tensor<4xf16>) attributes {
    afir.arg_buffer_descs = [#afir.buffer_desc<space = #afir.space<gm>,>]
  } {
    return
  }
}

// -----

module {
  func.func @legacy_operand_buffer_descs(%input: tensor<4xf16>) -> tensor<4xf16> {
    // expected-error @+1 {{UnsupportedOperation: unknown AFIR operation attribute '"afir.operand_buffer_descs"'}}
    %result = afir.load %input {
      afir.operand_buffer_descs = [#afir.buffer_desc<space = #afir.space<gm>,>]
    } : tensor<4xf16> -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
}

// -----

module {
  func.func @legacy_destination_buffer_descs(%input: tensor<4xf16>) -> tensor<4xf16> {
    // expected-error @+1 {{UnsupportedOperation: unknown AFIR operation attribute '"afir.destination_buffer_descs"'}}
    %result = afir.load %input {
      afir.destination_buffer_descs = [#afir.buffer_desc<space = #afir.space<ub>,>]
    } : tensor<4xf16> -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
}

// -----

module {
  // expected-error @+1 {{UnsupportedOperation: unknown AFIR function attribute '"afir.tmp_buffers"'}}
  func.func @legacy_tmp_buffers() attributes {
    afir.tmp_buffers = []
  }
}

// -----

module {
  func.func @invalid_slot_count() attributes {
    // expected-error @+1 {{InvalidBinding: slot_count must be -1 or at least 1}}
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<ub>, buffer_id = -1, reuse_id = -1, slot_count = 0,
      queue_depth = -1, alignment_bytes = -1, life_time_axis_id = -1>]
  } {
    return
  }
}

// -----

module {
  func.func @invalid_negative_buffer_id() attributes {
    // expected-error @+1 {{InvalidBinding: buffer_id must be -1 or non-negative}}
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<gm>, buffer_id = -2,>]
  } {
    return
  }
}

// -----

module {
  func.func @invalid_negative_reuse_id() attributes {
    // expected-error @+1 {{InvalidBinding: reuse_id must be -1 or non-negative}}
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<gm>, reuse_id = -2,>]
  } {
    return
  }
}

// -----

module {
  func.func @invalid_queue_depth() attributes {
    // expected-error @+1 {{InvalidBinding: queue_depth must be -1 or at least 1}}
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<gm>, queue_depth = 0,>]
  } {
    return
  }
}

// -----

module {
  func.func @invalid_alignment() attributes {
    // expected-error @+1 {{InvalidBinding: alignment_bytes must be -1 or at least 1}}
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<gm>, alignment_bytes = 0,>]
  } {
    return
  }
}

// -----

module {
  func.func @invalid_lifetime_axis_id() attributes {
    // Values below -1 are rejected by the common binding-range check first.
    // expected-error @+1 {{InvalidBinding: life_time_axis_id must be -1 or non-negative}}
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<gm>, life_time_axis_id = -2>]
  } {
    return
  }
}

// -----

module {
  // expected-error @+1 {{InvalidLifetime: 'afir.workspace_plan' element 0 references unknown life_time_axis_id 999}}
  func.func @unknown_lifetime_axis_id() attributes {
    afir.asc_graph = #afir.asc_graph<axes = [
      <id = 0, name = "tile", axis_type = TileInner, size = #afir.expr<"4">>
    ], type = Compute>,
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<gm>, life_time_axis_id = 999>]
  } {
    return
  }
}

// -----

module {
  // expected-error @+1 {{InvalidLifetime: 'afir.workspace_plan' element 0 requires life_time_axis_id}}
  func.func @missing_workspace_lifetime() attributes {
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<gm>,
      size_bytes_expr = #afir.expr<"256">, alignment_bytes = 16,>]
  } {
    return
  }
}

// -----

module {
  func.func @workspace_plan_on_operation(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    // expected-error @+1 {{InvalidDescriptor: 'afir.workspace_plan' is only valid on func.func}}
    %0 = afir.load %arg0 {afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<gm>,>]} : tensor<1xf32> -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// -----

// 尺寸的两个来源互斥，避免同时声明后出现不一致。
module {
  func.func @size_source_conflict() attributes {
    // expected-error @+1 {{InvalidBinding: size_from_resource and size_bytes_expr are mutually exclusive}}
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<ub>, buffer_id = 2,
      size_bytes_expr = #afir.expr<"(2 * s0)">, size_from_resource = true, life_time_axis_id = -1>]
  } {
    return
  }
}

// -----

// 引用的目标是 descriptor 自己的 buffer_id，没有 id 就拼不出 `q<id>_size`。
module {
  func.func @size_from_resource_without_id() attributes {
    // expected-error @+1 {{InvalidBinding: size_from_resource requires buffer_id}}
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<ub>, size_from_resource = true,
      life_time_axis_id = -1>]
  } {
    return
  }
}

// -----

// 只有片上资源在 TilingData 里按 id 存了尺寸字段；GM 的大小是算出来的。
module {
  func.func @size_from_resource_on_gm() attributes {
    // expected-error @+1 {{InvalidBinding: size_from_resource is only valid for UB resources, got gm}}
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<gm>, buffer_id = 2, size_from_resource = true,
      life_time_axis_id = -1>]
  } {
    return
  }
}

// -----

// 拒绝在 size_bytes_expr 中手写 canonical 资源名，避免 buffer_id=2 却引用 q7_size。
// 资源尺寸应通过 descriptor 自身身份引用，在属性构造时检查。
module {
  func.func @resource_size_name_as_expression() attributes {
    // expected-error @+1 {{InvalidBinding: 'q7_size' names a canonical resource size field; use size_from_resource instead}}
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<ub>, buffer_id = 2,
      size_bytes_expr = #afir.expr<"q7_size">, life_time_axis_id = -1>]
  } {
    return
  }
}

// -----

// workspace 是 host 分配并传入的 GM 临时存储；ATT 为其设置 GM 并清除 buf/que id。
// 片上临时量应使用带 descriptor 的 memref.alloc，不属于 workspace 表。
// 拒绝片上空间条目，避免为不存在的资源生成 b2_size 等 TilingData 字段。
module {
  // 这条由 func.func 的属性 verifier 输出，所以诊断挂在 Op 上而不是属性上。
  // expected-error @+1 {{InvalidDescriptor: 'afir.workspace_plan' element 0 requires gm storage, got ub}}
  func.func @workspace_plan_requires_gm() attributes {
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<ub>, buffer_id = 2,
      size_bytes_expr = #afir.expr<"256">, life_time_axis_id = -1>]
  } {
    return
  }
}
