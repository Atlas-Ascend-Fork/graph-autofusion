// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// descriptor 编码的等价性基线。
//
// 设计文档 5.3 定的硬纪律是「每个字段迁移必须先补 round-trip 与 negative LIT 锁住
// 等价性，再改生产代码」，而 I7 记录的现状是 descriptor 编码没有等价性测试。本文件
// 逐字段锁住 TensorDescAttr / BufferDescAttr / AxisAttr / AscGraphAttrGroupsAttr
// 的打印与解析契约。BufferDescAttr 自步 8 起挂在值的定义点（这里是 kernel 参数的
// arg attr），不再是与参数按下标对齐的数组属性，使 3.1（space/layout 进 Type）、3.3（属性收敛）的每一步改动都
// 能被 diff 出来，而不是静默丢字段。
//
// 第二条 RUN 做二次 round-trip，确认打印形式是解析的不动点。

// RUN: af-opt %s | FileCheck %s
// RUN: af-opt %s | af-opt | FileCheck %s

module {
  // 每个可选字段都取非默认值，锁住"全字段"形态。
  func.func private @desc_all_fields(%arg: tensor<8xf16> {afir.buffer_desc =
      #afir.buffer_desc<space = #afir.space<l0c>, buffer_id = 3, reuse_id = 5,
      slot_count = 2, queue_depth = 4, size_bytes_expr = #afir.expr<"(2 * s0)">,
      alignment_bytes = 64, life_time_axis_id = 1>}) -> tensor<8xf16> attributes {
    afir.function_role = #afir.function_role<kernel_body>,
    afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],
      repeats = [#afir.expr<"8">, #afir.expr<"s0">],
      strides = [#afir.expr<"1">, #afir.expr<"8">],
      vectorized_axis = [1],
      vectorized_strides = [#afir.expr<"1">],
      role = #afir.role<input>>],
    afir.asc_graph = #afir.asc_graph<tiling_key = 11,
      axes = [<id = 0, name = "a0", axis_type = Original, bind_block = true, size = #afir.expr<"s0">,
               align = #afir.expr<"32">, from = [1, 2], split_pair_other_id = 9>,
              <id = 1, name = "a1", axis_type = TileInner, size = #afir.expr<"16">>],
      type = Compute, size_var = [#afir.expr<"s0">]>
  } {
    return %arg : tensor<8xf16>
  }

  // 只给必填字段，锁住"最小"形态。BufferDescAttr 的 `space` 后面跟的是无条件逗号，
  // 因此最小形态必须写成 `<space = <gm>,>`——这是编码契约的一部分，不是笔误。
  func.func private @desc_minimal(%arg: tensor<8xf16>
      {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<gm>,>}) -> tensor<8xf16> attributes {
    afir.function_role = #afir.function_role<kernel_body>,
    afir.arg_descs = [#afir.tensor_desc<role = #afir.role<output>>],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"1">>], type = Compute>
  } {
    return %arg : tensor<8xf16>
  }

  // 默认值消隐：显式写出的默认值在打印时会被丢掉，"显式 -1"与"字段缺席"在一次
  // round-trip 之后不可区分。
  //
  // buffer_id / reuse_id / slot_count / queue_depth / alignment_bytes 的 ODS 默认值
  // 是 -1，因此显式写 -1 会被消隐；life_time_axis_id 的默认值是 int64_t::min()，
  // 因此 -1 会被保留。这个不对称是 3.3「消除重复编码」时最容易踩的坑：把扁平的
  // afir.alignment_bytes / afir.life_time_axis_id 合入 BufferDescAttr 时，必须先确认
  // 语义上"未指定"用的是哪个哨兵值，否则信息会静默丢失且不报错。
  func.func private @desc_defaults_elided(%arg: tensor<8xf16> {afir.buffer_desc =
      #afir.buffer_desc<space = #afir.space<ub>, buffer_id = -1, reuse_id = -1,
      slot_count = -1, queue_depth = -1, alignment_bytes = -1, life_time_axis_id = -1>})
      -> tensor<8xf16> attributes {
    afir.function_role = #afir.function_role<kernel_body>,
    afir.arg_descs = [#afir.tensor_desc<role = #afir.role<input>>]
  } {
    return %arg : tensor<8xf16>
  }

  // 每个 space 取值各出现一次，锁住 #afir.space 的全枚举。3.1 把 space 迁到 memref
  // memory space 时，这一组是新旧形态的对照基准。
  func.func private @desc_all_spaces(
      %gm: tensor<8xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<gm>,>},
      %ub: tensor<8xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<ub>,>},
      %l1: tensor<8xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<l1>,>},
      %l0a: tensor<8xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<l0a>,>},
      %l0b: tensor<8xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<l0b>,>},
      %l0c: tensor<8xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<l0c>,>})
      -> tensor<8xf16> attributes {
    afir.function_role = #afir.function_role<kernel_body>,
    afir.arg_descs = [#afir.tensor_desc<role = #afir.role<input>>,
                      #afir.tensor_desc<role = #afir.role<input>>,
                      #afir.tensor_desc<role = #afir.role<input>>,
                      #afir.tensor_desc<role = #afir.role<input>>,
                      #afir.tensor_desc<role = #afir.role<input>>,
                      #afir.tensor_desc<role = #afir.role<output>>]
  } {
    return %gm : tensor<8xf16>
  }
}

// 全字段形态逐字段打印，顺序与 ODS assemblyFormat 一致。
// CHECK-LABEL: func.func private @desc_all_fields
// CHECK-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <l0c>, buffer_id = 3, reuse_id = 5, slot_count = 2, queue_depth = 4, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 64, life_time_axis_id = 1>}
// CHECK-SAME: afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"8">, <"s0">],strides = [<"1">, <"8">],vectorized_axis = [1],vectorized_strides = [<"1">],role = <input>>]
// CHECK-SAME: afir.asc_graph = #afir.asc_graph<tiling_key = 11,axes = [<id = 0, name = "a0", axis_type = Original, bind_block = true, size = <"s0">, align = <"32">, from = [1, 2], split_pair_other_id = 9>, <id = 1, name = "a1", axis_type = TileInner, size = <"16">>], type = Compute, size_var = [<"s0">]>

// 最小形态：BufferDescAttr 保留 space 后的逗号，AxisAttr 省略全部可选字段。
// CHECK-LABEL: func.func private @desc_minimal
// CHECK-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <gm>,>}
// CHECK-SAME: afir.arg_descs = [#afir.tensor_desc<role = <output>>]
// CHECK-SAME: afir.asc_graph = #afir.asc_graph<axes = [<id = 0, name = "z", axis_type = Original, size = <"1">>], type = Compute>

// 默认值消隐：五个显式 -1 全部消失，只有哨兵值不同的 life_time_axis_id 留下。
// CHECK-LABEL: func.func private @desc_defaults_elided
// CHECK-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <ub>, life_time_axis_id = -1>}
// CHECK-SAME: afir.arg_descs = [#afir.tensor_desc<role = <input>>]

// space 全枚举。
// CHECK-LABEL: func.func private @desc_all_spaces
// CHECK-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <gm>,>}
// CHECK-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <ub>,>}
// CHECK-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <l1>,>}
// CHECK-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <l0a>,>}
// CHECK-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <l0b>,>}
// CHECK-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <l0c>,>}
