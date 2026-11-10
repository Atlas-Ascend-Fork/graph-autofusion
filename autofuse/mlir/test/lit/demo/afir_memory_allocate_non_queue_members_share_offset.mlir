// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms of
// the CANN Open Software License Version 2.0.
// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s

// 非队列资源上的两个成员不并排：各自从自己的 `view.offset` 起算，都是 0。
//
// 「一个 buffer 一份存储，成员大小与资源大小是同一个数」是 `materializeLocalMemberSize`
// 非队列分支本来就写着的意思；`reuse_id = -1` 的两个成员偏移都是 0，说的是「在时间上
// 复用这块存储」，legacy 对 `add(x, y) -> 其中一个的 buffer` 这种形状生成的就是一次原地计算。
//
// 此前 `materializeLocalMember` 对两种资源一视同仁地累加偏移。队列上是对的（累加就是摆
// slot），非队列上则要求 `b<id>_size` 是成员字节数之和，而那个数是 ATT 按
// `max(同 id 的每个张量) ⊔ Σ(同 reuse 组)` 给的，于是第二个成员一定指到资源外面。
// 失败方式不是报错，是运行期 `LocalTensor::operator[]` abort——CPU 模拟器上的原话是
// 「offset is 32, which can not be larger than data len 32」，而且输出一个字节都没写。
// 所以这一格钉的是偏移，不是「能不能编过」：编得过而偏移错的那一版，这份 lit 照样绿不了，
// 但任何只看 `init_buffer` 的断言都抓不住它。
//
// 真实场景是广播作分叉点（探针 T5，E2E `load_broadcast_diamond_store`）：分叉点的结果与
// `add` 的结果都落到同一个非队列 `b<id>` 且都不带 reuse_id。
//
// 钉法：两个成员的 `reinterpret_cast` 吃的是同一个 `tbuf.get_tensor` 结果。偏移 0 的
// `sub_index` 会被折叠掉，所以正确的那一版看不到 `sub_index`；累加的那一版第二个成员是
// `reinterpret_cast` 一个 `sub_index %raw, %c32`，捕获变量对不上就红。
// 同时钉 `init_buffer` 的字节数是 32（一份存储），不是两个成员之和。

module {
  func.func private @non_queue_two_members(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %in = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 1, queue_depth = 2,
        size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    // 同一个非队列 b3 上的两块，descriptor 逐字相同、都不带 reuse_id：于是同组、并排。
    %dstAbs = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3,
        size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    %dstNeg = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3,
        size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    %out = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 5, slot_count = 1, queue_depth = 2,
        size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    %loaded = afir.load %input outs(%in) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %valNeg = afir.neg %loaded outs(%dstNeg) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %sum = afir.add %loaded, %valNeg outs(%dstAbs) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>, #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    %doubled = afir.add %sum, %valNeg outs(%out) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>, #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %doubled outs(%input) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// CHECK: %[[SIZE:.*]] = arith.constant 32 : index
// CHECK: ascendc.pipe.init_buffer %{{.*}}, %{{.*}}, %[[SIZE]]
// CHECK: %[[RAW:.*]] = ascendc.tbuf.get_tensor
// CHECK: ascendc.reinterpret_cast %[[RAW]] {{.*}}buffer_id = 3
// CHECK: ascendc.reinterpret_cast %[[RAW]] {{.*}}buffer_id = 3
