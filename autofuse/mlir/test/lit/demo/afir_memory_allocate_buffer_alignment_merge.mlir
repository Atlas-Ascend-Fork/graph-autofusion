// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms of
// the CANN Open Software License Version 2.0.
// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s

// 同一块非队列 UB 上的两个张量，`alignment_bytes` 一个是 32、一个是 -1：要合得上。
//
// `alignment_bytes` 不是资源的身份，是**某个张量对这块存储字节数的要求**（-1 = 没有要求），
// 多个张量共用时这块存储要同时满足它们，聚合取最小公倍数。判据在
// `afir::canonicalResourceFactsAgree` 的清单上（它不在清单里），聚合在 `registerResource`。
//
// 这种组合是真的会出现、而不是编出来的：unary 算子的 tmp buffer 由 bridge 固定带 32
// （`AfirBuilder` 的 `kTmpBufferAlignmentBytes`），而 tmp buffer 与普通计算缓冲**同在一个
// id 空间**（`ScheduleToAfir` 里 `info.buffer_ids.push_back(tmpBuffer.id)` 旁边那段），
// 于是「某个中间结果与某个 tmp buffer 落到同一个 `b<id>`」就只差这一个字段。
// 此前要求相等，这张图整张报 `InvalidBinding: conflicting descriptors for one canonical
// resource`——单输入单输出算子作分叉点时命中（探针 T5c / T5e，2026-10-12）。
//
// 钉的是 `arith.ceildivsi` 的个数：聚合取 32 之后 `alignResourceSize` 会对 b4 生成一次
// 向上取整，取成 -1（「没有要求」赢）的话这一句就不见了，而 CPU 侧数值断言看不出区别——
// 那一侧只有在合并后的 TBuf 装不下时才会炸。
//
// 所以这张图里一块队列都没有：队列成员自己也会生成一次 32 对齐
// （`materializeLocalMemberSize` 的队列分支），混进来就分不出这一句是谁生成的。
// 全是非队列缓冲之后，`alignment_bytes != -1` 全图只有 b4 一处，计数才是判据。
//
// 还有一处：b4 的字节数走符号（`s0`）而不是字面量。写 `"32"` 的话
// `arith.ceildivsi(32, 32)` 会被折叠掉，整张图一句 `ceildivsi` 也看不见——
// 于是这份用例在改动前后都「绿」，什么都没钉住。这一步是实测出来的，不是推的。
//
// 这一格钉得住什么、钉不住什么（变异实测）：
//   - 退回「要求两者相等」：这张图直接报 conflicting descriptors，红。钉住了。
//   - 聚合时把要求丢掉（取后看到的那个 -1）：`ceildivsi` 变 0，红。钉住了——
//     这是真正要命的那一种，丢掉之后合并出来的 TBuf 装不下。
//   - 聚合取「先看到的那个」：钉不住。这张图里带 32 的那份正好先注册，取第一个
//     也得 32。要分开得控制注册顺序，而那是 `values` 的遍历顺序，不该让用例依赖它。
//   - 取 max 还是取最小公倍数：钉不住，而且今天钉不了——`alignment_bytes` 实际取值
//     只有 -1 与 32 两种（32 是 bridge 的 `kTmpBufferAlignmentBytes`），两种聚合同解。
//     取最小公倍数是为了在值变多之后仍然对（48 与 32 的公共解是 96，取 48 会落空），
//     理由写在 `aggregateAlignmentBytes` 上，不是这一格能证的事。

module {
  func.func private @buffer_alignment_merge(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %in = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2,
        size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    // `abs` 的结果与 `abs` 自己的 tmp buffer 分开两块，所以这里的 b3 只有 abs 的结果。
    %dstAbs = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3,
        size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    // 同一个 b4 上的两块：tmp buffer 带 32，`neg` 的结果不带。两者各自独占 b4（tmp buffer
    // 在 `CanonicalGroupKey` 里按序号单独成组），所以这里考的是聚合而不是并排。
    %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 4,
        size_bytes_expr = #afir.expr<"s0">, alignment_bytes = 32, life_time_axis_id = -1>} : memref<32xui8>
    %dstNeg = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 4,
        size_bytes_expr = #afir.expr<"s0">, life_time_axis_id = -1>} : memref<16xf16>
    %out = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 5,
        size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    %loaded = afir.load %input outs(%in) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %valAbs = afir.abs %loaded outs(%dstAbs) tmp(%tmp) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : memref<16xf16>, memref<16xf16>, memref<32xui8> -> memref<16xf16>
    %valNeg = afir.neg %loaded outs(%dstNeg) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %sum = afir.add %valAbs, %valNeg outs(%out) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>, #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %sum outs(%input) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// CHECK-COUNT-1: arith.ceildivsi
// CHECK-NOT: arith.ceildivsi
