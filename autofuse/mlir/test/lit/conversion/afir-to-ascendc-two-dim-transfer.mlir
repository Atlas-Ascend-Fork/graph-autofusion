// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 二维 tile 的搬运。`afir-to-ascendc-scheduled-transfer.mlir` 钉的是同一条发射路径在一维
// 上的**退化**形态（blockCount 恒 1、两侧缝隙恒 0），本用例钉它在二维上的真实形态。
//
// 轴形态：两根原始轴各切一刀 tile，外层那根再 block 切一刀。
//
//   z0 (Original, s0)                  z1 (Original, s1)
//     ├─ tile  → z0T / z0t(z0t_size)     └─ tile → z1T / z1t(z1t_size)
//     └─ z0T 再 block → z0TB / z0Tb(z0Tb_size)
//
// tile 是 `z0t_size × z1t_size` 的一块，在 GM 里**不连续**：相邻两行隔着整整一个 s1，
// 而 tile 只占其中 z1t_size 个元素。这正是一维路径搬不了的形态——按一维发射会把行间那
// 段不属于本 tile 的数据一并搬走。
//
// 有效视图归约到两项就停：
//
//   内层 z1t: extent=z1t_size  gm=1   ub=1
//   外层 z0t: extent=z0t_size  gm=s1  ub=z1t_size
//
// 合并不成立——内层累积出的 gm stride 是 `1 × z1t_size`，而 z0t 实际是 `s1`，两者证不出
// 相等（判不出就不合并，见设计文档 4.2.2）。于是两项分别进 blockLen 与 blockCount。

// 这里逐个列 pass 而不是走 `--afir-plan-candidate-lowering-pipeline`，因为夹具里**没有
// 计算算子**：二维 tile 的 elementwise 是本期明确拒绝的（一次 `Add(dst, a, b, len)` 盖不
// 住带缝隙的两段），所以只能 Load 直接接 Store，而那条 MTE2 -> MTE3 的边同步 pass 还不
// 支持。同步与寻址无关，跳过它不影响本用例要钉的东西。计算侧的拒绝本身由
// `TransferPlanTest` 与能力查询覆盖。
//
// V2201 与 V3510 各发一遍：二维发射不含任何目标相关的分支，mem 使用对齐行距，reg 使用 Compact 紧凑行距。

// RUN: af-opt --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc --target=Ascend910B1 %s | FileCheck %s --check-prefixes=IR,MEM
// RUN: af-opt --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc --target=Ascend950DT_950x %s | FileCheck %s --check-prefixes=IR,REG

#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0 * s1)">, life_time_axis_id = -1>
#ub_queue = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * z0t_size * z1t_size)">, alignment_bytes = 32, life_time_axis_id = -1>

// 完整轴表（外 -> 内）：块轴、块内轴、A 的 tile 内轴、R 侧的 tile 外轴、tile 内轴。
// `vectorized_axis` 只有 tile 内的那两根，差集 [z0TB, z0Tb, z1T] 就是 tile 外面的三层。
#gm_desc = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">],
  strides = [#afir.expr<"(s1 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * z0t_size)">, #afir.expr<"s1">,
             #afir.expr<"z1t_size">, #afir.expr<"1">],
  vectorized_axis = [2, 7], vectorized_strides = [#afir.expr<"z1t_size">, #afir.expr<"1">],
  role = <input>>
#gm_out_desc = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">],
  strides = [#afir.expr<"(s1 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * z0t_size)">, #afir.expr<"s1">,
             #afir.expr<"z1t_size">, #afir.expr<"1">],
  vectorized_axis = [2, 7], vectorized_strides = [#afir.expr<"z1t_size">, #afir.expr<"1">],
  role = <output>>
#ub_desc = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">],
  strides = [#afir.expr<"(s1 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * z0t_size)">, #afir.expr<"s1">,
             #afir.expr<"z1t_size">, #afir.expr<"1">],
  vectorized_axis = [2, 7], vectorized_strides = [#afir.expr<"z1t_size">, #afir.expr<"1">],
  role = <compute>>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  // 没有计算算子：二维 tile 的 elementwise 在本期是明确拒绝的（一次 `Add(dst, a, b, len)`
  // 盖不住带缝隙的两段），所以这里只有 Load 与 Store。
  func.func private @stage0_template0(
      %input: memref<2x4x8x4x16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.buffer_desc = #gm_buffer},
      %output: memref<2x4x8x4x16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.buffer_desc = #gm_buffer}) attributes {
    afir.arg_descs = [#gm_desc, #gm_out_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 0,
      axes = [
        <id = 0, name = "z0", axis_type = Original, size = #afir.expr<"s0">>,
        <id = 1, name = "z0T", axis_type = TileOuter, size = #afir.expr<"Ceiling((s0 / (z0t_size)))">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 2>,
        <id = 2, name = "z0t", axis_type = TileInner, size = #afir.expr<"z0t_size">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 1>,
        <id = 3, name = "z0TB", axis_type = BlockOuter, size = #afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">,
          align = #afir.expr<"1">, from = [1], split_pair_other_id = 4>,
        <id = 4, name = "z0Tb", axis_type = BlockInner, size = #afir.expr<"z0Tb_size">,
          align = #afir.expr<"1">, from = [1], split_pair_other_id = 3>,
        <id = 5, name = "z1", axis_type = Original, size = #afir.expr<"s1">>,
        <id = 6, name = "z1T", axis_type = TileOuter, size = #afir.expr<"Ceiling((s1 / (z1t_size)))">,
          align = #afir.expr<"1">, from = [5], split_pair_other_id = 7>,
        <id = 7, name = "z1t", axis_type = TileInner, size = #afir.expr<"z1t_size">,
          align = #afir.expr<"1">, from = [5], split_pair_other_id = 6>],
      type = Compute,
      size_var = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"z0t_size">, #afir.expr<"z0Tb_size">,
                  #afir.expr<"z1t_size">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %tile = memref.alloc() {afir.buffer_desc = #ub_queue} : memref<2x4x8x4x16xf16>
    %loaded = afir.load %input outs(%tile) {
      afir.destination_descs = [#ub_desc],
      afir.operand_descs = [#gm_desc],
      afir.result_descs = [#ub_desc],
      afir.sched = #afir.sched<axis = [3, 4, 6], loop_axis = 6>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]
    } : memref<2x4x8x4x16xf16>, memref<2x4x8x4x16xf16> -> memref<2x4x8x4x16xf16>
    afir.store %loaded outs(%output) {
      afir.destination_descs = [#gm_out_desc],
      afir.operand_descs = [#ub_desc],
      afir.sched = #afir.sched<axis = [3, 4, 6], loop_axis = 6>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]
    } : memref<2x4x8x4x16xf16>, memref<2x4x8x4x16xf16>
    return
  }
}

// IR-LABEL: func.func private @stage0_template0

// 尺寸序言：五个 size_var 按 `size_var` 的顺序连续物化，逐条 NEXT 锚定。
//
// 字段名后面那个 `:` 不能省：TilingData 的**类型签名里也列着全部字段名**，
// `emitasc.member %x "s1"` 会先撞上 `... "s0" : !emitasc.py_struct<..., ["block_dim",
// "s0", "s1", ...]>` 那一行，于是 S1 捕获到 s0 的值。加上 `:` 才只匹配取成员那一处。
// IR: emitasc.member %{{.*}} "s0" :
// IR-NEXT: arith.index_cast
// IR-NEXT: emitasc.member %{{.*}} "s1" :
// IR-NEXT: %[[S1:.*]] = arith.index_cast
// IR-NEXT: emitasc.member %{{.*}} "z0t_size" :
// IR-NEXT: %[[Z0T:.*]] = arith.index_cast
// IR-NEXT: emitasc.member %{{.*}} "z0Tb_size" :
// IR-NEXT: %[[Z0TB:.*]] = arith.index_cast
// IR-NEXT: emitasc.member %{{.*}} "z1t_size" :
// IR-NEXT: %[[Z1T:.*]] = arith.index_cast

// 块轴不发循环，块号是 `get_block_idx` 对该轴 loop_size 取模。
// IR: %[[BLOCKIDX:.*]] = arith.remsi {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 3, role = axis_index>}
// 两层循环：block-inner（z0Tb）在外，R 侧的 tile-outer（z1T）在内。
// IR: scf.for %[[ROUND:.*]] = %{{.*}} to %{{.*}}
// A 侧 tile 内轴的 actual_size：block-inner 循环体里反推出 z0T 的序号后配对发出。它是
// 二维搬运的 blockCount——"搬几行"。
// IR: %[[BLOCKCOUNT:.*]] = arith.select {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 2, role = actual_size>}
// IR: scf.for %[[COL:.*]] = %{{.*}} to %{{.*}}
// R 侧 tile 内轴的 actual_size：z1T 自己生成循环，进循环体就为配对 inner 发一份。它是
// blockLen——"每行搬几个"。
// IR: %[[BLOCKLEN:.*]] = arith.select {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 7, role = actual_size>}

// --- Load ---
// GM 偏移三项求和，对应差集 [z0TB, z0Tb, z1T]，每项乘自己那一格 stride。一维路径给不出
// 这个形态：那时只有「块号 × 块内 tile 数 + 轮次」两层，第三层循环无处安放。
// IR: %[[SB0:.*]] = index.mul %[[S1]], %[[Z0TB]]
// IR: %[[SB:.*]] = index.mul %[[SB0]], %[[Z0T]]
// IR: %[[T0:.*]] = arith.muli %[[BLOCKIDX]], %[[SB]]
// IR: %[[SI:.*]] = index.mul %[[S1]], %[[Z0T]]
// IR: %[[T1:.*]] = arith.muli %[[ROUND]], %[[SI]]
// IR: %[[SUM0:.*]] = arith.addi %[[T0]], %[[T1]]
// IR: %[[T2:.*]] = arith.muli %[[COL]], %[[Z1T]]
// IR: %[[OFF:.*]] = arith.addi %[[SUM0]], %[[T2]]

// 两侧缝隙各按自己那一侧的口径算，**不能互换**：
//   GM 缝隙 = s1 − blockLen        行距就是张量的实际排布
//   UB 缝隙 = z1t_size − blockLen  UB 按整块 tile 铺，行距是标称值、与尾块无关
// legacy `SetDmaParams` 的 `ActualSize(gm − repeat)` 与 `Size(ub) − ActualSize(repeat)`
// 就是这个区别。写成同一个口径在整块上看不出来，只有尾块会错。
// IR: %[[GMGAP:.*]] = arith.subi %[[S1]], %[[BLOCKLEN]]
// MEM: %[[UBGAP:.*]] = arith.subi %[[Z1T]], %[[BLOCKLEN]]
// REG: %[[UBGAP:.*]] = arith.subi %[[BLOCKLEN]], %[[BLOCKLEN]]
// 方向映射：Load 是 `LocalTensor ← GlobalTensor`，所以 srcStride 取 GM 侧、dstStride 取
// UB 侧。GM 侧带偏移、UB 侧恒为 0 的规则在二维下不变。
// IR: ascendc.global_tensor.subindex %{{.*}}[%[[OFF]]]
// IR: ascendc.local_tensor.subindex %{{.*}}[%[[ZERO:.*]]]
// IR: ascendc.data_copy_pad_extend{{(_reg_base)?}} %{{.*}}, %{{.*}}, %[[BLOCKCOUNT]], %[[BLOCKLEN]], %[[GMGAP]], %[[UBGAP]]

// --- Store ---
// 同一个发射点，方向相反：src 是 UB 侧、dst 是 GM 侧，于是两个缝隙参数**交换位置**。
//
// **这一行是方向映射唯一会变红的地方**。把 `emitTransfer` 退回「不看方向、按
// global/local 原样传」之后，Load 那条仍然对（它的顺序恰好就是 global, local），只有
// 这一条会挂——变异验证的判据就是"只有二维 Store 一条失败"。
// IR: %[[SGMGAP:.*]] = arith.subi %[[S1]], %[[BLOCKLEN]]
// MEM: %[[SUBGAP:.*]] = arith.subi %[[Z1T]], %[[BLOCKLEN]]
// REG: %[[SUBGAP:.*]] = arith.subi %[[BLOCKLEN]], %[[BLOCKLEN]]
// IR: ascendc.local_tensor.subindex
// IR: ascendc.global_tensor.subindex %{{.*}}[%[[SOFF:.*]]]
// IR: ascendc.data_copy_pad_extend{{(_reg_base)?}} %{{.*}}, %{{.*}}, %[[BLOCKCOUNT]], %[[BLOCKLEN]], %[[SUBGAP]], %[[SGMGAP]]
