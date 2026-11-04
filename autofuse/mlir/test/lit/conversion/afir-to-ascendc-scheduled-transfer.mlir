// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `TransferPlan` 的**调度路径**（`TransferPlanKind::Scheduled`）此前零覆盖：全仓没有一个
// lit 产出 `GlobalTensorSubIndexOp`，带 `vectorized_axis` 的用例全停在 memory-allocate
// 或 parse/print roundtrip。于是 `emitScheduledTransfer` 里的 GM 偏移与 tile
// 长度——搬运算子唯一会算错地址的地方——改了不会变红。
//
// 本用例把那条路钉住。轴形态取的是最常见的一种：原始轴先 tile 切、TileOuter 再 block 切。
//
//   z (Original, s0)
//     ├─ tile  → zT (TileOuter)   zt (TileInner, zt_size)   ← zt 是向量化轴
//     └─ zT 再 block → zTB (BlockOuter)  zTb (BlockInner, zTb_size)  ← zTb 是 loop_axis
//
// GM 偏移是集合差求和：`axis` 表里不在 `vectorized_axis` 里的每根轴，各乘自己那一格
// `strides`。这里差集是 `[zTB, zTb]`，于是偏移 = 块号 × (zTb_size × zt_size) + 轮次 ×
// zt_size。与它替换掉的 `(块号 × zTb_size + 轮次) × zt_size` **数值相等**——旧式子是这个
// 和的 Horner 因式分解，而能因式分解依赖 `stride(zTB) = stride(zTb) × size(zTb)` 这层
// 嵌套关系。reduce 不满足那层关系，所以本用例钉的是**求和形态**，不是那次外提。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR

#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>
#ub_queue = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * zt_size)">, alignment_bytes = 32, life_time_axis_id = -1>
#ub_out_queue = #afir.buffer_desc<space = <ub>, buffer_id = 1, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * zt_size)">, alignment_bytes = 32, life_time_axis_id = -1>

// GM 侧：完整轴表（外 -> 内），三根轴各有自己的 stride。
#gm_desc = #afir.tensor_desc<
  axis = [3, 4, 2],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (zt_size))) / (zTb_size)))">, #afir.expr<"zTb_size">, #afir.expr<"zt_size">],
  strides = [#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">, #afir.expr<"1">],
  vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">],
  role = <input>>
#gm_out_desc = #afir.tensor_desc<
  axis = [3, 4, 2],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (zt_size))) / (zTb_size)))">, #afir.expr<"zTb_size">, #afir.expr<"zt_size">],
  strides = [#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">, #afir.expr<"1">],
  vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">],
  role = <output>>
// UB 侧：只有 tile 内轴。
#ub_desc = #afir.tensor_desc<
  axis = [3, 4, 2],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (zt_size))) / (zTb_size)))">, #afir.expr<"zTb_size">, #afir.expr<"zt_size">],
  strides = [#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">, #afir.expr<"1">],
  vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">],
  role = <compute>>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %input: memref<2x8x16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.buffer_desc = #gm_buffer},
      %output: memref<2x8x16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.buffer_desc = #gm_buffer}) attributes {
    afir.arg_descs = [#gm_desc, #gm_out_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 0,
      axes = [
        <id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>,
        <id = 1, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((s0 / (zt_size)))">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 2>,
        <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 1>,
        <id = 3, name = "zTB", axis_type = BlockOuter, size = #afir.expr<"Ceiling((Ceiling((s0 / (zt_size))) / (zTb_size)))">,
          align = #afir.expr<"1">, from = [1], split_pair_other_id = 4>,
        <id = 4, name = "zTb", axis_type = BlockInner, size = #afir.expr<"zTb_size">,
          align = #afir.expr<"1">, from = [1], split_pair_other_id = 3>],
      type = Compute,
      size_var = [#afir.expr<"s0">, #afir.expr<"zt_size">, #afir.expr<"zTb_size">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %tile = memref.alloc() {afir.buffer_desc = #ub_queue} : memref<2x8x16xf16>
    %result = memref.alloc() {afir.buffer_desc = #ub_out_queue} : memref<2x8x16xf16>
    %loaded = afir.load %input outs(%tile) {
      afir.destination_descs = [#ub_desc],
      afir.operand_descs = [#gm_desc],
      afir.result_descs = [#ub_desc],
      afir.sched = #afir.sched<axis = [3, 4], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
    } : memref<2x8x16xf16>, memref<2x8x16xf16> -> memref<2x8x16xf16>
    %computed = afir.add %loaded, %loaded outs(%result) {
      afir.destination_descs = [#ub_desc],
      afir.operand_descs = [#ub_desc, #ub_desc],
      afir.result_descs = [#ub_desc],
      afir.sched = #afir.sched<axis = [3, 4], loop_axis = 4>
    } : (memref<2x8x16xf16>, memref<2x8x16xf16>), memref<2x8x16xf16> -> memref<2x8x16xf16>
    afir.store %computed outs(%output) {
      afir.destination_descs = [#gm_out_desc],
      afir.operand_descs = [#ub_desc],
      afir.sched = #afir.sched<axis = [3, 4], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
    } : memref<2x8x16xf16>, memref<2x8x16xf16>
    return
  }
}

// IR-LABEL: func.func private @stage0_template0

// 尺寸序言：两个切分轴各自的 inner 尺寸。轴的 outer 尺寸是 `Ceiling(parent/inner)`
// 派生式，不是符号，所以 TilingData 里只有 inner——这条规律见设计文档 4.2.5。
// 走完整 plan pipeline 后 `afir.tiling_param` 已降成 `emitasc.member` + index_cast。
// 三个符号按 size_var 顺序连续出现；逐条 NEXT 锚定，不能只 match 字段名——TilingData
// 的**类型签名里也列着全部字段名**，单独 match `"zt_size"` 会先撞上 `s0` 那一行。
// IR: emitasc.member %{{.*}} "s0"
// IR-NEXT: arith.index_cast
// IR-NEXT: emitasc.member %{{.*}} "zt_size"
// IR-NEXT: %[[ZT:.*]] = arith.index_cast
// IR-NEXT: emitasc.member %{{.*}} "zTb_size"
// IR-NEXT: %[[ZTB:.*]] = arith.index_cast

// 分核轴不发 scf.for：块号是 `get_block_idx` 对该轴 loop_size 取模，打 axis_index 标记
// （LoopGenerate.cpp:525-531）。二维重构后这个值仍然是 GM 偏移的一项。
// IR: ascendc.get_block_idx
// IR: %[[BLOCKIDX:.*]] = arith.remsi {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 3, role = axis_index>}
// block-inner 那层才是循环，循环上界是它的 actual_size（整块/尾块二选一）。
// IR: %[[ACTUAL:.*]] = arith.select {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 4, role = actual_size>}
// IR: scf.for %[[IV:.*]] = %{{.*}} to %[[ACTUAL]]

// 向量化轴自己的 actual_size：循环生成为配对 inner 发的那一份，在循环体内。
//
// 这个值**曾经被删掉**：memory-allocate 里下降符号求值那一步顺手扫了全函数、顺手做了
// 死代码消除，而它当时没有使用者——`#afir.axis_binding` 是「从轴找回值」的唯一通路，
// 标记本身不算使用。于是「一个轴绑定能不能活到下降阶段」取决于此前碰巧有没有消费者。
// 搬运长度改成查这个绑定之后它才有了使用者。本行把它的存活钉住。
// IR: %[[TILEACTUAL:.*]] = arith.select {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 2, role = actual_size>}

// --- Load ---
// GM 偏移 = 块号 × strides[zTB] + 轮次 × strides[zTb]。两项都直接来自 descriptor 的
// `strides` 表：复合的那格物化成一次 `index.mul`，裸符号那格直接绑回 TilingData 字段。
//
// 这里**没有**外提的公因子。旧实现把两项合成 `(块号 × zTb_size + 轮次) × zt_size`，
// 靠的是 stride 的嵌套关系；求和式不作这个假设，代价是循环体内多一次乘法。
// IR: %[[STRIDE:.*]] = index.mul %[[ZTB]], %[[ZT]]
// IR: %[[BLOCKTERM:.*]] = arith.muli %[[BLOCKIDX]], %[[STRIDE]]
// IR: %[[ROUNDTERM:.*]] = arith.muli %[[IV]], %[[ZT]]
// IR: %[[OFF:.*]] = arith.addi %[[BLOCKTERM]], %[[ROUNDTERM]]
// 一维退化：有效视图只有一项，于是 blockCount 恒 1、两侧缝隙恒 0。一维与二维共用同一条
// 发射路径，这一行钉的就是「退化得对」——二维用例在
// `afir-to-ascendc-two-dim-transfer.mlir`，两者的 `data_copy_pad_extend` 是同一个发射点。
// IR: %[[ONE:.*]] = arith.constant 1 : index
// IR: %[[ZEROSTRIDE:.*]] = arith.constant 0 : index
// GM 侧带偏移、UB 侧恒为 0——`8fcd2ebe` 定下的规则，二维重构后必须保持。
// IR: ascendc.global_tensor.subindex %{{.*}}[%[[OFF]]]
// IR: ascendc.local_tensor.subindex %{{.*}}[%[[ZERO:.*]]]
// 搬运长度是上面那个 actual_size 绑定本身，不再就地重算一次尾块 select。
// IR: ascendc.data_copy_pad_extend %{{.*}}, %{{.*}}, %[[ONE]], %[[TILEACTUAL]], %[[ZEROSTRIDE]], %[[ZEROSTRIDE]]

// --- 计算 ---
// 一维分段直接取 tile 的 actual_size；二维行为由
// `afir-two-dim-compute.mlir` 单独钉住。
// IR: ascendc.add_l2 %{{.*}}, %{{.*}}, %{{.*}}, %[[TILEACTUAL]]

// --- Store ---
// 方向映射：本地侧取源、GM 侧取目的，偏移仍只落在 GM 侧。求和形态与 Load 逐项相同——
// 两侧共用一条发射路径，写错一侧的表现是 Store 搬到错的地方而 Load 正常。
// IR: %[[SSTRIDE:.*]] = index.mul %[[ZTB]], %[[ZT]]
// IR: %[[SBLOCK:.*]] = arith.muli %[[BLOCKIDX]], %[[SSTRIDE]]
// IR: %[[SROUND:.*]] = arith.muli %[[IV]], %[[ZT]]
// IR: %[[SOFF:.*]] = arith.addi %[[SBLOCK]], %[[SROUND]]
// IR: ascendc.local_tensor.subindex
// IR: ascendc.global_tensor.subindex %{{.*}}[%[[SOFF]]]
// IR: ascendc.data_copy_pad_extend %{{.*}}, %{{.*}}, %{{.*}}, %[[TILEACTUAL]]
