// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `#afir.layout` 进 memref layout 的 round-trip（设计文档 5.1 步 6、3.1、ADR-16）。
//
// 迁移前 `MemRefType::get(shape, elem)` 既无 space 也无 layout（D8）：NZ / fractal
// 在 IR 里根本表达不了，这是 Cube 路线的根因阻塞。本用例锁住布局进入类型后的序列化形态。
//
// 布局按正交分解表达，而不是 `nd|nz|dn|fractal` 扁平枚举：
//   outer   基块网格（未分块时即元素）的遍历序
//   tiling  基块内部的遍历序，none 表示不分块
//   tile    基块在最后两维上的长度，单位是元素
// 四种 (outer, tiling) 组合与昇腾两字母格式名一一对应，见下方各 case 的注释。
//
// 布局只作用于**最后两维**：@stage_zn 的第 0 维是双缓冲的 stage 维，原样透传。

// RUN: af-opt %s | af-opt | FileCheck %s

// ND：连续行优先。等价于不写 layout——`isIdentity()` 对两者都为真，判「是不是连续 ND」
// 要用它，而不是用「有没有 LayoutAttr」。
func.func private @nd(%x: memref<64x32xf16, #afir.layout<outer = row_major, rank = 2>>)

// DN：转置，最后两维互换。
func.func private @dn(%x: memref<64x32xf16, #afir.layout<outer = col_major, rank = 2>>)

// FRACTAL_NZ：外层按列（N）、块内按行（Z）。fp16 的基块是 16x16 = 512 字节。
func.func private @nz(%x: memref<64x32xf16, #afir.layout<outer = col_major, tiling = row_major, tile = [16, 16], rank = 2>>)

// FRACTAL_ZN：外层按行（Z）、块内按列（N）。与上一条只差两维取值，正交分解的价值就在这里。
func.func private @zn(%x: memref<64x32xf16, #afir.layout<outer = row_major, tiling = col_major, tile = [16, 16], rank = 2>>)

// 前导维透传：stage 维不参与矩阵解释，rank 记的是 memref 的完整 rank。
func.func private @stage_zn(%x: memref<2x64x32xf16, #afir.layout<outer = col_major, tiling = row_major, tile = [16, 16], rank = 3>>)

// 基块尺寸是参数而不是枚举值：int8 的 16x32 与 fp16 的 16x16 都是 512 字节基块，
// 跨代变化时不必新增枚举项（ADR-16）。
func.func private @nz_i8(%x: memref<64x64xi8, #afir.layout<outer = col_major, tiling = row_major, tile = [16, 32], rank = 2>>)

// 布局与地址空间正交，同一个 memref 上各占各的扩展点（步 3 + 步 6）。
func.func private @nz_on_l0a(%x: memref<64x32xf16, #afir.layout<outer = col_major, tiling = row_major, tile = [16, 16], rank = 2>, #afir.space<l0a>>)

// CHECK: func.func private @nd(memref<64x32xf16, #afir.layout<outer = row_major, rank = 2>>)
// CHECK: func.func private @dn(memref<64x32xf16, #afir.layout<outer = col_major, rank = 2>>)
// CHECK: func.func private @nz(memref<64x32xf16, #afir.layout<outer = col_major, tiling = row_major, tile = [16, 16], rank = 2>>)
// CHECK: func.func private @zn(memref<64x32xf16, #afir.layout<outer = row_major, tiling = col_major, tile = [16, 16], rank = 2>>)
// CHECK: func.func private @stage_zn(memref<2x64x32xf16, #afir.layout<outer = col_major, tiling = row_major, tile = [16, 16], rank = 3>>)
// CHECK: func.func private @nz_i8(memref<64x64xi8, #afir.layout<outer = col_major, tiling = row_major, tile = [16, 32], rank = 2>>)
// CHECK: func.func private @nz_on_l0a(memref<64x32xf16, #afir.layout<outer = col_major, tiling = row_major, tile = [16, 16], rank = 2>, #afir.space<l0a>>)
