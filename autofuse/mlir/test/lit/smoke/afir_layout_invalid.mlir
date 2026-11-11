// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `#afir.layout` 的 negative 用例（设计文档 5.1 步 6、5.3）。
//
// 分两组：
//   属性自身的不变量（`LayoutAttr::verify`）——基块与分块布局必须同时在或同时不在，
//     基块必须给两维且为正，非 identity 布局要求 rank >= 2；
//   布局与 memref 的一致性（`verifyLayout`，由 `MemRefType` 在构造期调用）——rank 必须
//     相等，基块必须整除对应维度。
//
// 布局进入类型后，MemRefType 构造时可检查 rank 和基块整除性（D8）。

// RUN: af-opt --split-input-file --verify-diagnostics %s

// 不分块却带基块尺寸：自相矛盾。
// expected-error@+1 {{afir.layout tile is only valid when tiling is not 'none'}}
func.func private @tile_without_tiling(%x: memref<64x32xf16, #afir.layout<outer = row_major, tile = [16, 16], rank = 2>>)

// -----

// 分块却不给基块尺寸：affine map 无从构造。
// expected-error@+1 {{afir.layout tile must give exactly two extents, one per tiled dimension}}
func.func private @tiling_without_tile(%x: memref<64x32xf16, #afir.layout<outer = col_major, tiling = row_major, rank = 2>>)

// -----

// 基块只给一维。布局作用于最后两维，基块也必须逐维给全。
// expected-error@+1 {{afir.layout tile must give exactly two extents, one per tiled dimension}}
func.func private @tile_rank_one(%x: memref<64x32xf16, #afir.layout<outer = col_major, tiling = row_major, tile = [16], rank = 2>>)

// -----

// expected-error@+1 {{afir.layout tile extents must be positive}}
func.func private @tile_zero(%x: memref<64x32xf16, #afir.layout<outer = col_major, tiling = row_major, tile = [16, 0], rank = 2>>)

// -----

// identity 之外的布局都在谈「最后两维」，rank 不足 2 时这句话没有意义。
// expected-error@+1 {{afir.layout requires rank >= 2 unless it is the identity layout}}
func.func private @col_major_rank_one(%x: memref<64xf16, #afir.layout<outer = col_major, rank = 1>>)

// -----

// getAffineMap 不接收 rank，因此属性需记录 rank，并在 MemRefLayoutAttrInterface 校验中核对。
// expected-error@+1 {{afir.layout rank 3 does not match memref rank 2}}
func.func private @rank_mismatch(%x: memref<64x32xf16, #afir.layout<outer = col_major, rank = 3>>)

// -----

// 基块除不尽意味着最后一块越出逻辑张量——NZ 输入需要先补齐到基块整数倍，原因就在这里。
// expected-error@+1 {{afir.layout tile extent 16 does not divide dimension 1 of size 30}}
func.func private @tile_does_not_divide(%x: memref<64x30xf16, #afir.layout<outer = col_major, tiling = row_major, tile = [16, 16], rank = 2>>)
