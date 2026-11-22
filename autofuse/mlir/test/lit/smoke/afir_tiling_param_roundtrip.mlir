// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 检查 tiling_param 的 round-trip：将符号尺寸读取为 index SSA 值，
// 供 memref.alloc 等操作使用。
//
// 两条 RUN 缺一不可：第一条只验打印，第二条才是 round-trip。

// RUN: af-opt %s | FileCheck %s
// RUN: af-opt %s | af-opt | FileCheck %s

module {
  // 名字是符号名，不是序号——`q0_size` 同时是 host 侧 TilingData 的字段名。
  // CHECK-LABEL: func.func @named_params
  // CHECK: %{{.*}} = afir.tiling_param "q0_size" : index
  // CHECK: %{{.*}} = afir.tiling_param "b5_size" : index
  // CHECK: %{{.*}} = afir.tiling_param "s0" : index
  func.func @named_params() {
    %queue = afir.tiling_param "q0_size" : index
    %buffer = afir.tiling_param "b5_size" : index
    %shape = afir.tiling_param "s0" : index
    return
  }

  // 动态尺寸的缓冲——这是本 Op 存在的理由，而不是一个额外用法。
  // CHECK-LABEL: func.func @dynamic_alloc
  // CHECK: %[[N:.*]] = afir.tiling_param "q0_size" : index
  // CHECK: memref.alloc(%[[N]]) : memref<?xf16, #afir.space<ub>>
  func.func @dynamic_alloc() {
    %n = afir.tiling_param "q0_size" : index
    %buf = memref.alloc(%n) : memref<?xf16, #afir.space<ub>>
    memref.dealloc %buf : memref<?xf16, #afir.space<ub>>
    return
  }
}
