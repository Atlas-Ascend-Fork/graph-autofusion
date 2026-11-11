// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt %s | FileCheck %s
//
// Target ascendc IR shape for a full queue-faithful VECTOR kernel
// (load -> add -> store), reverse-engineered from the mlir-af translator
// inputs (test/Target/*-input.mlir, test/Conversion/linalg-to-ascendc.mlir).
//
// Target IR produced by queue-faithful AFIR lowering and consumed by CannTranslation
// for AscendC emission.
//
// Buffer/queue positions here are the schedule's decisions (VECIN for loads,
// VECOUT for the result) — the lowering honors them, it does not invent them.

module {
  func.func @elewise_add(
      %x: memref<?xf16, 22 : i32>,
      %y: memref<?xf16, 22 : i32>,
      %out: memref<?xf16, 22 : i32>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["N"]>
  ) attributes {ascendc.aicore, ascendc.global, ascendc.num_inputs = 2 : i32} {
    %off = arith.constant 0 : i32
    %cnt = arith.constant 32 : index
    %depth = arith.constant 1 : i32
    %bytes = arith.constant 64 : index

    %pipe = ascendc.pipe
    %qx = ascendc.queue : <vecin, 1>
    %qy = ascendc.queue : <vecin, 1>
    %qo = ascendc.queue : <vecout, 1>

    // load x: GM -> VECIN
    ascendc.pipe.init_queue %pipe, %qx, %depth, %bytes : !ascendc.queue<vecin, 1>, i32, index
    %tx = ascendc.que_bind.alloc_tensor %qx : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
    %gx = ascendc.global_tensor : !ascendc.global_tensor<32xf16>
    ascendc.global_tensor.set_global_buffer %gx, %x, %off : !ascendc.global_tensor<32xf16>, memref<?xf16, 22 : i32>, i32
    ascendc.data_copy_l2 %tx, %gx, %cnt : !ascendc.local_tensor<32xf16>, !ascendc.global_tensor<32xf16>, index
    ascendc.que_bind.enque_tensor %qx, %tx : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
    %dx = ascendc.que_bind.deque_tensor %qx : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>

    // load y: GM -> VECIN
    ascendc.pipe.init_queue %pipe, %qy, %depth, %bytes : !ascendc.queue<vecin, 1>, i32, index
    %ty = ascendc.que_bind.alloc_tensor %qy : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
    %gy = ascendc.global_tensor : !ascendc.global_tensor<32xf16>
    ascendc.global_tensor.set_global_buffer %gy, %y, %off : !ascendc.global_tensor<32xf16>, memref<?xf16, 22 : i32>, i32
    ascendc.data_copy_l2 %ty, %gy, %cnt : !ascendc.local_tensor<32xf16>, !ascendc.global_tensor<32xf16>, index
    ascendc.que_bind.enque_tensor %qy, %ty : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
    %dy = ascendc.que_bind.deque_tensor %qy : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>

    // add: VECIN x VECIN -> VECOUT
    ascendc.pipe.init_queue %pipe, %qo, %depth, %bytes : !ascendc.queue<vecout, 1>, i32, index
    %to = ascendc.que_bind.alloc_tensor %qo : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<32xf16>
    ascendc.add_l2 %to, %dx, %dy, %cnt {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, index
    ascendc.que_bind.enque_tensor %qo, %to : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<32xf16>
    %do = ascendc.que_bind.deque_tensor %qo : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<32xf16>

    // store out: VECOUT -> GM
    %go = ascendc.global_tensor : !ascendc.global_tensor<32xf16>
    ascendc.global_tensor.set_global_buffer %go, %out, %off : !ascendc.global_tensor<32xf16>, memref<?xf16, 22 : i32>, i32
    ascendc.data_copy_l2 %go, %do, %cnt : !ascendc.global_tensor<32xf16>, !ascendc.local_tensor<32xf16>, index
    ascendc.que_bind.free_tensor %qo, %do : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<32xf16>

    func.return
  }
}

// CHECK-LABEL: func.func @elewise_add
// CHECK: ascendc.pipe
// CHECK: ascendc.queue : <vecin, 1>
// CHECK: ascendc.queue : <vecout, 1>
// CHECK: ascendc.data_copy_l2 {{.*}} : !ascendc.local_tensor<32xf16>, !ascendc.global_tensor<32xf16>, index
// CHECK: ascendc.add_l2 {{.*}} {ascendc.unit = "AiCore.Vector"}
// CHECK: ascendc.data_copy_l2 {{.*}} : !ascendc.global_tensor<32xf16>, !ascendc.local_tensor<32xf16>, index
// CHECK: ascendc.que_bind.free_tensor
