// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 同步族的 round-trip：`!afir.token` 与 `afir.event_record` / `afir.event_wait` /
// `afir.pipe_barrier`（设计文档 3.6「流水与同步」、ADR-19）。
//
// `!afir.token` 是 AFIR 的第一个自有类型。payload 类型全部复用标准方言（约束 6），
// 自有类型只用来表达标准方言表达不了的昇腾语义——跨 pipe 的顺序边就是这样一件事。
//
// 两条 RUN 缺一不可：第一条只验打印，第二条才是 round-trip。

// RUN: af-opt %s | FileCheck %s
// RUN: af-opt %s | af-opt | FileCheck %s

module {
  // 跨 pipe 的一条边：搬入完成之后计算才能开始。
  // CHECK-LABEL: func.func @transfer_in_to_vector
  // CHECK: %[[T:.*]] = afir.event_record : <from = transfer_in, to = vector>
  // CHECK: afir.event_wait %[[T]] : <from = transfer_in, to = vector>
  func.func @transfer_in_to_vector() {
    %t = afir.event_record : !afir.token<from = transfer_in, to = vector>
    afir.event_wait %t : !afir.token<from = transfer_in, to = vector>
    return
  }

  // 反方向：计算写完之后才能搬出。方向由类型本身携带，不靠 Op 的名字或位置区分。
  // CHECK-LABEL: func.func @vector_to_transfer_out
  // CHECK: afir.event_record : <from = vector, to = transfer_out>
  func.func @vector_to_transfer_out() {
    %t = afir.event_record : !afir.token<from = vector, to = transfer_out>
    afir.event_wait %t : !afir.token<from = vector, to = transfer_out>
    return
  }

  // Cube 路线的那条链：`GM -> L1` 之后是 `L1 -> L0A/L0B`（另一台搬运引擎），然后是
  // 乘加，再是结果的后处理与搬出。四台引擎彼此异步，所以是四条边而不是一条。
  // CHECK-LABEL: func.func @cube_chain
  // CHECK: afir.event_record : <from = transfer_in, to = transfer_local>
  // CHECK: afir.event_record : <from = transfer_local, to = cube>
  // CHECK: afir.event_record : <from = cube, to = fixpipe>
  // CHECK: afir.event_record : <from = fixpipe, to = transfer_out>
  func.func @cube_chain() {
    %l1 = afir.event_record : !afir.token<from = transfer_in, to = transfer_local>
    afir.event_wait %l1 : !afir.token<from = transfer_in, to = transfer_local>
    %mad = afir.event_record : !afir.token<from = transfer_local, to = cube>
    afir.event_wait %mad : !afir.token<from = transfer_local, to = cube>
    %fix = afir.event_record : !afir.token<from = cube, to = fixpipe>
    afir.event_wait %fix : !afir.token<from = cube, to = fixpipe>
    %drain = afir.event_record : !afir.token<from = fixpipe, to = transfer_out>
    afir.event_wait %drain : !afir.token<from = fixpipe, to = transfer_out>
    return
  }

  // 标量引擎也是独立的一台：它算出的地址或标量参数，矢量要等。
  // CHECK-LABEL: func.func @scalar_to_vector
  // CHECK: afir.event_record : <from = scalar, to = vector>
  func.func @scalar_to_vector() {
    %t = afir.event_record : !afir.token<from = scalar, to = vector>
    afir.event_wait %t : !afir.token<from = scalar, to = vector>
    return
  }

  // 单 pipe 全序点。七台引擎各有一个。
  // CHECK-LABEL: func.func @barriers
  // CHECK: afir.pipe_barrier transfer_in
  // CHECK: afir.pipe_barrier transfer_out
  // CHECK: afir.pipe_barrier vector
  // CHECK: afir.pipe_barrier transfer_local
  // CHECK: afir.pipe_barrier cube
  // CHECK: afir.pipe_barrier fixpipe
  // CHECK: afir.pipe_barrier scalar
  func.func @barriers() {
    afir.pipe_barrier transfer_in
    afir.pipe_barrier transfer_out
    afir.pipe_barrier vector
    afir.pipe_barrier transfer_local
    afir.pipe_barrier cube
    afir.pipe_barrier fixpipe
    afir.pipe_barrier scalar
    return
  }

  // 循环携带的 token：第 i+1 次迭代的 wait 等的是第 i 次迭代的 record。SSA 里 wait
  // 消费的是 iter_arg 而不是 record 的结果——边跨的正好是一次迭代，这是双缓冲 WAR
  // 防护的形态（见 AFIRRaceCheckTest 的正反两条用例）。
  // CHECK-LABEL: func.func @loop_carried_token
  // CHECK: scf.for {{.*}} iter_args(%[[W:.*]] = %{{.*}}) -> (!afir.token<from = transfer_out, to = transfer_in>)
  // CHECK: afir.event_wait %[[W]] : <from = transfer_out, to = transfer_in>
  // CHECK: scf.yield %{{.*}} : !afir.token<from = transfer_out, to = transfer_in>
  func.func @loop_carried_token() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.constant 8 : index
    %seed = afir.event_record : !afir.token<from = transfer_out, to = transfer_in>
    %last = scf.for %i = %c0 to %n step %c1 iter_args(%w = %seed)
        -> (!afir.token<from = transfer_out, to = transfer_in>) {
      afir.event_wait %w : !afir.token<from = transfer_out, to = transfer_in>
      %next = afir.event_record : !afir.token<from = transfer_out, to = transfer_in>
      scf.yield %next : !afir.token<from = transfer_out, to = transfer_in>
    }
    return
  }
}
