// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.lowered.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=fused_graph %t.lowered.mlir | FileCheck %s

// AscGraph 的 `ScalarData`：主机在运行期按值传进来的标量，在 kernel 边界上是一个
// **非 memref 的形参**。与 `afir-scalar-binary-api.mlir` 的差别不在计算——发出来的调用
// 是同一个 `Adds`——而在**参数表**：那份的标量是函数体里的 `arith.constant`，这份的标量
// 是第 1 号 kernel 形参。
//
// 钉三件事，三件各自能单独错：
//
//   1. 签名物化不把它改成 memref。物化前这里是 `cast<MemRefType>`，遇到标量是 assert
//      失败而不是诊断；
//   2. **它按 index 排在两个指针形参中间，不是排到末尾。** `KernelArgumentKind::ScalarInput`
//      的枚举值追加在末尾（ABI 兼容要求），所以排序不折秩的话它会落到 output / workspace
//      之后；
//   3. host 包装器两处都分流：签名里 `half input1` 而不是 `void* input1`，调用处传
//      `input1` 而不是 `(half)input1`。两处写法抄自 legacy 的
//      `GetScheduledResultInputOutput`。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#scalar_desc = #afir.tensor_desc<role = <input>>
#output_desc = #afir.tensor_desc<axis = [0], role = <output>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>
#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>
#load_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>
#output_queue = #afir.buffer_desc<space = <ub>, buffer_id = 3, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  // 标量形参**不挂** `afir.buffer_desc`：它不是一块存储。`afir.arg_descs` 里它仍然占一格
  // （退化视图，只留角色），否则描述子下标就跟参数下标错开。
  func.func private @stage0_template0(
      %input: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.buffer_desc = #gm_buffer},
      %scale: f16 {afir.kernel_arg = #afir.kernel_arg<role = scalar_input, index = 1>},
      %output: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.buffer_desc = #gm_buffer}) attributes {
    afir.arg_descs = [#input_desc, #scalar_desc, #output_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 10,
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %load_tile = memref.alloc() {afir.buffer_desc = #load_buffer} : memref<16xf16>
    %compute_tile = memref.alloc() {afir.buffer_desc = #output_queue} : memref<16xf16>
    %loaded = afir.load %input outs(%load_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#input_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed = afir.add %loaded, %scale outs(%compute_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#compute_desc, #scalar_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, f16), memref<16xf16> -> memref<16xf16>
    afir.store %computed outs(%output) {
      afir.destination_descs = [#output_desc],
      afir.operand_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// CHECK: // -----KERNEL_BODY-----
// 1. kernel body 的标量形参仍是 `half`，没有被改成指针。
// CHECK: __aicore__ void plan0_stage0_template0({{.*}}half {{v[0-9]+}}, __gm__ half*
// 计算发的是标量形态的调用，与常量标量那条一模一样——conversion 层不区分标量从哪来。
// CHECK: AscendC::Adds
// 2. 到了 kernel 入口，标量夹在两个 GM 指针中间：形参序是 input0、input1、output0。
// CHECK: extern "C"  __global__ __aicore__ void fused_graph(__gm__ half* {{v[0-9]+}}, half {{v[0-9]+}}, __gm__ half* {{v[0-9]+}}, __gm__ uint8_t*
// CHECK: // -----AUTOFUSE_LAUNCH-----
// 3. host 包装器：签名按值、调用处不 cast。
// CHECK: extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, half input1, void* output0, void* workspace,
// CHECK: fused_graph<<<blockDim, nullptr, stream>>>((__gm__ half*)input0, input1, (__gm__ half*)output0,
