// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 标量写入者与读取者位于两个兄弟 scf.if 中。
// AFIRExecConditionPass 为每个带 exec_condition 的算子分别建立 scf.if；旧查询仅扫描当前
// block 的前驱，无法发现其他条件块中的写入者，导致实现选择失败。
//
// 读取前连续生成 SetFlag / WaitFlag，等待相应流水线在同步点之前的任务完成。
// 因此，分析只需确定可能的写入流水线，不要求写入者与读取者具有相同的控制流条件。
// 该同步只保证执行顺序，不负责保证输入数据已初始化。
//
// 本用例覆盖两个入口：独立实现选择在属性缺失时直接分析 AFIR；完整下降流水线提前记录
// afir.scalar_read_sync，下降时仅读取属性。

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --afir-scalar-read-sync %s | FileCheck %s --check-prefix=ATTR

#one = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">],
                         vectorized_axis = [0], vectorized_strides = [#afir.expr<"0">], role = <compute>>
#full = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"32">], strides = [#afir.expr<"1">],
                          vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>
#gm_one = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">], role = <input>>

func.func @minimum_scalar_written_in_sibling_if(%input: !ascendc.local_tensor<32xf32>,
                                                %dst: !ascendc.local_tensor<32xf32>,
                                                %seed: !ascendc.local_tensor<1xf32>, %guard: i1) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.alloc_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1xf32>
  // 写在第一个条件块里：矢量计算，所以读之前要等 V 排空。
  scf.if %guard {
    %neg = afir.neg %seed outs(%one) {afir.operand_descs = [#one], afir.destination_descs = [#one],
                                      afir.result_descs = [#one]}
        : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1xf32> -> !ascendc.local_tensor<1xf32>
  }
  // 读在第二个条件块里。两个 `scf.if` 是兄弟，谓词是同一个值。
  scf.if %guard {
    %r = afir.minimum %input, %one outs(%dst) {afir.operand_descs = [#full, #one],
                                               afir.destination_descs = [#full], afir.result_descs = [#full]}
        : (!ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32>
          -> !ascendc.local_tensor<32xf32>
  }
  return
}

// 将写入流水线 vector 记录在标量操作数对应的项中（输入下标为 1）。
// ATTR-LABEL: func.func @minimum_scalar_written_in_sibling_if
// ATTR: afir.minimum
// ATTR-SAME: afir.scalar_read_sync = {{\[}}[], [#afir<pipe vector>]]

// 确认选择支持标量操作数的实现。
// SELECT: id = "minimum.tensor_scalar"

// 事件是 `V_S` 而不是 `MTE2_S`：队列位置是 VECIN，但真实写入者是矢量计算。按位置猜会发
// `MTE2_S`，会导致数值错误。
// IR-LABEL: func.func @minimum_scalar_written_in_sibling_if
// IR: ascendc.pipe.fetch_event_id {{.*}}v_s
// IR: ascendc.set_flag
// IR: ascendc.wait_flag
// IR: ascendc.local_tensor.get_value
// IR: ascendc.mins

// 写入者在程序序上位于读取者之后。循环中的前次迭代可能先写入同一存储，因此分析不能仅扫描
// 前驱。本片段检查写入者的收集范围，不验证首次读取时数据是否已初始化。
func.func @minimum_scalar_written_after_read(%input: !ascendc.local_tensor<32xf32>,
                                             %dst: !ascendc.local_tensor<32xf32>,
                                             %seed: !ascendc.local_tensor<1xf32>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.alloc_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1xf32>
  %r = afir.minimum %input, %one outs(%dst) {afir.operand_descs = [#full, #one],
                                             afir.destination_descs = [#full], afir.result_descs = [#full]}
      : (!ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32>
        -> !ascendc.local_tensor<32xf32>
  %neg = afir.neg %seed outs(%one) {afir.operand_descs = [#one], afir.destination_descs = [#one],
                                    afir.result_descs = [#one]}
      : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1xf32> -> !ascendc.local_tensor<1xf32>
  return
}

// IR-LABEL: func.func @minimum_scalar_written_after_read
// IR: ascendc.pipe.fetch_event_id {{.*}}v_s
