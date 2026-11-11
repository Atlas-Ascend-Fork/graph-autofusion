// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// L2 elementwise 的下降在带值规划结果时的行为。
//
// 步 8 之后 `afir.buffer_desc` 挂在值的定义点上；这里的值是 kernel 参数，定义点就是
// 它的 arg attr（设计文档 5.1 步 8）。这条覆盖单独成文件的原因：`convert-afir-to-ascendc`
// 会把 `afir.*` 的 arg attr 原样留在产出的 AscendC 函数上，而 `ascir-translate` 只注册
// AscendC 方言，解析不了它们——所以 afir-to-ascendc-binary-l2.mlir（跑 ascir-translate）
// 必须不带 descriptor，两件事拆在两个文件里各测各的。
//
// 断言两件事：
//   1) descriptor 在场时下降结果与不在场时完全一致——它不参与非调度形态的下降决策；
//   2) descriptor 的空间与队列字段被如实保留在参数上，没有在下降途中被改写或丢弃。

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s -o %t.selected.mlir
// RUN: af-opt --convert-afir-to-ascendc --target=Ascend910B1 %t.selected.mlir | FileCheck %s --check-prefix=CONVERT

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>
#lhs_queue = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, life_time_axis_id = -1>
#rhs_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 1, life_time_axis_id = -1>
#destination_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 2, life_time_axis_id = -1>

module {
  func.func @add_l2_f16(%lhs: !ascendc.local_tensor<?xf16> {afir.buffer_desc = #lhs_queue},
                        %rhs: !ascendc.local_tensor<?xf16> {afir.buffer_desc = #rhs_buffer},
                        %destination: !ascendc.local_tensor<?xf16> {afir.buffer_desc = #destination_buffer}) {
    %result = afir.add %lhs, %rhs outs(%destination) {
      "ascend.implementation" = {kind = "ascendc_api", id = "add.tensor_tensor"},
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } :
        (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }
}

// IR-LABEL: func.func @add_l2_f16
// 队列缓冲与普通缓冲的 descriptor 都原样留在各自参数的定义点上。
// IR-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, life_time_axis_id = -1>}
// IR-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, life_time_axis_id = -1>}
// IR-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, life_time_axis_id = -1>}
// 非调度形态下降结果与不带 descriptor 时逐条一致（对照 afir-to-ascendc-binary-l2.mlir）。
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.add_l2 %arg2, %arg0, %arg1, %[[COUNT]] {ascendc.execution_model = "SIMD", ascendc.instruction_form = "MemoryBased", ascendc.unit = "AiCore.Vector"}
// IR-NOT: afir.add
// descriptor 不参与非调度形态的取址，因此不得凭空生出 tile 偏移。
// IR-NOT: ascendc.local_tensor.sub_index

// 4 个旧数组名一个都不会回来。
// CONVERT-NOT: afir.operand_buffer_descs
// CONVERT-NOT: afir.result_buffer_descs
// CONVERT-NOT: afir.destination_buffer_descs
// CONVERT-NOT: afir.arg_buffer_descs
// CONVERT-LABEL: func.func @add_l2_f16
// CONVERT: ascendc.add_l2
