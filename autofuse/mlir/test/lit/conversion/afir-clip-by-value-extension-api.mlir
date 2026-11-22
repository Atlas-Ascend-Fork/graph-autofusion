// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// RUN: not af-opt --afir-select-implementation --target=Ascend950DT_950x %s 2>&1 | FileCheck %s --check-prefix=WRONGTARGET

// clip 族的 mem-based 一侧。与选择族(afir-select-extension-api.mlir)逐字同形——同一个
// 连续 tile 形态、同一个 emitter——只差一条:第一个操作数不是 ui8 掩码,它是 x,
// 与 min / max 和结果同 dtype。所以这一类在 ODS 上绑的是四个而不是后三个
// (`AFIR_BufferizableTernaryHomoElementwiseOp`),在 capability 上走的是
// `conditionIsMask = false` 那一支。
//
// 两侧同名 `ClipByValue`,靠 tmp buffer 与扩展头分辨:mem 侧 `clipbyvalue.h` 要那块 UB,
// reg 侧 `clipbyvalue_reg_base.h` 转发给 `Clamp`、自己开寄存器。所以这份带 tmp buffer
// 的输入在 reg-base 一侧是 unsupported,见文件末尾的 WRONGTARGET。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @clip_by_value_extend_f32(%x: !ascendc.local_tensor<?xf32>,
                                      %lower: !ascendc.local_tensor<?xf32>,
                                      %upper: !ascendc.local_tensor<?xf32>,
                                      %destination: !ascendc.local_tensor<?xf32>,
                                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.clip_by_value %x, %lower, %upper outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @clip_by_value_extend_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg3
// IR-NEXT: ascendc.clipbyvalue_contiguous %arg3, %arg0, %arg1, %arg2, %[[COUNT]], %arg4
// IR-NOT: afir.clip_by_value

// 包装是全局函数,不带 `AscendC::` 限定,也没有显式模板实参。`calCount` 排在 tmp buffer
// 之前:`clipbyvalue.h` 原有的那个重载是反过来的,头里另补了一个这个顺序的转发重载,
// 好让这条路上只有一种顺序。把 emitter 改回「tmp 在前」这条会红。
// CPP-LABEL: void clip_by_value_extend_f32(
// CPP: {{^ *}}ClipByValue({{.*}}, static_cast<uint32_t>({{[^)]*}}), {{[^,]*}});
// CPP-NOT: AscendC::ClipByValue
// CPP-NOT: ClipByValue<

// SELECT: ascend.implementation = {id = "clip_by_value.tensor.extend", kind = "ascendc_api"}

// WRONGTARGET: no supported
