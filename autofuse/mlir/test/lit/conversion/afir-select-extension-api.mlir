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

// 选择族的 mem-based 一侧。与 abs / reciprocal 那一档的不同之处有两条,都钉在这里:
//
//   - **两侧都是 Autofuse 的包装**,不是「一侧标准、一侧扩展」。所以 reg-base 一侧不是
//     「换个标准调用」,而是换一个包装(`WhereExtend`)且**不要 tmp buffer**——这份带
//     tmp buffer 的输入在那一侧因此是 unsupported,见文件末尾的 WRONGTARGET。
//   - **第一个输入的元素类型与另外两个不同**:它是 ui8 真值。判据要分两条查,合成一条
//     「三个输入元素类型相同」的话整条链就只剩 ui8 能过。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @select_extend_f32(%condition: !ascendc.local_tensor<?xui8>,
                               %lhs: !ascendc.local_tensor<?xf32>,
                               %rhs: !ascendc.local_tensor<?xf32>,
                               %destination: !ascendc.local_tensor<?xf32>,
                               %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.select %condition, %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @select_extend_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg3
// IR-NEXT: ascendc.where_contiguous %arg3, %arg0, %arg1, %arg2, %[[COUNT]], %arg4
// IR-NOT: afir.select

// 包装是全局函数,不带 `AscendC::` 限定;而且**没有显式模板实参**——legacy 的
// `where_api_call.cpp` 让元素类型自己推导,写上一个会挑到另一个重载。
// CPP-LABEL: void select_extend_f32(
// CPP: {{^ *}}Where(
// CPP-NOT: AscendC::Where
// CPP-NOT: Where<

// SELECT: ascend.implementation = {id = "select.tensor.extend", kind = "ascendc_api"}

// WRONGTARGET: no supported
