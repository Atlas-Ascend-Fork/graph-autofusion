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

// `afir.logical_not` 的 mem 侧：`LogicalNot(dst, src, blk_tensor_with_value_1, tmp_buf, size)`
// （`logical_not.h`）。
//
// 三件值得钉的：
//
//   - **apiName 是裸名 `LogicalNot`**，不带 `Extend` 后缀。它与 `AscendC::LogicalNot`
//     同名而不是同一个函数——包装是全局的，发出去的调用不带 `AscendC::` 限定，两者靠
//     实参个数分开（5 个 vs 3 个）。legacy 的 `LogicalNotAscIrCodegenImpl::GetApiName`
//     写的也是这个裸名。
//   - **这一侧两端同 dtype**。reg 侧结果恒为 ui8，两侧关系不同，所以这个 op 在 AFIR 上
//     不声明 `InferTypeOpInterface`（理由见 `AFIR_BufferizableLogicalNotOp`）。
//   - 那块预先填好 1 的常量是 **`half`**，与正在算的 dtype 无关：f32 的 logical_not
//     用的仍然是 half 那一块，因为包装的形参写死了 `LocalTensor<half>`。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @logical_not_extend_f32(%input: !ascendc.local_tensor<?xf32>,
                                    %destination: !ascendc.local_tensor<?xf32>,
                                    %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.logical_not %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @logical_not_extend_i32(%input: !ascendc.local_tensor<?xi32>,
                                    %destination: !ascendc.local_tensor<?xi32>,
                                    %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.logical_not %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xi32>
    return
  }
}

// IR-LABEL: func.func @logical_not_extend_f32
// IR: %[[BLK:.*]] = ascendc.tbuf.get_tensor {{.*}} {ascendc.const_block = 1.000000e+00 : f16}
// IR: ascendc.duplicate_l2 %[[BLK]]
// IR-NEXT: ascendc.pipe_barrier pipe_v
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.logical_not_extend %arg1, %arg0, %[[BLK]], %arg2, %[[COUNT]]
// IR-NOT: ascendc.logical_not_extend_reg_base
// IR-NOT: afir.logical_not

// IR-LABEL: func.func @logical_not_extend_i32
// IR: ascendc.logical_not_extend

// CPP-LABEL: void logical_not_extend_f32(
// CPP: AscendC::LocalTensor<half> {{.*}} = {{.*}}.Get<half>()
// CPP: LogicalNot<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(
// CPP-NOT: LogicalNotExtend

// SELECT-COUNT-2: ascend.implementation = {id = "logical_not.tensor.extend", kind = "ascendc_api"}
