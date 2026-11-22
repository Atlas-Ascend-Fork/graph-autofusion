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

// `afir.bitwise_and` 是**二元**一类里第一个两侧调用不同名的 op，与一元的
// `afir.reciprocal` 同一档：mem 侧走包装且要 tmp buffer，reg 侧是标准调用、不要。
// 这一份钉 mem 侧；reg 侧在 afir-bitwise-and-stock-api.mlir。
//
// 包装的实参顺序是 `(dst, src0, src1, tmp, count)`——tmp 在 count 之前，与一元包装
// 同向、与三元的选择 / clip 族反向。写反了 device 上一个重载都对不上，所以 CPP 那条
// check 把五个实参连着钉。
//
// 元素类型是位模式不是数值：这里用 i16（包装对 16 位直接转发 `AscendC::And`，不走
// `ReinterpretCast` 那一支）。f16 / f32 在这个 op 上根本不该过 ODS。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @bitwise_and_extend_i16(%lhs: !ascendc.local_tensor<?xi16>, %rhs: !ascendc.local_tensor<?xi16>,
                                    %destination: !ascendc.local_tensor<?xi16>,
                                    %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.bitwise_and %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi16>, !ascendc.local_tensor<?xi16>),
        !ascendc.local_tensor<?xi16>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xi16>
    return
  }
}

// IR-LABEL: func.func @bitwise_and_extend_i16
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.bitwise_and_extend %arg2, %arg0, %arg1, %arg3, %[[COUNT]]
// IR-NOT: afir.bitwise_and

// 全局函数，不带 `AscendC::` 限定；也不带显式模板实参——legacy 的 `BinaryApiTmpCall`
// 就是让元素类型自己推导的。
// CPP-LABEL: void bitwise_and_extend_i16(
// CPP: {{^ *}}BitwiseAndExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(
// CPP-NOT: AscendC::BitwiseAnd

// SELECT: ascend.implementation = {id = "bitwise_and.tensor.extend", kind = "ascendc_api"}

// reg 侧的标准调用不收 tmp buffer，所以同一段 IR 在那张支持表上选不出实现。
// WRONGTARGET: no supported
