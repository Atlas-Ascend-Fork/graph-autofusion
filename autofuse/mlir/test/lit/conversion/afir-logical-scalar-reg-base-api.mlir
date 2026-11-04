// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=SELECT

// reg 侧的标量形态。**只有 `logical_and` 有**——`logical_or` 在这一侧一个标量都不收，
// 见 afir-logical-scalar-reg-base-unsupported.mlir。
//
// 调用是 `LogicalAndExtends`，四个实参、没有 tmp：与 mem 侧那份五参的
// `LogicalAndScalarExtend` 既不同名也不同元。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

func.func @logical_and_scalar_reg_base_f16(%input: !ascendc.local_tensor<?xf16>,
                                           %destination: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 2.000000e+00 : f16
  %result = afir.logical_and %input, %scalar outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf16>, f16),
      !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xui8>
  return
}

// IR-LABEL: func.func @logical_and_scalar_reg_base_f16
// IR: %[[SCALAR:.*]] = arith.constant 2.000000e+00 : f16
// IR: ascendc.logical_and_extends_reg_base %arg1, %arg0, %[[SCALAR]]
// IR-NOT: ascendc.logical_and_scalar_extend
// IR-NOT: afir.logical_and

// 末尾是 `Extends(`，不是 mem 侧的 `ScalarExtend(`。
// CPP-LABEL: void logical_and_scalar_reg_base_f16(
// CPP: {{^ *}}LogicalAndExtends({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT: ascend.implementation = {id = "logical_and.tensor_scalar.extend_reg_base", kind = "ascendc_api"}
