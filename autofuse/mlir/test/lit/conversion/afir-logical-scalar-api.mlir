// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=MEM
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=MEMCPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=MEMSELECT

// 逻辑族的标量形态。右操作数换成一个标量 SSA 值，结果 dtype 照旧恒为 `ui8`。
//
// 与 `add` / `mul` 那一类的标量形态不同，这一类两侧不是同一个调用，所以它没有走
// `registerScalarBinary`（那个包装把家族写死成 `ApiFamily::Any`）。两侧的包装名各按各
// 的规矩派生，都不是 apiName 加个后缀：
//
//   - mem 侧 `binary_api_tmp_call.cpp` 拼的是 `api_name_ + "ScalarExtend"`，于是
//     `LogicalAndScalarExtend(dst, src, scalar, tmp, size)`——**五参，tmp 照旧要**；
//   - reg 侧 `binary_api_call.cpp` 走通用的 `api_name_ + "s"` 分支，于是
//     `LogicalAndExtends(dst, src, scalar, size)`——四参，无 tmp。
//
// reg 侧的 `logical_or` 一个标量都不收，那一条在本文件末尾单独一份用例里钉。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

func.func @logical_and_scalar_f32(%input: !ascendc.local_tensor<?xf32>,
                                  %destination: !ascendc.local_tensor<?xui8>,
                                  %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 2.000000e+00 : f32
  %result = afir.logical_and %input, %scalar outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf32>, f32),
      !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xui8>
  return
}

func.func @logical_or_scalar_i32(%input: !ascendc.local_tensor<?xi32>,
                                 %destination: !ascendc.local_tensor<?xui8>,
                                 %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 3 : i32
  %result = afir.logical_or %input, %scalar outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xi32>, i32),
      !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xui8>
  return
}

// 标量不取块内下标：它在整条循环里就是同一个值，`LocalTensorSubIndexOp` 只落在 tile 上。
// MEM-LABEL: func.func @logical_and_scalar_f32
// MEM: %[[SCALAR:.*]] = arith.constant 2.000000e+00 : f32
// MEM: ascendc.logical_and_scalar_extend %arg1, %arg0, %[[SCALAR]], %arg2
// MEM-NOT: ascendc.logical_and_extend %
// MEM-NOT: afir.logical_and

// MEM-LABEL: func.func @logical_or_scalar_i32
// MEM: ascendc.logical_or_scalar_extend
// MEM-NOT: afir.logical_or

// MEMCPP-LABEL: void logical_and_scalar_f32(
// MEMCPP: {{^ *}}LogicalAndScalarExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(
// MEMCPP-LABEL: void logical_or_scalar_i32(
// MEMCPP: {{^ *}}LogicalOrScalarExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// MEMSELECT: ascend.implementation = {id = "logical_and.tensor_scalar", kind = "ascendc_api"}
// MEMSELECT: ascend.implementation = {id = "logical_or.tensor_scalar", kind = "ascendc_api"}
