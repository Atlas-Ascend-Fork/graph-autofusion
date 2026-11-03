// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR \
// RUN:   --implicit-check-not=afir. --implicit-check-not=ascendc.add_l2 --implicit-check-not=ascendc.mul_l2
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR \
// RUN:   --implicit-check-not=afir. --implicit-check-not=ascendc.add_l2 --implicit-check-not=ascendc.mul_l2
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP

// 同一个 `afir.add`，第二个操作数换成标量 SSA 值。**不是另一族 op**：`add.tensor_tensor`
// 与 `add.tensor_scalar` 是同一个 op 的两个 implementation，选哪个由操作数形态定死，
// 两者的判据互斥（`getTensorTypes` 两个方向都挡），不靠优先级。
//
// 这一族在两张供给表上是同一个 `AscendC::` 调用、都不要 tmp buffer，所以两个 `--target`
// 共用同一份 IR 和同一组 CHECK——它跟平台没关系，`--implicit-check-not` 里那两条 tensor
// 形态的名字是这份用例真正在钉的东西：**标量没有被当成一块 tile**。
//
// 标量也不取块内下标：它在整条循环里就是同一个值，`LocalTensorSubIndexOp` 只落在
// 两块 tile 上。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

func.func @add_scalar(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>) {
  %scalar = arith.constant 2.000000e+00 : f16
  %result = afir.add %input, %scalar outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf16>, f16), !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
  return
}

// IR-LABEL: func.func @add_scalar
// IR: %[[SCALAR:.*]] = arith.constant 2.000000e+00 : f16
// IR: ascendc.adds_l2 %arg1, %arg0, %[[SCALAR]]

func.func @mul_scalar(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>) {
  %scalar = arith.constant 3.000000e+00 : f32
  %result = afir.mul %input, %scalar outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf32>, f32), !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
  return
}

// IR-LABEL: func.func @mul_scalar
// IR: %[[MULS:.*]] = arith.constant 3.000000e+00 : f32
// IR: ascendc.muls_l2 %arg1, %arg0, %[[MULS]]

// 四个 op 各自钉一遍，因为「哪个 AFIR op 配哪个 AscendC op」是一张手写的对照表
// （`registerScalarBinary` 的四行）。只验 add 的话，把 Maxs 与 Mins 互换、或者把
// MaximumOp 写成 MinimumOp，整套用例还是全绿。

func.func @maximum_scalar(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>) {
  %scalar = arith.constant 1.000000e+00 : f16
  %result = afir.maximum %input, %scalar outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf16>, f16), !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
  return
}

// IR-LABEL: func.func @maximum_scalar
// IR: %[[MAXS:.*]] = arith.constant 1.000000e+00 : f16
// IR: ascendc.maxs_l2 %arg1, %arg0, %[[MAXS]]

func.func @minimum_scalar(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>) {
  %scalar = arith.constant -1.000000e+00 : f16
  %result = afir.minimum %input, %scalar outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf16>, f16), !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
  return
}

// IR-LABEL: func.func @minimum_scalar
// IR: %[[MINS:.*]] = arith.constant -1.000000e+00 : f16
// IR: ascendc.mins_l2 %arg1, %arg0, %[[MINS]]

// 生成的调用里标量是第三个实参，跟着一个元素个数——`Adds(dst, src, scalar, count)`。
// 实参序单独钉一遍：MLIR 那一层的序由上面的 `%[[...]]` 管着，但 emitter 是另一份手写
// 代码，它把 scalar 和 count 写反了上面全都看不出来。
// CPP: {{^ *}}AscendC::Adds<half, 1>({{v[0-9]+}}, {{v[0-9]+}}, c2_f16, {{v[0-9]+}});
// CPP: {{^ *}}AscendC::Muls
// CPP: {{^ *}}AscendC::Maxs
// CPP: {{^ *}}AscendC::Mins
