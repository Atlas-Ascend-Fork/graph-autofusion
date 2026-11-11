// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT

// 标量在第一个操作数（矩阵上 8 个 op 的 `swap(2nd)` 那半张格子）。
//
// 这 8 个全是可交换的，所以生成的调用与标量在第二个逐字相同：`Adds(dst, tile,
// scalar, count)`。这份用例钉的就是这件事——`ScalarOperand::Any` 只把有形状的那个操作数
// 挪到第一个实参上，不新开一个 implementation、也不换调用形状。三条 RUN 各钉一层：
//
//   IR     —— 建的是标量那一类的 def，而且 tile 在前（`%arg1, %arg0, %cst`）；
//   CPP    —— emitter 打出来的实参序，tile 在 scalar 前面。写反了 IR 全绿而这里红；
//   SELECT —— 选中的 implementation 是 `<op>.tensor_scalar`，与不交换时同一个 id。
//             要是有人把交换方向做成第二个 implementation，这里会变成另一个名字。
//
// `lt` / `le` / `gt` / `ge` 不在这里：它们不可交换，`scalarOperand` 是 `"rhs"`，标量在左
// 必须拒收——那条在 afir-scalar-binary-invalid.mlir。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

func.func @add_scalar_lhs(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>) {
  %scalar = arith.constant 2.000000e+00 : f16
  %result = afir.add %scalar, %input outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f16, !ascendc.local_tensor<?xf16>), !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
  return
}

func.func @mul_scalar_lhs(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>) {
  %scalar = arith.constant 3.000000e+00 : f32
  %result = afir.mul %scalar, %input outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f32, !ascendc.local_tensor<?xf32>), !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
  return
}

func.func @maximum_scalar_lhs(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>) {
  %scalar = arith.constant 1.000000e+00 : f16
  %result = afir.maximum %scalar, %input outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f16, !ascendc.local_tensor<?xf16>), !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
  return
}

func.func @minimum_scalar_lhs(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>) {
  %scalar = arith.constant -1.000000e+00 : f16
  %result = afir.minimum %scalar, %input outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f16, !ascendc.local_tensor<?xf16>), !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
  return
}

func.func @logical_and_scalar_lhs(%input: !ascendc.local_tensor<?xf32>,
                                  %destination: !ascendc.local_tensor<?xui8>,
                                  %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 2.000000e+00 : f32
  %result = afir.logical_and %scalar, %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f32, !ascendc.local_tensor<?xf32>),
      !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xui8>
  return
}

func.func @logical_or_scalar_lhs(%input: !ascendc.local_tensor<?xi32>,
                                 %destination: !ascendc.local_tensor<?xui8>,
                                 %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 3 : i32
  %result = afir.logical_or %scalar, %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (i32, !ascendc.local_tensor<?xi32>),
      !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xui8>
  return
}

func.func @eq_scalar_lhs(%input: !ascendc.local_tensor<?xf32>,
                         %destination: !ascendc.local_tensor<?xui8>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 1.500000e+00 : f32
  %result = afir.eq %scalar, %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f32, !ascendc.local_tensor<?xf32>),
      !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xui8>
  return
}

func.func @ne_scalar_lhs(%input: !ascendc.local_tensor<?xf16>,
                         %destination: !ascendc.local_tensor<?xui8>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 2.500000e+00 : f16
  %result = afir.ne %scalar, %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f16, !ascendc.local_tensor<?xf16>),
      !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xui8>
  return
}

// tile 在前、标量在后：`%arg1` 是目的地，`%arg0` 是 tile，`%cst` 是标量。次序反过来
// （`%arg0, %arg1`）说明 `emissionOrder` 没有把 `Any` 那一档换过来。
// IR-LABEL: func.func @add_scalar_lhs
// IR: %[[ADD_S:.*]] = arith.constant 2.000000e+00 : f16
// IR: ascendc.adds_l2 %arg1, %arg0, %[[ADD_S]]
// IR-NOT: ascendc.add_l2
// IR-NOT: afir.add

// IR-LABEL: func.func @mul_scalar_lhs
// IR: %[[MUL_S:.*]] = arith.constant 3.000000e+00 : f32
// IR: ascendc.muls_l2 %arg1, %arg0, %[[MUL_S]]
// IR-NOT: ascendc.mul_l2

// IR-LABEL: func.func @maximum_scalar_lhs
// IR: %[[MAX_S:.*]] = arith.constant 1.000000e+00 : f16
// IR: ascendc.maxs_l2 %arg1, %arg0, %[[MAX_S]]

// IR-LABEL: func.func @minimum_scalar_lhs
// IR: %[[MIN_S:.*]] = arith.constant -1.000000e+00 : f16
// IR: ascendc.mins_l2 %arg1, %arg0, %[[MIN_S]]

// IR-LABEL: func.func @logical_and_scalar_lhs
// IR: %[[AND_S:.*]] = arith.constant 2.000000e+00 : f32
// IR: ascendc.logical_and_scalar_extend %arg1, %arg0, %[[AND_S]], %arg2
// IR-NOT: ascendc.logical_and_extend %

// IR-LABEL: func.func @logical_or_scalar_lhs
// IR: %[[OR_S:.*]] = arith.constant 3 : i32
// IR: ascendc.logical_or_scalar_extend %arg1, %arg0, %[[OR_S]], %arg2
// IR-NOT: ascendc.logical_or_extend %

// 比较模式跟着 AFIR op 走，交换操作数不改它：EQ 是 2、NE 是 5。
// IR-LABEL: func.func @eq_scalar_lhs
// IR: %[[EQ_S:.*]] = arith.constant 1.500000e+00 : f32
// IR: ascendc.compare_scalar_extend_contiguous %arg1, %arg0, %[[EQ_S]]
// IR-SAME: cmpMode = 2 : i64

// IR-LABEL: func.func @ne_scalar_lhs
// IR: %[[NE_S:.*]] = arith.constant 2.500000e+00 : f16
// IR: ascendc.compare_scalar_extend_contiguous %arg1, %arg0, %[[NE_S]]
// IR-SAME: cmpMode = 5 : i64

// 字面量写全：emitter 打的是定长形式，写成 `c2_f16` 之外的缩写会匹配上别的常量名。
// CPP: {{^ *}}AscendC::Adds<half, 1>({{v[0-9]+}}, {{v[0-9]+}}, c2_f16, {{v[0-9]+}});
// CPP: {{^ *}}AscendC::Muls<float, 1>({{v[0-9]+}}, {{v[0-9]+}}, c3_f32, {{v[0-9]+}});
// CPP: {{^ *}}AscendC::Maxs<half, 1>({{v[0-9]+}}, {{v[0-9]+}}, c1_f16, {{v[0-9]+}});
// CPP: {{^ *}}AscendC::Mins<half, 1>({{v[0-9]+}}, {{v[0-9]+}}, cm1_f16, {{v[0-9]+}});
// CPP: {{^ *}}LogicalAndScalarExtend({{v[0-9]+}}, {{v[0-9]+}}, c2_f32, {{v[0-9]+}}, static_cast<uint32_t>({{v[0-9]+}}));
// CPP: {{^ *}}LogicalOrScalarExtend({{v[0-9]+}}, {{v[0-9]+}}, c3_i32, {{v[0-9]+}}, static_cast<uint32_t>({{v[0-9]+}}));
// CPP: {{^ *}}CompareScalarExtend({{v[0-9]+}}, {{v[0-9]+}}, c1_5_f32, CMPMODE::EQ, static_cast<uint32_t>({{v[0-9]+}}), {{v[0-9]+}});
// CPP: {{^ *}}CompareScalarExtend({{v[0-9]+}}, {{v[0-9]+}}, c2_5_f16, CMPMODE::NE, static_cast<uint32_t>({{v[0-9]+}}), {{v[0-9]+}});

// 与标量在右时同一个 id：交换方向不是第二个 implementation。
// SELECT: ascend.implementation = {id = "add.tensor_scalar", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "mul.tensor_scalar", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "maximum.tensor_scalar", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "minimum.tensor_scalar", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "logical_and.tensor_scalar", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "logical_or.tensor_scalar", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "eq.tensor_scalar", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "ne.tensor_scalar", kind = "ascendc_api"}
