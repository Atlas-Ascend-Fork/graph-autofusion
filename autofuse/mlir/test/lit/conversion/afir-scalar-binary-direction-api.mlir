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

// 方向要生成的那三个标量形态：`sub` / `div` / `truediv`（mem 侧）。
//
// 与 `afir-scalar-binary-swapped-api.mlir` 的那 8 个恰好相反：那 8 个可交换，交换完生成的
// 调用逐字相同；这三个不可交换，legacy 把方向编码成一个 bool 模板实参
// `IS_SCALAR_LATTER`（`Subs<T, false>` = 标量在左）。所以每个 op 两个方向各写一份，
// 两份的差别只在模板实参的最后一位——这正是唯一会写错又不报错的地方。
//
// 三条 RUN 各钉一层：
//
//   IR     —— 建的是标量那一类的 def，tile 在前，`isScalarLatter` 属性取值正确；
//   CPP    —— 模板实参列写全了。`Subs<half>` 少一位照样编得过（bool 有缺省值 true），
//             只是方向永远是 true，标量在左时静默算错另一个数；
//   SELECT —— 两个方向选中的是同一个 id，交换不是第二个 implementation。
//
// `TrueDivExtends` 是三个里唯一有**两个** dtype 模板实参的：`<T, U, LATTER>`，`U` 是结果
// 类型。相同也要写——少写一个的话 `LATTER` 会落到 `U` 的位置上，bool 当类型用编不过。
// 这里特意取 i32 输入 / f32 结果那一行，两个 dtype 不同，写反了当场看得见。
//
// reg 侧只有 `div` 一个（`sub` / `truediv` 的 v2 `IsScalarInputSupported` 返回 false），
// 在 afir-scalar-binary-direction-reg-base-api.mlir，那边不收 tmp buffer 所以图不一样。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

func.func @sub_scalar_rhs(%input: !ascendc.local_tensor<?xf16>,
                          %destination: !ascendc.local_tensor<?xf16>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 2.000000e+00 : f16
  %result = afir.sub %input, %scalar outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf16>, f16),
      !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf16>
  return
}

func.func @sub_scalar_lhs(%input: !ascendc.local_tensor<?xf16>,
                          %destination: !ascendc.local_tensor<?xf16>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 2.000000e+00 : f16
  %result = afir.sub %scalar, %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f16, !ascendc.local_tensor<?xf16>),
      !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf16>
  return
}

func.func @div_scalar_rhs(%input: !ascendc.local_tensor<?xf32>,
                          %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 4.000000e+00 : f32
  %result = afir.div %input, %scalar outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf32>, f32),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

func.func @div_scalar_lhs(%input: !ascendc.local_tensor<?xf32>,
                          %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 4.000000e+00 : f32
  %result = afir.div %scalar, %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f32, !ascendc.local_tensor<?xf32>),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

func.func @truediv_scalar_rhs(%input: !ascendc.local_tensor<?xi32>,
                              %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 3 : i32
  %result = afir.truediv %input, %scalar outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xi32>, i32),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

func.func @truediv_scalar_lhs(%input: !ascendc.local_tensor<?xi32>,
                              %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 3 : i32
  %result = afir.truediv %scalar, %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (i32, !ascendc.local_tensor<?xi32>),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

// 标量在右：`isScalarLatter` 是缺省值 true，所以属性不出现在 IR 里——`-NOT` 钉的就是
// 这一点，写成 `false` 的话它会被打出来。
//
// 正向那一条只匹配到操作数列为止，属性不写进同一条：`%{{.*}}` 是贪婪的，会吃到行尾，
// 跟一条 `IR-SAME` 就永远无处可匹配（这份用例第一版就是这么写错的）。要么像下面标量在左
// 那几条一样整行写一条，要么像这里一样让 `-NOT` 去管这一条之后的那一段。
// IR-LABEL: func.func @sub_scalar_rhs
// IR: %[[SUB_R:.*]] = arith.constant 2.000000e+00 : f16
// IR: ascendc.subs %arg1, %arg0, %[[SUB_R]], %arg2
// IR-NOT: isScalarLatter

// 标量在左：tile 仍然在第一个输入位（`%arg0` 是 input、`%arg1` 是 destination，所以操作数
// 列与上面逐字相同），差别全在属性上。
// IR-LABEL: func.func @sub_scalar_lhs
// IR: ascendc.subs %arg1, %arg0, {{.*}}isScalarLatter = false

// IR-LABEL: func.func @div_scalar_rhs
// IR: ascendc.divs %arg1, %arg0,
// IR-NOT: isScalarLatter

// IR-LABEL: func.func @div_scalar_lhs
// IR: ascendc.divs %arg1, %arg0, {{.*}}isScalarLatter = false

// IR-LABEL: func.func @truediv_scalar_rhs
// IR: ascendc.true_div_extends %arg1, %arg0,
// IR-NOT: isScalarLatter

// IR-LABEL: func.func @truediv_scalar_lhs
// IR: ascendc.true_div_extends %arg1, %arg0, {{.*}}isScalarLatter = false

// 模板实参写全，字面量写全。最后一位是方向，两两成对比着看。
// CPP: {{^ *}}Subs<half, true>({{v[0-9]+}}, {{v[0-9]+}}, c2_f16, {{v[0-9]+}}, static_cast<uint32_t>({{v[0-9]+}}));
// CPP: {{^ *}}Subs<half, false>({{v[0-9]+}}, {{v[0-9]+}}, c2_f16, {{v[0-9]+}}, static_cast<uint32_t>({{v[0-9]+}}));
// CPP: {{^ *}}Divs<float, true>({{v[0-9]+}}, {{v[0-9]+}}, c4_f32, {{v[0-9]+}}, static_cast<uint32_t>({{v[0-9]+}}));
// CPP: {{^ *}}Divs<float, false>({{v[0-9]+}}, {{v[0-9]+}}, c4_f32, {{v[0-9]+}}, static_cast<uint32_t>({{v[0-9]+}}));
// CPP: {{^ *}}TrueDivExtends<int32_t, float, true>({{v[0-9]+}}, {{v[0-9]+}}, c3_i32, {{v[0-9]+}}, static_cast<uint32_t>({{v[0-9]+}}));
// CPP: {{^ *}}TrueDivExtends<int32_t, float, false>({{v[0-9]+}}, {{v[0-9]+}}, c3_i32, {{v[0-9]+}}, static_cast<uint32_t>({{v[0-9]+}}));

// 两个方向同一个 id。
// SELECT: ascend.implementation = {id = "sub.tensor_scalar", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "sub.tensor_scalar", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "div.tensor_scalar", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "div.tensor_scalar", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "truediv.tensor_scalar.extend", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "truediv.tensor_scalar.extend", kind = "ascendc_api"}
