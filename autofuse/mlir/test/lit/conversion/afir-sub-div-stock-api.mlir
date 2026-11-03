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

// `afir.sub` / `afir.div` 的 mem 侧。两个 op 合在一份用例里，因为它们**是同一件事**：
// legacy 的两个 impl 类逐行同形，两边这里要钉的也是同一条性质——
//
//   **发的是标准的 `AscendC::Sub` / `AscendC::Div`（四个实参，没有 tmp 的位置），
//   而 AFIR op 上挂着一块 tmp buffer。**
//
// 那块是调度按 `CalcBinaryApiTmpSize` 分的，给标量形态的 `Subs` / `Divs` 用；
// tensor-tensor 这条路只是收下它，不往下传。这是 `TmpBufferUse::Unused` 唯一的现场，
// 所以 IR 那两条 `-SAME` 很重要：多传一个实参就不是这个重载了。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @sub_stock_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                           %destination: !ascendc.local_tensor<?xf32>,
                           %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.sub %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  // 整数只有 `sub` 收——`div` 两侧都只有 f16 / f32。
  func.func @sub_stock_i32(%lhs: !ascendc.local_tensor<?xi32>, %rhs: !ascendc.local_tensor<?xi32>,
                           %destination: !ascendc.local_tensor<?xi32>,
                           %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.sub %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>),
        !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xi32>
    return
  }

  func.func @div_stock_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                           %destination: !ascendc.local_tensor<?xf32>,
                           %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.div %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @sub_stock_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.sub_l2 %arg2, %arg0, %arg1, %[[COUNT]]
// tmp（%arg3）没有出现在这一行里——这就是 `Unused` 与 `Passed` 的区别。
// IR-NOT: ascendc.sub_l2 %arg2, %arg0, %arg1, %arg3
// IR-NOT: afir.sub

// IR-LABEL: func.func @sub_stock_i32
// IR: ascendc.sub_l2

// IR-LABEL: func.func @div_stock_f32
// IR: %[[DCOUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.div_l2 %arg2, %arg0, %arg1, %[[DCOUNT]]
// IR-NOT: afir.div

// 标准调用，带 `AscendC::` 限定——与走包装的那些（全局函数、不带限定）正好相反。
//
// 实参数恰好四个：`[^,()]` 不许嵌套括号，结尾的 `);` 不许再有第五个。**这条就是
// `Unused` 与 `Passed` 在生成代码上的全部差别**，多传一块 tmp 它立刻红。
//
// count 是裸的 SSA 名，不是 `static_cast<uint32_t>(...)`——那层转换出自
// `printApiExtensionBinary`，只有走包装的那条路有（见 reg 侧那份用例）。
// CPP-LABEL: void sub_stock_f32(
// CPP: {{^ *}}AscendC::Sub({{[^,()]*}}, {{[^,()]*}}, {{[^,()]*}}, {{[^,()]*}});
// CPP-LABEL: void div_stock_f32(
// CPP: {{^ *}}AscendC::Div({{[^,()]*}}, {{[^,()]*}}, {{[^,()]*}}, {{[^,()]*}});

// SELECT: ascend.implementation = {id = "sub.tensor_tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "sub.tensor_tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "div.tensor_tensor", kind = "ascendc_api"}
