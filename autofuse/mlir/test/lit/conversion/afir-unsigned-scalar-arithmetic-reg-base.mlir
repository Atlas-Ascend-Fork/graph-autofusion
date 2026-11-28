// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// ui16 / ui32 的 tensor-scalar：`Adds` / `Maxs` / `Mins` 的连续 LocalTensor 接口不收这两种
// 类型，改选 `*.tensor_scalar.extend_reg_base`，发 fork 的 reg 侧包装 `AddsExtend` / `MaxsExtend` /
// `MinsExtend`（`api_regbase/scalar_add.h` 等三个头），接法与 `floor_div.tensor.extend_reg_base`
// 相同。包装不收 tmp buffer、三个算子都可交换，实参恒是 `(dst, src, scalar, count)`：标量在
// 第二个操作数的位置。Maximum 的两个函数把标量写在左操作数（scalarOperand 是 "any"），另外
// 两个 op 写在右侧，两个方向各钉一遍。mem 侧没有这两种类型的标量实现。
//
// 包装里的循环不在这条流水线的产物里——头是按文本内联进最终 kernel 的，`ascir-translate` 的
// 输出只有调用。循环的形状（寄存器声明在循环外、uint16_t 归纳变量、每轮一次掩码更新、一进
// 一出）由 e2e 用例 `load_unsigned_scalar_arithmetic_reg_base_store` 对着内联了包装头的 kernel 钉住。
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP --implicit-check-not=AscendC::Adds( --implicit-check-not=AscendC::Maxs( --implicit-check-not=AscendC::Mins( --implicit-check-not=__VEC_SCOPE__ --implicit-check-not=MicroAPI::
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %s 2>&1 | FileCheck %s --check-prefix=MEM
// MEM: no supported Ascend implementation

#src = #afir.tensor_desc<axis = [0], role = <input>>
#scalar = #afir.tensor_desc<role = <input>>
#dst = #afir.tensor_desc<axis = [0], role = <compute>>

// 实参顺序是 (dst, src, scalar, count)：dst 是 out，src 是入参里的张量，标量是标量形参。
// IR-LABEL: func.func @add_ui16
// IR: ascendc.adds_extend_reg_base %arg2, %arg0, %arg1, {{.*}} : !ascendc.local_tensor<?xui16>, !ascendc.local_tensor<?xui16>, ui16, i32
// CPP-LABEL: void add_ui16(
// CPP: AscendC::LocalTensor<uint16_t> [[SRC:v[0-9]+]], uint16_t [[SC:v[0-9]+]], AscendC::LocalTensor<uint16_t> [[DST:v[0-9]+]])
// CPP-NEXT: int32_t v{{[0-9]+}} = [[DST]].GetSize();
// CPP-NEXT: AddsExtend([[DST]], [[SRC]], [[SC]], static_cast<uint32_t>(v{{[0-9]+}}));
// CPP-NEXT: return;
func.func @add_ui16(%lhs: !ascendc.local_tensor<?xui16>, %rhs: ui16, %out: !ascendc.local_tensor<?xui16>) {
  %0 = afir.add %lhs, %rhs outs(%out) {
    afir.operand_descs = [#src, #scalar],
    afir.destination_descs = [#dst], afir.result_descs = [#dst]
  } : (!ascendc.local_tensor<?xui16>, ui16), !ascendc.local_tensor<?xui16> -> !ascendc.local_tensor<?xui16>
  return
}

// IR-LABEL: func.func @add_ui32
// IR: ascendc.adds_extend_reg_base %arg2, %arg0, %arg1, {{.*}} : !ascendc.local_tensor<?xui32>, !ascendc.local_tensor<?xui32>, ui32, i32
// CPP-LABEL: void add_ui32(
// CPP: AscendC::LocalTensor<uint32_t> [[SRC:v[0-9]+]], uint32_t [[SC:v[0-9]+]], AscendC::LocalTensor<uint32_t> [[DST:v[0-9]+]])
// CPP-NEXT: int32_t v{{[0-9]+}} = [[DST]].GetSize();
// CPP-NEXT: AddsExtend([[DST]], [[SRC]], [[SC]], static_cast<uint32_t>(v{{[0-9]+}}));
// CPP-NEXT: return;
func.func @add_ui32(%lhs: !ascendc.local_tensor<?xui32>, %rhs: ui32, %out: !ascendc.local_tensor<?xui32>) {
  %0 = afir.add %lhs, %rhs outs(%out) {
    afir.operand_descs = [#src, #scalar],
    afir.destination_descs = [#dst], afir.result_descs = [#dst]
  } : (!ascendc.local_tensor<?xui32>, ui32), !ascendc.local_tensor<?xui32> -> !ascendc.local_tensor<?xui32>
  return
}

// IR-LABEL: func.func @maximum_ui16
// IR: ascendc.maxs_extend_reg_base %arg2, %arg0, %arg1, {{.*}} : !ascendc.local_tensor<?xui16>, !ascendc.local_tensor<?xui16>, ui16, i32
// CPP-LABEL: void maximum_ui16(
// CPP: AscendC::LocalTensor<uint16_t> [[SRC:v[0-9]+]], uint16_t [[SC:v[0-9]+]], AscendC::LocalTensor<uint16_t> [[DST:v[0-9]+]])
// CPP-NEXT: int32_t v{{[0-9]+}} = [[DST]].GetSize();
// CPP-NEXT: MaxsExtend([[DST]], [[SRC]], [[SC]], static_cast<uint32_t>(v{{[0-9]+}}));
// CPP-NEXT: return;
func.func @maximum_ui16(%lhs: !ascendc.local_tensor<?xui16>, %rhs: ui16, %out: !ascendc.local_tensor<?xui16>) {
  %0 = afir.maximum %rhs, %lhs outs(%out) {
    afir.operand_descs = [#scalar, #src],
    afir.destination_descs = [#dst], afir.result_descs = [#dst]
  } : (ui16, !ascendc.local_tensor<?xui16>), !ascendc.local_tensor<?xui16> -> !ascendc.local_tensor<?xui16>
  return
}

// IR-LABEL: func.func @maximum_ui32
// IR: ascendc.maxs_extend_reg_base %arg2, %arg0, %arg1, {{.*}} : !ascendc.local_tensor<?xui32>, !ascendc.local_tensor<?xui32>, ui32, i32
// CPP-LABEL: void maximum_ui32(
// CPP: AscendC::LocalTensor<uint32_t> [[SRC:v[0-9]+]], uint32_t [[SC:v[0-9]+]], AscendC::LocalTensor<uint32_t> [[DST:v[0-9]+]])
// CPP-NEXT: int32_t v{{[0-9]+}} = [[DST]].GetSize();
// CPP-NEXT: MaxsExtend([[DST]], [[SRC]], [[SC]], static_cast<uint32_t>(v{{[0-9]+}}));
// CPP-NEXT: return;
func.func @maximum_ui32(%lhs: !ascendc.local_tensor<?xui32>, %rhs: ui32, %out: !ascendc.local_tensor<?xui32>) {
  %0 = afir.maximum %rhs, %lhs outs(%out) {
    afir.operand_descs = [#scalar, #src],
    afir.destination_descs = [#dst], afir.result_descs = [#dst]
  } : (ui32, !ascendc.local_tensor<?xui32>), !ascendc.local_tensor<?xui32> -> !ascendc.local_tensor<?xui32>
  return
}

// IR-LABEL: func.func @minimum_ui16
// IR: ascendc.mins_extend_reg_base %arg2, %arg0, %arg1, {{.*}} : !ascendc.local_tensor<?xui16>, !ascendc.local_tensor<?xui16>, ui16, i32
// CPP-LABEL: void minimum_ui16(
// CPP: AscendC::LocalTensor<uint16_t> [[SRC:v[0-9]+]], uint16_t [[SC:v[0-9]+]], AscendC::LocalTensor<uint16_t> [[DST:v[0-9]+]])
// CPP-NEXT: int32_t v{{[0-9]+}} = [[DST]].GetSize();
// CPP-NEXT: MinsExtend([[DST]], [[SRC]], [[SC]], static_cast<uint32_t>(v{{[0-9]+}}));
// CPP-NEXT: return;
func.func @minimum_ui16(%lhs: !ascendc.local_tensor<?xui16>, %rhs: ui16, %out: !ascendc.local_tensor<?xui16>) {
  %0 = afir.minimum %lhs, %rhs outs(%out) {
    afir.operand_descs = [#src, #scalar],
    afir.destination_descs = [#dst], afir.result_descs = [#dst]
  } : (!ascendc.local_tensor<?xui16>, ui16), !ascendc.local_tensor<?xui16> -> !ascendc.local_tensor<?xui16>
  return
}

// IR-LABEL: func.func @minimum_ui32
// IR: ascendc.mins_extend_reg_base %arg2, %arg0, %arg1, {{.*}} : !ascendc.local_tensor<?xui32>, !ascendc.local_tensor<?xui32>, ui32, i32
// CPP-LABEL: void minimum_ui32(
// CPP: AscendC::LocalTensor<uint32_t> [[SRC:v[0-9]+]], uint32_t [[SC:v[0-9]+]], AscendC::LocalTensor<uint32_t> [[DST:v[0-9]+]])
// CPP-NEXT: int32_t v{{[0-9]+}} = [[DST]].GetSize();
// CPP-NEXT: MinsExtend([[DST]], [[SRC]], [[SC]], static_cast<uint32_t>(v{{[0-9]+}}));
// CPP-NEXT: return;
func.func @minimum_ui32(%lhs: !ascendc.local_tensor<?xui32>, %rhs: ui32, %out: !ascendc.local_tensor<?xui32>) {
  %0 = afir.minimum %lhs, %rhs outs(%out) {
    afir.operand_descs = [#src, #scalar],
    afir.destination_descs = [#dst], afir.result_descs = [#dst]
  } : (!ascendc.local_tensor<?xui32>, ui32), !ascendc.local_tensor<?xui32> -> !ascendc.local_tensor<?xui32>
  return
}

// 标量不是函数形参，而是 bridge 为字面量建的 `afir.scalar`：带符号性的整数在 AFIR 层由它
// 承载，下降时才换成后端常量 `emitc.constant`；算子一侧选中的实现不变。字面量按自己的类型
// 印成 C++ 常量，并被那次包装调用读到。
// IR-LABEL: func.func @add_ui32_literal
// IR-NEXT: "emitc.constant"() <{value = 3 : ui32}> : () -> ui32
// IR: ascendc.adds_extend_reg_base %arg1, %arg0, {{.*}}, {{.*}} : !ascendc.local_tensor<?xui32>, !ascendc.local_tensor<?xui32>, ui32, i32
// CPP-LABEL: void add_ui32_literal(
// CPP: AscendC::LocalTensor<uint32_t> [[SRC:v[0-9]+]], AscendC::LocalTensor<uint32_t> [[DST:v[0-9]+]])
// CPP-NEXT: constexpr uint32_t [[SC:v[0-9]+]] = 3;
// CPP-NEXT: int32_t v{{[0-9]+}} = [[DST]].GetSize();
// CPP-NEXT: AddsExtend([[DST]], [[SRC]], [[SC]], static_cast<uint32_t>(v{{[0-9]+}}));
// CPP-NEXT: return;
func.func @add_ui32_literal(%lhs: !ascendc.local_tensor<?xui32>, %out: !ascendc.local_tensor<?xui32>) {
  %k = afir.scalar {value = 3 : ui32} : ui32
  %0 = afir.add %lhs, %k outs(%out) {
    afir.operand_descs = [#src, #scalar],
    afir.destination_descs = [#dst], afir.result_descs = [#dst]
  } : (!ascendc.local_tensor<?xui32>, ui32), !ascendc.local_tensor<?xui32> -> !ascendc.local_tensor<?xui32>
  return
}

// IR-LABEL: func.func @maximum_ui16_literal
// IR-NEXT: "emitc.constant"() <{value = 7 : ui16}> : () -> ui16
// IR: ascendc.maxs_extend_reg_base %arg1, %arg0, {{.*}}, {{.*}} : !ascendc.local_tensor<?xui16>, !ascendc.local_tensor<?xui16>, ui16, i32
// CPP-LABEL: void maximum_ui16_literal(
// CPP: AscendC::LocalTensor<uint16_t> [[SRC:v[0-9]+]], AscendC::LocalTensor<uint16_t> [[DST:v[0-9]+]])
// CPP-NEXT: constexpr uint16_t [[SC:v[0-9]+]] = 7;
// CPP-NEXT: int32_t v{{[0-9]+}} = [[DST]].GetSize();
// CPP-NEXT: MaxsExtend([[DST]], [[SRC]], [[SC]], static_cast<uint32_t>(v{{[0-9]+}}));
// CPP-NEXT: return;
func.func @maximum_ui16_literal(%lhs: !ascendc.local_tensor<?xui16>, %out: !ascendc.local_tensor<?xui16>) {
  %k = afir.scalar {value = 7 : ui16} : ui16
  %0 = afir.maximum %k, %lhs outs(%out) {
    afir.operand_descs = [#scalar, #src],
    afir.destination_descs = [#dst], afir.result_descs = [#dst]
  } : (ui16, !ascendc.local_tensor<?xui16>), !ascendc.local_tensor<?xui16> -> !ascendc.local_tensor<?xui16>
  return
}

// 调用就是函数体里唯一的一句：上面每个函数都把「取长度、调包装、return」三行逐行钉住了。
// 连续 LocalTensor 接口收不了这两种类型，emitter 里也不再有拼出来的 `MicroAPI::` 循环：这两类
// 文本不该出现在 C++ 产物里，由 RUN 行上的 `--implicit-check-not` 对整份输入兜住。
