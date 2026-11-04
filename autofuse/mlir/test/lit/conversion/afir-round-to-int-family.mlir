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
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %s 2>&1 | FileCheck %s --check-prefix=WRONGTARGET

// step 2a 接通的两个：`ceil2int` 与 `floor_to_int`。两个的**结果 dtype 规则相同**
// （`ResultElementRule::Int32`，输出恒 i32 与输入无关）而**供给不同**，这一份要钉的就是
// 这两件事各自对了。
//
// 供给那一半最容易挑错，而且挑错不会在这一层报错：
//
//   - 对照表的 `reg_api_name` 列对这一族给的是 legacy 的 `GetApiName()`，**不是发出去的
//     函数名**。CANN 全树里没有 `AscendC::FloorToInt`——照那个名字建一个 op 会发出一个
//     不存在的调用，要到设备编译器才报。`CPP-NOT: FloorToInt(` 盯这一条。
//   - `ceil2int` 反过来：它有包装（`Ceil2IntExtend`，在 fork 的 `cast.h` 里转发给
//     `CastExtend<..., CAST_CEIL>`），**不能**退化成标准 `AscendC::Cast`，否则 RoundMode
//     就没人写了。`CPP-NOT: AscendC::Cast` 在那一格盯这一条。
//
// 结果 dtype 那一半由 `IR` 组的 `i32` 落点钉：输入 f32、目的地 i32，而这一族的 ODS
// `resultRule` 要是写成缺省的 `Same`，op 在 verifier 上就过不去。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @ceil2int_f32(%input: !ascendc.local_tensor<?xf32>,
                          %destination: !ascendc.local_tensor<?xi32>) {
    %result = afir.ceil2int %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xi32>
        -> !ascendc.local_tensor<?xi32>
    return
  }

  func.func @floor_to_int_f32(%input: !ascendc.local_tensor<?xf32>,
                              %destination: !ascendc.local_tensor<?xi32>) {
    %result = afir.floor_to_int %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xi32>
        -> !ascendc.local_tensor<?xi32>
    return
  }

}

// 包装那一支：三个实参（第三个是 calCount，两个 stride 由 emitter 打字面量）。
// IR-LABEL: func.func @ceil2int_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.ceil2int_extend_contiguous_reg_base %arg1, %arg0, %[[COUNT]]
// IR-NOT: ascendc.cast_l2
// IR-NOT: afir.ceil2int

// 标准调用那一支：带一个 `roundMode` 属性，而且必须是 CAST_FLOOR。
// IR-LABEL: func.func @floor_to_int_f32
// IR: ascendc.cast_l2 %arg1, %arg0
// `roundMode` 在 IR 里打的是枚举的原始值，2 就是 `CAST_FLOOR`
// （`Core/Attributes.td` 的 `I32EnumAttrCase<"CAST_FLOOR", 2>`）。可读名由下面 CPP 那一组
// 钉，所以这里钉数值不是在钉一个看不懂的东西——两组合起来才说明「数值与名字都对」。
// IR-SAME: roundMode = 2 : i32
// IR-NOT: ascendc.ceil2int
// IR-NOT: afir.floor_to_int

// CPP-LABEL: void ceil2int_f32(
// CPP: {{^ *}}Ceil2IntExtend({{[^,]*}}, {{[^,]*}}, {static_cast<uint32_t>(
// CPP-NOT: AscendC::Cast

// CPP-LABEL: void floor_to_int_f32(
// 显式模板实参 `<int32_t, float>` 是上游 L2 cast emitter 的行为，legacy 的
// `floor_to_int_api_call.cpp` 不打（靠推导）。两者语义相同、文本不同——与 `isSetMask`、
// `isReuseSource` 同一类已被接受的偏差，所以 CHECK 收 `Cast<`。
// CPP: AscendC::Cast<{{[^>]*}}>({{[^,]*}}, {{[^,]*}}, AscendC::RoundMode::CAST_FLOOR,
// CPP-NOT: FloorToInt(
// CPP-NOT: FloorToIntExtend

// SELECT: ascend.implementation = {id = "ceil2int.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "floor_to_int.tensor", kind = "ascendc_api"}

// 两个都只有 reg 侧的 implementation，mem 目标上一条都选不出来。
// WRONGTARGET: no supported
