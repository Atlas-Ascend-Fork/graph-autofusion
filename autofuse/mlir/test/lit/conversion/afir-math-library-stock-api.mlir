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

// `afir.tanh` / `afir.erf` 的 mem 侧：标准的 `AscendC::Tanh` / `AscendC::Erf`，要 tmp
// buffer，**一个扩展头都不引**。
//
// 这份用例要钉的是 `isReuseSource`。这一族的 ascendc op 带一个必填的 `isReuseSource`
// 操作数，被 emitter 当**模板实参**打印；lowering 物化一个 `arith.constant false`，
// 它经 `printConstantOp` 打成 `constexpr bool c0_i1 = false;`，再被填进
// `<float, c0_i1>` 那一位。下面 CPP 那两条 check 写全了模板实参，就是为了「常量没被
// 物化」或者「模板实参位空了」当场红——只写 `AscendC::Tanh(` 的话两种都照样绿。
//
// 它证不了的是那份 C++ 真的编得过（lit 不编译发出来的文本），那一条由 CPU E2E 的
// `load_tanh_store` 钉。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @tanh_stock_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>,
                            %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.tanh %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @erf_stock_f16(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>,
                           %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.erf %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf16>
    return
  }
}

// IR-LABEL: func.func @tanh_stock_f32
// IR-DAG: %[[FALSE:.*]] = arith.constant false
// IR-DAG: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR: ascendc.tanh %arg1, %arg0, %arg2, %[[COUNT]], %[[FALSE]]
// IR-NOT: ascendc.tanh_extend_reg_base
// IR-NOT: afir.tanh

// IR-LABEL: func.func @erf_stock_f16
// IR: ascendc.erf %arg1, %arg0, %arg2
// IR-NOT: ascendc.erf_extend_reg_base
// IR-NOT: afir.erf

// 模板实参写全，而且**与上一行那个常量同名**：`[[FLAG]]` 绑的就是 `constexpr bool` 那
// 个名字，常量没物化、或者名字没被填进模板实参位，这两行都不匹配。
//
// 最后一个实参是 `v4` 本身，**不带 `static_cast<uint32_t>`**——那个强转出自
// `printApiExtensionUnary`（包装那条路），Math 一族的打印器不加，直接把
// `local_tensor.get_size` 的 `int32_t` 传给形参 `const uint32_t calCount`。
// 隐式转换，编得过（由 load_tanh_store 证），别照抄 sign / remainder 那几份的形状。
// CPP-LABEL: void tanh_stock_f32(
// CPP: constexpr bool [[FLAG:.*]] = false;
// CPP: {{^ *}}AscendC::Tanh<float, [[FLAG]]>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,)]*}});

// CPP-LABEL: void erf_stock_f16(
// CPP: constexpr bool [[ERFFLAG:.*]] = false;
// CPP: {{^ *}}AscendC::Erf<half, [[ERFFLAG]]>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,)]*}});

// SELECT: ascend.implementation = {id = "tanh.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "erf.tensor", kind = "ascendc_api"}
