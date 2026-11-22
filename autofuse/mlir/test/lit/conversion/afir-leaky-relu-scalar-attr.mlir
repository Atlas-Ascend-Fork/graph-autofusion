// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT

// `afir.leaky_relu`：这一类里第一个**由 ODS 属性变成 API 实参**的 op。斜率在 AFIR 上是
// 属性不是操作数——它不进三段划分、不占缓冲——conversion 里现建一个常量塞进调用。
//
// 两张支持表给的是同一个 `AscendC::LeakyRelu`，都不要 tmp buffer、都不引扩展头，所以
// 只有一个 implementation、家族恒为 `Any`：两个 target 走的是同一条路，IR 那两条 RUN
// 用的是同一组 check。
//
// 常量按 tile 的元素类型建。包装是 `LeakyRelu(const LocalTensor<T>&, const
// LocalTensor<T>&, const T&, uint32_t)`：给 f16 的 tile 配一个 f32 常量，T 会同时被推成
// half 和 float，一个重载都对不上。下面 f16 那一份钉的就是这条。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @leaky_relu_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.leaky_relu %input outs(%destination) {
      negative_slop = 1.250000e-01 : f32,
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @leaky_relu_f16(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>) {
    %result = afir.leaky_relu %input outs(%destination) {
      negative_slop = 1.250000e-01 : f32,
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }
}

// IR-LABEL: func.func @leaky_relu_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR: %[[SLOPE:.*]] = arith.constant 1.250000e-01 : f32
// IR: ascendc.leaky_relu_l2 %arg1, %arg0, %[[SLOPE]], %[[COUNT]]
// IR-NOT: afir.leaky_relu

// 斜率跟着 tile 走：f16 的这一份拿到的是 f16 常量，不是 f32。这一条红了就说明常量建成了
// 固定的 f32——那会让 `LeakyRelu` 的 T 同时被推成 half 和 float，device 上一个重载都对不上。
// IR-LABEL: func.func @leaky_relu_f16
// IR: %[[SLOPE16:.*]] = arith.constant 1.250000e-01 : f16
// IR: ascendc.leaky_relu_l2 %arg1, %arg0, %[[SLOPE16]], %{{.*}}

// 常量按元素类型生成：f32 那份是 `(float)`，f16 那份是一个 `half` 变量。
//
// 字面量写全，并且把变量名绑进调用。 emitter 打的是定长的科学计数形式
// （`1.250000000e-01` 就是 0.125），原先这两条写的是 `(float)1.25` / `= 1.25`，
// 而 FileCheck 是子串匹配——`1.25` 恰好是 `1.250000000e-01` 的前缀，于是斜率改成 1.25
// 或 12.5 它照样绿，值本身其实没人钉。f32 与 f16 的位数还不一样（`e-01` 前面一个是
// 九位、一个是五位），所以两条各写各的，不能共用一个模式。
// CPP-LABEL: void leaky_relu_f32(
// CPP: constexpr float [[SLOPE:[A-Za-z0-9_]+]] = (float)1.250000000e-01;
// CPP: AscendC::LeakyRelu<float, 1>({{[^,]*}}, {{[^,]*}}, [[SLOPE]],
// CPP-LABEL: void leaky_relu_f16(
// CPP: half [[SLOPE16:[A-Za-z0-9_]+]] = 1.25000e-01;
// CPP: AscendC::LeakyRelu<half, 1>({{[^,]*}}, {{[^,]*}}, [[SLOPE16]],
//
// `<T, 1>` 里的 1 是 `isSetMask`，`1` 与 legacy 不带模板实参时用的默认值同义。
// 这一位曾经打的是 `0`：ODS 上它原本是 `UnitAttr`，缺省（属性不写）就是 `false`，而
// AscendC 声明上的缺省是 `true`——两个缺省极性正好相反，没人设它就一路落在错的那侧。
// `false` 的含义是「调用方已经把 mask 置成 count 模式并设好 count」，API 于是整段跳过
// 设 mask、按 `repeatTimes = 1` 直生成一条指令；目前没有任何 pass 提前设 mask，生成就
// 只算一个 repeat。详见 `Base.td` 的 `VecScalarL2Op`。
//
// **这条 CHECK 红了先看那个缺省有没有被改回 `UnitAttr`**，不要就地把 `1` 改成 `0`。

// SELECT: ascend.implementation = {id = "leaky_relu.tensor", kind = "ascendc_api"}
