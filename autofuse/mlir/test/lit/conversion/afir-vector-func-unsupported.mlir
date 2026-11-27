// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not af-opt --afir-vector-func-to-reg-base --split-input-file %s 2>&1 | FileCheck %s

// 这一期（M2）收不了的几种 `afir.vector_func`。一格只违反一条：一格同时违反两条时，
// 两条判据互相遮蔽——去掉其中一条也仍然红，变异验证就失去意义。做不到分开的那一格要把
// 「为什么分不开」写在格子上（第五格）。
//
// 这几条全部报在能力层（`emitCapabilityShortfall`），不是 verifier。两者的区别不是
// 措辞：能力不足会被 `RunMlirCodegen` 判成 `kNotSupported` 并整图干净回落给 legacy，而
// verifier 失败是硬失败。所以「这一期还收不了」只能走这条路——写成 verifier 的话，
// 今天能跑的图会当场红。
//
// 验收的另一半不在这里：真的回落了由 legacy 接管表看（`strict` 档的
// `shortfall=yes`），这里只钉「拒收的理由说得对、而且指得到那个算子」。

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// 一、region 中的 Pow 尚无寄存器映射，应报告能力不足。
// CHECK: UnsupportedOperation: 'afir.pow' has no register-form lowering
func.func @no_register_form(%a: !ascendc.local_tensor<?xf32>, %b: !ascendc.local_tensor<?xf32>,
                            %y: !ascendc.local_tensor<?xf32>) {
  %0 = afir.vector_func srcs(%a, %b) dsts(%y) attributes {
    afir.operand_descs = [#in, #in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xf32>, %lb: tensor<?xf32>):
    %res = afir.pow %la, %lb {afir.operand_descs = [#in, #in], afir.result_descs = [#out]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %res : tensor<?xf32>
  } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>)
      -> !ascendc.local_tensor<?xf32>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// 二、多输出。可表达（算子收 variadic dsts），但这一期的下降只生成一条 `StoreAlign`，
// 所以在能力层拒掉——不在 verifier 里拒：多输出是 M4 的事，不是非法 IR。
// CHECK: UnsupportedOperation: afir.vector_func with 2 destinations has no register-form lowering yet
func.func @two_destinations(%a: !ascendc.local_tensor<?xf32>, %y0: !ascendc.local_tensor<?xf32>,
                            %y1: !ascendc.local_tensor<?xf32>) {
  %0:2 = afir.vector_func srcs(%a) dsts(%y0, %y1) attributes {
    afir.operand_descs = [#in], afir.destination_descs = [#out, #out], afir.result_descs = [#out, #out]
  } {
  ^bb0(%la: tensor<?xf32>):
    %res = afir.abs %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
        : tensor<?xf32> -> tensor<?xf32>
    afir.vector_func_yield %res, %res : tensor<?xf32>, tensor<?xf32>
  } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>)
      -> (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>)
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// 三、`afir.cast` 这一对 dtype 没有 trait。`Reg::Cast` 的第三个模板实参是个
// `const CastTrait&`，挑哪一个取决于 (输入, 输出)（主要是舍入与饱和），表抄自 legacy 的
// `micro_cast_api_call.cpp:18`。表外的对**不回落到缺省 trait**：legacy 查不到时用
// `cast_trait_none`，而那条路上「没列过」与「列过且是 none」分不开，这里宁可干净回落。
//
// f16 -> i32 在 AFIR 的 reg 侧对表里（`Ops/Cast.cpp` 的 `getRegBasedCastPairs`），
// 所以 verifier 放它过去；trait 那张表里没有这一行，于是拒在能力层。两张表不是同一件事：
// 前者说「这一对转得出来」，后者说「这一对该怎么舍入」。
//
// 这一格只违反这一条：lane 宽度是 i32，边界 f16 有一步到位的 `DIST_UNPACK_B16`、
// 边界 i32 与 lane 等宽，所以搬运那两条都过得去。
// CHECK: UnsupportedOperation: 'afir.cast' from 'f16' to 'i32' has no register-form cast trait
func.func @cast_without_trait(%a: !ascendc.local_tensor<?xf16>, %y: !ascendc.local_tensor<?xi32>) {
  %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
    afir.operand_descs = [#in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xf16>):
    %res = afir.cast %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
        : tensor<?xf16> -> tensor<?xi32>
    afir.vector_func_yield %res : tensor<?xi32>
  } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xi32>) -> !ascendc.local_tensor<?xi32>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// 四、算的那一步不在最宽的 dtype 上。lane 的宽度由 VF 里最宽的元素类型定（这一格是
// f32），而窄 dtype 的寄存器里元素是按宽槽位排的（边界被 `DIST_UNPACK_B16` 撑开过）——
// `Reg::Abs(RegTensor<half>&, mask)` 读的是 `VL/2` 个 half，与 `UpdateMask<float>` 给的
// 那张 mask 对不上几个 lane 活着。这件事只能在设备上量，量不出来就不该答「支持」。
//
// 今天跨 dtype 的那几格都是「先 cast 到宽的、算完再 cast 回去」，所以这一条不挡它们。
// CHECK: UnsupportedOperation: 'afir.abs' computes in 'f16' while the vector function's lanes are 'f32' wide
func.func @compute_narrower_than_lanes(%a: !ascendc.local_tensor<?xf16>, %y: !ascendc.local_tensor<?xf32>) {
  %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
    afir.operand_descs = [#in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xf16>):
    %narrow = afir.abs %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
        : tensor<?xf16> -> tensor<?xf16>
    %res = afir.cast %narrow {afir.operand_descs = [#out], afir.result_descs = [#out]}
        : tensor<?xf16> -> tensor<?xf32>
    afir.vector_func_yield %res : tensor<?xf32>
  } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf32>) -> !ascendc.local_tensor<?xf32>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// 五、窄边界撑不进 lane 槽位：一条 `LoadDist` 不够。legacy 在这种时候会再补一串
// `Reg::UnPack`（`micro_load_api_call.cpp:57` 那个 while），那要多两个 op 与一个临时
// 寄存器，归后续。
//
// 这一格必然同时违反取与存两条：lane 宽度 8 字节要由 region 内某个结果带出来，而
// `afir.vector_func` 的边界 dtype 域里根本没有 8 字节的那一位
// （`VectorFunc.td:50`），所以两侧边界都只能是窄的——分不开。能钉的是报哪一条：
// 判据的顺序决定诊断说的是取而不是存。
// CHECK: UnsupportedOperation: afir.vector_func loads 'f32' into 'i64' lanes
func.func @unpack_sequence_needed(%a: !ascendc.local_tensor<?xf32>, %y: !ascendc.local_tensor<?xi32>) {
  %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
    afir.operand_descs = [#in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xf32>):
    %wide = afir.cast %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
        : tensor<?xf32> -> tensor<?xi64>
    %res = afir.cast %wide {afir.operand_descs = [#out], afir.result_descs = [#out]}
        : tensor<?xi64> -> tensor<?xi32>
    afir.vector_func_yield %res : tensor<?xi32>
  } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xi32>) -> !ascendc.local_tensor<?xi32>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// 六、谓词被当成一段值消费。`afir.eq` 在寄存器形态下产出的是一条 `MaskReg`，而这一格把
// 它直接当成 VF 的输出存回 UB。legacy 在消费者不是 `Select` 的 mask 时补一条
// `Reg::Duplicate(dst, static_cast<T>(1), mask)` 把谓词物化回 0/1 的向量
// （`micro_compare_api_call.cpp:48`），本期不建模——判据与 legacy 的
// `ShouldInitAsMaskReg`（`vec_func_call.cpp:264`）同一条。
//
// 这一格只违反这一条：lane 宽度是 f32，两路边界与它等宽，目的地 ui8 有一步到位的
// `DIST_PACK4_B32`，`eq` 也算在最宽的 f32 上。
// CHECK: UnsupportedOperation: 'afir.eq' produces a predicate register that is consumed as a value
func.func @predicate_used_as_value(%a: !ascendc.local_tensor<?xf32>, %b: !ascendc.local_tensor<?xf32>,
                                   %y: !ascendc.local_tensor<?xui8>) {
  %0 = afir.vector_func srcs(%a, %b) dsts(%y) attributes {
    afir.operand_descs = [#in, #in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xf32>, %lb: tensor<?xf32>):
    %res = afir.eq %la, %lb {afir.operand_descs = [#in, #in], afir.result_descs = [#out]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xui8>
    afir.vector_func_yield %res : tensor<?xui8>
  } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>)
      -> !ascendc.local_tensor<?xui8>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// 七、`afir.where` 的条件不是谓词，而是从 UB 取进来的一段 ui8 真值。legacy 在这种时候
// 先补一条 `Reg::CompareScalar<uint8_t, CMPMODE::NE>(tmp, reg, 0, pReg)` 把它转成谓词
// （`micro_where_api_call.cpp:30`），本期不建模。
//
// 这一格只违反这一条：lane 宽度是 f32，ui8 边界有一步到位的 `DIST_UNPACK4_B8`，两路
// 候选与结果都在 f32 上。
// CHECK: UnsupportedOperation: 'afir.where' takes a condition that is not a predicate register
func.func @condition_is_not_a_predicate(%c: !ascendc.local_tensor<?xui8>, %t: !ascendc.local_tensor<?xf32>,
                                        %e: !ascendc.local_tensor<?xf32>, %y: !ascendc.local_tensor<?xf32>) {
  %0 = afir.vector_func srcs(%c, %t, %e) dsts(%y) attributes {
    afir.operand_descs = [#in, #in, #in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%lc: tensor<?xui8>, %lt: tensor<?xf32>, %le: tensor<?xf32>):
    %res = afir.where %lc, %lt, %le {afir.operand_descs = [#in, #in, #in], afir.result_descs = [#out]}
        : (tensor<?xui8>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %res : tensor<?xf32>
  } : (!ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>,
       !ascendc.local_tensor<?xf32>) -> !ascendc.local_tensor<?xf32>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>

// 八、这一行在寄存器形态下不收这个元素类型。每个 `AscendC::Reg` API 有自己的类型白名单，与
// AFIR 算子的 ODS 域不是一回事：`relu` 的 ODS 收 ui8，`Reg::Relu` 不收；`neg` 的 ODS 收 bf16，
// `Reg::Neg` 不收。表里每一行带着量出来的 dtype 域（`AFIRVectorFuncToRegBase.cpp` 的
// `kRegOp*ElementTypes`），能力判定统一查表。放行的后果是编 kernel 时 `static_assert`——那是
// 硬失败，比回落晚一整截。
//
// 下面一格一个格子，覆盖表里今天收不了的每一种组合。每格都是单违反：边界与计算同一个 dtype，
// 等宽的取与存不带 dist，除了 dtype 域那一条，其余判据都过得去，所以每格报的都是同一条诊断。
// 顺序与表里的行序相同。
//
// 调度器会把 bf16 的 `exp` / `ln` / `sqrt` 升到 f32、把 int8 的乘法升到 int16，所以这几格只有 IR
// 本身已经在那个 dtype 上算时才走到。

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// `Reg::Div`：`div` / `truediv` 两行共用它，只收 half 与 float。
// CHECK: UnsupportedOperation: 'afir.truediv' has no register-form implementation for element type 'bf16'
func.func @truediv_in_bfloat16(%a: !ascendc.local_tensor<?xbf16>, %b: !ascendc.local_tensor<?xbf16>,
                                %y: !ascendc.local_tensor<?xbf16>) {
  %0 = afir.vector_func srcs(%a, %b) dsts(%y) attributes {
    afir.operand_descs = [#in, #in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xbf16>, %lb: tensor<?xbf16>):
    %res = afir.truediv %la, %lb {afir.operand_descs = [#in, #in], afir.result_descs = [#out]}
        : (tensor<?xbf16>, tensor<?xbf16>) -> tensor<?xbf16>
    afir.vector_func_yield %res : tensor<?xbf16>
  } : (!ascendc.local_tensor<?xbf16>, !ascendc.local_tensor<?xbf16>,
        !ascendc.local_tensor<?xbf16>) -> !ascendc.local_tensor<?xbf16>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// `Reg::Abs` / `Neg`：浮点与带符号整数，不收 bf16。
// CHECK: UnsupportedOperation: 'afir.abs' has no register-form implementation for element type 'bf16'
func.func @abs_in_bfloat16(%a: !ascendc.local_tensor<?xbf16>, %y: !ascendc.local_tensor<?xbf16>) {
  %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
    afir.operand_descs = [#in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xbf16>):
    %res = afir.abs %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
        : tensor<?xbf16> -> tensor<?xbf16>
    afir.vector_func_yield %res : tensor<?xbf16>
  } : (!ascendc.local_tensor<?xbf16>, !ascendc.local_tensor<?xbf16>) -> !ascendc.local_tensor<?xbf16>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// 同上，`Reg::Abs` 也不收 ui8。
// CHECK: UnsupportedOperation: 'afir.abs' has no register-form implementation for element type 'ui8'
func.func @abs_in_uint8(%a: !ascendc.local_tensor<?xui8>, %y: !ascendc.local_tensor<?xui8>) {
  %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
    afir.operand_descs = [#in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xui8>):
    %res = afir.abs %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
        : tensor<?xui8> -> tensor<?xui8>
    afir.vector_func_yield %res : tensor<?xui8>
  } : (!ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xui8>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// `Reg::Exp` / `Ln` / `Sqrt`：只收 half 与 float。
// CHECK: UnsupportedOperation: 'afir.exp' has no register-form implementation for element type 'bf16'
func.func @exp_in_bfloat16(%a: !ascendc.local_tensor<?xbf16>, %y: !ascendc.local_tensor<?xbf16>) {
  %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
    afir.operand_descs = [#in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xbf16>):
    %res = afir.exp %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
        : tensor<?xbf16> -> tensor<?xbf16>
    afir.vector_func_yield %res : tensor<?xbf16>
  } : (!ascendc.local_tensor<?xbf16>, !ascendc.local_tensor<?xbf16>) -> !ascendc.local_tensor<?xbf16>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// 同上。
// CHECK: UnsupportedOperation: 'afir.ln' has no register-form implementation for element type 'bf16'
func.func @ln_in_bfloat16(%a: !ascendc.local_tensor<?xbf16>, %y: !ascendc.local_tensor<?xbf16>) {
  %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
    afir.operand_descs = [#in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xbf16>):
    %res = afir.ln %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
        : tensor<?xbf16> -> tensor<?xbf16>
    afir.vector_func_yield %res : tensor<?xbf16>
  } : (!ascendc.local_tensor<?xbf16>, !ascendc.local_tensor<?xbf16>) -> !ascendc.local_tensor<?xbf16>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// 同上。
// CHECK: UnsupportedOperation: 'afir.sqrt' has no register-form implementation for element type 'bf16'
func.func @sqrt_in_bfloat16(%a: !ascendc.local_tensor<?xbf16>, %y: !ascendc.local_tensor<?xbf16>) {
  %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
    afir.operand_descs = [#in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xbf16>):
    %res = afir.sqrt %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
        : tensor<?xbf16> -> tensor<?xbf16>
    afir.vector_func_yield %res : tensor<?xbf16>
  } : (!ascendc.local_tensor<?xbf16>, !ascendc.local_tensor<?xbf16>) -> !ascendc.local_tensor<?xbf16>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// `Reg::Relu`：浮点与 32 / 64 位整数，不收 ui8。
// CHECK: UnsupportedOperation: 'afir.relu' has no register-form implementation for element type 'ui8'
func.func @relu_in_uint8(%a: !ascendc.local_tensor<?xui8>, %y: !ascendc.local_tensor<?xui8>) {
  %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
    afir.operand_descs = [#in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xui8>):
    %res = afir.relu %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
        : tensor<?xui8> -> tensor<?xui8>
    afir.vector_func_yield %res : tensor<?xui8>
  } : (!ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xui8>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// 同 `abs`：`Reg::Neg` 不收 bf16。
// CHECK: UnsupportedOperation: 'afir.neg' has no register-form implementation for element type 'bf16'
func.func @neg_in_bfloat16(%a: !ascendc.local_tensor<?xbf16>, %y: !ascendc.local_tensor<?xbf16>) {
  %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
    afir.operand_descs = [#in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xbf16>):
    %res = afir.neg %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
        : tensor<?xbf16> -> tensor<?xbf16>
    afir.vector_func_yield %res : tensor<?xbf16>
  } : (!ascendc.local_tensor<?xbf16>, !ascendc.local_tensor<?xbf16>) -> !ascendc.local_tensor<?xbf16>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// `Reg::Mul`：除两种 8 位整数之外都收。
// CHECK: UnsupportedOperation: 'afir.mul' has no register-form implementation for element type 'i8'
func.func @mul_computes_in_int8(%a: !ascendc.local_tensor<?xi8>, %b: !ascendc.local_tensor<?xi8>,
                                 %y: !ascendc.local_tensor<?xi8>) {
  %0 = afir.vector_func srcs(%a, %b) dsts(%y) attributes {
    afir.operand_descs = [#in, #in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xi8>, %lb: tensor<?xi8>):
    %res = afir.mul %la, %lb {afir.operand_descs = [#in, #in], afir.result_descs = [#out]}
        : (tensor<?xi8>, tensor<?xi8>) -> tensor<?xi8>
    afir.vector_func_yield %res : tensor<?xi8>
  } : (!ascendc.local_tensor<?xi8>, !ascendc.local_tensor<?xi8>,
        !ascendc.local_tensor<?xi8>) -> !ascendc.local_tensor<?xi8>
  return
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

// 同上，无符号的那一位。ui8 早就在 VF 的边界 dtype 域里，不是哪一次放开带进来的。
// CHECK: UnsupportedOperation: 'afir.mul' has no register-form implementation for element type 'ui8'
func.func @mul_computes_in_uint8(%a: !ascendc.local_tensor<?xui8>, %b: !ascendc.local_tensor<?xui8>,
                                  %y: !ascendc.local_tensor<?xui8>) {
  %0 = afir.vector_func srcs(%a, %b) dsts(%y) attributes {
    afir.operand_descs = [#in, #in], afir.destination_descs = [#out], afir.result_descs = [#out]
  } {
  ^bb0(%la: tensor<?xui8>, %lb: tensor<?xui8>):
    %res = afir.mul %la, %lb {afir.operand_descs = [#in, #in], afir.result_descs = [#out]}
        : (tensor<?xui8>, tensor<?xui8>) -> tensor<?xui8>
    afir.vector_func_yield %res : tensor<?xui8>
  } : (!ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>,
        !ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xui8>
  return
}
