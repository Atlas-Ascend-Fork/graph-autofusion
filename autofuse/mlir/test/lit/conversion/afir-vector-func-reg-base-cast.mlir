// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-vector-func-to-reg-base %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-vector-func-to-reg-base %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP

// 中途换 dtype 的 `afir.vector_func`（M2）。同 dtype 那一格在
// `afir-vector-func-reg-base.mlir`，这里只钉**跨 dtype 多出来的那三件事**：
//
//   1. lane 的宽度由最宽的那个元素类型定，不由边界定。这一格是 half 进、half 出、
//      中间走 float，所以 `UpdateMask` 是 `<float>` 而不是 `<half>`——写错的后果是每趟
//      多算一倍的 lane，而尾段之外的元素照样算对，只有尾巴和越界会暴露。
//   2. 窄边界的搬运带 dist：取进来 `DIST_UNPACK_B16`（把 half 撑进 4 字节的 lane
//      槽位）、存回去 `DIST_PACK_B32`（压回 half）。不带就是 `DIST_NORM`——编得过、
//      跑得完、静默算错（每个 lane 读的是相邻两个 half 而不是一个）。
//   3. `Reg::Cast` 的三个模板实参：`<OutT, InT, trait>` 一个都不能省，而 `trait` 是
//      `api_regbase/utils.h` 里那个具名 `constexpr` 的名字，一来一回挑的是**不同**的两个
//      （舍入方向不一样）。
//
// 寄存器声明的类型也在这里钉：每个值按自己的 dtype 声明，不按 lane 宽度——`RegTensor<half>`
// 与 `RegTensor<float>` 是同一个物理寄存器的两种读法，而 `Cast` 要的正是对得上的那两个。

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @vector_func_cast_abs_cast(%a: !ascendc.local_tensor<?xf16>, %y: !ascendc.local_tensor<?xf16>) {
    %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
      afir.operand_descs = [#in],
      afir.destination_descs = [#out],
      afir.result_descs = [#out]
    } {
    ^bb0(%la: tensor<?xf16>):
      %wide = afir.cast %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
          : tensor<?xf16> -> tensor<?xf32>
      %res = afir.abs %wide {afir.operand_descs = [#out], afir.result_descs = [#out]}
          : tensor<?xf32> -> tensor<?xf32>
      %narrow = afir.cast %res {afir.operand_descs = [#out], afir.result_descs = [#out]}
          : tensor<?xf32> -> tensor<?xf16>
      afir.vector_func_yield %narrow : tensor<?xf16>
    } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>) -> !ascendc.local_tensor<?xf16>
    return
  }
}

// IR-LABEL: func.func @vector_func_cast_abs_cast

// 一个块参数加三个中间结果，恰好四条声明。
// 类型钉在下面的 CPP 段（`RegTensor<half>` / `RegTensor<float>`）：类型写错不会少一条
// 声明，所以条数与类型要分开钉，而 FileCheck 的一遍扫描里「数条数」与「逐条看类型」两个
// 写法会互相吃掉匹配位置——数在 IR 这一遍，类型在 CPP 那一遍。
// IR: emitasc.vec_scope
// IR-COUNT-4: ascendc.reg_tensor
// IR-NOT: ascendc.reg_tensor

// 恰好一层 lane 循环，mask 按最宽的那个 dtype 更新。
// IR-COUNT-1: emitasc.vf_for
// IR-NOT: emitasc.vf_for
// IR: ascendc.update_mask f32

// 取一次、转一次、算一次、再转一次、存一次。
//
// 两条 `dist` 在 IR 里是裸整数：`AscendC_LoadDist` / `AscendC_StoreDist` 是
// `I32EnumAttr`，没有包成 `EnumAttr`，所以打印出来是枚举值而不是名字（既有的
// `ascendc.data_copy_vld_reg` 也是这样）。14 是 `DIST_UNPACK_B16`、7 是 `DIST_PACK_B32`
// ——名字那一侧由下面的 CPP 段钉，那也是真正会被编译器读到的那一份。
// IR: ascendc.load_align_reg {{.*}}dist = 14
// IR: ascendc.cast_reg {{.*}}trait = "cast_trait_bf16_2_float"
// IR: ascendc.abs_reg
// IR: ascendc.cast_reg {{.*}}trait = "cast_trait_float_2_half"
// IR: ascendc.store_align_reg {{.*}}dist = 7

// IR-NOT: afir.vector_func
// IR-NOT: afir.cast

// 生成的 C++。`UpdateMask<float>`、两条 dist、两个方向各自的 trait。
// CPP-LABEL: void vector_func_cast_abs_cast(
// CPP: __VEC_SCOPE__
// 四条声明，各按自己的 dtype：half 进、中间两步 float、half 出。
// CPP: AscendC::Reg::RegTensor<half>
// CPP: AscendC::Reg::RegTensor<float>
// CPP: AscendC::Reg::RegTensor<float>
// CPP: AscendC::Reg::RegTensor<half>
// CPP: AscendC::Reg::UpdateMask<float>(
// CPP: AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
// CPP: AscendC::Reg::Cast<float, half, cast_trait_bf16_2_float>(
// CPP: AscendC::Reg::Abs(
// CPP: AscendC::Reg::Cast<half, float, cast_trait_float_2_half>(
// CPP: AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_PACK_B32>(
