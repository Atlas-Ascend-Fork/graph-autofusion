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

// `afir.rsqrt` 的 mem 侧：`RsqrtExtend(dst, src, blk_tensor_with_value_1, tmp_buf, size)`。
//
// 这是头一个要「预先填好 1 的常量 block」的 op。 那一块不是第二块 tmp buffer——
// AFIR 上根本没有它，它是 lowering 在 kernel 序言里现造的一块 32 字节常量，元素类型由
// 包装的形参钉死（这里恒为 `float`，与正在算的 dtype 无关）。
//
// 第二个函数钉的是共用：按 (值, dtype) 整个 kernel 一份。ATT 侧的 UB 预算
// （`CalcExtraTmpBufForAscGraph`）就是按这个规则去重后每块 32 字节记的，一个 op 一块
// 会超预算，而超出去的症状是运行期 UB 越界，不是编译报错。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @rsqrt_extend_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>,
                              %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.rsqrt %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  // 同一个 kernel 里两个 rsqrt：常量 block 只能有一块。
  func.func @rsqrt_extend_f16_twice(%input: !ascendc.local_tensor<?xf16>, %mid: !ascendc.local_tensor<?xf16>,
                                    %destination: !ascendc.local_tensor<?xf16>,
                                    %tmp: !ascendc.local_tensor<?xui8>) {
    %first = afir.rsqrt %input outs(%mid) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf16>
    %second = afir.rsqrt %first outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#compute_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf16>
    return
  }
}

// IR-LABEL: func.func @rsqrt_extend_f32
// IR: %[[BUF:.*]] = ascendc.tbuf : <veccalc>
// IR: ascendc.pipe.init_buffer {{.*}}, %[[BUF]], %c32
// IR: %[[BLK:.*]] = ascendc.tbuf.get_tensor %[[BUF]] {ascendc.const_block = 1.000000e+00 : f32}
// IR: ascendc.duplicate_l2 %[[BLK]]
// IR-NEXT: ascendc.pipe_barrier pipe_v
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.rsqrt_extend %arg1, %arg0, %[[BLK]], %arg2, %[[COUNT]]
// IR-NOT: afir.rsqrt

// 常量 block 的元素类型跟着包装走，不跟着 tile 走：f16 的 rsqrt 用的仍然是 f32 那一块。
// 两个 op 共用一块，所以整个函数里 `duplicate_l2` 只有一条。
// IR-LABEL: func.func @rsqrt_extend_f16_twice
// IR: ascendc.tbuf.get_tensor {{.*}} {ascendc.const_block = 1.000000e+00 : f32}
// IR-COUNT-1: ascendc.duplicate_l2
// IR-COUNT-2: ascendc.rsqrt_extend
// IR-NOT: ascendc.duplicate_l2

// 全局函数、不带 `AscendC::` 限定、元素类型是显式模板实参——与其余包装一致；常量 block
// 排在 tmp buffer 之前，与包装的形参顺序一致。
// CPP-LABEL: void rsqrt_extend_f32(
// CPP: AscendC::Duplicate(
// CPP: RsqrtExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT-COUNT-3: ascend.implementation = {id = "rsqrt.tensor.extend", kind = "ascendc_api"}
