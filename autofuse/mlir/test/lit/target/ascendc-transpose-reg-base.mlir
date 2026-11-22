// Copyright (c) 2026 Huawei Technologies Co, Ltd.
// This program is free software and you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `ascendc.transpose_extend_reg_base` / `ascendc.confusion_transpose` 的 C++ 形态。
//
// 这一层**只钉发出来的文本**，不涉及 AFIR。两个手写 printer 的理由就是这两个形态自动
// emitter 发不出：
//
//   - reg 侧是「双模板实参 + 三个内联花括号列表」——`TransposeExtend<inner, total>(dst, src,
//     tmp, {...}, {...}, {...})`。T 不发（包装从 `LocalTensor` 形参推导），每个元素按
//     `RangeType<T>` 的位宽 cast：f32（4 字节）→ `int32_t`，f16（2 字节）→ `int16_t`。
//   - mem 侧是「`codegen::` 限定 + 枚举实参 + 尾随屏障」——`codegen::ConfusionTranspose<T>(
//     dst, src, tmp, AutoFuseTransposeType::X, tiling);` 紧跟 `AscendC::PipeBarrier<PIPE_ALL>();`。
//     屏障是调用的一部分（legacy 每条 ConfusionTranspose 后都跟它），不由下降侧记得。
//
// tiling 实参来自 `emitasc.member`（一份局部拷贝，包装的非 const 引用绑上去）——这里把
// member 一起放进 fixture，钉的是「调用吃的是拷贝，不是 tiling 实参本身」。

// RUN: af-opt %s | FileCheck %s --check-prefix=IR
// RUN: af-opt %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP32 --check-prefix=CPP16 --check-prefix=CPPMEM

module {
  func.func @three_segments_f32(%dst: !ascendc.local_tensor<?xf32>, %src: !ascendc.local_tensor<?xf32>,
                                %tmp: !ascendc.local_tensor<?xui8>,
                                %d0: index, %d1: index, %d2: index,
                                %s0: index, %s1: index, %s2: index,
                                %t0: index, %t1: index, %t2: index)
      attributes {ascendc.aicore} {
    "ascendc.transpose_extend_reg_base"(%dst, %src, %tmp, %d0, %d1, %d2, %s0, %s1, %s2, %t0, %t1, %t2)
        {operandSegmentSizes = array<i32: 1, 1, 1, 3, 3, 3>, innerDim = 3 : i8, totalDim = 3 : i8}
        : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>,
           index, index, index, index, index, index, index, index, index) -> ()
    return
  }

  // 2 字节元素 → `int16_t`：`RangeType<T>` 的位宽规则。
  func.func @three_segments_f16(%dst: !ascendc.local_tensor<?xf16>, %src: !ascendc.local_tensor<?xf16>,
                                %tmp: !ascendc.local_tensor<?xui8>,
                                %d0: index, %d1: index, %d2: index,
                                %s0: index, %s1: index, %s2: index,
                                %t0: index, %t1: index, %t2: index)
      attributes {ascendc.aicore} {
    "ascendc.transpose_extend_reg_base"(%dst, %src, %tmp, %d0, %d1, %d2, %s0, %s1, %s2, %t0, %t1, %t2)
        {operandSegmentSizes = array<i32: 1, 1, 1, 3, 3, 3>, innerDim = 2 : i8, totalDim = 3 : i8}
        : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>,
           index, index, index, index, index, index, index, index, index) -> ()
    return
  }

  // mem 侧：tiling 从 `emitasc.member` 的局部拷贝来；tmp 是 ui8 字节 scratch
  // （`ConfusionTranspose` 的形参是 `LocalTensor<uint8_t>`）。
  func.func @mem_confusion(%dst: !ascendc.local_tensor<?xf32>, %src: !ascendc.local_tensor<?xf32>,
                           %tmp: !ascendc.local_tensor<?xui8>,
                           %tiling: !emitasc.py_struct<"TilingData", [ui32, ui32], ["block_dim", "api"]>)
      attributes {ascendc.aicore} {
    %api = emitasc.member %tiling "api"
        : !emitasc.py_struct<"TilingData", [ui32, ui32], ["block_dim", "api"]>,
          !emitasc.py_struct<"ConfusionTransposeTiling", [ui32, ui32], ["param0", "param1"]>
    "ascendc.confusion_transpose"(%dst, %src, %tmp, %api)
        {transposeType = 4 : i32}
        : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>,
           !emitasc.py_struct<"ConfusionTransposeTiling", [ui32, ui32], ["param0", "param1"]>) -> ()
    return
  }
}

// IR-LABEL: func.func @three_segments_f32
// IR: ascendc.transpose_extend_reg_base
// IR-SAME: innerDim = 3 : i8
// IR-SAME: totalDim = 3 : i8
// IR-LABEL: func.func @three_segments_f16
// IR: ascendc.transpose_extend_reg_base
// IR-SAME: innerDim = 2 : i8
// IR-LABEL: func.func @mem_confusion
// IR: emitasc.member {{.*}} "api"
// IR: ascendc.confusion_transpose
// 枚举按底层 i32 装配/打印（`AutoFuseTransposeType` 的声明序，021 = 4）。
// IR-SAME: transposeType = 4 : i32

// 模板实参只有 inner/total 两个（T 不发），三组尺寸内联花括号列表；4 字节元素 cast
// `int32_t`。
// CPP32-LABEL: void three_segments_f32
// CPP32: TransposeExtend<3, 3>({{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}},
// CPP32-SAME: {static_cast<int32_t>({{[^)]*}}), static_cast<int32_t>({{[^)]*}}), static_cast<int32_t>({{[^)]*}})},
// CPP32-SAME: {static_cast<int32_t>({{[^)]*}}), static_cast<int32_t>({{[^)]*}}), static_cast<int32_t>({{[^)]*}})},
// CPP32-SAME: {static_cast<int32_t>({{[^)]*}}), static_cast<int32_t>({{[^)]*}}), static_cast<int32_t>({{[^)]*}})});

// 2 字节元素 → `int16_t`；innerDim 不是 inner 的专利，与 totalDim 独立取值。
// CPP16-LABEL: void three_segments_f16
// CPP16: TransposeExtend<2, 3>({{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}},
// CPP16-SAME: {static_cast<int16_t>({{[^)]*}}), static_cast<int16_t>({{[^)]*}}), static_cast<int16_t>({{[^)]*}})},

// `codegen::` 限定 + `<float>` 模板实参 + 折好的枚举 + member 的局部拷贝；尾随屏障。
// CPPMEM-LABEL: void mem_confusion
// CPPMEM: {{[a-zA-Z0-9_]+ = [a-zA-Z0-9_]+\.api;}}
// CPPMEM-NEXT: codegen::ConfusionTranspose<float>({{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}},
// CPPMEM-SAME: AutoFuseTransposeType::TRANSPOSE_ND2ND_021, {{[a-zA-Z0-9_]+}});
// CPPMEM-NEXT: AscendC::PipeBarrier<PIPE_ALL>();
