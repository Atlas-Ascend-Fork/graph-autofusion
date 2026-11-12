// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `ascendc.binary_brc_inline` 的 C++ 形态。只钉发出来的文本，不涉及 AFIR——`matchBroadcastInlineOperands`
// 判据与 `BinaryBrcInlineConversionPattern` 建 op 的覆盖在
// `test/ut/conversion/ElementwiseTransferPlanTest.cpp` 与 CPU E2E（`load_add_row_broadcast_inline`）里。
// 这里只核两处会手写错的地方：
//
//   - 模板实参打两处：包装自己的 `<T>`（由 `dst` 的元素类型推）与尾部函数指针上的
//     `<T, false>`（`wrappedApiName` 属性记的名字 + 固定的 `isSetMask=false`）；
//   - 五个标量实参逐字照抄 C++ 顺序（`shape0, shape1, is_input0_block_brc,
//     is_input1_block_brc, first_axis_v_stride`），不是 AFIR 操作数顺序。
//
// RUN: af-opt %s | FileCheck %s --check-prefix=IR
// RUN: af-opt %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP

module {
  func.func @add_row_broadcast(%dst: !ascendc.local_tensor<?xf32>, %src0: !ascendc.local_tensor<?xf32>,
                               %src1: !ascendc.local_tensor<?xf32>, %shape0: index, %shape1: index,
                               %isInput0BlockBrc: i8, %isInput1BlockBrc: i8, %firstAxisVStride: index)
      attributes {ascendc.aicore} {
    ascendc.binary_brc_inline %dst, %src0, %src1, %shape0, %shape1, %isInput0BlockBrc, %isInput1BlockBrc,
        %firstAxisVStride {wrappedApiName = "Add"}
        : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, index, index,
          i8, i8, index
    return
  }

  // 另一个被包装 API（`Mul`），且元素类型是 `half`——钉两处模板实参用的是同一个 `T`，
  // 不是「dst 的类型打一次、随便一个常量类型打第二次」。
  func.func @mul_row_broadcast(%dst: !ascendc.local_tensor<?xf16>, %src0: !ascendc.local_tensor<?xf16>,
                               %src1: !ascendc.local_tensor<?xf16>, %shape0: index, %shape1: index,
                               %isInput0BlockBrc: i8, %isInput1BlockBrc: i8, %firstAxisVStride: index)
      attributes {ascendc.aicore} {
    ascendc.binary_brc_inline %dst, %src0, %src1, %shape0, %shape1, %isInput0BlockBrc, %isInput1BlockBrc,
        %firstAxisVStride {wrappedApiName = "Mul"}
        : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>, index, index,
          i8, i8, index
    return
  }
}

// IR-LABEL: func.func @add_row_broadcast
// IR: ascendc.binary_brc_inline
// IR-SAME: wrappedApiName = "Add"
// IR-LABEL: func.func @mul_row_broadcast
// IR: ascendc.binary_brc_inline
// IR-SAME: wrappedApiName = "Mul"

// 五个标量实参的次序，和尾部函数指针上的第二个模板实参——写漏了 `<T, false>`、或者把
// `wrappedApiName` 打到包装自己的名字那一处，这条就分不开。
// CPP-LABEL: void add_row_broadcast
// CPP: BinaryBrcInlineApiWithTwoVectorizedAxisV2<float>(
// CPP-SAME: {{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}},
// CPP-SAME: static_cast<int64_t>({{[a-zA-Z0-9_]+}}), static_cast<int64_t>({{[a-zA-Z0-9_]+}}),
// CPP-SAME: static_cast<uint8_t>({{[a-zA-Z0-9_]+}}), static_cast<uint8_t>({{[a-zA-Z0-9_]+}}),
// CPP-SAME: static_cast<int64_t>({{[a-zA-Z0-9_]+}}), &AscendC::Add<float, false>)

// CPP-LABEL: void mul_row_broadcast
// CPP: BinaryBrcInlineApiWithTwoVectorizedAxisV2<half>(
// CPP-SAME: &AscendC::Mul<half, false>)
