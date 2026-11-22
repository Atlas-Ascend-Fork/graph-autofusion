// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s
func.func @reduce_sum_int32_ar(%dst: !ascendc.local_tensor<4x1xi32>, %src: !ascendc.local_tensor<4x8xi32>,
                            %tmp: !ascendc.local_tensor<?xui8>, %a: i32, %r: i32) {
  ascendc.reduce_sum_int32 %dst, %src, %tmp, %a, %r {layout = 0 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xi32>, !ascendc.local_tensor<4x8xi32>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}


// CHECK: ReduceSumInt32<int32_t, AscendC::Pattern::Reduce::AR, false>

func.func @reduce_max_half_ar(%dst: !ascendc.local_tensor<4x1xf16>, %src: !ascendc.local_tensor<4x8xf16>,
                            %tmp: !ascendc.local_tensor<?xui8>, %a: i32, %r: i32) {
  ascendc.reduce_max_2d_l2 %dst, %src, %tmp, %a, %r {layout = 0 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xf16>, !ascendc.local_tensor<4x8xf16>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}


// CHECK: AscendC::ReduceMax<half, AscendC::Pattern::Reduce::AR, false>

func.func @reduce_min_half_ar(%dst: !ascendc.local_tensor<4x1xf16>, %src: !ascendc.local_tensor<4x8xf16>,
                            %tmp: !ascendc.local_tensor<?xui8>, %a: i32, %r: i32) {
  ascendc.reduce_min_2d_l2 %dst, %src, %tmp, %a, %r {layout = 0 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xf16>, !ascendc.local_tensor<4x8xf16>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}


// CHECK: AscendC::ReduceMin<half, AscendC::Pattern::Reduce::AR, false>

func.func @reduce_sum_int32_ra(%dst: !ascendc.local_tensor<4x1xi32>, %src: !ascendc.local_tensor<4x8xi32>,
                            %tmp: !ascendc.local_tensor<?xui8>, %a: i32, %r: i32) {
  ascendc.reduce_sum_int32 %dst, %src, %tmp, %a, %r {layout = 1 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xi32>, !ascendc.local_tensor<4x8xi32>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}
// CHECK: ReduceSumInt32<int32_t, AscendC::Pattern::Reduce::RA, false>

func.func @reduce_max_half_ra(%dst: !ascendc.local_tensor<4x1xf16>, %src: !ascendc.local_tensor<4x8xf16>,
                            %tmp: !ascendc.local_tensor<?xui8>, %a: i32, %r: i32) {
  ascendc.reduce_max_2d_l2 %dst, %src, %tmp, %a, %r {layout = 1 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xf16>, !ascendc.local_tensor<4x8xf16>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}
// CHECK: AscendC::ReduceMax<half, AscendC::Pattern::Reduce::RA, false>

func.func @reduce_min_half_ra(%dst: !ascendc.local_tensor<4x1xf16>, %src: !ascendc.local_tensor<4x8xf16>,
                            %tmp: !ascendc.local_tensor<?xui8>, %a: i32, %r: i32) {
  ascendc.reduce_min_2d_l2 %dst, %src, %tmp, %a, %r {layout = 1 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xf16>, !ascendc.local_tensor<4x8xf16>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}
// CHECK: AscendC::ReduceMin<half, AscendC::Pattern::Reduce::RA, false>
