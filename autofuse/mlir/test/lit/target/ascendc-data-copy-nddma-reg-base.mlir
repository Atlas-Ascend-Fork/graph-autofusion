// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `ascendc.data_copy_nddma_reg_base` → reg 侧的 `DataCopyNddma` 包装
// （`v35/ascendc/api_regbase/datacopy_nddma.h`）。
//
// 这一层只钉生成的 C++，不涉及 AFIR：三组尺寸落成三个栈上 `const int64_t[dim]` 数组，
// `dim` 作为第二个模板实参，与包装的 `const int64_t (&output_dims)[dim]` 对上。手写 printer
// 的理由就是这个形态——自动 emitter 生成不出「栈上数组 + 两个模板实参」。
//
// 数组名带 `<src>_nddma_to_<dst>` 后缀：同一个作用域里两条 ND-DMA 时只按 dst 命名会撞名，
// 与 `BroadcastExtendRegBaseOp` 的 `dst_shape_<src>_brc_to_<dst>` 同一条规则。

// RUN: af-opt %s | FileCheck %s --check-prefix=IR
// RUN: af-opt %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP

module {
  func.func @two_segments(%dst: !ascendc.local_tensor<?xf32>, %src: !ascendc.global_tensor<?xf32>,
                          %d0: index, %d1: index, %o0: index, %o1: index, %i0: index, %i1: index)
      attributes {ascendc.aicore} {
    "ascendc.data_copy_nddma_reg_base"(%dst, %src, %d0, %d1, %o0, %o1, %i0, %i1)
        {operandSegmentSizes = array<i32: 1, 1, 2, 2, 2>}
        : (!ascendc.local_tensor<?xf32>, !ascendc.global_tensor<?xf32>,
           index, index, index, index, index, index) -> ()
    return
  }
}

// IR-LABEL: func.func @two_segments
// IR: ascendc.data_copy_nddma_reg_base %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7

// 三个数组各自一行，长度就是段数；形参是 `const int64_t (&)[dim]`，而操作数是 index，
// 所以每一项 `static_cast<int64_t>`。
// CPP: const int64_t output_dims_[[SRC:[a-zA-Z0-9_]+]]_nddma_to_[[DST:[a-zA-Z0-9_]+]][2] = {static_cast<int64_t>({{[^)]*}}), static_cast<int64_t>({{[^)]*}})};
// CPP-NEXT: const int64_t output_stride_[[SRC]]_nddma_to_[[DST]][2] = {
// CPP-NEXT: const int64_t input_stride_[[SRC]]_nddma_to_[[DST]][2] = {
// 调用在三个数组之后，`dim` 与数组长度一致。
// CPP-NEXT: DataCopyNddma<float, 2>([[DST]], [[SRC]], output_dims_[[SRC]]_nddma_to_[[DST]], output_stride_[[SRC]]_nddma_to_[[DST]], input_stride_[[SRC]]_nddma_to_[[DST]]);
// 生成的不是原生那个同名邻居（`AscendC::DataCopy(dst, src, NdDmaParams, dims)`）。
// CPP-NOT: NdDmaParams
