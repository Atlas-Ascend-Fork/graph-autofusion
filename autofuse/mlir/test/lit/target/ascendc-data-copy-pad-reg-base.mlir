// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt %s | ascir-translate -mlir-to-ascendc | FileCheck %s

module {
  func.func @packed_rows(%dst: !ascendc.local_tensor<?xf32>, %src: !ascendc.global_tensor<?xf32>,
                         %count: index, %len: index, %gm_gap: index, %ub_gap: index)
      attributes {ascendc.aicore} {
    "ascendc.data_copy_pad_extend_reg_base"(%dst, %src, %count, %len, %gm_gap, %ub_gap)
        {compact = true}
        : (!ascendc.local_tensor<?xf32>, !ascendc.global_tensor<?xf32>, index, index, index, index) -> ()
    "ascendc.data_copy_pad_extend_reg_base"(%src, %dst, %count, %len, %ub_gap, %gm_gap)
        {compact = false}
        : (!ascendc.global_tensor<?xf32>, !ascendc.local_tensor<?xf32>, index, index, index, index) -> ()
    return
  }
}

// CHECK: DataCopyPadExtend<float, AscendC::PaddingMode::Compact>(
// CHECK: DataCopyPadExtend<float, AscendC::PaddingMode::Normal>(
