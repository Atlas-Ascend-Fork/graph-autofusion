// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.ub2ub` → 一条 `AscendC::DataCopy(dst, src, KernelUtils::BlkAlign<T>(元素数))`，
// 与 legacy `Ub2ubApiCall::Generate`（`codegen/api_call/elewise/ub2ub_api_call.cpp:26`）
// 逐字同形：目的地在前、源在后，count 经 `BlkAlign` 抬到 32B 对齐。
//
// 目标调用与 `afir.pad` 没补位那一支共用同一个 `ascendc.data_copy_blk_aligned`
// （见 `afir-pad-ub-ub.mlir` 的 else 支），所以这里不再钉 include 块——那一格已经钉了
// `utils.h` 走前导。本格钉的是「ub2ub 选中了哪一个 implementation、发出来的是哪一句」。
//
// 两张支持表给的是同一个调用（矩阵 tier D0），所以 910B1 与 950 选中的是同一个
// implementation，不是两条路。

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @ub2ub_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.ub2ub %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @ub2ub_f16(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>) {
    %result = afir.ub2ub %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }
}

// 目的地在第一个实参位：拷反了是静默算错，所以这里钉的是 SSA 值而不只是调用名。
// IR-LABEL: func.func @ub2ub_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.data_copy_blk_aligned %arg1, %arg0, %[[COUNT]]
// IR-LABEL: func.func @ub2ub_f16
// IR: ascendc.data_copy_blk_aligned
// IR-NOT: afir.ub2ub

// CPP-LABEL: void ub2ub_f32(
// CPP: AscendC::DataCopy({{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}}, KernelUtils::BlkAlign<float>({{.*}}));
// CPP-LABEL: void ub2ub_f16(
// CPP: AscendC::DataCopy({{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}}, KernelUtils::BlkAlign<half>({{.*}}));

// SELECT: ascend.implementation = {id = "ub2ub.tensor", kind = "ascendc_api"}
