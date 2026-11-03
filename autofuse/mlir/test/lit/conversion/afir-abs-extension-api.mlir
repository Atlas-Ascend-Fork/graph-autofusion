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
// RUN: not af-opt --afir-select-implementation --target=Ascend950DT_950x %s 2>&1 | FileCheck %s --check-prefix=WRONGTARGET

// `afir.abs` on a target that does not ship the stock `AscendC::Abs`. It lowers
// onto the Autofuse `AbsExtend` wrapper, which takes the tmp buffer the
// schedule allocated. The companion file afir-abs-stock-api.mlir covers the
// other family; between them they pin the whole of a "D2" platform difference
// -- different API name, different header, different tmp buffer requirement --
// without either the IR or the stable ids naming a chip.
//
// The pairing of the two is what makes it a difference rather than two
// unrelated lowerings: the tmp buffer is not optional decoration. A schedule
// that allocated one cannot be lowered onto the stock call and vice versa, so
// the mismatch is reported rather than silently papered over (WRONGTARGET).

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @abs_extend_f16(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>,
                            %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.abs %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf16>
    return
  }

  // int32 is the reason the wrapper exists: the stock API has no integer Abs,
  // and the wrapper synthesizes it through the tmp buffer.
  func.func @abs_extend_i32(%input: !ascendc.local_tensor<?xi32>, %destination: !ascendc.local_tensor<?xi32>,
                            %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.abs %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xi32>
    return
  }
}

// IR-LABEL: func.func @abs_extend_f16
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.abs_extend %arg1, %arg0, %arg2, %[[COUNT]]
// IR-LABEL: func.func @abs_extend_i32
// IR: ascendc.abs_extend
// IR-NOT: afir.abs

// The wrapper is a global function, so no `AscendC::` qualifier, and the
// element type is an explicit template argument.
// CPP-LABEL: void abs_extend_f16(
// CPP: AbsExtend<half>(
// CPP-LABEL: void abs_extend_i32(
// CPP: AbsExtend<int32_t>(

// The stable id says which API, not which chip.
// SELECT: ascend.implementation = {id = "abs.tensor.extend", kind = "ascendc_api"}

// WRONGTARGET: no supported
