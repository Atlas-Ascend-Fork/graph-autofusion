// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=SELECT
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %s 2>&1 | FileCheck %s --check-prefix=WRONGTARGET

// clip 族的 reg-base 一侧,与 afir-clip-by-value-extension-api.mlir 互为对照:**同一个
// AFIR op,少一块 tmp buffer**。两个方向都钉住——这一份没有 tmp buffer,在 mem-based
// 一侧就是 unsupported(见 WRONGTARGET),因为 `clipbyvalue.h` 的 `ClipByValue` 要调用方
// 给那块 UB,而 AFIR 造不出来。
//
// 这一类两侧同名(都叫 `ClipByValue`),与选择族的 `Where` / `WhereExtend` 不同,所以
// 单看生成的调用分不出走的是哪一侧。真正的判别式是扩展头名:`clipbyvalue.h` vs
// `clipbyvalue_reg_base.h`,注册期的 `checkFamilyMatchesHeader` 逐项比对它们,写反了整张
// 注册表建不起来。这里靠 SELECT 那条(implementation id)把它钉住。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @clip_by_value_reg_base_f32(%x: !ascendc.local_tensor<?xf32>,
                                        %lower: !ascendc.local_tensor<?xf32>,
                                        %upper: !ascendc.local_tensor<?xf32>,
                                        %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.clip_by_value %x, %lower, %upper outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @clip_by_value_reg_base_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg3
// IR-NEXT: ascendc.clipbyvalue_contiguous_reg_base %arg3, %arg0, %arg1, %arg2, %[[COUNT]]
// IR-NOT: afir.clip_by_value

// CPP-LABEL: void clip_by_value_reg_base_f32(
// CPP: {{^ *}}ClipByValue({{.*}}, static_cast<uint32_t>({{[^)]*}}));
// CPP-NOT: AscendC::

// SELECT: ascend.implementation = {id = "clip_by_value.tensor.extend_reg_base", kind = "ascendc_api"}

// WRONGTARGET: no supported
