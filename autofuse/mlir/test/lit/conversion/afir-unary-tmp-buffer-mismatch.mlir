// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend910B1 %s
// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend950DT_950x %s

// 「tmp buffer 的有无必须与选中的 API 匹配」这条判据单独的负例。
//
// 它必须和「供给表不匹配」分开验：afir-reciprocal-{extension,stock}-api.mlir 里那条
// `WRONGTARGET` 两条判据同时触发，把 tmp buffer 这一整段删掉它照样绿。这里选的两个 op
// **两张供给表上的要求相同**（neg 两侧都不要、sigmoid 两侧都要），所以两个 `--target`
// 用同一份 IR、同一条 expected-error——能触发的只剩 tmp buffer 这一条。
//
// 两个方向都要报：多给了一块要报（不能把那块 UB 悄悄丢掉），少给了一块也要报
// （AFIR 造不出这块 buffer，它的大小来自 ATT 侧的聚合）。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 多给了一块：两张表上的 `NegExtend` 都只有 (dst, src, size)。
func.func @neg_with_unwanted_tmp_buffer(%input: !ascendc.local_tensor<?xf16>,
                                     %destination: !ascendc.local_tensor<?xf16>,
                                     %tmp: !ascendc.local_tensor<?xui8>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.neg %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf16>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 少给了一块：标准 `AscendC::Sigmoid` 和 `SigmoidExtend` 都要 tmp buffer。
func.func @sigmoid_without_required_tmp_buffer(%input: !ascendc.local_tensor<?xf16>,
                                            %destination: !ascendc.local_tensor<?xf16>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.sigmoid %input outs(%destination) {
    afir.operand_descs = [#input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
  return
}
