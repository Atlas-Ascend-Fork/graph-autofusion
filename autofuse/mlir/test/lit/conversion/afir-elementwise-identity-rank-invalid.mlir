// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend910B1 %s
// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend950DT_950x %s

// elementwise 三条路在未调度那条路（`TransferPlanKind::Identity`）上都只收秩为 1 的
// tile。走 `Scheduled` 时不查秩：元素个数由有效视图折算出的局部跨度说了算，多维 tile 上
// LocalTensor 的秩跟着 UB buffer 的 memref 走，与它无关。
//
// 这份用例存在的理由有两条，都不是「补一条负例」那么轻：
//
// 一、一元路径原先根本没有这条判据。同一份 `2x8xf16`，`afir.add` 当场报 unsupported，
//     而 `afir.exp` 会被放行、按 `GetSize()` 生成整块的元素数——三条路对同一份 IR 给
//     不同答案，而且松的那一条没有任何用例拦着。
//
// 二、`afir-to-ascendc-binary-l2-invalid.mlir` 里的 `unsupported_add_rank` **并没有真的
//     钉住秩这条判据**。那份 IR 一个 descriptor 都不写，于是先被 `hasCanonicalDescriptors`
//     判成 MissingInformation，报出来的错和秩无关；把二元的秩判据整条删掉，它照样绿。
//     所以这里重做一份：descriptor 三件写全，让秩成为唯一能触发的那一条。
//
// 挑 op 的判据是「只有秩这一条能触发」：
//
//   - 一元用 `afir.exp`、二元用 `afir.add`——两者的家族都是 `Any`（两张表同一个
//     `AscendC::` 调用）、都不要 tmp buffer、元素类型也收 f16，于是 `checkApiFamily` /
//     `checkTmpBuffer` / `areElementTypesSupported` 三条全过得去。换成 `afir.abs` 就不行：
//     910B1 上它会先被 tmp buffer 那条拦掉，用例把秩的判据删光也照样绿。
//   - 三元没有「家族 Any 且不要 tmp」的 op，`afir.select` 在 mem 侧必须收一块 tmp buffer，
//     所以这一份把 tmp 一起喂上，其余判据同样全过。

#input_desc = #afir.tensor_desc<axis = [0, 1], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0, 1], role = <compute>>

// 一元：这一条原先整个不存在。
func.func @unsupported_exp_rank(%input: !ascendc.local_tensor<2x8xf16>,
                                %destination: !ascendc.local_tensor<2x8xf16>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.exp %input outs(%destination) {
    afir.operand_descs = [#input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : !ascendc.local_tensor<2x8xf16>, !ascendc.local_tensor<2x8xf16> -> !ascendc.local_tensor<2x8xf16>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0, 1], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0, 1], role = <compute>>

// 二元：判据一直都在，但此前那条负例拦住它的其实是缺 descriptor，不是秩。
func.func @unsupported_add_rank(%lhs: !ascendc.local_tensor<2x8xf16>,
                                %rhs: !ascendc.local_tensor<2x8xf16>,
                                %destination: !ascendc.local_tensor<2x8xf16>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.add %lhs, %rhs outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<2x8xf16>, !ascendc.local_tensor<2x8xf16>),
      !ascendc.local_tensor<2x8xf16> -> !ascendc.local_tensor<2x8xf16>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0, 1], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0, 1], role = <compute>>

// 三元：连 tmp buffer 一起喂上，这样能触发的仍然只剩秩。
func.func @unsupported_select_rank(%condition: !ascendc.local_tensor<2x8xui8>,
                                   %lhs: !ascendc.local_tensor<2x8xf32>,
                                   %rhs: !ascendc.local_tensor<2x8xf32>,
                                   %destination: !ascendc.local_tensor<2x8xf32>,
                                   %tmp: !ascendc.local_tensor<?xui8>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.select %condition, %lhs, %rhs outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<2x8xui8>, !ascendc.local_tensor<2x8xf32>, !ascendc.local_tensor<2x8xf32>),
      !ascendc.local_tensor<2x8xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<2x8xf32>
  return
}
