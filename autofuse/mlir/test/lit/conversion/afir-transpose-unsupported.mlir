// Copyright (c) 2026 Huawei Technologies Co, Ltd.
// This program is free software and you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 四格反例，一格一个拒因（简单形态，`afir.abs` 的 `afir-abs-extension-api.mlir` 同款）：
//
//   - `@bf16_tile`：**dtype 的家族分界**。bf16 在 ODS 类型集内、reg 侧收（`RangeType` 按
//     位宽选索引），mem 侧不收（`TransDataTo5HD` 通路没有 bf16 形态）。reg 侧的正例在
//     `afir-transpose-reg-base-bf16.mlir`；本文件只钉「mem 拒」这一半。
//   - `@i32_tile`：**dtype 的有意摘除（2026-10-22）**。i32/ui32 在 mem 档因 v1 布局
//     （32B→8 元素）与 ConfusionTranspose 公式（写死 16 元素）错位而摘除，legacy 生产
//     同错（实测 1700/1702）。reg 档不受影响（regbase 显式传 strides）。详见下方分片
//     注释与 `Implementations/Transpose.cpp` 的「口径 A 修订」。
//   - `@no_tmp`：**tmp 缺席**。两个包装的形参都要 tmp（mem 的摆位区 / reg 的 Gather 索引
//     缓冲），`checkTmpBuffer` 两档同判——950 上同样拒，所以本文件不做 950 的 run。
//   - `@no_maps`：**索引映射缺失**。简单形态下 maps 是可选的（memref 形态缺 maps 是
//     verifier 硬错，轮不到 capability 说话），能力层的 `hasTwoIndexingMaps` 挡。
//
// 三格都报 `no supported`——**这是上报不是兜底**：调用方（`GenerateForInductor`）按这个
// 结论决定整图回落 legacy，静默选错一条路才是事故。
//
// **横线注释行是分片标记**（`afir-broadcast-unsupported.mlir` 同款，标记长什么样见下面
// 模块之间的那几行）：select 对单个分片 fail-fast，不切开的话第二格永远走不到，计数就
// 钉不住。别名不跨片可见，descriptor 直接内联。

// RUN: not af-opt --split-input-file --afir-select-implementation --target=Ascend910B1 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOSUPPORT
// 四格各报一次，计数钉住「一格一格拒、不是一票全否」。
// NOSUPPORT-COUNT-4: no supported

// dtype：mem 侧的 `isSupportedTransposeElementType` 家族分支。
module {
  func.func @bf16_tile(%input: !ascendc.local_tensor<?xbf16>, %destination: !ascendc.local_tensor<?xbf16>,
                       %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.transpose %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : !ascendc.local_tensor<?xbf16>, !ascendc.local_tensor<?xbf16>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xbf16>
    return
  }
}

// -----

// dtype（2026-10-22 摘除）：i32 在 mem 档**有意**拒收——v1 布局对非 float 的 transpose
// 只给 32B（i32 折 8 元素），而 ConfusionTranspose 公式写死尾轴 16 元素对齐，布局与
// 公式错位数值必错（legacy 生产同错，实测 1700/1702；其 ST 因手搓布局掩盖且无 int32
// 用例）。本 target（910B1）只有 mem 档可用，i32 因此「no supported」——生产链路据此
// 回落 legacy（维持存量行为）。reg 档（950，regbase 显式传 strides）不受影响仍收
// i32。恢复条件与证据链：`Implementations/Transpose.cpp` 的「口径 A 修订」注释、
// `ElementTypeMatrixTest` 的 `kKnownNarrower`、设计文档收尾结论「int32」两条。
module {
  func.func @i32_tile(%input: !ascendc.local_tensor<?xi32>, %destination: !ascendc.local_tensor<?xi32>,
                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.transpose %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xi32>
    return
  }
}

// -----

// tmp：ODS 允许不写（`tmp(...)` 可选），调度没给就是 `checkTmpBuffer` 的 unsupported。
module {
  func.func @no_tmp(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.transpose %input outs(%destination) {
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// -----

// 索引映射：descriptors 齐全、dtype 齐整，等量证明的原料缺席。
module {
  func.func @no_maps(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.transpose %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}
