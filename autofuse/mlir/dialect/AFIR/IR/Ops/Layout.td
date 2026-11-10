// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Layout.td - AFIR layout transform operations ---------*- tablegen -*-===//
//
// 布局变换族（设计文档 2026-09-17 划的第 7 族）：同一批元素换一种片上摆法，逻辑形状与
// 元素类型都不变，变的只是 descriptor 的 `vectorized_strides`。`RemovePad` / `Transpose` /
// `Pad` 已落在本文件，`Ub2ub` 后续落进来。
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_LAYOUT_OPS
#define AFIR_LAYOUT_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
include "Ops/Elementwise.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

// 元素类型取 legacy 两侧支持表的那一份，两侧逐字相同
// （`ascir_builtin_ops_v1.cpp:163` 与 `ascir_builtin_ops_v2.cpp:558` 共用同一个 `TensorType`）。
// `ui32` 与 `bf16` 在 lowering 那一层的 `ElementTypes` 里没有位，所以 implementation 收不到
// 它们；ODS 比 implementation 宽是对的——这里说的是 op 收什么，不是某个 implementation 生成什么。
defvar AFIR_RemovePadElementTypes = [I16, UI16, I32, UI32, F16, F32, BF16];
defvar AFIR_PadElementTypes = [I16, UI16, I32, UI32, F16, F32];

// 转置的 dtype 口径是**两侧注册表的并集**：mem 侧六种（`ascir_builtin_ops_v1.cpp:688`）、
// reg 侧八种含 BF16/BOOL（`ascir_builtin_ops_v2.cpp:1073`）。BOOL 是唯一不接的例外——
// MLIR 没有 I1 元素类型通路（本方言 ODS 无 I1、ascendc Types 无 Bool），reg 侧 e2e 对
// DT_BOOL 在 CPU 仿真上也是 `GTEST_SKIP`，接了验不了。
defvar AFIR_TransposeElementTypes = [I16, UI16, I32, UI32, F16, F32, BF16];

// RemovePad 复用一元 bufferizable op 的形状、类型推导、verifier 和 destination-style 契约，
// 但不是逐元素计算。AddRemovePadAfter 拷贝生产者属性，因此两端逻辑 axis / repeats 相同。
// lowering 通过 GatherMask 包装读取不同的 vectorized_strides，将带 padding 的输入压紧；
// 物理布局差异由具体 implementation 处理。
def AFIR_RemovePadOp : AFIR_BufferizableUnaryElementwiseOp<"remove_pad", AFIR_RemovePadElementTypes> {
  let afirStability = AFIR_Experimental;
  let summary = "Compact a padded on-chip tile by dropping the tail-axis padding";
  let description = [{
    把片上 tile 里按 32B 对齐补出来的尾轴填充挤掉：逻辑形状、元素类型都不变，输入的行与行
    之间相距一个对齐后的行距，输出连续无间隙。

    它解决的是**非对齐小尾轴下 MTE3 带宽利用率低**的问题——尾轴补齐之后搬出要按行发，
    行短则每行一次；先在 UB 内挤掉填充，搬出就变成一次连续搬运。

    调度在两处插入它，**两处的条件不同**：主对齐策略在「尾向量化轴 stride ≠ 1」的 Load 之后插
    （`optimize/platform/common/base_alignment_strategy.cpp:302`，只有 mem 侧），非对齐模板在
    非 scalar 的 Broadcast 与非连续的 Load 之后插（`optimize/platform/v1/template/unaligned_template.cpp:69`）。

    两侧走同一个扩展包装 `RemovePadExtend`（`removepad.h`，矩阵 tier D1），所以只有一个
    implementation。本 Op 在 `compute` pipe 上执行。
  }];
}

// **转置没有 perm 属性**——轴序差完全由两端 descriptor 的 `axis` 序表达，perm 由下降侧
// 从两端轴 ID 对应关系推导（legacy `ReorderInputStrideByOutputAxisOrder` 的对应物）。两端
// 形状不同（重排）、元素数相同，所以 `verifyIndexingMaps` 走 `equalElementCount = true`
// （与 `afir.load` 同档）；「重排不增删轴」由 verifier 追加判（`verifyTransposeSameRank`，
// `Ops/Data.cpp`）。
//
// 接口三件与 load 相同、方向 trait 不挂：两端都在片上、不跨 GM 边界，落 compute pipe。
// `api_tiling_field` 是 mem 档实现要用的节点级属性：bridge 按节点名生成
// `<nodeName>_tilingData_<templateId>`，func 级 `afir.api_tiling_fields` 声明那份嵌套
// struct 的字段布局，本属性说这个节点取哪个字段。reg 档不需要 host tiling，所以 Optional。
def AFIR_TransposeOp : AFIR_BufferizableUnaryOpBase<"transpose",
    [AFIRDataMoveInterface, AFIRDestinationStyleInterface, AFIRIndexingMapInterface,
     DeclareOpInterfaceMethods<AFIRDestinationStyleInterface>,
     DeclareOpInterfaceMethods<AFIRIndexingMapInterface>], 1,
    AFIR_BufferizableShapedType<AFIR_TransposeElementTypes>,
    AFIR_BufferizableShapedType<AFIR_TransposeElementTypes>,
    (ins OptionalAttr<StrAttr>:$api_tiling_field)> {
  let afirStability = AFIR_Experimental;
  let summary = "Reorder the axes of an on-chip tile";
  let extraClassDeclaration = baseMethodDecl
      # AFIR_OdsElementTypes<AFIR_TransposeElementTypes>.odsElementTypesDecl;
  let description = [{
    同一批元素换一种片上摆法：两端 rank 与元素数不变，变的只是轴序。轴序差不另存属性，
    它就在两端 descriptor 的 `axis` 数组里，perm 由下降侧推导。

    转置是布局族里唯一有独立调度路径的算子（`ComputeType::kComputeTranspose`，tiling 走
    x/y 双切分）。两档下降是覆盖矩阵上差异最大的 D2 档（API 名 / 扩展头 / tmp / host
    tiling 四项全差）：mem 档 `codegen::ConfusionTranspose`（21 种固定 perm 折 7 个枚举，
    host tiling 带全部尺寸）；reg 档 `TransposeExtend`（任意 perm，dims/strides 逐轴传参，
    超过 4 根向量化轴的外层轴各生成一层循环）。两侧包装都吃 tmp buffer。

    本 Op 在 `compute` pipe 上执行。
  }];
}

// Pad 与 `RemovePad` 对偶，同样骑在 `AFIR_BufferizableUnaryElementwiseOp` 的形状契约上
// （同 RemovePad 的理由：调度对两端给同一套 `axis` / `repeats`，差别全在
// `vectorized_strides`——只是方向反过来：输入紧凑、输出带填充）。
//
// 与 RemovePad 的两处结构差，都在 legacy 的两侧供给表里写得死死的：
//   - **要 tmp buffer**：官方 `AscendC::Pad` 的形参表带 `sharedTmpBuffer`，两侧的
//     `CalcPadTmpSize` 都实打实算尺寸（`v1_ascir_codegen_impl.h:271` / v2 同名），所以
//     本 op 落在带 `$tmpBuffer` 的基类上而不是像 RemovePad 那样空着；
//   - **要 host tiling**：`AscendC::Pad` 的最后一个形参是 `PadTiling&`，字段由 host 的
//     `AscendC::PadTilingFunc` 算（`att/gen_model_info/api_tiling_gen/api/pad.cpp:52`）。
//     两侧 codegen 都走同一个 `PadApiCall`、都发 tiling 字段（tier D1），没有
//     `Transpose` reg 档那种「不依赖 host tiling」的一侧，所以 `api_tiling_field` 对
//     本 op 的下降是必需项——缺了在下降侧报不支持，不是 Optional 的语义。
def AFIR_PadOp : AFIR_BufferizableUnaryElementwiseOp<"pad",
    AFIR_PadElementTypes,
    AFIR_PadElementTypes,
    "Same",
    (ins OptionalAttr<StrAttr>:$api_tiling_field)> {
  let afirStability = AFIR_Experimental;
  let summary = "Re-pad a compact on-chip tile to the 32B-aligned tail-axis layout";
  let description = [{
    把紧凑 tile 的尾轴按 32B 对齐重新补齐（或反向）：逻辑形状、元素类型不变，输入行与行
    之间紧挨着，输出行距对齐后的行距。它是 `RemovePad` 的对偶——调度在对齐行的消费端
    之前插入它（比如按 desc 布局转置的 Store 之前），与 RemovePad 的「生产端之后」
    相对。

    legacy 是 `PadApiCall::Generate`（`codegen/api_call/elewise/pad_api_call.cpp:26`），
    一条**运行期**二选一：倒数第二根向量化轴的对齐行距 ≠ 尾轴实际长度（布局里真的有
    补位）走 `AscendC::Pad`（`PadParams{0, padNum, 0}` + tmp buffer + `PadTiling`）；
    相等（没有补位）退化为一条 `AscendC::DataCopy(dst, src, BlkAlign<T>(元素数))`。

    两侧供给表共用 `PadApiCall`（矩阵 tier D1），只有一个 implementation。dtype 集与
    RemovePad 同列。本 Op 在 `compute` pipe 上执行。
  }];
}

#endif // AFIR_LAYOUT_OPS
