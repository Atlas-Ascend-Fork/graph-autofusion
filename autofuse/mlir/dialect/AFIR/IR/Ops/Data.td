// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Data.td - AFIR data operations ----------------------*- tablegen -*-===//

#ifndef AFIR_DATA_OPS
#define AFIR_DATA_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def AFIR_ScalarOp : AFIR_NullaryOpBase<"scalar", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32, BF16], 1> {
  let arguments = !con(
    (ins
      AnyAttrOf<[I8Attr, UI8Attr, I16Attr, UI16Attr, I32Attr, UI32Attr, I64Attr, UI64Attr, F16Attr, F32Attr]>:$value
    ),
    commonArgs
  );
}
def AFIR_IndexExprOp : AFIR_NullaryOpBase<"index_expr", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32, BF16], 1>;

// bf16 在搬运这一类上两侧都有（`ascir_op_lowering_matrix.csv` 里 Load / Store /
// Output / Workspace 的 `mem_dtypes` 与 `reg_dtypes` 都列着它），与计算族那批 reg 侧
// 独有的位不同。这一条是硬要求，不是顺手加宽：bf16 一旦过了 bridge 的
// `IsAfirModeledDType`，一张 bf16 图里的 `afir.load` 就会建出来，而 ODS 不收它时
// 失败在 verifier 上——那是硬失败，把本可以干净回落的图变红。探针上实测过一次。
defvar AFIR_DataMoveElementTypes = [I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32, BF16];

def AFIR_LoadOp : AFIR_BufferizableUnaryOpBase<"load",
    [AFIRDataMoveInterface, AFIRDestinationStyleInterface, AFIRIndexingMapInterface,
     AFIR_TransferIn,
     DeclareOpInterfaceMethods<AFIRDestinationStyleInterface>,
     DeclareOpInterfaceMethods<AFIRIndexingMapInterface>], 1,
     AFIR_BufferizableShapedType<AFIR_DataMoveElementTypes>>,
    AFIR_OdsElementTypes<AFIR_DataMoveElementTypes> {
  let extraClassDeclaration = baseMethodDecl # odsElementTypesDecl;
  let afirStability = AFIR_Experimental;
  let summary = "Move a tile from outer storage into on-chip storage";
  let description = [{
    把源值搬到 destination 所在的存储上。源与落点各自的地址空间写在它们**自己的类型**
    上（`#afir.space<*>`），因此本 Op 不需要「从哪搬到哪」这样的参数——搬运方向是两端
    类型的推论，不是另存的一份属性（设计文档 3.1 / D8）。

    destination-style：写入的那块存储既是操作数（`outs`）也是结果。消费者必须使用
    **结果值**而不是绕回去用 destination——那条 use-def 边就是 AFIR 层的
    happens-before，绕过去意味着搬入还没落地、消费者已经开始读（竞争检查器把它报成
    「未覆盖的 producer」，见完备性判据 C2）。

    `indexing_maps` 给出源与落点的下标对应，形状可以不同：调度切分之后落点通常是
    源的一个 tile。

    本 Op 在 `transfer_in` pipe 上执行（`AFIR_TransferIn` trait），与计算和搬出彼此
    异步；跨 pipe 的顺序要由 `afir.event_record` / `afir.event_wait` 显式表达。
  }];
}

def AFIR_StoreOp : AFIR_BufferizableUnaryOptionalResultOpBase<"store",
    [AFIRDataMoveInterface, AFIRDestinationStyleInterface, AFIRIndexingMapInterface,
     AFIR_TransferOut, DeclareOpInterfaceMethods<AFIRDestinationStyleInterface>,
     DeclareOpInterfaceMethods<AFIRIndexingMapInterface>], 1,
     AFIR_BufferizableShapedType<AFIR_DataMoveElementTypes>>,
    AFIR_OdsElementTypes<AFIR_DataMoveElementTypes> {
  let extraClassDeclaration = baseMethodDecl # odsElementTypesDecl;
  let afirStability = AFIR_Experimental;
  let summary = "Move a tile from on-chip storage out to outer storage";
  let description = [{
    `afir.load` 的反向：把源值搬到 destination 所在的存储上，方向同样由两端类型的
    地址空间决定，不另存属性。

    结果可缺省。搬出到 kernel 输出参数时这块存储没有后继读者，结果值也就没有消费者；
    需要把「搬出已完成」串给下游时（例如后面还要复用同一块片上缓冲），取结果值用。

    本 Op 在 `transfer_out` pipe 上执行（`AFIR_TransferOut` trait）。
  }];
}

// 与 afir.load 一样，允许源和 destination 的 rank、shape 不同。
// 源使用 GM 原始轴，destination 使用调度切分、合轴后的坐标，不能按右对齐维度比较形状。
// 因此不挂 AFIRBroadcastInterface；错误的形状限制会使本可按能力边界回落的图在 verifier 失败。
def AFIR_NddmaOp : AFIR_BufferizableUnaryOpBase<"nddma",
    [AFIRDataMoveInterface, AFIRDestinationStyleInterface, AFIRIndexingMapInterface,
     AFIR_TransferIn,
     DeclareOpInterfaceMethods<AFIRDestinationStyleInterface>,
     DeclareOpInterfaceMethods<AFIRIndexingMapInterface>], 1,
     AFIR_BufferizableShapedType<AFIR_DataMoveElementTypes>>,
    AFIR_OdsElementTypes<AFIR_DataMoveElementTypes> {
  let extraClassDeclaration = baseMethodDecl # odsElementTypesDecl;
  let afirStability = AFIR_Experimental;
  let summary = "Move a tile from outer storage into on-chip storage while broadcasting it";
  let description = [{
    把源值搬到 destination 所在的存储上，**并在搬运的同时把它广播开**：源在某些轴上只有
    一个元素，落点在那些轴上有多个，一次搬运把那一个元素铺满。方向同 `afir.load`，由两端
    类型的地址空间决定，不另存属性。

    `indexing_maps` 给出源与落点的下标对应；两端形状不同，而且**元素数也不同**——这是它
    与 `afir.load` 的唯一形态差别。哪些轴是广播轴不另存一份属性：它由两端 descriptor 的
    `vectorized_strides` 定出来（源在该轴上 stride 为 0、落点不为 0），与
    `afir.broadcast` 的判法是同一条。

    destination-style 与 pipe 的约定同 `afir.load`：写入的那块存储既是操作数也是结果，
    消费者必须用结果值；本 Op 在 `transfer_in` pipe 上执行。
  }];
}

#endif // AFIR_DATA_OPS
