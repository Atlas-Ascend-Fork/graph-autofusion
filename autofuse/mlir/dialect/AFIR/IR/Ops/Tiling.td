// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// -----------------------------------------------------------------------------------------------------------

#ifndef AFIR_TILING_OPS
#define AFIR_TILING_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Runtime tiling parameters
//===----------------------------------------------------------------------===//

def AFIR_TilingParamOp : AFIR_Op<"tiling_param", [Pure]> {
  let afirStability = AFIR_Experimental;
  let summary = "Read one runtime tiling parameter by its symbolic name";
  let description = [{
    取出 host 在本次执行前填好的一个 tiling 参数，作为一个普通的 `index` SSA 值。

    它补的是 AFIR 的一个地基缺口：队列与缓冲的尺寸是符号表达式（`#afir.buffer_desc`
    的 `size_bytes_expr`，例如 `q0_size`），而 `memref.alloc(%n)` 要的是一个 index
    **操作数**。没有本 Op 时，AFIR 层拿不到那个值，尺寸只能留在属性里、等到下降阶段
    才物化——于是 AFIR 表达不了「一块运行期尺寸的缓冲」。

    ```mlir
    %n = afir.tiling_param "q0_size" : index
    %buf = memref.alloc(%n) : memref<?xf16, #afir.space<ub>>
    ```

    **按名字，不按序号。** `q0_size` 是一个已经存在且有含义的名字（它同时是 host 侧
    TilingData 的字段名），换成序号读 IR 的人就对不上指代什么。序号方案适用于本来
    就无名的 push constant，不适用于这里。

    参数集合本身不在 AFIR 层建模：AFIR 只说「我要这个名字的值」，名字到具体 ABI 的
    对应关系属于目标。降到 AscendC 是 `tiling.q0_size`（`emitasc.member` 加一次
    index cast），host ABI 逐字节不变；名字在目标的 schema 里不存在时，由下降阶段
    给出诊断，而不是在这里假定某一种 ABI。

    `Pure`：同一个名字在一次执行中取值恒定，可以自由 CSE 与外提。
  }];

  let arguments = (ins StrAttr:$symbol);
  let results = (outs Index:$result);
  let assemblyFormat = "$symbol attr-dict `:` type($result)";
  let hasVerifier = 1;
}

#endif // AFIR_TILING_OPS
