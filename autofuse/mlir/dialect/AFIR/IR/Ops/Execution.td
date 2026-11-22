// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// -----------------------------------------------------------------------------------------------------------

#ifndef AFIR_EXECUTION_OPS
#define AFIR_EXECUTION_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
include "mlir/IR/CommonTypeConstraints.td"

def AFIR_BlockIdOp : AFIR_Op<"block_id", [Pure]> {
  let afirStability = AFIR_Experimental;
  let summary = "Logical index of the current block along one dimension";
  let description = [{
    当前执行块在第 `dimension` 维上的**逻辑**下标，不是物理核号。怎么取到这个下标由
    目标决定（AscendC 侧下降为 `GetBlockIdx`），AFIR 层不出现后端 API 名，也不出现
    芯片型号（设计约束 3）。

    它与调度侧的对应关系是：轴的 `axis_type` 为 `BlockOuter` 时，该轴不产生 `scf.for`
    ——它的迭代由多个块并行承担，块内看到的就是本 Op 给出的下标。

    `Pure`：同一个 `dimension` 在一次执行中取值恒定，可以自由 CSE 与外提。
  }];

  let arguments = (ins I64Attr:$dimension);
  let results = (outs Index:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

#endif // AFIR_EXECUTION_OPS
