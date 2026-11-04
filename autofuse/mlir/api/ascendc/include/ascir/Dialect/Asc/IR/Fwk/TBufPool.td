/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_FWK_TBUF_POOL_TD
#define ASC_FWK_TBUF_POOL_TD
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

//===----------------------------------------------------------------------===//
// TBufPool operations
//===----------------------------------------------------------------------===//

def AscendC_TBufPoolOp : AscendC_Op<"tBufPool", [AscConstructor]> {
  let summary = "Instantiate tBufPool";
// let arguments = (ins AscendC_TPositionAttr:$pos, AnyType:$bufIDSize);
  let results = (outs AscendC_TBufPool:$tBufPool);
  let assemblyFormat = "attr-dict `:` type($tBufPool)";
}

def AscendC_TBufPoolInitBufPoolOp : APIOp<"tbuf_pool.init_buf_pool", "InitBufPool", [AscMemberFunc]> {
  let summary = "TBufPool InitBufPool";
  let arguments = (ins AscendC_TBufPool:$origin, AnyType:$bufPool, AnyType:$len, Optional<AnyType>:$shareBuf);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    `(` $origin `,` $bufPool `,` $len `)`
    attr-dict (` ` `,` ` ` $shareBuf^)?
    `:` type($origin) `,` type($bufPool) `,` type($len)
    (` ` `,` ` ` type($shareBuf)^)? `->` type($value)
  }];
}

def AscendC_TBufPoolInitBufferOp : APIOp<"tbuf_pool.init_buffer", "InitBuffer", [AscMemberFunc]> {
  let summary = "TBufPool initialize raw buffer";
  let arguments = (ins AscendC_TBufPool:$bufPool, AscendC_TBuf:$buffer,
                       AnyType:$length);
  let assemblyFormat = [{
    $bufPool `,` $buffer `,` $length attr-dict `:` qualified(type($bufPool)) `,` qualified(type($buffer)) `,`
    type($length)
  }];
}

def AscendC_TBufPoolInitQueueOp : APIOp<"tbuf_pool.init_queue", "InitBuffer", [AscMemberFunc]> {
  let summary = "TBufPool initialize queue";
  let arguments = (ins AscendC_TBufPool:$bufPool, AscendC_BaseQueueTypeInterface:$queue, AnyType:$num,
                       AnyType:$length);
  let assemblyFormat = [{
    $bufPool `,` $queue `,` $num `,` $length attr-dict `:` qualified(type($bufPool)) `,` qualified(type($queue))
    `,` type($num) `,` type($length)
  }];
}

def AscendC_TBufPoolResetOp : APIOp<"tbuf_pool.reset", "Reset", [AscMemberFunc]> {
  let summary = "TBufPool Reset";
  let arguments = (ins AscendC_TBufPool:$bufPool);
}

#endif
