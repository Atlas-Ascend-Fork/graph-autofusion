/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_FWK_TPIPE_TD
#define ASC_FWK_TPIPE_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

//===----------------------------------------------------------------------===//
// Pipe operations (TPipe)
//===----------------------------------------------------------------------===//

def AscendC_PipeOp : AscendC_Op<"pipe", [AscConstructor]> {
  let summary = "Instantiate pipe";
  let results = (outs AscendC_Pipe:$pipe);
  let assemblyFormat = "attr-dict";
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, $_builder.getType<PipeType>());
    }]>,
  ];
}

def AscendC_TPipeInitBufferOp : APIOp<"pipe.init_buffer", "InitBuffer", [AscMemberFunc]> {
  let summary = "Initialize raw buffer";
  let arguments = (ins AscendC_Pipe:$pipe, AscendC_TBuf:$buffer,
                       AnyType:$length);
  let assemblyFormat = [{
    $pipe `,` $buffer `,` $length attr-dict `:` qualified(type($buffer)) `,`
    type($length)
  }];
}

def AscendC_TPipeInitQueueOp : APIOp<"pipe.init_queue", "InitBuffer", [AscMemberFunc]> {
  let summary = "Initialize queue";
  let arguments = (ins AscendC_Pipe:$pipe, AscendC_BaseQueueTypeInterface:$queue, AnyType:$num,
                       AnyType:$length);
  let assemblyFormat = [{
    $pipe `,` $queue `,` $num `,` $length attr-dict `:` qualified(type($queue))
    `,` type($num) `,` type($length)
  }];
}

def AscendC_TPipeInitOp : APIOp<"pipe.init", "Init", [AscMemberFunc]> {
  let summary = "TPipe Initialize";
  let arguments = (ins AscendC_Pipe:$pipe);
}

def AscendC_TPipeTQueBindDestroyOp : APIOp<"pipe.destroy", "Destroy", [AscMemberFunc]> {
  let summary = "TPipe Destroy";
  let arguments = (ins AscendC_Pipe:$pipe);
}

def AscendC_TPipeResetOp : APIOp<"pipe.reset", "Reset", [AscMemberFunc]> {
  let summary = "TPipe Reset";
  let arguments = (ins AscendC_Pipe:$pipe);
}

def AscendC_TPipeAllocEventIDOp : APIOp<"pipe.alloc_event_id", "AllocEventID", [AscMemberFunc]> {
  let summary = "TPipe AllocEventID";
  let arguments = (ins AscendC_Pipe:$pipe, AscendC_HardEventAttr:$event);
  let results = (outs AnySignlessIntegerOrIndex:$event_id);
  let paramTypeLists = [0, 3];
  let assemblyFormat = "$pipe `,` $event  attr-dict `:` type($event_id)";
}

def AscendC_TPipeReleaseEventIDOp : APIOp<"pipe.release_event_id", "ReleaseEventID", [AscMemberFunc]> {
  let summary = "TPipe ReleaseEventID";
  let arguments = (ins AscendC_Pipe:$pipe, AnyType:$id, AscendC_HardEventAttr:$event);
  let paramTypeLists = [0, 0, 3];
}

def AscendC_TPipeFetchEventIDOp : APIOp<"pipe.fetch_event_id", "FetchEventID", [AscMemberFunc]> {
  let summary = "TPipe FetchEventID";
  let arguments = (ins AscendC_Pipe:$pipe, AscendC_HardEventAttr:$event);
  let results = (outs AnySignlessIntegerOrIndex:$event_id);
  let paramTypeLists = [0, 3];
  let assemblyFormat = "$pipe `,` $event  attr-dict `:` type($event_id)";
}

def AscendC_TPipeGetBaseAddrOp : APIOp<"pipe.get_base_addr", "GetBaseAddr", [AscMemberFunc]> {
  let summary = "TPipe GetBaseAddr";
  let arguments = (ins AscendC_Pipe:$pipe, AscendC_TPositionAttr:$logicPos);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$pipe `,` $logicPos  attr-dict `:` type($value)";
}

def AscendC_TPipeInitBufPoolOp : APIOp<"pipe.init_buf_pool", "InitBufPool", [AscMemberFunc]> {
  let summary = "TPipe InitBufPool";
  let arguments = (ins AscendC_Pipe:$pipe, AnyType:$bufPool, AnyType:$len, Optional<AnyType>:$shareBuf);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    `(` $pipe `,` $bufPool `,` $len `)`
    attr-dict (` ` `,` ` ` $shareBuf^)?
    `:` type($pipe) `,` type($bufPool) `,` type($len)
    (` ` `,` ` ` type($shareBuf)^)? `->` type($value)
  }];
}

#endif
