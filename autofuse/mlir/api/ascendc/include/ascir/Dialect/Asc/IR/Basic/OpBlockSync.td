/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_BLOCKSYNC_TD
#define ASC_BASIC_OP_BLOCKSYNC_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Data Sync Barrier Op
//===----------------------------------------------------------------------===//

def AscendC_DataSyncBarrierOp
    : APIOp<"data_sync_barrier", "DataSyncBarrier", [AscFunc]> {
  let description = "Synchronize data accesses by waiting for specified memory operations, "
                    "controlled by the MemDsbT parameter.";
  let arguments = (ins
    AscendC_MemDsbAttr:$memDsbType
  );
  let paramTypeLists = [3];
}

//===----------------------------------------------------------------------===//
// Synchronization operations
//===----------------------------------------------------------------------===//

def PipeBarrierOp : APIOp<"pipe_barrier", "PipeBarrier"> {
  let arguments = (ins AscendC_PipeAttr:$pipe);
  let assemblyFormat = "$pipe attr-dict";
  let hasCanonicalizeMethod = 1;
}

def SetFlagOp : APIOp<"set_flag", "SetFlag", [AscFunc]> {
  let arguments = (ins AscendC_HardEventAttr:$event, AnyType:$eventId);
  let assemblyFormat = "$event `,` $eventId attr-dict `:` type($eventId)";
  let paramTypeLists = [3, 0];
}

def WaitFlagOp : APIOp<"wait_flag", "WaitFlag"> {
  let arguments = (ins AscendC_HardEventAttr:$event, AnyType:$eventId);
  let assemblyFormat = "$event `,` $eventId attr-dict `:` type($eventId)";
}

def IBSetOp : APIOp<"ib_set", "IBSet", [AscFunc]> {
  let arguments = (ins
    AscendC_GlobalTensor:$gmWorkspace,
    AscendC_LocalTensor:$ubWorkspace,
    AnyType:$blockIdx,
    AnyType:$eventId,
    UnitAttr:$isAIVOnly
  );
  let paramTypeLists = [0, 0, 0, 0];
}

def IBWaitOp : APIOp<"ib_wait", "IBWait", [AscFunc]> {
  let arguments = (ins
    AscendC_GlobalTensor:$gmWorkspace,
    AscendC_LocalTensor:$ubWorkspace,
    AnyType:$blockIdx,
    AnyType:$eventId,
    UnitAttr:$isAIVOnly
  );
  let paramTypeLists = [0, 0, 0, 0];
}

def SyncAllSoftOp : APIOp<"sync_all_soft", "SyncAll", [AscFunc]> {
  let arguments = (ins
    AscendC_GlobalTensor:$gmWorkspace,
    AscendC_LocalTensor:$ubWorkspace,
    AnyType:$usedCores,
    UnitAttr:$isAIVOnly
  );
  let paramTypeLists = [0, 0, 0];
}

def SyncAllHardOp : APIOp<"sync_all_hard", "SyncAll", [AscFunc]> {
  let arguments = (ins UnitAttr:$isAIVOnly);
  let paramTypeLists = [];
  let assemblyFormat = "attr-dict";
}

def CrossCoreSetFlagOp : APIOp<"cross_core_set_flag", "CrossCoreSetFlag"> {
  let arguments = (ins AnyType:$flagId, UI8Attr:$modeId, AscendC_PipeAttr:$pipe);
  let assemblyFormat = "$flagId `,` $modeId `,` $pipe attr-dict `:` type($flagId)";
}

def CrossCoreWaitFlagOp : APIOp<"cross_core_wait_flag", "CrossCoreWaitFlag"> {
  let arguments = (ins AnyType:$flagId, UI8Attr:$modeId, AscendC_PipeAttr:$pipe);
  let assemblyFormat = "$flagId `,` $modeId `,` $pipe attr-dict `:` type($flagId)";
}

def GetBufOp : APIOp<"get_buf", "get_buf"> {
  let arguments = (ins AscendC_PipeAttr:$pipe, I32Attr:$bufId, UnitAttr:$mode);
  let assemblyFormat = "$pipe `,` $bufId attr-dict";
}

def RlsBufOp : APIOp<"rls_buf", "rls_buf"> {
  let arguments = (ins AscendC_PipeAttr:$pipe, I32Attr:$bufId, UnitAttr:$mode);
  let assemblyFormat = "$pipe `,` $bufId attr-dict";
}

#endif // ASC_BASIC_OP_BLOCKSYNC_TD
