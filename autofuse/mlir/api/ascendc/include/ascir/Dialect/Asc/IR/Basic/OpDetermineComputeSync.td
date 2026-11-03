/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_DETERMINE_COMPUTE_SYNC_TD
#define ASC_BASIC_OP_DETERMINE_COMPUTE_SYNC_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def NotifyNextBlockOp
    : APIOp<"notify_next_block", "NotifyNextBlock", [AscFunc]> {
  let summary = "Notify the next core that the current core's operation is complete.";
  let arguments = (ins
    AscendC_GlobalTensor:$gmWorkspace,
    AscendC_LocalTensor:$ubWorkspace
  );
  let paramTypeLists = [0, 0];
  let assemblyFormat = "$gmWorkspace `,` $ubWorkspace attr-dict `:` type($gmWorkspace) `,` type($ubWorkspace)";
}

def WaitPreBlockOp
    : APIOp<"wait_pre_block", "WaitPreBlock", [AscFunc]> {
  let summary = "Wait for the previous core to complete its operation.";
  let arguments = (ins
    AscendC_GlobalTensor:$gmWorkspace,
    AscendC_LocalTensor:$ubWorkspace
  );
  let paramTypeLists = [0, 0];
  let assemblyFormat = "$gmWorkspace `,` $ubWorkspace attr-dict `:` type($gmWorkspace) `,` type($ubWorkspace)";
}

#endif //ASC_BASIC_OP_DETERMINE_COMPUTE_SYNC_TD
