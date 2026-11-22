/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_FIXPIPE_TD
#define ASC_BASIC_OP_FIXPIPE_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_FixpipeOp : DataCopyOp<"fixpipe", "Fixpipe", [OpWithDstInterface, OpWithSrcInterface]> {
  let summary = "Call `AscendC::Fixpipe` function without cbufWorkspace";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst, AscendC_LocalTensor:$src,
                    AnyTypeOf<[AscendC_FixpipeParamsV220, AscendC_FixpipeParamsC310]>:$intriParams,
                    AscendC_FixpipeConfig:$fixpipeConfig);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def AscendC_FixpipeWithWorkspaceOp : APIOp<"fixpipe_with_workspace", "Fixpipe"> {
  let summary = "Call `AscendC::Fixpipe` function with cbufWorkspace";
  let arguments = (ins AscendC_GlobalTensor:$dst, AscendC_LocalTensor:$src,
                     AscendC_LocalTensor:$cbufWorkspace, AscendC_FixpipeParamsV220:$intriParams,
                     AscendC_FixpipeConfig:$fixpipeConfig);
}

def AscendC_SetFixpipePreQuantFlagOp : APIOp<"set_fixpipe_pre_quant_flag", "SetFixpipePreQuantFlag", [AscFunc]> {
  let description = "Set scalar quantization parameters in fixpipe quantization process";
  let arguments = (ins AnyType:$config);
  let assemblyFormat = "$config attr-dict `:` type($config)";
}
#endif // ASC_BASIC_OP_FIXPIPE_TD
