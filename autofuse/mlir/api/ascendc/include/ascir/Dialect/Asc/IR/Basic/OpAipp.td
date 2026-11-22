/*
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_AIPP_TD
#define ASC_BASIC_OP_AIPP_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

//===----------------------------------------------------------------------===//
// AIPP operations
//===----------------------------------------------------------------------===//

def AscendC_SetAippFunctionsOp
    : APIOp<"set_aipp_functions", "SetAippFunctions"> {
  let summary = "Call `AscendC::SetAippFunctions` method for AIPP configuration";
  let arguments = (ins
    AscendC_GlobalTensor:$src0,
    Optional<AscendC_GlobalTensor>:$src1,
    AscendC_AippInputFormatAttr:$format,
    AscendC_AippParams:$config
  );
  let assemblyFormat = [{
    $src0 (`,` $src1^)? `,` $config `,` `format` `=` $format
    attr-dict
    `:` functional-type(operands, results)
  }];
}

#endif // ASC_BASIC_OP_AIPP_TD