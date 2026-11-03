/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_ADV_KFC_TD
#define ASC_ADV_KFC_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"


def AscendC_KfcInitOp : APIOp<"kfc_server.init", "Init"> {
  let arguments = (ins AscendC_KfcServer:$server, AnyType:$workspace);
}

def AscendC_KfcIsRunOp : APIOp<"kfc_server.is_run", "isRun"> {
  let arguments = (ins AscendC_KfcServer:$server);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$server attr-dict `:` type($value)";
}

def AscendC_KfcInitObjOp : APIOp<"kfc_server.init_obj", "InitObj"> {
  let arguments = (ins AscendC_KfcServer:$server, AscendC_Pipe:$pipe,
                    Variadic<AnyType>:$ops);
}

def AscendC_KfcRunOp : APIOp<"kfc_server.run", "Run"> {
  let arguments = (ins AscendC_KfcServer:$server, AscendC_Matmul:$matmul);
}

def AscendC_KfcQuitOp : APIOp<"kfc_server.quit", "Quit"> {
  let arguments = (ins AscendC_KfcServer:$server);
}
#endif //ASC_ADV_KFC_TD
