/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_CORE_INTERFACES_TD
#define ASC_CORE_INTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class AscendC_TypeInterface<string name, list<Interface> baseInterfaces = []>
    : TypeInterface<name, baseInterfaces> {
  let cppNamespace = "::mlir::ascendc";
}

def AscendC_BaseQueueTypeInterface : AscendC_TypeInterface<"BaseQueueType"> {
  let summary = "base queue type";
}

def AscendC_BaseTensorTypeInterface
    : AscendC_TypeInterface<"BaseTensorType", [ShapedTypeInterface]> {
  let summary = "base tensor type";
}

def AscendC_BaseLoadDataParamsTypeInterface
    : AscendC_TypeInterface<"BaseLoadDataParamsType"> {
  let summary = "base load data params type";
}

#endif // ASC_CORE_INTERFACES_TD
