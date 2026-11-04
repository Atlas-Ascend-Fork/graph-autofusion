/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_DIALECT_TD
#define ASC_DIALECT_TD

include "mlir/IR/OpBase.td"

def AscendC_Dialect : Dialect {
  let name = "ascendc";
  let summary = "A special dialect to support Ascend C API";

  let description = [{
    The AscendC dialect provides a number of operations that represent Ascend C
    library API for Huawei NPU.
  }];

  let cppNamespace = "::mlir::ascendc";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
    void registerOps();
  }];
}

#endif // ASC_DIALECT_TD
