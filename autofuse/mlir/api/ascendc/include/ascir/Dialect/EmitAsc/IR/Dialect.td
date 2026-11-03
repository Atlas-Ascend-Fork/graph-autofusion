/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EMITASC_DIALECT_TD
#define EMITASC_DIALECT_TD

include "mlir/IR/OpBase.td"

def EmitAsc_Dialect : Dialect {
  let name = "emitasc";
  let summary = "C++ emission support dialect";

  let cppNamespace = "::mlir::emitasc";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
    void registerOps();
  }];
}

#endif // EMITASC_DIALECT_TD
