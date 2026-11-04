/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_EMITNAMESTACK_H
#define ASCIR_TARGET_ASC_EMITNAMESTACK_H

#include "mlir/IR/Value.h"

#include <stack>
#include <string>
#include <unordered_map>

namespace mlir {

struct EmitNameStack {
    using CountStack = std::stack<int64_t>;
    std::unordered_map<std::string, CountStack> valueInScopeCount;
    CountStack labelInScopeCount;

    EmitNameStack();

    void pushScope();
    void popScope();

    CountStack& getCountStack(const std::string& prefix);
    std::string getNameForEmission(Value val);
};

} // namespace mlir

#endif // ASCIR_TARGET_ASC_EMITNAMESTACK_H
