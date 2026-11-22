/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_UTILS_INLINING_H
#define ASCIR_DIALECT_UTILS_INLINING_H

#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace ascir {

template <typename... AllowedOpT>
struct AllowlistInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation* op, Region*, bool, IRMapping&) const override { return isa<AllowedOpT...>(op); }
};

struct PermissiveInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation*, Operation*, bool) const override { return true; }

    bool isLegalToInline(Region*, Region*, bool, IRMapping&) const override { return true; }

    bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const override { return true; }

    void handleTerminator(Operation*, Block*) const override {}

    void handleTerminator(Operation*, ValueRange) const override {}
};

} // namespace ascir
} // namespace mlir

#endif // ASCIR_DIALECT_UTILS_INLINING_H
