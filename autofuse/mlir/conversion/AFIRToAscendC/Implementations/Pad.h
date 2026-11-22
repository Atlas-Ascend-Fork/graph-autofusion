/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_PAD_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_PAD_H

#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend::instruction {
class ImplementationRegistry;
class SelectionPolicyRegistry;
}  // namespace mlir::ascend::instruction

namespace mlir::ascend::kernel {

// 注册 `afir.pad` 的 implementation。**只有一个**：legacy 两侧 codegen 走同一个
// `PadApiCall`、发同一个官方 `AscendC::Pad`（矩阵 tier D1），家族 `Shared`。
// 收的形态边界与理由见 Pad.cpp 的文件头注释；收不了的一律报 unsupported，
// 含它的图不走 MLIR。
LogicalResult registerPadImplementations(ascend::instruction::ImplementationRegistry &registry,
                                         ascend::instruction::SelectionPolicyRegistry *policies = nullptr);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_PAD_H
