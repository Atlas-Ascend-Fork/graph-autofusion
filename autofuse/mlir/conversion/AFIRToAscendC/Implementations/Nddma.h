/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_NDDMA_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_NDDMA_H

#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend::instruction {
class ImplementationRegistry;
class SelectionPolicyRegistry;
}  // namespace mlir::ascend::instruction

namespace mlir::ascend::kernel {

// 注册 `afir.nddma` 的 implementation。只有 reg 侧一个——它生成的那个包装
// （`DataCopyNddma`）只在 reg-base 那张支持表上。收不了的形态报 unsupported、整张图回落
// legacy。边界与理由见 Nddma.cpp 的文件头注释。
LogicalResult registerNddmaImplementations(ascend::instruction::ImplementationRegistry &registry,
                                           ascend::instruction::SelectionPolicyRegistry *policies = nullptr);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_NDDMA_H
