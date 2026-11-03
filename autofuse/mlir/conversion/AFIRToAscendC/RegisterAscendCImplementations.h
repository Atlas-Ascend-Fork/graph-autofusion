/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_REGISTERASCENDCIMPLEMENTATIONS_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_REGISTERASCENDCIMPLEMENTATIONS_H

#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend::instruction {
class ImplementationRegistry;
class SelectionPolicyRegistry;
}  // namespace mlir::ascend::instruction

namespace mlir::ascend::kernel {

// `policies` is optional: pass nullptr to register implementations only, or a
// live registry to also register each capability module's selection
// policies. Adding a capability module means adding one entry to the
// registrar list in the .cpp file, not editing this function.
LogicalResult registerAscendCImplementations(ascend::instruction::ImplementationRegistry &registry,
                                             ascend::instruction::SelectionPolicyRegistry *policies = nullptr);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_REGISTERASCENDCIMPLEMENTATIONS_H
