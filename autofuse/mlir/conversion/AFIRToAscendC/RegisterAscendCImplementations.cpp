/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/RegisterAscendCImplementations.h"

#include "conversion/AFIRToAscendC/Implementations/BinaryElementwise.h"
#include "conversion/AFIRToAscendC/Implementations/BroadcastImplementation.h"
#include "conversion/AFIRToAscendC/Implementations/DataMovement.h"
#include "conversion/AFIRToAscendC/Implementations/Execution.h"
#include "conversion/AFIRToAscendC/Implementations/Nddma.h"
#include "conversion/AFIRToAscendC/Implementations/Pad.h"
#include "conversion/AFIRToAscendC/Implementations/RemovePad.h"
#include "conversion/AFIRToAscendC/Implementations/Reduce.h"
#include "conversion/AFIRToAscendC/Implementations/Synchronization.h"
#include "conversion/AFIRToAscendC/Implementations/TernaryElementwise.h"
#include "conversion/AFIRToAscendC/Implementations/TilingParam.h"
#include "conversion/AFIRToAscendC/Implementations/Transpose.h"
#include "conversion/AFIRToAscendC/Implementations/UnaryElementwise.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"

namespace mlir::ascend::kernel {
namespace {

using Registrar = LogicalResult (*)(ascend::instruction::ImplementationRegistry &,
                                    ascend::instruction::SelectionPolicyRegistry *);

// Every AscendC capability module registers itself here. Adding a module
// means adding one entry to this list, not editing the loop below.
constexpr Registrar kCapabilityRegistrars[] = {
    registerBinaryElementwiseImplementations,
    registerBroadcastImplementations,
    registerDataMovementImplementations,
    registerExecutionImplementations,
    registerNddmaImplementations,
    registerPadImplementations,
    registerRemovePadImplementations,
    registerReduceImplementations,
    registerSynchronizationImplementations,
    registerTernaryElementwiseImplementations,
    registerTilingParamImplementations,
    registerTransposeImplementations,
    registerUnaryElementwiseImplementations,
};

}  // namespace

LogicalResult registerAscendCImplementations(ascend::instruction::ImplementationRegistry &registry,
                                             ascend::instruction::SelectionPolicyRegistry *policies) {
  for (Registrar registrar : kCapabilityRegistrars) {
    if (failed(registrar(registry, policies))) {
      return failure();
    }
  }
  return success();
}

}  // namespace mlir::ascend::kernel
