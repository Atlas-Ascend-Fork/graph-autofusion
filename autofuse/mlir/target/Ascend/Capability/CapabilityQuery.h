/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TARGET_ASCEND_CAPABILITY_CAPABILITYQUERY_H
#define AUTOFUSE_MLIR_TARGET_ASCEND_CAPABILITY_CAPABILITYQUERY_H

#include "target/Ascend/Capability/ImplementationRegistry.h"

namespace mlir::ascend::instruction {

class CapabilityQuery {
 public:
  explicit CapabilityQuery(const ImplementationRegistry &registry);

  FailureOr<Candidates> queryCapability(const CapabilityQueryRequest &request, const QueryContext &context) const;

  FailureOr<Candidates> queryLowering(Operation *op, const QueryContext &context) const;

 private:
  bool isValidOperation(Operation *op) const;
  bool isValidCapabilityRequest(const CapabilityQueryRequest &request) const;
  FailureOr<Candidates> collectCapabilityCandidates(const CapabilityQueryRequest &request,
                                                    const QueryContext &context) const;
  FailureOr<Candidates> collectLoweringCandidates(Operation *op, const QueryContext &context) const;

  const ImplementationRegistry &registry_;
};

}  // namespace mlir::ascend::instruction

#endif  // AUTOFUSE_MLIR_TARGET_ASCEND_CAPABILITY_CAPABILITYQUERY_H
