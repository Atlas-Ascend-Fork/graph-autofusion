/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "target/Ascend/AscendTargetProfile.h"

namespace mlir::ascend::kernel::elementwise {

using instruction::CapabilityReason;
using instruction::CapabilitySupport;

uint32_t elementTypeBit(Type type) {
  return afir::afirElementTypeBit(type);
}

bool isSupportedElementType(uint32_t allowed, Type type) {
  const uint32_t bit = elementTypeBit(type);
  return bit != 0U && (allowed & bit) != 0U;
}

bool hasCanonicalDescriptors(Operation *op) {
  return op->hasAttr(afir::kOperandDescsAttrName) && op->hasAttr(afir::kDestinationDescsAttrName) &&
         op->hasAttr(afir::kResultDescsAttrName);
}

namespace {

// 进程级 API 家族侧表。注册入口为自由函数；生命周期和限制见 registeredFamilies()。
std::vector<RegisteredFamily> &familyTable() {
  static std::vector<RegisteredFamily> table;
  return table;
}

}  // namespace

void recordRegisteredFamily(llvm::StringRef afirOpName, llvm::StringRef id, ApiFamily family, uint32_t elementTypes) {
  std::vector<RegisteredFamily> &table = familyTable();
  for (const RegisteredFamily &entry : table) {
    if (entry.afirOpName == afirOpName && entry.id == id) {
      return;
    }
  }
  table.push_back(RegisteredFamily{afirOpName.str(), id.str(), family, elementTypes});
}

llvm::ArrayRef<RegisteredFamily> registeredFamilies() {
  return familyTable();
}

std::optional<CapabilitySupport> checkApiFamily(ApiFamily family, const instruction::QueryContext &context) {
  // Any 使用两侧共有的标准调用，Shared 使用两侧共有的扩展包装，均无需查询家族 feature。
  // 两者的头文件校验不同，headersMatchFamily 仍需分别处理。
  if (family == ApiFamily::Any || family == ApiFamily::Shared) {
    return std::nullopt;
  }
  const FeatureState regBased = context.target.getFeatureState(TargetFeature::RegBasedApiFamily);
  if (regBased == FeatureState::Unknown) {
    return CapabilitySupport::undetermined(CapabilityReason::TargetFeatureNotModeled);
  }
  const bool isRegBased = regBased == FeatureState::Available;
  if (isRegBased != (family == ApiFamily::RegBased)) {
    return CapabilitySupport::unsupported(CapabilityReason::TargetFeatureUnavailable);
  }
  return std::nullopt;
}

std::optional<CapabilitySupport> checkTmpBuffer(Operation *op, bool needsTmpBuffer) {
  auto destinationStyle = dyn_cast<afir::AFIRDestinationStyleInterface>(op);
  // 调用接口前必须检查其存在，并在此统一验证 destination 数量。
  // 调用方的 getOperandPartition 也校验该数量，保留此处检查以明确接口自身的不变量。
  if (!destinationStyle || destinationStyle.getNumDestinations() != 1) {
    return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
  }
  const bool hasTmpBuffer = destinationStyle.getNumTmpBuffers() == 1;
  if (hasTmpBuffer != needsTmpBuffer) {
    return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
  }
  return std::nullopt;
}

}  // namespace mlir::ascend::kernel::elementwise
