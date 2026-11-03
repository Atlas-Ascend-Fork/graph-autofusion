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

// 侧表本体。进程级，因为注册函数是自由函数、没有别的地方挂得住；理由与代价见头文件
// `registeredFamilies()` 的注释。
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
  // `Any`（两侧同一个标准调用）与 `Shared`（两侧同一个扩展包装）都没有可选项，也就
  // 没有要问的 feature。两者在这里同一个答案，但在 `headersMatchFamily` 那边不是——
  // 所以它们是两档，不是一档。
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
  // 接口的空判是必需的（下一行要调它）。目的地计数这一条是冗余的——两条调用路径随后
  // 都会用 `getOperandPartition()` 再查一次 `destinationCount != 1`，结论和 reason 都
  // 相同。留着是因为这里是唯一拿得到 destination-style 接口的地方，删掉之后这个不变量
  // 就只剩在调用方那边各写一遍。
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
