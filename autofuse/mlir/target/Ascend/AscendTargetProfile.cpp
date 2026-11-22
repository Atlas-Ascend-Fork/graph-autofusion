/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target/Ascend/AscendTargetProfile.h"

#include <utility>

namespace mlir::ascend {

std::optional<AICoreUnit> getMemoryOwner(MemorySpace space) {
  switch (space) {
    case MemorySpace::GM:
    case MemorySpace::L2:
      return std::nullopt;
    case MemorySpace::L1:
    case MemorySpace::L0A:
    case MemorySpace::L0B:
    case MemorySpace::L0C:
      return AICoreUnit::Cube;
    case MemorySpace::UB:
      return AICoreUnit::Vector;
  }
  llvm_unreachable("unknown MemorySpace");
}

LogicalResult validateExecutionSpec(const ExecutionSpec &spec) {
  if (spec.processor == ProcessorType::AICPU) {
    return success(!spec.unit && !spec.executionModel && !spec.instructionForm);
  }
  if (!spec.unit) {
    return failure();
  }
  if (*spec.unit != AICoreUnit::Vector) {
    return success(!spec.executionModel && !spec.instructionForm);
  }
  if (!spec.executionModel) {
    return failure();
  }
  if (*spec.executionModel == ExecutionModel::SIMD) {
    return success(spec.instructionForm.has_value());
  }
  return success(!spec.instructionForm);
}

TargetProfile::TargetProfile(TargetProfileContents contents, std::string fingerprint)
    : identity_(std::move(contents.identity)),
      topology_(contents.topology),
      memories_(std::move(contents.memories)),
      computeUnits_(std::move(contents.computeUnits)),
      features_(std::move(contents.features)),
      computeAccesses_(std::move(contents.computeAccesses)),
      transferEdges_(std::move(contents.transferEdges)),
      fingerprint_(std::move(fingerprint)) {}

const TargetIdentity &TargetProfile::getIdentity() const {
  return identity_;
}

const CoreTopology &TargetProfile::getCoreTopology() const {
  return topology_;
}

bool TargetProfile::hasMemory(MemorySpace space) const {
  for (const MemoryResource &entry : memories_) {
    if (entry.space == space) {
      return true;
    }
  }
  return false;
}

FailureOr<MemorySpec> TargetProfile::getMemorySpec(MemorySpace space) const {
  for (const MemoryResource &entry : memories_) {
    if (entry.space == space) {
      return entry.spec;
    }
  }
  return failure();
}

bool TargetProfile::hasComputeUnit(AICoreUnit unit) const {
  return unit == AICoreUnit::Cube ? topology_.cubeCoreCount > 0 : topology_.vectorCoreCount > 0;
}

FailureOr<ComputeUnitSpec> TargetProfile::getComputeUnitSpec(AICoreUnit unit) const {
  if (!hasComputeUnit(unit)) {
    return failure();
  }
  for (const ComputeUnitSpecEntry &entry : computeUnits_) {
    if (entry.unit == unit) {
      return entry.spec;
    }
  }
  return ComputeUnitSpec{};
}

// 未显式配置的 feature 按指令族查询；未知家族保持 Unknown，由调用点诊断，不能默认支持。
// getFeatureState 先查 features_，但 ReadFeatures 目前只加载 support_bf16 和
// support_inf_nan，因此此处列出的 feature 尚不能由 profile 覆盖。
// DataCopyPadExtend 不按家族区分：mem 侧 datacopy.h 也定义该包装，legacy 两侧均使用。
// 若 mem 侧改为裸 DataCopy，非 32B 对齐的尾块可能丢失数据。
FeatureState TargetProfile::getFeatureStateFromArch(TargetFeature feature) const {
  switch (feature) {
    case TargetFeature::RegBasedApiFamily:
      // reg-base 的扩展头（`xxx_reg_base.h`）只在 V3510 这一类上可用实现。
      return identity_.npuArch == NpuArch::V3510   ? FeatureState::Available
             : identity_.npuArch == NpuArch::V2201 ? FeatureState::Unavailable
                                                   : FeatureState::Unknown;
    default:
      return FeatureState::Unknown;
  }
}

FeatureState TargetProfile::getFeatureState(TargetFeature feature) const {
  for (const TargetFeatureSpec &entry : features_) {
    if (entry.feature == feature) {
      return entry.state;
    }
  }
  return getFeatureStateFromArch(feature);
}

bool TargetProfile::hasFeature(TargetFeature feature) const {
  return getFeatureState(feature) == FeatureState::Available;
}

FeatureState TargetProfile::getExecutionSupport(const ExecutionSpec &spec) const {
  if (failed(validateExecutionSpec(spec))) {
    return FeatureState::Unavailable;
  }
  if (spec.processor == ProcessorType::AICore && spec.unit && !hasComputeUnit(*spec.unit)) {
    return FeatureState::Unavailable;
  }
  if (spec.processor != ProcessorType::AICore || spec.unit != AICoreUnit::Vector || !spec.executionModel) {
    return FeatureState::Unknown;
  }
  if (*spec.executionModel == ExecutionModel::SIMD && spec.instructionForm == InstructionForm::MemoryBased) {
    return FeatureState::Available;
  }
  switch (identity_.npuArch) {
    case NpuArch::V2201:
      return FeatureState::Unavailable;
    case NpuArch::V3510:
      return FeatureState::Available;
    case NpuArch::Unknown:
      return FeatureState::Unknown;
  }
  return FeatureState::Unknown;
}

llvm::ArrayRef<MemoryResource> TargetProfile::getMemoryResources() const {
  return memories_;
}

llvm::ArrayRef<ComputeAccess> TargetProfile::getComputeAccesses() const {
  return computeAccesses_;
}

bool TargetProfile::canAccess(AICoreUnit unit, MemorySpace space, AccessMode mode) const {
  if (!hasComputeUnit(unit) || !hasMemory(space)) {
    return false;
  }
  for (const ComputeAccess &access : computeAccesses_) {
    if (access.unit == unit && access.space == space && access.mode == mode) {
      return true;
    }
  }
  return false;
}

llvm::ArrayRef<TransferEdge> TargetProfile::getTransferEdges() const {
  return transferEdges_;
}

bool TargetProfile::hasDirectTransfer(MemorySpace source, MemorySpace destination) const {
  if (!hasMemory(source) || !hasMemory(destination)) {
    return false;
  }
  for (const TransferEdge &edge : transferEdges_) {
    if (edge.source == source && edge.destination == destination) {
      return true;
    }
  }
  return false;
}

llvm::StringRef TargetProfile::getFingerprint() const {
  return fingerprint_;
}

}  // namespace mlir::ascend
