/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TARGET_ASCEND_ASCENDTARGETPROFILE_H
#define AUTOFUSE_MLIR_TARGET_ASCEND_ASCENDTARGETPROFILE_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend {

enum class NpuArch {
  V2201,
  V3510,
  Unknown,
};

enum class ProcessorType {
  AICore,
  AICPU,
};

enum class AICoreUnit {
  Cube,
  Vector,
};

enum class ExecutionModel {
  SIMD,
  SIMT,
};

enum class InstructionForm {
  MemoryBased,
  RegBased,
};

struct ExecutionSpec {
  ProcessorType processor;
  std::optional<AICoreUnit> unit;
  std::optional<ExecutionModel> executionModel;
  std::optional<InstructionForm> instructionForm;
};

LogicalResult validateExecutionSpec(const ExecutionSpec &spec);

enum class MemorySpace {
  GM,
  L2,
  L1,
  L0A,
  L0B,
  L0C,
  UB,
};

std::optional<AICoreUnit> getMemoryOwner(MemorySpace space);

enum class TargetFeature {
  BF16,
  InfNan,
  NDDMA,
  // 这个平台的 Autofuse 扩展 API 由哪张供给表提供：reg-base 的 `xxx_reg_base.h`
  // （Available），还是 mem-based 的 `xxx.h`（Unavailable）。
  //
  // 它不等于 `ExecutionSpec::instructionForm`：mem-based 指令两张表上都有，问的是
  // 「哪张表」而不是「哪种指令」。也不等于「标准 API 可不可用」——两个方向都出现过，
  // `afir.abs` 的标准调用在 reg-base 一侧，`afir.tanh` 的标准调用在 mem-based 一侧。
  // 所以这里只描述家族，由各 implementation 自己说它属于哪一家。
  RegBasedApiFamily,
};

enum class FeatureState {
  Available,
  Unavailable,
  Unknown,
};

enum class CubeVectorTopology {
  Combined,
  Split,
  Unknown,
};

struct TargetIdentity {
  std::string socVersion;
  std::string shortSocVersion;
  NpuArch npuArch = NpuArch::Unknown;
};

struct CoreTopology {
  int64_t aiCoreCount;
  int64_t cubeCoreCount;
  int64_t vectorCoreCount;
  CubeVectorTopology cubeVectorTopology;
};

struct MemorySpec {
  std::optional<int64_t> capacityBytes;
  std::optional<int64_t> blockBytes;
  std::optional<int64_t> bankBytes;
  std::optional<int64_t> bankCount;
  std::optional<int64_t> subBankCount;
  std::optional<int64_t> bankGroupCount;
};

struct ComputeUnitSpec {
  std::optional<int64_t> frequencyMHz;
  std::optional<int64_t> vectorCalculationBytes;
  std::optional<int64_t> vectorRegisterBytes;
  std::optional<int64_t> predicateRegisterBytes;
  std::optional<int64_t> wideRegisterBytes;
};

enum class AccessMode {
  Read,
  Write,
};

struct ComputeAccess {
  AICoreUnit unit;
  MemorySpace space;
  AccessMode mode;
};

enum class TransferEngine {
  MTE1,
  MTE2,
  MTE3,
  Fixpipe,
};

struct TransferEdge {
  MemorySpace source;
  MemorySpace destination;
  TransferEngine engine;
  std::optional<int64_t> bytesPerCycle;
};

struct MemoryResource {
  MemorySpace space;
  MemorySpec spec;
};

struct ComputeUnitSpecEntry {
  AICoreUnit unit;
  ComputeUnitSpec spec;
};

struct TargetFeatureSpec {
  TargetFeature feature;
  FeatureState state;
};

// 一份目标描述的全部内容，不含指纹。指纹是对这些内容算出的摘要，由加载器先算好再连同
// 内容一起交给 TargetProfile；把内容收成一个整体，构造与指纹计算都只看这一个入参。
struct TargetProfileContents {
  TargetIdentity identity;
  CoreTopology topology;
  std::vector<MemoryResource> memories;
  std::vector<ComputeUnitSpecEntry> computeUnits;
  std::vector<TargetFeatureSpec> features;
  std::vector<ComputeAccess> computeAccesses;
  std::vector<TransferEdge> transferEdges;
};

class TargetProfile {
 public:
  TargetProfile(TargetProfileContents contents, std::string fingerprint);

  const TargetIdentity &getIdentity() const;
  const CoreTopology &getCoreTopology() const;

  bool hasMemory(MemorySpace space) const;
  FailureOr<MemorySpec> getMemorySpec(MemorySpace space) const;
  bool hasComputeUnit(AICoreUnit unit) const;
  FailureOr<ComputeUnitSpec> getComputeUnitSpec(AICoreUnit unit) const;

  FeatureState getFeatureState(TargetFeature feature) const;

  // profile 没写明时按 `NpuArch` 回落的那一半，逐 feature 一行，见实现处的注释。
  FeatureState getFeatureStateFromArch(TargetFeature feature) const;
  bool hasFeature(TargetFeature feature) const;
  FeatureState getExecutionSupport(const ExecutionSpec &spec) const;

  llvm::ArrayRef<MemoryResource> getMemoryResources() const;
  llvm::ArrayRef<ComputeAccess> getComputeAccesses() const;
  bool canAccess(AICoreUnit unit, MemorySpace space, AccessMode mode) const;

  llvm::ArrayRef<TransferEdge> getTransferEdges() const;
  bool hasDirectTransfer(MemorySpace source, MemorySpace destination) const;

  llvm::StringRef getFingerprint() const;

 private:
  TargetIdentity identity_;
  CoreTopology topology_;
  std::vector<MemoryResource> memories_;
  std::vector<ComputeUnitSpecEntry> computeUnits_;
  std::vector<TargetFeatureSpec> features_;
  std::vector<ComputeAccess> computeAccesses_;
  std::vector<TransferEdge> transferEdges_;
  std::string fingerprint_;
};

}  // namespace mlir::ascend

#endif  // AUTOFUSE_MLIR_TARGET_ASCEND_ASCENDTARGETPROFILE_H
