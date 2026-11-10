/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target/Ascend/TargetProfileLoader.h"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::ascend {
namespace {

using FieldMap = std::map<std::string, std::string>;

constexpr uint64_t kMaxTargetProfileBytes = 256 * 1024;
constexpr size_t kMaxTargetProfileLineBytes = 4096;
constexpr size_t kMaxTargetProfileFields = 4096;
constexpr size_t kMaxPlatformNameLength = 128;
constexpr llvm::StringLiteral kSocVersionField = "version.SoC_version";
constexpr llvm::StringLiteral kNpuArchField = "version.NpuArch";
constexpr llvm::StringLiteral kCubeVectorCombineField = "SoCInfo.cube_vector_combine";
constexpr llvm::StringLiteral kAICoreIntrinsicMapPrefix = "AICoreintrinsicDtypeMap.";

struct BuiltinProfileResource {
  llvm::StringLiteral canonicalPlatform;
  llvm::StringLiteral platformAliases;
  llvm::StringLiteral ini;
};

#include "BuiltinTargetProfiles.inc"

struct IdentityOverride {
  llvm::StringRef sourceSocVersion;
  llvm::StringRef requestedSocVersion;
};

void EmitError(llvm::raw_ostream &diagnostic, llvm::StringRef identifier, const llvm::Twine &message) {
  llvm::printEscapedString(identifier, diagnostic);
  diagnostic << ": " << message << '\n';
}

// INI 解析上下文：buffer 标识、诊断流和当前行号，由逐行解析函数共用。
struct IniParseContext {
  llvm::StringRef identifier;
  llvm::raw_ostream &diagnostic;
  unsigned lineNumber = 0;

  LogicalResult Error(const llvm::Twine &message) const {
    EmitError(diagnostic, identifier, message);
    return failure();
  }
  LogicalResult LineError(const llvm::Twine &message) const {
    return Error(llvm::Twine("line ") + llvm::Twine(lineNumber) + ": " + message);
  }
};

// 解析 `[section]` 行，成功时把当前节名换成它。
LogicalResult ParseIniSection(llvm::StringRef line, const IniParseContext &ctx, std::string &section) {
  if (!line.ends_with("]")) {
    return ctx.LineError("malformed section");
  }
  llvm::StringRef sectionName = line.drop_front().drop_back().trim();
  if (sectionName.empty()) {
    return ctx.LineError("empty section name");
  }
  section = sectionName.str();
  return success();
}

// 解析 `key = value` 行，以 `节名.键名` 写入字段表。
LogicalResult ParseIniField(llvm::StringRef line, const std::string &section, const IniParseContext &ctx,
                            FieldMap &fields) {
  size_t separator = line.find('=');
  if (separator == llvm::StringRef::npos) {
    return ctx.LineError("expected key=value");
  }
  llvm::StringRef key = line.take_front(separator).trim();
  llvm::StringRef value = line.drop_front(separator + 1).trim();
  if (key.empty()) {
    return ctx.LineError("empty field name");
  }

  std::string qualifiedKey = section + "." + key.str();
  if (fields.size() >= kMaxTargetProfileFields) {
    return ctx.Error(llvm::Twine("target profile exceeds maximum field count of ") +
                     llvm::Twine(kMaxTargetProfileFields));
  }
  if (!fields.emplace(qualifiedKey, value.str()).second) {
    return ctx.Error("duplicate field " + qualifiedKey);
  }
  return success();
}

LogicalResult ParseIni(llvm::MemoryBufferRef buffer, FieldMap &fields, llvm::raw_ostream &diagnostic) {
  IniParseContext ctx{buffer.getBufferIdentifier(), diagnostic};
  if (buffer.getBufferSize() > kMaxTargetProfileBytes) {
    return ctx.Error(llvm::Twine("target profile exceeds maximum size of ") + llvm::Twine(kMaxTargetProfileBytes) +
                     " bytes");
  }

  llvm::StringRef remaining = buffer.getBuffer();
  std::string section;

  while (!remaining.empty()) {
    auto [rawLine, rest] = remaining.split('\n');
    remaining = rest;
    ++ctx.lineNumber;

    if (rawLine.size() > kMaxTargetProfileLineBytes) {
      return ctx.LineError(llvm::Twine("exceeds maximum length of ") + llvm::Twine(kMaxTargetProfileLineBytes) +
                           " bytes");
    }

    llvm::StringRef line = rawLine.rtrim("\r").trim();
    if (line.empty() || line.starts_with("#") || line.starts_with(";")) {
      continue;
    }

    if (line.starts_with("[")) {
      if (failed(ParseIniSection(line, ctx, section))) {
        return failure();
      }
      continue;
    }

    if (section.empty()) {
      return ctx.LineError("field appears before any section");
    }
    if (failed(ParseIniField(line, section, ctx, fields))) {
      return failure();
    }
  }
  return success();
}

std::optional<llvm::StringRef> Lookup(const FieldMap &fields, llvm::StringRef key) {
  auto iterator = fields.find(key.str());
  if (iterator == fields.end()) {
    return std::nullopt;
  }
  return iterator->second;
}

// profile 读取上下文：字段表、来源标识和诊断流，供各 Read* / Build* 函数共用。
struct ProfileSource {
  const FieldMap &fields;
  llvm::StringRef identifier;
  llvm::raw_ostream &diagnostic;

  std::optional<llvm::StringRef> Find(llvm::StringRef key) const {
    return Lookup(fields, key);
  }
  LogicalResult Error(const llvm::Twine &message) const {
    EmitError(diagnostic, identifier, message);
    return failure();
  }
};

LogicalResult ReadRequiredString(const ProfileSource &profile, llvm::StringRef key, std::string &result) {
  std::optional<llvm::StringRef> value = profile.Find(key);
  if (!value || value->empty()) {
    return profile.Error("missing required field " + key);
  }
  result = value->str();
  return success();
}

LogicalResult ReadInteger(const ProfileSource &profile, llvm::StringRef key, int64_t minimum, int64_t &result,
                          bool required) {
  std::optional<llvm::StringRef> value = profile.Find(key);
  if (!value) {
    if (required) {
      return profile.Error("missing required field " + key);
    }
    result = 0;
    return success();
  }
  if (value->getAsInteger(10, result)) {
    return profile.Error("invalid integer in field " + key);
  }
  if (result < minimum) {
    return profile.Error(minimum == 1 ? "field " + key + " must be greater than zero"
                                      : "field " + key + " must not be negative");
  }
  return success();
}

LogicalResult ReadOptionalPositiveInteger(const ProfileSource &profile, llvm::StringRef key,
                                          std::optional<int64_t> &result) {
  std::optional<llvm::StringRef> value = profile.Find(key);
  if (!value || value->empty()) {
    result.reset();
    return success();
  }
  int64_t parsed = 0;
  if (value->getAsInteger(10, parsed)) {
    return profile.Error("invalid integer in field " + key);
  }
  if (parsed <= 0) {
    return profile.Error("field " + key + " must be greater than zero");
  }
  result = parsed;
  return success();
}

LogicalResult ReadOptionalPositiveInteger(const ProfileSource &profile, llvm::ArrayRef<llvm::StringRef> keys,
                                          std::optional<int64_t> &result) {
  for (llvm::StringRef key : keys) {
    if (profile.Find(key)) {
      return ReadOptionalPositiveInteger(profile, key, result);
    }
  }
  result.reset();
  return success();
}

LogicalResult ReadOptionalFeature(const ProfileSource &profile, llvm::StringRef key, TargetFeature feature,
                                  std::vector<TargetFeatureSpec> &features) {
  std::optional<llvm::StringRef> value = profile.Find(key);
  if (!value || value->empty()) {
    return success();
  }
  if (*value == "0") {
    features.push_back({feature, FeatureState::Unavailable});
    return success();
  }
  if (*value == "1") {
    features.push_back({feature, FeatureState::Available});
    return success();
  }
  return profile.Error("field " + key + " must be 0 or 1");
}

LogicalResult ReadTopology(const ProfileSource &profile, CubeVectorTopology &topology) {
  std::optional<llvm::StringRef> value = profile.Find(kCubeVectorCombineField);
  if (!value) {
    topology = CubeVectorTopology::Unknown;
    return success();
  }
  if (*value == "split") {
    topology = CubeVectorTopology::Split;
    return success();
  }
  if (*value == "fuse" || *value == "combined") {
    topology = CubeVectorTopology::Combined;
    return success();
  }
  return profile.Error(llvm::Twine("invalid value in field ") + kCubeVectorCombineField);
}

llvm::StringLiteral GetTopologyName(CubeVectorTopology topology) {
  switch (topology) {
    case CubeVectorTopology::Combined:
      return "combined";
    case CubeVectorTopology::Split:
      return "split";
    case CubeVectorTopology::Unknown:
      return "unknown";
  }
  llvm_unreachable("unknown CubeVectorTopology");
}

NpuArch ParseNpuArch(const FieldMap &fields) {
  std::optional<llvm::StringRef> value = Lookup(fields, kNpuArchField);
  if (!value) {
    return NpuArch::Unknown;
  }
  if (*value == "2201") {
    return NpuArch::V2201;
  }
  if (*value == "3510") {
    return NpuArch::V3510;
  }
  return NpuArch::Unknown;
}

bool HasAICoreIntrinsic(const FieldMap &fields, llvm::StringRef intrinsic) {
  for (const auto &[key, value] : fields) {
    if (!llvm::StringRef(key).starts_with(kAICoreIntrinsicMapPrefix)) {
      continue;
    }
    if (llvm::StringRef(value).split('|').first == intrinsic) {
      return true;
    }
  }
  return false;
}

bool HasFieldWithPrefix(const FieldMap &fields, llvm::StringRef prefix) {
  for (const auto &[key, value] : fields) {
    (void)value;
    if (llvm::StringRef(key).starts_with(prefix)) {
      return true;
    }
  }
  return false;
}

bool HasMemory(llvm::ArrayRef<MemoryResource> memories, MemorySpace space) {
  for (const MemoryResource &memory : memories) {
    if (memory.space == space) {
      return true;
    }
  }
  return false;
}

void AddTransferEdge(std::vector<TransferEdge> &edges, MemorySpace source, MemorySpace destination,
                     TransferEngine engine, std::optional<int64_t> bytesPerCycle = std::nullopt) {
  for (TransferEdge &edge : edges) {
    if (edge.source == source && edge.destination == destination && edge.engine == engine) {
      if (!edge.bytesPerCycle && bytesPerCycle) {
        edge.bytesPerCycle = bytesPerCycle;
      }
      return;
    }
  }
  edges.push_back({source, destination, engine, bytesPerCycle});
}

std::vector<ComputeAccess> BuildComputeAccesses(const CoreTopology &topology, llvm::ArrayRef<MemoryResource> memories) {
  std::vector<ComputeAccess> accesses;
  if (topology.cubeCoreCount > 0) {
    if (HasMemory(memories, MemorySpace::L0A)) {
      accesses.push_back({AICoreUnit::Cube, MemorySpace::L0A, AccessMode::Read});
    }
    if (HasMemory(memories, MemorySpace::L0B)) {
      accesses.push_back({AICoreUnit::Cube, MemorySpace::L0B, AccessMode::Read});
    }
    if (HasMemory(memories, MemorySpace::L0C)) {
      accesses.push_back({AICoreUnit::Cube, MemorySpace::L0C, AccessMode::Write});
    }
  }
  if (topology.vectorCoreCount > 0 && HasMemory(memories, MemorySpace::UB)) {
    accesses.push_back({AICoreUnit::Vector, MemorySpace::UB, AccessMode::Read});
    accesses.push_back({AICoreUnit::Vector, MemorySpace::UB, AccessMode::Write});
  }
  return accesses;
}

FailureOr<std::vector<ComputeUnitSpecEntry>> BuildComputeUnitSpecs(const ProfileSource &profile,
                                                                   const CoreTopology &topology) {
  std::vector<ComputeUnitSpecEntry> computeUnits;
  if (topology.cubeCoreCount > 0) {
    ComputeUnitSpec cube;
    if (failed(ReadOptionalPositiveInteger(profile, "AICoreSpec.cube_freq", cube.frequencyMHz))) {
      return failure();
    }
    computeUnits.push_back({AICoreUnit::Cube, cube});
  }
  if (topology.vectorCoreCount > 0) {
    ComputeUnitSpec vector;
    if (failed(ReadOptionalPositiveInteger(profile, "VectorCoreSpec.vec_freq", vector.frequencyMHz)) ||
        failed(ReadOptionalPositiveInteger(profile, {"VectorCoreSpec.vec_calc_size", "AICoreSpec.vec_calc_size"},
                                           vector.vectorCalculationBytes)) ||
        failed(ReadOptionalPositiveInteger(profile, {"VectorCoreSpec.vector_reg_width", "AICoreSpec.vector_reg_width"},
                                           vector.vectorRegisterBytes)) ||
        failed(ReadOptionalPositiveInteger(profile,
                                           {"VectorCoreSpec.predicate_reg_width", "AICoreSpec.predicate_reg_width"},
                                           vector.predicateRegisterBytes)) ||
        failed(ReadOptionalPositiveInteger(profile, {"VectorCoreSpec.wide_reg_width", "AICoreSpec.wide_reg_width"},
                                           vector.wideRegisterBytes))) {
      return failure();
    }
    computeUnits.push_back({AICoreUnit::Vector, vector});
  }
  return computeUnits;
}

FailureOr<std::vector<TransferEdge>> BuildTransferEdges(const ProfileSource &profile,
                                                        llvm::ArrayRef<MemoryResource> memories,
                                                        const CoreTopology &topology) {
  struct IntrinsicTransfer {
    llvm::StringLiteral intrinsic;
    MemorySpace source;
    MemorySpace destination;
    TransferEngine engine;
  };
  const std::array mappings = {
      IntrinsicTransfer{"Intrinsic_data_move_out2l1", MemorySpace::GM, MemorySpace::L1, TransferEngine::MTE2},
      IntrinsicTransfer{"Intrinsic_data_move_out2l0a", MemorySpace::GM, MemorySpace::L0A, TransferEngine::MTE2},
      IntrinsicTransfer{"Intrinsic_data_move_out2l0b", MemorySpace::GM, MemorySpace::L0B, TransferEngine::MTE2},
      IntrinsicTransfer{"Intrinsic_data_move_l12l0a", MemorySpace::L1, MemorySpace::L0A, TransferEngine::MTE1},
      IntrinsicTransfer{"Intrinsic_data_move_l12l0b", MemorySpace::L1, MemorySpace::L0B, TransferEngine::MTE1},
      IntrinsicTransfer{"Intrinsic_data_move_l12out", MemorySpace::L1, MemorySpace::GM, TransferEngine::MTE3},
      IntrinsicTransfer{"Intrinsic_data_move_l12ub", MemorySpace::L1, MemorySpace::UB, TransferEngine::MTE2},
      IntrinsicTransfer{"Intrinsic_data_move_ub2l1", MemorySpace::UB, MemorySpace::L1, TransferEngine::MTE3},
      IntrinsicTransfer{"Intrinsic_data_move_ub2out", MemorySpace::UB, MemorySpace::GM, TransferEngine::MTE3},
      IntrinsicTransfer{"Intrinsic_data_move_out2ub", MemorySpace::GM, MemorySpace::UB, TransferEngine::MTE2},
      IntrinsicTransfer{"Intrinsic_fix_pipe_l0c2l1", MemorySpace::L0C, MemorySpace::L1, TransferEngine::Fixpipe},
      IntrinsicTransfer{"Intrinsic_fix_pipe_l0c2out", MemorySpace::L0C, MemorySpace::GM, TransferEngine::Fixpipe},
      IntrinsicTransfer{"Intrinsic_fix_pipe_l0c2ub", MemorySpace::L0C, MemorySpace::UB, TransferEngine::Fixpipe},
  };

  if (!HasFieldWithPrefix(profile.fields, kAICoreIntrinsicMapPrefix)) {
    return profile.Error("missing required section AICoreintrinsicDtypeMap");
  }

  std::vector<TransferEdge> edges;
  for (const IntrinsicTransfer &mapping : mappings) {
    if (HasAICoreIntrinsic(profile.fields, mapping.intrinsic) && HasMemory(memories, mapping.source) &&
        HasMemory(memories, mapping.destination)) {
      AddTransferEdge(edges, mapping.source, mapping.destination, mapping.engine);
    }
  }

  // GM and Vector UB transfer is a baseline MTE2/MTE3 capability. The INI intrinsic map does not enumerate this
  // basic operation consistently, so the schema adapter supplies it when both resources and Vector Core exist.
  if (topology.vectorCoreCount > 0 && HasMemory(memories, MemorySpace::GM) && HasMemory(memories, MemorySpace::UB)) {
    AddTransferEdge(edges, MemorySpace::GM, MemorySpace::UB, TransferEngine::MTE2);
    AddTransferEdge(edges, MemorySpace::UB, MemorySpace::GM, TransferEngine::MTE3);
  }
  return edges;
}

// 一条传输边的身份：起止空间与搬运引擎。速率是这条边的属性，按身份定位后写入。
struct TransferEdgeKey {
  MemorySpace source;
  MemorySpace destination;
  TransferEngine engine;
};

void SetTransferRateIfPresent(const FieldMap &fields, llvm::StringRef key, TransferEdgeKey target,
                              std::vector<TransferEdge> &edges) {
  std::optional<llvm::StringRef> value = Lookup(fields, key);
  if (!value || value->empty()) {
    return;
  }
  int64_t bytesPerCycle = 0;
  if (value->getAsInteger(10, bytesPerCycle) || bytesPerCycle <= 0) {
    return;
  }
  for (TransferEdge &edge : edges) {
    if (edge.source == target.source && edge.destination == target.destination && edge.engine == target.engine) {
      edge.bytesPerCycle = bytesPerCycle;
      return;
    }
  }
}

void ApplyTransferRates(const FieldMap &fields, std::vector<TransferEdge> &edges) {
  SetTransferRateIfPresent(fields, "AICoreMemoryRates.l1_to_l0_a_rate",
                           {MemorySpace::L1, MemorySpace::L0A, TransferEngine::MTE1}, edges);
  SetTransferRateIfPresent(fields, "AICoreMemoryRates.l1_to_l0_b_rate",
                           {MemorySpace::L1, MemorySpace::L0B, TransferEngine::MTE1}, edges);
  SetTransferRateIfPresent(fields, "AICoreMemoryRates.l1_to_ub_rate",
                           {MemorySpace::L1, MemorySpace::UB, TransferEngine::MTE2}, edges);
  SetTransferRateIfPresent(fields, "AICoreMemoryRates.l0_c_to_ub_rate",
                           {MemorySpace::L0C, MemorySpace::UB, TransferEngine::Fixpipe}, edges);
  SetTransferRateIfPresent(fields, "AICoreMemoryRates.ub_to_l1_rate",
                           {MemorySpace::UB, MemorySpace::L1, TransferEngine::MTE3}, edges);

  llvm::StringRef readRateKey = Lookup(fields, "VectorCoreMemoryRates.ddr_read_rate")
                                    ? "VectorCoreMemoryRates.ddr_read_rate"
                                    : "AICoreMemoryRates.ddr_read_rate";
  llvm::StringRef writeRateKey = Lookup(fields, "VectorCoreMemoryRates.ub_to_ddr_rate")
                                     ? "VectorCoreMemoryRates.ub_to_ddr_rate"
                                     : "AICoreMemoryRates.ub_to_ddr_rate";
  SetTransferRateIfPresent(fields, readRateKey, {MemorySpace::GM, MemorySpace::UB, TransferEngine::MTE2}, edges);
  SetTransferRateIfPresent(fields, writeRateKey, {MemorySpace::UB, MemorySpace::GM, TransferEngine::MTE3}, edges);
}

llvm::StringLiteral GetNpuArchName(NpuArch npuArch) {
  switch (npuArch) {
    case NpuArch::V2201:
      return "2201";
    case NpuArch::V3510:
      return "3510";
    case NpuArch::Unknown:
      return "unknown";
  }
  llvm_unreachable("unknown NpuArch");
}

llvm::StringLiteral GetAICoreUnitName(AICoreUnit unit) {
  switch (unit) {
    case AICoreUnit::Cube:
      return "cube";
    case AICoreUnit::Vector:
      return "vector";
  }
  llvm_unreachable("unknown AICoreUnit");
}

llvm::StringLiteral GetMemorySpaceName(MemorySpace space) {
  switch (space) {
    case MemorySpace::GM:
      return "gm";
    case MemorySpace::L2:
      return "l2";
    case MemorySpace::L1:
      return "l1";
    case MemorySpace::L0A:
      return "l0a";
    case MemorySpace::L0B:
      return "l0b";
    case MemorySpace::L0C:
      return "l0c";
    case MemorySpace::UB:
      return "ub";
  }
  llvm_unreachable("unknown MemorySpace");
}

llvm::StringLiteral GetAccessModeName(AccessMode mode) {
  switch (mode) {
    case AccessMode::Read:
      return "read";
    case AccessMode::Write:
      return "write";
  }
  llvm_unreachable("unknown AccessMode");
}

llvm::StringLiteral GetTransferEngineName(TransferEngine engine) {
  switch (engine) {
    case TransferEngine::MTE1:
      return "mte1";
    case TransferEngine::MTE2:
      return "mte2";
    case TransferEngine::MTE3:
      return "mte3";
    case TransferEngine::Fixpipe:
      return "fixpipe";
  }
  llvm_unreachable("unknown TransferEngine");
}

llvm::StringLiteral GetTargetFeatureName(TargetFeature feature) {
  switch (feature) {
    case TargetFeature::BF16:
      return "bf16";
    case TargetFeature::InfNan:
      return "inf_nan";
    case TargetFeature::NDDMA:
      return "nddma";
    case TargetFeature::RegBasedApiFamily:
      return "reg_based_api_family";
  }
  llvm_unreachable("unknown TargetFeature");
}

llvm::StringLiteral GetFeatureStateName(FeatureState state) {
  switch (state) {
    case FeatureState::Available:
      return "available";
    case FeatureState::Unavailable:
      return "unavailable";
    case FeatureState::Unknown:
      return "unknown";
  }
  llvm_unreachable("unknown FeatureState");
}

std::string BuildFingerprint(const TargetProfileContents &contents) {
  const TargetIdentity &identity = contents.identity;
  const CoreTopology &topology = contents.topology;
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream << "schema=4\n";
  stream << "identity.soc_version=" << identity.socVersion << '\n';
  stream << "identity.short_soc_version=" << identity.shortSocVersion << '\n';
  stream << "identity.npu_arch=" << GetNpuArchName(identity.npuArch) << '\n';
  stream << "topology.ai_core_count=" << topology.aiCoreCount << '\n';
  stream << "topology.cube_core_count=" << topology.cubeCoreCount << '\n';
  stream << "topology.vector_core_count=" << topology.vectorCoreCount << '\n';
  stream << "topology.cube_vector=" << GetTopologyName(topology.cubeVectorTopology) << '\n';
  for (const MemoryResource &entry : contents.memories) {
    stream << "memory." << GetMemorySpaceName(entry.space) << '=' << entry.spec.capacityBytes.value_or(0) << ','
           << entry.spec.blockBytes.value_or(0) << '\n';
  }
  for (const ComputeUnitSpecEntry &entry : contents.computeUnits) {
    stream << "compute." << GetAICoreUnitName(entry.unit) << '=' << entry.spec.frequencyMHz.value_or(0) << ','
           << entry.spec.vectorCalculationBytes.value_or(0) << ',' << entry.spec.vectorRegisterBytes.value_or(0) << ','
           << entry.spec.predicateRegisterBytes.value_or(0) << ',' << entry.spec.wideRegisterBytes.value_or(0) << '\n';
  }
  for (const TargetFeatureSpec &entry : contents.features) {
    stream << "feature." << GetTargetFeatureName(entry.feature) << '=' << GetFeatureStateName(entry.state) << '\n';
  }
  for (const ComputeAccess &access : contents.computeAccesses) {
    stream << "access." << GetAICoreUnitName(access.unit) << '.' << GetAccessModeName(access.mode) << '='
           << GetMemorySpaceName(access.space) << '\n';
  }
  for (const TransferEdge &edge : contents.transferEdges) {
    stream << "transfer." << GetMemorySpaceName(edge.source) << "->" << GetMemorySpaceName(edge.destination) << '='
           << GetTransferEngineName(edge.engine) << ',' << edge.bytesPerCycle.value_or(0) << '\n';
  }
  stream.flush();

  llvm::SHA256 sha256;
  sha256.update(canonical);
  return llvm::toHex(sha256.final(), true);
}

bool HasPlatformAlias(llvm::StringRef aliases, llvm::StringRef platform) {
  while (!aliases.empty()) {
    auto [alias, remaining] = aliases.split(',');
    if (alias == platform) {
      return true;
    }
    aliases = remaining;
  }
  return false;
}

bool IsValidPlatformName(llvm::StringRef platform) {
  if (platform.empty() || platform.size() > kMaxPlatformNameLength) {
    return false;
  }
  for (char character : platform) {
    if (!llvm::isAlnum(character) && character != '_' && character != '-') {
      return false;
    }
  }
  return true;
}

LogicalResult ReadIdentity(const ProfileSource &profile, std::optional<IdentityOverride> identityOverride,
                           TargetIdentity &identity) {
  identity.npuArch = ParseNpuArch(profile.fields);
  if (failed(ReadRequiredString(profile, kSocVersionField, identity.socVersion))) {
    return failure();
  }
  if (!IsValidPlatformName(identity.socVersion)) {
    return profile.Error(llvm::Twine("invalid platform name in field ") + kSocVersionField);
  }
  if (identityOverride) {
    if (identity.socVersion != identityOverride->sourceSocVersion) {
      return profile.Error("builtin profile identity does not match manifest");
    }
    identity.socVersion = identityOverride->requestedSocVersion.str();
  }
  if (std::optional<llvm::StringRef> value = profile.Find("version.Short_SoC_version")) {
    identity.shortSocVersion = value->str();
  }
  return success();
}

LogicalResult ReadCoreTopology(const ProfileSource &profile, CoreTopology &topology) {
  if (failed(ReadInteger(profile, "SoCInfo.ai_core_cnt", 1, topology.aiCoreCount, true)) ||
      failed(ReadInteger(profile, "SoCInfo.cube_core_cnt", 0, topology.cubeCoreCount, true)) ||
      failed(ReadInteger(profile, "SoCInfo.vector_core_cnt", 0, topology.vectorCoreCount, true)) ||
      failed(ReadTopology(profile, topology.cubeVectorTopology))) {
    return failure();
  }
  return success();
}

// Cube / Vector 不存在时，它们私有的那几块空间既不读也不列入结果。
FailureOr<std::vector<MemoryResource>> ReadMemories(const ProfileSource &profile, const CoreTopology &topology) {
  std::optional<int64_t> gmBytes;
  std::optional<int64_t> l2Bytes;
  if (failed(ReadOptionalPositiveInteger(profile, "SoCInfo.memory_size", gmBytes)) ||
      failed(ReadOptionalPositiveInteger(profile, "SoCInfo.l2_size", l2Bytes))) {
    return failure();
  }
  std::vector<MemoryResource> memories;
  memories.reserve(7);
  memories.push_back({MemorySpace::GM, {gmBytes}});
  memories.push_back({MemorySpace::L2, {l2Bytes}});

  if (topology.cubeCoreCount > 0) {
    std::optional<int64_t> l1Bytes;
    std::optional<int64_t> l0aBytes;
    std::optional<int64_t> l0bBytes;
    std::optional<int64_t> l0cBytes;
    if (failed(ReadOptionalPositiveInteger(profile, "AICoreSpec.l1_size", l1Bytes)) ||
        failed(ReadOptionalPositiveInteger(profile, "AICoreSpec.l0_a_size", l0aBytes)) ||
        failed(ReadOptionalPositiveInteger(profile, "AICoreSpec.l0_b_size", l0bBytes)) ||
        failed(ReadOptionalPositiveInteger(profile, "AICoreSpec.l0_c_size", l0cBytes))) {
      return failure();
    }
    memories.push_back({MemorySpace::L1, {l1Bytes}});
    memories.push_back({MemorySpace::L0A, {l0aBytes}});
    memories.push_back({MemorySpace::L0B, {l0bBytes}});
    memories.push_back({MemorySpace::L0C, {l0cBytes}});
  }
  if (topology.vectorCoreCount > 0) {
    std::optional<int64_t> ubBytes;
    std::optional<int64_t> ubBlockBytes;
    if (failed(ReadOptionalPositiveInteger(profile, "AICoreSpec.ub_size", ubBytes)) ||
        failed(ReadOptionalPositiveInteger(profile, "AICoreSpec.ubblock_size", ubBlockBytes))) {
      return failure();
    }
    memories.push_back({MemorySpace::UB, {ubBytes, ubBlockBytes}});
  }
  return memories;
}

LogicalResult ReadFeatures(const ProfileSource &profile, std::vector<TargetFeatureSpec> &features) {
  if (failed(ReadOptionalFeature(profile, "SoCInfo.support_bf16", TargetFeature::BF16, features)) ||
      failed(ReadOptionalFeature(profile, "SoCInfo.support_inf_nan", TargetFeature::InfNan, features))) {
    return failure();
  }
  return success();
}

}  // namespace

static FailureOr<TargetProfile> LoadFromBuffer(llvm::MemoryBufferRef buffer, llvm::raw_ostream &diagnostic,
                                               std::optional<IdentityOverride> identityOverride) {
  FieldMap fields;
  if (failed(ParseIni(buffer, fields, diagnostic))) {
    return failure();
  }
  ProfileSource profile{fields, buffer.getBufferIdentifier(), diagnostic};

  TargetProfileContents contents;
  if (failed(ReadIdentity(profile, identityOverride, contents.identity)) ||
      failed(ReadCoreTopology(profile, contents.topology))) {
    return failure();
  }

  FailureOr<std::vector<MemoryResource>> memories = ReadMemories(profile, contents.topology);
  if (failed(memories)) {
    return failure();
  }
  contents.memories = std::move(*memories);

  FailureOr<std::vector<ComputeUnitSpecEntry>> computeUnits = BuildComputeUnitSpecs(profile, contents.topology);
  if (failed(computeUnits)) {
    return failure();
  }
  contents.computeUnits = std::move(*computeUnits);

  if (failed(ReadFeatures(profile, contents.features))) {
    return failure();
  }

  contents.computeAccesses = BuildComputeAccesses(contents.topology, contents.memories);
  FailureOr<std::vector<TransferEdge>> transferEdges =
      BuildTransferEdges(profile, contents.memories, contents.topology);
  if (failed(transferEdges)) {
    return failure();
  }
  contents.transferEdges = std::move(*transferEdges);
  ApplyTransferRates(fields, contents.transferEdges);

  std::string fingerprint = BuildFingerprint(contents);
  return TargetProfile(std::move(contents), std::move(fingerprint));
}

FailureOr<TargetProfile> TargetProfileLoader::loadFromBuffer(llvm::MemoryBufferRef buffer,
                                                             llvm::raw_ostream &diagnostic) {
  return LoadFromBuffer(buffer, diagnostic, std::nullopt);
}

FailureOr<TargetProfile> TargetProfileLoader::load(const TargetSelection &selection, llvm::raw_ostream &diagnostic) {
  if (selection.getMode() == TargetSelectionMode::BuiltinPlatform) {
    return loadBuiltin(selection.getValue(), diagnostic);
  }

  llvm::sys::fs::file_status status;
  std::error_code statusError = llvm::sys::fs::status(selection.getValue(), status);
  if (statusError || !llvm::sys::fs::is_regular_file(status)) {
    EmitError(diagnostic, selection.getValue(), "target profile is not a readable regular file");
    return failure();
  }
  if (status.getSize() > kMaxTargetProfileBytes) {
    EmitError(diagnostic, selection.getValue(),
              llvm::Twine("target profile exceeds maximum size of ") + llvm::Twine(kMaxTargetProfileBytes) + " bytes");
    return failure();
  }

  auto buffer = llvm::MemoryBuffer::getFile(selection.getValue());
  if (!buffer) {
    EmitError(diagnostic, selection.getValue(), "failed to read target profile");
    return failure();
  }
  return loadFromBuffer((*buffer)->getMemBufferRef(), diagnostic);
}

FailureOr<TargetProfile> TargetProfileLoader::loadBuiltin(llvm::StringRef platform, llvm::raw_ostream &diagnostic) {
  if (!IsValidPlatformName(platform)) {
    EmitError(diagnostic, platform, "invalid builtin target platform name");
    return failure();
  }
  for (const BuiltinProfileResource &resource : kBuiltinProfileResources) {
    if (HasPlatformAlias(resource.platformAliases, platform)) {
      llvm::MemoryBufferRef buffer(resource.ini, resource.canonicalPlatform);
      return LoadFromBuffer(buffer, diagnostic, IdentityOverride{resource.canonicalPlatform, platform});
    }
  }
  EmitError(diagnostic, platform, "unknown builtin target platform");
  return failure();
}

}  // namespace mlir::ascend
