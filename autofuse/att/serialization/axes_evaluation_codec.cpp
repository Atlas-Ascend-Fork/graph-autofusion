/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "axes_evaluation_codec.h"
#include <algorithm>
#include <cstring>
#include "aftp_v1.h"
#include "axes_callback_codec.h"
#include "tiling_record_io.h"
#include "tiling_result_layout_codec.h"
#include "tiling_stage_codec.h"

namespace att {
namespace {
using Status = TilingExpressionStatus;
using tiling_record::kWordBits;
using tiling_record::kWordBytes;
using tiling_record::Reader;
using tiling_record::Writer;
constexpr size_t kVariableFixedBytes = 4 * sizeof(int64_t) + 3 * kWordBytes;
constexpr size_t kMinVariableBytes = kVariableFixedBytes + kWordBytes;  // includes constraint count
constexpr size_t kConstraintBytes = kWordBytes + sizeof(int64_t);
constexpr size_t kConstraintWithSourceBytes = kConstraintBytes + kWordBytes;
constexpr size_t kMinConstraintBindingBytes = 3 * kWordBytes;  // record length and two index counts
constexpr size_t kMinUpperBoundBindingBytes = 3 * kWordBytes;  // record length, target and parent count
constexpr size_t kManifestIdentityCount = 4;                   // graph case, shape slots, companion assets, layout
constexpr uint32_t kVersion = 1;
constexpr uint32_t kSolveManifestMagic = 0x4d534641U;
constexpr uint32_t kSolvePolicyVersion = 2;
constexpr uint32_t kSolveResourceVersion = 3;
constexpr uint32_t kSolveDerivedVersion = 5;
constexpr uint32_t kSolveResourceWritebackVersion = 6;
constexpr uint32_t kKnownSolveFlags = kAxesSolveTradeOff | kAxesSolveTuneBlock | kAxesSolveTune;

template <class Bounds>
bool HasSolveUpperBounds(const AxesSolveTopology &topology, const Bounds &bounds) {
  std::vector<bool> defined(topology.variables.size());
  for (const auto &bound : bounds) {
    if (bound.target_index >= defined.size() || defined[bound.target_index]) {
      return false;
    }
    defined[bound.target_index] = true;
  }
  const auto covered = [&defined](const auto &group) {
    return std::all_of(group.begin(), group.end(), [&defined](uint32_t index) { return defined[index]; });
  };
  return covered(topology.pure_mc) && covered(topology.local_buffer);
}

bool MatchesEnvelope(const AftpV1Header &header, const AxesEvaluationEnvelopeRequirements &requirements) {
  return header.schema_hash == requirements.schema_hash &&
         header.target_constraints == requirements.target_constraints &&
         (header.required_capabilities & ~requirements.supported_capabilities) == 0;
}

template <class Output, class Decoder>
Status ReadRecord(Reader &reader, Output &output, Decoder decode) {
  uint32_t length;
  if (!reader.Word(length) || length > reader.Size()) {
    return Status::kInvalidProgram;
  }
  const auto status = decode(reader.Data(), length, &output);
  reader.Skip(length);
  return status;
}
bool WriteRecord(Writer &writer, const std::vector<uint8_t> &record) {
  if (writer.bytes.size() > kAftpV1MaxBytes - kWordBytes ||
      record.size() > kAftpV1MaxBytes - kWordBytes - writer.bytes.size()) {
    return false;
  }
  writer.Word(static_cast<uint32_t>(record.size()));
  writer.bytes.insert(writer.bytes.end(), record.begin(), record.end());
  return true;
}
bool ReadIndices(Reader &reader, std::vector<uint32_t> &indices) {
  uint32_t count;
  if (!reader.Count(count, kWordBytes)) {
    return false;
  }
  indices.resize(count);
  for (auto &index : indices) {
    if (!reader.Word(index)) {
      return false;
    }
  }
  return true;
}
bool WriteIndices(Writer &writer, const std::vector<uint32_t> &indices) {
  if (indices.size() > kTilingExpressionMaxItems || writer.bytes.size() > kAftpV1MaxBytes - kWordBytes ||
      indices.size() > (kAftpV1MaxBytes - kWordBytes - writer.bytes.size()) / kWordBytes) {
    return false;
  }
  writer.Word(static_cast<uint32_t>(indices.size()));
  for (auto index : indices) {
    writer.Word(index);
  }
  return true;
}
void WriteInt64(Writer &writer, int64_t value) {
  const auto bits = static_cast<uint64_t>(value);
  writer.Word(static_cast<uint32_t>(bits));
  writer.Word(static_cast<uint32_t>(bits >> kWordBits));
}

bool ReadInt64(Reader &reader, int64_t &value) {
  uint32_t low, high;
  if (!reader.Word(low) || !reader.Word(high)) {
    return false;
  }
  const uint64_t bits = low | (static_cast<uint64_t>(high) << kWordBits);
  value = CopyTilingScalarBits<int64_t>(bits);
  return true;
}

// Keep the original variable record width. No-tail relations are outside this wire contract.
constexpr uint32_t kReservedVariableWord = UINT32_MAX;

bool WriteTopology(Writer &writer, const AxesSolveTopology &topology, bool bind_resources) {
  writer.Word(static_cast<uint32_t>(topology.variables.size()));
  for (const auto &variable : topology.variables) {
    if (writer.bytes.size() > kAftpV1MaxBytes - kVariableFixedBytes) {
      return false;
    }
    WriteInt64(writer, variable.initial_value);
    WriteInt64(writer, variable.align);
    WriteInt64(writer, variable.prompt_align);
    WriteInt64(writer, variable.data_type_size);
    writer.Word(variable.mc_related ? 1 : 0);
    writer.Word(kReservedVariableWord);
    writer.Word(static_cast<uint32_t>(variable.order));
    if (!WriteIndices(writer, variable.constraints)) {
      return false;
    }
  }
  const size_t constraint_bytes = bind_resources ? kConstraintWithSourceBytes : kConstraintBytes;
  if (writer.bytes.size() > kAftpV1MaxBytes - kWordBytes ||
      topology.constraints.size() > (kAftpV1MaxBytes - kWordBytes - writer.bytes.size()) / constraint_bytes) {
    return false;
  }
  writer.Word(static_cast<uint32_t>(topology.constraints.size()));
  for (const auto &constraint : topology.constraints) {
    writer.Word(constraint.type);
    WriteInt64(writer, constraint.resource);
    if (bind_resources) {
      writer.Word(static_cast<uint32_t>(constraint.resource_source));
    }
  }
  return WriteIndices(writer, topology.pure_mc) && WriteIndices(writer, topology.local_buffer) &&
         WriteIndices(writer, topology.ordered_to_canonical);
}

bool ReadTopology(Reader &reader, AxesSolveTopology &topology, bool bind_resources) {
  uint32_t count;
  if (!reader.Count(count, kMinVariableBytes)) {
    return false;
  }
  topology.variables.resize(count);
  for (auto &variable : topology.variables) {
    uint32_t mc, reserved, order;
    if (!ReadInt64(reader, variable.initial_value) || !ReadInt64(reader, variable.align) ||
        !ReadInt64(reader, variable.prompt_align) || !ReadInt64(reader, variable.data_type_size) || !reader.Word(mc) ||
        mc > 1 || !reader.Word(reserved) || reserved != kReservedVariableWord || !reader.Word(order) ||
        !ReadIndices(reader, variable.constraints)) {
      return false;
    }
    variable.mc_related = mc != 0;
    variable.order = order;
  }
  if (!reader.Count(count, bind_resources ? kConstraintWithSourceBytes : kConstraintBytes)) {
    return false;
  }
  topology.constraints.resize(count);
  for (auto &constraint : topology.constraints) {
    uint32_t source = 0;
    if (!reader.Word(constraint.type) || !ReadInt64(reader, constraint.resource) ||
        (bind_resources && !reader.Word(source))) {
      return false;
    }
    constraint.resource_source = static_cast<AxesConstraintResource>(source);
  }
  return ReadIndices(reader, topology.pure_mc) && ReadIndices(reader, topology.local_buffer) &&
         ReadIndices(reader, topology.ordered_to_canonical) && reader.Done() && VerifyAxesSolveTopology(topology);
}
Status VerifySolveManifest(const AftpV1PayloadDescriptor &manifest, uint64_t layout_identity, bool require_structured) {
  Reader reader(manifest.data, manifest.size);
  uint32_t magic;
  if (!reader.Word(magic) || magic != kSolveManifestMagic) {
    return require_structured ? Status::kInvalidProgram : Status::kOk;
  }
  uint32_t version;
  if (!reader.Word(version)) {
    return Status::kInvalidProgram;
  }
  if (version != kVersion) {
    return Status::kUnsupported;
  }
  uint64_t value = 0;
  for (size_t i = 0; i < kManifestIdentityCount; ++i) {
    uint32_t low, high;
    if (!reader.Word(low) || !reader.Word(high) || (low == 0 && high == 0)) {
      return Status::kInvalidProgram;
    }
    value = low | (static_cast<uint64_t>(high) << kWordBits);
  }
  return reader.Done() && value == layout_identity ? Status::kOk : Status::kInvalidProgram;
}

}  // namespace

TilingExpressionStatus GetAxesEvaluationScratchElements(const AxesEvaluationRecords &records, size_t *elements) {
  if (elements == nullptr) {
    return Status::kInvalidArgument;
  }
  size_t capacity = 0;
  for (const auto &method : records.methods) {
    if (!method) {
      return Status::kInvalidProgram;
    }
    if (method->Stage() != nullptr) {
      capacity = std::max(capacity, GetTilingStageScratchElements(*method->Stage()));
    }
  }
  const auto include_callbacks = [&capacity](const auto &entries) {
    for (const auto &entry : entries) {
      if (!entry.callback) {
        return false;
      }
      capacity = std::max(capacity, GetTilingStageScratchElements(entry.callback->Stage()));
    }
    return true;
  };
  if (!include_callbacks(records.bindings.constraints) || !include_callbacks(records.bindings.upper_bounds)) {
    return Status::kInvalidProgram;
  }
  *elements = capacity;
  return Status::kOk;
}

TilingExpressionStatus DecodeAxesEvaluationV1(const void *data, size_t size, AxesEvaluationRecords *output) {
  if (data == nullptr || output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (size > kAftpV1MaxBytes) {
    return Status::kInvalidProgram;
  }
  Reader reader(data, size);
  uint32_t version, count;
  if (!reader.Word(version)) {
    return Status::kInvalidProgram;
  }
  if (version != kVersion) {
    return Status::kUnsupported;
  }
  AxesEvaluationRecords decoded;
  if (!reader.Word(decoded.tiling_count) || decoded.tiling_count > kTilingExpressionMaxItems) {
    return Status::kInvalidProgram;
  }
  auto status = ReadRecord(reader, decoded.methods, DecodeAxesModelMethodsV1);
  if (status != Status::kOk) {
    return status;
  }
  if (!reader.Count(count, kMinConstraintBindingBytes)) {
    return Status::kInvalidProgram;
  }
  decoded.bindings.constraints.resize(count);
  for (auto &binding : decoded.bindings.constraints) {
    status = ReadRecord(reader, binding.callback, DecodeAxesCallbackV1);
    if (status != Status::kOk) {
      return status;
    }
    if (!ReadIndices(reader, binding.tiling_indices) || !ReadIndices(reader, binding.shape_indices)) {
      return Status::kInvalidProgram;
    }
  }
  if (!reader.Count(count, kMinUpperBoundBindingBytes) || count > decoded.tiling_count) {
    return Status::kInvalidProgram;
  }
  decoded.bindings.upper_bounds.resize(count);
  for (auto &binding : decoded.bindings.upper_bounds) {
    status = ReadRecord(reader, binding.callback, DecodeAxesCallbackV1);
    if (status != Status::kOk) {
      return status;
    }
    if (!reader.Word(binding.target_index) || !ReadIndices(reader, binding.parent_indices)) {
      return Status::kInvalidProgram;
    }
  }
  if (!reader.Done()) {
    return Status::kInvalidProgram;
  }
  status = VerifyAxesCallbackBindings(decoded.bindings, decoded.tiling_count, decoded.methods.front()->InputCount());
  if (status == Status::kOk) {
    *output = std::move(decoded);
  }
  return status;
}

TilingExpressionStatus EncodeAxesEvaluationV1(const AxesEvaluationDefinition &definition,
                                              std::vector<uint8_t> *output) {
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (definition.tiling_count > kTilingExpressionMaxItems ||
      definition.constraints.size() > kTilingExpressionMaxItems ||
      definition.upper_bounds.size() > definition.tiling_count) {
    return Status::kInvalidProgram;
  }
  Writer writer;
  writer.Word(kVersion);
  writer.Word(definition.tiling_count);
  std::vector<uint8_t> record;
  auto status = EncodeAxesModelMethodsV1(definition.methods, &record);
  if (status != Status::kOk) {
    return status;
  }
  if (!WriteRecord(writer, record)) {
    return Status::kInvalidProgram;
  }
  writer.Word(static_cast<uint32_t>(definition.constraints.size()));
  for (const auto &binding : definition.constraints) {
    status = EncodeAxesCallbackV1(binding.callback, &record);
    if (status != Status::kOk) {
      return status;
    }
    if (!WriteRecord(writer, record) || !WriteIndices(writer, binding.tiling_indices) ||
        !WriteIndices(writer, binding.shape_indices)) {
      return Status::kInvalidProgram;
    }
  }
  writer.Word(static_cast<uint32_t>(definition.upper_bounds.size()));
  for (const auto &binding : definition.upper_bounds) {
    status = EncodeAxesCallbackV1(binding.callback, &record);
    if (status != Status::kOk) {
      return status;
    }
    if (!WriteRecord(writer, record)) {
      return Status::kInvalidProgram;
    }
    writer.Word(binding.target_index);
    if (!WriteIndices(writer, binding.parent_indices)) {
      return Status::kInvalidProgram;
    }
  }
  AxesEvaluationRecords validated;
  status = DecodeAxesEvaluationV1(writer.bytes.data(), writer.bytes.size(), &validated);
  if (status == Status::kOk) {
    *output = std::move(writer.bytes);
  }
  return status;
}
AxesEvaluationEnvelopeError EncodeAxesEvaluationEnvelopeV1(const AxesEvaluationDefinition &definition,
                                                           const AftpV1EncodeOptions &options,
                                                           std::vector<uint8_t> *output) {
  if (output == nullptr || options.schema_hash == 0) {
    return {AftpV1Status::kInvalidArgument, Status::kOk};
  }
  std::vector<uint8_t> payload;
  const auto status = EncodeAxesEvaluationV1(definition, &payload);
  if (status != Status::kOk) {
    return {AftpV1Status::kOk, status};
  }
  return {
      EncodeAftpSectionPayloadV1(payload.data(), payload.size(), AftpV1SectionType::kTypedPlanPayload, options, output),
      Status::kOk};
}

AxesEvaluationEnvelopeError DecodeAxesEvaluationEnvelopeV1(const void *data, size_t size,
                                                           const AxesEvaluationEnvelopeRequirements &requirements,
                                                           AftpV1Header *header, AxesEvaluationRecords *output) {
  if (header == nullptr || output == nullptr || requirements.schema_hash == 0) {
    return {AftpV1Status::kInvalidArgument, Status::kOk};
  }
  AftpV1Header decoded_header{};
  AftpV1ProgramPayload payload;
  const auto status = DecodeAftpProgramWithManifestV1(data, size, &decoded_header, &payload);
  if (status != AftpV1Status::kOk) {
    return {status, Status::kOk};
  }
  if (payload.section_type != AftpV1SectionType::kTypedPlanPayload || !payload.manifest_payload.empty()) {
    return {AftpV1Status::kInvalidSection, Status::kOk};
  }
  if (!MatchesEnvelope(decoded_header, requirements)) {
    return {AftpV1Status::kUnsupportedFeature, Status::kOk};
  }
  const auto evaluation = DecodeAxesEvaluationV1(payload.plan_payload.data(), payload.plan_payload.size(), output);
  if (evaluation == Status::kOk) {
    *header = decoded_header;
  }
  return {AftpV1Status::kOk, evaluation};
}
TilingExpressionStatus EncodeAxesEvaluationWithLayoutV1(const AxesEvaluationDefinition &definition,
                                                        uint32_t payload_size,
                                                        const std::vector<TilingResultField> &fields,
                                                        std::vector<uint8_t> *output) {
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  std::vector<uint8_t> evaluation;
  auto status = EncodeAxesEvaluationV1(definition, &evaluation);
  if (status != Status::kOk) {
    return status;
  }
  std::vector<uint8_t> layout;
  status = EncodeTilingResultLayoutV1(payload_size, definition.tiling_count, fields, &layout);
  if (status != Status::kOk) {
    return status;
  }
  Writer writer;
  writer.Word(kVersion);
  if (!WriteRecord(writer, evaluation) || layout.size() > kAftpV1MaxBytes - writer.bytes.size()) {
    return Status::kInvalidProgram;
  }
  writer.bytes.insert(writer.bytes.end(), layout.begin(), layout.end());
  *output = std::move(writer.bytes);
  return Status::kOk;
}

TilingExpressionStatus DecodeAxesEvaluationWithLayoutV1(const void *data, size_t size,
                                                        AxesEvaluationWithLayout *output) {
  if (data == nullptr || output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (size > kAftpV1MaxBytes) {
    return Status::kInvalidProgram;
  }
  Reader reader(data, size);
  uint32_t version;
  if (!reader.Word(version)) {
    return Status::kInvalidProgram;
  }
  if (version != kVersion) {
    return Status::kUnsupported;
  }
  AxesEvaluationWithLayout decoded;
  auto status = ReadRecord(reader, decoded.evaluation, DecodeAxesEvaluationV1);
  if (status != Status::kOk) {
    return status;
  }
  status = DecodeTilingResultLayoutV1(reader.Data(), reader.Size(), &decoded.layout);
  if (status != Status::kOk) {
    return status;
  }
  if (decoded.layout->ValueCount() != decoded.evaluation.tiling_count) {
    return Status::kInvalidProgram;
  }
  decoded.layout_identity = AftpV1Hash(reader.Data(), reader.Size());
  *output = std::move(decoded);
  return Status::kOk;
}
static bool ValidateDerivedLayout(const TilingResultLayout &tiles, const TilingResultLayout &combined,
                                  size_t stage_count) {
  if (tiles.PayloadSize() != combined.PayloadSize() || !combined.HasAllValues() ||
      combined.ValueCount() != tiles.ValueCount() + stage_count) {
    return false;
  }
  size_t tile_fields = 0;
  for (const auto &field : combined.Fields()) {
    if (field.type != TilingProgramIRValueType::kUint32) {
      return false;
    }
    if (field.value_index < tiles.ValueCount()) {
      if (tile_fields == tiles.Fields().size()) {
        return false;
      }
      const auto &expected = tiles.Fields()[tile_fields++];
      if (field.value_index != expected.value_index || field.byte_offset != expected.byte_offset ||
          field.type != expected.type) {
        return false;
      }
    }
  }
  return tile_fields == tiles.Fields().size();
}

static TilingExpressionStatus EncodeAxesSolve(const AxesEvaluationDefinition &definition,
                                              const AxesSolveTopology &topology, uint32_t payload_size,
                                              const std::vector<TilingResultField> &fields,
                                              std::vector<uint8_t> *output, uint32_t solve_flags,
                                              const AxesDerivedDefinition *derived) {
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  if ((solve_flags & ~kKnownSolveFlags) != 0 || !VerifyAxesSolveTopology(topology) ||
      !std::all_of(fields.begin(), fields.end(), TilingResultLayout::IsTileField) ||
      topology.variables.size() != definition.tiling_count ||
      topology.constraints.size() != definition.constraints.size() ||
      !HasSolveUpperBounds(topology, definition.upper_bounds)) {
    return Status::kInvalidProgram;
  }
  for (const auto &method : definition.methods) {
    if (method.pure_mc_count != topology.pure_mc.size() || method.local_buffer_count != topology.local_buffer.size()) {
      return Status::kInvalidProgram;
    }
  }
  std::vector<uint8_t> evaluation;
  const auto status = EncodeAxesEvaluationWithLayoutV1(definition, payload_size, fields, &evaluation);
  if (status != Status::kOk) {
    return status;
  }
  Writer writer;
  const bool bind_resources =
      derived != nullptr ||
      std::any_of(topology.constraints.begin(), topology.constraints.end(), [](const auto &constraint) {
        return constraint.resource_source == AxesConstraintResource::kRuntimeUb;
      });
  writer.Word(derived != nullptr
                  ? kSolveResourceWritebackVersion
                  : (bind_resources ? kSolveResourceVersion : (solve_flags == 0 ? kVersion : kSolvePolicyVersion)));
  if (bind_resources || solve_flags != 0) {
    writer.Word(solve_flags);
  }
  if (!WriteRecord(writer, evaluation)) {
    return Status::kInvalidProgram;
  }
  if (derived != nullptr) {
    if (derived->stages.empty() || derived->stages.size() > kTilingExpressionMaxItems ||
        definition.tiling_count + derived->stages.size() > kTilingExpressionMaxItems) {
      return Status::kInvalidProgram;
    }
    writer.Word(derived->shape_count);
    writer.Word(static_cast<uint32_t>(derived->stages.size()));
    std::vector<uint8_t> record;
    for (const auto &stage : derived->stages) {
      if (EncodeTilingStageV1(stage, &record) != Status::kOk || !WriteRecord(writer, record)) {
        return Status::kInvalidProgram;
      }
    }
    if (EncodeTilingResultLayoutV1(payload_size, definition.tiling_count + derived->stages.size(), derived->fields,
                                   &record) != Status::kOk ||
        !WriteRecord(writer, record)) {
      return Status::kInvalidProgram;
    }
  }
  if (!WriteTopology(writer, topology, bind_resources)) {
    return Status::kInvalidProgram;
  }
  *output = std::move(writer.bytes);
  return Status::kOk;
}

TilingExpressionStatus EncodeAxesSolveV1(const AxesEvaluationDefinition &definition, const AxesSolveTopology &topology,
                                         uint32_t payload_size, const std::vector<TilingResultField> &fields,
                                         std::vector<uint8_t> *output, uint32_t solve_flags) {
  return EncodeAxesSolve(definition, topology, payload_size, fields, output, solve_flags, nullptr);
}

TilingExpressionStatus DecodeAxesSolveV1(const void *data, size_t size, AxesSolveRecords *output) {
  if (data == nullptr || output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (size > kAftpV1MaxBytes) {
    return Status::kInvalidProgram;
  }
  Reader reader(data, size);
  uint32_t version;
  if (!reader.Word(version)) {
    return Status::kInvalidProgram;
  }
  if (version != kVersion && version != kSolvePolicyVersion && version != kSolveResourceVersion &&
      version != kSolveDerivedVersion && version != kSolveResourceWritebackVersion) {
    return Status::kUnsupported;
  }
  AxesSolveRecords decoded;
  if (version != kVersion &&
      (!reader.Word(decoded.solve_flags) || (version == kSolvePolicyVersion && decoded.solve_flags == 0) ||
       (decoded.solve_flags & ~kKnownSolveFlags) != 0)) {
    return Status::kInvalidProgram;
  }
  const auto status =
      ReadRecord(reader, static_cast<AxesEvaluationWithLayout &>(decoded), DecodeAxesEvaluationWithLayoutV1);
  if (status != Status::kOk) {
    return status;
  }
  if (version >= kSolveDerivedVersion) {
    uint32_t count;
    if (!reader.Word(decoded.derived.shape_count) || decoded.derived.shape_count > kTilingExpressionMaxItems ||
        !reader.Count(count, kWordBytes) || count == 0) {
      return Status::kInvalidProgram;
    }
    for (uint32_t i = 0; i < count; ++i) {
      TilingProgramIRStageDefinition stage;
      std::shared_ptr<const TilingProgramIRCompiledStage> compiled;
      const auto slots = decoded.derived.shape_count + decoded.evaluation.tiling_count;
      if (ReadRecord(reader, stage, DecodeTilingStageV1) != Status::kOk ||
          (stage.input_types.size() < slots ||
           stage.input_types.size() > slots + (version == kSolveDerivedVersion ? 1U : 2U)) ||
          !std::all_of(stage.input_types.begin(), stage.input_types.end(),
                       [](auto type) { return type == TilingProgramIRValueType::kUint32; }) ||
          BuildTilingProgramIRStage(stage, &compiled).status != Status::kOk) {
        return Status::kInvalidProgram;
      }
      decoded.derived.stages.push_back(std::move(compiled));
    }
    const auto decode_derived_layout = [&decoded](const void *data, size_t size,
                                                  std::shared_ptr<const TilingResultLayout> *layout) {
      const auto result = DecodeTilingResultLayoutV1(data, size, layout);
      if (result == Status::kOk) {
        decoded.layout_identity = AftpV1Hash(data, size);
      }
      return result;
    };
    if (ReadRecord(reader, decoded.derived.layout, decode_derived_layout) != Status::kOk ||
        !ValidateDerivedLayout(*decoded.layout, *decoded.derived.layout, count)) {
      return Status::kInvalidProgram;
    }
  }
  if (!decoded.layout->HasAllValues() || !decoded.layout->HasOnlyTileFields() ||
      !ReadTopology(reader, decoded.topology, version >= kSolveResourceVersion) ||
      decoded.topology.variables.size() != decoded.evaluation.tiling_count ||
      decoded.topology.constraints.size() != decoded.evaluation.bindings.constraints.size() ||
      !HasSolveUpperBounds(decoded.topology, decoded.evaluation.bindings.upper_bounds)) {
    return Status::kInvalidProgram;
  }
  for (const auto &method : decoded.evaluation.methods) {
    if (method->PureMcCount() != decoded.topology.pure_mc.size() ||
        method->LocalBufferCount() != decoded.topology.local_buffer.size()) {
      return Status::kInvalidProgram;
    }
  }
  *output = std::move(decoded);
  return Status::kOk;
}
uint64_t HashAxesShapeSlots(const std::vector<std::string> &ordered_names) {
  if (ordered_names.size() > kTilingExpressionMaxItems) {
    return 0;
  }
  Writer writer;
  writer.Word(static_cast<uint32_t>(ordered_names.size()));
  for (const auto &name : ordered_names) {
    if (name.empty() || writer.bytes.size() > kAftpV1MaxBytes - kWordBytes ||
        name.size() > kAftpV1MaxBytes - kWordBytes - writer.bytes.size()) {
      return 0;
    }
    writer.Word(static_cast<uint32_t>(name.size()));
    writer.bytes.insert(writer.bytes.end(), name.begin(), name.end());
  }
  return AftpV1Hash(writer.bytes.data(), writer.bytes.size());
}

TilingExpressionStatus EncodeAxesSolveManifestV1(const AxesSolveManifestIdentity &identity, uint32_t payload_size,
                                                 uint32_t value_count, const std::vector<TilingResultField> &fields,
                                                 std::vector<uint8_t> *output) {
  if (output == nullptr || identity.graph_case == 0 || identity.shape_slots == 0 || identity.companion_assets == 0) {
    return Status::kInvalidArgument;
  }
  std::vector<uint8_t> layout;
  const auto status = EncodeTilingResultLayoutV1(payload_size, value_count, fields, &layout);
  if (status != Status::kOk) {
    return status;
  }
  Writer writer;
  writer.Word(kSolveManifestMagic);
  writer.Word(kVersion);
  for (const auto value : {identity.graph_case, identity.shape_slots, identity.companion_assets,
                           AftpV1Hash(layout.data(), layout.size())}) {
    writer.Word(static_cast<uint32_t>(value));
    writer.Word(static_cast<uint32_t>(value >> kWordBits));
  }
  *output = std::move(writer.bytes);
  return Status::kOk;
}

AxesEvaluationEnvelopeError EncodeAxesSolveEnvelopeV1(const AxesEvaluationDefinition &definition,
                                                      const AxesSolveTopology &topology, uint32_t payload_size,
                                                      const std::vector<TilingResultField> &fields,
                                                      const AxesSolveEnvelopeOptions &options,
                                                      std::vector<uint8_t> *output) {
  const bool structured = options.structured_identity != nullptr;
  const auto &provided = options.manifest;
  if (output == nullptr || (!structured && options.derived != nullptr) ||
      (structured && (provided.data != nullptr || provided.size != 0)) ||
      (!structured && (provided.data == nullptr || provided.size == 0))) {
    return {AftpV1Status::kInvalidArgument, Status::kOk};
  }
  std::vector<uint8_t> manifest_bytes;
  auto manifest = provided;
  if (structured) {
    const auto &manifest_fields = options.derived == nullptr ? fields : options.derived->fields;
    const auto stage_count = options.derived == nullptr ? 0U : options.derived->stages.size();
    if (stage_count > kTilingExpressionMaxItems || definition.tiling_count + stage_count > kTilingExpressionMaxItems) {
      return {AftpV1Status::kOk, Status::kInvalidProgram};
    }
    const auto status = EncodeAxesSolveManifestV1(*options.structured_identity, payload_size,
                                                  definition.tiling_count + static_cast<uint32_t>(stage_count),
                                                  manifest_fields, &manifest_bytes);
    if (status != Status::kOk) {
      return {AftpV1Status::kOk, status};
    }
    manifest = {manifest_bytes.data(), manifest_bytes.size()};
  }
  std::vector<uint8_t> plan;
  const auto status =
      EncodeAxesSolve(definition, topology, payload_size, fields, &plan, options.solve_flags, options.derived);
  if (status != Status::kOk) {
    return {AftpV1Status::kOk, status};
  }
  return {EncodeAftpProgramWithManifestV1(
              {{plan.data(), plan.size()}, manifest, AftpV1SectionType::kAxesSolvePayload, options.header}, output),
          Status::kOk};
}

AxesEvaluationEnvelopeError DecodeAxesSolveEnvelopeV1(const void *data, size_t size,
                                                      const AxesEvaluationEnvelopeRequirements &requirements,
                                                      const AftpV1PayloadDescriptor &expected_manifest,
                                                      AftpV1Header *header, AxesSolveRecords *output) {
  if (header == nullptr || output == nullptr || requirements.schema_hash == 0 || expected_manifest.data == nullptr ||
      expected_manifest.size == 0) {
    return {AftpV1Status::kInvalidArgument, Status::kOk};
  }
  AftpV1Header decoded_header{};
  AftpV1ProgramPayload payload;
  const auto status = DecodeAftpProgramWithManifestV1(data, size, &decoded_header, &payload);
  if (status != AftpV1Status::kOk) {
    return {status, Status::kOk};
  }
  if (payload.section_type != AftpV1SectionType::kAxesSolvePayload || payload.manifest_payload.empty()) {
    return {AftpV1Status::kInvalidSection, Status::kOk};
  }
  if (!MatchesEnvelope(decoded_header, requirements) || payload.manifest_payload.size() != expected_manifest.size ||
      std::memcmp(payload.manifest_payload.data(), expected_manifest.data, expected_manifest.size) != 0) {
    return {AftpV1Status::kUnsupportedFeature, Status::kOk};
  }
  AxesSolveRecords decoded;
  auto solve = DecodeAxesSolveV1(payload.plan_payload.data(), payload.plan_payload.size(), &decoded);
  if (solve == Status::kOk) {
    solve = VerifySolveManifest({payload.manifest_payload.data(), payload.manifest_payload.size()},
                                decoded.layout_identity, decoded.derived.layout != nullptr);
  }
  if (solve == Status::kOk) {
    *header = decoded_header;
    *output = std::move(decoded);
  }
  return {AftpV1Status::kOk, solve};
}
}  // namespace att
