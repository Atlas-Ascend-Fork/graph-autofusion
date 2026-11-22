/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aftp_v1.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace att {
namespace {

void AppendUint16(std::vector<uint8_t> *output, uint16_t value) {
  for (size_t index = 0U; index < sizeof(value); ++index) {
    output->push_back(static_cast<uint8_t>(value >> (index * 8U)));
  }
}

void AppendUint32(std::vector<uint8_t> *output, uint32_t value) {
  for (size_t index = 0U; index < sizeof(value); ++index) {
    output->push_back(static_cast<uint8_t>(value >> (index * 8U)));
  }
}

void AppendUint64(std::vector<uint8_t> *output, uint64_t value) {
  for (size_t index = 0U; index < sizeof(value); ++index) {
    output->push_back(static_cast<uint8_t>(value >> (index * 8U)));
  }
}

uint16_t ReadUint16(const uint8_t *data) {
  uint16_t value = 0U;
  for (size_t index = 0U; index < sizeof(value); ++index) {
    value |= static_cast<uint16_t>(static_cast<uint16_t>(data[index]) << (index * 8U));
  }
  return value;
}

uint32_t ReadUint32(const uint8_t *data) {
  uint32_t value = 0U;
  for (size_t index = 0U; index < sizeof(value); ++index) {
    value |= static_cast<uint32_t>(static_cast<uint32_t>(data[index]) << (index * 8U));
  }
  return value;
}

uint64_t ReadUint64(const uint8_t *data) {
  uint64_t value = 0U;
  for (size_t index = 0U; index < sizeof(value); ++index) {
    value |= static_cast<uint64_t>(static_cast<uint64_t>(data[index]) << (index * 8U));
  }
  return value;
}

bool CheckedAdd(size_t left, size_t right, size_t *result) {
  if (left > std::numeric_limits<size_t>::max() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

AftpV1Header ReadHeader(const uint8_t *data) {
  AftpV1Header header{};
  header.magic = ReadUint32(data);
  header.format_major = ReadUint16(data + 4U);
  header.format_minor = ReadUint16(data + 6U);
  header.producer_build_id = ReadUint64(data + 8U);
  header.min_runtime_abi = ReadUint64(data + 16U);
  header.max_runtime_abi = ReadUint64(data + 24U);
  header.total_size = ReadUint64(data + 32U);
  header.section_count = ReadUint32(data + 40U);
  header.flags = ReadUint32(data + 44U);
  header.program_id = ReadUint64(data + 48U);
  header.program_hash = ReadUint64(data + 56U);
  header.schema_hash = ReadUint64(data + 64U);
  header.required_capabilities = ReadUint64(data + 72U);
  header.target_constraints = ReadUint64(data + 80U);
  return header;
}

void WriteHeader(std::vector<uint8_t> *output, const AftpV1Header &header) {
  AppendUint32(output, header.magic);
  AppendUint16(output, header.format_major);
  AppendUint16(output, header.format_minor);
  AppendUint64(output, header.producer_build_id);
  AppendUint64(output, header.min_runtime_abi);
  AppendUint64(output, header.max_runtime_abi);
  AppendUint64(output, header.total_size);
  AppendUint32(output, header.section_count);
  AppendUint32(output, header.flags);
  AppendUint64(output, header.program_id);
  AppendUint64(output, header.program_hash);
  AppendUint64(output, header.schema_hash);
  AppendUint64(output, header.required_capabilities);
  AppendUint64(output, header.target_constraints);
}

void AppendSectionHeader(std::vector<uint8_t> *output, const AftpV1SectionType section_type, const uint64_t offset,
                         const uint64_t size, const uint64_t hash) {
  AppendUint32(output, static_cast<uint32_t>(section_type));
  AppendUint32(output, 0U);
  AppendUint64(output, offset);
  AppendUint64(output, size);
  AppendUint64(output, hash);
}

struct AftpV1SectionRecord {
  AftpV1SectionType type;
  uint64_t offset;
  uint64_t size;
  uint64_t hash;
};

bool IsSupportedSectionType(const uint32_t type) {
  return type == static_cast<uint32_t>(AftpV1SectionType::kPlanPayload) ||
         type == static_cast<uint32_t>(AftpV1SectionType::kTypedPlanPayload) ||
         type == static_cast<uint32_t>(AftpV1SectionType::kManifestPayload) ||
         type == static_cast<uint32_t>(AftpV1SectionType::kAxesSolvePayload);
}

bool IsPlanSection(const AftpV1SectionType type) {
  return type == AftpV1SectionType::kPlanPayload || type == AftpV1SectionType::kTypedPlanPayload ||
         type == AftpV1SectionType::kAxesSolvePayload;
}

AftpV1Status ParseAftpSections(const uint8_t *bytes, const size_t size, const AftpV1Header &header,
                               std::vector<AftpV1SectionRecord> *sections) {
  const size_t table_size = static_cast<size_t>(header.section_count) * kAftpV1SectionSize;
  const size_t table_end = kAftpV1HeaderSize + table_size;
  if (size < table_end) {
    return AftpV1Status::kInvalidSize;
  }
  sections->clear();
  for (uint32_t index = 0U; index < header.section_count; index++) {
    const uint8_t *section = bytes + kAftpV1HeaderSize + static_cast<size_t>(index) * kAftpV1SectionSize;
    const uint32_t type_value = ReadUint32(section);
    const uint32_t reserved = ReadUint32(section + 4U);
    const uint64_t offset = ReadUint64(section + 8U);
    const uint64_t section_size = ReadUint64(section + 16U);
    const uint64_t hash = ReadUint64(section + 24U);
    if (!IsSupportedSectionType(type_value)) {
      return AftpV1Status::kUnsupportedFeature;
    }
    if (reserved != 0U || offset < table_end || section_size == 0U || offset > size || section_size > size - offset) {
      return AftpV1Status::kInvalidSection;
    }
    sections->push_back({static_cast<AftpV1SectionType>(type_value), offset, section_size, hash});
  }
  std::sort(sections->begin(), sections->end(), [](const AftpV1SectionRecord &left, const AftpV1SectionRecord &right) {
    return left.offset < right.offset;
  });
  for (size_t index = 1U; index < sections->size(); index++) {
    if ((*sections)[index - 1U].offset + (*sections)[index - 1U].size != (*sections)[index].offset) {
      return AftpV1Status::kInvalidSection;
    }
  }
  if (sections->front().offset != table_end || sections->back().offset + sections->back().size != size) {
    return AftpV1Status::kInvalidSection;
  }
  return AftpV1Status::kOk;
}

AftpV1Status ValidateEnvelope(const void *data, size_t size, AftpV1Header *header,
                              std::vector<AftpV1SectionRecord> *sections) {
  if (data == nullptr || size < kAftpV1HeaderSize) {
    return AftpV1Status::kInvalidArgument;
  }
  const auto *bytes = static_cast<const uint8_t *>(data);
  const auto parsed = ReadHeader(bytes);
  if (parsed.magic != kAftpV1Magic) {
    return AftpV1Status::kInvalidMagic;
  }
  if (parsed.format_major != kAftpV1FormatMajor || parsed.format_minor != kAftpV1FormatMinor) {
    return AftpV1Status::kUnsupportedVersion;
  }
  if (parsed.total_size != size || size > kAftpV1MaxBytes || parsed.producer_build_id == 0U ||
      parsed.program_id == 0U || parsed.schema_hash == 0U || parsed.min_runtime_abi > parsed.max_runtime_abi) {
    return AftpV1Status::kInvalidSize;
  }
  if (parsed.min_runtime_abi > kAftpV1RuntimeAbi || parsed.max_runtime_abi < kAftpV1RuntimeAbi) {
    return AftpV1Status::kUnsupportedVersion;
  }
  if (parsed.flags != 0U) {
    return AftpV1Status::kUnsupportedFeature;
  }
  if (parsed.section_count == 0U || parsed.section_count > 2U) {
    return AftpV1Status::kInvalidSection;
  }
  const auto status = ParseAftpSections(bytes, size, parsed, sections);
  if (status != AftpV1Status::kOk) {
    return status;
  }
  size_t plan_count = 0U;
  size_t manifest_count = 0U;
  for (const auto &section : *sections) {
    if (IsPlanSection(section.type)) {
      ++plan_count;
    } else {
      ++manifest_count;
    }
  }
  if (plan_count != 1U || manifest_count > 1U) {
    return AftpV1Status::kInvalidSection;
  }
  for (const auto &section : *sections) {
    const auto actual_hash = AftpV1Hash(bytes + section.offset, static_cast<size_t>(section.size));
    if (actual_hash != section.hash || (IsPlanSection(section.type) && actual_hash != parsed.program_hash)) {
      return AftpV1Status::kHashMismatch;
    }
  }
  *header = parsed;
  return AftpV1Status::kOk;
}

}  // namespace

uint64_t AftpV1Hash(const void *data, const size_t size) {
  const auto *bytes = static_cast<const uint8_t *>(data);
  uint64_t hash = 1469598103934665603ULL;
  for (size_t index = 0U; index < size; ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ULL;
  }
  return hash;
}

bool LooksLikeAftpV1(const void *data, const size_t size) {
  if (data == nullptr || size < sizeof(uint32_t)) {
    return false;
  }
  return ReadUint32(static_cast<const uint8_t *>(data)) == kAftpV1Magic;
}

AftpV1Status PeekAftpV1Header(const void *data, const size_t size, AftpV1Header *header) {
  if (header == nullptr) {
    return AftpV1Status::kInvalidArgument;
  }
  std::vector<AftpV1SectionRecord> sections;
  return ValidateEnvelope(data, size, header, &sections);
}

AftpV1Status EncodeAftpSectionPayloadV1(const void *payload_data, const size_t payload_size,
                                        const AftpV1SectionType section_type, const AftpV1EncodeOptions &options,
                                        std::vector<uint8_t> *output) {
  if (payload_data == nullptr || payload_size == 0U || output == nullptr || options.producer_build_id == 0U ||
      options.program_id == 0U || options.schema_hash == 0U) {
    return AftpV1Status::kInvalidArgument;
  }
  if (section_type != AftpV1SectionType::kPlanPayload && section_type != AftpV1SectionType::kTypedPlanPayload) {
    return AftpV1Status::kInvalidSection;
  }
  size_t total_size = 0U;
  if (!CheckedAdd(kAftpV1HeaderSize, kAftpV1SectionSize, &total_size) ||
      !CheckedAdd(total_size, payload_size, &total_size) || total_size > kAftpV1MaxBytes) {
    return AftpV1Status::kInvalidSize;
  }

  AftpV1Header header{};
  header.magic = kAftpV1Magic;
  header.format_major = kAftpV1FormatMajor;
  header.format_minor = kAftpV1FormatMinor;
  header.producer_build_id = options.producer_build_id;
  header.min_runtime_abi = kAftpV1RuntimeAbi;
  header.max_runtime_abi = kAftpV1RuntimeAbi;
  header.total_size = total_size;
  header.section_count = 1U;
  header.flags = 0U;
  header.program_id = options.program_id;
  header.program_hash = AftpV1Hash(payload_data, payload_size);
  header.schema_hash = options.schema_hash;
  header.required_capabilities = options.required_capabilities;
  header.target_constraints = options.target_constraints;

  std::vector<uint8_t> encoded;
  encoded.reserve(total_size);
  WriteHeader(&encoded, header);
  AppendUint32(&encoded, static_cast<uint32_t>(section_type));
  AppendUint32(&encoded, 0U);
  AppendUint64(&encoded, kAftpV1HeaderSize + kAftpV1SectionSize);
  AppendUint64(&encoded, payload_size);
  AppendUint64(&encoded, header.program_hash);
  const auto *payload_bytes = static_cast<const uint8_t *>(payload_data);
  encoded.insert(encoded.end(), payload_bytes, payload_bytes + payload_size);
  output->swap(encoded);
  return AftpV1Status::kOk;
}

AftpV1Status DecodeAftpSectionPayloadV1(const void *data, const size_t size, AftpV1Header *header,
                                        AftpV1SectionType *section_type, std::vector<uint8_t> *payload) {
  if (header == nullptr || section_type == nullptr || payload == nullptr) {
    return AftpV1Status::kInvalidArgument;
  }
  AftpV1ProgramPayload decoded;
  const AftpV1Status status = DecodeAftpProgramWithManifestV1(data, size, header, &decoded);
  if (status == AftpV1Status::kOk) {
    *section_type = decoded.section_type;
    *payload = std::move(decoded.plan_payload);
  }
  return status;
}

AftpV1Status EncodeAftpProgramWithManifestV1(const AftpV1ProgramWithManifestDescriptor &descriptor,
                                             std::vector<uint8_t> *output) {
  if (descriptor.plan.data == nullptr || descriptor.plan.size == 0U || descriptor.manifest.data == nullptr ||
      descriptor.manifest.size == 0U || output == nullptr || descriptor.options.producer_build_id == 0U ||
      descriptor.options.program_id == 0U || descriptor.options.schema_hash == 0U ||
      !IsPlanSection(descriptor.plan_section_type)) {
    return AftpV1Status::kInvalidArgument;
  }
  size_t total_size = kAftpV1HeaderSize + 2U * kAftpV1SectionSize;
  if (!CheckedAdd(total_size, descriptor.plan.size, &total_size) ||
      !CheckedAdd(total_size, descriptor.manifest.size, &total_size) || total_size > kAftpV1MaxBytes) {
    return AftpV1Status::kInvalidSize;
  }

  const auto *plan_bytes = static_cast<const uint8_t *>(descriptor.plan.data);
  const auto *manifest_bytes = static_cast<const uint8_t *>(descriptor.manifest.data);
  const uint64_t plan_hash = AftpV1Hash(plan_bytes, descriptor.plan.size);
  const uint64_t manifest_hash = AftpV1Hash(manifest_bytes, descriptor.manifest.size);
  const uint64_t plan_offset = kAftpV1HeaderSize + 2U * kAftpV1SectionSize;
  const uint64_t manifest_offset = plan_offset + descriptor.plan.size;

  AftpV1Header header{};
  header.magic = kAftpV1Magic;
  header.format_major = kAftpV1FormatMajor;
  header.format_minor = kAftpV1FormatMinor;
  header.producer_build_id = descriptor.options.producer_build_id;
  header.min_runtime_abi = kAftpV1RuntimeAbi;
  header.max_runtime_abi = kAftpV1RuntimeAbi;
  header.total_size = total_size;
  header.section_count = 2U;
  header.flags = 0U;
  header.program_id = descriptor.options.program_id;
  header.program_hash = plan_hash;
  header.schema_hash = descriptor.options.schema_hash;
  header.required_capabilities = descriptor.options.required_capabilities;
  header.target_constraints = descriptor.options.target_constraints;

  std::vector<uint8_t> encoded;
  encoded.reserve(total_size);
  WriteHeader(&encoded, header);
  AppendSectionHeader(&encoded, descriptor.plan_section_type, plan_offset, descriptor.plan.size, plan_hash);
  AppendSectionHeader(&encoded, AftpV1SectionType::kManifestPayload, manifest_offset, descriptor.manifest.size,
                      manifest_hash);
  encoded.insert(encoded.end(), plan_bytes, plan_bytes + descriptor.plan.size);
  encoded.insert(encoded.end(), manifest_bytes, manifest_bytes + descriptor.manifest.size);
  output->swap(encoded);
  return AftpV1Status::kOk;
}

AftpV1Status DecodeAftpProgramWithManifestV1(const void *data, const size_t size, AftpV1Header *header,
                                             AftpV1ProgramPayload *payload) {
  if (header == nullptr || payload == nullptr) {
    return AftpV1Status::kInvalidArgument;
  }
  AftpV1Header validated{};
  std::vector<AftpV1SectionRecord> sections;
  const auto status = ValidateEnvelope(data, size, &validated, &sections);
  if (status != AftpV1Status::kOk) {
    return status;
  }
  const auto *bytes = static_cast<const uint8_t *>(data);
  AftpV1ProgramPayload decoded;
  for (const auto &section : sections) {
    if (IsPlanSection(section.type)) {
      decoded.section_type = section.type;
      decoded.plan_payload.assign(bytes + section.offset, bytes + section.offset + section.size);
    } else {
      decoded.manifest_payload.assign(bytes + section.offset, bytes + section.offset + section.size);
    }
  }
  // Publish only after every section is checked and both owned copies exist.
  *payload = std::move(decoded);
  *header = validated;
  return AftpV1Status::kOk;
}

AftpV1Status EncodeAftpPlanPayloadV1(const void *plan_data, const size_t plan_size, const AftpV1EncodeOptions &options,
                                     std::vector<uint8_t> *output) {
  return EncodeAftpSectionPayloadV1(plan_data, plan_size, AftpV1SectionType::kPlanPayload, options, output);
}

AftpV1Status DecodeAftpPlanPayloadV1(const void *data, const size_t size, AftpV1Header *header,
                                     std::vector<uint8_t> *plan_payload) {
  if (header == nullptr || plan_payload == nullptr) {
    return AftpV1Status::kInvalidArgument;
  }
  AftpV1Header validated{};
  AftpV1ProgramPayload decoded;
  const auto status = DecodeAftpProgramWithManifestV1(data, size, &validated, &decoded);
  if (status != AftpV1Status::kOk) {
    return status;
  }
  if (decoded.section_type != AftpV1SectionType::kPlanPayload) {
    return AftpV1Status::kInvalidSection;
  }
  *plan_payload = std::move(decoded.plan_payload);
  *header = validated;
  return AftpV1Status::kOk;
}

const char *AftpV1StatusMessage(const AftpV1Status status) {
  switch (status) {
    case AftpV1Status::kOk:
      return "ok";
    case AftpV1Status::kInvalidArgument:
      return "invalid argument";
    case AftpV1Status::kInvalidMagic:
      return "invalid AFTP magic";
    case AftpV1Status::kUnsupportedVersion:
      return "unsupported AFTP version";
    case AftpV1Status::kInvalidSize:
      return "invalid AFTP size or metadata";
    case AftpV1Status::kInvalidSection:
      return "invalid AFTP section";
    case AftpV1Status::kUnsupportedFeature:
      return "unsupported AFTP feature";
    case AftpV1Status::kHashMismatch:
      return "AFTP payload hash mismatch";
    default:
      return "unknown AFTP status";
  }
}

}  // namespace att
