/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_SERIALIZATION_AFTP_V1_H_
#define ATT_SERIALIZATION_AFTP_V1_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace att {

constexpr uint32_t kAftpV1Magic = 0x50544641U;
constexpr uint16_t kAftpV1FormatMajor = 1U;
constexpr uint16_t kAftpV1FormatMinor = 0U;
constexpr uint64_t kAftpV1RuntimeAbi = 1U;
constexpr size_t kAftpV1HeaderSize = 88U;
constexpr size_t kAftpV1SectionSize = 32U;
constexpr size_t kAftpV1MaxBytes = 64U * 1024U * 1024U;

enum class AftpV1SectionType : uint32_t {
  kPlanPayload = 1U,
  kTypedPlanPayload = 2U,
  kManifestPayload = 3U,
  kAxesSolvePayload = 4U,
};

enum class AftpV1Status {
  kOk,
  kInvalidArgument,
  kInvalidMagic,
  kUnsupportedVersion,
  kInvalidSize,
  kInvalidSection,
  kHashMismatch,
  kUnsupportedFeature,
};

struct AftpV1Header {
  uint32_t magic;
  uint16_t format_major;
  uint16_t format_minor;
  uint64_t producer_build_id;
  uint64_t min_runtime_abi;
  uint64_t max_runtime_abi;
  uint64_t total_size;
  uint32_t section_count;
  uint32_t flags;
  uint64_t program_id;
  uint64_t program_hash;
  uint64_t schema_hash;
  uint64_t required_capabilities;
  uint64_t target_constraints;
};

static_assert(sizeof(AftpV1Header) == kAftpV1HeaderSize, "AFTP V1 header layout must be fixed");

struct AftpV1EncodeOptions {
  uint64_t producer_build_id = 1U;
  uint64_t program_id = 1U;
  uint64_t schema_hash = 0U;
  uint64_t required_capabilities = 0U;
  uint64_t target_constraints = 0U;
};

struct AftpV1PayloadDescriptor {
  const void *data = nullptr;
  size_t size = 0U;
};

struct AftpV1ProgramWithManifestDescriptor {
  AftpV1PayloadDescriptor plan;
  AftpV1PayloadDescriptor manifest;
  AftpV1SectionType plan_section_type = AftpV1SectionType::kPlanPayload;
  AftpV1EncodeOptions options;
};

struct AftpV1ProgramPayload {
  AftpV1SectionType section_type = AftpV1SectionType::kPlanPayload;
  std::vector<uint8_t> plan_payload;
  std::vector<uint8_t> manifest_payload;
};

// Internal payload checksum; data must address size readable bytes (or size must be zero).
uint64_t AftpV1Hash(const void *data, size_t size);
bool LooksLikeAftpV1(const void *data, size_t size);
// All decoding entries validate the full envelope and publish outputs only on success.
// Capability/target matching and inner Program verification belong to Program Create.
// Allocation failure propagates to the owning API exception boundary.
AftpV1Status PeekAftpV1Header(const void *data, size_t size, AftpV1Header *header);
AftpV1Status EncodeAftpSectionPayloadV1(const void *payload_data, size_t payload_size, AftpV1SectionType section_type,
                                        const AftpV1EncodeOptions &options, std::vector<uint8_t> *output);
AftpV1Status DecodeAftpSectionPayloadV1(const void *data, size_t size, AftpV1Header *header,
                                        AftpV1SectionType *section_type, std::vector<uint8_t> *payload);
AftpV1Status EncodeAftpProgramWithManifestV1(const AftpV1ProgramWithManifestDescriptor &descriptor,
                                             std::vector<uint8_t> *output);
AftpV1Status DecodeAftpProgramWithManifestV1(const void *data, size_t size, AftpV1Header *header,
                                             AftpV1ProgramPayload *payload);
AftpV1Status EncodeAftpPlanPayloadV1(const void *plan_data, size_t plan_size, const AftpV1EncodeOptions &options,
                                     std::vector<uint8_t> *output);
AftpV1Status DecodeAftpPlanPayloadV1(const void *data, size_t size, AftpV1Header *header,
                                     std::vector<uint8_t> *plan_payload);
const char *AftpV1StatusMessage(AftpV1Status status);

}  // namespace att

#endif  // ATT_SERIALIZATION_AFTP_V1_H_
