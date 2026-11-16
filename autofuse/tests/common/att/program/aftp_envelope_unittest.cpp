/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "serialization/aftp_v1.h"

namespace att {
namespace {
constexpr uint64_t kSentinelId = 777U;
const std::vector<uint8_t> kSentinelBytes{91U, 92U};

void SetU64(std::vector<uint8_t> &data, size_t offset, uint64_t value) {
  for (size_t i = 0U; i < 8U; ++i) {
    data.at(offset + i) = static_cast<uint8_t>(value >> (i * 8U));
  }
}

std::vector<uint8_t> Envelope(bool manifest = true) {
  const std::vector<uint8_t> plan{1U, 3U, 5U};
  const std::vector<uint8_t> metadata{7U, 9U};
  AftpV1EncodeOptions options;
  options.schema_hash = 11U;
  options.program_id = 123U;
  options.producer_build_id = 456U;
  std::vector<uint8_t> result;
  if (manifest) {
    AftpV1ProgramWithManifestDescriptor descriptor;
    descriptor.plan = {plan.data(), plan.size()};
    descriptor.manifest = {metadata.data(), metadata.size()};
    descriptor.plan_section_type = AftpV1SectionType::kTypedPlanPayload;
    descriptor.options = options;
    EXPECT_EQ(EncodeAftpProgramWithManifestV1(descriptor, &result), AftpV1Status::kOk);
  } else {
    EXPECT_EQ(EncodeAftpPlanPayloadV1(plan.data(), plan.size(), options, &result), AftpV1Status::kOk);
  }
  return result;
}

void ExpectRejectedWithoutPublication(const std::vector<uint8_t> &data, AftpV1Status expected) {
  AftpV1Header header{};
  header.program_id = kSentinelId;
  AftpV1ProgramPayload payload;
  payload.section_type = AftpV1SectionType::kManifestPayload;
  payload.plan_payload = kSentinelBytes;
  payload.manifest_payload = kSentinelBytes;
  ASSERT_EQ(DecodeAftpProgramWithManifestV1(data.data(), data.size(), &header, &payload), expected);
  EXPECT_EQ(header.program_id, kSentinelId);
  EXPECT_EQ(payload.section_type, AftpV1SectionType::kManifestPayload);
  EXPECT_EQ(payload.plan_payload, kSentinelBytes);
  EXPECT_EQ(payload.manifest_payload, kSentinelBytes);
  EXPECT_EQ(PeekAftpV1Header(data.data(), data.size(), &header), expected);
  EXPECT_EQ(header.program_id, kSentinelId);
}
}  // namespace

TEST(AftpEnvelopeTest, RoundTripIsDeterministicAndOwned) {
  for (bool manifest : {false, true}) {
    auto data = Envelope(manifest);
    EXPECT_EQ(data, Envelope(manifest));
    AftpV1Header header{};
    AftpV1ProgramPayload payload;
    ASSERT_EQ(DecodeAftpProgramWithManifestV1(data.data(), data.size(), &header, &payload), AftpV1Status::kOk);
    EXPECT_EQ(header.program_id, 123U);
    EXPECT_EQ(header.producer_build_id, 456U);
    EXPECT_EQ(header.schema_hash, 11U);
    EXPECT_EQ(header.section_count, manifest ? 2U : 1U);
    EXPECT_EQ(payload.plan_payload, (std::vector<uint8_t>{1U, 3U, 5U}));
    EXPECT_EQ(payload.manifest_payload, manifest ? (std::vector<uint8_t>{7U, 9U}) : std::vector<uint8_t>{});
    std::fill(data.begin(), data.end(), 0U);
    EXPECT_EQ(payload.plan_payload, (std::vector<uint8_t>{1U, 3U, 5U}));
  }
}

TEST(AftpEnvelopeTest, CorruptManifestCannotPublishValidPlan) {
  auto data = Envelope();
  data.back() ^= 1U;
  ExpectRejectedWithoutPublication(data, AftpV1Status::kHashMismatch);
}

TEST(AftpEnvelopeTest, CorruptPlanAndHeaderHashAreRejected) {
  auto data = Envelope(false);
  data.back() ^= 1U;
  ExpectRejectedWithoutPublication(data, AftpV1Status::kHashMismatch);
  data = Envelope(false);
  data[56U] ^= 1U;
  ExpectRejectedWithoutPublication(data, AftpV1Status::kHashMismatch);
}

TEST(AftpEnvelopeTest, UnknownRequiredFlagsAndSectionsAreUnsupported) {
  auto data = Envelope();
  data[44U] = 1U;
  ExpectRejectedWithoutPublication(data, AftpV1Status::kUnsupportedFeature);
  data = Envelope();
  data[kAftpV1HeaderSize] = 255U;
  ExpectRejectedWithoutPublication(data, AftpV1Status::kUnsupportedFeature);
}

TEST(AftpEnvelopeTest, IncompatibleFormatOrRuntimeIsUnsupported) {
  auto data = Envelope();
  data[4U] = 2U;
  ExpectRejectedWithoutPublication(data, AftpV1Status::kUnsupportedVersion);
  data = Envelope();
  SetU64(data, 16U, 2U);
  SetU64(data, 24U, 2U);
  ExpectRejectedWithoutPublication(data, AftpV1Status::kUnsupportedVersion);
  SetU64(data, 16U, 3U);
  ExpectRejectedWithoutPublication(data, AftpV1Status::kInvalidSize);
}

TEST(AftpEnvelopeTest, DuplicatePlanOrMissingPlanIsInvalid) {
  auto data = Envelope();
  data[kAftpV1HeaderSize + kAftpV1SectionSize] = static_cast<uint8_t>(AftpV1SectionType::kPlanPayload);
  ExpectRejectedWithoutPublication(data, AftpV1Status::kInvalidSection);
  data = Envelope();
  data[kAftpV1HeaderSize] = static_cast<uint8_t>(AftpV1SectionType::kManifestPayload);
  ExpectRejectedWithoutPublication(data, AftpV1Status::kInvalidSection);
}

TEST(AftpEnvelopeTest, SectionTableMayBeReorderedWithoutChangingPayloadOrder) {
  auto data = Envelope();
  std::swap_ranges(data.begin() + kAftpV1HeaderSize, data.begin() + kAftpV1HeaderSize + kAftpV1SectionSize,
                   data.begin() + kAftpV1HeaderSize + kAftpV1SectionSize);
  AftpV1Header header{};
  AftpV1ProgramPayload payload;
  ASSERT_EQ(DecodeAftpProgramWithManifestV1(data.data(), data.size(), &header, &payload), AftpV1Status::kOk);
  EXPECT_EQ(payload.plan_payload, (std::vector<uint8_t>{1U, 3U, 5U}));
}

TEST(AftpEnvelopeTest, GapOverlapAndOverflowAreRejected) {
  auto data = Envelope();
  const size_t second = kAftpV1HeaderSize + kAftpV1SectionSize;
  const size_t payload_start = kAftpV1HeaderSize + 2U * kAftpV1SectionSize;
  data.insert(data.begin() + payload_start + 3U, 0U);
  SetU64(data, 32U, data.size());
  SetU64(data, second + 8U, payload_start + 4U);
  ExpectRejectedWithoutPublication(data, AftpV1Status::kInvalidSection);
  data = Envelope();
  SetU64(data, second + 8U, payload_start + 2U);
  ExpectRejectedWithoutPublication(data, AftpV1Status::kInvalidSection);
  data = Envelope();
  SetU64(data, second + 8U, UINT64_MAX);
  ExpectRejectedWithoutPublication(data, AftpV1Status::kInvalidSection);
  data = Envelope();
  SetU64(data, second + 16U, UINT64_MAX);
  ExpectRejectedWithoutPublication(data, AftpV1Status::kInvalidSection);
}

TEST(AftpEnvelopeTest, EveryTruncationIsRejectedWithoutPublication) {
  const auto valid = Envelope();
  for (size_t size = 0U; size < valid.size(); ++size) {
    const std::vector<uint8_t> data(valid.begin(), valid.begin() + size);
    ExpectRejectedWithoutPublication(
        data, size < kAftpV1HeaderSize ? AftpV1Status::kInvalidArgument : AftpV1Status::kInvalidSize);
  }
}

TEST(AftpEnvelopeTest, SectionCountAndReservedFieldsAreBounded) {
  auto data = Envelope();
  std::fill(data.begin() + 40U, data.begin() + 44U, 255U);
  ExpectRejectedWithoutPublication(data, AftpV1Status::kInvalidSection);
  data = Envelope();
  data[kAftpV1HeaderSize + 4U] = 1U;
  ExpectRejectedWithoutPublication(data, AftpV1Status::kInvalidSection);
}

TEST(AftpEnvelopeTest, WrongPlanKindDoesNotOverwriteLegacyOutput) {
  const auto data = Envelope();
  AftpV1Header header{};
  header.program_id = kSentinelId;
  auto payload = kSentinelBytes;
  EXPECT_EQ(DecodeAftpPlanPayloadV1(data.data(), data.size(), &header, &payload), AftpV1Status::kInvalidSection);
  EXPECT_EQ(header.program_id, kSentinelId);
  EXPECT_EQ(payload, kSentinelBytes);
}

TEST(AftpEnvelopeTest, EncoderMayReuseItsInputBuffer) {
  std::vector<uint8_t> data{1U, 3U, 5U};
  AftpV1EncodeOptions options;
  options.schema_hash = 11U;
  options.program_id = 123U;
  options.producer_build_id = 456U;
  ASSERT_EQ(EncodeAftpPlanPayloadV1(data.data(), data.size(), options, &data), AftpV1Status::kOk);
  EXPECT_EQ(data, Envelope(false));
  const auto saved = data;
  EXPECT_EQ(EncodeAftpPlanPayloadV1(data.data(), std::numeric_limits<size_t>::max(), options, &data),
            AftpV1Status::kInvalidSize);
  EXPECT_EQ(data, saved);
}

TEST(AftpEnvelopeTest, DeterministicMutationsKeepPeekAndDecodeConsistent) {
  const auto valid = Envelope();
  std::mt19937 random(20261102U);
  for (size_t i = 0U; i < 5000U; ++i) {
    auto data = valid;
    data[random() % data.size()] ^= static_cast<uint8_t>(1U << (random() % 8U));
    AftpV1Header header{};
    AftpV1ProgramPayload payload;
    const auto status = DecodeAftpProgramWithManifestV1(data.data(), data.size(), &header, &payload);
    if (status != AftpV1Status::kOk) {
      ExpectRejectedWithoutPublication(data, status);
    } else {
      EXPECT_EQ(PeekAftpV1Header(data.data(), data.size(), &header), AftpV1Status::kOk);
      EXPECT_EQ(payload.plan_payload, (std::vector<uint8_t>{1U, 3U, 5U}));
      EXPECT_EQ(payload.manifest_payload, (std::vector<uint8_t>{7U, 9U}));
    }
  }
}
// Golden bytes were exported by the unmodified reference codec at a0484ee948b2.
TEST(AftpEnvelopeTest, KeepsReferenceWireBytes) {
  const std::vector<uint8_t> plan{
      0x41U, 0x46U, 0x54U, 0x50U, 0x01U, 0x00U, 0x00U, 0x00U, 0xc8U, 0x01U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x01U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x01U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x7bU, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x01U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x7bU, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x2eU, 0x68U, 0xa7U, 0x35U, 0x4fU, 0x5fU, 0x97U, 0xa0U,
      0x0bU, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x01U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x78U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x03U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x2eU, 0x68U, 0xa7U, 0x35U, 0x4fU, 0x5fU, 0x97U, 0xa0U, 0x01U, 0x03U, 0x05U};
  EXPECT_EQ(Envelope(false), plan);
  const std::vector<uint8_t> manifest{
      0x41U, 0x46U, 0x54U, 0x50U, 0x01U, 0x00U, 0x00U, 0x00U, 0xc8U, 0x01U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x01U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x01U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x9dU, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x02U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x7bU, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x2eU, 0x68U, 0xa7U, 0x35U, 0x4fU, 0x5fU, 0x97U, 0xa0U,
      0x0bU, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x02U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x98U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x03U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x2eU, 0x68U, 0xa7U, 0x35U, 0x4fU, 0x5fU, 0x97U, 0xa0U, 0x03U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x9bU, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x02U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
      0x3fU, 0x90U, 0x4bU, 0xc5U, 0x00U, 0x72U, 0x6cU, 0x9aU, 0x01U, 0x03U, 0x05U, 0x07U, 0x09U};
  EXPECT_EQ(Envelope(true), manifest);
}
}  // namespace att
