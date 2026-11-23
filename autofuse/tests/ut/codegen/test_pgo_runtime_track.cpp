/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#include "codegen/pgo/pgo_runtime_track.h"

#include <gtest/gtest.h>

#include <array>

namespace codegen::pgo {
namespace {

struct RawCompactInfo {
  uint16_t magic = 0x5a5a;
  uint16_t level = 5000;
  uint32_t type = 802;
  uint32_t thread_id = 17;
  uint32_t data_len = 40;
  uint64_t timestamp = 123;
  uint16_t device_id = 2;
  uint16_t stream_id = 7;
  uint32_t task_info = 99;
  uint64_t task_type = 0x42;
  uint64_t kernel_name = 0xabcdef;
  std::array<unsigned char, 16> reserved{};
};

TEST(PgoRuntimeTrack, DecodesCannCompactRuntimeTrack) {
  RawCompactInfo raw;
  PgoRuntimeTrack track;
  ASSERT_EQ(DecodeRuntimeTrack(1, &raw, sizeof(raw), &track), PGO_SUCCESS);
  EXPECT_EQ(track.device_id, 2U);
  EXPECT_EQ(track.stream_id, 7U);
  EXPECT_EQ(track.task_info, 99U);
  EXPECT_EQ(track.task_type, 0x42U);
  EXPECT_EQ(track.kernel_name_hash, 0xabcdefU);
  EXPECT_TRUE(track.aging);
}

TEST(PgoRuntimeTrack, RejectsMalformedCompactInfo) {
  std::array<unsigned char, 64> raw{};
  PgoRuntimeTrack track;
  EXPECT_EQ(DecodeRuntimeTrack(0, raw.data(), raw.size(), &track), PGO_DATA_UNAVAILABLE);
  EXPECT_EQ(DecodeRuntimeTrack(0, nullptr, 0, &track), PGO_INVALID_ARGUMENT);
}

TEST(PgoRuntimeTrack, ReporterOwnershipIsSingleSlot) {
  EXPECT_EQ(RegisterRuntimeTrackReporter(PgoReporterOwnership::kUnknown, &QueueRuntimeTrack), PGO_REPORTER_BUSY);
  ASSERT_EQ(RegisterRuntimeTrackReporter(PgoReporterOwnership::kExclusive, &QueueRuntimeTrack), PGO_SUCCESS);
  EXPECT_EQ(RegisterRuntimeTrackReporter(PgoReporterOwnership::kExclusive, &QueueRuntimeTrack), PGO_REPORTER_BUSY);
  EXPECT_EQ(UnregisterRuntimeTrackReporter(), PGO_SUCCESS);
  EXPECT_EQ(UnregisterRuntimeTrackReporter(), PGO_SUCCESS);
}

TEST(PgoRuntimeTrack, TypeInfoIsSessionScopedAndFailsClosed) {
  ASSERT_EQ(RegisterRuntimeTrackReporter(PgoReporterOwnership::kExclusive, &QueueRuntimeTrack), PGO_SUCCESS);
  EXPECT_FALSE(IsKernelTaskType(7U));
  SetRuntimeTypeInfoAvailable(true);
  constexpr char kKernelName[] = "KERNEL_AICORE";
  ASSERT_EQ(DispatchRuntimeTypeInfo(5000U, 7U, kKernelName, sizeof(kKernelName) - 1U), PGO_SUCCESS);
  EXPECT_TRUE(IsKernelTaskType(7U));
  EXPECT_FALSE(IsKernelTaskType(8U));
  ASSERT_EQ(UnregisterRuntimeTrackReporter(), PGO_SUCCESS);
  EXPECT_FALSE(IsKernelTaskType(7U));
}

TEST(PgoRuntimeTrack, ReportsQueueOverflowAndClearsItWithSession) {
  ASSERT_EQ(RegisterRuntimeTrackReporter(PgoReporterOwnership::kExclusive, &QueueRuntimeTrack), PGO_SUCCESS);
  SetRuntimeTrackCaptureEnabled(true);
  PgoRuntimeTrack track{};
  for (size_t index = 0U; index < kRuntimeTrackQueueCapacity; ++index) {
    ASSERT_EQ(QueueRuntimeTrack(track), PGO_SUCCESS);
  }
  EXPECT_EQ(QueueRuntimeTrack(track), PGO_DATA_UNAVAILABLE);
  EXPECT_TRUE(RuntimeTrackOverflowed());
  ClearRuntimeTracks();
  EXPECT_FALSE(RuntimeTrackOverflowed());
  EXPECT_EQ(UnregisterRuntimeTrackReporter(), PGO_SUCCESS);
}

TEST(PgoRuntimeTrack, DecodesStreamExpandControlRecord) {
  ASSERT_EQ(RegisterRuntimeTrackReporter(PgoReporterOwnership::kExclusive, &QueueRuntimeTrack), PGO_SUCCESS);
  RawCompactInfo raw;
  raw.type = 804;
  raw.data_len = 4;
  auto *payload = reinterpret_cast<unsigned char *>(&raw) + 24U;
  payload[0] = 1U;
  ASSERT_EQ(DispatchRuntimeTrack(0, &raw, sizeof(raw)), PGO_SUCCESS);
  EXPECT_TRUE(IsStreamExpandEnabled());
  payload[0] = 0U;
  raw.data_len = 40;
  ASSERT_EQ(DispatchRuntimeTrack(0, &raw, sizeof(raw)), PGO_SUCCESS);
  EXPECT_FALSE(IsStreamExpandEnabled());
  raw.data_len = 3;
  payload[0] = 1U;
  EXPECT_EQ(DispatchRuntimeTrack(0, &raw, sizeof(raw)), PGO_DATA_UNAVAILABLE);
  EXPECT_FALSE(IsStreamExpandEnabled());
  EXPECT_EQ(UnregisterRuntimeTrackReporter(), PGO_SUCCESS);
}

}  // namespace
}  // namespace codegen::pgo
