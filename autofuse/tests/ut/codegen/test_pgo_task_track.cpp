/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/pgo/pgo_stars_parser.h"
#include "codegen/pgo/pgo_capability.h"
#include "codegen/pgo/pgo_task_record.h"
#include "codegen/pgo/pgo_timestamp.h"

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

namespace codegen::pgo {
namespace {

TEST(PgoTimestamp, MatchesMsptiIntegerConversion) {
  PgoDeviceTimeInfo info;
  info.frequency = 50;
  info.start_sys_count = 1000;
  info.start_real_time_ns = 5000000000ULL;
  uint64_t start = 0;
  uint64_t end = 0;
  ASSERT_EQ(ConvertCounterToRealTime(1050, info, &start), 0);
  ASSERT_EQ(ConvertCounterToRealTime(1100, info, &end), 0);
  EXPECT_EQ(start, 6000000000ULL);
  EXPECT_EQ(end, 7000000000ULL);
  uint64_t duration = 0;
  ASSERT_EQ(ConvertTaskDuration(1050, 1100, info, &duration), 0);
  EXPECT_EQ(duration, 1000000000ULL);
}

TEST(PgoTimestamp, UsesRawCounterWhenFrequencyUnavailable) {
  PgoDeviceTimeInfo info;
  uint64_t converted = 0;
  ASSERT_EQ(ConvertCounterToRealTime(1250, info, &converted), 0);
  EXPECT_EQ(converted, 1250ULL);
  uint64_t duration = 0;
  ASSERT_EQ(ConvertTaskDuration(1250, 1300, info, &duration), 0);
  EXPECT_EQ(duration, 50ULL);
}

TEST(PgoTimestamp, PreservesSingleCounterWrapWithSignedDelta) {
  PgoDeviceTimeInfo info;
  info.frequency = 1;
  info.start_sys_count = UINT64_MAX - 5U;
  info.start_real_time_ns = 1000U;
  uint64_t duration = 0;
  ASSERT_EQ(ConvertTaskDuration(UINT64_MAX - 2U, 2U, info, &duration), 0);
  EXPECT_EQ(duration, 5ULL * 1000000000ULL);
}

TEST(PgoTimestamp, RejectsMultipleOrBackwardCounterRange) {
  PgoDeviceTimeInfo info;
  info.frequency = 1;
  uint64_t duration = 0;
  EXPECT_NE(ConvertTaskDuration(100U, 50U, info, &duration), 0);
}

TEST(PgoTimestamp, TruncatesFractionalTicksTowardZero) {
  PgoDeviceTimeInfo info;
  info.frequency = 3;
  info.start_sys_count = 10;
  info.start_real_time_ns = 2000000000ULL;
  uint64_t converted = 0;
  ASSERT_EQ(ConvertCounterToRealTime(14, info, &converted), 0);
  EXPECT_EQ(converted, 3333333333ULL);
  ASSERT_EQ(ConvertCounterToRealTime(6, info, &converted), 0);
  EXPECT_EQ(converted, 666666667ULL);
  ASSERT_EQ(ConvertCounterToRealTime(11, info, &converted), 0);
  EXPECT_EQ(converted, 2333333333ULL);
  ASSERT_EQ(ConvertCounterToRealTime(9, info, &converted), 0);
  EXPECT_EQ(converted, 1666666667ULL);
}

TEST(PgoTimestamp, ConvertsFullSignedCounterRange) {
  PgoDeviceTimeInfo info;
  info.frequency = UINT64_MAX;
  info.start_sys_count = uint64_t{1} << 63;
  info.start_real_time_ns = 1000000000ULL;
  uint64_t converted = 0;
  ASSERT_EQ(ConvertCounterToRealTime(INT64_MAX, info, &converted), 0);
  EXPECT_EQ(converted, 2000000000ULL);
  info.start_sys_count = INT64_MAX;
  ASSERT_EQ(ConvertCounterToRealTime(uint64_t{1} << 63, info, &converted), 0);
  EXPECT_EQ(converted, 0ULL);
}

TEST(PgoTimestamp, RejectsOutOfRangeRealTime) {
  PgoDeviceTimeInfo info;
  info.frequency = 1;
  info.start_sys_count = 1;
  uint64_t converted = 0;
  EXPECT_NE(ConvertCounterToRealTime(0, info, &converted), 0);
  info.start_real_time_ns = UINT64_MAX;
  EXPECT_NE(ConvertCounterToRealTime(2, info, &converted), 0);
}

TEST(PgoTimestamp, RejectsNullOutput) {
  PgoDeviceTimeInfo info;
  EXPECT_NE(ConvertCounterToRealTime(0, info, nullptr), 0);
  EXPECT_NE(ConvertTaskDuration(0, 1, info, nullptr), 0);
}

TEST(PgoTimestamp, RejectsBackwardRawCounterRange) {
  PgoDeviceTimeInfo info;
  uint64_t duration = 0;
  EXPECT_NE(ConvertTaskDuration(2, 1, info, &duration), 0);
}

TEST(PgoStarsParser, ParsesStarsBeginEnd) {
  struct RawStars {
    uint16_t func_type : 6;
    uint16_t count : 4;
    uint16_t task_type : 6;
    uint16_t reserved0;
    uint16_t stream_id;
    uint16_t task_id;
    uint64_t timestamp;
    uint16_t reserved1;
    uint16_t acc_id : 6;
    uint16_t acsq_id : 10;
    uint32_t reserved2[11];
  } raw{};
  raw.func_type = 0;
  raw.task_type = 0;
  raw.stream_id = 7;
  raw.task_id = 11;
  raw.timestamp = 1234;
  PgoStarsEvent event{};
  ASSERT_TRUE(ParseStarsRecord(&raw, sizeof(raw), 2, &event));
  EXPECT_EQ(event.device_id, 2U);
  EXPECT_EQ(event.stream_id, 7U);
  EXPECT_EQ(event.task_id, 11U);
  EXPECT_EQ(event.timestamp, 1234U);
}

TEST(PgoStarsParser, ResolvesExpandedStreamAndTaskIdentity) {
  uint32_t stream_id = 0;
  uint32_t task_id = 0;
  ASSERT_TRUE(ResolveStarsTaskIdentity(0x8001U, 0x8002U, true, false, &stream_id, &task_id));
  EXPECT_EQ(stream_id, 2U);
  EXPECT_EQ(task_id, 0x8001U);
}

TEST(PgoStarsParser, ResolvesLegacyStreamAndTaskIdentity) {
  uint32_t stream_id = 0;
  uint32_t task_id = 0;
  ASSERT_TRUE(ResolveStarsTaskIdentity(0x9007U, 0x1234U, false, false, &stream_id, &task_id));
  EXPECT_EQ(stream_id, 0x7U);
  EXPECT_EQ(task_id, 0x9234U);
  ASSERT_TRUE(ResolveStarsTaskIdentity(0x2007U, 0x3456U, false, false, &stream_id, &task_id));
  EXPECT_EQ(stream_id, 0x456U);
  EXPECT_EQ(task_id, 0x3007U);
}

TEST(PgoStarsParser, RejectsUnknownChipFormatInsteadOfGuessingRecordSize) {
  EXPECT_EQ(ResolvePgoPlatform(0, static_cast<int64_t>(5U << 8U)), PgoPlatformType::kAscend910B);
  EXPECT_EQ(ResolvePgoPlatform(0, static_cast<int64_t>(7U << 8U)), PgoPlatformType::kAscend310B);
  EXPECT_EQ(ResolvePgoPlatform(0, static_cast<int64_t>(15U << 8U)), PgoPlatformType::kAscendV6);
  EXPECT_EQ(ResolvePgoPlatform(0, static_cast<int64_t>(99U << 8U)), PgoPlatformType::kUnknown);
  EXPECT_EQ(ResolvePgoPlatform(-1, 0), PgoPlatformType::kUnknown);
  EXPECT_EQ(ResolveStarsFormat(PgoPlatformType::kAscend910B), PgoStarsFormat::kLegacy64);
  EXPECT_EQ(ResolveStarsFormat(PgoPlatformType::kAscend310B), PgoStarsFormat::kLegacy64);
  EXPECT_EQ(ResolveStarsFormat(PgoPlatformType::kAscendV6), PgoStarsFormat::kV6_32);
  EXPECT_EQ(ResolveStarsFormat(PgoPlatformType::kUnknown), PgoStarsFormat::kUnsupported);
  EXPECT_EQ(StarsRecordSize(PgoStarsFormat::kLegacy64), 64U);
  EXPECT_EQ(StarsRecordSize(PgoStarsFormat::kV6_32), 32U);
  EXPECT_EQ(StarsRecordSize(PgoStarsFormat::kUnsupported), 0U);
}

TEST(PgoTaskRecord, PairsStableDeviceStreamTaskAndLaunch) {
  PgoDeviceTimeInfo info;
  info.frequency = 1;
  std::vector<PgoStarsEvent> events;
  for (size_t i = 0U; i < kPgoMeasureSamples; ++i) {
    const uint64_t start = 10U + i * 10U;
    events.push_back({PgoStarsFuncType::kBegin, 0, 2, 3, 9, start});
    events.push_back({PgoStarsFuncType::kEnd, 0, 2, 3, 9, start + 2U + i});
  }
  std::vector<PgoTaskRecord> records;
  std::vector<uint64_t> launches(kPgoMeasureSamples);
  std::iota(launches.begin(), launches.end(), 7U);
  ASSERT_EQ(PairPgoTaskEvents(events, launches, "candidate", info, &records), 0);
  ASSERT_EQ(records.size(), kPgoMeasureSamples);
  EXPECT_EQ(records[0].launch_sequence, 7U);
  EXPECT_EQ(records[1].launch_sequence, 8U);
  EXPECT_EQ(records[0].duration_ns, 2000000000ULL);
  EXPECT_EQ(records[1].duration_ns, 3000000000ULL);
}

TEST(PgoTaskRecord, PairsBatchEventsByCandidateOrder) {
  PgoDeviceTimeInfo info;
  info.frequency = 1;
  std::vector<PgoStarsEvent> events;
  for (size_t candidate = 0U; candidate < 2U; ++candidate) {
    for (size_t sample = 0U; sample < kPgoMeasureSamples; ++sample) {
      const uint64_t start = 10U + (candidate * kPgoMeasureSamples + sample) * 10U;
      events.push_back({PgoStarsFuncType::kBegin, 0, 2, 3, static_cast<uint32_t>(9U + candidate), start});
      events.push_back({PgoStarsFuncType::kEnd, 0, 2, 3, static_cast<uint32_t>(9U + candidate), start + 2U + sample});
    }
  }
  std::vector<PgoBatchCandidate> candidates(2);
  candidates[0].candidate_key = "candidate_0";
  candidates[1].candidate_key = "candidate_1";
  candidates[0].launch_sequences.resize(kPgoMeasureSamples);
  candidates[1].launch_sequences.resize(kPgoMeasureSamples);
  std::iota(candidates[0].launch_sequences.begin(), candidates[0].launch_sequences.end(), 0U);
  std::iota(candidates[1].launch_sequences.begin(), candidates[1].launch_sequences.end(), 100U);

  std::vector<std::vector<PgoTaskRecord>> records;
  ASSERT_EQ(PairPgoTaskEventsBatch(events, candidates, info, &records), 0);
  ASSERT_EQ(records.size(), 2U);
  ASSERT_EQ(records[0].size(), kPgoMeasureSamples);
  ASSERT_EQ(records[1].size(), kPgoMeasureSamples);
  EXPECT_EQ(records[0].front().candidate_key, "candidate_0");
  EXPECT_EQ(records[1].front().candidate_key, "candidate_1");
  EXPECT_EQ(records[0].front().launch_sequence, 0U);
  EXPECT_EQ(records[1].front().launch_sequence, 100U);

  uint64_t duration = 0U;
  ASSERT_EQ(CalculatePgoDurationNs(records[0], &duration), 0);
  EXPECT_EQ(duration, 18000000000ULL);
}

TEST(PgoTaskRecord, RejectsBatchEventsAcrossStreams) {
  PgoDeviceTimeInfo info;
  info.frequency = 1;
  std::vector<PgoStarsEvent> events;
  for (size_t sample = 0U; sample < kPgoMeasureSamples; ++sample) {
    const uint64_t start = 10U + sample * 10U;
    const uint32_t stream = sample % 2U == 0U ? 2U : 4U;
    events.push_back({PgoStarsFuncType::kBegin, 0, 0, stream, 3, start});
    events.push_back({PgoStarsFuncType::kEnd, 0, 0, stream, 3, start + 2U});
  }
  std::vector<PgoBatchCandidate> candidates(1);
  candidates[0].candidate_key = "candidate";
  candidates[0].launch_sequences.resize(kPgoMeasureSamples);
  std::iota(candidates[0].launch_sequences.begin(), candidates[0].launch_sequences.end(), 0U);
  std::vector<std::vector<PgoTaskRecord>> records;
  EXPECT_NE(PairPgoTaskEventsBatch(events, candidates, info, &records), 0);
}

TEST(PgoTaskRecord, RejectsMissingAndDuplicateEndEvents) {
  PgoDeviceTimeInfo info;
  info.frequency = 1;
  std::vector<PgoTaskRecord> records;
  const PgoStarsEvent begin{PgoStarsFuncType::kBegin, 0, 0, 1, 1, 1};
  const PgoStarsEvent end{PgoStarsFuncType::kEnd, 0, 0, 1, 1, 2};
  std::vector<uint64_t> launches(kPgoMeasureSamples);
  std::iota(launches.begin(), launches.end(), 1U);
  EXPECT_NE(PairPgoTaskEvents({end}, launches, "candidate", info, &records), 0);
  std::vector<PgoStarsEvent> valid_events;
  for (size_t i = 0U; i < kPgoMeasureSamples; ++i) {
    valid_events.push_back(begin);
    valid_events.push_back({PgoStarsFuncType::kEnd, 0, 0, 1, 1, static_cast<uint64_t>(2U + i)});
  }
  EXPECT_EQ(PairPgoTaskEvents(valid_events, launches, "candidate", info, &records), 0);
  EXPECT_EQ(records.size(), kPgoMeasureSamples);
  std::vector<PgoStarsEvent> duplicate_end = valid_events;
  duplicate_end.insert(duplicate_end.begin() + 2U, end);
  EXPECT_NE(PairPgoTaskEvents(duplicate_end, launches, "candidate", info, &records), 0);
  EXPECT_NE(PairPgoTaskEvents({begin}, launches, "candidate", info, &records), 0);
}

TEST(PgoTaskRecord, RejectsMoreTaskPairsThanLaunches) {
  PgoDeviceTimeInfo info;
  info.frequency = 1;
  const PgoStarsEvent begin{PgoStarsFuncType::kBegin, 0, 0, 1, 1, 1};
  const PgoStarsEvent end{PgoStarsFuncType::kEnd, 0, 0, 1, 1, 2};
  std::vector<PgoTaskRecord> records;
  std::vector<PgoStarsEvent> events;
  for (size_t i = 0U; i < kPgoMeasureSamples + 1U; ++i) {
    events.push_back(begin);
    events.push_back(end);
  }
  std::vector<uint64_t> launches(kPgoMeasureSamples);
  std::iota(launches.begin(), launches.end(), 1U);
  EXPECT_NE(PairPgoTaskEvents(events, launches, "candidate", info, &records), 0);
}

TEST(PgoTaskRecord, SerialFallbackRequiresSingleStream) {
  const PgoStarsEvent first{PgoStarsFuncType::kBegin, 6, 0, 7, 1, 10};
  const PgoStarsEvent second{PgoStarsFuncType::kEnd, 6, 0, 7, 1, 20};
  EXPECT_TRUE(CanUseSerialPgoEvents({first, second}));
  const PgoStarsEvent other_stream{PgoStarsFuncType::kBegin, 6, 0, 8, 2, 30};
  EXPECT_FALSE(CanUseSerialPgoEvents({first, other_stream}));
  EXPECT_FALSE(CanUseSerialPgoEvents({}));
}

TEST(PgoStarsParser, AcceptsKernelTaskTypesWithoutFixedWhitelist) {
  struct RawStars {
    uint16_t func_type : 6;
    uint16_t count : 4;
    uint16_t task_type : 6;
    uint16_t reserved0;
    uint16_t stream_id;
    uint16_t task_id;
    uint64_t timestamp;
    uint16_t reserved1;
    uint16_t acc_id : 6;
    uint16_t acsq_id : 10;
    uint32_t reserved2[11];
  } raw{};
  raw.func_type = 0;
  PgoStarsEvent event{};
  raw.task_type = 6;
  EXPECT_TRUE(ParseStarsRecord(&raw, sizeof(raw), 0, &event));
  raw.task_type = 63;
  EXPECT_TRUE(ParseStarsRecord(&raw, sizeof(raw), 0, &event));
}

}  // namespace
}  // namespace codegen::pgo
