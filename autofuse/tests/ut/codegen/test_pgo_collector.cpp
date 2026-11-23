/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/pgo/pgo_collector.h"
#include "codegen/pgo/pgo_profapi_adapter.h"
#include "codegen/pgo/pgo_task_record.h"
#include "../../depends/profiling/pgo_profiler_stub.h"

#include <dlfcn.h>
#include <memory>

#include <gtest/gtest.h>

namespace codegen::pgo {
namespace {

TEST(PgoCollector, NullArgumentsReturnFailure) {
  EXPECT_NE(AutofusePgoCollectorCreate(0, nullptr, nullptr), 0);
  EXPECT_NE(AutofusePgoCollectorRecordLaunch(nullptr, 0), 0);
  EXPECT_NE(AutofusePgoCollectorEnd(nullptr), 0);
  uint64_t duration = 0;
  EXPECT_NE(AutofusePgoCollectorGetDurationNs(nullptr, &duration), 0);
  AutofusePgoCollectorDestroy(nullptr);
}

TEST(PgoCollector, CreatePropagatesCapabilityStatus) {
  void *collector = nullptr;
  EXPECT_EQ(AutofusePgoCollectorCreate(0, nullptr, &collector), PGO_CAPABILITY_UNAVAILABLE);
  EXPECT_EQ(collector, nullptr);
}

class PgoCollectorWithStub : public testing::Test {
 protected:
  void SetUp() override {
    library_ = dlopen(PGO_PROFILER_STUB_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(library_, nullptr);
    set_failure_ = reinterpret_cast<SetPgoStubFailureFn>(dlsym(library_, "SetPgoStubFailure"));
    ASSERT_NE(set_failure_, nullptr);
    set_failure_(PgoStubFailure::None);
    SetPgoProfilerStubPath(PGO_PROFILER_STUB_PATH);
  }

  void TearDown() override {
    collector_.reset();
    SetPgoProfilerStubPath(nullptr);
    if (library_ != nullptr) {
      dlclose(library_);
    }
  }

  void Create() {
    void *collector = nullptr;
    ASSERT_EQ(AutofusePgoCollectorCreate(0U, &stream_, &collector), PGO_SUCCESS);
    ASSERT_NE(collector, nullptr);
    collector_.reset(collector);
  }

  void RecordCandidate() {
    ASSERT_EQ(AutofusePgoCollectorBegin(collector_.get(), "candidate"), PGO_SUCCESS);
    for (uint64_t launch = 1; launch <= kPgoMeasureSamples; ++launch) {
      ASSERT_EQ(AutofusePgoCollectorRecordLaunch(collector_.get(), launch), PGO_SUCCESS);
    }
    ASSERT_EQ(AutofusePgoCollectorEnd(collector_.get()), PGO_SUCCESS);
  }

  SetPgoStubFailureFn set_failure_ = nullptr;
  std::unique_ptr<void, decltype(&AutofusePgoCollectorDestroy)> collector_{nullptr, &AutofusePgoCollectorDestroy};

 private:
  void *library_ = nullptr;
  int stream_ = 0;
};

TEST_F(PgoCollectorWithStub, AbortsIncompleteCandidateAndRejectsEmptyBatchData) {
  ASSERT_NO_FATAL_FAILURE(Create());
  ASSERT_EQ(AutofusePgoCollectorBegin(collector_.get(), "candidate"), PGO_SUCCESS);
  EXPECT_EQ(AutofusePgoCollectorEnd(collector_.get()), PGO_DATA_UNAVAILABLE);
  ASSERT_EQ(AutofusePgoCollectorAbort(collector_.get()), PGO_SUCCESS);

  ASSERT_NO_FATAL_FAILURE(RecordCandidate());
  EXPECT_EQ(AutofusePgoCollectorFinalizeBatch(collector_.get()), PGO_DATA_UNAVAILABLE);
  uint64_t duration = 0;
  EXPECT_NE(AutofusePgoCollectorGetDurationNs(collector_.get(), &duration), PGO_SUCCESS);
}

TEST_F(PgoCollectorWithStub, PropagatesStartFailure) {
  ASSERT_NO_FATAL_FAILURE(Create());
  set_failure_(PgoStubFailure::Start);
  EXPECT_EQ(AutofusePgoCollectorBegin(collector_.get(), "candidate"), PGO_PROFILING_START_FAILED);
}

TEST_F(PgoCollectorWithStub, PropagatesCollectionFailures) {
  for (const auto failure : {PgoStubFailure::Stop, PgoStubFailure::Flush, PgoStubFailure::Read}) {
    SCOPED_TRACE(static_cast<int>(failure));
    set_failure_(PgoStubFailure::None);
    ASSERT_NO_FATAL_FAILURE(Create());
    ASSERT_NO_FATAL_FAILURE(RecordCandidate());
    set_failure_(failure);
    const int expected = failure == PgoStubFailure::Stop ? PGO_PROFILING_STOP_FAILED : PGO_DATA_UNAVAILABLE;
    EXPECT_EQ(AutofusePgoCollectorFinalizeBatch(collector_.get()), expected);
    set_failure_(PgoStubFailure::None);
    collector_.reset();
  }
}

TEST(PgoProfApiAdapter, StopStatusPreservesChannelFailure) {
  EXPECT_EQ(CombinePgoStopStatus(PGO_SUCCESS, PGO_SUCCESS), PGO_SUCCESS);
  EXPECT_EQ(CombinePgoStopStatus(PGO_CHANNEL_UNAVAILABLE, PGO_SUCCESS), PGO_CHANNEL_UNAVAILABLE);
  EXPECT_EQ(CombinePgoStopStatus(PGO_SUCCESS, PGO_PROFILING_STOP_FAILED), PGO_PROFILING_STOP_FAILED);
}

TEST(PgoProfApiAdapter, PollPositiveChannelCountIsSuccess) {
  EXPECT_EQ(NormalizePgoPollStatus(0), PGO_SUCCESS);
  EXPECT_EQ(NormalizePgoPollStatus(1), PGO_SUCCESS);
  EXPECT_EQ(NormalizePgoPollStatus(-1), PGO_CHANNEL_UNAVAILABLE);
}

TEST(PgoProfApiAdapter, FlushNotSupportMatchesMsptiNoOp) {
  EXPECT_TRUE(IsPgoFlushUnsupportedStatus(static_cast<int32_t>(0xfffe)));
  EXPECT_FALSE(IsPgoFlushUnsupportedStatus(1));
}

TEST(PgoProfApiAdapter, DlopenModeKeepsProfilerLibrariesResident) {
  const int flags = PgoDlopenFlags();
  EXPECT_EQ(flags & RTLD_NOW, RTLD_NOW);
#if defined(RTLD_LOCAL) && RTLD_LOCAL != 0
  EXPECT_EQ(flags & RTLD_LOCAL, RTLD_LOCAL);
#endif
#if defined(RTLD_NODELETE)
  EXPECT_EQ(flags & RTLD_NODELETE, RTLD_NODELETE);
#endif
}

}  // namespace
}  // namespace codegen::pgo
