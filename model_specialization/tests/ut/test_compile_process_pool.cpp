/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compile_process_pool.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <boost/filesystem.hpp>
#include <fstream>
#include <poll.h>
#include <set>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include <gtest/gtest.h>

namespace {
enum class PollFailurePoint { None, Startup, AfterFirstResult };
thread_local PollFailurePoint pollFailurePoint = PollFailurePoint::None;
thread_local bool jobSent = false;
thread_local bool firstResultReadable = false;
thread_local bool pollFailureInjected = false;
}  // namespace

extern "C" int __real_poll(pollfd *fds, nfds_t count, int timeout);

extern "C" int __wrap_poll(pollfd *fds, nfds_t count, int timeout) {
  const bool canWrite =
      count != 0 && std::any_of(fds, fds + count, [](const pollfd &fd) { return (fd.events & POLLOUT) != 0; });
  if (pollFailurePoint == PollFailurePoint::Startup ||
      (pollFailurePoint == PollFailurePoint::AfterFirstResult && firstResultReadable && canWrite)) {
    pollFailureInjected = true;
    errno = EINVAL;
    return -1;
  }
  const int result = __real_poll(fds, count, timeout);
  if (pollFailurePoint == PollFailurePoint::AfterFirstResult && result > 0) {
    if (jobSent && count != 0 &&
        std::any_of(fds, fds + count, [](const pollfd &fd) { return (fd.revents & POLLIN) != 0; })) {
      firstResultReadable = true;
    }
    jobSent = jobSent || canWrite;
  }
  return result;
}

namespace model_spec::compile {
namespace {
namespace fs = boost::filesystem;

class CompileProcessPoolTest : public testing::Test {
 protected:
  void SetUp() override {
    const char *root = std::getenv("TMPDIR");
    if (root == nullptr || root[0] == '\0') {
      root = MODEL_SPEC_TEST_TMP_DIR;
    }
    fs::create_directories(root);
    std::string pattern = (fs::path(root) / "compile_pool_ut_XXXXXX").string();
    ASSERT_NE(mkdtemp(pattern.data()), nullptr);
    directory_ = pattern;
    config_.outputDirectory = directory_;
    config_.workerPath = MODEL_SPEC_FAULT_WORKER_PATH;
    config_.workerCount = 2;
    config_.startupTimeout = std::chrono::milliseconds(1000);
    config_.taskTimeout = std::chrono::milliseconds(1000);
    config_.shutdownTimeout = std::chrono::milliseconds(50);
    config_.termGrace = std::chrono::milliseconds(20);
    config_.groupCheckTimeout = std::chrono::milliseconds(100);
    config_.reapInterval = std::chrono::milliseconds(5);
  }

  void TearDown() override {
    pollFailurePoint = PollFailurePoint::None;
    jobSent = false;
    firstResultReadable = false;
    pollFailureInjected = false;
    unsetenv("MODEL_SPEC_FAULT_MODE");
    fs::remove_all(directory_);
  }

  std::vector<CompileJob> Jobs(std::initializer_list<std::string> entries) {
    std::vector<CompileJob> jobs;
    for (const auto &entry : entries) {
      CompileJob job;
      job.jobId = jobs.size();
      job.resourceId = "resource";
      job.kernelEntry = entry;
      job.arguments = {{1, 2, 3, 4}};
      job.outputElfPath = (fs::path(directory_) / (std::to_string(job.jobId) + ".elf")).string();
      jobs.push_back(std::move(job));
    }
    return jobs;
  }

  void ExpectPollFailedBatch(const PoolRunResult &result, size_t count) {
    EXPECT_TRUE(pollFailureInjected);
    ASSERT_EQ(result.jobs.size(), count);
    for (size_t index = 0; index < count; ++index) {
      EXPECT_EQ(result.jobs[index].jobId, index);
      EXPECT_EQ(result.jobs[index].outcome, CompileOutcome::ExecutionFailed);
      EXPECT_FALSE(result.jobs[index].aclrtcCode.has_value());
      EXPECT_NE(result.jobs[index].detail.find("parent poll failed"), std::string::npos);
    }
    EXPECT_FALSE(result.preserveOutputDirectory);
    errno = 0;
    EXPECT_EQ(waitpid(-1, nullptr, WNOHANG), -1);
    EXPECT_EQ(errno, ECHILD);
  }

  PoolConfig config_;
  std::string directory_;
};

TEST_F(CompileProcessPoolTest, ReusesWorkersAndReturnsResultsInOriginalOrder) {
  auto jobs = Jobs({"delay", "good", "good", "good", "good"});
  CompileProcessPool pool(config_);
  const auto result = pool.Run(jobs);
  ASSERT_EQ(result.jobs.size(), jobs.size());
  EXPECT_FALSE(result.preserveOutputDirectory);
  std::set<std::string> pids;
  for (size_t i = 0; i < jobs.size(); ++i) {
    EXPECT_EQ(result.jobs[i].jobId, i);
    EXPECT_EQ(result.jobs[i].outcome, CompileOutcome::Success) << result.jobs[i].detail;
    std::ifstream file(jobs[i].outputElfPath);
    std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    ASSERT_GE(bytes.size(), 4U);
    pids.insert(bytes.substr(4));
  }
  EXPECT_EQ(pids.size(), 2U);
  const auto cleanup = pool.CloseAndReap();
  EXPECT_FALSE(cleanup.preserveOutputDirectory);
  EXPECT_FALSE(pool.CloseAndReap().preserveOutputDirectory);
}

TEST_F(CompileProcessPoolTest, OrdinaryCompileFailureKeepsWorkerReusable) {
  config_.workerCount = 1;
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"fail", "good"}));
  ASSERT_EQ(result.jobs.size(), 2U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::CompileFailed);
  EXPECT_EQ(result.jobs[0].aclrtcCode, 37);
  EXPECT_EQ(result.jobs[1].outcome, CompileOutcome::Success);
}

TEST_F(CompileProcessPoolTest, MissingWorkerSkipsBatchBeforeDispatch) {
  config_.workerPath = directory_ + "/missing_worker";
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good"}));
  ASSERT_EQ(result.jobs.size(), 1U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::Skipped);
  EXPECT_FALSE(result.preserveOutputDirectory);
}

TEST_F(CompileProcessPoolTest, WrongReadyIdentityFailsBeforeDispatch) {
  ASSERT_EQ(setenv("MODEL_SPEC_FAULT_MODE", "wrong-ready", 1), 0);
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good"}));
  ASSERT_EQ(result.jobs.size(), 1U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::Skipped);
  EXPECT_FALSE(fs::exists(directory_ + "/0.elf"));
}

TEST_F(CompileProcessPoolTest, StartupDeadlineSkipsBatchAfterCleanup) {
  ASSERT_EQ(setenv("MODEL_SPEC_FAULT_MODE", "silent-startup", 1), 0);
  config_.startupTimeout = std::chrono::milliseconds(50);
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good"}));
  ASSERT_EQ(result.jobs.size(), 1U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::Skipped);
  EXPECT_FALSE(result.preserveOutputDirectory);
}

TEST_F(CompileProcessPoolTest, StartupPollFailureMarksEntireBatchExecutionFailed) {
  config_.workerCount = 1;
  pollFailurePoint = PollFailurePoint::Startup;
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good", "good", "good"}));
  ExpectPollFailedBatch(result, 3);
  EXPECT_FALSE(fs::exists(directory_ + "/0.elf"));
}

TEST_F(CompileProcessPoolTest, PollFailureDiscardsEarlierSuccessAndQueuedJobs) {
  config_.workerCount = 1;
  pollFailurePoint = PollFailurePoint::AfterFirstResult;
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good", "good", "good"}));
  ExpectPollFailedBatch(result, 3);
  EXPECT_TRUE(fs::exists(directory_ + "/0.elf"));
  EXPECT_FALSE(fs::exists(directory_ + "/1.elf"));
}

TEST_F(CompileProcessPoolTest, ShutdownPollFailureDiscardsSuccessfulBatch) {
  config_.workerCount = 1;
  pollFailurePoint = PollFailurePoint::AfterFirstResult;
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good"}));
  ExpectPollFailedBatch(result, 1);
  EXPECT_TRUE(fs::exists(directory_ + "/0.elf"));
}

TEST_F(CompileProcessPoolTest, CrashDoesNotRetryAndHealthyWorkerDrainsQueue) {
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"crash", "delay", "good", "good"}));
  ASSERT_EQ(result.jobs.size(), 4U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::ExecutionFailed);
  for (size_t i = 1; i < result.jobs.size(); ++i) {
    EXPECT_EQ(result.jobs[i].outcome, CompileOutcome::Success) << result.jobs[i].detail;
  }
}

TEST_F(CompileProcessPoolTest, TaskTimeoutDoesNotBlockHealthyWorker) {
  config_.taskTimeout = std::chrono::milliseconds(100);
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"hang", "good", "good"}));
  ASSERT_EQ(result.jobs.size(), 3U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::TimedOut);
  EXPECT_EQ(result.jobs[1].outcome, CompileOutcome::Success);
  EXPECT_EQ(result.jobs[2].outcome, CompileOutcome::Success);
  EXPECT_FALSE(result.preserveOutputDirectory);
}

TEST_F(CompileProcessPoolTest, PreviousSuccessSurvivesLaterCrash) {
  config_.workerCount = 1;
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good", "crash", "good"}));
  ASSERT_EQ(result.jobs.size(), 3U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::Success);
  EXPECT_EQ(result.jobs[1].outcome, CompileOutcome::ExecutionFailed);
  EXPECT_EQ(result.jobs[2].outcome, CompileOutcome::Skipped);
}

TEST_F(CompileProcessPoolTest, SuccessfulResultDoesNotValidateOutputFiles) {
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"corrupt", "missing", "good"}));
  ASSERT_EQ(result.jobs.size(), 3U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::Success);
  EXPECT_EQ(result.jobs[1].outcome, CompileOutcome::Success);
  EXPECT_EQ(result.jobs[2].outcome, CompileOutcome::Success);
}

TEST_F(CompileProcessPoolTest, WrongResultIdDisablesWorker) {
  config_.workerCount = 1;
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"wrong-id", "good"}));
  ASSERT_EQ(result.jobs.size(), 2U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::ExecutionFailed);
  EXPECT_EQ(result.jobs[1].outcome, CompileOutcome::Skipped);
}

TEST_F(CompileProcessPoolTest, FaultyWorkerIsTerminatedOnlyAfterHealthyJobsComplete) {
  config_.taskTimeout = std::chrono::seconds(5);
  auto jobs = Jobs({"wrong-id-live", "await-disabled", "check-deferred", "good"});
  CompileProcessPool pool(config_);
  const auto result = pool.Run(jobs);
  ASSERT_EQ(result.jobs.size(), jobs.size());
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::ExecutionFailed);
  for (size_t index = 1; index < jobs.size(); ++index) {
    EXPECT_EQ(result.jobs[index].outcome, CompileOutcome::Success) << result.jobs[index].detail;
  }
  pid_t faultPid = -1;
  std::ifstream(directory_ + "/deferred-worker.pid") >> faultPid;
  ASSERT_GT(faultPid, 0);
  EXPECT_FALSE(result.preserveOutputDirectory);
  errno = 0;
  EXPECT_EQ(waitpid(faultPid, nullptr, WNOHANG), -1);
  EXPECT_EQ(errno, ECHILD);
  errno = 0;
  EXPECT_EQ(kill(faultPid, 0), -1);
  EXPECT_EQ(errno, ESRCH);
}

TEST_F(CompileProcessPoolTest, ShutdownDeadlineEscalatesWithoutDiscardingSuccess) {
  ASSERT_EQ(setenv("MODEL_SPEC_FAULT_MODE", "shutdown-hang", 1), 0);
  config_.workerCount = 1;
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good"}));
  ASSERT_EQ(result.jobs.size(), 1U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::Success);
  EXPECT_FALSE(result.preserveOutputDirectory);
}

TEST_F(CompileProcessPoolTest, NonzeroReturnCodeFailsOnlyCurrentJob) {
  config_.workerCount = 1;
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"nonzero-success", "good"}));
  ASSERT_EQ(result.jobs.size(), 2U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::CompileFailed);
  EXPECT_EQ(result.jobs[0].aclrtcCode, 37);
  EXPECT_EQ(result.jobs[1].outcome, CompileOutcome::Success);
}

TEST_F(CompileProcessPoolTest, ExecutionFailureIncludesWorkerExitDiagnostic) {
  config_.workerCount = 1;
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"crash"}));
  ASSERT_EQ(result.jobs.size(), 1U);
  EXPECT_NE(result.jobs[0].detail.find("PID="), std::string::npos);
  EXPECT_NE(result.jobs[0].detail.find("exit code 17"), std::string::npos);
}

TEST_F(CompileProcessPoolTest, KillsWorkerThatIgnoresTermAfterGrace) {
  config_.workerCount = 1;
  config_.taskTimeout = std::chrono::milliseconds(50);
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"ignore-term", "good"}));
  ASSERT_EQ(result.jobs.size(), 2U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::TimedOut);
  EXPECT_NE(result.jobs[0].detail.find("signal 9"), std::string::npos);
  EXPECT_EQ(result.jobs[1].outcome, CompileOutcome::Skipped);
  EXPECT_FALSE(result.preserveOutputDirectory);
}

TEST_F(CompileProcessPoolTest, NonReadingWorkerCannotBlockDeadlineDuringPartialSend) {
  ASSERT_EQ(setenv("MODEL_SPEC_FAULT_MODE", "nonreader", 1), 0);
  config_.workerCount = 1;
  config_.taskTimeout = std::chrono::milliseconds(50);
  auto jobs = Jobs({"good"});
  jobs[0].arguments[0].resize(2U * 1024U * 1024U, 1);
  CompileProcessPool pool(config_);
  const auto result = pool.Run(jobs);
  ASSERT_EQ(result.jobs.size(), 1U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::TimedOut);
  EXPECT_FALSE(result.preserveOutputDirectory);
}

TEST_F(CompileProcessPoolTest, UnconfirmedDescendantCleanupPreservesSuccessAndDirectory) {
  int wasSubreaper = 0;
  ASSERT_EQ(prctl(PR_GET_CHILD_SUBREAPER, &wasSubreaper), 0);
  ASSERT_EQ(prctl(PR_SET_CHILD_SUBREAPER, 1), 0);
  config_.workerCount = 1;
  auto jobs = Jobs({"descendant"});
  CompileProcessPool pool(config_);
  const auto result = pool.Run(jobs);
  EXPECT_TRUE(result.preserveOutputDirectory);
  ASSERT_EQ(result.jobs.size(), 1U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::Success);
  EXPECT_TRUE(pool.CloseAndReap().preserveOutputDirectory);
  pid_t descendant = -1;
  std::ifstream(jobs[0].outputElfPath + ".descendant") >> descendant;
  if (descendant > 0) {
    int status = 0;
    EXPECT_EQ(waitpid(descendant, &status, 0), descendant);
  } else {
    ADD_FAILURE() << "fault worker did not record descendant PID";
  }
  EXPECT_EQ(prctl(PR_SET_CHILD_SUBREAPER, wasSubreaper), 0);
}

TEST_F(CompileProcessPoolTest, PoolCannotRunTwice) {
  CompileProcessPool pool(config_);
  EXPECT_TRUE(pool.Run({}).jobs.empty());
  EXPECT_THROW(pool.Run({}), std::logic_error);
}

TEST_F(CompileProcessPoolTest, EmptyBatchDoesNotRequireWorker) {
  config_.workerPath = "/does/not/exist";
  CompileProcessPool pool(config_);
  const auto result = pool.Run({});
  EXPECT_TRUE(result.jobs.empty());
}

}  // namespace
}  // namespace model_spec::compile
