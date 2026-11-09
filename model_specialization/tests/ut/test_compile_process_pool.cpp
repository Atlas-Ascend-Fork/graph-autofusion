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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include <gtest/gtest.h>

namespace model_spec::compile {
namespace {
namespace fs = std::filesystem;

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
      job.argumentBytes = {4};
      job.outputElfPath = (fs::path(directory_) / (std::to_string(job.jobId) + ".elf")).string();
      jobs.push_back(std::move(job));
    }
    return jobs;
  }

  PoolConfig config_;
  std::string directory_;
};

TEST_F(CompileProcessPoolTest, ReusesWorkersAndReturnsResultsInOriginalOrder) {
  auto jobs = Jobs({"delay", "good", "good", "good", "good"});
  CompileProcessPool pool(config_);
  const auto result = pool.Run(jobs);
  ASSERT_EQ(result.disposition, PoolDisposition::Finished);
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

TEST_F(CompileProcessPoolTest, MissingWorkerAllowsSerialOnlyBeforeDispatch) {
  config_.workerPath = directory_ + "/missing_worker";
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good"}));
  EXPECT_EQ(result.disposition, PoolDisposition::RetrySerialBeforeDispatch);
  EXPECT_FALSE(result.preserveOutputDirectory);
}

TEST_F(CompileProcessPoolTest, WrongReadyIdentityFailsBeforeDispatch) {
  ASSERT_EQ(setenv("MODEL_SPEC_FAULT_MODE", "wrong-ready", 1), 0);
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good"}));
  EXPECT_EQ(result.disposition, PoolDisposition::RetrySerialBeforeDispatch);
  EXPECT_FALSE(fs::exists(directory_ + "/0.elf"));
}

TEST_F(CompileProcessPoolTest, StartupDeadlineCleansWorkersBeforeSerialFallback) {
  ASSERT_EQ(setenv("MODEL_SPEC_FAULT_MODE", "silent-startup", 1), 0);
  config_.startupTimeout = std::chrono::milliseconds(50);
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good"}));
  EXPECT_EQ(result.disposition, PoolDisposition::RetrySerialBeforeDispatch);
  EXPECT_FALSE(result.preserveOutputDirectory);
}

TEST_F(CompileProcessPoolTest, CrashDoesNotRetryAndHealthyWorkerDrainsQueue) {
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"crash", "delay", "good", "good"}));
  EXPECT_EQ(result.disposition, PoolDisposition::Finished);
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

TEST_F(CompileProcessPoolTest, PreviousProducedArtifactSurvivesLaterCrash) {
  config_.workerCount = 1;
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good", "crash", "good"}));
  ASSERT_EQ(result.jobs.size(), 3U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::Success);
  EXPECT_EQ(result.jobs[1].outcome, CompileOutcome::ExecutionFailed);
  EXPECT_EQ(result.jobs[2].outcome, CompileOutcome::Skipped);
}

TEST_F(CompileProcessPoolTest, CorruptArtifactOnlyFailsItsOwnRequest) {
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"corrupt", "good"}));
  ASSERT_EQ(result.jobs.size(), 2U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::ArtifactFailed);
  EXPECT_EQ(result.jobs[1].outcome, CompileOutcome::Success);
}

TEST_F(CompileProcessPoolTest, WrongResultIdDisablesWorker) {
  config_.workerCount = 1;
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"wrong-id", "good"}));
  ASSERT_EQ(result.jobs.size(), 2U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::ExecutionFailed);
  EXPECT_EQ(result.jobs[1].outcome, CompileOutcome::Skipped);
}

TEST_F(CompileProcessPoolTest, ShutdownDeadlineEscalatesWithoutDiscardingProducedOutput) {
  ASSERT_EQ(setenv("MODEL_SPEC_FAULT_MODE", "shutdown-hang", 1), 0);
  config_.workerCount = 1;
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"good"}));
  ASSERT_EQ(result.jobs.size(), 1U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::Success);
  EXPECT_FALSE(result.preserveOutputDirectory);
}

TEST_F(CompileProcessPoolTest, ContradictoryResultCodeDisablesWorker) {
  CompileProcessPool pool(config_);
  const auto result = pool.Run(Jobs({"nonzero-success", "zero-failure"}));
  ASSERT_EQ(result.jobs.size(), 2U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::ExecutionFailed);
  EXPECT_EQ(result.jobs[1].outcome, CompileOutcome::ExecutionFailed);
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
  jobs[0].argumentBytes[0] = jobs[0].arguments[0].size();
  CompileProcessPool pool(config_);
  const auto result = pool.Run(jobs);
  EXPECT_EQ(result.disposition, PoolDisposition::Finished);
  ASSERT_EQ(result.jobs.size(), 1U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::TimedOut);
  EXPECT_FALSE(result.preserveOutputDirectory);
}

TEST_F(CompileProcessPoolTest, UnconfirmedDescendantCleanupRevokesProducedArtifactAndPreservesDirectory) {
  int wasSubreaper = 0;
  ASSERT_EQ(prctl(PR_GET_CHILD_SUBREAPER, &wasSubreaper), 0);
  ASSERT_EQ(prctl(PR_SET_CHILD_SUBREAPER, 1), 0);
  config_.workerCount = 1;
  auto jobs = Jobs({"descendant"});
  CompileProcessPool pool(config_);
  const auto result = pool.Run(jobs);
  EXPECT_EQ(result.disposition, PoolDisposition::Finished);
  EXPECT_TRUE(result.preserveOutputDirectory);
  ASSERT_EQ(result.jobs.size(), 1U);
  EXPECT_EQ(result.jobs[0].outcome, CompileOutcome::ArtifactFailed);
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
  EXPECT_EQ(pool.Run({}).disposition, PoolDisposition::Finished);
  EXPECT_THROW(pool.Run({}), std::logic_error);
}

TEST_F(CompileProcessPoolTest, EmptyBatchDoesNotRequireWorker) {
  config_.workerPath = "/does/not/exist";
  CompileProcessPool pool(config_);
  const auto result = pool.Run({});
  EXPECT_EQ(result.disposition, PoolDisposition::Finished);
  EXPECT_TRUE(result.jobs.empty());
}

}  // namespace
}  // namespace model_spec::compile
