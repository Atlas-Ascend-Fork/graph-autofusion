/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compile_artifact.h"
#include "compile_protocol.h"

#include <gtest/gtest.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

extern char **environ;

namespace model_spec::compile {
namespace {

class CompileWorkerTest : public testing::Test {
 protected:
  void SetUp() override {
    const char *tempRoot = std::getenv("TMPDIR");
    if (tempRoot == nullptr || tempRoot[0] == '\0') {
      tempRoot = MODEL_SPEC_TEST_TMP_DIR;
    }
    std::string pattern = std::string(tempRoot) + "/compile_worker_XXXXXX";
    ASSERT_NE(mkdtemp(pattern.data()), nullptr);
    directory_ = pattern;
    int sockets[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets), 0);
    posix_spawn_file_actions_t actions;
    ASSERT_EQ(posix_spawn_file_actions_init(&actions), 0);
    ASSERT_EQ(posix_spawn_file_actions_addclose(&actions, sockets[0]), 0);
    ASSERT_EQ(posix_spawn_file_actions_adddup2(&actions, sockets[1], 3), 0);
    if (sockets[1] != 3) {
      ASSERT_EQ(posix_spawn_file_actions_addclose(&actions, sockets[1]), 0);
    }
    char *arguments[] = {const_cast<char *>(MODEL_SPEC_TEST_WORKER_PATH), const_cast<char *>("--control-fd"),
                         const_cast<char *>("3"), nullptr};
    const int result = posix_spawn(&pid_, MODEL_SPEC_TEST_WORKER_PATH, &actions, nullptr, arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(sockets[1]);
    ASSERT_EQ(result, 0);
    channel_ = std::make_unique<FramedChannel>(sockets[0]);
    Message ready;
    ASSERT_TRUE(Receive(ready));
    ASSERT_EQ(ready.kind, MessageKind::READY);
    ASSERT_EQ(ready.jobId, NO_JOB_ID);
    ASSERT_EQ(ready.buildId, WORKER_BUILD_ID);
  }

  void TearDown() override {
    channel_.reset();
    if (pid_ > 0) {
      kill(pid_, SIGKILL);
      while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
    if (!directory_.empty()) {
      std::filesystem::remove_all(directory_);
    }
  }

  bool Send(const Message &message) {
    std::string error;
    if (!channel_->Queue(message, error)) {
      ADD_FAILURE() << error;
      return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (channel_->WantsWrite() && std::chrono::steady_clock::now() < deadline) {
      const IoStatus status = channel_->PumpWrite(error);
      if (status == IoStatus::Error || status == IoStatus::PeerClosed) {
        ADD_FAILURE() << error;
        return false;
      }
      pollfd fd{channel_->Fd(), POLLOUT, 0};
      poll(&fd, 1, 10);
    }
    return !channel_->WantsWrite();
  }

  bool Receive(Message &message) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      std::vector<Message> messages;
      std::string error;
      const IoStatus status = channel_->PumpRead(messages, error);
      if (!messages.empty()) {
        message = std::move(messages.front());
        return true;
      }
      if (status == IoStatus::Error || status == IoStatus::PeerClosed) {
        ADD_FAILURE() << "worker closed before expected message: " << error;
        return false;
      }
      pollfd fd{channel_->Fd(), POLLIN, 0};
      poll(&fd, 1, 10);
    }
    ADD_FAILURE() << "worker message timed out";
    return false;
  }

  Message Job(uint64_t id, const std::string &entry = "specialized_kernel") {
    Message message;
    message.kind = MessageKind::JOB;
    message.jobId = id;
    message.job.jobId = id;
    message.job.resourceId = "resource_中文";
    message.job.kernelEntry = entry;
    message.job.arguments = {{0x00, 0xff, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc}, {}, {0x42}};
    message.job.argumentBytes = {8, 0, 1};
    message.job.basicOptions = {"--optimization=3", "--include=中文路径", ""};
    message.job.skOptions = {"--sk-option=value", "--second"};
    message.job.outputElfPath = directory_ + "/job_" + std::to_string(id) + ".elf";
    return message;
  }

  void Stop() {
    Message shutdown;
    shutdown.kind = MessageKind::SHUTDOWN;
    ASSERT_TRUE(Send(shutdown));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
      if (waitpid(pid_, &status, WNOHANG) == pid_) {
        pid_ = -1;
        ASSERT_TRUE(WIFEXITED(status));
        EXPECT_EQ(WEXITSTATUS(status), 0);
        return;
      }
      poll(nullptr, 0, 10);
    }
    FAIL() << "SHUTDOWN did not exit worker";
  }

  std::unique_ptr<FramedChannel> channel_;
  pid_t pid_ = -1;
  std::string directory_;
};

TEST_F(CompileWorkerTest, RebuildsCompleteRequestAndPublishesElf) {
  Message job = Job(7);
  ASSERT_TRUE(Send(job));
  Message result;
  ASSERT_TRUE(Receive(result));
  EXPECT_EQ(result.kind, MessageKind::RESULT);
  EXPECT_EQ(result.jobId, 7U);
  EXPECT_EQ(result.reply.kind, WorkerResultKind::Produced);
  EXPECT_EQ(result.reply.aclrtcCode, 0);
  ASSERT_TRUE(std::filesystem::exists(job.job.outputElfPath));
  EXPECT_FALSE(std::filesystem::exists(job.job.outputElfPath + ".partial"));
  std::ifstream file(job.job.outputElfPath, std::ios::binary);
  const std::string actual((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_EQ(actual,
            "\177ELF\nresource_中文\nspecialized_kernel\nnonblock=1 cloexec=1\n"
            "arguments=3\n8:00ff123456789abc\n0:\n1:42\noptions=3\n--optimization=3\n"
            "--include=中文路径\n\nskOptions=2\n--sk-option=value\n--second\nreservedZero=1\n");
  ArtifactInfo info;
  std::string error;
  ASSERT_TRUE(InspectArtifact(job.job.outputElfPath, info, error)) << error;
  EXPECT_EQ(result.reply.artifactSize, info.size);
  EXPECT_EQ(result.reply.artifactSha256, info.sha256);
  Stop();
}

TEST_F(CompileWorkerTest, ContinuesAfterCompileAndArtifactFailures) {
  const std::vector<std::string> entries = {"compile_failure", "missing_output", "invalid_output",
                                            "specialized_kernel"};
  const std::vector<WorkerResultKind> kinds = {WorkerResultKind::CompileFailed, WorkerResultKind::ArtifactFailed,
                                               WorkerResultKind::ArtifactFailed, WorkerResultKind::Produced};
  for (size_t i = 0; i < entries.size(); ++i) {
    Message job = Job(i, entries[i]);
    ASSERT_TRUE(Send(job));
    Message result;
    ASSERT_TRUE(Receive(result));
    EXPECT_EQ(result.jobId, i);
    EXPECT_EQ(result.reply.kind, kinds[i]);
    EXPECT_EQ(result.reply.aclrtcCode, i == 0 ? 12345 : 0);
    if (kinds[i] != WorkerResultKind::Produced) {
      EXPECT_FALSE(std::filesystem::exists(job.job.outputElfPath));
      EXPECT_FALSE(result.reply.detail.empty());
    }
  }
  Stop();
}

TEST_F(CompileWorkerTest, ReportsRenameFailureAndContinues) {
  Message job = Job(0);
  ASSERT_TRUE(std::filesystem::create_directory(job.job.outputElfPath));
  ASSERT_TRUE(Send(job));
  Message result;
  ASSERT_TRUE(Receive(result));
  EXPECT_EQ(result.reply.kind, WorkerResultKind::ArtifactFailed);
  EXPECT_EQ(result.reply.aclrtcCode, 0);
  ASSERT_TRUE(Send(Job(1)));
  ASSERT_TRUE(Receive(result));
  EXPECT_EQ(result.reply.kind, WorkerResultKind::Produced);
  Stop();
}

TEST_F(CompileWorkerTest, ExitsWhenParentClosesChannel) {
  channel_.reset();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  int status = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    if (waitpid(pid_, &status, WNOHANG) == pid_) {
      pid_ = -1;
      ASSERT_TRUE(WIFEXITED(status));
      EXPECT_EQ(WEXITSTATUS(status), 0);
      return;
    }
    poll(nullptr, 0, 10);
  }
  FAIL() << "EOF did not exit worker";
}

}  // namespace
}  // namespace model_spec::compile
