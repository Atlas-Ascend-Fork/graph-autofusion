/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;

namespace sk::test {
class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    char pattern[] = "/tmp/sk-aot-process-XXXXXX";
    const char *path = mkdtemp(pattern);
    if (path == nullptr) {
      throw std::runtime_error("cannot create ST temporary directory");
    }
    path_ = path;
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  const std::filesystem::path &Path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

// Each child starts with fresh production singletons and exits normally so
// recorder shutdown and gcov flushing are exercised.
inline int RunProcess(const std::vector<std::string> &arguments, const std::string &extraEnvironment = "") {
  std::vector<char *> argv;
  for (const auto &arg : arguments) {
    argv.push_back(const_cast<char *>(arg.c_str()));
  }
  argv.push_back(nullptr);
  std::vector<char *> environment;
  for (char **entry = environ; *entry != nullptr; ++entry) {
    const std::string value = *entry;
    if (value.rfind("GTEST_TOTAL_SHARDS=", 0) == 0 || value.rfind("GTEST_SHARD_INDEX=", 0) == 0 ||
        value.rfind("GTEST_SHARD_STATUS_FILE=", 0) == 0) {
      continue;
    }
    environment.push_back(*entry);
  }
  if (!extraEnvironment.empty()) {
    environment.push_back(const_cast<char *>(extraEnvironment.c_str()));
  }
  environment.push_back(nullptr);
  pid_t child = -1;
  if (posix_spawn(&child, argv[0], nullptr, nullptr, argv.data(), environment.data()) != 0) {
    return -1;
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  while (std::chrono::steady_clock::now() < deadline) {
    int status = 0;
    const auto result = waitpid(child, &status, WNOHANG);
    if (result == child) {
      return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    if (result == -1 && errno != EINTR) {
      return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  kill(child, SIGKILL);
  int status = 0;
  while (waitpid(child, &status, 0) == -1 && errno == EINTR) {
  }
  return -1;
}
}  // namespace sk::test
