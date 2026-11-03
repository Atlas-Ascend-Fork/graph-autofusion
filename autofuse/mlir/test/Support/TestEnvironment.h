/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_SUPPORT_TEST_ENVIRONMENT_H
#define AUTOFUSE_MLIR_TEST_SUPPORT_TEST_ENVIRONMENT_H

#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

namespace mlir::test {

// Temporarily sets or unsets a process environment variable and restores its previous state on destruction.
// The guard is intentionally non-copyable because copying would make two objects restore the same variable.
class ScopedEnvVar {
 public:
  ScopedEnvVar(std::string name, std::string value) : ScopedEnvVar(std::move(name), std::move(value), true) {}

  static ScopedEnvVar Unset(std::string name) {
    return ScopedEnvVar(std::move(name), "", false);
  }

  ~ScopedEnvVar() {
    if (had_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvVar(const ScopedEnvVar &) = delete;
  ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

 private:
  ScopedEnvVar(std::string name, std::string value, bool set_value)
      : name_(std::move(name)), had_value_(std::getenv(name_.c_str()) != nullptr) {
    if (had_value_) {
      old_value_ = std::getenv(name_.c_str());
    }
    if (set_value) {
      setenv(name_.c_str(), value.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  std::string name_;
  bool had_value_;
  std::string old_value_;
};

// 按 fd 收 stderr。**按 fd 而不是换 `llvm::errs()` 的流**：要收的既有 `llvm::errs()`，也有
// MLIR 诊断引擎默认 handler 打出去的那些，两者只有 fd 2 这一个共同出口。与
// `test/probe/CapabilityBoundaryProbe.cpp` 的 `StderrCapture` 同一写法。
class ScopedStderrCapture {
 public:
  ScopedStderrCapture() {
    saved_ = ::dup(STDERR_FILENO);
    file_ = std::tmpfile();
    if (file_ != nullptr) {
      ::dup2(::fileno(file_), STDERR_FILENO);
    }
  }

  ~ScopedStderrCapture() {
    Restore();
    if (file_ != nullptr) {
      std::fclose(file_);
    }
  }

  ScopedStderrCapture(const ScopedStderrCapture &) = delete;
  ScopedStderrCapture &operator=(const ScopedStderrCapture &) = delete;

  // 收到的全文。**先 `Restore()` 再读**，否则断言失败时 gtest 的输出也进了这个临时文件。
  std::string Text() {
    Restore();
    if (file_ == nullptr) {
      return {};
    }
    std::rewind(file_);
    std::string text;
    std::array<char, 4096> chunk{};
    size_t read = 0;
    while ((read = std::fread(chunk.data(), 1, chunk.size(), file_)) > 0U) {
      text.append(chunk.data(), read);
    }
    return text;
  }

 private:
  void Restore() {
    if (saved_ >= 0) {
      std::fflush(stderr);
      ::dup2(saved_, STDERR_FILENO);
      ::close(saved_);
      saved_ = -1;
    }
  }

  int saved_ = -1;
  std::FILE *file_ = nullptr;
};

}  // namespace mlir::test

#endif  // AUTOFUSE_MLIR_TEST_SUPPORT_TEST_ENVIRONMENT_H
