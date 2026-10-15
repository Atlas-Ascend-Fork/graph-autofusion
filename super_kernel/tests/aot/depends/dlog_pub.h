/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <string>
#include <vector>
#include "securec.h"
#ifdef __cplusplus
extern "C" {
#endif

// Module ID for logging
typedef enum {
  ASCENDCKERNEL = 0,
  ASCENDSLOG = 1,
} AscendcModule;

// Log level definitions
typedef enum {
  DLOG_DEBUG = 0,
  DLOG_INFO = 1,
  DLOG_WARN = 2,
  DLOG_ERROR = 3,
  DLOG_FATAL = 4,
} DlogLevel;

#ifdef __cplusplus
}
#endif

// ============================================================================
// 日志缓冲区：测试失败时才打印，成功则丢弃
// ============================================================================
#ifdef __cplusplus

#include <sstream>
#include <mutex>

namespace ut_log {

// 日志缓冲区类（每个测试用例独立）
class LogBuffer {
 public:
  __attribute__((visibility("default"))) static LogBuffer &Instance();

  // 添加日志
  void Append(const std::string &level, const char *format, va_list args) {
    std::lock_guard<std::mutex> lock(mutex_);
    char buf[4096];
    vsnprintf_s(buf, sizeof(buf), 4096, format, args);
    buffer_ << "[" << level << "] " << buf << "\n";
  }

  // 打印并清空缓冲区（测试失败时调用）
  void Flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string content = buffer_.str();
    if (!content.empty()) {
      fprintf(stderr, "\n=== UT LOG (test failed) ===\n%s=== END UT LOG ===\n", content.c_str());
      fflush(stderr);
    }
    buffer_.str("");
    buffer_.clear();
  }

  // 清空缓冲区（测试成功时调用）
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.str("");
    buffer_.clear();
  }

  // 获取缓冲区内容
  std::string GetContent() {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.str();
  }

 private:
  LogBuffer() = default;
  std::ostringstream buffer_;
  std::mutex mutex_;
};

}  // namespace ut_log

// C++ 版本的日志函数（写入缓冲区）
static inline void dlog_debug(int32_t moduleId, const char *format, ...) {
  va_list args;
  va_start(args, format);
  ut_log::LogBuffer::Instance().Append("DEBUG", format, args);
  va_end(args);
}

static inline void dlog_info(int32_t moduleId, const char *format, ...) {
  va_list args;
  va_start(args, format);
  ut_log::LogBuffer::Instance().Append("INFO", format, args);
  va_end(args);
}

static inline void dlog_warn(int32_t moduleId, const char *format, ...) {
  va_list args;
  va_start(args, format);
  ut_log::LogBuffer::Instance().Append("WARN", format, args);
  va_end(args);
}

static inline void dlog_error(int32_t moduleId, const char *format, ...) {
  va_list args;
  va_start(args, format);
  ut_log::LogBuffer::Instance().Append("ERROR", format, args);
  va_end(args);
}

static inline void dlog_fatal(int32_t moduleId, const char *format, ...) {
  va_list args;
  va_start(args, format);
  ut_log::LogBuffer::Instance().Append("FATAL", format, args);
  va_end(args);
}

#else  // __cplusplus

// C 版本：直接打印（fallback）
static inline void dlog_debug(int32_t moduleId, const char *format, ...) {
  va_list args;
  va_start(args, format);
  printf("[DEBUG] ");
  vprintf(format, args);
  printf("\n");
  va_end(args);
}

static inline void dlog_info(int32_t moduleId, const char *format, ...) {
  va_list args;
  va_start(args, format);
  printf("[INFO] ");
  vprintf(format, args);
  printf("\n");
  va_end(args);
}

static inline void dlog_warn(int32_t moduleId, const char *format, ...) {
  va_list args;
  va_start(args, format);
  printf("[WARN] ");
  vprintf(format, args);
  printf("\n");
  va_end(args);
}

static inline void dlog_error(int32_t moduleId, const char *format, ...) {
  va_list args;
  va_start(args, format);
  printf("[ERROR] ");
  vprintf(format, args);
  printf("\n");
  va_end(args);
}

static inline void dlog_fatal(int32_t moduleId, const char *format, ...) {
  va_list args;
  va_start(args, format);
  printf("[FATAL] ");
  vprintf(format, args);
  printf("\n");
  va_end(args);
}

#endif  // __cplusplus
