/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_log.h
 * \brief Super Kernel Log Module (Merged File Logger)
 *
 * Features:
 * 1. Supports on/off control, passthrough to dlog_* when disabled
 * 2. Creates directory structure: sk_meta/{Pid}/{ModelRI} when enabled
 * 3. Provides five log levels: trace/debug/info/warning/error
 * 4. Supports segmentation for long logs
 * 5. RAII-style log context manager
 * 6. Thread-safe file handle management
 */

#ifndef __SK_LOG_H__
#define __SK_LOG_H__

#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <fstream>
#include <sstream>
#include <cstring>
#include <chrono>
#include <cerrno>
#include <iomanip>
#include <thread>
#include <unistd.h>
#include <sys/stat.h>
#include <cstdlib>
#include "dlog_pub.h"

// Forward declaration for aclmdlRI
typedef void *aclmdlRI;

// ==================== CANN Module Definition ====================
#define ASCENDC_MODULE_NAME static_cast<int32_t>(ASCENDCKERNEL)

// ==================== Low-Level Log Macros (SK_DLOG*) ====================
// These macros directly call dlog_* interfaces, used for passthrough
#define SK_DLOGD(format, ...)                                                              \
  do {                                                                                     \
    dlog_debug(ASCENDC_MODULE_NAME, "[SK][%s] " format "\n", __FUNCTION__, ##__VA_ARGS__); \
  } while (0)

#define SK_DLOGW(format, ...)                                                             \
  do {                                                                                    \
    dlog_warn(ASCENDC_MODULE_NAME, "[SK][%s] " format "\n", __FUNCTION__, ##__VA_ARGS__); \
  } while (0)

#define SK_DLOGI(format, ...)                                                             \
  do {                                                                                    \
    dlog_info(ASCENDC_MODULE_NAME, "[SK][%s] " format "\n", __FUNCTION__, ##__VA_ARGS__); \
  } while (0)

#define SK_DLOGE(format, ...)                                                              \
  do {                                                                                     \
    dlog_error(ASCENDC_MODULE_NAME, "[SK][%s] " format "\n", __FUNCTION__, ##__VA_ARGS__); \
  } while (0)

// ==================== Helper Macro Definitions ====================
#define CHECK_ACL(x)                   \
  do {                                 \
    aclError __ret = x;                \
    if (__ret != ACL_ERROR_NONE) {     \
      SK_DLOGE("aclError: %d", __ret); \
    }                                  \
  } while (0)

#define REPORT_ERROR_MESSAGE(...)    \
  do {                               \
    ReportErrorMessage(__VA_ARGS__); \
  } while (0)

#define SK_ASSERT_RETVAL(cond, ret)                    \
  do {                                                 \
    if (!(cond)) {                                     \
      SK_DLOGE_WITH_REPORT("Assert %s failed", #cond); \
      return (ret);                                    \
    }                                                  \
  } while (0)

#define SK_DLOGE_WITH_REPORT(format, ...)                                                  \
  do {                                                                                     \
    dlog_error(ASCENDC_MODULE_NAME, "[SK][%s] " format "\n", __FUNCTION__, ##__VA_ARGS__); \
    REPORT_ERROR_MESSAGE(format, ##__VA_ARGS__);                                           \
  } while (0)

// ==================== Helper Functions ====================
constexpr mode_t SK_DIRECTORY_MODE = 0755;

constexpr const char *GetFileName(const char *path) {
  const char *file = path;
  while (*path != '\0') {
    if (*path++ == '/') {
      file = path;
    }
  }
  return file;
}

void ReportErrorMessageInner(const std::string &code, const char *fmt, ...);

template <typename... Arguments>
void ReportErrorMessage(const char *fmt, Arguments &&...args) {
  std::string errorCode = "EZ9999";
  return ReportErrorMessageInner(errorCode, fmt, std::forward<Arguments>(args)...);
}

inline std::string SanitizePathComponent(const std::string &component) {
  std::string result = component;
  for (char &pathChar : result) {
    if (pathChar == '/' || pathChar == '\\' || pathChar == ':' || pathChar == '*' || pathChar == '?' ||
        pathChar == '"' || pathChar == '<' || pathChar == '>' || pathChar == '|') {
      pathChar = '_';
    }
  }
  return result;
}

inline std::string GetSkMetaBasePath() {
  return "sk_meta/" + std::to_string(getpid());
}

inline std::string GetSkMetaPath(const std::string &modelId) {
  if (modelId.empty()) {
    return "";
  }
  return GetSkMetaBasePath() + "/" + SanitizePathComponent(modelId);
}

inline bool CreateDirectoryRecursive(const std::string &path) {
  if (path.empty()) {
    return false;
  }

  size_t pos = 0;
  do {
    pos = path.find('/', pos + 1);
    std::string subPath = path.substr(0, pos);
    if (subPath.empty()) {
      continue;
    }

    struct stat st;
    if (stat(subPath.c_str(), &st) != 0 && mkdir(subPath.c_str(), SK_DIRECTORY_MODE) != 0 && errno != EEXIST) {
      return false;
    }
  } while (pos != std::string::npos && pos < path.size());
  return true;
}

inline std::string CreateSkMetaDirectory(const std::string &modelId) {
  std::string dirPath = GetSkMetaPath(modelId);
  if (!CreateDirectoryRecursive(dirPath)) {
    return "";
  }
  return dirPath;
}

// ==================== File Logger Namespace ====================
namespace sk {
namespace logger {

class FileLogSuppressGuard;

// ==================== Log Level Enumeration ====================
enum class LogLevel : uint8_t {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARNING = 3,
  ERROR = 4,
  OFF = 5  // Disable all logs
};

// Convert log level to string
inline const char *LogLevelToString(LogLevel level) {
  switch (level) {
    case LogLevel::TRACE:
      return "TRACE";
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::WARNING:
      return "WARN";
    case LogLevel::ERROR:
      return "ERROR";
    default:
      return "UNKNOWN";
  }
}

// ==================== Configuration Structure ====================
struct LoggerConfig {
  bool enabled = false;                    // Enable file logging
  LogLevel minLevel = LogLevel::INFO;      // Minimum log level
  std::string baseDir = "sk_meta";         // Base directory name
  std::string modelId;                     // Current model ID, e.g. model_48_1
  size_t maxFileSize = 100 * 1024 * 1024;  // Max file size: 100MB
  size_t maxLineLength = 4096;             // Max line length (for segmentation)
  bool enableTimestamp = true;             // Add timestamp
  bool enablePidTid = true;                // Add process/thread ID
};

// ==================== File Handle Information ====================
struct FileHandleInfo {
  std::string filePath;                              // Full file path
  std::ofstream fileStream;                          // File stream
  size_t currentSize = 0;                            // Current file size
  size_t writeCount = 0;                             // Write count
  std::chrono::system_clock::time_point createTime;  // Creation time

  FileHandleInfo() = default;
  FileHandleInfo(const FileHandleInfo &) = delete;
  FileHandleInfo &operator=(const FileHandleInfo &) = delete;

  FileHandleInfo(FileHandleInfo &&other) noexcept
      : filePath(std::move(other.filePath)),
        fileStream(std::move(other.fileStream)),
        currentSize(other.currentSize),
        writeCount(other.writeCount),
        createTime(other.createTime) {}

  FileHandleInfo &operator=(FileHandleInfo &&other) noexcept {
    if (this != &other) {
      if (fileStream.is_open()) {
        fileStream.close();
      }
      filePath = std::move(other.filePath);
      fileStream = std::move(other.fileStream);
      currentSize = other.currentSize;
      writeCount = other.writeCount;
      createTime = other.createTime;
    }
    return *this;
  }

  ~FileHandleInfo() {
    if (fileStream.is_open()) {
      fileStream.flush();
      fileStream.close();
    }
  }
};

// ==================== File Handle Manager ====================
class FileHandleManager {
 public:
  static FileHandleManager &Instance() {
    static FileHandleManager instance;
    return instance;
  }

  // Register log file
  bool RegisterFile(const std::string &name, const std::string &path);

  // Switch to specified file handle
  bool SwitchToFile(const std::string &name);

  // Switch to default handle
  void SwitchToDefault();

  // Write log
  bool Write(const std::string &name, const std::string &content);

  // Write to current handle
  bool WriteToCurrent(const std::string &content);

  // Get current handle name
  std::string GetCurrentHandle() const;

  // Get file size
  size_t GetFileSize(const std::string &name);

  // Close specified file
  void CloseFile(const std::string &name);

  // Initialize default file
  bool InitializeDefault(const std::string &modelId);

 private:
  class ThreadAwareMutex {
   public:
    void lock() {
      mutex_.lock();
      heldByCurrentThread_ = true;
    }

    void unlock() {
      heldByCurrentThread_ = false;
      mutex_.unlock();
    }

    bool IsHeldByCurrentThread() const {
      return heldByCurrentThread_;
    }

   private:
    std::mutex mutex_;
    inline static thread_local bool heldByCurrentThread_ = false;
  };

  FileHandleManager();
  ~FileHandleManager();

  FileHandleManager(const FileHandleManager &) = delete;
  FileHandleManager &operator=(const FileHandleManager &) = delete;

  bool IsLockedByCurrentThread() const {
    return mutex_.IsHeldByCurrentThread();
  }

  friend class FileLogger;

  std::unordered_map<std::string, FileHandleInfo> handles_;
  mutable ThreadAwareMutex mutex_;

  // Thread-local current handle to avoid multi-threading conflicts
  static thread_local std::string currentHandle_;
};

// ==================== RAII Log Context Manager ====================
class LogContextGuard {
 public:
  // Ensure and route to a model's default log file.
  explicit LogContextGuard(const std::string &modelId);
  // Register and route to a specific log file.
  explicit LogContextGuard(const std::string &handleName, const std::string &filePath);
  ~LogContextGuard();

  // Disable copy
  LogContextGuard(const LogContextGuard &) = delete;
  LogContextGuard &operator=(const LogContextGuard &) = delete;

  // Support move
  LogContextGuard(LogContextGuard &&other) noexcept;
  LogContextGuard &operator=(LogContextGuard &&other) noexcept;

  bool IsActive() const {
    return active_;
  }

 private:
  void Restore();

  std::string previousModelId_;
  std::string previousHandle_;
  bool active_ = false;
};

// ==================== Main Logger Class ====================
class FileLogger {
 public:
  static FileLogger &Instance();

  // Initialize
  bool Initialize(const LoggerConfig &config);

  // Write log to file if enabled (called by SK_LOG* macros after passthrough)
  template <typename... Args>
  void WriteLogIfEnabled(LogLevel level, const char *funcName, const char *fileName, int lineNum, const char *format,
                         Args &&...args) {
    if (!suppressFileLog_ && enabled_.load(std::memory_order_relaxed) &&
        level >= minLevel_.load(std::memory_order_relaxed)) {
      // SK_LOG* has already emitted its passthrough log. Skip the file sink when re-entered by its manager.
      if (FileHandleManager::Instance().IsLockedByCurrentThread()) {
        return;
      }
      LoggerConfig config = GetConfigSnapshot();
      if (!config.enabled || level < config.minLevel) {
        return;
      }
      std::string message =
          FormatMessage(config, level, funcName, fileName, lineNum, format, std::forward<Args>(args)...);
      WriteLog(message, config);
    }
  }

  // ========== Thread-local model ID management ==========
  static void SetCurrentModelId(const std::string &modelId) {
    currentModelId_ = modelId;
  }

  static const std::string &GetCurrentModelId();

  // File handle management
  bool RegisterLogFile(const std::string &name, const std::string &subPath = "");
  bool SwitchToFile(const std::string &name);
  void SwitchToDefault();

  // RAII context management
  std::unique_ptr<LogContextGuard> CreateContext(const std::string &fileName, const std::string &modelId);

  // Configuration management
  void SetEnabled(bool enabled);
  bool IsEnabled() const;
  void SetMinLevel(LogLevel level);
  void SetModelId(const std::string &modelId);
  bool IsInitialized() const;

 private:
  FileLogger() = default;
  ~FileLogger() = default;

  FileLogger(const FileLogger &) = delete;
  FileLogger &operator=(const FileLogger &) = delete;

  // Format message (using variadic arguments to avoid format-security warning)
  std::string FormatMessage(const LoggerConfig &config, LogLevel level, const char *funcName, const char *fileName,
                            int lineNum, const char *format, ...);

  void WriteLog(const std::string &message, const LoggerConfig &config);

  // Get the effective model ID. Prefer the thread-local ID and fall back to config_.
  std::string GetEffectiveModelId() const;
  LoggerConfig GetConfigSnapshot() const;

 private:
  friend class FileLogSuppressGuard;

  LoggerConfig config_;
  std::atomic<bool> enabled_{false};
  std::atomic<LogLevel> minLevel_{LogLevel::INFO};
  std::atomic<bool> initialized_{false};
  std::atomic<pid_t> pid_{0};
  // Serialize initialization without preventing log calls from taking a configuration snapshot.
  std::mutex initializationMutex_;
  mutable std::mutex mutex_;

  // Thread-local model ID, used to isolate concurrent aclskOptimize calls.
  static thread_local std::string currentModelId_;
  // Thread-local file log suppression, used by APIs that must keep dlog/plog but never write sk_meta files.
  static thread_local bool suppressFileLog_;
};

class FileLogSuppressGuard {
 public:
  FileLogSuppressGuard();
  ~FileLogSuppressGuard();

  FileLogSuppressGuard(const FileLogSuppressGuard &) = delete;
  FileLogSuppressGuard &operator=(const FileLogSuppressGuard &) = delete;

 private:
  bool previousSuppressFileLog_ = false;
};

}  // namespace logger
}  // namespace sk

// ==================== User Log Macros (SK_LOG*) ====================
// These macros provide passthrough + file logging functionality
// Design: Directly call SK_DLOG* for passthrough (captures real __FUNCTION__),
//         then write to file if enabled

#define SK_LOGT(format, ...)                                                                                  \
  do {                                                                                                        \
    SK_DLOGD(format, ##__VA_ARGS__);                                                                          \
    sk::logger::FileLogger::Instance().WriteLogIfEnabled(sk::logger::LogLevel::TRACE, __FUNCTION__, __FILE__, \
                                                         __LINE__, format, ##__VA_ARGS__);                    \
  } while (0)

#define SK_LOGD(format, ...)                                                                                  \
  do {                                                                                                        \
    SK_DLOGD(format, ##__VA_ARGS__);                                                                          \
    sk::logger::FileLogger::Instance().WriteLogIfEnabled(sk::logger::LogLevel::DEBUG, __FUNCTION__, __FILE__, \
                                                         __LINE__, format, ##__VA_ARGS__);                    \
  } while (0)

#define SK_LOGI(format, ...)                                                                                           \
  do {                                                                                                                 \
    SK_DLOGI(format, ##__VA_ARGS__);                                                                                   \
    sk::logger::FileLogger::Instance().WriteLogIfEnabled(sk::logger::LogLevel::INFO, __FUNCTION__, __FILE__, __LINE__, \
                                                         format, ##__VA_ARGS__);                                       \
  } while (0)

#define SK_LOGW(format, ...)                                                                                    \
  do {                                                                                                          \
    SK_DLOGW(format, ##__VA_ARGS__);                                                                            \
    sk::logger::FileLogger::Instance().WriteLogIfEnabled(sk::logger::LogLevel::WARNING, __FUNCTION__, __FILE__, \
                                                         __LINE__, format, ##__VA_ARGS__);                      \
  } while (0)

#define SK_LOGE(format, ...)                                                                                  \
  do {                                                                                                        \
    SK_DLOGE(format, ##__VA_ARGS__);                                                                          \
    sk::logger::FileLogger::Instance().WriteLogIfEnabled(sk::logger::LogLevel::ERROR, __FUNCTION__, __FILE__, \
                                                         __LINE__, format, ##__VA_ARGS__);                    \
  } while (0)

// ==================== RAII Context Macros ====================
#define SK_LOG_CONTEXT(fileName, modelId) \
  auto _sk_log_context_guard = sk::logger::FileLogger::Instance().CreateContext(fileName, modelId)

#define SK_LOG_CONTEXT_SIMPLE(fileName) \
  auto _sk_log_context_guard =          \
      sk::logger::FileLogger::Instance().CreateContext(fileName, sk::logger::FileLogger::GetCurrentModelId())

// ==================== Global Initialization Helper Functions and Macros ====================

/**
 * @brief Initialize logger with the current model ID
 * @param modelId Current model ID, e.g. model_48_1
 *
 * Reads environment variable ASCEND_OP_COMPILE_SAVE_KERNEL_META to enable/disable file logging.
 * - ASCEND_OP_COMPILE_SAVE_KERNEL_META=1: Enable file logging
 * - ASCEND_OP_COMPILE_SAVE_KERNEL_META=0 or unset: Disable file logging
 *
 * @example
 *   InitSkLogger(modelId);
 *   // Creates directory: sk_meta/{pid}/model_305419896/
 */
inline void InitSkLogger(const std::string &modelId) {
  sk::logger::FileLogger::SetCurrentModelId(modelId);

  // Read environment variable
  const char *envVar = std::getenv("ASCEND_OP_COMPILE_SAVE_KERNEL_META");
  bool enabled = false;

  if (envVar != nullptr) {
    std::string value(envVar);
    // Trim leading and trailing whitespace
    value.erase(0, value.find_first_not_of(" \t\n\r\f\v"));
    value.erase(value.find_last_not_of(" \t\n\r\f\v") + 1);

    // Check if value is "1" (enable file logging)
    enabled = (value == "1");

    if (enabled) {
      SK_DLOGI("File logger enabled by environment variable ASCEND_OP_COMPILE_SAVE_KERNEL_META=1");
    } else {
      SK_DLOGI("File logger disabled by environment variable ASCEND_OP_COMPILE_SAVE_KERNEL_META=%s", value.c_str());
    }
  } else {
    SK_DLOGI("File logger disabled (environment variable ASCEND_OP_COMPILE_SAVE_KERNEL_META not set)");
  }

  // Initialize file logger
  sk::logger::LoggerConfig config;
  config.enabled = enabled;
  config.modelId = modelId;
  config.minLevel = sk::logger::LogLevel::DEBUG;
  config.baseDir = "sk_meta";

  if (!sk::logger::FileLogger::Instance().Initialize(config)) {
    SK_DLOGW("Failed to initialize file logger, falling back to dlog only");
    // Disable file logging on initialization failure
    sk::logger::FileLogger::Instance().SetEnabled(false);
  }
}

// Manual initialization function
bool InitializeSkFileLogger(bool enabled, const std::string &modelId = "",
                            sk::logger::LogLevel minLevel = sk::logger::LogLevel::INFO);

#define INIT_SK_FILE_LOGGER(modelId) InitializeSkFileLogger(true, modelId, sk::logger::LogLevel::DEBUG)

#define INIT_SK_FILE_LOGGER_MINIMAL(modelId) InitializeSkFileLogger(true, modelId, sk::logger::LogLevel::INFO)

#define DISABLE_SK_FILE_LOGGER() sk::logger::FileLogger::Instance().SetEnabled(false)

#endif  // __SK_LOG_H__
