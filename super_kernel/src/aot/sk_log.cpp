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
 * \file sk_log.cpp
 * \brief Super Kernel Log Module Implementation (Merged File Logger)
 */

#include "sk_log.h"
#include "sk_common.h"
#include "securec.h"
#include "base/err_mgr.h"
#include <iostream>
#include <cstdarg>
#include <cstdlib>
#include <limits.h>

// ==================== Original Error Report Function ====================
constexpr size_t LIMIT_PREDEFINED_MESSAGE = 1024U;

void ReportErrorMessageInner(const std::string &code, const char *fmt, ...) {
  std::vector<char> buf(LIMIT_PREDEFINED_MESSAGE, '\0');
  va_list argList;
  va_start(argList, fmt);
  auto ret = vsnprintf_s(buf.data(), LIMIT_PREDEFINED_MESSAGE, LIMIT_PREDEFINED_MESSAGE - 1U, fmt, argList);
  if (ret == -1) {
    SK_DLOGE("Construct error message failed, maybe the length of error message exceed limits: %zu",
             LIMIT_PREDEFINED_MESSAGE);
  }
  va_end(argList);
  const std::vector<const char *> msgKey = {"message"};
  const std::vector<const char *> msgValue = {buf.data()};
  REPORT_PREDEFINED_ERR_MSG(code.c_str(), msgKey, msgValue);
}

// ==================== FileHandleManager Implementation ====================
namespace sk {
namespace logger {

bool FileHandleManager::RegisterFile(const std::string &name, const std::string &path) {
  {
    std::lock_guard<ThreadAwareMutex> lock(mutex_);

    if (handles_.find(name) != handles_.end()) {
      return true;  // Already registered
    }

    FileHandleInfo handle;
    handle.filePath = path;
    handle.fileStream.open(path, std::ios::app | std::ios::out);

    if (handle.fileStream.is_open()) {
      handle.createTime = std::chrono::system_clock::now();

      // Get current file size for existing files
      handle.fileStream.seekp(0, std::ios::end);
      handle.currentSize = static_cast<size_t>(handle.fileStream.tellp());

      handles_.emplace(name, std::move(handle));
      return true;
    }
  }

  SK_LOGE("Failed to open log file: %s", path.c_str());
  return false;
}

bool FileHandleManager::SwitchToFile(const std::string &name) {
  {
    std::lock_guard<ThreadAwareMutex> lock(mutex_);

    auto it = handles_.find(name);
    if (it != handles_.end()) {
      currentHandle_ = name;
      return true;
    }
  }

  SK_LOGE("File handle not found: %s", name.c_str());
  return false;
}

void FileHandleManager::SwitchToDefault() {
  std::lock_guard<ThreadAwareMutex> lock(mutex_);
  currentHandle_ = "default";
}

bool FileHandleManager::Write(const std::string &name, const std::string &content) {
  std::lock_guard<ThreadAwareMutex> lock(mutex_);

  auto it = handles_.find(name);
  if (it == handles_.end() || !it->second.fileStream.is_open()) {
    return false;
  }

  auto &handle = it->second;
  handle.fileStream << content;
  handle.fileStream.flush();

  handle.currentSize += content.size();
  handle.writeCount++;

  return true;
}

bool FileHandleManager::WriteToCurrent(const std::string &content) {
  return Write(currentHandle_, content);
}

std::string FileHandleManager::GetCurrentHandle() const {
  std::lock_guard<ThreadAwareMutex> lock(mutex_);
  return currentHandle_;
}

size_t FileHandleManager::GetFileSize(const std::string &name) {
  std::lock_guard<ThreadAwareMutex> lock(mutex_);

  auto it = handles_.find(name);
  if (it == handles_.end()) {
    return 0;
  }
  return it->second.currentSize;
}

void FileHandleManager::CloseFile(const std::string &name) {
  std::lock_guard<ThreadAwareMutex> lock(mutex_);

  auto it = handles_.find(name);
  if (it != handles_.end()) {
    if (it->second.fileStream.is_open()) {
      it->second.fileStream.flush();
      it->second.fileStream.close();
    }
    handles_.erase(it);
  }
}

bool FileHandleManager::InitializeDefault(const std::string &modelId) {
  std::string dirPath = GetSkMetaPath(modelId);
  if (!CreateDirectoryRecursive(dirPath)) {
    return false;
  }

  std::string handleName = SanitizePathComponent(modelId);

  std::string defaultPath = dirPath + "/super_kernel.log";
  return RegisterFile(handleName, defaultPath);
}

// Thread-local current handle initialization
thread_local std::string FileHandleManager::currentHandle_ = "default";

// Thread-local current model ID
thread_local std::string FileLogger::currentModelId_;

// Thread-local file log suppression
thread_local bool FileLogger::suppressFileLog_ = false;

FileLogSuppressGuard::FileLogSuppressGuard() : previousSuppressFileLog_(FileLogger::suppressFileLog_) {
  FileLogger::suppressFileLog_ = true;
}

FileLogSuppressGuard::~FileLogSuppressGuard() {
  FileLogger::suppressFileLog_ = previousSuppressFileLog_;
}

FileHandleManager::FileHandleManager() {}

FileHandleManager::~FileHandleManager() {
  std::lock_guard<ThreadAwareMutex> lock(mutex_);
  for (auto &pair : handles_) {
    if (pair.second.fileStream.is_open()) {
      pair.second.fileStream.flush();
      pair.second.fileStream.close();
    }
  }
}

// ==================== LogContextGuard Implementation ====================
LogContextGuard::LogContextGuard(const std::string &modelId) {
  auto &logger = FileLogger::Instance();
  if (!logger.IsInitialized() || !logger.IsEnabled()) {
    return;
  }

  previousModelId_ = FileLogger::GetCurrentModelId();
  previousHandle_ = FileHandleManager::Instance().GetCurrentHandle();
  FileLogger::SetCurrentModelId(modelId);
  FileHandleManager::Instance().SwitchToDefault();
  active_ = true;

  if (!FileHandleManager::Instance().InitializeDefault(modelId)) {
    SK_DLOGE("Failed to initialize default log file for model: %s", modelId.c_str());
  }
}

LogContextGuard::LogContextGuard(const std::string &handleName, const std::string &filePath)
    : previousModelId_(FileLogger::GetCurrentModelId()),
      previousHandle_(FileHandleManager::Instance().GetCurrentHandle()) {
  if (!FileHandleManager::Instance().RegisterFile(handleName, filePath)) {
    SK_DLOGE("Failed to register log file: %s", handleName.c_str());
    return;
  }

  if (!FileHandleManager::Instance().SwitchToFile(handleName)) {
    SK_DLOGE("Failed to switch to log file: %s", handleName.c_str());
    return;
  }

  active_ = true;
}

LogContextGuard::~LogContextGuard() {
  Restore();
}

LogContextGuard::LogContextGuard(LogContextGuard &&other) noexcept
    : previousModelId_(std::move(other.previousModelId_)),
      previousHandle_(std::move(other.previousHandle_)),
      active_(other.active_) {
  other.active_ = false;
}

LogContextGuard &LogContextGuard::operator=(LogContextGuard &&other) noexcept {
  if (this != &other) {
    Restore();
    previousModelId_ = std::move(other.previousModelId_);
    previousHandle_ = std::move(other.previousHandle_);
    active_ = other.active_;
    other.active_ = false;
  }
  return *this;
}

void LogContextGuard::Restore() {
  if (!active_) {
    return;
  }

  FileLogger::SetCurrentModelId(previousModelId_);
  if (previousHandle_ == "default") {
    FileHandleManager::Instance().SwitchToDefault();
  } else if (!previousHandle_.empty()) {
    FileHandleManager::Instance().SwitchToFile(previousHandle_);
  }
  active_ = false;
}

// ==================== FileLogger Implementation ====================
FileLogger &FileLogger::Instance() {
  static FileLogger instance;
  return instance;
}

bool FileLogger::Initialize(const LoggerConfig &config) {
  std::lock_guard<std::mutex> initializationLock(initializationMutex_);

  LoggerConfig effectiveConfig = config;
  if (effectiveConfig.modelId.empty()) {
    effectiveConfig.modelId = GetCurrentModelId();
  }
  if (effectiveConfig.modelId.empty()) {
    std::lock_guard<std::mutex> lock(mutex_);
    effectiveConfig.modelId = config_.modelId;
  }

  initialized_.store(false, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = effectiveConfig;
    enabled_.store(false, std::memory_order_relaxed);
    minLevel_.store(effectiveConfig.minLevel, std::memory_order_relaxed);
  }
  if (!effectiveConfig.enabled) {
    SK_DLOGI("File logger is disabled");
    initialized_.store(true, std::memory_order_relaxed);
    return true;
  }

  // Use the current model ID to create sk_meta directory structure.
  std::string logDir = CreateSkMetaDirectory(effectiveConfig.modelId);
  if (logDir.empty()) {
    SK_DLOGE("Failed to create sk_meta directory");
    return false;
  }

  // Extract PID from created directory
  pid_.store(getpid(), std::memory_order_relaxed);

  // 为新 model ID 注册日志文件
  std::string modelIdForLog = effectiveConfig.modelId;
  std::string handleName = SanitizePathComponent(modelIdForLog);
  std::string defaultPath = logDir + "/super_kernel.log";

  if (!FileHandleManager::Instance().RegisterFile(handleName, defaultPath)) {
    SK_DLOGE("Failed to register log file for model: %s", modelIdForLog.c_str());
    return false;
  }

  // 切换到新的日志文件
  FileHandleManager::Instance().SwitchToFile(handleName);

  initialized_.store(true, std::memory_order_relaxed);
  enabled_.store(true, std::memory_order_relaxed);

  // Convert to absolute path for better visibility
  char absPath[PATH_MAX] = {0};  // Initialize buffer with zeros
  if (realpath(logDir.c_str(), absPath) != nullptr) {
    // Ensure null termination
    absPath[PATH_MAX - 1] = '\0';
    SK_DLOGI("File logger initialized: dir=%s, model=%s", absPath, modelIdForLog.c_str());
  } else {
    SK_DLOGI("File logger initialized: dir=%s, model=%s", logDir.c_str(), modelIdForLog.c_str());
  }
  return true;
}

std::string FileLogger::GetEffectiveModelId() const {
  if (!currentModelId_.empty()) {
    return currentModelId_;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  return config_.modelId;
}

LoggerConfig FileLogger::GetConfigSnapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  LoggerConfig config = config_;
  config.enabled = enabled_.load(std::memory_order_relaxed);
  config.minLevel = minLevel_.load(std::memory_order_relaxed);
  return config;
}

bool FileLogger::RegisterLogFile(const std::string &name, const std::string &subPath) {
  if (!initialized_.load() || !enabled_.load(std::memory_order_relaxed)) {
    return false;
  }

  std::string modelId = GetEffectiveModelId();
  if (modelId.empty()) {
    return false;
  }

  std::string basePath = GetSkMetaPath(modelId);
  std::string filePath = basePath + "/" + name;

  // If subPath provided, insert it before filename
  if (!subPath.empty()) {
    filePath = basePath + "/" + SanitizePathComponent(subPath) + "/" + name;
  }

  // Ensure directory exists using common utility
  size_t lastSlash = filePath.find_last_of('/');
  if (lastSlash != std::string::npos) {
    std::string dir = filePath.substr(0, lastSlash);
    if (!CreateDirectoryRecursive(dir)) {
      return false;
    }
  }

  std::string handleName = SanitizePathComponent(modelId) + "_" + name;

  return FileHandleManager::Instance().RegisterFile(handleName, filePath);
}

bool FileLogger::SwitchToFile(const std::string &name) {
  if (!initialized_.load() || !enabled_.load(std::memory_order_relaxed)) {
    return false;
  }
  return FileHandleManager::Instance().SwitchToFile(name);
}

void FileLogger::SwitchToDefault() {
  FileHandleManager::Instance().SwitchToDefault();
}

std::unique_ptr<LogContextGuard> FileLogger::CreateContext(const std::string &fileName, const std::string &modelId) {
  if (suppressFileLog_ || !initialized_.load() || !enabled_.load(std::memory_order_relaxed)) {
    return nullptr;
  }

  if (modelId.empty()) {
    return nullptr;
  }

  std::string sanitizedModelId = SanitizePathComponent(modelId);
  std::string dirPath = GetSkMetaBasePath() + "/" + sanitizedModelId;
  if (!CreateDirectoryRecursive(dirPath)) {
    SK_DLOGE("Failed to create directory for context");
    return nullptr;
  }

  std::string filePath = dirPath + "/" + fileName;
  std::string handleName = sanitizedModelId + "_" + fileName;

  return std::make_unique<LogContextGuard>(handleName, filePath);
}

void FileLogger::SetEnabled(bool enabled) {
  enabled_.store(enabled, std::memory_order_relaxed);
}

bool FileLogger::IsEnabled() const {
  return enabled_.load(std::memory_order_relaxed);
}

void FileLogger::SetMinLevel(LogLevel level) {
  minLevel_.store(level, std::memory_order_relaxed);
}

void FileLogger::SetModelId(const std::string &modelId) {
  std::lock_guard<std::mutex> initializationLock(initializationMutex_);
  std::lock_guard<std::mutex> lock(mutex_);
  config_.modelId = modelId;
}

bool FileLogger::IsInitialized() const {
  return initialized_.load();
}

const std::string &FileLogger::GetCurrentModelId() {
  return currentModelId_;
}

std::string FileLogger::FormatMessage(const LoggerConfig &config, LogLevel level, const char *funcName,
                                      const char *fileName, int lineNum, const char *format, ...) {
  std::ostringstream oss;

  // Timestamp
  if (config.enableTimestamp) {
    auto now = std::chrono::system_clock::now();
    auto timeValue = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm localTime{};
    if (localtime_r(&timeValue, &localTime) != nullptr) {
      oss << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S");
      oss << "." << std::setfill('0') << std::setw(3) << ms.count();
      oss << " ";
    }
  }

  // Process/Thread ID
  if (config.enablePidTid) {
    oss << "[" << pid_.load(std::memory_order_relaxed) << ":" << std::this_thread::get_id() << "] ";
  }

  // Log level
  oss << "[" << LogLevelToString(level) << "] ";

  // File name and line number (extract just the filename from full path)
  if (fileName != nullptr) {
    const char *baseName = strrchr(fileName, '/');
    if (baseName != nullptr) {
      baseName++;  // Skip '/'
    } else {
      baseName = fileName;
    }
    oss << "[" << baseName << ":" << lineNum << "] ";
  }

  // Function name
  if (funcName != nullptr) {
    oss << "[" << funcName << "] ";
  }

  // Log content - use secure function
  va_list args;
  va_start(args, format);
  char buffer[4096];
  int ret = vsnprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, format, args);
  va_end(args);

  if (ret < 0) {
    // Format error, use error message
    oss << "[FORMAT_ERROR] ";
    oss << "Failed to format log message\n";
  } else {
    oss << buffer << "\n";
  }

  return oss.str();
}

void FileLogger::WriteLog(const std::string &message, const LoggerConfig &config) {
  if (message.empty()) {
    return;
  }

  // 获取当前 handle，判断是否在 LogContextGuard 上下文中
  auto &handleManager = FileHandleManager::Instance();
  std::string currentHandle = handleManager.GetCurrentHandle();
  std::string targetHandle;

  // "default" 表示模型级日志上下文，按当前 model ID 选择对应 handle。
  if (currentHandle == "default") {
    const std::string &modelId = currentModelId_.empty() ? config.modelId : currentModelId_;
    targetHandle = SanitizePathComponent(modelId);
  } else {
    // 在 LogContextGuard 上下文中，使用当前 handle
    targetHandle = currentHandle;
  }

  // Long log segmentation handling
  if (message.size() > config.maxLineLength) {
    size_t offset = 0;
    size_t segmentNum = 0;

    while (offset < message.size()) {
      size_t length = std::min(config.maxLineLength, message.size() - offset);
      std::string segment;

      if (segmentNum == 0) {
        segment = message.substr(offset, length);
      } else {
        segment = "[CONT:" + std::to_string(segmentNum) + "] " + message.substr(offset, length);
      }

      handleManager.Write(targetHandle, segment);
      offset += length;
      segmentNum++;
    }
  } else {
    handleManager.Write(targetHandle, message);
  }
}

}  // namespace logger
}  // namespace sk

// ==================== Global Initialization Helper Functions ====================

bool InitializeSkFileLogger(bool enabled, const std::string &modelId, sk::logger::LogLevel minLevel) {
  sk::logger::FileLogger::Instance().SetCurrentModelId(modelId);
  sk::logger::LoggerConfig config;
  config.enabled = enabled;
  config.modelId = modelId;
  config.minLevel = minLevel;

  if (!sk::logger::FileLogger::Instance().Initialize(config)) {
    // Disable file logging on initialization failure
    sk::logger::FileLogger::Instance().SetEnabled(false);
    return false;
  }
  return true;
}
