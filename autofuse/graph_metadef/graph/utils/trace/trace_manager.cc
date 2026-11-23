/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/util/trace_manager/trace_manager.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <pthread.h>

#include "mmpa/mmpa_api.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/ge_context.h"
#include "graph_metadef/graph/utils/file_utils.h"

namespace af {
namespace {
class TraceFileHolder {
 public:
  explicit TraceFileHolder(int32_t fd) : fd_(fd) {}
  TraceFileHolder(TraceFileHolder const &) = delete;
  TraceFileHolder &operator=(TraceFileHolder const &) = delete;
  ~TraceFileHolder() {
    if (fd_ >= 0) {
      (void)mmClose(fd_);
      fd_ = -1;
    }
  }

  void Write(const char_t *data, const char *separator = "\r\n") const {
    if (fd_ >= 0) {
      const mmSsize_t written_count = mmWrite(fd_, const_cast<char_t *>(data), strlen(data));
      if ((written_count == EN_INVALID_PARAM) || (written_count == EN_ERROR)) {
        GELOGE(INTERNAL_ERROR, "[trace] Failed to write trace info to trace file, fd:%d, data:%s", fd_, data);
      }
      (void)mmWrite(fd_, const_cast<char_t *>(separator), strlen(separator));
    }
  }

  bool Valid() const {
    return fd_ >= 0;
  }

 private:
  int32_t fd_;
};

std::string CurrentTimeInSecondsStr() {
  mmSystemTime_t sysTime;
  if (mmGetSystemTime(&sysTime) != EN_OK) {
    GELOGE(INTERNAL_ERROR, "Get current time failed");
    const static std::string kInvalidTimeStr;
    return kInvalidTimeStr;
  }

  std::stringstream ss;
  ss << sysTime.wYear << sysTime.wMonth << sysTime.wDay << sysTime.wHour << sysTime.wMinute << sysTime.wSecond;
  return ss.str();
}

constexpr uint64_t kTraceSaveArraySize = (kTraceSaveTriggerNum << 1U);
constexpr uint64_t kTraceSaveCountsPerFile = 2000000U;
}  // namespace

thread_local std::string TraceManager::trace_header_;
thread_local std::string TraceManager::graph_name_;

TraceManager &TraceManager::GetInstance() {
  static TraceManager instance;
  return instance;
}

// Set owner
void TraceManager::SetTraceOwner(const std::string &owner, const std::string &stage, const std::string &graph_name) {
  if (!enabled_) {
    return;
  }
  trace_header_ = owner + ":" + stage;
  graph_name_ = graph_name;
}

// Clear owner
void TraceManager::ClearTraceOwner() {
  if (!enabled_) {
    return;
  }
  trace_header_.clear();
  graph_name_.clear();
}

std::string TraceManager::NextFileName() {
  static std::atomic<uint64_t> uuid(0U);

  std::stringstream ss;
  // need 3 widths to express uuid
  ss << trace_save_file_path_ << "trace_" << CurrentTimeInSecondsStr() << "_" << std::setw(3) << std::setfill('0')
     << uuid++ << ".txt";

  return ss.str();
}

std::unique_ptr<TraceFileHolder> OpenOrCreateFile(const std::string &file_path) {
  if (strnlen(file_path.c_str(), MMPA_MAX_PATH) >= MMPA_MAX_PATH) {
    GELOGE(PATH_INVALID, "[trace] Trace file name %s exceeds max length %u", file_path.c_str(),
           static_cast<uint32_t>(MMPA_MAX_PATH));
    return nullptr;
  }

  char_t real_path[MMPA_MAX_PATH] = {};
  if (mmRealPath(file_path.c_str(), &real_path[0], MMPA_MAX_PATH) != EN_OK) {
    GELOGI("[trace] Create new trace file %s", file_path.c_str());
  }

  const static auto kFlag = static_cast<int32_t>(static_cast<uint32_t>(M_WRONLY) | static_cast<uint32_t>(M_CREAT) |
                                                 static_cast<uint32_t>(M_APPEND));
  const static auto kMode = static_cast<mmMode_t>(static_cast<uint32_t>(M_IRUSR) | static_cast<uint32_t>(M_IWUSR));

  return ComGraphMakeUnique<TraceFileHolder>(mmOpen2(&real_path[0], kFlag, kMode));
}

void TraceManager::SaveTraceBufferToFile(const ReadyPart ready_part) {
  if (ready_part == ReadyPart::None) {
    return;
  }

  ge::ScopeGuard guard([this, ready_part]() {
    // Saved count must update for un-block add tracing thread
    if (ready_part == ReadyPart::A) {
      part1_ready_nums_ = 0U;
    } else {
      part2_ready_nums_ = 0U;
    }
    // Must update save nums after clear part ready nums
    total_saved_nums_ += kTraceSaveTriggerNum;
  });

  if (current_saving_file_name_.empty() || (current_file_saved_nums_ >= kTraceSaveCountsPerFile)) {
    current_saving_file_name_ = NextFileName();
    current_file_saved_nums_ = 0U;
  }

  auto fh = OpenOrCreateFile(current_saving_file_name_);
  if (fh == nullptr || (!fh->Valid())) {
    GELOGE(INTERNAL_ERROR, "[trace] Failed to get file holder for %s", current_saving_file_name_.c_str());
    return;
  }

  while (((ready_part == ReadyPart::A) && (part1_ready_nums_ < kTraceSaveTriggerNum)) ||
         ((ready_part == ReadyPart::B) && (part2_ready_nums_ < kTraceSaveTriggerNum))) {
  }
  const size_t start = (ready_part == ReadyPart::A) ? 0U : kTraceSaveTriggerNum;
  for (size_t i = start; i < (start + kTraceSaveTriggerNum); i++) {
    if (!trace_array_[i].empty()) {
      current_file_saved_nums_++;
      fh->Write(trace_array_[i].c_str());
    }
  }
}

void TraceManager::SaveBufferToFileThreadFunc() {
  (void)pthread_setname_np(pthread_self(), "ge_trace_savbuf");
  while (true) {
    std::unique_lock<std::mutex> lock_file(mu_);
    while ((ready_part_ == ReadyPart::None) && (!stopped_)) {
      data_ready_var_.wait(lock_file);
    }
    if (stopped_ && (ready_part_ == ReadyPart::None)) {  // Keep save remain trace even request stop
      break;
    }
    const auto ready_part = ready_part_;
    ready_part_ = ReadyPart::None;
    lock_file.unlock();

    SaveTraceBufferToFile(ready_part);
  }
}

Status TraceManager::Initialize(const char_t *file_save_path) {
  // init data
  std::stringstream ss;
  ss << file_save_path << MMPA_PATH_SEPARATOR_STR << "extra-info" << MMPA_PATH_SEPARATOR_STR << "graph_trace"
     << MMPA_PATH_SEPARATOR_STR << af::GetContext().DeviceId() << MMPA_PATH_SEPARATOR_STR;
  trace_save_file_path_ = ss.str();
  if (CreateDir(trace_save_file_path_) != 0) {
    GELOGE(INTERNAL_ERROR, "[trace] Trace not enabled as failed to create trace file save directory[%s]",
           trace_save_file_path_.c_str());
    return FAILED;
  }
  trace_array_.resize(kTraceSaveTriggerNum << 1U);
  try {
    save_thread_ = std::thread(&TraceManager::SaveBufferToFileThreadFunc, this);
  } catch (const std::system_error &) {
    GELOGE(INTERNAL_ERROR, "[trace] Trace not enabled as failed to start trace saving thread");
    return FAILED;
  }
  return SUCCESS;
}

void TraceManager::Finalize() {
  std::thread([this]() {
    (void)pthread_setname_np(pthread_self(), "ge_trace_final");
    // Trigger save for left trace info, trace added when or after dtor may lose
    for (size_t i = 1; i < kTraceSaveTriggerNum; i++) {
      AddTrace("");
    }
  }).join();
  // After join the thread above, remain trace must have trigger save part A or B
  std::unique_lock<std::mutex> lk(mu_);
  stopped_ = true;  // stopping record any new trace here
  data_ready_var_.notify_all();
  lk.unlock();
  if (save_thread_.joinable()) {
    save_thread_.join();
  }
}

TraceManager::TraceManager() {
  const char_t *trace_env_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_NPU_COLLECT_PATH, trace_env_path);
  enabled_ = (trace_env_path != nullptr) && (trace_env_path[0U] != '\0');
  if (!enabled_) {
    GELOGI("[trace] Trace not enabled as env 'NPU_COLLECT_PATH' not set");
    return;
  }

  if (Initialize(trace_env_path) != SUCCESS) {
    enabled_ = false;
    GELOGE(INTERNAL_ERROR, "[trace] Trace not enabled as initialize failed");
  }
}

TraceManager::~TraceManager() {
  if (!enabled_) {
    return;
  }
  Finalize();
}

void TraceManager::AddTrace(std::string &&trace_info) {
  if (!enabled_) {
    return;
  }
  // Assume kTraceSaveArraySize = 2 * kTraceSaveTriggerNum
  const auto current_trace_nums = trace_index_.fetch_add(1);
  // blocking when almost full to prevent re-trigger save
  const static uint64_t kLeftNumTriggerBlock = 1U;
  while (((current_trace_nums - total_saved_nums_) >= (kTraceSaveArraySize - kLeftNumTriggerBlock)) && (!stopped_)) {
  }
  if (stopped_) {  // Drop trace after request stopping
    return;
  }
  const auto index = current_trace_nums % kTraceSaveArraySize;
  trace_array_[index] = std::move(trace_info);
  if (index < kTraceSaveTriggerNum) {
    part1_ready_nums_++;
  } else {
    part2_ready_nums_++;
  }
  // assume kTraceSaveTriggerNum is an aliquot part of kTraceSaveArraySize
  if ((index + 1U) % kTraceSaveTriggerNum == 0) {
    std::unique_lock<std::mutex> lk(mu_);
    ready_part_ = (index < kTraceSaveTriggerNum) ? ReadyPart::A : ReadyPart::B;
    lk.unlock();
    data_ready_var_.notify_all();
  }
}
}  // namespace af
