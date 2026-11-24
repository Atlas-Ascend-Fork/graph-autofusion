/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tracing_recorder.h"
#include <map>
#include <string>
#include "mmpa/mmpa_api.h"
#include "atrace_pub_af.h"

namespace ge {
namespace {
constexpr char_t const *kGEInitTraceHandleName = "GE_Init_Trace";
constexpr char_t const *kGECompileTraceHandleName = "GE_Compile_Trace";
constexpr char_t const *kAutoFuseBackendTraceHandleName = "AutoFuse_Backend_Trace";
constexpr char_t const *kGELoadTraceHandleName = "GE_Load_Trace";
constexpr char_t const *kGeFinalizeEventName = "GE_Finalize_Event";
constexpr char_t const *kInvalidHandleName = "Invalid_Trace";
std::map<TracingModule, std::string> kModuleToHandleName = {
    {TracingModule::kCANNInitialize, kGEInitTraceHandleName},
    {TracingModule::kModelCompile, kGECompileTraceHandleName},
    {TracingModule::kAutoFuseBackend, kAutoFuseBackendTraceHandleName},
    {TracingModule::kModelLoad, kGELoadTraceHandleName},
};
const std::string &GetModuleHandleName(const TracingModule module) {
  const auto &iter = kModuleToHandleName.find(module);
  if (iter != kModuleToHandleName.end()) {
    return iter->second;
  }
  static std::string invalid_name(kInvalidHandleName);
  return invalid_name;
}
}  // namespace
std::string TracingRecorder::GetHandleName() const {
  return GetModuleHandleName(module_) + std::to_string(handles_.size());
}

TracingRecorder::TracingRecorder(const TracingModule module) {
  module_ = module;
  Initialize();
}

void TracingRecorder::Initialize() {
  const auto &handle_name = GetHandleName();
  if (handle_name.find(kInvalidHandleName) != std::string::npos) {
    GELOGW("Tracing recorder init failed, invalid module[%u].", module_);
    return;
  }
  if (static_cast<int32_t>(handles_.size()) >= kMaxAtracingProfilingHandleSzie) {
    GELOGW("Tracing recorder init failed, handle size[%zu] is larger than max handle size[%d].", handles_.size(),
           kMaxAtracingProfilingHandleSzie);
    return;
  }
  TraceAttr attr{};
  attr.exitSave = true;  // 退出进程时落盘
  attr.msgSize = kMaxAtracingProfilingMsgSize;
  attr.msgNum = kMaxAtracingProfilingRecordNum;
  GELOGD("Init trace recorder:exitSave[%d], msgSize[%d], msgNum[%d]", attr.exitSave, attr.msgSize, attr.msgNum);
  auto handle = AtraceCreateWithAttr(TracerType::TRACER_TYPE_SCHEDULE, handle_name.c_str(), &attr);
  std::string event_name =
      std::string(kGeFinalizeEventName)
          .append("_")
          .append(std::to_string(mmGetPid()).append("_").append(std::to_string(static_cast<int32_t>(module_))))
          .append("_");
  TraEventHandle finalize_event_handle = kInvalidHandle;
  // 首次event和绑定handle超过上限的场景需要重新创建event handle
  if (finalize_event_handles_.empty() || (++event_bind_num_ > kMaxEventBindNum)) {
    event_name.append(std::to_string(finalize_event_handles_.size() + 1UL));
    finalize_event_handle = AtraceEventCreate(event_name.c_str());
    finalize_event_handles_.emplace_back(finalize_event_handle);
    event_bind_num_ = 1;
    GELOGI("Create event handle[%s] to module[%s].", event_name.c_str(), handle_name.c_str());
  } else {
    event_name.append(std::to_string(finalize_event_handles_.size()));
  }
  if ((handle < 0) || (finalize_event_handle < 0)) {
    GELOGW("Create trace handle or event handle failed for module[%s].", handle_name.c_str());
  } else {
    (void)AtraceEventBindTrace(finalize_event_handle, handle);
    GELOGI("Create trace handle and event handle %s for module[%s] successfully.", event_name.c_str(),
           handle_name.c_str());
    is_ready_ = true;
  }
  handles_.emplace_back(handle);
}

void TracingRecorder::SubmitTraceMsgs(const TracingRecord *tracing_record) {
  AtracingReporter reporter(handles_.back(), tracing_record);
  if (reporter.Report() != af::SUCCESS) {
    GELOGW("Report failed of module[%s] for record[%s].", GetHandleName().c_str(), tracing_record->Debug().c_str());
  }
}

void TracingRecorder::Report(const TracingRecord *tracing_record, int64_t record_num) {
  for (int64_t i = 0L; i < record_num; ++i) {
    SubmitTraceMsgs(&tracing_record[i]);
  }
}

void TracingRecorder::Finalize() {
  if (is_ready_) {
    for (const auto &finalize_event_handle : finalize_event_handles_) {
      (void)AtraceEventReportSync(finalize_event_handle);
      AtraceEventDestroy(finalize_event_handle);
    }
    for (const auto &handle : handles_) {
      AtraceDestroy(handle);
    }
    is_ready_ = false;
  }
  finalize_event_handles_.clear();
  handles_.clear();
  event_bind_num_ = 0;
  GELOGI("Event handle and handle of [%s] have been destroyed.", GetHandleName().c_str());
}

void TracingRecorder::Report() {
  if (module_ >= TracingModule::kTracingModuleEnd) {
    GELOGW("Report tracing record failed, as module id is %u", module_);
  }
  // 保证进程内Report多并发安全
  std::lock_guard guard(mu_);
  auto records_num = static_cast<int32_t>(records_.size());
  int32_t offset = 0;
  while (records_num > 0) {
    if (!is_ready_) {
      GELOGI("%d msgs have been left, need reinit and report again.", records_num);
      Initialize();
    }
    if (is_ready_) {
      auto loop_record_num = std::min(records_num, kMaxAtracingProfilingRecordNum);
      Report(&records_[offset], loop_record_num);
      records_num -= loop_record_num;
      offset += loop_record_num;
      GELOGI("%d msgs have been recorded, and the next step is to persist them to disk of module[%s].", loop_record_num,
             GetHandleName().c_str());
      is_ready_ = false;
    } else {
      break;
    }
  }
  records_.clear();
}

TracingRecorder::~TracingRecorder() {
  GELOGI("Enter tracing recorder destructor, module[%s]", GetHandleName().c_str());
  Report();
  Finalize();
}

void TracingRecorder::RecordDuration(const std::vector<std::string> &tracing_msg, uint64_t start, uint64_t duration) {
  std::lock_guard guard(mu_);
  if (module_ >= TracingModule::kTracingModuleEnd) {
    GELOGW("Record tracing record failed, as module id is %u", module_);
  }
  auto record = RecordMsgs(tracing_msg, TracingEvent::kEventDuration);
  if (record != nullptr) {
    record->start = start;
    record->duration = duration;
  }
  GELOGD("%s has been recorded to atrace log.", record->Debug().c_str());
}

TracingRecord *TracingRecorder::RecordMsgs(const std::vector<std::string> &tracing_msg, const TracingEvent ev) {
  TracingRecord record{};
  record.pid = mmGetPid();
  record.thread = mmGetTid();
  record.tracing_msgs = tracing_msg;
  record.event = static_cast<uint8_t>(ev);
  records_.emplace_back(record);
  return &records_.back();
}
}  // namespace ge
