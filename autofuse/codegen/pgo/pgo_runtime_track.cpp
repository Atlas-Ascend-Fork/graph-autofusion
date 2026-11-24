/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#include "pgo_runtime_track.h"

#include <array>
#include <condition_variable>
#include <mutex>
#include <securec.h>
#include <string>
#include <unordered_set>

#include "pgo_error.h"

namespace codegen::pgo {
namespace {
constexpr uint16_t kMagic = 0x5a5a;
constexpr uint32_t kTaskTrackType = 802;
constexpr uint32_t kStreamExpandType = 804;
constexpr uint32_t kRuntimeDataLength = 40;
constexpr uint32_t kStreamExpandDataLength = 4;
constexpr size_t kCompactInfoLength = 64;
// A 1000-candidate batch produces at most 20,000 kernel task tracks. Keep
// callback insertion allocation-free while covering the complete batch.
constexpr size_t kMaxQueuedTracks = kRuntimeTrackQueueCapacity;
constexpr uint16_t kRuntimeLevel = 5000;

#pragma pack(push, 1)
struct MsprofRuntimeTrack {
  uint16_t device_id;
  uint16_t stream_id;
  uint32_t task_info;
  uint64_t task_type;
  uint64_t kernel_name;
  uint8_t reserved[16];
};

struct MsprofCompactInfo {
  uint16_t magic;
  uint16_t level;
  uint32_t type;
  uint32_t thread_id;
  uint32_t data_len;
  uint64_t timestamp;
  uint8_t data[kRuntimeDataLength];
};
#pragma pack(pop)

static_assert(sizeof(MsprofRuntimeTrack) == 40, "CANN runtime track ABI changed");
static_assert(sizeof(MsprofCompactInfo) == kCompactInfoLength, "CANN compact info ABI changed");

std::mutex g_reporter_mutex;
std::condition_variable g_reporter_cv;
RuntimeTrackSink g_sink = nullptr;
std::array<PgoRuntimeTrack, kMaxQueuedTracks> g_tracks{};
size_t g_track_head = 0;
size_t g_track_count = 0;
size_t g_active_callbacks = 0;
bool g_capture_enabled = false;
bool g_stream_expand = false;
bool g_runtime_type_info_available = false;
bool g_track_overflow = false;
std::unordered_set<uint32_t> g_kernel_task_types;

RuntimeTrackSink AdmitCallback() {
  std::lock_guard<std::mutex> lock(g_reporter_mutex);
  if (g_sink == nullptr) return nullptr;
  ++g_active_callbacks;
  return g_sink;
}

void ReleaseCallback() {
  std::lock_guard<std::mutex> lock(g_reporter_mutex);
  if (g_active_callbacks == 0U) return;
  --g_active_callbacks;
  if (g_active_callbacks == 0U) g_reporter_cv.notify_all();
}
}  // namespace

int DecodeRuntimeTrack(uint32_t aging_flag, const void *data, uint32_t length, PgoRuntimeTrack *track) {
  if (data == nullptr || track == nullptr) return PGO_INVALID_ARGUMENT;
  if (aging_flag > 1U) return PGO_DATA_UNAVAILABLE;
  if (length != sizeof(MsprofCompactInfo)) return PGO_DATA_UNAVAILABLE;
  MsprofCompactInfo compact{};
  if (memcpy_s(&compact, sizeof(compact), data, sizeof(compact)) != EOK) return PGO_DATA_UNAVAILABLE;
  if (compact.magic != kMagic || compact.level != kRuntimeLevel || compact.data_len != kRuntimeDataLength) {
    return PGO_DATA_UNAVAILABLE;
  }
  if (compact.type != kTaskTrackType) return PGO_DATA_UNAVAILABLE;
  MsprofRuntimeTrack runtime{};
  if (memcpy_s(&runtime, sizeof(runtime), compact.data, sizeof(runtime)) != EOK) return PGO_DATA_UNAVAILABLE;
  constexpr uint16_t kMaxDeviceId = 64U;
  if (runtime.device_id >= kMaxDeviceId) {
    return PGO_DATA_UNAVAILABLE;
  }
  track->device_id = runtime.device_id;
  track->stream_id = runtime.stream_id;
  track->task_info = runtime.task_info;
  track->task_type = runtime.task_type;
  track->kernel_name_hash = runtime.kernel_name;
  track->aging = aging_flag != 0;
  return PGO_SUCCESS;
}

int RegisterRuntimeTrackReporter(PgoReporterOwnership ownership, RuntimeTrackSink sink) {
  if (sink == nullptr) return PGO_INVALID_ARGUMENT;
  if (ownership != PgoReporterOwnership::kExclusive) return PGO_REPORTER_BUSY;
  std::lock_guard<std::mutex> lock(g_reporter_mutex);
  if (g_sink != nullptr) return PGO_REPORTER_BUSY;
  g_sink = sink;
  g_capture_enabled = false;
  g_stream_expand = false;
  g_track_head = 0;
  g_track_count = 0;
  g_track_overflow = false;
  g_runtime_type_info_available = false;
  g_kernel_task_types.clear();
  return PGO_SUCCESS;
}

int UnregisterRuntimeTrackReporter() {
  std::unique_lock<std::mutex> lock(g_reporter_mutex);
  g_capture_enabled = false;
  g_reporter_cv.wait(lock, [] { return g_active_callbacks == 0; });
  // Keep the ownership slot occupied until all callbacks admitted by the
  // previous session have left.  Otherwise a new session could register
  // while an old callback is still executing and append into its queue.
  g_sink = nullptr;
  g_track_head = 0;
  g_track_count = 0;
  g_track_overflow = false;
  g_runtime_type_info_available = false;
  g_kernel_task_types.clear();
  return PGO_SUCCESS;
}

int32_t QueueRuntimeTrack(const PgoRuntimeTrack &track) {
  std::lock_guard<std::mutex> lock(g_reporter_mutex);
  if (!g_capture_enabled) return PGO_SUCCESS;
  if (g_track_count >= kMaxQueuedTracks) {
    g_track_overflow = true;
    return PGO_DATA_UNAVAILABLE;
  }
  const size_t tail = (g_track_head + g_track_count) % kMaxQueuedTracks;
  g_tracks[tail] = track;
  ++g_track_count;
  return PGO_SUCCESS;
}

void SetRuntimeTrackCaptureEnabled(bool enabled) {
  std::lock_guard<std::mutex> lock(g_reporter_mutex);
  g_capture_enabled = enabled;
}

bool IsStreamExpandEnabled() {
  std::lock_guard<std::mutex> lock(g_reporter_mutex);
  return g_stream_expand;
}

void ClearRuntimeTracks() {
  std::lock_guard<std::mutex> lock(g_reporter_mutex);
  g_track_head = 0;
  g_track_count = 0;
  g_track_overflow = false;
}

void WaitRuntimeTrackCallbacks() {
  std::unique_lock<std::mutex> lock(g_reporter_mutex);
  g_reporter_cv.wait(lock, [] { return g_active_callbacks == 0; });
}

void DrainRuntimeTracks(std::vector<PgoRuntimeTrack> *tracks) {
  if (tracks == nullptr) return;
  std::lock_guard<std::mutex> lock(g_reporter_mutex);
  tracks->clear();
  tracks->reserve(g_track_count);
  for (size_t index = 0; index < g_track_count; ++index) {
    tracks->push_back(g_tracks[(g_track_head + index) % kMaxQueuedTracks]);
  }
  g_track_head = 0;
  g_track_count = 0;
}

int32_t DispatchRuntimeTrack(uint32_t aging_flag, const void *data, uint32_t length) {
  RuntimeTrackSink sink = AdmitCallback();
  if (sink == nullptr) return PGO_SUCCESS;

  if (data != nullptr && length == sizeof(MsprofCompactInfo)) {
    MsprofCompactInfo compact{};
    if (memcpy_s(&compact, sizeof(compact), data, sizeof(compact)) != EOK) {
      ReleaseCallback();
      return PGO_DATA_UNAVAILABLE;
    }
    if (compact.magic == kMagic && compact.level == kRuntimeLevel && compact.type == kStreamExpandType) {
      // The control payload is MsprofStreamExpandSpecInfo, a four-byte entry
      // carried in the same compact envelope as the forty-byte task track.
      if (compact.data_len != kStreamExpandDataLength && compact.data_len != kRuntimeDataLength) {
        ReleaseCallback();
        return PGO_DATA_UNAVAILABLE;
      }
      if (compact.data[0] > 1U) {
        ReleaseCallback();
        return PGO_DATA_UNAVAILABLE;
      }
      {
        std::lock_guard<std::mutex> lock(g_reporter_mutex);
        g_stream_expand = compact.data[0] == 1U;
      }
      ReleaseCallback();
      return PGO_SUCCESS;
    }
  }
  PgoRuntimeTrack track;
  const int decode_ret = DecodeRuntimeTrack(aging_flag, data, length, &track);
  if (decode_ret != PGO_SUCCESS) {
    ReleaseCallback();
    return decode_ret;
  }
  const int32_t ret = sink(track);
  ReleaseCallback();
  return ret;
}

int32_t DispatchRuntimeTypeInfo(uint16_t level, uint32_t type_id, const char *name, size_t length) {
  if (name == nullptr || length == 0U) return PGO_INVALID_ARGUMENT;
  if (level != kRuntimeLevel) return PGO_SUCCESS;
  const std::string type_name(name, length);
  if (AdmitCallback() == nullptr) return PGO_SUCCESS;
  {
    std::lock_guard<std::mutex> lock(g_reporter_mutex);
    if (type_name.rfind("KERNEL", 0U) == 0U) {
      g_kernel_task_types.insert(type_id);
    }
  }
  ReleaseCallback();
  return PGO_SUCCESS;
}

void SetRuntimeTypeInfoAvailable(bool available) {
  std::lock_guard<std::mutex> lock(g_reporter_mutex);
  g_runtime_type_info_available = available;
}

bool IsKernelTaskType(uint64_t task_type) {
  std::lock_guard<std::mutex> lock(g_reporter_mutex);
  if (!g_runtime_type_info_available || g_kernel_task_types.empty()) return false;
  return g_kernel_task_types.find(static_cast<uint32_t>(task_type)) != g_kernel_task_types.end();
}

bool RuntimeTrackOverflowed() {
  std::lock_guard<std::mutex> lock(g_reporter_mutex);
  return g_track_overflow;
}

}  // namespace codegen::pgo
