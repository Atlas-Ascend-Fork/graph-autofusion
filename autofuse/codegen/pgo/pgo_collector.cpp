/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#include "pgo_collector.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <securec.h>
#include <string>
#include <time.h>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "common/ge_common/debug/log.h"
#include "pgo_profapi_adapter.h"
#include "pgo_candidate_identity.h"
#include "pgo_runtime_track.h"
#include "pgo_task_record.h"
#include "pgo_stars_parser.h"

namespace codegen::pgo {
namespace {
constexpr int32_t kSystemModule = 0;
constexpr int32_t kSysCount = 13;
constexpr size_t kReadBufferSize = 2U * 1024U * 1024U;
constexpr size_t kCompactTypeOffset = 4U;
constexpr uint32_t kTaskTrackType = 802;
// Legacy64 records share the funcType field. FFTS+ start/end values are
// defined in MSPTI's channel_data.h and are only counted for diagnosis here.
constexpr uint8_t kFftsBegin = 34U;
constexpr uint8_t kFftsEnd = 35U;
#define PGO_WINDOW_LOG_FORMAT                                                                  \
  "[PGO] graph=%s device=%u stage=%s status=%d mode=%s candidates=%zu expected=%zu valid=%zu " \
  "bytes=%zu acsq=%zu ffts=%zu filtered_events=%zu tracks=%zu"

uint16_t ReadLegacyIdentityField(const uint8_t *data) {
  return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8U);
}

int CompactSink(uint32_t aging_flag, const void *data, uint32_t length) {
  const int ret = DispatchRuntimeTrack(aging_flag, data, length);
  if (ret == PGO_DATA_UNAVAILABLE && data != nullptr && length >= 12U) {
    uint32_t type = 0;
    if (memcpy_s(&type, sizeof(type), static_cast<const unsigned char *>(data) + kCompactTypeOffset, sizeof(type)) !=
        EOK) {
      return PGO_DATA_UNAVAILABLE;
    }
    // Compact callback carries several runtime record kinds.  Only task
    // track records are consumed by the internal collector; other valid
    // records are deliberately ignored.  A task-track queue overflow remains
    // an error and is returned to the profiler.
    if (type != kTaskTrackType) return PGO_SUCCESS;
  }
  return ret;
}

uint64_t HostRealTimeNs() {
  timespec ts{};
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) return 0;
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

class CollectorImpl final : public PgoCollector {
 public:
  ~CollectorImpl() override {
    Finalize();
  }

  int Initialize(uint32_t device_id, void *stream) override {
    device_id_ = device_id;
    const int capability_status =
        ProbePgoCapability(PgoReporterOwnership::kExclusive, device_id_, stream, &capability_);
    if (capability_status != PGO_SUCCESS) {
      return WindowStatus("capability", capability_status);
    }
    const int reporter_status = RegisterRuntimeTrackReporter(PgoReporterOwnership::kExclusive, &QueueRuntimeTrack);
    if (reporter_status != PGO_SUCCESS) {
      return WindowStatus("reporter", reporter_status);
    }
    reporter_registered_ = true;
    const int adapter_status = adapter_.Initialize(&CompactSink, PgoReporterOwnership::kExclusive);
    if (adapter_status != PGO_SUCCESS) {
      return WindowStatus("initialize", adapter_status);
    }
    ClearRuntimeTracks();
    const uint64_t host_before = HostRealTimeNs();
    int64_t start_count = 0;
    const PgoStarsFormat stars_format = ResolveStarsFormat(capability_.platform);
    record_size_ = StarsRecordSize(stars_format);
    if (record_size_ == 0U) {
      return WindowStatus("record-format", PGO_PROFILE_UNSUPPORTED);
    }
    const char *mode = std::getenv("AF_PGO_STARS_DIAGNOSTICS");
    capture_ffts_ = mode != nullptr && stars_format == PgoStarsFormat::kLegacy64 && std::strcmp(mode, "acsq_ffts") == 0;
    diagnose_stars_ = capture_ffts_ || (mode != nullptr && std::strcmp(mode, "acsq") == 0);
    const uint64_t ticks_per_ms = capability_.timestamp_ticks_per_ms;
    const bool frequency_valid = ticks_per_ms != 0U;
    if (frequency_valid) {
      // GetDevStartSysCnt() in MSPTI returns zero on failure and does not
      // reject profiling.  Preserve that behavior for the injected HAL ABI.
      (void)adapter_.GetDeviceInfo(device_id_, kSystemModule, kSysCount, &start_count);
    }
    const uint64_t host_after = HostRealTimeNs();
    if (frequency_valid) {
      time_info_.ticks_per_ms = ticks_per_ms;
      time_info_.start_sys_count = static_cast<uint64_t>(start_count);
      time_info_.start_real_time_ns = static_cast<uint64_t>((static_cast<__uint128_t>(host_before) + host_after) / 2U);
    }
    initialized_ = true;
    return 0;
  }

  int BeginCandidate(const char *candidate_key) override {
    if (!initialized_ || candidate_key == nullptr || *candidate_key == '\0' || candidate_active_) return -1;
    if (batch_finalized_) return -1;
    candidate_key_ = candidate_key;
    launch_sequences_.clear();
    // The generated runner creates one collector per ProfilingBatchProcess.
    // Keep the profiler/channel and all raw activity active across candidates;
    // EndCandidate only commits the candidate boundary.
    if (!profiling_started_) {
      const int status = adapter_.Start(device_id_, ResolvePgoStarsSamplePeriod(capability_.platform), capture_ffts_);
      if (status != PGO_SUCCESS) {
        return WindowStatus("start", status, 1U);
      }
      profiling_started_ = true;
    }
    SetRuntimeTrackCaptureEnabled(true);
    candidate_active_ = true;
    return 0;
  }

  int BeginCandidateTuple(const char *graph_name, const char *tiling_repr, uint64_t workspace_size,
                          uint32_t block_dim) {
    if (graph_name == nullptr || tiling_repr == nullptr || *graph_name == '\0' || *tiling_repr == '\0' ||
        block_dim == 0U) {
      return -1;
    }
    const std::string key = BuildCandidateKey(graph_name, tiling_repr, workspace_size, block_dim);
    return BeginCandidate(key.c_str());
  }

  int RecordLaunch(uint64_t launch_sequence) override {
    if (!candidate_active_ || (!launch_sequences_.empty() && launch_sequence <= launch_sequences_.back())) return -1;
    launch_sequences_.push_back(launch_sequence);
    return 0;
  }

  int EndCandidate() override {
    if (!initialized_ || !candidate_active_) return -1;
    if (launch_sequences_.size() != kPgoMeasureSamples) {
      return WindowStatus("launch-count", PGO_DATA_UNAVAILABLE, pending_candidates_.size() + 1U);
    }
    pending_candidates_.push_back(PgoBatchCandidate{candidate_key_, launch_sequences_});
    candidate_active_ = false;
    launch_sequences_.clear();
    candidate_key_.clear();
    return 0;
  }

  int AbortCandidate() override {
    SetRuntimeTrackCaptureEnabled(false);
    WaitRuntimeTrackCallbacks();
    const int stop_ret = profiling_started_ ? adapter_.Stop(device_id_) : PGO_SUCCESS;
    candidate_active_ = false;
    profiling_started_ = false;
    pending_candidates_.clear();
    candidate_records_.clear();
    candidate_durations_ns_.clear();
    batch_finalized_ = false;
    events_.clear();
    launch_sequences_.clear();
    pending_bytes_.clear();
    runtime_tracks_.clear();
    ClearRuntimeTracks();
    bytes_read_ = 0U;
    acsq_records_ = 0U;
    ffts_records_ = 0U;
    return stop_ret;
  }

  int FinalizeBatch() override {
    if (!initialized_ || candidate_active_ || pending_candidates_.empty() || batch_finalized_) return -1;
    uint32_t flushed = 0;
    const int flush_ret = adapter_.FlushStarsChannel(device_id_, &flushed);
    const bool drain_ok = flush_ret == PGO_SUCCESS && DrainRecords(flushed);
    SetRuntimeTrackCaptureEnabled(false);
    WaitRuntimeTrackCallbacks();
    DrainRuntimeTracks(&runtime_tracks_);
    const int stop_ret = profiling_started_ ? adapter_.Stop(device_id_) : PGO_SUCCESS;
    profiling_started_ = false;
    if (stop_ret != PGO_SUCCESS) return WindowStatus("stop", stop_ret);
    if (flush_ret != PGO_SUCCESS) return WindowStatus("flush", flush_ret);
    if (!drain_ok) return WindowStatus("read", PGO_DATA_UNAVAILABLE);
    if (RuntimeTrackOverflowed()) return WindowStatus("track-overflow", PGO_DATA_UNAVAILABLE);
    if (!ParsePendingRecords(IsStreamExpandEnabled())) return WindowStatus("decode", PGO_DATA_UNAVAILABLE);
    if (!runtime_tracks_.empty()) {
      if (!FilterEventsByRuntimeTracks()) return WindowStatus("track-filter", PGO_DATA_UNAVAILABLE);
    } else if (!CanUseSerialPgoEvents(events_)) {
      return WindowStatus("serial-filter", PGO_DATA_UNAVAILABLE);
    }
    if (PairPgoTaskEventsBatch(events_, pending_candidates_, time_info_, &candidate_records_) != 0) {
      return WindowStatus("pair", PGO_DATA_UNAVAILABLE);
    }
    candidate_durations_ns_.clear();
    candidate_durations_ns_.reserve(candidate_records_.size());
    for (const auto &records : candidate_records_) {
      uint64_t duration_ns = 0U;
      if (CalculatePgoDurationNs(records, &duration_ns) != 0 || duration_ns == 0U) {
        return WindowStatus("timestamp", PGO_DATA_UNAVAILABLE);
      }
      candidate_durations_ns_.push_back(duration_ns);
    }
    batch_finalized_ = true;
    return diagnose_stars_ ? WindowStatus("complete", PGO_SUCCESS) : PGO_SUCCESS;
  }

  int GetDurationNs(uint64_t *duration_ns) override {
    return GetCandidateDurationNs(0U, duration_ns);
  }

  int GetCandidateDurationNs(uint32_t candidate_index, uint64_t *duration_ns) override {
    if (!batch_finalized_ || duration_ns == nullptr || candidate_index >= candidate_durations_ns_.size()) return -1;
    *duration_ns = candidate_durations_ns_[candidate_index];
    return 0;
  }

  void Finalize() override {
    if (reporter_registered_) {
      SetRuntimeTrackCaptureEnabled(false);
    }
    if (initialized_ && profiling_started_) (void)adapter_.Stop(device_id_);
    adapter_.Finalize();
    if (reporter_registered_) {
      WaitRuntimeTrackCallbacks();
      (void)UnregisterRuntimeTrackReporter();
      reporter_registered_ = false;
    }
    candidate_active_ = false;
    initialized_ = false;
    profiling_started_ = false;
    runtime_tracks_.clear();
  }

 private:
  int WindowStatus(const char *stage, int status, size_t candidates = 0U) const {
    if (candidates == 0U) {
      candidates = pending_candidates_.size();
    }
    const size_t expected = candidates * kPgoMeasureSamples;
    const size_t valid = candidate_durations_ns_.size() * kPgoMeasureSamples;
    const char *mode = capture_ffts_ ? "acsq_ffts" : "acsq";
    const char *graph =
        pending_candidates_.empty() ? candidate_key_.c_str() : pending_candidates_.front().candidate_key.c_str();
    if (status == PGO_SUCCESS) {
      GELOGI(PGO_WINDOW_LOG_FORMAT, graph, device_id_, stage, status, mode, candidates, expected, valid, bytes_read_,
             acsq_records_, ffts_records_, events_.size(), runtime_tracks_.size());
    } else {
      GELOGW(PGO_WINDOW_LOG_FORMAT, graph, device_id_, stage, status, mode, candidates, expected, valid, bytes_read_,
             acsq_records_, ffts_records_, events_.size(), runtime_tracks_.size());
    }
    return status;
  }

  void LogLegacyIdentity(const char *source, uint16_t raw_stream, uint16_t raw_task, bool stream_expand,
                         uint16_t task_type) const {
    uint32_t stream = 0U;
    uint32_t task = 0U;
    const bool resolved = ResolveStarsTaskIdentity(raw_stream, raw_task, stream_expand, false, &stream, &task);
    GELOGI("[PGO] %s device=%u task_type=%u raw_stream=%u raw_task=%u resolved=%d stream=%u task=%u", source,
           device_id_, static_cast<unsigned>(task_type), static_cast<unsigned>(raw_stream),
           static_cast<unsigned>(raw_task), static_cast<int>(resolved), stream, task);
  }

  struct IdentityHash {
    size_t operator()(const std::tuple<uint32_t, uint32_t, uint32_t> &value) const {
      const auto [device, stream, task] = value;
      return (static_cast<size_t>(device) << 32U) ^ (static_cast<size_t>(stream) << 16U) ^ task;
    }
  };

  bool FilterEventsByRuntimeTracks() {
    if (runtime_tracks_.empty()) return false;
    using Identity = std::tuple<uint32_t, uint32_t, uint32_t>;
    std::unordered_set<Identity, IdentityHash> identities;
    if (diagnose_stars_ && record_size_ == 64U) {
      const auto &track = runtime_tracks_.front();
      LogLegacyIdentity("runtime track", static_cast<uint16_t>(track.stream_id), static_cast<uint16_t>(track.task_info),
                        IsStreamExpandEnabled(), static_cast<uint16_t>(track.task_type));
    }
    for (const auto &track : runtime_tracks_) {
      if (!IsKernelTaskType(track.task_type)) continue;
      uint32_t stream_id = 0;
      uint32_t task_id = 0;
      if (!ResolveStarsTaskIdentity(static_cast<uint16_t>(track.stream_id), track.task_info, IsStreamExpandEnabled(),
                                    record_size_ == 32U, &stream_id, &task_id)) {
        continue;
      }
      identities.emplace(track.device_id, stream_id, task_id);
    }
    if (identities.empty()) return false;
    std::vector<PgoStarsEvent> filtered;
    filtered.reserve(events_.size());
    for (const auto &event : events_) {
      if (identities.find(Identity{event.device_id, event.stream_id, event.task_id}) != identities.end()) {
        filtered.push_back(event);
      }
    }
    events_.swap(filtered);
    return !events_.empty();
  }

  bool DrainRecords(uint32_t flushed_bytes) {
    std::vector<uint8_t> buffer(kReadBufferSize);
    pending_bytes_.reserve(record_size_);
    size_t drained_bytes = 0U;
    size_t idle_polls = 0U;
    // MSPTI waits for the amount reported by halProfDataFlush and then drains
    // until two consecutive empty reads.  A bounded loop prevents a broken
    // driver from hanging the PGO worker indefinitely.
    for (size_t attempt = 0U; attempt < 64U && (drained_bytes < flushed_bytes || idle_polls < 2U); ++attempt) {
      const int poll_ret = adapter_.PollStarsChannel(1);
      if (poll_ret != PGO_SUCCESS) {
        pending_bytes_.clear();
        return false;
      }
      const int size = adapter_.ReadStarsChannel(device_id_, buffer);
      if (size < 0) {
        pending_bytes_.clear();
        return false;
      }
      if (size == 0) {
        ++idle_polls;
        continue;
      }
      if (static_cast<size_t>(size) > buffer.size()) {
        pending_bytes_.clear();
        return false;
      }
      idle_polls = 0U;
      drained_bytes += static_cast<size_t>(size);
      bytes_read_ = drained_bytes;
      pending_bytes_.insert(pending_bytes_.end(), buffer.begin(), buffer.begin() + size);
    }
    // Reaching the polling bound is not equivalent to draining the channel.
    // If flush reported bytes that were not read, or the channel kept
    // producing data without two quiescent reads, accepting the records here
    // would silently turn a truncated sample set into a valid timing result.
    if (drained_bytes < flushed_bytes || idle_polls < 2U) {
      events_.clear();
      pending_bytes_.clear();
      return false;
    }
    // A partial record is never silently accepted; it indicates truncated
    // driver data and will make pairing fail deterministically.
    if (pending_bytes_.size() % record_size_ != 0U) {
      events_.clear();
      pending_bytes_.clear();
      return false;
    }
    return true;
  }

  bool ParsePendingRecords(bool stream_expand) {
    events_.clear();
    for (size_t offset = 0U; offset < pending_bytes_.size(); offset += record_size_) {
      const auto *record = pending_bytes_.data() + offset;
      const uint8_t func_type = record[0] & 0x3fU;
      if (func_type <= 1U) {
        ++acsq_records_;
      } else if (record_size_ == 64U && (func_type == kFftsBegin || func_type == kFftsEnd)) {
        ++ffts_records_;
      }
      if (diagnose_stars_ && record_size_ == 64U) {
        const bool ffts = func_type == kFftsBegin || func_type == kFftsEnd;
        if ((ffts && ffts_records_ == 1U) || (func_type <= 1U && acsq_records_ == 1U)) {
          LogLegacyIdentity(ffts ? "FFTS+ record" : "ACSQ record", ReadLegacyIdentityField(record + 4U),
                            ReadLegacyIdentityField(record + 6U), stream_expand,
                            ReadLegacyIdentityField(record) >> 10U);
        }
      }
      PgoStarsEvent event{};
      if (ParseStarsRecord(record, record_size_, device_id_, &event, stream_expand)) {
        events_.push_back(event);
      }
    }
    pending_bytes_.clear();
    return !events_.empty();
  }

  uint32_t device_id_ = 0;
  bool initialized_ = false;
  bool reporter_registered_ = false;
  bool capture_ffts_ = false;
  bool diagnose_stars_ = false;
  bool profiling_started_ = false;
  bool candidate_active_ = false;
  bool batch_finalized_ = false;
  size_t record_size_ = 64;
  size_t bytes_read_ = 0U;
  size_t acsq_records_ = 0U;
  size_t ffts_records_ = 0U;
  std::string candidate_key_;
  std::vector<PgoBatchCandidate> pending_candidates_;
  std::vector<std::vector<PgoTaskRecord>> candidate_records_;
  std::vector<uint64_t> candidate_durations_ns_;
  PgoDeviceTimeInfo time_info_;
  PgoCapability capability_;
  PgoProfApiAdapter adapter_;
  std::vector<uint8_t> pending_bytes_;
  std::vector<PgoStarsEvent> events_;
  std::vector<uint64_t> launch_sequences_;
  std::vector<PgoRuntimeTrack> runtime_tracks_;
};
#undef PGO_WINDOW_LOG_FORMAT
}  // namespace

extern "C" int AutofusePgoCollectorCreate(uint32_t device_id, void *stream, void **collector) {
  if (collector == nullptr) return -1;
  auto *impl = new (std::nothrow) CollectorImpl();
  if (impl == nullptr) {
    delete impl;
    return -1;
  }
  const int initialize_status = impl->Initialize(device_id, stream);
  if (initialize_status != 0) {
    delete impl;
    return initialize_status;
  }
  *collector = impl;
  return 0;
}
extern "C" int AutofusePgoCollectorBegin(void *collector, const char *candidate_key) {
  return collector == nullptr ? -1 : static_cast<CollectorImpl *>(collector)->BeginCandidate(candidate_key);
}
extern "C" int AutofusePgoCollectorBeginCandidate(void *collector, const char *graph_name, const char *tiling_repr,
                                                  uint64_t workspace_size, uint32_t block_dim) {
  return collector == nullptr ? -1
                              : static_cast<CollectorImpl *>(collector)->BeginCandidateTuple(graph_name, tiling_repr,
                                                                                             workspace_size, block_dim);
}
extern "C" int AutofusePgoCollectorRecordLaunch(void *collector, uint64_t launch_sequence) {
  return collector == nullptr ? -1 : static_cast<CollectorImpl *>(collector)->RecordLaunch(launch_sequence);
}
extern "C" int AutofusePgoCollectorEnd(void *collector) {
  return collector == nullptr ? -1 : static_cast<CollectorImpl *>(collector)->EndCandidate();
}
extern "C" int AutofusePgoCollectorAbort(void *collector) {
  return collector == nullptr ? -1 : static_cast<CollectorImpl *>(collector)->AbortCandidate();
}
extern "C" int AutofusePgoCollectorFinalizeBatch(void *collector) {
  return collector == nullptr ? -1 : static_cast<CollectorImpl *>(collector)->FinalizeBatch();
}
extern "C" int AutofusePgoCollectorGetDurationNs(void *collector, uint64_t *duration_ns) {
  return collector == nullptr ? -1 : static_cast<CollectorImpl *>(collector)->GetDurationNs(duration_ns);
}
extern "C" int AutofusePgoCollectorGetCandidateDurationNs(void *collector, uint32_t candidate_index,
                                                          uint64_t *duration_ns) {
  return collector == nullptr
             ? -1
             : static_cast<CollectorImpl *>(collector)->GetCandidateDurationNs(candidate_index, duration_ns);
}
extern "C" void AutofusePgoCollectorDestroy(void *collector) {
  delete static_cast<CollectorImpl *>(collector);
}
}  // namespace codegen::pgo
