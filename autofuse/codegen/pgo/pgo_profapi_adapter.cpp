/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#include "pgo_profapi_adapter.h"

#include <dlfcn.h>
#include <link.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "pgo_runtime_track.h"

namespace codegen::pgo {
namespace {
constexpr int32_t kCompactCallbackType = 4;
constexpr int32_t kTypeInfoCallbackType = 11;
constexpr uint64_t kProfTaskTime = 0x00000800ULL;
constexpr uint64_t kProfRuntimeTrace = 0x0000004000000ULL;
constexpr uint32_t kCommandStart = 1;
constexpr uint32_t kCommandStop = 2;
constexpr uint32_t kStarsSocLogChannel = 50;
constexpr int32_t kSystemModule = 0;
constexpr int32_t kVersionInfo = 1;
constexpr uint32_t kProfileTypeTs = 0;
constexpr uint32_t kProfileRealTime = 1;
constexpr uint32_t kTaskEnable = 1;
constexpr uint32_t kTaskDisable = 2;
constexpr uint32_t kMaxChannelCount = 160;

struct PgoChannelInfo {
  char name[32];
  uint32_t type;
  uint32_t id;
};
struct PgoChannelList {
  uint32_t chip_type;
  uint32_t channel_num;
  PgoChannelInfo channels[kMaxChannelCount];
};
struct PgoStartPara {
  uint32_t channel_type;
  uint32_t sample_period;
  uint32_t real_time;
  void *user_data;
  uint32_t user_data_size;
};
struct PgoPollInfo {
  uint32_t device_id;
  uint32_t channel_id;
};
struct PgoStarsSocLogConfig {
  uint32_t acsq_task;
  uint32_t acc_pmu;
  uint32_t cdqm_reg;
  uint32_t dvpp_vpc_block;
  uint32_t dvpp_jpegd_block;
  uint32_t dvpp_jpede_block;
  uint32_t ffts_thread_task;
  uint32_t ffts_block;
  uint32_t sdma_dmu;
  uint32_t tag;
  uint32_t block_shink_flag;
};

static_assert(sizeof(PgoChannelInfo) == 40, "prof channel ABI changed");
static_assert(sizeof(PgoStarsSocLogConfig) == 44, "STARS config ABI changed");
static_assert(sizeof(PgoStartPara) == 32, "prof start ABI changed");

template <size_t PathCount>
void *OpenLibrary(const char *name, const std::array<const char *, PathCount> &relative_paths) {
  const int dlopen_flags = PgoDlopenFlags();
  if (void *handle = dlopen(name, dlopen_flags); handle != nullptr) return handle;
  const char *root = std::getenv("ASCEND_HOME_PATH");
  if (root == nullptr) root = std::getenv("ASCEND_TOOLKIT_HOME");
  if (root == nullptr) return nullptr;
  for (const char *relative : relative_paths) {
    const std::string path = std::string(root) + "/" + relative;
    if (void *handle = dlopen(path.c_str(), dlopen_flags); handle != nullptr) return handle;
  }
  return nullptr;
}

template <typename T>
T LoadSymbol(void *handle, const char *name) {
  return handle == nullptr ? nullptr : reinterpret_cast<T>(dlsym(handle, name));
}

int CheckLoadedObject(struct dl_phdr_info *info, size_t, void *data) {
  if (info == nullptr || info->dlpi_name == nullptr || data == nullptr) return 0;
  const char *name = info->dlpi_name;
  const char *base = std::strrchr(name, '/');
  base = base == nullptr ? name : base + 1;
  static constexpr size_t kMsptiNameLength = 11U;
  const volatile unsigned char kMsptiName[kMsptiNameLength] = {'l', 'i', 'b', 'm', 's', 'p', 't', 'i', '.', 's', 'o'};
  bool is_mspti = std::strlen(base) >= kMsptiNameLength;
  if (is_mspti) {
    for (size_t index = 0; index < kMsptiNameLength; ++index) {
      if (static_cast<unsigned char>(base[index]) != kMsptiName[index]) {
        is_mspti = false;
        break;
      }
    }
  }
  if (is_mspti && (base[kMsptiNameLength] == '\0' || base[kMsptiNameLength] == '.')) {
    *static_cast<bool *>(data) = true;
    return 1;
  }
  return 0;
}

bool IsMsptiLoaded() {
  bool loaded = false;
  (void)dl_iterate_phdr(CheckLoadedObject, &loaded);
  return loaded;
}

}  // namespace

int PgoDlopenFlags() {
  int flags = RTLD_NOW | RTLD_LOCAL;
#if defined(RTLD_NODELETE)
  flags |= RTLD_NODELETE;
#endif
  return flags;
}

bool IsPgoFlushUnsupportedStatus(int32_t status) {
  return status == static_cast<int32_t>(0xfffe);
}

int CombinePgoStopStatus(int channel_status, int command_status) {
  return channel_status != PGO_SUCCESS ? channel_status : command_status;
}

int NormalizePgoPollStatus(int poll_status) {
  // prof_channel_poll returns the number of readable channels (zero means
  // timeout), not a boolean status. Only negative values indicate failure.
  return poll_status < 0 ? PGO_CHANNEL_UNAVAILABLE : PGO_SUCCESS;
}

PgoProfApiAdapter::~PgoProfApiAdapter() {
  Finalize();
}

int PgoProfApiAdapter::Initialize(CompactCallback callback, PgoReporterOwnership ownership) {
  if (callback == nullptr) return PGO_INVALID_ARGUMENT;
  if (ownership != PgoReporterOwnership::kExclusive || IsMsptiLoaded()) return PGO_REPORTER_BUSY;
  if (initialized_) return PGO_SUCCESS;
  static constexpr std::array kProfapiPaths = {"x86_64-linux/lib64/libprofapi.so", "aarch64-linux/lib64/libprofapi.so",
                                               "lib64/libprofapi.so"};
  static constexpr std::array kHalPaths = {"x86_64-linux/devlib/libascend_hal.so",
                                           "x86_64-linux/lib64/device/lib64/libascend_hal.so",
                                           "x86_64-linux/lib64/libascend_hal.so",
                                           "aarch64-linux/devlib/libascend_hal.so",
                                           "aarch64-linux/lib64/device/lib64/libascend_hal.so",
                                           "aarch64-linux/lib64/libascend_hal.so"};
  profapi_handle_ = OpenLibrary("libprofapi.so", kProfapiPaths);
  hal_handle_ = OpenLibrary("libascend_hal.so", kHalPaths);
  if (profapi_handle_ == nullptr || hal_handle_ == nullptr) {
    Finalize();
    return PGO_PROFAPI_UNAVAILABLE;
  }
  set_command_ = LoadSymbol<decltype(set_command_)>(profapi_handle_, "profSetProfCommand");
  register_callback_ = LoadSymbol<decltype(register_callback_)>(profapi_handle_, "MsprofRegisterProfileCallback");
  get_channels_ = LoadSymbol<decltype(get_channels_)>(hal_handle_, "prof_drv_get_channels");
  start_channel_ = LoadSymbol<decltype(start_channel_)>(hal_handle_, "prof_drv_start");
  stop_channel_ = LoadSymbol<decltype(stop_channel_)>(hal_handle_, "prof_stop");
  read_channel_ = LoadSymbol<decltype(read_channel_)>(hal_handle_, "prof_channel_read");
  poll_channel_ = LoadSymbol<decltype(poll_channel_)>(hal_handle_, "prof_channel_poll");
  flush_channel_ = LoadSymbol<decltype(flush_channel_)>(hal_handle_, "halProfDataFlush");
  get_device_info_ = LoadSymbol<decltype(get_device_info_)>(hal_handle_, "halGetDeviceInfo");
  if (set_command_ == nullptr || register_callback_ == nullptr || get_channels_ == nullptr ||
      start_channel_ == nullptr || stop_channel_ == nullptr || read_channel_ == nullptr || poll_channel_ == nullptr) {
    Finalize();
    return PGO_PROFAPI_UNAVAILABLE;
  }
  if (register_callback_(kCompactCallbackType, reinterpret_cast<void *>(callback), sizeof(void *)) != 0) {
    Finalize();
    return PGO_PROFAPI_UNAVAILABLE;
  }
  // Keep initialization successful when type-info is not available, but mark
  // the capability explicitly.  The collector then fails closed instead of
  // classifying every runtime task as a kernel.
  const int type_info_ret =
      register_callback_(kTypeInfoCallbackType, reinterpret_cast<void *>(DispatchRuntimeTypeInfo), sizeof(void *));
  SetRuntimeTypeInfoAvailable(type_info_ret == 0);
  callback_registered_ = true;
  initialized_ = true;
  return PGO_SUCCESS;
}

int PgoProfApiAdapter::Start(uint32_t device_id) {
  int64_t version = 0;
  const int version_status = get_device_info_ == nullptr
                                 ? PGO_DATA_UNAVAILABLE
                                 : get_device_info_(device_id, kSystemModule, kVersionInfo, &version);
  return Start(device_id, ResolvePgoStarsSamplePeriod(ResolvePgoPlatform(version_status, version)));
}

int PgoProfApiAdapter::Start(uint32_t device_id, uint32_t sample_period, bool capture_ffts) {
  if (!initialized_ || set_command_ == nullptr) return PGO_PROFAPI_UNAVAILABLE;
  // MSPTI may be loaded after capability probing but before the first
  // candidate.  Re-check the global ownership boundary at the operation that
  // installs profiling state so callbacks cannot be overwritten mid-session.
  if (IsMsptiLoaded()) return PGO_REPORTER_BUSY;
  PgoProfCommand command;
  command.prof_switch = kProfTaskTime | kProfRuntimeTrace;
  command.device_count = 1;
  command.device_ids[0] = device_id;
  command.model_id = 0xffffffffU;
  command.type = kCommandStart;
  if (set_command_(&command, sizeof(command)) != 0) return PGO_PROFILING_START_FAILED;
  profiling_device_id_ = device_id;
  profiling_command_started_ = true;
  const int channel_ret = StartStarsChannel(device_id, sample_period, capture_ffts);
  if (channel_ret != PGO_SUCCESS) {
    command.type = kCommandStop;
    if (set_command_(&command, sizeof(command)) == 0) {
      profiling_command_started_ = false;
    }
  }
  return channel_ret;
}

int PgoProfApiAdapter::Stop(uint32_t device_id) {
  if (!initialized_ || set_command_ == nullptr) return PGO_PROFAPI_UNAVAILABLE;
  // MSPTI disables CANN profiling before stopping the STARS reader.  Keep the
  // same ordering so task-track callbacks generated by the final launches are
  // delivered while the reporter is still registered.
  PgoProfCommand command;
  command.prof_switch = kProfTaskTime | kProfRuntimeTrace;
  command.device_count = 1;
  command.device_ids[0] = device_id;
  command.model_id = 0xffffffffU;
  command.type = kCommandStop;
  const int command_ret = set_command_(&command, sizeof(command)) == 0 ? PGO_SUCCESS : PGO_PROFILING_STOP_FAILED;
  if (command_ret == PGO_SUCCESS) {
    profiling_command_started_ = false;
  }
  const int channel_ret = StopStarsChannel(device_id);
  return CombinePgoStopStatus(channel_ret, command_ret);
}

int PgoProfApiAdapter::StartStarsChannel(uint32_t device_id, uint32_t sample_period, bool capture_ffts) {
  if (!initialized_ || get_channels_ == nullptr || start_channel_ == nullptr) return PGO_CHANNEL_UNAVAILABLE;
  PgoChannelList channels{};
  if (get_channels_(device_id, &channels) != 0 || channels.channel_num == 0 ||
      channels.channel_num > kMaxChannelCount) {
    return PGO_CHANNEL_UNAVAILABLE;
  }
  bool found = false;
  for (uint32_t i = 0; i < channels.channel_num; ++i)
    if (channels.channels[i].id == kStarsSocLogChannel) {
      found = true;
      break;
    }
  if (!found) return PGO_CHANNEL_UNAVAILABLE;
  PgoStarsSocLogConfig config{};
  config.acsq_task = kTaskEnable;
  config.ffts_thread_task = capture_ffts ? kTaskEnable : kTaskDisable;
  PgoStartPara para{};
  para.channel_type = kProfileTypeTs;
  para.sample_period = sample_period;
  para.real_time = kProfileRealTime;
  para.user_data = &config;
  para.user_data_size = sizeof(config);
  if (start_channel_(device_id, kStarsSocLogChannel, &para) != 0) return PGO_PROFILING_START_FAILED;
  stars_device_id_ = device_id;
  stars_channel_started_ = true;
  return PGO_SUCCESS;
}

int PgoProfApiAdapter::StopStarsChannel(uint32_t device_id) {
  if (!stars_channel_started_) return PGO_SUCCESS;
  const bool failed = stop_channel_ == nullptr || stop_channel_(device_id, stars_channel_id_) != 0;
  if (!failed) {
    stars_channel_started_ = false;
  }
  return failed ? PGO_PROFILING_STOP_FAILED : PGO_SUCCESS;
}

int PgoProfApiAdapter::FlushStarsChannel(uint32_t device_id, uint32_t *data_len) {
  if (data_len == nullptr) return PGO_INVALID_ARGUMENT;
  *data_len = 0;
  if (!stars_channel_started_ || flush_channel_ == nullptr) return PGO_SUCCESS;
  const int32_t ret = flush_channel_(device_id, stars_channel_id_, data_len);
  if (ret == 0 || IsPgoFlushUnsupportedStatus(ret)) return PGO_SUCCESS;
  return PGO_DATA_UNAVAILABLE;
}

int PgoProfApiAdapter::ReadStarsChannel(uint32_t device_id, std::vector<uint8_t> &buffer) {
  if (buffer.empty()) return PGO_INVALID_ARGUMENT;
  if (!stars_channel_started_ || read_channel_ == nullptr) return PGO_CHANNEL_UNAVAILABLE;
  const int ret = read_channel_(device_id, stars_channel_id_, buffer.data(), static_cast<uint32_t>(buffer.size()));
  return ret < 0 ? PGO_DATA_UNAVAILABLE : ret;
}

int PgoProfApiAdapter::PollStarsChannel(uint32_t timeout_seconds) {
  if (!stars_channel_started_ || poll_channel_ == nullptr) return PGO_CHANNEL_UNAVAILABLE;
  PgoPollInfo info{stars_device_id_, stars_channel_id_};
  return NormalizePgoPollStatus(poll_channel_(&info, 1, static_cast<int>(timeout_seconds)));
}

void PgoProfApiAdapter::Finalize() {
  if (profiling_command_started_ && set_command_ != nullptr) {
    PgoProfCommand command;
    command.prof_switch = kProfTaskTime | kProfRuntimeTrace;
    command.device_count = 1;
    command.device_ids[0] = profiling_device_id_;
    command.model_id = 0xffffffffU;
    command.type = kCommandStop;
    if (set_command_(&command, sizeof(command)) == 0) {
      profiling_command_started_ = false;
    }
  }
  if (stars_channel_started_) (void)StopStarsChannel(stars_device_id_);
  stars_channel_started_ = false;
  if (callback_registered_ && register_callback_ != nullptr) {
    (void)register_callback_(kCompactCallbackType, nullptr, sizeof(void *));
    (void)register_callback_(kTypeInfoCallbackType, nullptr, sizeof(void *));
  }
  callback_registered_ = false;
  WaitRuntimeTrackCallbacks();
  initialized_ = false;
  set_command_ = nullptr;
  register_callback_ = nullptr;
  get_channels_ = nullptr;
  start_channel_ = nullptr;
  stop_channel_ = nullptr;
  read_channel_ = nullptr;
  poll_channel_ = nullptr;
  flush_channel_ = nullptr;
  get_device_info_ = nullptr;
  if (profapi_handle_ != nullptr) {
    dlclose(profapi_handle_);
    profapi_handle_ = nullptr;
  }
  if (hal_handle_ != nullptr) {
    dlclose(hal_handle_);
    hal_handle_ = nullptr;
  }
}

int PgoProfApiAdapter::GetDeviceInfo(uint32_t device_id, int32_t module, int32_t type, int64_t *value) {
  if (!initialized_ || get_device_info_ == nullptr || value == nullptr) return PGO_INVALID_ARGUMENT;
  return get_device_info_(device_id, module, type, value) == 0 ? PGO_SUCCESS : PGO_DATA_UNAVAILABLE;
}
}  // namespace codegen::pgo
