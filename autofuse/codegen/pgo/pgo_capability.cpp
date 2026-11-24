/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#include "pgo_capability.h"

#include <dlfcn.h>
#include <link.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <string>

namespace codegen::pgo {
namespace {
constexpr uint32_t kStarsChannel = 50;
constexpr uint32_t kSystemModule = 0;
constexpr uint32_t kVersionInfo = 1;
constexpr uint32_t kDeviceOscFrequency = 25;
constexpr uint32_t kMaxChannelCount = 160;
#pragma pack(push, 4)
struct ChannelInfo {
  char name[32];
  uint32_t type;
  uint32_t id;
};
struct ChannelList {
  uint32_t chip_type;
  uint32_t channel_num;
  ChannelInfo channel[kMaxChannelCount];
};
#pragma pack(pop)

template <size_t PathCount>
void *OpenLibrary(const char *name, const std::array<const char *, PathCount> &relative_paths) {
  if (void *handle = dlopen(name, RTLD_NOW | RTLD_LOCAL); handle != nullptr) return handle;
  const char *root = std::getenv("ASCEND_HOME_PATH");
  if (root == nullptr) root = std::getenv("ASCEND_TOOLKIT_HOME");
  if (root == nullptr) return nullptr;
  for (const char *relative : relative_paths) {
    const std::string path = std::string(root) + "/" + relative;
    if (void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL); handle != nullptr) return handle;
  }
  return nullptr;
}

void *OpenHal() {
  static constexpr std::array kPaths = {"x86_64-linux/devlib/libascend_hal.so",
                                        "x86_64-linux/lib64/device/lib64/libascend_hal.so",
                                        "x86_64-linux/lib64/libascend_hal.so",
                                        "aarch64-linux/devlib/libascend_hal.so",
                                        "aarch64-linux/lib64/device/lib64/libascend_hal.so",
                                        "aarch64-linux/lib64/libascend_hal.so"};
  return OpenLibrary("libascend_hal.so", kPaths);
}

void *OpenProfApi() {
  static constexpr std::array kPaths = {"x86_64-linux/lib64/libprofapi.so", "aarch64-linux/lib64/libprofapi.so",
                                        "lib64/libprofapi.so"};
  return OpenLibrary("libprofapi.so", kPaths);
}

int FindMspti(struct dl_phdr_info *info, size_t, void *data) {
  if (info == nullptr || info->dlpi_name == nullptr || data == nullptr) return 0;
  const char *base = std::strrchr(info->dlpi_name, '/');
  base = base == nullptr ? info->dlpi_name : base + 1;
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
  (void)dl_iterate_phdr(FindMspti, &loaded);
  return loaded;
}
}  // namespace

PgoPlatformType ResolvePgoPlatform(int version_status, int64_t version) {
  if (version_status != PGO_SUCCESS) return PgoPlatformType::kUnknown;
  const uint32_t chip_id = static_cast<uint32_t>((static_cast<uint64_t>(version) >> 8U) & 0xffU);
  switch (chip_id) {
    case static_cast<uint32_t>(PgoPlatformType::kAscend910B):
      return PgoPlatformType::kAscend910B;
    case static_cast<uint32_t>(PgoPlatformType::kAscend310B):
      return PgoPlatformType::kAscend310B;
    case static_cast<uint32_t>(PgoPlatformType::kAscendV6):
      return PgoPlatformType::kAscendV6;
    default:
      return PgoPlatformType::kUnknown;
  }
}

uint64_t ResolvePgoDeviceTicksPerMs(int frequency_status, int64_t ticks_per_ms, PgoPlatformType platform) {
  if (frequency_status == PGO_SUCCESS) return ticks_per_ms > 0 ? static_cast<uint64_t>(ticks_per_ms) : 0U;
  return platform == PgoPlatformType::kAscendV6 ? 1000U : 50U;
}

uint32_t ResolvePgoStarsSamplePeriod(PgoPlatformType platform) {
  return platform == PgoPlatformType::kAscendV6 ? 0U : 20U;
}

int ResolvePgoCapabilityStatus(bool profapi_ready, bool channel_symbols_ready, bool stars_channel_present,
                               PgoPlatformType platform) {
  if (!profapi_ready || !channel_symbols_ready) return PGO_CAPABILITY_UNAVAILABLE;
  if (!stars_channel_present || platform == PgoPlatformType::kUnknown) return PGO_PROFILE_UNSUPPORTED;
  return PGO_SUCCESS;
}

int ProbePgoCapability(PgoReporterOwnership ownership, uint32_t device_id, void *stream, PgoCapability *capability) {
  if (capability == nullptr) return PGO_INVALID_ARGUMENT;
  *capability = {};
  if (ownership != PgoReporterOwnership::kExclusive || IsMsptiLoaded()) return PGO_REPORTER_BUSY;
  if (stream == nullptr) return PGO_CAPABILITY_UNAVAILABLE;
  void *profapi = OpenProfApi();
  void *hal = OpenHal();
  if (profapi == nullptr || hal == nullptr) {
    if (profapi != nullptr) dlclose(profapi);
    if (hal != nullptr) dlclose(hal);
    return PGO_CAPABILITY_UNAVAILABLE;
  }
  const bool profapi_ready =
      dlsym(profapi, "profSetProfCommand") != nullptr && dlsym(profapi, "MsprofRegisterProfileCallback") != nullptr;
  auto get_channels = reinterpret_cast<int (*)(uint32_t, ChannelList *)>(dlsym(hal, "prof_drv_get_channels"));
  auto get_info = reinterpret_cast<int (*)(uint32_t, int32_t, int32_t, int64_t *)>(dlsym(hal, "halGetDeviceInfo"));
  const bool channel_symbols = get_channels != nullptr && get_info != nullptr &&
                               dlsym(hal, "prof_drv_start") != nullptr && dlsym(hal, "prof_stop") != nullptr &&
                               dlsym(hal, "prof_channel_read") != nullptr && dlsym(hal, "prof_channel_poll") != nullptr;
  ChannelList channels{};
  bool stars_channel = false;
  if (get_channels != nullptr && get_channels(device_id, &channels) == 0 && channels.channel_num > 0 &&
      channels.channel_num <= kMaxChannelCount) {
    for (uint32_t i = 0; i < channels.channel_num; ++i) {
      if (channels.channel[i].id == kStarsChannel) {
        stars_channel = true;
        break;
      }
    }
  }
  int64_t version = 0;
  const int version_status = get_info == nullptr ? PGO_DATA_UNAVAILABLE
                                                 : get_info(device_id, static_cast<int32_t>(kSystemModule),
                                                            static_cast<int32_t>(kVersionInfo), &version);
  const PgoPlatformType platform = ResolvePgoPlatform(version_status, version);
  capability->profapi_available = profapi_ready;
  capability->platform = platform;
  capability->profile_status = ResolvePgoCapabilityStatus(profapi_ready, channel_symbols, stars_channel, platform);
  capability->stars_task_track_supported = capability->profile_status == PGO_SUCCESS;
  capability->runtime_kernel_identity_supported = profapi_ready;
  capability->kernel_activity_reporter_supported = false;
  capability->serial_stream_supported = capability->stars_task_track_supported;
  if (get_info != nullptr) {
    int64_t ticks_per_ms = 0;
    const int frequency_status = get_info(device_id, static_cast<int32_t>(kSystemModule),
                                          static_cast<int32_t>(kDeviceOscFrequency), &ticks_per_ms);
    capability->timestamp_ticks_per_ms = ResolvePgoDeviceTicksPerMs(frequency_status, ticks_per_ms, platform);
  }
  capability->device_timestamp_supported = capability->timestamp_ticks_per_ms != 0;
  dlclose(profapi);
  dlclose(hal);
  // Match MSPTI: a failed oscillator query falls back to the platform default
  // instead of rejecting profiling.
  return capability->profile_status;
}

int ProbePgoCapability(PgoReporterOwnership ownership, void *stream, PgoCapability *capability) {
  return ProbePgoCapability(ownership, 0U, stream, capability);
}

int ProbePgoCapability(PgoReporterOwnership ownership, PgoCapability *capability) {
  return ProbePgoCapability(ownership, reinterpret_cast<void *>(1), capability);
}

int ProbePgoCapability(PgoCapability *capability) {
  // Kept for existing callers.  A non-null sentinel means "probe the default
  // device"; callers that own a runtime stream should use the stream overload.
  return ProbePgoCapability(PgoReporterOwnership::kExclusive, reinterpret_cast<void *>(1), capability);
}

int ProbePgoCapability(void *stream, PgoCapability *capability) {
  return ProbePgoCapability(PgoReporterOwnership::kExclusive, stream, capability);
}
}  // namespace codegen::pgo
