/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "pgo_profiler_stub.h"

#include <cstdint>
namespace {
constexpr uint32_t kStarsChannel = 50U;
constexpr int32_t kVersionInfo = 1;
constexpr int32_t kOscillatorFrequency = 25;
constexpr int64_t kLegacyChipVersion = 5LL << 8U;
constexpr uint32_t kMaxChannels = 160U;

// Driver ABI consumed by both capability probing and the profiling adapter.
struct ChannelInfo {
  char name[32];
  uint32_t type;
  uint32_t id;
};
struct ChannelList {
  uint32_t chip_type;
  uint32_t count;
  ChannelInfo channels[kMaxChannels];
};
static_assert(sizeof(ChannelInfo) == 40U, "Unexpected channel ABI");

thread_local PgoStubFailure g_failure = PgoStubFailure::None;
}  // namespace

extern "C" void SetPgoStubFailure(PgoStubFailure failure) {
  g_failure = failure;
}

extern "C" int32_t profSetProfCommand(void *, uint32_t) {
  return 0;
}

extern "C" int32_t MsprofRegisterProfileCallback(int32_t, void *, uint32_t) {
  return 0;
}

extern "C" int prof_drv_get_channels(uint32_t, void *output) {
  ChannelList list{};
  list.count = 1U;
  list.channels[0].id = kStarsChannel;
  *static_cast<ChannelList *>(output) = list;
  return 0;
}

extern "C" int halGetDeviceInfo(uint32_t, int32_t, int32_t type, int64_t *value) {
  if (type == kVersionInfo) {
    *value = kLegacyChipVersion;
  } else if (type == kOscillatorFrequency) {
    *value = 50;
  } else {
    *value = 0;
  }
  return 0;
}

extern "C" int prof_drv_start(uint32_t, uint32_t, void *) {
  return g_failure == PgoStubFailure::Start ? -1 : 0;
}

extern "C" int prof_stop(uint32_t, uint32_t) {
  return g_failure == PgoStubFailure::Stop ? -1 : 0;
}

extern "C" int halProfDataFlush(uint32_t, uint32_t, uint32_t *size) {
  *size = 0U;
  return g_failure == PgoStubFailure::Flush ? -1 : 0;
}

extern "C" int prof_channel_read(uint32_t, uint32_t, void *, uint32_t) {
  return g_failure == PgoStubFailure::Read ? -1 : 0;
}

extern "C" int prof_channel_poll(void *, int, int) {
  return 0;
}
