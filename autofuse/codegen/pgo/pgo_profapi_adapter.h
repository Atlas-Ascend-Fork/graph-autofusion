/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_CODEGEN_PGO_PGO_PROFAPI_ADAPTER_H_
#define AUTOFUSE_CODEGEN_PGO_PGO_PROFAPI_ADAPTER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pgo_error.h"
#include "pgo_capability.h"

namespace codegen::pgo {

using CompactCallback = int32_t (*)(uint32_t aging_flag, const void *data, uint32_t length);

struct PgoProfCommandParams {
  uint32_t path_len = 0;
  uint32_t storage_limit = 0;
  uint32_t prof_data_len = 0;
  char path[1024] = {};
  char prof_data[4096] = {};
};

struct PgoProfCommand {
  uint64_t prof_switch = 0;
  uint64_t prof_switch_hi = 0;
  uint32_t device_count = 0;
  uint32_t device_ids[64] = {};
  uint32_t model_id = 0xffffffffU;
  uint32_t type = 0;
  uint32_t cache_flag = 0;
  PgoProfCommandParams params;
};

static_assert(sizeof(PgoProfCommandParams) == 5132, "prof command parameter ABI changed");
static_assert(sizeof(PgoProfCommand) == 5424, "prof command ABI changed");

int CombinePgoStopStatus(int channel_status, int command_status);
int NormalizePgoPollStatus(int poll_status);
bool IsPgoFlushUnsupportedStatus(int32_t status);
int PgoDlopenFlags();

class PgoProfApiAdapter {
 public:
  PgoProfApiAdapter() = default;
  ~PgoProfApiAdapter();
  PgoProfApiAdapter(const PgoProfApiAdapter &) = delete;
  PgoProfApiAdapter &operator=(const PgoProfApiAdapter &) = delete;

  int Initialize(CompactCallback callback, PgoReporterOwnership ownership = PgoReporterOwnership::kExclusive);
  int Start(uint32_t device_id);
  int Start(uint32_t device_id, uint32_t sample_period, bool capture_ffts = false);
  int Stop(uint32_t device_id);
  // Start/stop the driver STARS_SOC_LOG channel.  The profapi command and
  // channel are intentionally separate, matching MSPTI's DevTaskManager and
  // DevProfTaskStars lifecycle.
  int StartStarsChannel(uint32_t device_id, uint32_t sample_period = 20, bool capture_ffts = false);
  int StopStarsChannel(uint32_t device_id);
  int FlushStarsChannel(uint32_t device_id, uint32_t *data_len);
  int ReadStarsChannel(uint32_t device_id, std::vector<uint8_t> &buffer);
  int PollStarsChannel(uint32_t timeout_seconds);
  int GetDeviceInfo(uint32_t device_id, int32_t module, int32_t type, int64_t *value);
  bool IsStarsChannelStarted() const {
    return stars_channel_started_;
  }
  uint32_t stars_channel_id() const {
    return stars_channel_id_;
  }
  void Finalize();

 private:
  void *profapi_handle_ = nullptr;
  int32_t (*set_command_)(void *, uint32_t) = nullptr;
  int32_t (*register_callback_)(int32_t, void *, uint32_t) = nullptr;
  int (*get_channels_)(uint32_t, void *) = nullptr;
  int (*start_channel_)(uint32_t, uint32_t, void *) = nullptr;
  int (*stop_channel_)(uint32_t, uint32_t) = nullptr;
  int (*read_channel_)(uint32_t, uint32_t, void *, uint32_t) = nullptr;
  int (*poll_channel_)(void *, int, int) = nullptr;
  int (*flush_channel_)(uint32_t, uint32_t, uint32_t *) = nullptr;
  int (*get_device_info_)(uint32_t, int32_t, int32_t, int64_t *) = nullptr;
  void *hal_handle_ = nullptr;
  uint32_t stars_channel_id_ = 50;
  uint32_t stars_device_id_ = 0;
  bool stars_channel_started_ = false;
  uint32_t profiling_device_id_ = 0;
  bool profiling_command_started_ = false;
  bool initialized_ = false;
  bool callback_registered_ = false;
};

}  // namespace codegen::pgo

#endif  // AUTOFUSE_CODEGEN_PGO_PGO_PROFAPI_ADAPTER_H_
