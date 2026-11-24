/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_CODEGEN_PGO_PGO_CAPABILITY_H_
#define AUTOFUSE_CODEGEN_PGO_PGO_CAPABILITY_H_

#include <cstdint>

#include "pgo_error.h"

namespace codegen::pgo {

enum class PgoReporterOwnership : uint8_t { kUnknown = 0, kExclusive = 1, kExternal = 2 };

enum class PgoPlatformType : uint8_t {
  kUnknown = 0,
  kAscend910B = 5,
  kAscend310B = 7,
  kAscendV6 = 15,
};

struct PgoCapability {
  bool profapi_available = false;
  bool stars_task_track_supported = false;
  bool device_timestamp_supported = false;
  bool serial_stream_supported = false;
  bool runtime_kernel_identity_supported = false;
  bool kernel_activity_reporter_supported = false;
  int profile_status = PGO_PROFILE_UNSUPPORTED;
  uint64_t timestamp_ticks_per_ms = 0;
  PgoPlatformType platform = PgoPlatformType::kUnknown;
};

int ProbePgoCapability(PgoReporterOwnership ownership, void *stream, PgoCapability *capability);
int ProbePgoCapability(PgoReporterOwnership ownership, uint32_t device_id, void *stream, PgoCapability *capability);
int ProbePgoCapability(PgoReporterOwnership ownership, PgoCapability *capability);
int ProbePgoCapability(void *stream, PgoCapability *capability);
int ProbePgoCapability(PgoCapability *capability);
PgoPlatformType ResolvePgoPlatform(int version_status, int64_t version);
uint64_t ResolvePgoDeviceTicksPerMs(int frequency_status, int64_t ticks_per_ms, PgoPlatformType platform);
uint32_t ResolvePgoStarsSamplePeriod(PgoPlatformType platform);
int ResolvePgoCapabilityStatus(bool profapi_ready, bool channel_symbols_ready, bool stars_channel_present,
                               PgoPlatformType platform);

}  // namespace codegen::pgo

#endif  // AUTOFUSE_CODEGEN_PGO_PGO_CAPABILITY_H_
