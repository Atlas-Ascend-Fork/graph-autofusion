/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#include "pgo_timestamp.h"

#include <limits>
#include <securec.h>

namespace codegen::pgo {
namespace {
constexpr uint64_t kNsPerSecond = 1000000000ULL;

int64_t SignedCounter(uint64_t value) {
  int64_t signed_value = 0;
  if (memcpy_s(&signed_value, sizeof(signed_value), &value, sizeof(value)) != EOK) return 0;
  return signed_value;
}

}  // namespace

int ConvertCounterToRealTime(uint64_t counter, const PgoDeviceTimeInfo &info, uint64_t *real_time_ns) {
  if (real_time_ns == nullptr) {
    return -1;
  }
  if (info.frequency == 0) {
    // MSPTI returns the raw device counter when the oscillator frequency is
    // unavailable.  Do not reinterpret a device counter as a host monotonic
    // timestamp: the two clocks have different epochs and units.
    *real_time_ns = counter;
    return 0;
  }

  // MSPTI treats the device counter as a signed two's-complement 64-bit value
  // before applying the integer quotient/remainder formula. Preserve that
  // bit-level conversion explicitly while using 128-bit intermediates.
  const auto delta =
      static_cast<__int128>(SignedCounter(counter)) - static_cast<__int128>(SignedCounter(info.start_sys_count));
  // |delta| <= UINT64_MAX, so scaling by 1e9 fits in __int128 and preserves
  // truncation toward zero. A single division avoids the newer libgcc
  // dependency introduced by a fused signed division/remainder operation.
  const auto real = (delta * static_cast<__int128>(kNsPerSecond)) / static_cast<__int128>(info.frequency) +
                    static_cast<__int128>(info.start_real_time_ns);
  if (real < 0 || real > std::numeric_limits<uint64_t>::max()) {
    return -1;
  }
  *real_time_ns = static_cast<uint64_t>(real);
  return 0;
}

int ConvertTaskDuration(uint64_t start_counter, uint64_t end_counter, const PgoDeviceTimeInfo &info,
                        uint64_t *duration_ns) {
  if (duration_ns == nullptr) {
    return -1;
  }
  if (info.frequency == 0) {
    if (end_counter < start_counter) return -1;
    *duration_ns = end_counter - start_counter;
    return 0;
  }
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  if (ConvertCounterToRealTime(start_counter, info, &start_ns) != 0 ||
      ConvertCounterToRealTime(end_counter, info, &end_ns) != 0 || end_ns < start_ns) {
    return -1;
  }
  *duration_ns = end_ns - start_ns;
  return 0;
}
}  // namespace codegen::pgo
