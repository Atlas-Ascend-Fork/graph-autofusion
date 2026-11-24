/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_CODEGEN_PGO_PGO_TIMESTAMP_H_
#define AUTOFUSE_CODEGEN_PGO_PGO_TIMESTAMP_H_

#include <cstdint>

namespace codegen::pgo {

struct PgoDeviceTimeInfo {
  uint64_t ticks_per_ms = 0;
  uint64_t start_sys_count = 0;
  uint64_t start_real_time_ns = 0;
  bool monotonic_counter = false;
  uint64_t start_monotonic_ns = 0;
};

int ConvertCounterToRealTime(uint64_t counter, const PgoDeviceTimeInfo &info, uint64_t *real_time_ns);
int ConvertTaskDuration(uint64_t start_counter, uint64_t end_counter, const PgoDeviceTimeInfo &info,
                        uint64_t *duration_ns);

}  // namespace codegen::pgo

#endif  // AUTOFUSE_CODEGEN_PGO_PGO_TIMESTAMP_H_
