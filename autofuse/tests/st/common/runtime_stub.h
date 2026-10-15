/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_RUNTIME_STUB_H
#define AUTOFUSE_RUNTIME_STUB_H
#include <cstring>
#include_next <runtime_stub.h>
#include "../../common/att/soc_perf_test_config.h"

namespace af {
using ge::RuntimeStub;
class RuntimeStubV2 : public ge::RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) override {
    (void)strcpy_s(version, maxLen, "Ascend910_9591");
    return RT_ERROR_NONE;
  }

  rtError_t rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen) override {
    return ge::CopyRuntimeSocSpecValue(label, key, val, maxLen, ge::RuntimeSocSpecDefaults{"3510", "0"});
  }
};

inline void InstallV35SocPerfRuntimeStub() {
  ge::RuntimeStub::SetInstance(std::make_shared<RuntimeStubV2>());
  if (att::soc_perf_test::SetSocPerfTestPlatform(att::soc_perf_test::kV35AltNpuArch, att::soc_perf_test::kSocEmpty) !=
      af::SUCCESS) {
    ge::RuntimeStub::Reset();
  }
}
}  // namespace af

#endif  // AUTOFUSE_RUNTIME_STUB_H
