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
#include "tests/depends/runtime/src/runtime_stub.h"
#include "../../../common/att/soc_perf_test_config.h"

namespace af {
class RuntimeStubV2 : public ge::RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) override {
    (void)strcpy_s(version, maxLen, "Ascend910_9591");
    return RT_ERROR_NONE;
  }

  rtError_t rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen) override {
    return ge::CopyRuntimeSocSpecValue(label, key, val, maxLen, ge::RuntimeSocSpecDefaults{"3510", "3510"});
  }

  // reg-base 那一侧的 SoC 名。legacy 一路只读 NpuArch（上面那两个覆写），所以这一处缺了
  // 一直没人发现；而 MLIR 的 `ResolveSocVersion` 读的正是 `aclrtGetSocName`——用例给的
  // `CodegenOptions{}` soc_version 为空时就落到它身上。不覆写就继承基类那个
  // `"Ascend910B1"`，于是 MLIR 会按 **mem 侧** profile 对着一份 reg-base 的调度产物下降，
  // **而且不报错**。
  //
  // 取值与 `mlir/test/e2e/cpu/AddCpuCase.cmake` 的 `reg_base` 那一支一致，那里抄的是
  // `mlir/target/Ascend/Profiles/platform_config/Ascend950PR_9599.ini`（`NpuArch=3510`，
  // 与上面 `rtGetSocSpec` 答的那个 arch 对得上）。**两处必须同一个 arch**，不然调度与
  // MLIR 两边按不同平台算。
  const char *aclrtGetSocName() override {
    return "Ascend950PR_9599";
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
