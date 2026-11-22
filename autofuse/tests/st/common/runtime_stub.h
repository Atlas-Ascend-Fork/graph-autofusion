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

  // reg-base 那一侧的 SoC 名。legacy 一路只读 NpuArch（上面那两个覆写），所以缺这一处
  // 不影响它；而 MLIR 的 `ResolveSocVersion` 读的正是 `aclrtGetSocName`
  // （`bridge/CodegenOrchestration/CodegenOrchestration.cpp:173`）——用例给的
  // `CodegenOptions{}` soc_version 为空时就落到它身上。不覆写就继承基类那个
  // `"Ascend910B1"`，于是 MLIR 会按 **mem 侧** profile 对着一份 reg-base 的调度产物
  // 下降，**而且不报错**。
  //
  // **这一处 2026-10-17 补过，但补错了文件。** 那一提交（4ca857ac，「补齐 reg 侧接管表」）
  // 补的是 `tests/v35/st/optimize/runtime_stub.h` 里的同名类，而接管表真正量的那 200 格
  // （`v35/st/backend_e2e_v2` 的 `*_codegen_v2`）按 include 路径解析到的是**本文件**
  // （`-I .../tests/st/common`）。两个同名 `RuntimeStubV2` 是分头演化的，改一个不会让
  // 另一个变红。后果是 2026-10-17 与 2026-10-19 两张「reg 侧」接管表里，legacy 的调度按
  // NpuArch 3510 走 reg 侧、而 MLIR 按 mem 侧 profile 下降——两边不是同一个平台。
  //
  // 这件事被看见是因为 `afir.ndtr`：它是第一个 **legacy 只有 reg 侧有** 的 op（对照表
  // `mem_present=0`），在 mem profile 上一个 implementation 都选不出来。在它之前接通的
  // op 两侧都有实现，按 mem 侧选中一个照样能走完，所以这个错配一直是静默的。
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
