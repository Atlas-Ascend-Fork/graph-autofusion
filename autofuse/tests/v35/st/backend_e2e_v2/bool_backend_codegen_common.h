/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_BOOL_BACKEND_CODEGEN_COMMON_H_
#define AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_BOOL_BACKEND_CODEGEN_COMMON_H_

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "backend_codegen_common.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

class BoolBackendCodegenE2e : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    af::InstallV35SocPerfRuntimeStub();
  }

  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
  }
};

inline void GenerateBoolBackendKernel(const af::ComputeGraphPtr &graph) {
  std::string tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  GenerateBackendKernelWithCheck(graph, shape_info, tiling_stub, [](const std::string &) {});
}

#endif  // AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_BOOL_BACKEND_CODEGEN_COMMON_H_
