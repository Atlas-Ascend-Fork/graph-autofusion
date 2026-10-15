/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <exception>
#include <filesystem>
#include "codegen.h"
#include "optimize.h"
#include "share_graph.h"
#include "../backend_codegen_common.h"

#include <iostream>
#include <vector>
#include <string>
#include "runtime_stub.h"
#include "common/platform_context.h"

class TestBackendFloortointFloatE2e : public testing::Test {
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

TEST_F(TestBackendFloortointFloatE2e, FloortointFloatE2eCodegen) {
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::FloorToIntFloatFusedGraph(2);
  GenerateBackendKernelWithCheck(graph, shape_info, tilig_stub, [](const std::string &kernel) {
    EXPECT_NE(kernel.find("AscendC::Cast(local_3[0], local_2[0], AscendC::RoundMode::CAST_FLOOR, "
                          "local_2_actual_size);"),
              std::string::npos);
    EXPECT_NE(kernel.find("DataCopyPadExtend<int32_t, AscendC::PaddingMode::Normal>"), std::string::npos);
  });
}
