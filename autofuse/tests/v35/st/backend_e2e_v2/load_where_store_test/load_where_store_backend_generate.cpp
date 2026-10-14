/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
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
#include "runtime_stub.h"
#include "common/platform_context.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

class TestBackendLoadWhereStoreE2e : public testing::Test {
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

TEST_F(TestBackendLoadWhereStoreE2e, LoadWhereStoreE2eCodegen) {
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  // shape_info 和 AddAbsFusedGraph入参dims_size匹配（个数相同，命名规则为s开头、编号从0开始）
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}, {"s2", "stub_s2"}});
  auto graph = ascir::ShareGraph::LoadWhereReduceStoreFusedGraph(3, false, false);
  GenerateBackendKernelWithCheck(graph, shape_info, tilig_stub, [](const std::string &kernel) {
    EXPECT_NE(kernel.find("WhereExtend<false, false>(local_8[0], local_7[0], local_6[0], local_5[0], "
                          "{static_cast<uint16_t>(z0z1t_actual_size)"),
              std::string::npos);
    EXPECT_NE(kernel.find("{static_cast<uint16_t>(((8 * Ceiling((Rational(1 , 8) * t->z2t_size))))/(1)), "
                          "static_cast<uint16_t>(1)}"),
              std::string::npos);
  });
}
