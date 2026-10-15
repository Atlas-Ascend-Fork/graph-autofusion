/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/platform_context.h"
#include "../../../../common/att/soc_perf_test_config.h"
#include "runtime_stub.h"
#include "common/test_common_headers.h"
#include "common/st_scenario_utils.h"

using namespace att;
class STestReduceSplitPenaltyV2 : public ::testing::Test {
 public:
  static af::RuntimeStubV2 stub_v_2;
  void SetUp() override {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
    AutoFuseConfig::MutableAttStrategyConfig().Reset();
    ge::RuntimeStub::Install(&stub_v_2);
    ge::PlatformContext::GetInstance().Reset();
    ASSERT_EQ(
        att::soc_perf_test::SetSocPerfTestPlatform(att::soc_perf_test::kV35NpuArch, att::soc_perf_test::kSocEmpty),
        af::SUCCESS);
    // V2形态：通过TilingScheduleConfigTableV2自动返回true，无需手动设置
  }
  void TearDown() override {
    ge::RuntimeStub::UnInstall(&stub_v_2);
    ge::PlatformContext::GetInstance().Reset();
  }
};
af::RuntimeStubV2 STestReduceSplitPenaltyV2::stub_v_2;

namespace {

using test_common::ConstructSingleCaseForReduceSplitPenalty;
using test_common::GenTilingImpl;
using test_common::PrintTilingDebugInfo;
using test_common::RunBasicTilingTest;
using test_common::ValidateTilingResult;

// 创建运行测试的main函数内容 (V2版本，包含两个测试用例)
std::string CreateRunTestMainV2() {
  return R"(
#include <iostream>
using namespace optiling;
namespace optiling {
bool GetTiling(ReduceSplitPenaltyTilingData &tiling_data, int32_t tiling_case_id, double *perf);
}

void Test1(ReduceSplitPenaltyTilingData &tilingData) {
  std::cout << "====================================================" << std::endl;
  auto block_dim = tilingData.get_block_dim();
  std::cout << "get_block_dim"<< " = " << block_dim << std::endl;
  // cache_line_size从64统一为128后，block_dim从64变为34
  MY_ASSERT_EQ(block_dim, 34);
  std::cout << "====================================================" << std::endl;
}

void Test2(ReduceSplitPenaltyTilingData &tilingData) {
  std::cout << "====================================================" << std::endl;
  auto block_dim = tilingData.get_block_dim();
  std::cout << "get_block_dim"<< " = " << block_dim << std::endl;
  MY_ASSERT_EQ(block_dim < 4, true);
  std::cout << "====================================================" << std::endl;
}

int main() {
  ReduceSplitPenaltyTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  // 新增：测试S0=1024,S1=1024时的block_dim
  tilingData.set_S0(1024);
  tilingData.set_S1(1024);
  if (GetTiling(tilingData, -1, nullptr)) {
    Test1(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }

  // 新增：测试S0=1024,S1=32时的block_dim
  tilingData.set_S0(1024);
  tilingData.set_S1(32);
  if (GetTiling(tilingData, -1, nullptr)) {
    Test2(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
}

// Test Reduce split penalty scenario with Reduce on s2t axis, multi-core on merged axis
TEST_F(STestReduceSplitPenaltyV2, reduce_split_penalty_basic) {
  // Build graph with Reduce on s2t axis, multi-core on merged s1Ts2T axis
  // - s2t is the Reduce axis (reduced by Mean operation)
  // - s1Ts2T is the merged axis that gets split for multi-core
  std::vector<ascir::ScheduledResult> schedule_results;
  ASSERT_EQ(ConstructSingleCaseForReduceSplitPenalty(schedule_results), af::SUCCESS);
  RunBasicTilingTest("ReduceSplitPenalty", schedule_results);
}

TEST_F(STestReduceSplitPenaltyV2, reduce_split_penalty_run) {
  int32_t ret = 0;
  std::vector<ascir::ScheduledResult> schedule_results;
  ASSERT_EQ(ConstructSingleCaseForReduceSplitPenalty(schedule_results), af::SUCCESS);
  ASSERT_EQ(GenTilingImpl(schedule_results), af::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_ReduceSplitPenalty.cpp", std::ios::out);
  oss << "#include \"ReduceSplitPenalty_tiling_data.h\"" << ResultCheckerUtils::DefineCheckerFunction()
      << CreateRunTestMainV2();
  oss.close();
  ret = std::system(
      "g++ -std=c++17 -ggdb3 -O0 tiling_func_main_ReduceSplitPenalty.cpp ReduceSplitPenalty_tiling_func.cpp -o "
      "tiling_func_main_ReduceSplitPenalty -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_ReduceSplitPenalty");
  EXPECT_EQ(ret, 0);
}
}  // namespace
