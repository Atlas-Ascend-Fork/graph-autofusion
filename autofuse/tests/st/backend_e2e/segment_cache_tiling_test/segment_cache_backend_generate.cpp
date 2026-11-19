/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. Do not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <string>

#include "asc_graph_builder.h"
#include "autofuse_config/auto_fuse_config.h"
#include "backend_common.h"
#include "codegen.h"
#include "optimize.h"

namespace {
// HOST_ONLY（tiling-only）形态：KERNEL_SRC_LIST 只含 tiling.cpp 与 autofuse_tiling_data.h
constexpr size_t kTilingFileIndex = 0U;
constexpr size_t kTilingDataFileIndex = 1U;

// 动态shape二维reduce图：Data(s0*s1)→Load→Sum(axis0)→Store→Output。
// 轴尺寸为运行期SizeVar（host侧TilingData字段），R轴(被reduce的s0)参与分段key公式①；
// s1参与公式②（load/store最内轴）。图简洁优先，重在验证分段缓存生成与执行链路。
af::AscGraph MakeSegmentCacheTilingGraph() {
  af::testing::AscGraphBuilder builder("segment_cache_tiling");
  const auto s0 = builder.SizeVar("s0");
  const auto s1 = builder.SizeVar("s1");
  auto graph = builder.Loops({s0, s1})
                   .Data("data", 0, {s0, s1}, {s1, af::testing::Sym(1)})
                   .Load("load0", "data")
                   .Sum("sum0", "load0", {0})
                   .Store("store", "sum0")
                   .Output("output", "store", 0)
                   .Build();
  return graph;
}

void WriteGeneratedFiles(const codegen::CodegenResult &result) {
  const auto parts = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_EQ(parts.size(), kTilingDataFileIndex + 1U);
  std::fstream tiling_file(parts[kTilingFileIndex], std::ios::out);
  std::fstream tiling_data_file(parts[kTilingDataFileIndex], std::ios::out);
  ASSERT_TRUE(tiling_file.is_open());
  ASSERT_TRUE(tiling_data_file.is_open());
  tiling_file << result.tiling;
  tiling_data_file << result.tiling_data;
}
}  // namespace

class TestBackendSegmentCacheTiling : public testing::Test {};

TEST_F(TestBackendSegmentCacheTiling, SegmentCacheTilingCodegen) {
  // 开启动态shape分段缓存：att_accuracy_level=0（默认关闭）
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 1);
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
  ASSERT_EQ(att::AutoFuseConfig::MutableAttStrategyConfig().Init(), af::SUCCESS);

  auto graph = MakeSegmentCacheTilingGraph();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  codegen::Codegen codegen(codegen::CodegenOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), af::SUCCESS);

  const std::map<std::string, std::string> shape_info({{"s0", "s0"}, {"s1", "s1"}});
  codegen::CodegenResult result;
  ASSERT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), af::SUCCESS);

  // 分段缓存关键生成片段：key计算/查找/保存/失效齐全，证明分段路径真实生成
  EXPECT_NE(result.tiling.find("ComputeSegmentKey"), std::string::npos);
  EXPECT_NE(result.tiling.find("FindSegmentCache"), std::string::npos);
  EXPECT_NE(result.tiling.find("SaveSegmentCache"), std::string::npos);
  EXPECT_NE(result.tiling.find("InvalidateSegmentCache"), std::string::npos);
  WriteGeneratedFiles(result);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}
