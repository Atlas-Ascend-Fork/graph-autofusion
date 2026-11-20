/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include "ascendc_ir.h"
#include "ascir_utils.h"

namespace {
const std::string kDumpRoot = "./DumpGraphLongNameSt";

bool HasPrefix(const std::string &name, const std::string &prefix) {
  return name.rfind(prefix, 0UL) == 0UL;
}

bool HasSuffix(const std::string &name, const std::string &suffix) {
  return name.length() >= suffix.length() &&
         name.compare(name.length() - suffix.length(), suffix.length(), suffix) == 0;
}
}  // namespace

class AscirDumpGraphSt : public testing::Test {
 protected:
  void SetUp() override {
    std::error_code error_code;
    std::filesystem::remove_all(kDumpRoot, error_code);
    setenv("AUTOFUSE_DFX_FLAGS", ("codegen_compile_debug=true;debug_dir=" + kDumpRoot).c_str(), 1);
    ::ascir::utils::ResetDumpConfig();
  }

  void TearDown() override {
    unsetenv("AUTOFUSE_DFX_FLAGS");
    ::ascir::utils::ResetDumpConfig();
    std::error_code error_code;
    std::filesystem::remove_all(kDumpRoot, error_code);
  }
};

TEST_F(AscirDumpGraphSt, dump_graph_limits_long_file_name) {
  const std::string long_name(300U, 'g');
  af::AscGraph graph(long_name.c_str());
  ::ascir::utils::DumpGraph(graph, std::string(200U, 's'));
  ::ascir::utils::DumpPyCode(graph);

  size_t ascgraph_file_count = 0U;
  size_t ge_onnx_file_count = 0U;
  size_t py_code_file_count = 0U;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(kDumpRoot)) {
    const std::string file_name = entry.path().filename().string();
    if (HasPrefix(file_name, "ascgraph_")) {
      ++ascgraph_file_count;
      EXPECT_EQ(file_name.length(), 255U);
      EXPECT_TRUE(HasSuffix(file_name, ".txt"));
      EXPECT_GT(std::filesystem::file_size(entry.path()), 0U);
    } else if (HasPrefix(file_name, "ge_onnx_")) {
      ++ge_onnx_file_count;
      EXPECT_EQ(file_name.length(), 255U);
      EXPECT_TRUE(HasSuffix(file_name, ".pbtxt"));
      EXPECT_GT(std::filesystem::file_size(entry.path()), 0U);
    } else if (HasPrefix(file_name, "py_code_")) {
      ++py_code_file_count;
      EXPECT_EQ(file_name.length(), 255U);
      EXPECT_TRUE(HasSuffix(file_name, ".py"));
      EXPECT_GT(std::filesystem::file_size(entry.path()), 0U);
    }
  }
  EXPECT_EQ(ascgraph_file_count, 1U);
  EXPECT_EQ(ge_onnx_file_count, 1U);
  EXPECT_EQ(py_code_file_count, 1U);
}
