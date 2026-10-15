/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "st_process.h"
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

TEST(AotProfilingSystemTest, ExportsEmptyTimelineOnStopAndAtProcessShutdown) {
  for (const std::string scenario : {"normal", "missing_stop"}) {
    SCOPED_TRACE(scenario);
    sk::test::TemporaryDirectory directory;
    ASSERT_EQ(sk::test::RunProcess({SK_PROFILING_PROCESS_PATH, directory.Path().string(), "1", scenario}), 0);
    std::ifstream output(directory.Path() / "export/sk_prof_device_0.json");
    ASSERT_TRUE(output.is_open());
    EXPECT_EQ(nlohmann::json::parse(output), nlohmann::json::array({nlohmann::json::object()}));
    size_t localFiles = 0;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(directory.Path() / "sk_meta")) {
      if (entry.path().filename() == "sk_prof_device_0.json") {
        ++localFiles;
        std::ifstream local(entry.path());
        EXPECT_EQ(nlohmann::json::parse(local), nlohmann::json::array({nlohmann::json::object()}));
      }
    }
    EXPECT_EQ(localFiles, 1U);
  }
}

TEST(AotProfilingSystemTest, InvalidAndDisabledEnvironmentPreservesOptimization) {
  for (const std::string value : {"0", "", "abc", "-1", "5121"}) {
    SCOPED_TRACE(value);
    sk::test::TemporaryDirectory directory;
    ASSERT_EQ(sk::test::RunProcess({SK_PROFILING_PROCESS_PATH, directory.Path().string(), value, "disabled"}), 0);
    EXPECT_FALSE(std::filesystem::exists(directory.Path() / "sk_meta"));
    EXPECT_TRUE(std::filesystem::is_empty(directory.Path() / "export"));
  }
}

TEST(AotProfilingSystemTest, RecorderFailuresDoNotPreventOptimizationOrProcessExit) {
  for (const std::string scenario : {"memset_failure", "directory_failure", "empty_path"}) {
    SCOPED_TRACE(scenario);
    sk::test::TemporaryDirectory directory;
    ASSERT_EQ(sk::test::RunProcess({SK_PROFILING_PROCESS_PATH, directory.Path().string(), "1", scenario}), 0);
    EXPECT_TRUE(std::filesystem::is_empty(directory.Path() / "export"));
  }
}
