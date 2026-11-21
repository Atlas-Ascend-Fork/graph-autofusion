/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "sk_log.h"

class TestSkLogContext : public testing::Test {
 protected:
  void TearDown() override {
    sk::logger::FileLogger::Instance().SetEnabled(false);
    sk::logger::FileLogger::SetCurrentModelId("");
    sk::logger::FileHandleManager::Instance().SwitchToDefault();
  }
};

TEST_F(TestSkLogContext, SanitizePathComponent_ReplacesInvalidCharacters) {
  EXPECT_EQ(SanitizePathComponent("model_42_1"), "model_42_1");
  EXPECT_EQ(SanitizePathComponent("a/b\\c:d*e?f\"g<h>i|j"), "a_b_c_d_e_f_g_h_i_j");
  EXPECT_EQ(SanitizePathComponent(""), "");
}

TEST_F(TestSkLogContext, GetSkMetaPath_UsesCanonicalModelIdDirectly) {
  EXPECT_EQ(GetSkMetaPath("model_42_1"), GetSkMetaBasePath() + "/model_42_1");
  EXPECT_EQ(GetSkMetaPath("model_path_component"), GetSkMetaBasePath() + "/model_path_component");
  EXPECT_TRUE(GetSkMetaPath("").empty());
}

TEST_F(TestSkLogContext, CreateDirectoryRecursive_CreatesNestedPathAndIsIdempotent) {
  const std::string path = GetSkMetaBasePath() + "/log_context_nested/child";
  EXPECT_TRUE(CreateDirectoryRecursive(path));
  EXPECT_TRUE(CreateDirectoryRecursive(path));

  struct stat st;
  ASSERT_EQ(stat(path.c_str(), &st), 0);
  EXPECT_TRUE(S_ISDIR(st.st_mode));
}

TEST_F(TestSkLogContext, CreateSkMetaDirectory_DoesNotAddAnotherModelPrefix) {
  const std::string path = CreateSkMetaDirectory("model_77_2");
  EXPECT_EQ(path, GetSkMetaBasePath() + "/model_77_2");

  struct stat st;
  ASSERT_EQ(stat(path.c_str(), &st), 0);
  EXPECT_TRUE(S_ISDIR(st.st_mode));
}

TEST_F(TestSkLogContext, FileLogger_InitializationFailureReportsDirectory) {
  const std::string modelId(256, 'a');
  const std::string targetPath = GetSkMetaPath(modelId);

  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelId = modelId;
  EXPECT_FALSE(sk::logger::FileLogger::Instance().Initialize(config));
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find("Failed to create sk_meta directory: " + targetPath),
            std::string::npos);
}

TEST_F(TestSkLogContext, FileLogger_RegistrationFailureReportsHandleAndPath) {
  const std::string modelId = "model_logger_registration_failure";
  const std::string targetPath = GetSkMetaPath(modelId) + "/super_kernel.log";
  ASSERT_TRUE(CreateDirectoryRecursive(targetPath));

  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelId = modelId;
  EXPECT_FALSE(sk::logger::FileLogger::Instance().Initialize(config));
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find("Failed to register log file for model: " + modelId +
                                                            ", handle: " + modelId + ", path: " + targetPath),
            std::string::npos);
}

TEST_F(TestSkLogContext, LogContextGuard_RoutesNestedModelsAndRestoresPreviousState) {
  const std::string originalModelId = "model_route_original";
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelId = originalModelId;
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));
  sk::logger::FileLogger::SetCurrentModelId(originalModelId);
  ASSERT_TRUE(sk::logger::FileHandleManager::Instance().SwitchToFile(originalModelId));

  {
    sk::logger::LogContextGuard outerContext("model_route_outer");
    EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), "model_route_outer");
    EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");

    {
      sk::logger::LogContextGuard innerContext("model_route_inner");
      EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), "model_route_inner");
    }
    EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), "model_route_outer");
  }

  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), originalModelId);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), originalModelId);
}

TEST_F(TestSkLogContext, LogContextGuard_DisabledLoggerLeavesContextUnchanged) {
  sk::logger::FileLogger::SetCurrentModelId("model_disabled_original");
  sk::logger::FileLogger::Instance().SetEnabled(false);

  sk::logger::LogContextGuard context("model_disabled_target");
  EXPECT_FALSE(context.IsActive());
  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), "model_disabled_original");
}

TEST_F(TestSkLogContext, LogContextGuard_InitializationFailureDoesNotWriteToPreviousModel) {
  const std::string originalModelId = "model_init_failure_original";
  const std::string targetModelId = "model_init_failure_target";
  const std::string logMarker = "target model log must not reach the previous model";
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelId = originalModelId;
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));
  sk::logger::FileLogger::SetCurrentModelId(originalModelId);
  sk::logger::FileHandleManager::Instance().SwitchToDefault();

  const std::string targetPath = GetSkMetaPath(targetModelId);
  std::ofstream blockedTargetPath(targetPath);
  ASSERT_TRUE(blockedTargetPath.good());
  blockedTargetPath.close();

  {
    sk::logger::LogContextGuard context(targetModelId);
    EXPECT_TRUE(context.IsActive());
    EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), targetModelId);
    EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");
    SK_LOGI("%s", logMarker.c_str());
  }
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find("directory: " + targetPath), std::string::npos);

  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), originalModelId);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");

  std::ifstream originalLog(GetSkMetaPath(originalModelId) + "/super_kernel.log");
  ASSERT_TRUE(originalLog.good());
  std::string content((std::istreambuf_iterator<char>(originalLog)), std::istreambuf_iterator<char>());
  EXPECT_EQ(content.find(logMarker), std::string::npos);
}

TEST_F(TestSkLogContext, FileLogger_ContextDirectoryFailureReportsPath) {
  const std::string originalModelId = "model_context_directory_original";
  const std::string targetModelId(256, 'b');
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelId = originalModelId;
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));

  const std::string targetPath = GetSkMetaPath(targetModelId);

  EXPECT_EQ(sk::logger::FileLogger::Instance().CreateContext("context.log", targetModelId), nullptr);
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find("Failed to create directory for context: " + targetPath),
            std::string::npos);
}

TEST_F(TestSkLogContext, LogContextGuard_ConcurrentThreadsKeepModelIdsIsolated) {
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelId = "model_thread_config";
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));

  std::atomic<uint32_t> readyCount{0U};
  std::array<bool, 2> routeChecks{false, false};
  std::array<bool, 2> restoreChecks{false, false};
  std::vector<std::thread> workers;
  for (size_t i = 0; i < routeChecks.size(); ++i) {
    workers.emplace_back([i, &readyCount, &routeChecks, &restoreChecks]() {
      const std::string previousModelId = "model_thread_previous_" + std::to_string(i);
      const std::string targetModelId = "model_thread_target_" + std::to_string(i);
      sk::logger::FileLogger::SetCurrentModelId(previousModelId);
      sk::logger::FileHandleManager::Instance().SwitchToDefault();
      {
        sk::logger::LogContextGuard context(targetModelId);
        readyCount.fetch_add(1U, std::memory_order_release);
        while (readyCount.load(std::memory_order_acquire) < routeChecks.size()) {
          std::this_thread::yield();
        }
        routeChecks[i] = sk::logger::FileLogger::GetCurrentModelId() == targetModelId &&
                         sk::logger::FileHandleManager::Instance().GetCurrentHandle() == "default";
      }
      restoreChecks[i] = sk::logger::FileLogger::GetCurrentModelId() == previousModelId &&
                         sk::logger::FileHandleManager::Instance().GetCurrentHandle() == "default";
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_TRUE(routeChecks[0]);
  EXPECT_TRUE(routeChecks[1]);
  EXPECT_TRUE(restoreChecks[0]);
  EXPECT_TRUE(restoreChecks[1]);
}

TEST_F(TestSkLogContext, LogContextGuard_FileAndModelContextsRestoreInNestedOrder) {
  const std::string originalModelId = "model_context_original";
  const std::string nestedModelId = "model_context_nested";
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelId = originalModelId;
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));
  sk::logger::FileLogger::SetCurrentModelId(originalModelId);
  sk::logger::FileHandleManager::Instance().SwitchToDefault();

  auto fileContext = sk::logger::FileLogger::Instance().CreateContext("nested_context.log", originalModelId);
  ASSERT_NE(fileContext, nullptr);
  const std::string fileHandle = sk::logger::FileHandleManager::Instance().GetCurrentHandle();
  EXPECT_NE(fileHandle, "default");

  {
    sk::logger::LogContextGuard modelContext(nestedModelId);
    EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), nestedModelId);
    EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");
  }

  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), originalModelId);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), fileHandle);
  fileContext.reset();
  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), originalModelId);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");
}

TEST_F(TestSkLogContext, LogContextGuard_MoveTransfersContextOwnership) {
  const std::string originalModelId = "model_move_original";
  const std::string targetModelId = "model_move_target";
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelId = originalModelId;
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));
  sk::logger::FileLogger::SetCurrentModelId(originalModelId);
  sk::logger::FileHandleManager::Instance().SwitchToDefault();

  {
    sk::logger::LogContextGuard source(targetModelId);
    sk::logger::LogContextGuard moved(std::move(source));
    EXPECT_FALSE(source.IsActive());
    EXPECT_TRUE(moved.IsActive());
    EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), targetModelId);
  }
  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), originalModelId);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");

  {
    sk::logger::FileLogger::Instance().SetEnabled(false);
    sk::logger::LogContextGuard moved("model_move_inactive");
    sk::logger::FileLogger::Instance().SetEnabled(true);
    sk::logger::LogContextGuard source(targetModelId);
    moved = std::move(source);
    EXPECT_FALSE(source.IsActive());
    EXPECT_TRUE(moved.IsActive());
  }
  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), originalModelId);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");
}

TEST_F(TestSkLogContext, ExplicitLogContextWritesUnderCanonicalModelIdDirectory) {
  const std::string modelId = "model_explicit_log_context";
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelId = modelId;
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));

  {
    auto context = sk::logger::FileLogger::Instance().CreateContext("explicit.log", modelId);
    ASSERT_NE(context, nullptr);
    SK_LOGI("log with explicit model id");
  }

  std::ifstream file(GetSkMetaPath(modelId) + "/explicit.log");
  ASSERT_TRUE(file.good());
  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("log with explicit model id"), std::string::npos);
}
