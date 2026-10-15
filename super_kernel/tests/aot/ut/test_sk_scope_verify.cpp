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

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits.h>
#include <limits>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

#define private public
#define protected public
#include "super_kernel.h"
#include "sk_graph.h"
#include "sk_lock_detector.h"
#include "sk_log.h"
#include "sk_scope_verify.h"
#include "sk_verify_graph.h"

namespace {

aclskScopeVerifyNodeInfo MakeScopeVerifyComputeNode(int64_t taskId, int64_t streamId, int32_t scopeId,
                                                    aclskScopeVerifyKernelType kernelType, uint32_t numBlocks,
                                                    uint32_t taskRatio0, uint32_t taskRatio1, int32_t scheMode) {
  aclskScopeVerifyNodeInfo node{};
  node.taskId = taskId;
  node.streamId = streamId;
  node.scopeId = scopeId;
  node.taskType = ACLSK_SCOPE_VERIFY_NODE_COMPUTE;
  node.kernelType = kernelType;
  node.numBlocks = numBlocks;
  node.taskRatio[0] = taskRatio0;
  node.taskRatio[1] = taskRatio1;
  node.scheMode = scheMode;
  return node;
}

aclskScopeVerifyNodeInfo MakeScopeVerifyEventNode(int64_t taskId, int64_t streamId, int64_t eventId, int32_t scopeId,
                                                  aclskScopeVerifyNodeType taskType) {
  aclskScopeVerifyNodeInfo node{};
  node.taskId = taskId;
  node.streamId = streamId;
  node.eventId = eventId;
  node.scopeId = scopeId;
  node.taskType = taskType;
  return node;
}

constexpr const char *SCOPE_VERIFY_LOG_CHILD_ENV = "SK_SCOPE_VERIFY_LOG_CHILD";
constexpr const char *SCOPE_VERIFY_LOG_CHILD_FILTER =
    "--gtest_filter=ScopeVerifyLogIsolationTest.ChildAclskScopeVerifyDoesNotCreateSkMeta";

std::filesystem::path CurrentSkMetaPidDir() {
  return std::filesystem::path("sk_meta") / std::to_string(getpid());
}

std::filesystem::path ModelSkMetaDir(const std::string &modelId) {
  return CurrentSkMetaPidDir() / SanitizePathComponent(modelId);
}

std::string FileContent(const std::filesystem::path &filePath) {
  std::ifstream file(filePath);
  std::ostringstream oss;
  oss << file.rdbuf();
  return oss.str();
}

const char *BoolToString(bool value) {
  return value ? "true" : "false";
}

std::string CurrentExecutablePath() {
  char path[PATH_MAX] = {};
  ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (len <= 0) {
    return "";
  }
  path[len] = '\0';
  return std::string(path);
}

int RunScopeVerifyLogChildProcess() {
  std::string executable = CurrentExecutablePath();
  if (executable.empty()) {
    return -1;
  }

  pid_t pid = fork();
  if (pid < 0) {
    return -1;
  }
  if (pid == 0) {
    setenv(SCOPE_VERIFY_LOG_CHILD_ENV, "1", 1);
    unsetenv("ASCEND_OP_COMPILE_SAVE_KERNEL_META");
    execl(executable.c_str(), executable.c_str(), SCOPE_VERIFY_LOG_CHILD_FILTER, "--gtest_color=no", nullptr);
    _exit(127);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    return -1;
  }
  if (!WIFEXITED(status)) {
    return -1;
  }
  return WEXITSTATUS(status);
}

class ScopedLockDetectorCoreNums {
 public:
  ScopedLockDetectorCoreNums(int64_t cubeNum, int64_t vecNum)
      : oldCubeNum_(LockDetector::deviceRealCubeNum), oldVecNum_(LockDetector::deviceRealVecNum) {
    LockDetector::deviceRealCubeNum = cubeNum;
    LockDetector::deviceRealVecNum = vecNum;
  }

  ~ScopedLockDetectorCoreNums() {
    LockDetector::deviceRealCubeNum = oldCubeNum_;
    LockDetector::deviceRealVecNum = oldVecNum_;
  }

 private:
  int64_t oldCubeNum_;
  int64_t oldVecNum_;
};

class ScopedModelFileLog {
 public:
  explicit ScopedModelFileLog(std::string modelId) : modelId_(std::move(modelId)) {
    RemoveModelLogDir();
  }

  ~ScopedModelFileLog() {
    sk::logger::FileLogger::SetCurrentModelId("");
    sk::logger::FileLogger::Instance().SetEnabled(false);
    sk::logger::FileHandleManager::Instance().SwitchToDefault();
    sk::logger::FileHandleManager::Instance().CloseFile(ModelDefaultHandle());
    sk::logger::FileHandleManager::Instance().CloseFile(ModelScopeSplitHandle());
    RemoveModelLogDir();
  }

  void InitializeEnabledLogger() const {
    sk::logger::LoggerConfig config;
    config.enabled = true;
    config.modelId = modelId_;
    config.minLevel = sk::logger::LogLevel::DEBUG;
    ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));
    sk::logger::FileLogger::SetCurrentModelId(modelId_);
    sk::logger::FileHandleManager::Instance().SwitchToDefault();
  }

  std::filesystem::path ModelDir() const {
    return ModelSkMetaDir(modelId_);
  }

  std::filesystem::path DefaultLogPath() const {
    return ModelDir() / "super_kernel.log";
  }

  std::filesystem::path ScopeSplitLogPath() const {
    return ModelDir() / "sk_scope_split.log";
  }

 private:
  std::string SanitizedModelId() const {
    return SanitizePathComponent(modelId_);
  }

  std::string ModelDefaultHandle() const {
    return SanitizedModelId();
  }

  std::string ModelScopeSplitHandle() const {
    return SanitizedModelId() + "_sk_scope_split.log";
  }

  void RemoveModelLogDir() const {
    std::error_code ec;
    std::filesystem::remove_all(ModelDir(), ec);
  }

  std::string modelId_;
};

void ExpectComputeGraphTopology(const SuperKernelGraph &graph) {
  ASSERT_EQ(graph.graphMap.size(), 3);
  ASSERT_EQ(graph.streams.size(), 2);
  ASSERT_EQ(graph.headNodes.size(), 2);
  EXPECT_EQ(graph.headNodes[0], 10);
  EXPECT_EQ(graph.headNodes[1], 12);
  ASSERT_EQ(graph.nodeSizeInStream.size(), 2);
  EXPECT_EQ(graph.nodeSizeInStream[0], 2);
  EXPECT_EQ(graph.nodeSizeInStream[1], 1);
}

void ExpectCubeNodeFields(const SuperKernelBaseNode *cubeNode) {
  ASSERT_NE(cubeNode, nullptr);
  EXPECT_EQ(cubeNode->GetNodeType(), SkNodeType::NODE_KERNEL);
  EXPECT_EQ(cubeNode->GetStreamId(), 100);
  EXPECT_EQ(cubeNode->GetStreamIdxInGraph(), 0);
  EXPECT_EQ(cubeNode->GetNodeIdxInStream(), 0);
  EXPECT_EQ(cubeNode->GetPreNodeId(), INVALID_TASK_ID);
  EXPECT_EQ(cubeNode->GetNextNodeId(), 11);
  EXPECT_EQ(cubeNode->GetKernelType(), SkKernelType::AIC_ONLY);
  EXPECT_EQ(cubeNode->GetNodeInfos().kernelInfos.kernelTypeInt, static_cast<uint32_t>(ACLSK_SCOPE_VERIFY_KERNEL_CUBE));
  EXPECT_EQ(cubeNode->GetNumBlocks(), 8);
  EXPECT_EQ(cubeNode->GetCubeNum(), 8);
  EXPECT_EQ(cubeNode->GetVecNum(), 0);
  EXPECT_EQ(cubeNode->GetNodeInfos().kernelInfos.taskRatio[0], 1);
  EXPECT_EQ(cubeNode->GetNodeInfos().kernelInfos.taskRatio[1], 0);
  EXPECT_FALSE(cubeNode->IsScheModeOn());
}

void ExpectVectorNodeFields(const SuperKernelBaseNode *vectorNode) {
  ASSERT_NE(vectorNode, nullptr);
  EXPECT_EQ(vectorNode->GetNodeType(), SkNodeType::NODE_KERNEL);
  EXPECT_EQ(vectorNode->GetStreamId(), 100);
  EXPECT_EQ(vectorNode->GetStreamIdxInGraph(), 0);
  EXPECT_EQ(vectorNode->GetNodeIdxInStream(), 1);
  EXPECT_EQ(vectorNode->GetPreNodeId(), 10);
  EXPECT_EQ(vectorNode->GetNextNodeId(), INVALID_TASK_ID);
  EXPECT_EQ(vectorNode->GetKernelType(), SkKernelType::AIV_ONLY);
  EXPECT_EQ(vectorNode->GetNodeInfos().kernelInfos.kernelTypeInt,
            static_cast<uint32_t>(ACLSK_SCOPE_VERIFY_KERNEL_VECTOR));
  EXPECT_EQ(vectorNode->GetNumBlocks(), 16);
  EXPECT_EQ(vectorNode->GetCubeNum(), 0);
  EXPECT_EQ(vectorNode->GetVecNum(), 16);
  EXPECT_EQ(vectorNode->GetNodeInfos().kernelInfos.taskRatio[0], 0);
  EXPECT_EQ(vectorNode->GetNodeInfos().kernelInfos.taskRatio[1], 1);
  EXPECT_TRUE(vectorNode->IsScheModeOn());
}

void ExpectMixNodeFields(const SuperKernelBaseNode *mixNode) {
  ASSERT_NE(mixNode, nullptr);
  EXPECT_EQ(mixNode->GetNodeType(), SkNodeType::NODE_KERNEL);
  EXPECT_EQ(mixNode->GetStreamId(), 101);
  EXPECT_EQ(mixNode->GetStreamIdxInGraph(), 1);
  EXPECT_EQ(mixNode->GetNodeIdxInStream(), 0);
  EXPECT_EQ(mixNode->GetPreNodeId(), INVALID_TASK_ID);
  EXPECT_EQ(mixNode->GetNextNodeId(), INVALID_TASK_ID);
  EXPECT_EQ(mixNode->GetKernelType(), SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(mixNode->GetNodeInfos().kernelInfos.kernelTypeInt, static_cast<uint32_t>(ACLSK_SCOPE_VERIFY_KERNEL_MIX));
  EXPECT_EQ(mixNode->GetNumBlocks(), 4);
  EXPECT_EQ(mixNode->GetCubeNum(), 4);
  EXPECT_EQ(mixNode->GetVecNum(), 8);
  EXPECT_EQ(mixNode->GetNodeInfos().kernelInfos.taskRatio[0], 1);
  EXPECT_EQ(mixNode->GetNodeInfos().kernelInfos.taskRatio[1], 2);
  EXPECT_TRUE(mixNode->IsScheModeOn());
}

void ExpectComputeScopeMapping(const std::vector<SuperKernelScopeInfo> &initialScopes) {
  ASSERT_EQ(initialScopes.size(), 1);
  const auto &scopeNodes = initialScopes[0].GetNodes();
  ASSERT_EQ(scopeNodes.size(), 3);
  EXPECT_EQ(scopeNodes[0]->GetNodeId(), 10);
  EXPECT_EQ(scopeNodes[1]->GetNodeId(), 11);
  EXPECT_EQ(scopeNodes[2]->GetNodeId(), 12);
  const auto &streamInfos = initialScopes[0].GetScopeStreamInfos();
  ASSERT_EQ(streamInfos.size(), 2);
  EXPECT_EQ(streamInfos[0].streamIdx, 0);
  EXPECT_EQ(streamInfos[0].headNodeIdx, 10);
  EXPECT_EQ(streamInfos[0].tailNodeIdx, 11);
  EXPECT_EQ(streamInfos[0].nodeSize, 2);
  EXPECT_EQ(streamInfos[1].streamIdx, 1);
  EXPECT_EQ(streamInfos[1].headNodeIdx, 12);
  EXPECT_EQ(streamInfos[1].tailNodeIdx, 12);
  EXPECT_EQ(streamInfos[1].nodeSize, 1);
}

void ExpectEventGraphTopology(const SuperKernelGraph &graph) {
  ASSERT_EQ(graph.graphMap.size(), 2);
  ASSERT_EQ(graph.streams.size(), 2);
  ASSERT_EQ(graph.headNodes.size(), 2);
  EXPECT_EQ(graph.headNodes[0], 20);
  EXPECT_EQ(graph.headNodes[1], 21);
}

void ExpectNotifyNodeFields(const SuperKernelBaseNode *notifyNode) {
  ASSERT_NE(notifyNode, nullptr);
  EXPECT_EQ(notifyNode->GetNodeType(), SkNodeType::NODE_NOTIFY);
  EXPECT_EQ(notifyNode->GetStreamId(), 200);
  EXPECT_EQ(notifyNode->GetStreamIdxInGraph(), 0);
  EXPECT_EQ(notifyNode->GetNodeIdxInStream(), 0);
  EXPECT_EQ(notifyNode->GetPreNodeId(), INVALID_TASK_ID);
  EXPECT_EQ(notifyNode->GetNextNodeId(), INVALID_TASK_ID);
  EXPECT_EQ(notifyNode->GetEventId(), 300);
  ASSERT_EQ(notifyNode->GetCorrespondingWaitNodeIds().size(), 1);
  EXPECT_EQ(notifyNode->GetCorrespondingWaitNodeIds()[0], 21);
}

void ExpectWaitNodeFields(const SuperKernelBaseNode *waitNode) {
  ASSERT_NE(waitNode, nullptr);
  EXPECT_EQ(waitNode->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_EQ(waitNode->GetStreamId(), 201);
  EXPECT_EQ(waitNode->GetStreamIdxInGraph(), 1);
  EXPECT_EQ(waitNode->GetNodeIdxInStream(), 0);
  EXPECT_EQ(waitNode->GetPreNodeId(), INVALID_TASK_ID);
  EXPECT_EQ(waitNode->GetNextNodeId(), INVALID_TASK_ID);
  EXPECT_EQ(waitNode->GetEventId(), 300);
  EXPECT_EQ(waitNode->GetCorrespondingNotifyNodeId(), 20);
}

void ExpectEventScopeMapping(const std::vector<SuperKernelScopeInfo> &initialScopes) {
  ASSERT_EQ(initialScopes.size(), 1);
  const auto &scopeNodes = initialScopes[0].GetNodes();
  ASSERT_EQ(scopeNodes.size(), 2);
  EXPECT_EQ(scopeNodes[0]->GetNodeId(), 20);
  EXPECT_EQ(scopeNodes[1]->GetNodeId(), 21);
  const auto &streamInfos = initialScopes[0].GetScopeStreamInfos();
  ASSERT_EQ(streamInfos.size(), 2);
  EXPECT_EQ(streamInfos[0].streamIdx, 0);
  EXPECT_EQ(streamInfos[0].headNodeIdx, 20);
  EXPECT_EQ(streamInfos[0].tailNodeIdx, 20);
  EXPECT_EQ(streamInfos[0].nodeSize, 1);
  EXPECT_EQ(streamInfos[1].streamIdx, 1);
  EXPECT_EQ(streamInfos[1].headNodeIdx, 21);
  EXPECT_EQ(streamInfos[1].tailNodeIdx, 21);
  EXPECT_EQ(streamInfos[1].nodeSize, 1);
}

}  // namespace

class ScopeVerifyTest : public testing::Test {
 protected:
  SuperKernelGraph graph;
};

TEST(ScopeVerifyLogIsolationTest, ChildAclskScopeVerifyDoesNotCreateSkMeta) {
  if (std::getenv(SCOPE_VERIFY_LOG_CHILD_ENV) == nullptr) {
    return;
  }

  ScopedLockDetectorCoreNums coreNums(25, 32);
  const std::filesystem::path skMetaPidDir = CurrentSkMetaPidDir();
  std::error_code ec;
  std::filesystem::remove_all(skMetaPidDir, ec);
  ASSERT_FALSE(std::filesystem::exists(skMetaPidDir));
  ASSERT_FALSE(sk::logger::FileLogger::Instance().IsInitialized());
  ASSERT_FALSE(sk::logger::FileLogger::Instance().IsEnabled());
  fprintf(stderr,
          "\n[ScopeVerifyLogIsolation] before aclskScopeVerify: initialized=%s, enabled=%s, skMetaPidDir=%s, "
          "exists=%s\n",
          BoolToString(sk::logger::FileLogger::Instance().IsInitialized()),
          BoolToString(sk::logger::FileLogger::Instance().IsEnabled()), skMetaPidDir.c_str(),
          BoolToString(std::filesystem::exists(skMetaPidDir)));

  std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
      MakeScopeVerifyComputeNode(1, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 4, 1, 0, 0),
      MakeScopeVerifyComputeNode(2, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_VECTOR, 4, 0, 1, 0),
      MakeScopeVerifyComputeNode(3, 1, 2, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 2, 1, 0, 0),
  };
  aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
  size_t realSplitResultCount = 123;

  aclError ret = aclskScopeVerify(&verifyGraph, 0, nullptr, &realSplitResultCount);
  fprintf(stderr,
          "[ScopeVerifyLogIsolation] after aclskScopeVerify: ret=%d, realSplitResultCount=%zu, initialized=%s, "
          "enabled=%s, skMetaPidDir=%s, exists=%s\n",
          ret, realSplitResultCount, BoolToString(sk::logger::FileLogger::Instance().IsInitialized()),
          BoolToString(sk::logger::FileLogger::Instance().IsEnabled()), skMetaPidDir.c_str(),
          BoolToString(std::filesystem::exists(skMetaPidDir)));

  std::string dlogContent = ut_log::LogBuffer::Instance().GetContent();
  fprintf(stderr, "[ScopeVerifyLogIsolation] captured dlog buffer begin\n%s", dlogContent.c_str());
  fprintf(stderr, "[ScopeVerifyLogIsolation] captured dlog buffer end\n\n");
  fflush(stderr);

  ASSERT_EQ(ret, ACL_SUCCESS);

  EXPECT_EQ(realSplitResultCount, 0);
  EXPECT_FALSE(sk::logger::FileLogger::Instance().IsInitialized());
  EXPECT_FALSE(sk::logger::FileLogger::Instance().IsEnabled());
  EXPECT_FALSE(std::filesystem::exists(skMetaPidDir));
}

TEST(ScopeVerifyLogIsolationTest, AclskScopeVerifyDoesNotCreateSkMetaInFreshProcess) {
  ASSERT_EQ(RunScopeVerifyLogChildProcess(), 0);
}

TEST(ScopeVerifyLogIsolationTest, AclskScopeVerifyDoesNotWriteFileLogsWithStaleThreadModelId) {
  ScopedLockDetectorCoreNums coreNums(25, 32);
  ScopedModelFileLog modelLog("scope_verify_same_thread_old_model");
  modelLog.InitializeEnabledLogger();

  std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
      MakeScopeVerifyComputeNode(1, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 4, 1, 0, 0),
      MakeScopeVerifyComputeNode(2, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_VECTOR, 4, 0, 1, 0),
      MakeScopeVerifyComputeNode(3, 1, 2, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 2, 1, 0, 0),
  };
  aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
  size_t realSplitResultCount = 123;
  ut_log::LogBuffer::Instance().Clear();

  ASSERT_EQ(aclskScopeVerify(&verifyGraph, 0, nullptr, &realSplitResultCount), ACL_SUCCESS);

  const std::string dlogContent = ut_log::LogBuffer::Instance().GetContent();
  EXPECT_NE(dlogContent.find("Scope split results"), std::string::npos);
  EXPECT_EQ(FileContent(modelLog.DefaultLogPath()).find("Scope split results"), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(modelLog.ScopeSplitLogPath()));

  SK_LOGI("optimize file log remains enabled after same-thread scope verify");
  EXPECT_NE(FileContent(modelLog.DefaultLogPath()).find("optimize file log remains enabled"), std::string::npos);
}

TEST(ScopeVerifyLogIsolationTest, AclskScopeVerifyDoesNotUseOtherThreadModelIdFallback) {
  ScopedLockDetectorCoreNums coreNums(25, 32);
  ScopedModelFileLog modelLog("scope_verify_other_thread_old_model");
  modelLog.InitializeEnabledLogger();

  std::thread worker([&modelLog]() {
    std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
        MakeScopeVerifyComputeNode(1, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 4, 1, 0, 0),
        MakeScopeVerifyComputeNode(2, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_VECTOR, 4, 0, 1, 0),
        MakeScopeVerifyComputeNode(3, 1, 2, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 2, 1, 0, 0),
    };
    aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
    size_t realSplitResultCount = 123;

    EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), "");
    EXPECT_EQ(aclskScopeVerify(&verifyGraph, 0, nullptr, &realSplitResultCount), ACL_SUCCESS);
    EXPECT_EQ(realSplitResultCount, 0);
  });
  worker.join();

  EXPECT_EQ(FileContent(modelLog.DefaultLogPath()).find("Scope split results"), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(modelLog.ScopeSplitLogPath()));

  SK_LOGI("optimize file log remains enabled after other-thread scope verify");
  EXPECT_NE(FileContent(modelLog.DefaultLogPath()).find("optimize file log remains enabled"), std::string::npos);
}

TEST_F(ScopeVerifyTest, GraphBuilder_MapsComputeKernelCoreFields) {
  std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
      MakeScopeVerifyComputeNode(10, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 8, 1, 0, 0),
      MakeScopeVerifyComputeNode(11, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_VECTOR, 16, 0, 1, 1),
      MakeScopeVerifyComputeNode(12, 101, 7, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 1, 2, 1),
  };
  aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
  std::vector<SuperKernelScopeInfo> initialScopes;
  std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById;
  ScopeVerifyGraphBuilder builder;

  ASSERT_EQ(builder.Build(&verifyGraph, graph, initialScopes, verifyNodesById), ACL_SUCCESS);

  ASSERT_NO_FATAL_FAILURE(ExpectComputeGraphTopology(graph));
  ASSERT_NO_FATAL_FAILURE(ExpectCubeNodeFields(graph.GetNodeById(10)));
  ASSERT_NO_FATAL_FAILURE(ExpectVectorNodeFields(graph.GetNodeById(11)));
  ASSERT_NO_FATAL_FAILURE(ExpectMixNodeFields(graph.GetNodeById(12)));
  ASSERT_NO_FATAL_FAILURE(ExpectComputeScopeMapping(initialScopes));

  ASSERT_EQ(verifyNodesById.size(), 3);
  EXPECT_EQ(verifyNodesById[10], &verifyNodes[0]);
  EXPECT_EQ(verifyNodesById[11], &verifyNodes[1]);
  EXPECT_EQ(verifyNodesById[12], &verifyNodes[2]);
}

TEST_F(ScopeVerifyTest, GraphBuilder_RejectsNonNullNodeExtendInfo) {
  auto verifyNode = MakeScopeVerifyComputeNode(10, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 1, 2, 0);
  uint32_t extendInfo = 0;
  verifyNode.extendInfo = &extendInfo;
  aclskScopeVerifyGraphInfo verifyGraph{&verifyNode, 1, 0, nullptr};
  std::vector<SuperKernelScopeInfo> initialScopes;
  std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById;
  ScopeVerifyGraphBuilder builder;

  EXPECT_EQ(builder.Build(&verifyGraph, graph, initialScopes, verifyNodesById), ACL_ERROR_INVALID_PARAM);
}

TEST_F(ScopeVerifyTest, GraphBuilder_IgnoresCoreLimitWhenDynamicFlagIsUnset) {
  auto verifyNode = MakeScopeVerifyComputeNode(11, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 4, 1, 0, 0);
  verifyNode.flag = 2U;
  verifyNode.coreLimit[0] = 20;
  verifyNode.coreLimit[1] = 32;
  aclskScopeVerifyGraphInfo verifyGraph{&verifyNode, 1, 0, nullptr};
  std::vector<SuperKernelScopeInfo> initialScopes;
  std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById;
  ScopeVerifyGraphBuilder builder;

  ASSERT_EQ(builder.Build(&verifyGraph, graph, initialScopes, verifyNodesById), ACL_SUCCESS);

  auto *node = graph.GetNodeById(11);
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->GetCubeNum(), 4U);
  EXPECT_EQ(node->GetVecNum(), 0U);
}

TEST_F(ScopeVerifyTest, GraphBuilder_UsesDynamicCoreLimitFromNodeFields) {
  auto verifyNode = MakeScopeVerifyComputeNode(12, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 4, 1, 0, 0);
  verifyNode.flag = 1U;
  verifyNode.coreLimit[0] = 20;
  verifyNode.coreLimit[1] = 32;
  aclskScopeVerifyGraphInfo verifyGraph{&verifyNode, 1, 0, nullptr};
  std::vector<SuperKernelScopeInfo> initialScopes;
  std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById;
  ScopeVerifyGraphBuilder builder;

  ASSERT_EQ(builder.Build(&verifyGraph, graph, initialScopes, verifyNodesById), ACL_SUCCESS);

  auto *node = graph.GetNodeById(12);
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->GetCubeNum(), 20U);
  EXPECT_EQ(node->GetVecNum(), 32U);
}

TEST_F(ScopeVerifyTest, GraphBuilder_RejectsNegativeDynamicCoreLimit) {
  auto verifyNode = MakeScopeVerifyComputeNode(13, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 4, 1, 0, 0);
  verifyNode.flag = 1U;
  verifyNode.coreLimit[0] = -1;
  verifyNode.coreLimit[1] = 32;
  aclskScopeVerifyGraphInfo verifyGraph{&verifyNode, 1, 0, nullptr};
  std::vector<SuperKernelScopeInfo> initialScopes;
  std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById;
  ScopeVerifyGraphBuilder builder;

  EXPECT_EQ(builder.Build(&verifyGraph, graph, initialScopes, verifyNodesById), ACL_ERROR_INVALID_PARAM);
}

TEST_F(ScopeVerifyTest, GraphBuilder_RejectsDynamicCoreLimitForNonAiCoreNode) {
  auto verifyNode = MakeScopeVerifyEventNode(14, 100, 1, 7, ACLSK_SCOPE_VERIFY_NODE_WAIT);
  verifyNode.flag = 1U;
  verifyNode.coreLimit[0] = 20;
  verifyNode.coreLimit[1] = 32;
  aclskScopeVerifyGraphInfo verifyGraph{&verifyNode, 1, 0, nullptr};
  std::vector<SuperKernelScopeInfo> initialScopes;
  std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById;
  ScopeVerifyGraphBuilder builder;

  EXPECT_EQ(builder.Build(&verifyGraph, graph, initialScopes, verifyNodesById), ACL_ERROR_INVALID_PARAM);
}

TEST_F(ScopeVerifyTest, GraphBuilder_MapsZeroBlockComputeToDefaultNode) {
  std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
      MakeScopeVerifyComputeNode(13, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_NO_AICORE, 0, 0, 0, 0),
  };
  aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
  std::vector<SuperKernelScopeInfo> initialScopes;
  std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById;
  ScopeVerifyGraphBuilder builder;

  ASSERT_EQ(builder.Build(&verifyGraph, graph, initialScopes, verifyNodesById), ACL_SUCCESS);

  auto *defaultNode = graph.GetNodeById(13);
  ASSERT_NE(defaultNode, nullptr);
  EXPECT_EQ(defaultNode->GetNodeType(), SkNodeType::NODE_DEFAULT);
  EXPECT_EQ(defaultNode->GetNumBlocks(), 0);
  ASSERT_EQ(initialScopes.size(), 1);
  ASSERT_EQ(initialScopes[0].GetNodes().size(), 1);
  EXPECT_EQ(initialScopes[0].GetNodes()[0], defaultNode);
  ASSERT_EQ(verifyNodesById.size(), 1);
  EXPECT_EQ(verifyNodesById[13], &verifyNodes[0]);
}

TEST_F(ScopeVerifyTest, GraphBuilder_SkipsNegativeScopeIdNodeFromInitialScopes) {
  std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
      MakeScopeVerifyComputeNode(13, 100, -1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 4, 1, 0, 0),
      MakeScopeVerifyComputeNode(14, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_VECTOR, 8, 0, 1, 0),
  };
  aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
  std::vector<SuperKernelScopeInfo> initialScopes;
  std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById;
  ScopeVerifyGraphBuilder builder;

  ASSERT_EQ(builder.Build(&verifyGraph, graph, initialScopes, verifyNodesById), ACL_SUCCESS);

  auto *outOfScopeNode = graph.GetNodeById(13);
  ASSERT_NE(outOfScopeNode, nullptr);
  auto *inScopeNode = graph.GetNodeById(14);
  ASSERT_NE(inScopeNode, nullptr);
  EXPECT_EQ(outOfScopeNode->GetNextNodeId(), 14);
  EXPECT_EQ(inScopeNode->GetPreNodeId(), 13);

  ASSERT_EQ(initialScopes.size(), 1);
  const auto &scopeNodes = initialScopes[0].GetNodes();
  ASSERT_EQ(scopeNodes.size(), 1);
  EXPECT_EQ(scopeNodes[0], inScopeNode);

  ASSERT_EQ(verifyNodesById.size(), 2);
  EXPECT_EQ(verifyNodesById[13], &verifyNodes[0]);
  EXPECT_EQ(verifyNodesById[14], &verifyNodes[1]);
}

TEST_F(ScopeVerifyTest, GraphBuilder_MapsMixTaskRatioToCoreFields) {
  std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
      MakeScopeVerifyComputeNode(30, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 0, 1, 0),
      MakeScopeVerifyComputeNode(31, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 1, 0, 0),
      MakeScopeVerifyComputeNode(32, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 1, 1, 0),
      MakeScopeVerifyComputeNode(33, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 1, 2, 0),
  };
  aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
  std::vector<SuperKernelScopeInfo> initialScopes;
  std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById;
  ScopeVerifyGraphBuilder builder;

  ASSERT_EQ(builder.Build(&verifyGraph, graph, initialScopes, verifyNodesById), ACL_SUCCESS);

  auto *mixAivNode = graph.GetNodeById(30);
  ASSERT_NE(mixAivNode, nullptr);
  EXPECT_EQ(mixAivNode->GetKernelType(), SkKernelType::MIX_AIV_1_0);
  EXPECT_EQ(mixAivNode->GetCubeNum(), 0);
  EXPECT_EQ(mixAivNode->GetVecNum(), 4);

  auto *mixAicNode = graph.GetNodeById(31);
  ASSERT_NE(mixAicNode, nullptr);
  EXPECT_EQ(mixAicNode->GetKernelType(), SkKernelType::MIX_AIC_1_0);
  EXPECT_EQ(mixAicNode->GetCubeNum(), 4);
  EXPECT_EQ(mixAicNode->GetVecNum(), 0);

  auto *mixAicAivNode = graph.GetNodeById(32);
  ASSERT_NE(mixAicAivNode, nullptr);
  EXPECT_EQ(mixAicAivNode->GetKernelType(), SkKernelType::MIX_AIC_1_1);
  EXPECT_EQ(mixAicAivNode->GetCubeNum(), 4);
  EXPECT_EQ(mixAicAivNode->GetVecNum(), 4);

  auto *mixAicAiv2Node = graph.GetNodeById(33);
  ASSERT_NE(mixAicAiv2Node, nullptr);
  EXPECT_EQ(mixAicAiv2Node->GetKernelType(), SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(mixAicAiv2Node->GetCubeNum(), 4);
  EXPECT_EQ(mixAicAiv2Node->GetVecNum(), 8);
}

TEST_F(ScopeVerifyTest, GraphBuilder_RejectsInvalidMixTaskRatio) {
  std::vector<aclskScopeVerifyNodeInfo> invalidMixNodes = {
      MakeScopeVerifyComputeNode(40, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 2, 1, 0),
      MakeScopeVerifyComputeNode(41, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 0, 2, 0),
      MakeScopeVerifyComputeNode(42, 100, 7, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 2, 0, 0),
  };

  for (auto &verifyNode : invalidMixNodes) {
    aclskScopeVerifyGraphInfo verifyGraph{&verifyNode, 1, 0, nullptr};
    std::vector<SuperKernelScopeInfo> initialScopes;
    std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById;
    SuperKernelGraph testGraph;
    ScopeVerifyGraphBuilder builder;

    EXPECT_EQ(builder.Build(&verifyGraph, testGraph, initialScopes, verifyNodesById), ACL_ERROR_INVALID_PARAM);
  }
}

TEST_F(ScopeVerifyTest, GraphBuilder_MapsNotifyWaitAndEventRelations) {
  std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
      MakeScopeVerifyEventNode(20, 200, 300, 3, ACLSK_SCOPE_VERIFY_NODE_NOTIFY),
      MakeScopeVerifyEventNode(21, 201, 300, 3, ACLSK_SCOPE_VERIFY_NODE_WAIT),
  };
  aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
  std::vector<SuperKernelScopeInfo> initialScopes;
  std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById;
  ScopeVerifyGraphBuilder builder;

  ASSERT_EQ(builder.Build(&verifyGraph, graph, initialScopes, verifyNodesById), ACL_SUCCESS);

  ASSERT_NO_FATAL_FAILURE(ExpectEventGraphTopology(graph));
  ASSERT_NO_FATAL_FAILURE(ExpectNotifyNodeFields(graph.GetNodeById(20)));
  ASSERT_NO_FATAL_FAILURE(ExpectWaitNodeFields(graph.GetNodeById(21)));
  ASSERT_NO_FATAL_FAILURE(ExpectEventScopeMapping(initialScopes));

  ASSERT_EQ(verifyNodesById.size(), 2);
  EXPECT_EQ(verifyNodesById[20], &verifyNodes[0]);
  EXPECT_EQ(verifyNodesById[21], &verifyNodes[1]);
}

TEST_F(ScopeVerifyTest, NoSplitGraphReturnsZeroResults) {
  std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
      MakeScopeVerifyComputeNode(1, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 4, 1, 0, 0),
      MakeScopeVerifyComputeNode(2, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_VECTOR, 4, 0, 1, 0),
      MakeScopeVerifyComputeNode(3, 1, 2, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 2, 1, 0, 0),
  };
  aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
  size_t realSplitResultCount = 123;

  ASSERT_EQ(aclskScopeVerify(&verifyGraph, 0, nullptr, &realSplitResultCount), ACL_SUCCESS);

  EXPECT_EQ(realSplitResultCount, 0);
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find(
                "aclskScopeVerify input node[0]: taskId=1, streamId=0, eventId=0, scopeId=1, taskType=2, "
                "kernelType=1, numBlocks=4, taskRatio={1,0}, scheMode=0, flag=0x0, coreLimit={0,0}"),
            std::string::npos);
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find(
                "aclskScopeVerify split result: No risk of Superkernel fusion caused by deadlock or Syncall"),
            std::string::npos);
}

TEST_F(ScopeVerifyTest, ScheModeSplitFillsBeforeNodeResult) {
  std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
      MakeScopeVerifyComputeNode(1, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 1, 2, 1),
      MakeScopeVerifyComputeNode(2, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 2, 1, 0, 1),
  };
  aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
  aclskScopeVerifySplitResult splitResults[2]{};
  size_t realSplitResultCount = 0;

  ASSERT_EQ(aclskScopeVerify(&verifyGraph, 2, splitResults, &realSplitResultCount), ACL_SUCCESS);

  ASSERT_EQ(realSplitResultCount, 1);
  EXPECT_EQ(splitResults[0].splitNode, &verifyNodes[1]);
  EXPECT_EQ(splitResults[0].splitType, ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE);
  EXPECT_EQ(splitResults[0].splitReason, ACLSK_SCOPE_VERIFY_SYNCALL_OP_DROP);
  EXPECT_EQ(splitResults[0].extendType, 0);
  EXPECT_EQ(splitResults[0].extendInfo, nullptr);
  EXPECT_NE(
      ut_log::LogBuffer::Instance().GetContent().find(
          "aclskScopeVerify split result[0]: taskId=2, scopeId=1, splitType=ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE, "
          "splitReason=ACLSK_SCOPE_VERIFY_SYNCALL_OP_DROP"),
      std::string::npos);
}

TEST_F(ScopeVerifyTest, ResultBufferTooSmallReturnsInvalidParamAndDoesNotWriteResults) {
  std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
      MakeScopeVerifyComputeNode(1, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 1, 2, 1),
      MakeScopeVerifyComputeNode(2, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 2, 1, 0, 1),
      MakeScopeVerifyComputeNode(3, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 1, 1, 0, 1),
  };
  aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
  aclskScopeVerifySplitResult splitResults[1]{};
  int extendSentinel = 0;
  splitResults[0].splitNode = &verifyNodes[0];
  splitResults[0].splitType = ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE;
  splitResults[0].splitReason = ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED;
  splitResults[0].extendType = 123;
  splitResults[0].extendInfo = &extendSentinel;
  size_t realSplitResultCount = 0;

  ASSERT_EQ(aclskScopeVerify(&verifyGraph, 1, splitResults, &realSplitResultCount), ACL_ERROR_INVALID_PARAM);

  ASSERT_EQ(realSplitResultCount, 2);
  EXPECT_EQ(splitResults[0].splitNode, &verifyNodes[0]);
  EXPECT_EQ(splitResults[0].splitType, ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE);
  EXPECT_EQ(splitResults[0].splitReason, ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED);
  EXPECT_EQ(splitResults[0].extendType, 123);
  EXPECT_EQ(splitResults[0].extendInfo, &extendSentinel);
}

TEST_F(ScopeVerifyTest, DeadlockSplitFillsExcludeNodeResult) {
  ScopedLockDetectorCoreNums coreNums(25, 32);
  std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
      MakeScopeVerifyComputeNode(1, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 12, 1, 0, 0),
      MakeScopeVerifyEventNode(2, 0, 100, 1, ACLSK_SCOPE_VERIFY_NODE_WAIT),
      MakeScopeVerifyComputeNode(3, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 15, 1, 0, 0),
      MakeScopeVerifyComputeNode(4, 1, 2, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 12, 1, 0, 0),
      MakeScopeVerifyEventNode(5, 1, 100, 2, ACLSK_SCOPE_VERIFY_NODE_NOTIFY),
  };
  aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
  aclskScopeVerifySplitResult splitResults[2]{};
  size_t realSplitResultCount = 0;

  ASSERT_EQ(aclskScopeVerify(&verifyGraph, 2, splitResults, &realSplitResultCount), ACL_SUCCESS);

  ASSERT_EQ(realSplitResultCount, 1);
  EXPECT_EQ(splitResults[0].splitNode, &verifyNodes[1]);
  EXPECT_EQ(splitResults[0].splitType, ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE);
  EXPECT_EQ(splitResults[0].splitReason, ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED);
  EXPECT_EQ(splitResults[0].extendType, 0);
  EXPECT_EQ(splitResults[0].extendInfo, nullptr);
  EXPECT_NE(
      ut_log::LogBuffer::Instance().GetContent().find(
          "aclskScopeVerify split result[0]: taskId=2, scopeId=1, splitType=ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE, "
          "splitReason=ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED"),
      std::string::npos);
}

TEST_F(ScopeVerifyTest, ScheModeSplitBeforeDeadlockOnRefinedScope) {
  ScopedLockDetectorCoreNums coreNums(25, 32);
  std::vector<aclskScopeVerifyNodeInfo> verifyNodes = {
      MakeScopeVerifyComputeNode(1, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 12, 1, 2, 1),
      MakeScopeVerifyEventNode(2, 0, 100, 1, ACLSK_SCOPE_VERIFY_NODE_WAIT),
      MakeScopeVerifyComputeNode(3, 0, 1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 15, 1, 0, 1),
      MakeScopeVerifyComputeNode(4, 1, 2, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 12, 1, 0, 0),
      MakeScopeVerifyEventNode(5, 1, 100, 2, ACLSK_SCOPE_VERIFY_NODE_NOTIFY),
  };
  aclskScopeVerifyGraphInfo verifyGraph{verifyNodes.data(), verifyNodes.size(), 0, nullptr};
  aclskScopeVerifySplitResult splitResults[2]{};
  size_t realSplitResultCount = 0;

  ASSERT_EQ(aclskScopeVerify(&verifyGraph, 2, splitResults, &realSplitResultCount), ACL_SUCCESS);

  ASSERT_EQ(realSplitResultCount, 1);
  EXPECT_EQ(splitResults[0].splitNode, &verifyNodes[2]);
  EXPECT_EQ(splitResults[0].splitType, ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE);
  EXPECT_EQ(splitResults[0].splitReason, ACLSK_SCOPE_VERIFY_SYNCALL_OP_DROP);
  EXPECT_EQ(splitResults[0].extendType, 0);
  EXPECT_EQ(splitResults[0].extendInfo, nullptr);
}
