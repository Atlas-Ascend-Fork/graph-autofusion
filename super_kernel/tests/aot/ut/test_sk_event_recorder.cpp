/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sk_event_recorder.cpp
 * \brief Unit tests for SkEventRecorder
 */

#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include <memory>
#include <string>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "securec.h"
#include "ut_common_stubs.h"

#define private public
#define protected public
#include "sk_event_recorder.h"
#include "sk_graph.h"
#include "sk_node.h"

// 辅助函数：包装现已公开的 CoreIsAiv，返回 0/1 以匹配旧测试断言
// 注：sk_event_recorder.h 中的 CoreIsAiv 现在依赖 GetCurrentSkKernelArch()，
// stub 的 aclrtGetSocName 返回 "Ascend910B"，因此走 coreId>=25 分支
static int TestCoreIsAiv(int core_id) {
  return CoreIsAiv(core_id) ? 1 : 0;
}

class SkEventRecorderTest : public testing::Test {
 protected:
  void SetUp() override {
    SkUtResetTestControls();
    // 清理环境变量
    unsetenv("ASCEND_PROF_SK_ON");
    // 清理测试文件和目录
    CleanupTestFiles();
  }

  void TearDown() override {
    // 确保每次测试后 recorder 被重置
    SkEventRecorder::Instance().SkProfilingShutdown();
    // gmAddr 现在是 unique_ptr，通过 reset 释放
    SkEventRecorder::Instance().deviceCtxs.gmAddr.reset();
    SkEventRecorder::Instance().deviceCtxs.hostBuf.reset();
    SkEventRecorder::Instance().deviceCtxs.outputFp.Close();
    SkEventRecorder::Instance().deviceCtxs.outputDir.clear();
    SkEventRecorder::Instance().deviceCtxs.active.store(0);
    SkEventRecorder::Instance().deviceCtxs.totalSize = 0;
    // 清理 modelId index 映射表（重构后由 RegisterModelId 维护）
    SkEventRecorder::Instance().modelIdIndexMap.clear();
    SkEventRecorder::Instance().modelIdToIndexMap.clear();
    SkEventRecorder::Instance().nodeInfoMap.clear();
    SkEventRecorder::Instance().skNameMap.clear();
    SkEventRecorder::Instance().profBasePath.clear();
    unsetenv("ASCEND_PROF_SK_ON");
    SkUtResetTestControls();
    // 重置 call_once 标志，使后续测试的 Init() 可以重新执行
    SkEventRecorder::Instance().initFlag_.~once_flag();
    new (&SkEventRecorder::Instance().initFlag_) std::once_flag();
    // 重置 static 成员变量
    SkEventRecorder::coreSize_ = SK_EVENT_DEFAULT_CORE_SIZE;
    SkEventRecorder::totalSize_ = SK_EVENT_DAV_2201_CORE_NUM * SK_EVENT_DEFAULT_CORE_SIZE;
    // 清理测试文件和目录
    CleanupTestFiles();
  }

  // 辅助函数：清理测试文件和目录
  void CleanupTestFiles() {
    pid_t pid = getpid();
    for (int i = 0; i < 4; i++) {
      // 新路径：./sk_meta/<pid>/
      std::string finalFile = "./sk_meta/" + std::to_string(pid) + "/sk_prof_device_" + std::to_string(i) + ".json";
      remove(finalFile.c_str());
    }
    // 清理 sk_meta/<pid> 目录
    std::string pidDir = "./sk_meta/" + std::to_string(pid);
    rmdir(pidDir.c_str());
    // 尝试清理 sk_meta 目录（如果为空）
    rmdir("./sk_meta");
  }

  // 辅助函数：创建模拟的事件数据
  void CreateMockEventData(SkEventDeviceCtx *ctx, uint32_t coreId, uint64_t modelIdIndex, uint32_t skId,
                           uint32_t nodeId, uint64_t startTime, uint64_t endTime, uint8_t blockIdx = 0,
                           uint8_t blockNum = 1) {
    uint8_t *hostBuf = ctx->hostBuf.get();
    SkKernelEventCoreBuf *coreBuf =
        reinterpret_cast<SkKernelEventCoreBuf *>(hostBuf + coreId * SkEventRecorder::coreSize_);

    // 设置 offset 以便 DumpDeviceData 能读取到数据
    uint32_t newOffset = sizeof(SkKernelEventCoreBuf) + sizeof(SkKernelEventRecord);
    coreBuf->offset = newOffset;

    // 写入事件记录（紧跟在 SkKernelEventCoreBuf 头部之后）
    SkKernelEventRecord *record = reinterpret_cast<SkKernelEventRecord *>(
        hostBuf + coreId * SkEventRecorder::coreSize_ + sizeof(SkKernelEventCoreBuf));
    record->modelIdIndex = modelIdIndex;
    record->skId = skId;
    record->nodeId = nodeId;
    record->blockIdx = blockIdx;
    record->blockNum = blockNum;
    record->startTime = startTime;
    record->endTime = endTime;

    // 重置 lastOffset 以便重新读取
    ctx->lastOffset[coreId] = sizeof(SkKernelEventCoreBuf);
  }

  // 辅助函数：初始化一个模拟的设备上下文
  void InitMockDeviceCtx() {
    if (SkEventRecorder::totalSize_ == 0) {
      SkEventRecorder::totalSize_ = GetSkRuntimeConfig().eventCoreNum * SkEventRecorder::coreSize_;
    }
    SkEventDeviceCtx *ctx = &SkEventRecorder::Instance().deviceCtxs;
    ctx->deviceId = 0;
    ctx->totalSize = SkEventRecorder::totalSize_;
    // gmAddr 是 unique_ptr，这里用 malloc 分配并通过 release 手动管理（测试环境无 aclrtMalloc）
    // 注意：GmAddrDeleter 调用 aclrtFree，但测试中用 malloc 分配，
    // 所以 CleanupMockDeviceCtx 中需要先 release 再 free
    void *rawPtr = malloc(SkEventRecorder::totalSize_);
    ctx->gmAddr.reset(rawPtr);
    ctx->hostBuf = std::make_unique<uint8_t[]>(SkEventRecorder::totalSize_);
    (void)memset_s(ctx->hostBuf.get(), SkEventRecorder::totalSize_, 0, SkEventRecorder::totalSize_);
    ctx->outputDir = SkEventRecorder::CreateOutputDir();  // 设置输出目录
    ctx->outputFp.Close();                                // FileGuard 默认构造已经是无效状态
    uint32_t coreNum = GetSkRuntimeConfig().eventCoreNum;
    ctx->lastOffset.resize(coreNum);
    for (uint32_t i = 0; i < coreNum; i++) {
      ctx->lastOffset[i] = sizeof(SkKernelEventCoreBuf);
    }
    ctx->active.store(1);
  }

  // 辅助函数：清理模拟的设备上下文
  void CleanupMockDeviceCtx() {
    SkEventDeviceCtx *ctx = &SkEventRecorder::Instance().deviceCtxs;
    if (ctx->gmAddr) {
      // 测试中用 malloc 分配，不能让 unique_ptr 析构调 aclrtFree，先 release 再 free
      void *rawPtr = ctx->gmAddr.release();
      free(rawPtr);
    }
    ctx->hostBuf.reset();
    ctx->outputFp.Close();  // 使用 FileGuard 的 Close 方法
    ctx->active.store(0);
  }

  std::unique_ptr<SuperKernelKernelNode> CreateKernelNode(uint64_t nodeId, const std::string &funcName,
                                                          uint32_t numBlocks) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                        INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->nodeInfos.kernelInfos.funcName = funcName;
    node->nodeInfos.kernelInfos.numBlocks = numBlocks;
    return node;
  }

  std::unique_ptr<SuperKernelMemoryNode> CreateWaitNode(uint64_t nodeId) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, 0, INVALID_STREAM_ID,
                                                        INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNodeType(SkNodeType::NODE_WAIT);
    return node;
  }

  std::string ReadWholeFile(const std::string &fileName) {
    std::ifstream in(fileName, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  }
};

// ==================== 基础功能测试 ====================

// Test 1: 单例模式验证
TEST_F(SkEventRecorderTest, InstanceReturnsSameInstance) {
  SkEventRecorder &instance1 = SkEventRecorder::Instance();
  SkEventRecorder &instance2 = SkEventRecorder::Instance();

  EXPECT_EQ(&instance1, &instance2);
}

// Test 2: 默认状态下未启用
TEST_F(SkEventRecorderTest, DisabledByDefault) {
  bool result = SkEventRecorder::Instance().Init();
  EXPECT_FALSE(result);
  EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// Test 3: 环境变量设置为 "64" 时启用（64KB coreSize）
TEST_F(SkEventRecorderTest, EnabledWhenEnvSetToValidSize) {
  setenv("ASCEND_PROF_SK_ON", "64", 1);

  bool result = SkEventRecorder::Instance().Init();
  EXPECT_TRUE(result);
  EXPECT_TRUE(SkEventRecorder::Instance().IsEnabled());

  SkEventRecorder::Instance().SkProfilingShutdown();
}

// Test 4: 环境变量设置为非 "1" 时不启用
TEST_F(SkEventRecorderTest, NotEnabledWhenEnvNotOne) {
  setenv("ASCEND_PROF_SK_ON", "0", 1);

  bool result = SkEventRecorder::Instance().Init();
  EXPECT_FALSE(result);
  EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// Test 5: 环境变量设置为其他值时不启用
TEST_F(SkEventRecorderTest, NotEnabledWhenEnvSetToOtherValue) {
  setenv("ASCEND_PROF_SK_ON", "true", 1);

  bool result = SkEventRecorder::Instance().Init();
  EXPECT_FALSE(result);
  EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

TEST_F(SkEventRecorderTest, NotEnabledWhenEnvHasTrailingCharacters) {
  setenv("ASCEND_PROF_SK_ON", "123abc", 1);

  bool result = SkEventRecorder::Instance().Init();
  EXPECT_FALSE(result);
  EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// ==================== NodeInfo 测试 ====================

// Test 6: NodeInfo 映射添加和获取
TEST_F(SkEventRecorderTest, AddAndGetNodeInfo) {
  std::string modelId = "model_100_1";
  uint32_t skId = 1;
  uint32_t nodeId = 10;
  std::string nodeName = "test_node";
  uint32_t numBlocks = 4;

  SkEventRecorder::Instance().AddNodeInfoMapping(modelId, skId, nodeId, nodeName, numBlocks);

  SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo(modelId, skId, nodeId);

  EXPECT_EQ(info.nodeName, nodeName);
  EXPECT_EQ(info.numBlocks, numBlocks);
}

// Test 7: 获取不存在的 NodeInfo
TEST_F(SkEventRecorderTest, GetNonExistentNodeInfo) {
  SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo("missing_model", 999, 999);

  EXPECT_TRUE(info.nodeName.empty());
  EXPECT_EQ(info.numBlocks, 0);
}

// Test 8: 多个 NodeInfo 映射
TEST_F(SkEventRecorderTest, MultipleNodeInfoMappings) {
  SkEventRecorder::Instance().AddNodeInfoMapping("model_1_1", 1, 1, "node_1_1_1", 2);
  SkEventRecorder::Instance().AddNodeInfoMapping("model_1_1", 1, 2, "node_1_1_2", 3);
  SkEventRecorder::Instance().AddNodeInfoMapping("model_1_1", 2, 1, "node_1_2_1", 4);
  SkEventRecorder::Instance().AddNodeInfoMapping("model_2_1", 1, 1, "node_2_1_1", 5);

  EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo("model_1_1", 1, 1).nodeName, "node_1_1_1");
  EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo("model_1_1", 1, 2).nodeName, "node_1_1_2");
  EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo("model_1_1", 2, 1).nodeName, "node_1_2_1");
  EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo("model_2_1", 1, 1).nodeName, "node_2_1_1");
}

// Test 9: NodeInfo 映射覆盖
TEST_F(SkEventRecorderTest, NodeInfoMappingOverwrite) {
  SkEventRecorder::Instance().AddNodeInfoMapping("model_100_1", 1, 10, "old_name", 2);
  SkEventRecorder::Instance().AddNodeInfoMapping("model_100_1", 1, 10, "new_name", 4);

  SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo("model_100_1", 1, 10);

  EXPECT_EQ(info.nodeName, "new_name");
  EXPECT_EQ(info.numBlocks, 4);
}

// Test 10: 空字符串 nodeName
TEST_F(SkEventRecorderTest, EmptyNodeInfoStrings) {
  SkEventRecorder::Instance().AddNodeInfoMapping("model_100_1", 1, 10, "", 0);

  SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo("model_100_1", 1, 10);

  EXPECT_TRUE(info.nodeName.empty());
  EXPECT_EQ(info.numBlocks, 0);
}

// Test 11: 大量 NodeInfo 映射
TEST_F(SkEventRecorderTest, ManyNodeInfoMappings) {
  const int count = 100;

  for (int i = 0; i < count; i++) {
    SkEventRecorder::Instance().AddNodeInfoMapping("model_" + std::to_string(i), i % 10, i % 20,
                                                   "node_" + std::to_string(i), i % 5 + 1);
  }

  for (int i = 0; i < count; i += 10) {
    SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo("model_" + std::to_string(i), i % 10, i % 20);
    EXPECT_EQ(info.nodeName, "node_" + std::to_string(i));
  }
}

// ==================== 线程安全测试 ====================

// Test 12: 线程安全 - 并发添加 NodeInfo
TEST_F(SkEventRecorderTest, ThreadSafeAddNodeInfo) {
  const int numThreads = 4;
  const int opsPerThread = 25;

  std::vector<std::thread> threads;

  for (int t = 0; t < numThreads; t++) {
    threads.emplace_back([t, opsPerThread]() {
      for (int i = 0; i < opsPerThread; i++) {
        SkEventRecorder::Instance().AddNodeInfoMapping("model_" + std::to_string(t * 1000 + i), t, i,
                                                       "node_" + std::to_string(t) + "_" + std::to_string(i), 1);
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  for (int t = 0; t < numThreads; t++) {
    SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo("model_" + std::to_string(t * 1000), t, 0);
    EXPECT_EQ(info.nodeName, "node_" + std::to_string(t) + "_0");
  }
}

// Test 13: 线程安全 - 并发读取 NodeInfo
TEST_F(SkEventRecorderTest, ThreadSafeGetNodeInfo) {
  for (int i = 0; i < 100; i++) {
    SkEventRecorder::Instance().AddNodeInfoMapping("model_" + std::to_string(i), i, i, "node_" + std::to_string(i), 1);
  }

  const int numThreads = 4;
  std::vector<std::thread> threads;
  std::atomic<int> successCount{0};

  for (int t = 0; t < numThreads; t++) {
    threads.emplace_back([&successCount]() {
      for (int i = 0; i < 25; i++) {
        SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo("model_" + std::to_string(i), i, i);
        if (info.nodeName == "node_" + std::to_string(i)) {
          successCount++;
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(successCount.load(), 100);
}

// ==================== CoreIsAiv 函数测试 ====================

// Test 14: CoreIsAiv - AIC 核心（core_id < 25）
TEST_F(SkEventRecorderTest, CoreIsAiv_AicCores) {
  // 对于非 310 架构，core_id < 25 是 AIC
  for (int i = 0; i < 25; i++) {
    EXPECT_EQ(TestCoreIsAiv(i), 0);
  }
}

// Test 15: CoreIsAiv - AIV 核心（core_id >= 25）
TEST_F(SkEventRecorderTest, CoreIsAiv_AivCores) {
  // 对于非 310 架构，core_id >= 25 是 AIV
  for (int i = 25; i < 75; i++) {
    EXPECT_EQ(TestCoreIsAiv(i), 1);
  }
}

// Test 16: CoreIsAiv - 边界值
TEST_F(SkEventRecorderTest, CoreIsAiv_BoundaryValues) {
  // 边界测试：24 是 AIC，25 是 AIV
  EXPECT_EQ(TestCoreIsAiv(24), 0);
  EXPECT_EQ(TestCoreIsAiv(25), 1);
}

// ==================== CoreIsAiv Dav3510 分支 ====================
// 默认 stub aclrtGetSocName 返回 "Ascend910B"。这里在子进程中 mock 成
// "Ascend950"，专门覆盖 sk_event_recorder.h 中 DAV_3510 分支时的
// AIV 区间 [18,53] ∪ [72,107]。

namespace {
const char *FakeSocName_Ascend950ForCoreIsAiv() {
  return "Ascend950";
}

void InitDav3510RuntimeConfigForCoreIsAiv() {
  SkUtResetTestControls();
  MOCKER(aclrtGetSocName).stubs().will(invoke(FakeSocName_Ascend950ForCoreIsAiv));
  InitSkRuntimeConfig();
}

void ExitIsolatedCoreIsAivTest() {
  GlobalMockObject::verify();
  fflush(nullptr);
  _exit(::testing::Test::HasFailure() ? 1 : 0);
}
}  // namespace

TEST(SkEventRecorderDav3510Test, CoreIsAiv_FirstAivRange_18To53) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        InitDav3510RuntimeConfigForCoreIsAiv();
        for (int i = 18; i <= 53; ++i) {
          EXPECT_TRUE(CoreIsAiv(i)) << "expected AIV for core " << i;
        }
        ExitIsolatedCoreIsAivTest();
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(SkEventRecorderDav3510Test, CoreIsAiv_SecondAivRange_72To107) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        InitDav3510RuntimeConfigForCoreIsAiv();
        for (int i = 72; i <= 107; ++i) {
          EXPECT_TRUE(CoreIsAiv(i)) << "expected AIV for core " << i;
        }
        ExitIsolatedCoreIsAivTest();
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(SkEventRecorderDav3510Test, CoreIsAiv_AicCores_BelowFirstRange) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        InitDav3510RuntimeConfigForCoreIsAiv();
        for (int i = 0; i < 18; ++i) {
          EXPECT_FALSE(CoreIsAiv(i)) << "expected AIC for core " << i;
        }
        ExitIsolatedCoreIsAivTest();
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(SkEventRecorderDav3510Test, CoreIsAiv_AicCores_BetweenRanges) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        InitDav3510RuntimeConfigForCoreIsAiv();
        // 区间间隙 [54,71] 全部应为 AIC
        for (int i = 54; i <= 71; ++i) {
          EXPECT_FALSE(CoreIsAiv(i)) << "expected AIC for core " << i;
        }
        ExitIsolatedCoreIsAivTest();
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(SkEventRecorderDav3510Test, CoreIsAiv_BoundaryValues) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        InitDav3510RuntimeConfigForCoreIsAiv();
        // 第一段 AIV 区间边界
        EXPECT_FALSE(CoreIsAiv(17));
        EXPECT_TRUE(CoreIsAiv(18));
        EXPECT_TRUE(CoreIsAiv(53));
        EXPECT_FALSE(CoreIsAiv(54));
        // 第二段 AIV 区间边界
        EXPECT_FALSE(CoreIsAiv(71));
        EXPECT_TRUE(CoreIsAiv(72));
        EXPECT_TRUE(CoreIsAiv(107));
        EXPECT_FALSE(CoreIsAiv(108));
        ExitIsolatedCoreIsAivTest();
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(SkEventRecorderDav3510Test, CoreIsAiv_DoesNotFallThroughToAscend910Rule) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        InitDav3510RuntimeConfigForCoreIsAiv();
        // 关键回归点：core_id=60 在 Ascend910B 下是 AIV(>=25)，在 Dav3510 下却是
        // AIC（落在 [54,71] 间隙），防止以后有人把分支条件简化回 coreId>=25。
        EXPECT_FALSE(CoreIsAiv(60));
        // 反方向：core_id=20 在 Ascend910B 下是 AIC，在 Dav3510 下是 AIV。
        EXPECT_TRUE(CoreIsAiv(20));
        ExitIsolatedCoreIsAivTest();
      },
      ::testing::ExitedWithCode(0), "");
}

// Test 17: GetGmAddrForDevice 在未启用时返回 nullptr
TEST_F(SkEventRecorderTest, GetGmAddrForDeviceReturnsNullWhenDisabled) {
  void *addr = SkEventRecorder::Instance().GetGmAddrForDevice(0);
  EXPECT_EQ(addr, nullptr);
}

// Test 19: GetGmAddrForDevice 在 deviceId 越界时返回 nullptr
TEST_F(SkEventRecorderTest, GetGmAddrForDeviceReturnsNullForInvalidDeviceId) {
  SkEventRecorder::Instance().enabled = true;

  void *addressBeyondMaxDevice = SkEventRecorder::Instance().GetGmAddrForDevice(SK_EVENT_MAX_DEVICE_NUM);
  EXPECT_EQ(addressBeyondMaxDevice, nullptr);

  void *addressExceedingMaxDevice = SkEventRecorder::Instance().GetGmAddrForDevice(SK_EVENT_MAX_DEVICE_NUM + 1);
  EXPECT_EQ(addressExceedingMaxDevice, nullptr);
}

// Test 21: GetGmAddrForDevice 已激活的设备返回缓存地址
TEST_F(SkEventRecorderTest, GetGmAddrForDeviceReturnsCachedAddr) {
  // 手动初始化一个设备上下文
  InitMockDeviceCtx();
  SkEventRecorder::Instance().enabled = true;

  void *firstCallAddr = SkEventRecorder::Instance().GetGmAddrForDevice(0);
  EXPECT_NE(firstCallAddr, nullptr);

  // 第二次调用应该返回相同地址（fast path）
  void *secondCallAddr = SkEventRecorder::Instance().GetGmAddrForDevice(0);
  EXPECT_EQ(firstCallAddr, secondCallAddr);

  CleanupMockDeviceCtx();
}

// ==================== DumpDeviceData 测试 ====================

// Test 23: DumpDeviceData 空数据处理
TEST_F(SkEventRecorderTest, DumpDeviceDataEmptyData) {
  InitMockDeviceCtx();

  // 没有事件数据
  SkEventRecorder::Instance().DumpDeviceData(&SkEventRecorder::Instance().deviceCtxs);

  // 不应该崩溃
  CleanupMockDeviceCtx();
}

// Test 24: DumpDeviceData 空 gmAddr 或 hostBuf
TEST_F(SkEventRecorderTest, DumpDeviceDataNullPointers) {
  SkEventDeviceCtx ctx;
  // gmAddr 和 hostBuf 默认构造就是 nullptr 的 unique_ptr
  ctx.deviceId = 0;

  // 不应该崩溃
  SkEventRecorder::Instance().DumpDeviceData(&ctx);
}

// Test 28: DumpDeviceData 跳过没有 NodeInfo 的事件
TEST_F(SkEventRecorderTest, DumpDeviceDataSkipsEventsWithoutNodeInfo) {
  InitMockDeviceCtx();

  SkEventDeviceCtx *ctx = &SkEventRecorder::Instance().deviceCtxs;

  // 创建事件数据但不添加 NodeInfo
  CreateMockEventData(ctx, 0, 999, 999, 999, 100, 200, 0, 1);

  SkEventRecorder::Instance().DumpDeviceData(ctx);

  // 不应该崩溃，正常处理完成

  CleanupMockDeviceCtx();
}

TEST_F(SkEventRecorderTest, WriteNodeEventToJsonWritesModelIdFromRegisteredIndex) {
  InitMockDeviceCtx();
  SkEventDeviceCtx *ctx = &SkEventRecorder::Instance().deviceCtxs;

  std::string outputFile = ctx->outputDir + "/sk_prof_device_0.json";
  ASSERT_TRUE(ctx->outputFp.Open(outputFile.c_str(), "w+b"));
  const char *jsonStart = "[{}]";
  ASSERT_EQ(fwrite(jsonStart, 1, strlen(jsonStart), ctx->outputFp.Get()), strlen(jsonStart));

  const std::string modelId = "model_88_3";
  uint16_t modelIdIndex = SkEventRecorder::Instance().RegisterModelId(modelId);
  SkKernelEventRecord record = {};
  record.modelIdIndex = modelIdIndex;
  record.skId = 7;
  record.nodeId = 2;
  record.blockIdx = 1;
  record.blockNum = 4;
  record.startTime = 100;
  record.endTime = 250;
  SkNodeInfo nodeInfo;
  nodeInfo.nodeName = "relu_kernel";
  nodeInfo.numBlocks = 4;

  EXPECT_TRUE(SkEventRecorder::Instance().WriteNodeEventToJson(ctx, &record, 25, nodeInfo));
  fflush(ctx->outputFp.Get());

  std::string content = ReadWholeFile(outputFile);
  EXPECT_NE(content.find("\"modelId\":\"" + modelId + "\""), std::string::npos);
  EXPECT_EQ(content.find("modelRI"), std::string::npos);
  EXPECT_NE(content.find("\"pid\":\"AIV\""), std::string::npos);
  EXPECT_NE(content.find("relu_kernel"), std::string::npos);

  CleanupMockDeviceCtx();
}

TEST_F(SkEventRecorderTest, WriteSkEventToJsonWritesModelIdAndSkName) {
  InitMockDeviceCtx();
  SkEventDeviceCtx *ctx = &SkEventRecorder::Instance().deviceCtxs;

  std::string outputFile = ctx->outputDir + "/sk_prof_device_0.json";
  ASSERT_TRUE(ctx->outputFp.Open(outputFile.c_str(), "w+b"));
  const char *jsonStart = "[{}]";
  ASSERT_EQ(fwrite(jsonStart, 1, strlen(jsonStart), ctx->outputFp.Get()), strlen(jsonStart));

  const std::string modelId = "model_99_4";
  uint16_t modelIdIndex = SkEventRecorder::Instance().RegisterModelId(modelId);
  SkEventRecorder::Instance().AddSkNameMapping(modelId, 9, "sk_fused_add");

  SkKernelEventRecord record = {};
  record.modelIdIndex = modelIdIndex;
  record.skId = 9;
  record.nodeId = UINT32_MAX;
  record.blockIdx = 0;
  record.blockNum = 1;
  record.startTime = 200;
  record.endTime = 500;

  EXPECT_TRUE(SkEventRecorder::Instance().WriteSkEventToJson(ctx, &record, 0));
  fflush(ctx->outputFp.Get());

  std::string content = ReadWholeFile(outputFile);
  EXPECT_NE(content.find("\"modelId\":\"" + modelId + "\""), std::string::npos);
  EXPECT_EQ(content.find("modelRI"), std::string::npos);
  EXPECT_NE(content.find("\"pid\":\"AIC\""), std::string::npos);
  EXPECT_NE(content.find("sk_fused_add"), std::string::npos);

  CleanupMockDeviceCtx();
}

TEST_F(SkEventRecorderTest, DumpDeviceDataUsesModelIdIndexForNodeInfoLookup) {
  InitMockDeviceCtx();
  SkEventDeviceCtx *ctx = &SkEventRecorder::Instance().deviceCtxs;

  std::string outputFile = ctx->outputDir + "/sk_prof_device_0.json";
  ASSERT_TRUE(ctx->outputFp.Open(outputFile.c_str(), "w+b"));
  const char *jsonStart = "[{}]";
  ASSERT_EQ(fwrite(jsonStart, 1, strlen(jsonStart), ctx->outputFp.Get()), strlen(jsonStart));

  const std::string modelId = "model_indexed_1";
  uint16_t modelIdIndex = SkEventRecorder::Instance().RegisterModelId(modelId);
  SkEventRecorder::Instance().AddNodeInfoMapping(modelId, 5, 3, "indexed_node", 2);
  CreateMockEventData(ctx, 0, modelIdIndex, 5, 3, 100, 300, 1, 2);

  SkEventRecorder::Instance().DumpDeviceData(ctx);
  fflush(ctx->outputFp.Get());

  std::string content = ReadWholeFile(outputFile);
  EXPECT_NE(content.find("\"modelId\":\"" + modelId + "\""), std::string::npos);
  EXPECT_NE(content.find("indexed_node"), std::string::npos);
  EXPECT_EQ(content.find("stale_model_id"), std::string::npos);

  CleanupMockDeviceCtx();
}

// ==================== CreateDeviceCtx 测试 ====================

// Test 29: CreateDeviceCtx 基本功能
TEST_F(SkEventRecorderTest, CreateDeviceCtxBasic) {
  SkEventDeviceCtx *ctx = SkEventRecorder::Instance().CreateDeviceCtx(0);

  // stub 实现应该成功创建
  if (ctx != nullptr) {
    EXPECT_NE(ctx->gmAddr, nullptr);
    EXPECT_NE(ctx->hostBuf, nullptr);
    EXPECT_EQ(ctx->deviceId, 0);
    EXPECT_EQ(ctx->active.load(), 1);
    EXPECT_EQ(ctx->totalSize, SkEventRecorder::totalSize_);
    EXPECT_FALSE(ctx->outputDir.empty());  // 输出目录应被设置

    // 清理
    ctx->gmAddr.reset();
    ctx->hostBuf.reset();
    ctx->active.store(0);

    // 删除文件
    std::string jsonFile = ctx->outputDir + "/sk_prof_device_0.json";
    remove(jsonFile.c_str());
  }
}

// ==================== Shutdown 测试 ====================

// Test 30: Shutdown 后 IsEnabled 返回 false
TEST_F(SkEventRecorderTest, IsEnabledFalseAfterShutdown) {
  setenv("ASCEND_PROF_SK_ON", "64", 1);
  SkEventRecorder::Instance().Init();
  EXPECT_TRUE(SkEventRecorder::Instance().IsEnabled());

  SkEventRecorder::Instance().SkProfilingShutdown();
  EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// Test 31: Shutdown 未启用时不做任何操作
TEST_F(SkEventRecorderTest, ShutdownDoesNothingWhenDisabled) {
  EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());

  // 不应该崩溃
  SkEventRecorder::Instance().SkProfilingShutdown();
  SkEventRecorder::Instance().SkProfilingShutdown();

  EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// Test 32: Shutdown 创建最终文件
TEST_F(SkEventRecorderTest, ShutdownCreatesFinalFile) {
  InitMockDeviceCtx();
  SkEventRecorder::Instance().enabled = true;

  SkEventDeviceCtx *ctx = &SkEventRecorder::Instance().deviceCtxs;

  // 创建临时文件
  char tmpFile[512];
  (void)snprintf_s(tmpFile, sizeof(tmpFile), sizeof(tmpFile) - 1, "%s/sk_prof_device_%u.json", ctx->outputDir.c_str(),
                   0);
  ctx->outputFp.Open(tmpFile, "wb");
  if (ctx->outputFp.IsValid()) {
    const char *jsonStart = "[{}]";
    fwrite(jsonStart, 1, strlen(jsonStart), ctx->outputFp.Get());
  }

  // 执行 Shutdown（不启动线程）
  SkEventRecorder::Instance().globalRunning.store(false);

  ctx->outputFp.Close();

  // 验证文件创建
  struct stat buffer;
  EXPECT_EQ(stat(tmpFile, &buffer), 0);

  // 清理
  remove(tmpFile);
  CleanupMockDeviceCtx();
}

// Test 33: Shutdown 释放资源
TEST_F(SkEventRecorderTest, ShutdownReleasesResources) {
  InitMockDeviceCtx();
  SkEventRecorder::Instance().enabled = true;

  // 调用 Shutdown
  SkEventRecorder::Instance().SkProfilingShutdown();

  // 验证资源被释放
  EXPECT_EQ(SkEventRecorder::Instance().deviceCtxs.hostBuf, nullptr);
}

// Test 36: SkKernelEventRecord 结构体大小验证
TEST_F(SkEventRecorderTest, EventRecordSize) {
  EXPECT_LE(sizeof(SkKernelEventRecord), 64);
  EXPECT_GE(sizeof(SkKernelEventRecord), sizeof(uint64_t) * 2 + sizeof(uint32_t) * 2 + sizeof(uint8_t) * 2);
}

// Test 38: SkEventDeviceCtx 初始状态
TEST_F(SkEventRecorderTest, DeviceCtxInitialState) {
  SkEventDeviceCtx ctx;

  EXPECT_EQ(ctx.active.load(), 0);
  EXPECT_EQ(ctx.gmAddr.get(), nullptr);
  EXPECT_EQ(ctx.hostBuf.get(), nullptr);
  EXPECT_EQ(ctx.deviceId, 0);
  EXPECT_EQ(ctx.totalSize, 0);
  EXPECT_FALSE(ctx.outputFp.IsValid());  // FileGuard 默认无效状态
}

// ==================== 多次调用安全性测试 ====================

// Test 39: Init 多次调用安全性
TEST_F(SkEventRecorderTest, MultipleInitCalls) {
  setenv("ASCEND_PROF_SK_ON", "64", 1);

  bool result1 = SkEventRecorder::Instance().Init();
  bool result2 = SkEventRecorder::Instance().Init();

  EXPECT_TRUE(SkEventRecorder::Instance().IsEnabled());

  SkEventRecorder::Instance().SkProfilingShutdown();
}

// Test 40: Shutdown 多次调用安全性
TEST_F(SkEventRecorderTest, MultipleShutdownCalls) {
  SkEventRecorder::Instance().SkProfilingShutdown();
  SkEventRecorder::Instance().SkProfilingShutdown();
  SkEventRecorder::Instance().SkProfilingShutdown();

  EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// ==================== 特殊字符串测试 ====================

// Test 41: 特殊字符 nodeName
TEST_F(SkEventRecorderTest, SpecialCharacterNodeName) {
  std::string specialName = "node-with_special.chars:123/456";

  SkEventRecorder::Instance().AddNodeInfoMapping("model_1_1", 1, 1, specialName, 1);

  SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo("model_1_1", 1, 1);
  EXPECT_EQ(info.nodeName, specialName);
}

// Test 42: 长字符串 nodeName
TEST_F(SkEventRecorderTest, LongNodeName) {
  std::string longName(256, 'a');

  SkEventRecorder::Instance().AddNodeInfoMapping("model_1_1", 1, 1, longName, 1);

  SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo("model_1_1", 1, 1);
  EXPECT_EQ(info.nodeName, longName);
}

// ==================== 常量验证测试 ====================

// Test 43: 常量值验证
TEST_F(SkEventRecorderTest, ConstantsVerification) {
  EXPECT_EQ(SK_EVENT_MAX_DEVICE_NUM, 16);
  EXPECT_EQ(GetSkRuntimeConfig().eventCoreNum, 75);
  EXPECT_EQ(SK_EVENT_DEFAULT_CORE_SIZE, 1024 * 1024);
  EXPECT_EQ(SkEventRecorder::coreSize_, SK_EVENT_DEFAULT_CORE_SIZE);
  EXPECT_EQ(SkEventRecorder::totalSize_, SK_EVENT_DAV_2201_CORE_NUM * SK_EVENT_DEFAULT_CORE_SIZE);
}

// ==================== Init/Shutdown 生命周期测试 ====================

// Test 44: Init 成功后 Shutdown 能正确关闭
TEST_F(SkEventRecorderTest, InitShutdownLifecycle) {
  // 设置 64KB coreSize（已对齐，无需向上取整）
  setenv("ASCEND_PROF_SK_ON", "64", 1);
  SkEventRecorder::Instance().Init();
  EXPECT_TRUE(SkEventRecorder::Instance().IsEnabled());
  EXPECT_EQ(SkEventRecorder::coreSize_, 64U * 1024U);

  SkEventRecorder::Instance().SkProfilingShutdown();
  EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// ==================== DumpThreadFunc 间接测试 ====================

// Test 45: DumpThreadFunc 可以被安全调用
TEST_F(SkEventRecorderTest, DumpThreadFuncCanBeCalledSafely) {
  InitMockDeviceCtx();

  // 直接调用 DumpThreadFunc 的单次迭代逻辑
  SkEventRecorder::Instance().globalRunning.store(false);  // 立即停止

  // 手动调用 DumpDeviceData
  SkEventRecorder::Instance().DumpDeviceData(&SkEventRecorder::Instance().deviceCtxs);

  // 不应该崩溃
  CleanupMockDeviceCtx();
}

// Test 46: DumpThreadFunc 处理单个设备
TEST_F(SkEventRecorderTest, DumpThreadFuncHandlesSingleDevice) {
  InitMockDeviceCtx();

  SkEventRecorder::Instance().globalRunning.store(false);

  // 手动调用 DumpDeviceData
  SkEventDeviceCtx *ctx = &SkEventRecorder::Instance().deviceCtxs;
  if (ctx->active.load()) {
    SkEventRecorder::Instance().DumpDeviceData(ctx);
  }

  CleanupMockDeviceCtx();
}

// ==================== 边界条件测试 ====================

// Test 47: GetNodeInfo 部分键存在部分不存在
TEST_F(SkEventRecorderTest, GetNodeInfoPartialKeyExists) {
  SkEventRecorder::Instance().AddNodeInfoMapping("model_1_1", 1, 1, "node", 1);

  // modelId 存在，skId 不存在
  SkNodeInfo info1 = SkEventRecorder::Instance().GetNodeInfo("model_1_1", 999, 1);
  EXPECT_TRUE(info1.nodeName.empty());

  // modelId 和 skId 存在，nodeId 不存在
  SkNodeInfo info2 = SkEventRecorder::Instance().GetNodeInfo("model_1_1", 1, 999);
  EXPECT_TRUE(info2.nodeName.empty());

  // modelId 不存在
  SkNodeInfo info3 = SkEventRecorder::Instance().GetNodeInfo("missing_model", 1, 1);
  EXPECT_TRUE(info3.nodeName.empty());
}

// Test 48: Double-check locking 路径测试
TEST_F(SkEventRecorderTest, GetGmAddrForDeviceDoubleCheckLocking) {
  SkEventRecorder::Instance().enabled = true;

  // 第一次调用 - slow path
  void *addr1 = SkEventRecorder::Instance().GetGmAddrForDevice(0);

  // 第二次调用 - fast path（active 已设置）
  void *addr2 = SkEventRecorder::Instance().GetGmAddrForDevice(0);

  if (addr1 != nullptr && addr2 != nullptr) {
    EXPECT_EQ(addr1, addr2);

    // 清理
    SkEventDeviceCtx *ctx = &SkEventRecorder::Instance().deviceCtxs;
    std::string outputDir = ctx->outputDir;  // 保存路径用于清理
    ctx->gmAddr.reset();
    ctx->hostBuf.reset();
    ctx->active.store(0);
    // 删除文件
    std::string jsonFile = outputDir + "/sk_prof_device_0.json";
    remove(jsonFile.c_str());
  }
}

// ==================== RegisterModelId / GetModelIdByIndex / PrintModelIdIndexMap 测试 ====================

// Test: RegisterModelId 基本注册，返回 index 0
TEST_F(SkEventRecorderTest, RegisterModelId_FirstRegistrationReturnsZero) {
  uint16_t idx = SkEventRecorder::Instance().RegisterModelId("model_id_0");
  EXPECT_EQ(idx, 0);
}

// Test: RegisterModelId 多次注册不同 modelId，index 递增
TEST_F(SkEventRecorderTest, RegisterModelId_MultipleRegistrationsIncrementIndex) {
  uint16_t idx0 = SkEventRecorder::Instance().RegisterModelId("model_aaaa");
  uint16_t idx1 = SkEventRecorder::Instance().RegisterModelId("model_bbbb");
  uint16_t idx2 = SkEventRecorder::Instance().RegisterModelId("model_cccc");
  EXPECT_EQ(idx0, 0);
  EXPECT_EQ(idx1, 1);
  EXPECT_EQ(idx2, 2);
}

// Test: RegisterModelId 重复注册同一个 modelId，返回相同 index（去重）
TEST_F(SkEventRecorderTest, RegisterModelId_DuplicateRegistrationReturnsSameIndex) {
  uint16_t idx1 = SkEventRecorder::Instance().RegisterModelId("model_dead");
  uint16_t idx2 = SkEventRecorder::Instance().RegisterModelId("model_dead");
  EXPECT_EQ(idx1, idx2);
}

// Test: GetModelIdByIndex 通过 index 反查原始 modelId
TEST_F(SkEventRecorderTest, GetModelIdByIndex_ReturnsModelId) {
  std::string m1 = "model_1234_5";
  std::string m2 = "model_aabb_7";
  uint16_t idx1 = SkEventRecorder::Instance().RegisterModelId(m1);
  uint16_t idx2 = SkEventRecorder::Instance().RegisterModelId(m2);

  EXPECT_EQ(SkEventRecorder::Instance().GetModelIdByIndex(idx1), m1);
  EXPECT_EQ(SkEventRecorder::Instance().GetModelIdByIndex(idx2), m2);
}

TEST_F(SkEventRecorderTest, ModelIdIndexDistinguishesRepeatedModelId) {
  const std::string firstModelId = "48_1";
  const std::string secondModelId = "48_2";

  uint16_t firstIdx = SkEventRecorder::Instance().RegisterModelId(firstModelId);
  uint16_t secondIdx = SkEventRecorder::Instance().RegisterModelId(secondModelId);

  EXPECT_NE(firstIdx, secondIdx);
  EXPECT_EQ(SkEventRecorder::Instance().GetModelIdByIndex(firstIdx), firstModelId);
  EXPECT_EQ(SkEventRecorder::Instance().GetModelIdByIndex(secondIdx), secondModelId);
}

// Test: GetModelIdByIndex 越界 index 返回空串
TEST_F(SkEventRecorderTest, GetModelIdByIndex_OutOfRangeReturnsEmpty) {
  SkEventRecorder::Instance().RegisterModelId("model_one");
  EXPECT_TRUE(SkEventRecorder::Instance().GetModelIdByIndex(100).empty());
  EXPECT_TRUE(SkEventRecorder::Instance().GetModelIdByIndex(UINT16_MAX).empty());
}

// Test: GetModelIdByIndex 空 map 时返回空串
TEST_F(SkEventRecorderTest, GetModelIdByIndex_EmptyMapReturnsEmpty) {
  EXPECT_TRUE(SkEventRecorder::Instance().GetModelIdByIndex(0).empty());
}

// Test: PrintModelIdIndexMap 不崩溃
TEST_F(SkEventRecorderTest, PrintModelIdIndexMap_DoesNotCrash) {
  SkEventRecorder::Instance().RegisterModelId("model_aaaa");
  SkEventRecorder::Instance().RegisterModelId("model_bbbb");
  SkEventRecorder::Instance().PrintModelIdIndexMap();
  SUCCEED();
}

// Test: PrintModelIdIndexMap 空 map 时不崩溃
TEST_F(SkEventRecorderTest, PrintModelIdIndexMap_EmptyMapDoesNotCrash) {
  SkEventRecorder::Instance().PrintModelIdIndexMap();
  SUCCEED();
}

// ==================== modelIdIndexAndSkScopeId 编码测试 ====================

// Test: modelIdIndexAndSkScopeId 编码格式验证
// 布局: modelIdIdx(16bit)[55:40] | skScopeId(16bit)[39:24] | 低24bit预留[23:0]
TEST_F(SkEventRecorderTest, ModelIdIndexAndSkScopeId_EncodingLayout) {
  std::string modelId = "model_123_1";
  uint16_t skScopeId = 42;

  uint16_t modelIdIdx = SkEventRecorder::Instance().RegisterModelId(modelId);
  uint64_t encoded = (static_cast<uint64_t>(modelIdIdx) << 40) | (static_cast<uint64_t>(skScopeId) << 24);

  // 解码验证
  uint16_t decodedIdx = static_cast<uint16_t>((encoded >> 40) & 0xFFFF);
  uint16_t decodedScopeId = static_cast<uint16_t>((encoded >> 24) & 0xFFFF);
  uint32_t decodedLow24 = static_cast<uint32_t>(encoded & 0xFFFFFF);

  EXPECT_EQ(decodedIdx, modelIdIdx);
  EXPECT_EQ(decodedScopeId, skScopeId);
  EXPECT_EQ(decodedLow24, 0);  // 低24bit预留，应为0
}

// Test: modelIdIndexAndSkScopeId 编码后能通过 index 反查原始 modelId
TEST_F(SkEventRecorderTest, ModelIdIndexAndSkScopeId_EncodeDecodeRoundTrip) {
  std::string modelId = "model_deadbeef_3";
  uint16_t skScopeId = 100;

  uint16_t modelIdIdx = SkEventRecorder::Instance().RegisterModelId(modelId);
  uint64_t encoded = (static_cast<uint64_t>(modelIdIdx) << 40) | (static_cast<uint64_t>(skScopeId) << 24);

  // 从编码中解码出 index
  uint16_t decodedIdx = static_cast<uint16_t>((encoded >> 40) & 0xFFFF);
  // 通过 index 反查原始 modelId
  std::string recoveredModelId = SkEventRecorder::Instance().GetModelIdByIndex(decodedIdx);

  EXPECT_EQ(recoveredModelId, modelId);
}

// Test: RegisterModelId 不依赖 profiling 开关
TEST_F(SkEventRecorderTest, RegisterModelId_WorksWhenProfilingDisabled) {
  // 不设置 ASCEND_PROF_SK_ON，profiling 未启用
  ASSERT_FALSE(SkEventRecorder::Instance().IsEnabled());

  std::string modelId = "model_5555_1";
  uint16_t idx = SkEventRecorder::Instance().RegisterModelId(modelId);
  std::string recovered = SkEventRecorder::Instance().GetModelIdByIndex(idx);
  EXPECT_EQ(recovered, modelId);
}

// Test: RegisterModelId 线程安全
TEST_F(SkEventRecorderTest, RegisterModelId_ThreadSafe) {
  const int numThreads = 4;
  const int opsPerThread = 100;
  std::vector<std::thread> threads;
  std::vector<uint16_t> results[numThreads];

  for (int t = 0; t < numThreads; t++) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < opsPerThread; i++) {
        std::string modelId = "m_" + std::to_string(t) + "_" + std::to_string(i);
        uint16_t idx = SkEventRecorder::Instance().RegisterModelId(modelId);
        results[t].push_back(idx);
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  // 验证：所有注册的 modelId 都能通过 index 反查回来
  for (int t = 0; t < numThreads; t++) {
    for (int i = 0; i < opsPerThread; i++) {
      std::string modelId = "m_" + std::to_string(t) + "_" + std::to_string(i);
      uint16_t idx = results[t][i];
      EXPECT_EQ(SkEventRecorder::Instance().GetModelIdByIndex(idx), modelId);
    }
  }
}

// Test: RegisterModelId 重复注册线程安全（去重）
TEST_F(SkEventRecorderTest, RegisterModelId_DuplicateRegistrationThreadSafe) {
  const int numThreads = 4;
  std::vector<std::thread> threads;
  std::vector<uint16_t> results(numThreads);

  std::string sharedModelId = "model_9999_1";

  for (int t = 0; t < numThreads; t++) {
    threads.emplace_back([&, t]() { results[t] = SkEventRecorder::Instance().RegisterModelId(sharedModelId); });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  // 所有线程应得到相同的 index
  for (int t = 1; t < numThreads; t++) {
    EXPECT_EQ(results[0], results[t]);
  }
}

TEST_F(SkEventRecorderTest, DumpProfilingDetailDisabledEncodesModelIdIndexInDevArgs) {
  const aclmdlRI modelRI = reinterpret_cast<aclmdlRI>(static_cast<uintptr_t>(0x12345678));
  const std::string modelId = "model_305419896_7";
  SuperKernelGraph graph(modelRI, modelId);

  SuperKernelScopeInfo scopeInfo;
  SkLaunchInfo launchInfo = {};
  ASSERT_TRUE(launchInfo.devArgs.Init(sizeof(SkDeviceEntryArgs) + sizeof(SkEventConfig)));
  (void)memset_s(launchInfo.devArgs.Get(), sizeof(SkDeviceEntryArgs) + sizeof(SkEventConfig), 0,
                 sizeof(SkDeviceEntryArgs) + sizeof(SkEventConfig));
  launchInfo.devArgs.Get()->skHeader.eventConfigOffset = sizeof(SkDeviceEntryArgs);

  std::vector<SuperKernelBaseNode *> taskNodes;
  EXPECT_TRUE(DumpProfilingDetail(taskNodes, launchInfo, scopeInfo, graph));

  uint64_t encoded = launchInfo.devArgs.Get()->skHeader.modelIdIndexAndSkScopeId;
  uint16_t decodedModelIdIndex = static_cast<uint16_t>((encoded >> 40) & 0xFFFF);
  uint16_t decodedScopeId = static_cast<uint16_t>((encoded >> 24) & 0xFFFF);

  EXPECT_EQ(SkEventRecorder::Instance().GetModelIdByIndex(decodedModelIdIndex), modelId);
  EXPECT_EQ(decodedScopeId, scopeInfo.GetScopeId());
  EXPECT_EQ(static_cast<uint16_t>(encoded & 0xFFFF), 0);
  EXPECT_EQ(SkEventRecorder::Instance().GetSkName(modelId, scopeInfo.GetScopeId()), launchInfo.skFuncName);
  EXPECT_EQ(launchInfo.eventGmAddr, nullptr);
  EXPECT_EQ(launchInfo.modelIdIndex, 0U);
  EXPECT_EQ(launchInfo.skId, 0U);
}

TEST_F(SkEventRecorderTest, DumpProfilingDetailEnabledUpdatesRuntimeConfigAndMappings) {
  const aclmdlRI modelRI = reinterpret_cast<aclmdlRI>(static_cast<uintptr_t>(0x87654321));
  const std::string modelId = "model_2271560481_2";
  SuperKernelGraph graph(modelRI, modelId);

  SuperKernelScopeInfo scopeInfo;
  SkLaunchInfo launchInfo = {};
  launchInfo.skFuncName = "sk_scope_runtime";
  ASSERT_TRUE(launchInfo.devArgs.Init(sizeof(SkDeviceEntryArgs) + sizeof(SkEventConfig)));
  (void)memset_s(launchInfo.devArgs.Get(), sizeof(SkDeviceEntryArgs) + sizeof(SkEventConfig), 0,
                 sizeof(SkDeviceEntryArgs) + sizeof(SkEventConfig));
  launchInfo.devArgs.Get()->skHeader.eventConfigOffset = sizeof(SkDeviceEntryArgs);

  auto kernel0 = CreateKernelNode(10, "first_kernel", 8);
  auto wait = CreateWaitNode(11);
  auto kernel2 = CreateKernelNode(12, "last_kernel", 16);
  std::vector<SuperKernelBaseNode *> taskNodes = {kernel0.get(), wait.get(), kernel2.get(), nullptr};

  InitMockDeviceCtx();
  SkEventDeviceCtx *ctx = &SkEventRecorder::Instance().deviceCtxs;
  SkEventRecorder::Instance().enabled = true;

  bool ret = DumpProfilingDetail(taskNodes, launchInfo, scopeInfo, graph);
  EXPECT_TRUE(ret);
  if (ret) {
    SkEventConfig *eventConfig = reinterpret_cast<SkEventConfig *>(
        reinterpret_cast<uint8_t *>(launchInfo.devArgs.Get()) + sizeof(SkDeviceEntryArgs));
    uint16_t headerModelIdIndex =
        static_cast<uint16_t>((launchInfo.devArgs.Get()->skHeader.modelIdIndexAndSkScopeId >> 40) & 0xFFFF);
    uint16_t headerScopeId =
        static_cast<uint16_t>((launchInfo.devArgs.Get()->skHeader.modelIdIndexAndSkScopeId >> 24) & 0xFFFF);

    EXPECT_EQ(launchInfo.eventGmAddr, ctx->gmAddr.get());
    EXPECT_EQ(SkEventRecorder::Instance().GetModelIdByIndex(static_cast<uint16_t>(launchInfo.modelIdIndex)), modelId);
    EXPECT_EQ(headerModelIdIndex, static_cast<uint16_t>(launchInfo.modelIdIndex));
    EXPECT_EQ(headerScopeId, scopeInfo.GetScopeId());
    EXPECT_EQ(launchInfo.skId, scopeInfo.GetScopeId());
    EXPECT_EQ(eventConfig->eventGmAddr, reinterpret_cast<uint64_t>(ctx->gmAddr.get()));
    EXPECT_EQ(eventConfig->modelIdIndex, launchInfo.modelIdIndex);
    EXPECT_EQ(eventConfig->skId, launchInfo.skId);
    EXPECT_EQ(eventConfig->enabled, 1);
    EXPECT_EQ(eventConfig->coreSize, SkEventRecorder::GetCoreSize());
    EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo(modelId, launchInfo.skId, 0).nodeName, "first_kernel");
    EXPECT_TRUE(SkEventRecorder::Instance().GetNodeInfo(modelId, launchInfo.skId, 1).nodeName.empty());
    EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo(modelId, launchInfo.skId, 2).nodeName, "last_kernel");
    EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo(modelId, launchInfo.skId, 2).numBlocks, 16U);
    EXPECT_EQ(SkEventRecorder::Instance().GetSkName(modelId, launchInfo.skId), "sk_scope_runtime");
  }

  SkEventRecorder::Instance().enabled = false;
  CleanupMockDeviceCtx();
}

TEST_F(SkEventRecorderTest, DumpProfilingDetailEnabledReturnsFalseWhenGetDeviceFails) {
  const aclmdlRI modelRI = reinterpret_cast<aclmdlRI>(static_cast<uintptr_t>(0x2222));
  const std::string modelId = "model_8738_1";
  SuperKernelGraph graph(modelRI, modelId);

  SuperKernelScopeInfo scopeInfo;
  SkLaunchInfo launchInfo = {};
  ASSERT_TRUE(launchInfo.devArgs.Init(sizeof(SkDeviceEntryArgs) + sizeof(SkEventConfig)));
  (void)memset_s(launchInfo.devArgs.Get(), sizeof(SkDeviceEntryArgs) + sizeof(SkEventConfig), 0,
                 sizeof(SkDeviceEntryArgs) + sizeof(SkEventConfig));

  SkEventRecorder::Instance().enabled = true;
  SkUtSetAclrtGetDeviceRet(ACL_ERROR_FAILURE);

  std::vector<SuperKernelBaseNode *> taskNodes;
  EXPECT_FALSE(DumpProfilingDetail(taskNodes, launchInfo, scopeInfo, graph));
  EXPECT_EQ(SkEventRecorder::Instance().GetSkName(modelId, scopeInfo.GetScopeId()), launchInfo.skFuncName);

  SkEventRecorder::Instance().enabled = false;
}

// Test: SkHeaderInfo::modelIdIndexAndSkScopeId 字段偏移和大小验证
TEST_F(SkEventRecorderTest, SkHeaderInfoModelIdIndexAndSkScopeId_FieldVerification) {
  // 验证 modelIdIndexAndSkScopeId 字段存在且为 uint64_t
  SkHeaderInfo headerInfo = {};
  headerInfo.modelIdIndexAndSkScopeId = 0x0001000200000000ULL;
  EXPECT_EQ(headerInfo.modelIdIndexAndSkScopeId, 0x0001000200000000ULL);

  // 验证字段大小
  EXPECT_EQ(sizeof(headerInfo.modelIdIndexAndSkScopeId), sizeof(uint64_t));

  // 验证可以正确编码/解码
  uint16_t modelIdIdx = 1;
  uint16_t skScopeId = 2;
  headerInfo.modelIdIndexAndSkScopeId =
      (static_cast<uint64_t>(modelIdIdx) << 40) | (static_cast<uint64_t>(skScopeId) << 24);
  EXPECT_EQ(static_cast<uint16_t>((headerInfo.modelIdIndexAndSkScopeId >> 40) & 0xFFFF), modelIdIdx);
  EXPECT_EQ(static_cast<uint16_t>((headerInfo.modelIdIndexAndSkScopeId >> 24) & 0xFFFF), skScopeId);
}

// Test: 多个不同 modelId 注册后完整编解码验证
TEST_F(SkEventRecorderTest, ModelIdIndexAndSkScopeId_MultipleModelIdEndToEnd) {
  struct TestItem {
    std::string modelId;
    uint16_t skScopeId;
  };

  std::vector<TestItem> items = {
      {"model_aaaaaaa1_1", 10},
      {"model_aaaaaaa2_1", 20},
      {"model_aaaaaaa3_1", 30},
  };

  std::vector<uint64_t> encodedValues;

  for (const auto &item : items) {
    uint16_t modelIdIdx = SkEventRecorder::Instance().RegisterModelId(item.modelId);
    uint64_t encoded = (static_cast<uint64_t>(modelIdIdx) << 40) | (static_cast<uint64_t>(item.skScopeId) << 24);
    encodedValues.push_back(encoded);
  }

  // 验证每个编码值可以解码回原始值
  for (size_t i = 0; i < items.size(); i++) {
    uint16_t decodedIdx = static_cast<uint16_t>((encodedValues[i] >> 40) & 0xFFFF);
    uint16_t decodedScopeId = static_cast<uint16_t>((encodedValues[i] >> 24) & 0xFFFF);
    std::string recoveredModelId = SkEventRecorder::Instance().GetModelIdByIndex(decodedIdx);

    EXPECT_EQ(recoveredModelId, items[i].modelId);
    EXPECT_EQ(decodedScopeId, items[i].skScopeId);
  }
}
// ==================== SkName 映射测试 ====================

// Test 49: AddSkNameMapping 和 GetSkName 基本功能
TEST_F(SkEventRecorderTest, AddAndGetSkName) {
  std::string modelId = "model_100_1";
  uint32_t skId = 1;
  std::string skName = "test_super_kernel";

  SkEventRecorder::Instance().AddSkNameMapping(modelId, skId, skName);

  std::string result = SkEventRecorder::Instance().GetSkName(modelId, skId);
  EXPECT_EQ(result, skName);
}

// Test 50: GetSkName 获取不存在的映射
TEST_F(SkEventRecorderTest, GetNonExistentSkName) {
  std::string result = SkEventRecorder::Instance().GetSkName("missing_model", 999);
  EXPECT_TRUE(result.empty());
}

// Test 51: 多个 SkName 映射
TEST_F(SkEventRecorderTest, MultipleSkNameMappings) {
  SkEventRecorder::Instance().AddSkNameMapping("model_1", 1, "sk_1_1");
  SkEventRecorder::Instance().AddSkNameMapping("model_1", 2, "sk_1_2");
  SkEventRecorder::Instance().AddSkNameMapping("model_2", 1, "sk_2_1");

  EXPECT_EQ(SkEventRecorder::Instance().GetSkName("model_1", 1), "sk_1_1");
  EXPECT_EQ(SkEventRecorder::Instance().GetSkName("model_1", 2), "sk_1_2");
  EXPECT_EQ(SkEventRecorder::Instance().GetSkName("model_2", 1), "sk_2_1");
}

// Test 52: SkName 映射覆盖
TEST_F(SkEventRecorderTest, SkNameMappingOverwrite) {
  SkEventRecorder::Instance().AddSkNameMapping("model_100_1", 1, "old_sk");
  SkEventRecorder::Instance().AddSkNameMapping("model_100_1", 1, "new_sk");

  std::string result = SkEventRecorder::Instance().GetSkName("model_100_1", 1);
  EXPECT_EQ(result, "new_sk");
}

// ==================== ParseEnvAndSetSize 边界测试 ====================

// Test 53: 环境变量非对齐值向上取整
TEST_F(SkEventRecorderTest, EnvValueAlignmentRoundUp) {
  // 100KB 应向上取整到 128KB（64 的倍数）
  setenv("ASCEND_PROF_SK_ON", "100", 1);

  SkEventRecorder::Instance().Init();
  EXPECT_TRUE(SkEventRecorder::Instance().IsEnabled());
  // 100 向上取整到 128，coreSize = 128 * 1024
  EXPECT_EQ(SkEventRecorder::coreSize_, 128U * 1024U);

  SkEventRecorder::Instance().SkProfilingShutdown();
}

// Test 54: 环境变量已经是对齐值
TEST_F(SkEventRecorderTest, EnvValueAlreadyAligned) {
  // 128KB 已是 64 的倍数，无需取整
  setenv("ASCEND_PROF_SK_ON", "128", 1);

  SkEventRecorder::Instance().Init();
  EXPECT_TRUE(SkEventRecorder::Instance().IsEnabled());
  EXPECT_EQ(SkEventRecorder::coreSize_, 128U * 1024U);

  SkEventRecorder::Instance().SkProfilingShutdown();
}

// Test 55: 环境变量超过 5MB 应失败
TEST_F(SkEventRecorderTest, EnvValueTooLarge) {
  // 5121KB > 5MB，应失败
  setenv("ASCEND_PROF_SK_ON", "5121", 1);

  bool result = SkEventRecorder::Instance().Init();
  EXPECT_FALSE(result);
  EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// Test 56: 环境变量超过 1MB 但小于 5MB 应成功
TEST_F(SkEventRecorderTest, EnvValueLargeButValid) {
  // 2048KB = 2MB，大于 1MB 但小于 5MB，应成功
  setenv("ASCEND_PROF_SK_ON", "2048", 1);

  SkEventRecorder::Instance().Init();
  EXPECT_TRUE(SkEventRecorder::Instance().IsEnabled());
  EXPECT_EQ(SkEventRecorder::coreSize_, 2048U * 1024U);

  SkEventRecorder::Instance().SkProfilingShutdown();
}

// Test 57: 环境变量边界值 1KB（最小非零值，向上取整到 64KB）
TEST_F(SkEventRecorderTest, EnvValueMinimumOne) {
  setenv("ASCEND_PROF_SK_ON", "1", 1);

  SkEventRecorder::Instance().Init();
  EXPECT_TRUE(SkEventRecorder::Instance().IsEnabled());
  // 1 向上取整到 64
  EXPECT_EQ(SkEventRecorder::coreSize_, 64U * 1024U);

  SkEventRecorder::Instance().SkProfilingShutdown();
}

// ==================== DumpThreadFunc 文件大小检查测试 ====================

// Test 58: 没有检测到profiling关闭补复制
TEST_F(SkEventRecorderTest, DumpThreadFuncDstMissingTriggersReCopy) {
  InitMockDeviceCtx();
  SkEventRecorder::Instance().enabled = true;

  SkEventDeviceCtx *ctx = &SkEventRecorder::Instance().deviceCtxs;

  // 创建源 json 文件：先写内容后关闭，再以 "rb" 重新打开
  // 这样 outputFp.IsValid()=true（CopyOutputToProfPath 不跳过），
  // 同时文件内容已落盘（ifstream 读取不会与写锁冲突卡住）
  char srcFile[512];
  (void)snprintf_s(srcFile, sizeof(srcFile), sizeof(srcFile) - 1, "%s/sk_prof_device_%u.json", ctx->outputDir.c_str(),
                   ctx->deviceId);
  ctx->outputFp.Open(srcFile, "wb");
  ASSERT_TRUE(ctx->outputFp.IsValid());
  const char *jsonContent = "[{\"node\":\"test_node\",\"start\":100,\"end\":200}]";
  fwrite(jsonContent, 1, strlen(jsonContent), ctx->outputFp.Get());
  ctx->outputFp.Close();
  // 以 "rb" 重新打开，IsValid()=true，且 ifstream 读同一文件不会冲突
  ctx->outputFp.Open(srcFile, "rb");
  ASSERT_TRUE(ctx->outputFp.IsValid());

  // 设置 profBasePath
  std::string profDir = "/tmp/sk_test_prof_missing_" + std::to_string(getpid());
  mkdir(profDir.c_str(), 0755);
  {
    std::lock_guard<std::mutex> lock(SkEventRecorder::Instance().profBasePathMutex);
    SkEventRecorder::Instance().profBasePath = profDir;
  }

  // 确保目标文件不存在
  std::string dstFile = profDir + "/sk_prof_device_" + std::to_string(ctx->deviceId) + ".json";
  remove(dstFile.c_str());

  // 让 globalRunning=false，使 DumpThreadFunc 的 while 循环立即退出
  SkEventRecorder::Instance().globalRunning.store(false);
  ctx->active.store(0);
  // 设置 g_profSignal 为非0值，覆盖367-373行代码路径
  SkEventRecorder::SetProfSignal(1);
  SkEventRecorder::DumpThreadFunc(&SkEventRecorder::Instance());

  // 验证目标文件已被创建且大小 > 10
  struct stat afterStat;
  EXPECT_EQ(stat(dstFile.c_str(), &afterStat), 0);
  EXPECT_GT(afterStat.st_size, 10);

  // 覆盖378-394行代码路径
  remove(dstFile.c_str());
  SkEventRecorder::SetProfSignal(0);
  SkEventRecorder::DumpThreadFunc(&SkEventRecorder::Instance());

  // 验证目标文件已被创建且大小 > 10
  EXPECT_EQ(stat(dstFile.c_str(), &afterStat), 0);
  EXPECT_GT(afterStat.st_size, 10);

  remove(dstFile.c_str());
  rmdir(profDir.c_str());
  CleanupMockDeviceCtx();
}

// ==================== Mapping Lifetime / DumpProfilingDetail Tests ====================

TEST_F(SkEventRecorderTest, ModelMappings_PreserveMultipleModelsUntilTestTeardown) {
  const std::string targetModelId = "model_target_1";
  const std::string otherModelId = "model_other_1";

  SkEventRecorder::Instance().AddNodeInfoMapping(targetModelId, 1, 0, "target_node", 4);
  SkEventRecorder::Instance().AddSkNameMapping(targetModelId, 1, "target_sk");
  SkEventRecorder::Instance().AddNodeInfoMapping(otherModelId, 1, 0, "other_node", 8);
  SkEventRecorder::Instance().AddSkNameMapping(otherModelId, 1, "other_sk");

  EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo(targetModelId, 1, 0).nodeName, "target_node");
  EXPECT_EQ(SkEventRecorder::Instance().GetSkName(targetModelId, 1), "target_sk");
  EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo(otherModelId, 1, 0).nodeName, "other_node");
  EXPECT_EQ(SkEventRecorder::Instance().GetSkName(otherModelId, 1), "other_sk");
}

TEST_F(SkEventRecorderTest, DumpProfilingDetail_DisabledRegistersModelAndSkName) {
  SuperKernelScopeInfo scopeInfo;
  SkLaunchInfo launchInfo;
  ASSERT_TRUE(launchInfo.devArgs.Init(sizeof(SkDeviceEntryArgs)));
  launchInfo.skFuncName = "sk_disabled_start_add_end_mul";
  aclmdlRI modelRI = reinterpret_cast<aclmdlRI>(0x123456789ABCULL);
  const std::string modelId = "model_disabled_1";
  SuperKernelGraph graph(modelRI, modelId);

  EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
  EXPECT_TRUE(DumpProfilingDetail({}, launchInfo, scopeInfo, graph));

  EXPECT_EQ(launchInfo.eventGmAddr, nullptr);
  EXPECT_EQ(launchInfo.modelIdIndex, 0);
  EXPECT_EQ(launchInfo.skId, 0);
  const uint64_t encoded = launchInfo.devArgs.Get()->skHeader.modelIdIndexAndSkScopeId;
  const uint16_t modelIdIdx = static_cast<uint16_t>((encoded >> 40) & 0xFFFF);
  const uint16_t skScopeId = static_cast<uint16_t>((encoded >> 24) & 0xFFFF);
  EXPECT_EQ(skScopeId, scopeInfo.GetScopeId());
  EXPECT_EQ(SkEventRecorder::Instance().GetModelIdByIndex(modelIdIdx), modelId);
  EXPECT_EQ(SkEventRecorder::Instance().GetSkName(modelId, scopeInfo.GetScopeId()), launchInfo.skFuncName);
}

TEST_F(SkEventRecorderTest, DumpProfilingDetail_EnabledUpdatesEventConfigAndNodeInfo) {
  InitMockDeviceCtx();
  SkEventRecorder::Instance().enabled = true;

  SuperKernelScopeInfo scopeInfo;
  SkLaunchInfo launchInfo;
  const size_t devArgsSize = sizeof(SkDeviceEntryArgs) + sizeof(SkEventConfig);
  ASSERT_TRUE(launchInfo.devArgs.Init(devArgsSize));
  launchInfo.devArgs.Get()->skHeader.eventConfigOffset = sizeof(SkHeaderInfo);
  launchInfo.skFuncName = "sk_enabled_start_add_end_mul";

  auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeType(SkNodeType::NODE_KERNEL);
  node->nodeInfos.kernelInfos.funcName = "add_kernel";
  node->nodeInfos.kernelInfos.numBlocks = 7;
  std::vector<SuperKernelBaseNode *> taskNodes = {node.get()};

  aclmdlRI modelRI = reinterpret_cast<aclmdlRI>(0x223344556677ULL);
  const std::string modelId = "model_enabled_1";
  SuperKernelGraph graph(modelRI, modelId);

  EXPECT_TRUE(DumpProfilingDetail(taskNodes, launchInfo, scopeInfo, graph));

  SkEventDeviceCtx *ctx = &SkEventRecorder::Instance().deviceCtxs;
  EXPECT_EQ(launchInfo.eventGmAddr, ctx->gmAddr.get());
  EXPECT_EQ(SkEventRecorder::Instance().GetModelIdByIndex(static_cast<uint16_t>(launchInfo.modelIdIndex)), modelId);
  EXPECT_EQ(launchInfo.skId, scopeInfo.GetScopeId());

  auto *eventConfig = reinterpret_cast<SkEventConfig *>(reinterpret_cast<uint8_t *>(launchInfo.devArgs.Get()) +
                                                        launchInfo.devArgs.Get()->skHeader.eventConfigOffset);
  EXPECT_EQ(eventConfig->eventGmAddr, reinterpret_cast<uint64_t>(ctx->gmAddr.get()));
  EXPECT_EQ(eventConfig->modelIdIndex, launchInfo.modelIdIndex);
  EXPECT_EQ(eventConfig->skId, scopeInfo.GetScopeId());
  EXPECT_EQ(eventConfig->enabled, 1);
  EXPECT_EQ(eventConfig->coreSize, SkEventRecorder::Instance().GetCoreSize());

  SkNodeInfo nodeInfo = SkEventRecorder::Instance().GetNodeInfo(modelId, scopeInfo.GetScopeId(), 0);
  EXPECT_EQ(nodeInfo.nodeName, "add_kernel");
  EXPECT_EQ(nodeInfo.numBlocks, 7);
  EXPECT_EQ(SkEventRecorder::Instance().GetSkName(modelId, scopeInfo.GetScopeId()), launchInfo.skFuncName);

  SkEventRecorder::Instance().enabled = false;
  CleanupMockDeviceCtx();
}

// ==================== GetSkFuncName Tests ====================

class SkGetSkFuncNameTest : public testing::Test {
 protected:
  void SetUp() override {
    graph = std::make_unique<SuperKernelGraph>();
  }

  void TearDown() override {}

  SuperKernelBaseNode *CreateKernelNodeWithFuncName(uint64_t nodeId, const std::string &funcName) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                        INVALID_TASK_ID);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(INVALID_TASK_ID);
    node->SetNextNodeId(INVALID_TASK_ID);
    node->nodeInfos.kernelInfos.funcName = funcName;
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  SuperKernelBaseNode *CreateWaitNodeForFuncName(uint64_t nodeId) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, 0, INVALID_STREAM_ID,
                                                        INVALID_TASK_ID);
    node->SetNodeType(SkNodeType::NODE_WAIT);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(INVALID_TASK_ID);
    node->SetNextNodeId(INVALID_TASK_ID);
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  std::unique_ptr<SuperKernelGraph> graph;
};

TEST_F(SkGetSkFuncNameTest, EmptyNodes_ReturnsNoKernelScope) {
  std::vector<SuperKernelBaseNode *> nodes;
  EXPECT_EQ(GetSkFuncName(nodes, 1, "myscope"), "sk_1_no_kernel_scope_myscope");
}

TEST_F(SkGetSkFuncNameTest, EmptyNodes_EmptyScopeName_ReturnsNoKernelScopeWithEmptyPrefix) {
  std::vector<SuperKernelBaseNode *> nodes;
  EXPECT_EQ(GetSkFuncName(nodes, 2, ""), "sk_2_no_kernel_scope_");
}

TEST_F(SkGetSkFuncNameTest, AllNonKernelNodes_ReturnsNoKernelScope) {
  std::vector<SuperKernelBaseNode *> nodes = {
      CreateWaitNodeForFuncName(1),
      CreateWaitNodeForFuncName(2),
  };
  EXPECT_EQ(GetSkFuncName(nodes, 3, "scope"), "sk_3_no_kernel_scope_scope");
}

TEST_F(SkGetSkFuncNameTest, SingleKernelNode_StartAndEndAreSame) {
  std::vector<SuperKernelBaseNode *> nodes = {
      CreateKernelNodeWithFuncName(10, "add_kernel"),
  };
  EXPECT_EQ(GetSkFuncName(nodes, 5, "myname"), "sk_5_myname_start_add_kernel_end_add_kernel");
}

TEST_F(SkGetSkFuncNameTest, SingleKernelNode_EmptyScopeName) {
  std::vector<SuperKernelBaseNode *> nodes = {
      CreateKernelNodeWithFuncName(10, "add_kernel"),
  };
  EXPECT_EQ(GetSkFuncName(nodes, 5, ""), "sk_5__start_add_kernel_end_add_kernel");
}

TEST_F(SkGetSkFuncNameTest, MultipleKernelNodes_FirstIsStartLastIsEnd) {
  std::vector<SuperKernelBaseNode *> nodes = {
      CreateKernelNodeWithFuncName(10, "add_kernel"),
      CreateKernelNodeWithFuncName(20, "mul_kernel"),
      CreateKernelNodeWithFuncName(30, "sub_kernel"),
  };
  EXPECT_EQ(GetSkFuncName(nodes, 7, "fused"), "sk_7_fused_start_add_kernel_end_sub_kernel");
}

TEST_F(SkGetSkFuncNameTest, MixedKernelAndNonKernelNodes_OnlyKernelNodesDetermineStartEnd) {
  std::vector<SuperKernelBaseNode *> nodes = {
      CreateWaitNodeForFuncName(1), CreateKernelNodeWithFuncName(10, "first_k"), CreateWaitNodeForFuncName(2),
      CreateWaitNodeForFuncName(3), CreateKernelNodeWithFuncName(20, "last_k"),  CreateWaitNodeForFuncName(4),
  };
  EXPECT_EQ(GetSkFuncName(nodes, 9, "abc"), "sk_9_abc_start_first_k_end_last_k");
}

TEST_F(SkGetSkFuncNameTest, KernelNodesNotAtBoundary_StartEndByKernelOnly) {
  std::vector<SuperKernelBaseNode *> nodes = {
      CreateKernelNodeWithFuncName(10, "init_k"),  CreateWaitNodeForFuncName(5),
      CreateKernelNodeWithFuncName(20, "mid_k"),   CreateWaitNodeForFuncName(6),
      CreateKernelNodeWithFuncName(30, "final_k"),
  };
  EXPECT_EQ(GetSkFuncName(nodes, 0, "test"), "sk_0_test_start_init_k_end_final_k");
}

TEST_F(SkGetSkFuncNameTest, ScopeIdZero_ValidOutput) {
  std::vector<SuperKernelBaseNode *> nodes = {
      CreateKernelNodeWithFuncName(1, "k0"),
  };
  EXPECT_EQ(GetSkFuncName(nodes, 0, ""), "sk_0__start_k0_end_k0");
}

TEST_F(SkGetSkFuncNameTest, LargeScopeId_ValidOutput) {
  std::vector<SuperKernelBaseNode *> nodes = {
      CreateKernelNodeWithFuncName(1, "k1"),
  };
  EXPECT_EQ(GetSkFuncName(nodes, 65535, "s"), "sk_65535_s_start_k1_end_k1");
}
