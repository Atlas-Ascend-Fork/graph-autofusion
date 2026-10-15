/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sk_dump_json.cpp
 * \brief Unit tests for SuperKernel graph JSON dump utilities
 */

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include "mockcpp/mockcpp.hpp"
#include <fstream>
#include <cstring>
#include <memory>
#include <vector>

#define private public
#define protected public

#include "sk_dump_json.h"
#include "sk_common.h"
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_scope_info.h"
#include "sk_log.h"
#include "sk_options_manager.h"
#include "sk_types.h"
#include "ut_common_stubs.h"

#include <nlohmann/json.hpp>

#include "sk_dump_json.cpp"

// ==================== Test Fixture ====================

class SkDumpJsonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    sk::logger::FileLogger::Instance().SetEnabled(false);
  }

  void TearDown() override {
    GlobalMockObject::verify();
    sk::logger::FileLogger::Instance().SetEnabled(true);
  }

  std::unique_ptr<SuperKernelKernelNode> CreateKernelNode(uint64_t nodeId) {
    auto node =
        std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, nodeId, 0, 0, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    return node;
  }

  std::unique_ptr<SuperKernelMemoryNode> CreateMemoryNode(uint64_t nodeId, aclmdlRITaskType taskType) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, taskType, nodeId, 0, 0, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    return node;
  }

  void AddKernelInfoToNode(SuperKernelKernelNode *node) {
    node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
    node->nodeInfos.kernelInfos.numBlocks = 8;
    node->nodeInfos.kernelInfos.funcName = "test_kernel";
    node->nodeInfos.kernelInfos.vecNum = 4;
    node->nodeInfos.kernelInfos.cubeNum = 2;
  }
};

// ==================== SkTaskToQueueJson Tests ====================

class SkTaskToQueueJsonTest : public SkDumpJsonTest {};

TEST_F(SkTaskToQueueJsonTest, EmptyTasks) {
  SkTask aicTask;
  SkTask aivTask;

  Json result = SkTaskToQueueJson(aicTask, aivTask, 0);
  EXPECT_EQ(result["scopeId"], 0);
}

TEST_F(SkTaskToQueueJsonTest, TasksWithInit) {
  SkTask aicTask;
  SkTask aivTask;

  ASSERT_TRUE(aicTask.Init(4));
  ASSERT_TRUE(aivTask.Init(4));

  Json result = SkTaskToQueueJson(aicTask, aivTask, 1);
  EXPECT_EQ(result["scopeId"], 1);
  EXPECT_TRUE(result.contains("taskQueues"));
}

// ==================== PrintOriginalScopes Tests ====================

class PrintOriginalScopesTest : public SkDumpJsonTest {};

TEST_F(PrintOriginalScopesTest, EmptyGraph) {
  SuperKernelGraph graph(nullptr);
  PrintOriginalScopes(graph);
  SUCCEED();
}

TEST_F(PrintOriginalScopesTest, GraphWithKernelNodes) {
  SuperKernelGraph graph(nullptr);
  auto node = CreateKernelNode(10);
  AddKernelInfoToNode(node.get());
  graph.graphMap[10] = std::move(node);

  PrintOriginalScopes(graph);
  SUCCEED();
}

// ==================== PrintFusedScopes Tests ====================

class PrintFusedScopesTest : public SkDumpJsonTest {};

TEST_F(PrintFusedScopesTest, EmptyGraphAndScopes) {
  SuperKernelGraph graph(nullptr);
  std::vector<SuperKernelScopeInfo> emptyScopes;

  PrintFusedScopes(graph, emptyScopes);
  SUCCEED();
}

TEST_F(PrintFusedScopesTest, ScopeWithKernelNodes) {
  SuperKernelGraph graph(nullptr);
  auto node1 = CreateKernelNode(10);
  AddKernelInfoToNode(node1.get());
  graph.graphMap[10] = std::move(node1);

  auto node2 = CreateKernelNode(11);
  AddKernelInfoToNode(node2.get());
  graph.graphMap[11] = std::move(node2);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.SetNodes({graph.graphMap[10].get(), graph.graphMap[11].get()});
  scopeInfo.MutableExtInfo().filteredNodes = {graph.graphMap[10].get(), graph.graphMap[11].get()};

  std::vector<SuperKernelScopeInfo> scopeInfos;
  scopeInfos.push_back(std::move(scopeInfo));

  PrintFusedScopes(graph, scopeInfos);
  SUCCEED();
}

TEST_F(PrintFusedScopesTest, ScopeWithMemoryNode) {
  SuperKernelGraph graph(nullptr);
  auto memoryNode = CreateMemoryNode(10, ACL_MODEL_RI_TASK_EVENT_RECORD);
  graph.graphMap[10] = std::move(memoryNode);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.SetNodes({graph.graphMap[10].get()});

  std::vector<SuperKernelScopeInfo> scopeInfos;
  scopeInfos.push_back(std::move(scopeInfo));

  PrintFusedScopes(graph, scopeInfos);
  SUCCEED();
}

TEST_F(PrintFusedScopesTest, DebugPerOpMaxCoreSuppressesScopeBreakReason) {
  SuperKernelGraph graph(nullptr);
  auto node1 = CreateKernelNode(10);
  AddKernelInfoToNode(node1.get());
  SuperKernelBaseNode *node1Ptr = node1.get();
  graph.graphMap[10] = std::move(node1);

  auto node2 = CreateKernelNode(11);
  AddKernelInfoToNode(node2.get());
  SuperKernelBaseNode *node2Ptr = node2.get();
  graph.graphMap[11] = std::move(node2);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.SetNodes({node1Ptr, node2Ptr});
  scopeInfo.MutableExtInfo().filteredNodes = {node1Ptr, node2Ptr};
  scopeInfo.SetBreakInfo(ScopeBreakInfo().SetReason(ScopeBreakReason::UNFUSIBLE_NODE).SetTriggerNode(11, 0));

  std::vector<SuperKernelScopeInfo> scopeInfos;
  scopeInfos.push_back(std::move(scopeInfo));

  PrintFusedScopes(graph, scopeInfos, true);
  SUCCEED();
}

// ==================== SuperKernelScopeInfo Tests ====================

class SuperKernelScopeInfoTest : public SkDumpJsonTest {};

TEST_F(SuperKernelScopeInfoTest, EmptyScope) {
  SuperKernelScopeInfo scopeInfo;
  EXPECT_GE(scopeInfo.GetScopeId(), 0);
  EXPECT_EQ(scopeInfo.GetNodes().size(), 0);
}

TEST_F(SuperKernelScopeInfoTest, ScopeWithNodes) {
  auto node1 = CreateKernelNode(10);
  AddKernelInfoToNode(node1.get());
  auto node2 = CreateKernelNode(11);
  AddKernelInfoToNode(node2.get());

  std::vector<SuperKernelBaseNode *> nodes = {node1.get(), node2.get()};

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.SetNodes(nodes);

  EXPECT_GE(scopeInfo.GetScopeId(), 0);
  EXPECT_EQ(scopeInfo.GetNodes().size(), 2);
}

TEST_F(SuperKernelScopeInfoTest, ScopeSetNodes) {
  auto node = CreateKernelNode(10);
  AddKernelInfoToNode(node.get());

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.SetNodes({node.get()});

  auto retrievedNodes = scopeInfo.GetNodes();
  EXPECT_EQ(retrievedNodes.size(), 1);
  EXPECT_EQ(retrievedNodes[0]->GetNodeId(), 10);
}

TEST_F(SuperKernelScopeInfoTest, ScopeMutableExtInfo) {
  SuperKernelScopeInfo scopeInfo;
  auto &extInfo = scopeInfo.MutableExtInfo();

  extInfo.processStatus = ScopeProcessStatus::SUCCESS;
  extInfo.filteredNodes = {};

  EXPECT_EQ(extInfo.processStatus, ScopeProcessStatus::SUCCESS);
  EXPECT_EQ(extInfo.filteredNodes.size(), 0);
}

// ==================== SkTask Tests ====================

class SkTaskTest : public SkDumpJsonTest {};

TEST_F(SkTaskTest, EmptyTask) {
  SkTask task;
  EXPECT_EQ(task.numBlocks, 0);
  EXPECT_EQ(task.funcCnt, 0);
}

TEST_F(SkTaskTest, TaskInit) {
  SkTask task;
  ASSERT_TRUE(task.Init(4));
  EXPECT_TRUE(task.GetTaskQue() != nullptr);
}

TEST_F(SkTaskTest, TaskWithData) {
  SkTask task;
  task.numBlocks = 10;
  task.funcCnt = 2;
  task.nodeType = SkKernelType::AIC_ONLY;

  ASSERT_TRUE(task.Init(4));
  EXPECT_EQ(task.numBlocks, 10);
  EXPECT_EQ(task.funcCnt, 2);
  EXPECT_EQ(task.nodeType, SkKernelType::AIC_ONLY);
}

// ==================== SuperKernelGraph Tests ====================

class DumpJsonSuperKernelGraphTest : public SkDumpJsonTest {};

TEST_F(DumpJsonSuperKernelGraphTest, EmptyGraph) {
  SuperKernelGraph graph(nullptr);
  EXPECT_EQ(graph.GetSortedNodeIds().size(), 0);
}

TEST_F(DumpJsonSuperKernelGraphTest, GraphWithSingleNode) {
  SuperKernelGraph graph(nullptr);
  auto node = CreateKernelNode(10);
  AddKernelInfoToNode(node.get());
  graph.graphMap[10] = std::move(node);

  auto sortedIds = graph.GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 1);
  EXPECT_EQ(sortedIds[0], 10);
}

TEST_F(DumpJsonSuperKernelGraphTest, GraphWithMultipleNodes) {
  SuperKernelGraph graph(nullptr);

  auto node1 = CreateKernelNode(30);
  AddKernelInfoToNode(node1.get());
  graph.graphMap[30] = std::move(node1);

  auto node2 = CreateKernelNode(10);
  AddKernelInfoToNode(node2.get());
  graph.graphMap[10] = std::move(node2);

  auto node3 = CreateKernelNode(20);
  AddKernelInfoToNode(node3.get());
  graph.graphMap[20] = std::move(node3);

  auto sortedIds = graph.GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 3);
  EXPECT_EQ(sortedIds[0], 10);
  EXPECT_EQ(sortedIds[1], 20);
  EXPECT_EQ(sortedIds[2], 30);
}

// ==================== DumpModelRITasksToJson Tests ====================

class DumpModelRITasksToJsonTest : public SkDumpJsonTest {};

// ==================== Direct JSON Helper Tests ====================

class SkDumpJsonDirectHelperTest : public SkDumpJsonTest {};

namespace {

struct __attribute__((packed)) UtDumpSknlMapInfo {
  uint64_t cap;
  void *globalFunc;
  void *sknlFunc[4];
};

struct __attribute__((packed)) UtDumpSknlValuePayload {
  uint32_t res;
  UtDumpSknlMapInfo info;
};

struct UtRITaskInternal {
  uint32_t taskId;
  aclmdlRITaskType type;
  aclmdlRITaskParams params;
};

UtRITaskInternal g_utStreamTask;

int FakeRtBinaryGetMetaNumSuccess(void *binHdl, int typeEnum, size_t *metaNum) {
  (void)binHdl;
  (void)typeEnum;
  *metaNum = 2;
  return 0;
}

int FakeRtBinaryGetMetaNumFailure(void *binHdl, int typeEnum, size_t *metaNum) {
  (void)binHdl;
  (void)typeEnum;
  (void)metaNum;
  return -1;
}

int FakeRtBinaryGetMetaInfoSuccess(void *binHdl, int typeEnum, size_t metaNum, void **dataList, size_t *sizeList) {
  (void)binHdl;
  (void)typeEnum;
  UtDumpSknlValuePayload payloads[2]{};
  payloads[0].info.cap = 4;
  payloads[0].info.globalFunc = reinterpret_cast<void *>(0x100);
  payloads[0].info.sknlFunc[0] = reinterpret_cast<void *>(0x1000);
  payloads[0].info.sknlFunc[1] = reinterpret_cast<void *>(0x1100);
  payloads[0].info.sknlFunc[2] = reinterpret_cast<void *>(0x1200);
  payloads[0].info.sknlFunc[3] = reinterpret_cast<void *>(0x1300);
  payloads[1].info.cap = 4;
  payloads[1].info.globalFunc = reinterpret_cast<void *>(0x200);
  payloads[1].info.sknlFunc[0] = reinterpret_cast<void *>(0x2000);
  payloads[1].info.sknlFunc[1] = reinterpret_cast<void *>(0x2100);
  payloads[1].info.sknlFunc[2] = reinterpret_cast<void *>(0x2200);
  payloads[1].info.sknlFunc[3] = reinterpret_cast<void *>(0x2300);

  for (size_t i = 0; i < metaNum && i < 2; ++i) {
    *static_cast<UtDumpSknlValuePayload *>(dataList[i]) = payloads[i];
    sizeList[i] = sizeof(UtDumpSknlValuePayload);
  }
  return 0;
}

int FakeRtBinaryGetMetaInfoFailure(void *binHdl, int typeEnum, size_t metaNum, void **dataList, size_t *sizeList) {
  (void)binHdl;
  (void)typeEnum;
  (void)metaNum;
  (void)dataList;
  (void)sizeList;
  return -1;
}

int FakeRtGetBinBufferEmpty(void *binHdl, int addrType, void **buffer, uint32_t *size) {
  (void)binHdl;
  (void)addrType;
  *buffer = nullptr;
  *size = 0;
  return 0;
}

aclError FakeAclrtGetFunctionAddrForDump(aclrtFuncHandle funcHdl, void **addrAicore, void **addrAiv) {
  (void)funcHdl;
  *addrAicore = reinterpret_cast<void *>(0x1100);
  *addrAiv = reinterpret_cast<void *>(0x1200);
  return ACL_SUCCESS;
}

aclError FakeAclrtBinaryGetDevAddressForDump(aclrtBinHandle binHdl, void **devAddr, size_t *devSize) {
  (void)binHdl;
  *devAddr = reinterpret_cast<void *>(0x1000);
  *devSize = 0x10000;
  return ACL_SUCCESS;
}

aclError FakeAclmdlRIGetTasksByStreamOneTask(aclrtStream stream, aclmdlRITask *tasks, uint32_t *numTasks) {
  (void)stream;
  if (numTasks == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (tasks == nullptr) {
    *numTasks = 1;
    return ACL_SUCCESS;
  }
  tasks[0] = reinterpret_cast<aclmdlRITask>(&g_utStreamTask);
  *numTasks = 1;
  return ACL_SUCCESS;
}

aclError FakeAclmdlRIGetTasksByStreamCountFailure(aclrtStream stream, aclmdlRITask *tasks, uint32_t *numTasks) {
  (void)stream;
  (void)tasks;
  (void)numTasks;
  return ACL_ERROR_FAILURE;
}

aclError FakeAclmdlRIGetTasksByStreamFetchFailure(aclrtStream stream, aclmdlRITask *tasks, uint32_t *numTasks) {
  (void)stream;
  if (numTasks == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (tasks == nullptr) {
    *numTasks = 1;
    return ACL_SUCCESS;
  }
  return ACL_ERROR_FAILURE;
}

aclError FakeAclmdlRITaskGetParamsFailure(aclmdlRITask task, aclmdlRITaskParams *params) {
  (void)task;
  (void)params;
  return ACL_ERROR_FAILURE;
}

static const std::string &MaxLengthDumpFunctionName() {
  static const std::string name = [] {
    const std::string prefix = "dump_kernel_";
    return prefix + std::string(MAX_FUNC_NAME_LEN - 1U - prefix.size(), 'x');
  }();
  return name;
}

aclError FakeAclrtGetFunctionName_MaxLengthDump(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name) {
  (void)funcHandle;
  const std::string &functionName = MaxLengthDumpFunctionName();
  if (name == nullptr || maxLen <= functionName.size()) {
    return ACL_ERROR_INVALID_PARAM;
  }
  errno_t ret = memcpy_s(name, maxLen, functionName.c_str(), functionName.size() + 1U);
  return ret == EOK ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

}  // namespace

TEST_F(SkDumpJsonDirectHelperTest, TaskAndKernelTypeStringHelpers) {
  EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_KERNEL), "KERNEL");
  EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_VALUE_WRITE), "VALUE_WRITE");
  EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_VALUE_WAIT), "VALUE_WAIT");
  EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_EVENT_RECORD), "NOTIFY");
  EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_EVENT_WAIT), "WAIT");
  EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_EVENT_RESET), "RESET");
  EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_DEFAULT), "DEFAULT");

  uint32_t taskRatio[2] = {1, 0};
  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_CUBE, taskRatio), "AIC_ONLY");
  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_VECTOR, taskRatio), "AIV_ONLY");
  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_MIX, taskRatio), "AIC_ONLY");
  taskRatio[1] = 1;
  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_MIX, taskRatio), "MIX_AIC_1_1");
  taskRatio[1] = 2;
  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_MIX, taskRatio), "MIX_AIC_1_2");

  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_AICPU, taskRatio), "DEFAULT");

  EXPECT_EQ(PtrToHexString(reinterpret_cast<void *>(0x1234)), "0x1234");
  EXPECT_EQ(UintToHexString(0xabcd), "0xabcd");
}

TEST_F(SkDumpJsonDirectHelperTest, BasicKernelParamsAcceptMaximumFunctionName) {
  aclmdlRIKernelTaskParams params{};
  params.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x300);

  ASSERT_EQ(MaxLengthDumpFunctionName().size(), MAX_FUNC_NAME_LEN - 1U);
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionName_MaxLengthDump));

  Json kernelParamsJson;
  AddBasicKernelParams(kernelParamsJson, params);

  EXPECT_EQ(kernelParamsJson["funcName"].get<std::string>(), MaxLengthDumpFunctionName());
}

TEST_F(SkDumpJsonDirectHelperTest, BinaryBindMapAndResolvedFuncsCoverSuccessAndFailurePaths) {
  aclrtBinHandle binHdl = reinterpret_cast<aclrtBinHandle>(0xaaaa);
  EXPECT_TRUE(InitSuperKernelBindMap(binHdl).empty());

  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumFailure));
  EXPECT_TRUE(InitSuperKernelBindMap(binHdl).empty());
}

TEST_F(SkDumpJsonDirectHelperTest, BinaryBindMapAndResolvedFuncsSerializeResolvedEntries) {
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumSuccess));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSuccess));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferEmpty));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForDump));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForDump));

  aclrtBinHandle binHdl = reinterpret_cast<aclrtBinHandle>(0xbbbb);
  SkBindMap bindMap = InitSuperKernelBindMap(binHdl);
  EXPECT_EQ(bindMap.size(), 2);
  EXPECT_EQ(bindMap[0x100].cap, 4);
  EXPECT_EQ(bindMap[0x100].sknlFuncs[0], 0x1000);
  EXPECT_EQ(bindMap[0x200].cap, 4);
  EXPECT_EQ(bindMap[0x200].sknlFuncs[3], 0x2300);

  ResolvedFunctionInfo resolvedFuncs[K_MAX_SPLIT_BIN_COUNT];
  uint32_t resolvedNum = 0;
  GetResolvedFuncsForDump(reinterpret_cast<aclrtFuncHandle>(0xcccc), binHdl, resolvedFuncs, resolvedNum);
  EXPECT_EQ(resolvedNum, K_MAX_SPLIT_BIN_COUNT);
  EXPECT_EQ(resolvedFuncs[0].funcAddr[0], 0x2000);
  EXPECT_EQ(resolvedFuncs[0].funcAddr[1], 0x3000);
  Json resolvedJson;
  AddResolvedFuncsToJson(resolvedJson, reinterpret_cast<aclrtFuncHandle>(0xcccc), binHdl);
  EXPECT_EQ(resolvedJson["resolvedNum"], K_MAX_SPLIT_BIN_COUNT);
  EXPECT_EQ(resolvedJson["resolvedFuncs"].size(), K_MAX_SPLIT_BIN_COUNT);
}

TEST_F(SkDumpJsonDirectHelperTest, BinaryBindMapMetaInfoFailureReturnsEmpty) {
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumSuccess));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoFailure));
  EXPECT_TRUE(InitSuperKernelBindMap(reinterpret_cast<aclrtBinHandle>(0xdddd)).empty());
}

TEST_F(SkDumpJsonDirectHelperTest, ParseOriginalScopesGroupsNamedScopeNodesByBitFlags) {
  SuperKernelGraph graph(nullptr);
  std::bitset<MAX_SCOPE_NUM> flags;
  flags.set(3);

  auto marker = CreateKernelNode(1);
  marker->SetIsScopeNode(true);
  marker->isScopeBegin = true;
  marker->scopeName = "named_scope";
  marker->SetScopeBitFlags(flags);

  auto scopedNode = CreateKernelNode(2);
  scopedNode->SetNodeType(SkNodeType::NODE_KERNEL);
  scopedNode->SetScopeBitFlags(flags);

  auto emptyFlagsNode = CreateKernelNode(3);
  emptyFlagsNode->SetNodeType(SkNodeType::NODE_KERNEL);

  graph.graphMap[1] = std::move(marker);
  graph.graphMap[2] = std::move(scopedNode);
  graph.graphMap[3] = std::move(emptyFlagsNode);

  graph.ParseOriginalScopes();
  ASSERT_EQ(graph.GetOriginalScopeInfos().size(), 1);
  EXPECT_EQ(graph.GetOriginalScopeInfos()[0].nodeIds, std::vector<uint64_t>({2}));
  EXPECT_EQ(graph.GetOriginalScopeInfos()[0].GetScopeBitFlags(), flags);
}

TEST_F(SkDumpJsonDirectHelperTest, ParseOriginalScopesCreatesSingleScopeWhenNoScopeNodesExist) {
  SuperKernelGraph graph(nullptr);
  std::bitset<MAX_SCOPE_NUM> flags;
  flags.set(4);

  auto node1 = CreateKernelNode(10);
  node1->SetNodeType(SkNodeType::NODE_KERNEL);
  node1->SetScopeBitFlags(flags);
  auto node2 = CreateMemoryNode(11, ACL_MODEL_RI_TASK_EVENT_WAIT);
  node2->SetNodeType(SkNodeType::NODE_WAIT);
  node2->SetScopeBitFlags(flags);

  graph.graphMap[10] = std::move(node1);
  graph.graphMap[11] = std::move(node2);

  graph.ParseOriginalScopes();
  ASSERT_EQ(graph.GetOriginalScopeInfos().size(), 1);
  EXPECT_EQ(graph.GetOriginalScopeInfos()[0].nodeIds, std::vector<uint64_t>({10, 11}));
}

TEST_F(SkDumpJsonDirectHelperTest, ParseOriginalScopesExcludesUnnamedScopeBody) {
  SuperKernelGraph graph(nullptr);
  std::bitset<MAX_SCOPE_NUM> flags;
  flags.set(5);

  auto unnamedBegin = CreateKernelNode(20);
  unnamedBegin->SetIsScopeNode(true);
  unnamedBegin->isScopeBegin = true;
  unnamedBegin->SetScopeBitFlags(flags);

  auto insideNode = CreateKernelNode(21);
  insideNode->SetNodeType(SkNodeType::NODE_KERNEL);
  insideNode->SetScopeBitFlags(flags);

  auto unnamedEnd = CreateKernelNode(22);
  unnamedEnd->SetIsScopeNode(true);
  unnamedEnd->isScopeEnd = true;
  unnamedEnd->SetScopeBitFlags(flags);

  auto placeholder = CreateKernelNode(23);
  placeholder->SetIsScopeNode(true);
  placeholder->isPlaceholder = true;
  placeholder->SetScopeBitFlags(flags);

  auto outsideNode = CreateKernelNode(24);
  outsideNode->SetNodeType(SkNodeType::NODE_KERNEL);
  outsideNode->SetScopeBitFlags(flags);

  graph.graphMap[20] = std::move(unnamedBegin);
  graph.graphMap[21] = std::move(insideNode);
  graph.graphMap[22] = std::move(unnamedEnd);
  graph.graphMap[23] = std::move(placeholder);
  graph.graphMap[24] = std::move(outsideNode);

  graph.ParseOriginalScopes();
  ASSERT_EQ(graph.GetOriginalScopeInfos().size(), 1);
  EXPECT_EQ(graph.GetOriginalScopeInfos()[0].nodeIds, std::vector<uint64_t>({24}));
}

TEST_F(SkDumpJsonDirectHelperTest, SkTaskQueueAndFileWritingHelpers) {
  SkTask task;
  task.numBlocks = 16;
  task.funcCnt = 2;
  task.nodeType = SkKernelType::AIC_ONLY;
  ASSERT_TRUE(task.Init(2));

  TaskQue *queue = task.GetTaskQue();
  ASSERT_NE(queue, nullptr);
  queue->taskCnt = 2;
  queue->taskInfos[0].index = 10;
  queue->taskInfos[0].type = SkTaskType::TYPE_FUNC;
  queue->taskInfos[0].numBlocks = 8;
  queue->taskInfos[0].entryCnt = 2;
  queue->taskInfos[0].args = 0x1000;
  queue->taskInfos[0].entry[0] = 0x2000;
  queue->taskInfos[0].entry[1] = 0x3000;
  queue->taskInfos[0].debugOptions = 3;
  queue->taskInfos[0].isSimtKernel = 1;
  queue->taskInfos[1].index = 11;
  queue->taskInfos[1].type = SkTaskType::TYPE_EVENT_WAIT;
  queue->taskInfos[1].numBlocks = 1;
  queue->taskInfos[1].entryCnt = 0;
  queue->taskInfos[1].args = 0x4000;
  queue->taskInfos[1].entry[0] = 0x5000;
  queue->taskInfos[1].extraInfo = static_cast<uint64_t>(SkMemoryWaitFlag::AND);
  queue->taskInfos[1].debugOptions = 16;

  Json taskJson = SkTaskToJson(task);
  EXPECT_EQ(taskJson["taskQue"]["taskCnt"], 2);
  EXPECT_FALSE(taskJson["taskQue"]["taskInfos"][0].contains("isSimtKernel"));
  EXPECT_FALSE(taskJson["taskQue"]["taskInfos"][1].contains("isSimtKernel"));
  EXPECT_EQ(taskJson["taskQue"]["taskInfos"][0]["entries"].size(), 2);
  EXPECT_EQ(taskJson["taskQue"]["taskInfos"][1]["entryCnt"], 0);
  EXPECT_EQ(taskJson["taskQue"]["taskInfos"][1]["entries"].size(), 0);
  EXPECT_EQ(taskJson["taskQue"]["taskInfos"][1]["eventValue"], "0x5000");
  EXPECT_EQ(taskJson["taskQue"]["taskInfos"][1]["waitFlag"], static_cast<uint32_t>(SkMemoryWaitFlag::AND));

  Json noQueueJson = SkTaskToJson(SkTask());
  EXPECT_FALSE(noQueueJson.contains("taskQue"));

  SuperKernelGraph graph(nullptr, "model_nullptr");
  sk::logger::FileLogger::Instance().SetEnabled(true);
  std::unordered_map<std::string, Json> taskQueues;
  taskQueues["99"] = SkTaskToQueueJson(task, task, 99);
  EXPECT_TRUE(DumpAllTaskQueuesToJson(graph, taskQueues));
  sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(SkDumpJsonDirectHelperTest, RawTaskParamJsonCoversAllTaskTypes) {
  aclmdlRITaskParams params{};
  params.type = ACL_MODEL_RI_TASK_KERNEL;
  params.taskGrp = reinterpret_cast<aclrtTaskGrp>(0x100);
  params.opInfoPtr = reinterpret_cast<void *>(0x200);
  params.opInfoSize = 16;
  params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x300);
  params.kernelTaskParams.args = reinterpret_cast<void *>(0x400);
  params.kernelTaskParams.argsSize = 32;
  params.kernelTaskParams.isHostArgs = 1;
  params.kernelTaskParams.numBlocks = 64;
  aclrtLaunchKernelAttr attr{};
  attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  attr.value.schemMode = 1;
  aclrtLaunchKernelCfg cfg{&attr, 1};
  params.kernelTaskParams.cfg = &cfg;

  Json kernelTaskJson = TaskToJson(0, 7, ACL_MODEL_RI_TASK_KERNEL, &params, nullptr);
  EXPECT_EQ(kernelTaskJson["streamId"], 7);
  EXPECT_EQ(kernelTaskJson["taskType"], "KERNEL");
  EXPECT_EQ(kernelTaskJson["kernelParams"]["numBlocks"], 64);
  EXPECT_EQ(kernelTaskJson["kernelParams"]["taskGrp"], "0x100");
  EXPECT_EQ(kernelTaskJson["kernelParams"]["kernelType"], "DEFAULT");
  EXPECT_EQ(kernelTaskJson["kernelParams"]["taskRatio"].size(), 2);

  params = {};
  params.type = ACL_MODEL_RI_TASK_EVENT_RECORD;
  params.eventRecordTaskParams.event = reinterpret_cast<aclrtEvent>(0x500);
  params.eventRecordTaskParams.eventFlag = 9;
  params.eventRecordTaskParams.recordFlag = ACL_EVENT_RECORD_EXTERNAL;
  Json recordJson;
  AddTaskParamsToJson(recordJson, params, nullptr);
  EXPECT_EQ(recordJson["eventRecordParams"]["eventId"], "0x500");
  EXPECT_EQ(recordJson["eventRecordParams"]["eventFlag"], 9);
  EXPECT_EQ(recordJson["eventRecordParams"]["recordFlag"], ACL_EVENT_RECORD_EXTERNAL);

  params = {};
  params.type = ACL_MODEL_RI_TASK_EVENT_WAIT;
  params.eventWaitTaskParams.event = reinterpret_cast<aclrtEvent>(0x600);
  params.eventWaitTaskParams.eventFlag = 10;
  params.eventWaitTaskParams.waitFlag = ACL_EVENT_WAIT_EXTERNAL;
  Json waitJson = TaskToJson(1, 8, ACL_MODEL_RI_TASK_EVENT_WAIT, &params, nullptr);
  EXPECT_EQ(waitJson["eventWaitParams"]["event"], "0x600");
  EXPECT_EQ(waitJson["eventWaitParams"]["eventFlag"], 10);
  EXPECT_EQ(waitJson["eventWaitParams"]["waitFlag"], ACL_EVENT_WAIT_EXTERNAL);

  params = {};
  params.type = ACL_MODEL_RI_TASK_EVENT_RESET;
  params.eventResetTaskParams.event = reinterpret_cast<aclrtEvent>(0x700);
  params.eventResetTaskParams.eventFlag = 11;
  params.eventResetTaskParams.resetFlag = 1;
  Json resetJson;
  AddTaskParamsToJson(resetJson, params, nullptr);
  EXPECT_EQ(resetJson["eventResetParams"]["event"], "0x700");
  EXPECT_EQ(resetJson["eventResetParams"]["eventFlag"], 11);
  EXPECT_EQ(resetJson["eventResetParams"]["resetFlag"], 1);

  params = {};
  params.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
  params.valueWriteTaskParams.devAddr = reinterpret_cast<void *>(0x800);
  params.valueWriteTaskParams.value = 0x900;
  Json valueWriteJson;
  AddTaskParamsToJson(valueWriteJson, params, nullptr);
  EXPECT_EQ(valueWriteJson["valueWriteParams"]["devAddr"], "0x800");
  EXPECT_EQ(valueWriteJson["valueWriteParams"]["value"], "0x900");

  params = {};
  params.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
  params.valueWaitTaskParams.devAddr = reinterpret_cast<void *>(0xa00);
  params.valueWaitTaskParams.value = 0xb00;
  params.valueWaitTaskParams.flag = 13;
  Json valueWaitJson;
  AddTaskParamsToJson(valueWaitJson, params, nullptr);
  EXPECT_EQ(valueWaitJson["valueWaitParams"]["devAddr"], "0xa00");
  EXPECT_EQ(valueWaitJson["valueWaitParams"]["value"], "0xb00");
  EXPECT_EQ(valueWaitJson["valueWaitParams"]["flag"], 13);

  params = {};
  params.type = ACL_MODEL_RI_TASK_DEFAULT;
  Json defaultJson;
  AddTaskParamsToJson(defaultJson, params, nullptr);
  EXPECT_TRUE(defaultJson.empty());
  EXPECT_EQ(TaskToJson(2, 9, ACL_MODEL_RI_TASK_DEFAULT, nullptr, nullptr)["taskType"], "DEFAULT");
}

TEST_F(SkDumpJsonDirectHelperTest, ScopePrintingHelpersCoverKernelSetAndBatchBranches) {
  EXPECT_STREQ(to_string(ScopeProcessStatus::INIT), "INIT");
  EXPECT_STREQ(to_string(ScopeProcessStatus::SUCCESS), "SUCCESS");
  EXPECT_STREQ(to_string(ScopeProcessStatus::RESOURCE_INSUFFICIENT), "RESOURCE_INSUFFICIENT");
  EXPECT_STREQ(to_string(ScopeProcessStatus::NO_TARGET_NODE), "NO_TARGET_NODE");
  EXPECT_STREQ(to_string(ScopeProcessStatus::UNRECOVERABLE_FAIL), "UNRECOVERABLE_FAIL");
  EXPECT_STREQ(to_string(static_cast<ScopeProcessStatus>(255)), "UNKNOWN");
  EXPECT_STREQ(ScopeProcessStatusDetail(ScopeProcessStatus::INIT), "Scope has not been processed");
  EXPECT_STREQ(ScopeProcessStatusDetail(ScopeProcessStatus::RESOURCE_INSUFFICIENT),
               "Insufficient stream task slots or event memory resources");
  EXPECT_STREQ(ScopeProcessStatusDetail(ScopeProcessStatus::NO_TARGET_NODE), "No target node remains after filtering");
  EXPECT_STREQ(ScopeProcessStatusDetail(ScopeProcessStatus::UNRECOVERABLE_FAIL),
               "Unrecoverable failure that cannot be skipped");
  EXPECT_STREQ(ScopeProcessStatusDetail(static_cast<ScopeProcessStatus>(255)), "");
  EXPECT_STREQ(to_string(ScopeBreakReason::NONE), "NONE");
  EXPECT_STREQ(to_string(ScopeBreakReason::UNFUSIBLE_NODE), "UNFUSIBLE_NODE");
  EXPECT_STREQ(to_string(ScopeBreakReason::DEADLOCK_DETECTED), "DEADLOCK_DETECTED");
  EXPECT_STREQ(to_string(ScopeBreakReason::SYNCALL_OP_DROP), "SYNCALL_OP_DROP");
  EXPECT_STREQ(to_string(ScopeBreakReason::DEBUG_PER_OP_MAX_CORE), "DEBUG_PER_OP_MAX_CORE");
  EXPECT_STREQ(ScopeBreakReasonDetail(ScopeBreakReason::UNFUSIBLE_NODE), "There exists unfusible node in scope");
  EXPECT_STREQ(ScopeBreakReasonDetail(ScopeBreakReason::DEADLOCK_DETECTED), "There exists deadlock in scope");
  EXPECT_NE(std::string(ScopeBreakReasonDetail(ScopeBreakReason::SYNCALL_OP_DROP))
                .find("greater than or equal to other operators"),
            std::string::npos);
  EXPECT_STREQ(ScopeBreakReasonDetail(ScopeBreakReason::DEBUG_PER_OP_MAX_CORE),
               "Per-Op debug mode: each operator is an independent scope");
  EXPECT_STREQ(to_string(static_cast<ScopeBreakReason>(255)), "UNKNOWN_SCOPE_BREAK_REASON");
  EXPECT_STREQ(ScopeBreakReasonDetail(static_cast<ScopeBreakReason>(255)), "");

  SuperKernelGraph graph(nullptr);
  graph.scopeIdxToName[1] = "scope_one";
  std::bitset<MAX_SCOPE_NUM> flags;
  flags.set(1);

  auto node1 = CreateKernelNode(201);
  node1->SetNodeType(SkNodeType::NODE_KERNEL);
  node1->nodeInfos.kernelInfos.funcName = "kernel_201";
  node1->SetScopeBitFlags(flags);
  auto node2 = CreateKernelNode(202);
  node2->SetNodeType(SkNodeType::NODE_KERNEL);
  node2->nodeInfos.kernelInfos.funcName = "kernel_202";
  node2->SetScopeBitFlags(flags);
  auto scopeMarker = CreateKernelNode(203);
  scopeMarker->SetNodeType(SkNodeType::NODE_KERNEL);
  scopeMarker->isScopeBegin = true;
  scopeMarker->SetScopeBitFlags(flags);

  SuperKernelBaseNode *node1Ptr = node1.get();
  SuperKernelBaseNode *node2Ptr = node2.get();
  SuperKernelBaseNode *markerPtr = scopeMarker.get();
  graph.graphMap[201] = std::move(node1);
  graph.graphMap[202] = std::move(node2);
  graph.graphMap[203] = std::move(scopeMarker);
  graph.originalScopeInfos_.push_back({1, flags, {201, 203, 999}});

  std::vector<uint64_t> longNodeIds;
  for (uint64_t i = 0; i < 700; ++i) {
    longNodeIds.push_back(1000000000000ULL + i);
  }
  PrintNodeIdsInBatches(longNodeIds, "    hugeNodeIds: [");
  PrintFusedNodeIds(longNodeIds);
  PrintOriginalScopes(graph);

  const SuperKernelBaseNode *nodes[] = {nullptr, node1Ptr, markerPtr, node2Ptr};
  auto kernelIds = CollectKernelIdsFromNodes(nodes, 4);
  EXPECT_EQ(kernelIds.size(), 2);
  EXPECT_EQ(kernelIds.count(201), 1);
  EXPECT_EQ(kernelIds.count(202), 1);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.SetScopeBitFlags(flags);
  scopeInfo.SetNodes({node1Ptr, node2Ptr, markerPtr});
  scopeInfo.MutableExtInfo().filteredNodes = {nullptr, markerPtr, node1Ptr, node2Ptr};
  scopeInfo.MutableExtInfo().processStatus = ScopeProcessStatus::RESOURCE_INSUFFICIENT;
  scopeInfo.SetBreakInfo(ScopeBreakInfo()
                             .SetReason(ScopeBreakReason::UNFUSIBLE_NODE)
                             .SetTriggerNode(202, 0)
                             .SetSyncAllNodeIds({201, 202}));
  std::string breakInfo = scopeInfo.GetBreakInfo().Format();
  EXPECT_NE(breakInfo.find("breakReason=UNFUSIBLE_NODE"), std::string::npos);
  EXPECT_NE(breakInfo.find("syncAllNodes=[201,202]"), std::string::npos);
  EXPECT_EQ(breakInfo.find("breakReasonDetail="), std::string::npos);

  auto originalSets = BuildOriginalKernelSets(graph, graph.GetOriginalScopeInfos());
  EXPECT_FALSE(IsKernelSetMatch(scopeInfo, originalSets, graph));

  std::vector<uint64_t> fusedNodeIds;
  bool hasKernel = false;
  CollectFusedNodes(scopeInfo.GetExtInfo(), fusedNodeIds, hasKernel);
  EXPECT_TRUE(hasKernel);
  EXPECT_EQ(fusedNodeIds.size(), 2);

  std::vector<SuperKernelScopeInfo> scopeInfos;
  scopeInfos.push_back(std::move(scopeInfo));
  PrintFusedScopes(graph, scopeInfos);
}

TEST_F(SkDumpJsonDirectHelperTest, ScopePrintingHelpersCoverSuccessScopeStatus) {
  SuperKernelGraph graph(nullptr);
  std::bitset<MAX_SCOPE_NUM> flags;
  flags.set(2);

  auto node = CreateKernelNode(301);
  node->SetNodeType(SkNodeType::NODE_KERNEL);
  node->SetScopeBitFlags(flags);
  SuperKernelBaseNode *nodePtr = node.get();
  graph.graphMap[301] = std::move(node);
  graph.originalScopeInfos_.push_back({2, flags, {301}});

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.SetScopeBitFlags(flags);
  scopeInfo.SetNodes({nodePtr});
  scopeInfo.MutableExtInfo().filteredNodes = {nodePtr};
  scopeInfo.MutableExtInfo().processStatus = ScopeProcessStatus::SUCCESS;

  std::vector<SuperKernelScopeInfo> scopeInfos;
  scopeInfos.push_back(std::move(scopeInfo));
  PrintFusedScopes(graph, scopeInfos);
}

TEST_F(SkDumpJsonDirectHelperTest, InjectSkInfosBuildsMetadataFromScopeInfos) {
  Json graphJson;
  graphJson["streams"] = Json::array();
  graphJson["streams"].push_back({{"nodes", Json::array({{{"nodeId", 100U}}, {{"nodeId", 200U}}})}});

  auto startNode = CreateKernelNode(10);
  startNode->SetNodeType(SkNodeType::NODE_KERNEL);
  startNode->nodeInfos.kernelInfos.funcName = "start_func";
  auto endNode = CreateKernelNode(11);
  endNode->SetNodeType(SkNodeType::NODE_KERNEL);
  endNode->nodeInfos.kernelInfos.funcName = "end_func";

  SuperKernelBaseNode *startNodePtr = startNode.get();
  SuperKernelBaseNode *endNodePtr = endNode.get();

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.MutableExtInfo().processStatus = ScopeProcessStatus::SUCCESS;
  scopeInfo.MutableExtInfo().skMainNodeId = 200;
  scopeInfo.MutableExtInfo().scopeName = "scope_a";
  scopeInfo.MutableExtInfo().filteredNodes = {startNodePtr, endNodePtr};

  std::vector<SuperKernelScopeInfo> scopeInfos;
  scopeInfos.push_back(std::move(scopeInfo));

  InjectSkInfos(graphJson, scopeInfos);

  EXPECT_FALSE(graphJson["streams"][0]["nodes"][0].contains("skinfos"));
  ASSERT_TRUE(graphJson["streams"][0]["nodes"][1].contains("skinfos"));
  EXPECT_EQ(graphJson["streams"][0]["nodes"][1]["skinfos"]["scopeId"], scopeInfos[0].GetScopeId());
  EXPECT_EQ(graphJson["streams"][0]["nodes"][1]["skinfos"]["scopeName"], "scope_a");
  EXPECT_EQ(graphJson["streams"][0]["nodes"][1]["skinfos"]["startKernelFuncName"], "start_func");
  EXPECT_EQ(graphJson["streams"][0]["nodes"][1]["skinfos"]["endKernelFuncName"], "end_func");
}

// ==================== DumpGraphJson Tests ====================

class DumpGraphJsonTest : public SkDumpJsonTest {};

TEST_F(DumpGraphJsonTest, DisabledLoggerSkipDump) {
  sk::logger::FileLogger::Instance().SetEnabled(false);

  SuperKernelOptionsManager opts;
  bool result = DumpGraphJson(nullptr, "model_nullptr", opts, "/tmp", "test_raw_tasks");

  EXPECT_TRUE(result);

  sk::logger::FileLogger::Instance().SetEnabled(true);
}

TEST_F(DumpGraphJsonTest, EmptyModelRISucceedsWithEmptyGraph) {
  sk::logger::FileLogger::Instance().SetEnabled(true);

  SuperKernelOptionsManager opts;
  std::string metaDir = CreateSkMetaDirectory("model_nullptr");
  bool result = DumpGraphJson(nullptr, "model_nullptr", opts, metaDir, "test_raw_tasks");

  EXPECT_TRUE(result);

  sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(DumpGraphJsonTest, InvalidMetaDirFails) {
  sk::logger::FileLogger::Instance().SetEnabled(true);

  SuperKernelOptionsManager opts;
  bool result = DumpGraphJson(nullptr, "model_nullptr", opts, "/nonexistent_dir_12345", "test_raw_tasks");

  EXPECT_FALSE(result);

  sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(DumpGraphJsonTest, EmptyFilenameWithValidDir) {
  sk::logger::FileLogger::Instance().SetEnabled(true);

  SuperKernelOptionsManager opts;
  std::string metaDir = CreateSkMetaDirectory("model_nullptr");
  bool result = DumpGraphJson(nullptr, "model_nullptr", opts, metaDir, "");

  EXPECT_FALSE(result);

  sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(DumpGraphJsonTest, ValidMetaDirAndFilename) {
  sk::logger::FileLogger::Instance().SetEnabled(true);

  SuperKernelOptionsManager opts;
  opts.AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1));
  opts.AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 4));

  std::string metaDir = CreateSkMetaDirectory("model_nullptr");
  std::string filename = "test_dump_valid";
  bool result = DumpGraphJson(nullptr, "model_nullptr", opts, metaDir, filename);

  EXPECT_TRUE(result);

  std::string jsonPath = metaDir + "/" + filename + ".json";
  std::ifstream file(jsonPath);
  EXPECT_TRUE(file.good());

  sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(DumpGraphJsonTest, ExistingGraphWritesWithoutRebuildingFromModelRI) {
  sk::logger::FileLogger::Instance().SetEnabled(true);
  SkUtResetTestControls();

  SuperKernelOptionsManager opts;
  aclmdlRI model = reinterpret_cast<aclmdlRI>(static_cast<uintptr_t>(0x40));  // id=64
  const std::string modelId = "model_64_1";

  SuperKernelGraph graph(nullptr, opts, modelId);
  auto node = CreateKernelNode(400);
  AddKernelInfoToNode(node.get());
  graph.graphMap[400] = std::move(node);

  std::string metaDir = CreateSkMetaDirectory(modelId);
  std::string filename = "test_existing_graph";
  bool result = DumpGraphJson(graph, opts, metaDir, filename);

  EXPECT_TRUE(result);

  std::ifstream file(metaDir + "/" + filename + ".json");
  ASSERT_TRUE(file.good());

  Json rootJson;
  file >> rootJson;
  EXPECT_EQ(rootJson["modelId"], modelId);
  EXPECT_EQ(rootJson["totalNodes"], 1);

  sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(DumpGraphJsonTest, WithPopulatedOptionsManager) {
  sk::logger::FileLogger::Instance().SetEnabled(true);

  SuperKernelOptionsManager opts;
  opts.AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 2));
  opts.AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 3));
  opts.AddOption(std::make_unique<NumberOptOption>("debug_sync_all", aclskOptionType::DEBUG_SYNC_ALL, 1));

  std::vector<std::string> dcciKernels = {"Add", "Mul", "Conv"};
  opts.AddOption(
      std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL, dcciKernels));

  std::string metaDir = CreateSkMetaDirectory("model_nullptr");
  std::string filename = "test_with_options";
  bool result = DumpGraphJson(nullptr, "model_nullptr", opts, metaDir, filename);

  EXPECT_TRUE(result);

  sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(DumpGraphJsonTest, FileAlreadyExists) {
  sk::logger::FileLogger::Instance().SetEnabled(true);

  SuperKernelOptionsManager opts;
  std::string metaDir = CreateSkMetaDirectory("model_nullptr");
  std::string filename = "test_existing";
  std::string jsonPath = metaDir + "/" + filename + ".json";

  std::ofstream existingFile(jsonPath);
  existingFile << "{\"existing\": true}";
  existingFile.close();

  bool result = DumpGraphJson(nullptr, "model_nullptr", opts, metaDir, filename);

  EXPECT_TRUE(result);

  std::ifstream file(jsonPath);
  std::string content;
  std::getline(file, content, '\0');
  EXPECT_TRUE(content.find("existing") == std::string::npos);

  sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(DumpGraphJsonTest, ExplicitModelIdWritesCanonicalValueToJson) {
  sk::logger::FileLogger::Instance().SetEnabled(true);
  SkUtResetTestControls();
  SkUtSetModelStreamNum(0);

  SuperKernelOptionsManager opts;
  aclmdlRI model = reinterpret_cast<aclmdlRI>(static_cast<uintptr_t>(0x30));  // id=48
  const std::string modelId = "model_48_1";
  std::string metaDir = CreateSkMetaDirectory(modelId);
  std::string filename = "test_model_id_context";

  bool result = DumpGraphJson(model, modelId, opts, metaDir, filename);
  EXPECT_TRUE(result);

  std::ifstream file(metaDir + "/" + filename + ".json");
  ASSERT_TRUE(file.good());

  Json rootJson;
  file >> rootJson;
  EXPECT_EQ(rootJson["modelId"], modelId);

  sk::logger::FileLogger::Instance().SetEnabled(false);
}

// ==================== SuperKernelGraph::ToJson Tests ====================

class SuperKernelGraphToJsonTest : public SkDumpJsonTest {};

TEST_F(SuperKernelGraphToJsonTest, EmptyGraphToJson) {
  SuperKernelGraph graph(nullptr);

  Json json = graph.ToJson();
  EXPECT_TRUE(json.contains("version"));
  EXPECT_TRUE(json.contains("description"));
  EXPECT_TRUE(json.contains("modelId"));
  EXPECT_TRUE(json.contains("deviceId"));
  EXPECT_TRUE(json.contains("options"));
  EXPECT_TRUE(json.contains("totalStreams"));
  EXPECT_TRUE(json.contains("totalNodes"));
  EXPECT_TRUE(json.contains("streams"));

  EXPECT_EQ(json["version"], "1.0");
  EXPECT_EQ(json["totalStreams"], 0);
  EXPECT_EQ(json["totalNodes"], 0);
  EXPECT_TRUE(json["streams"].is_array());
  EXPECT_EQ(json["streams"].size(), 0);
}

TEST_F(SuperKernelGraphToJsonTest, GraphWithKernelNodesToJson) {
  SuperKernelGraph graph(nullptr);
  graph.scopeIdxToName[0] = "(none)";
  graph.scopeIdxToName[1] = "scope_a";

  auto node1 = CreateKernelNode(10);
  AddKernelInfoToNode(node1.get());
  node1->SetNodeType(SkNodeType::NODE_KERNEL);
  node1->streamIdxInGraph = 0;
  node1->streamId = 5;
  graph.graphMap[10] = std::move(node1);

  auto node2 = CreateKernelNode(20);
  AddKernelInfoToNode(node2.get());
  node2->SetNodeType(SkNodeType::NODE_KERNEL);
  node2->streamIdxInGraph = 0;
  node2->streamId = 5;
  graph.graphMap[20] = std::move(node2);

  graph.streams.push_back(reinterpret_cast<aclrtStream>(0x100));
  graph.headNodes.push_back(10);
  graph.nodeSizeInStream.push_back(2);

  Json json = graph.ToJson();
  EXPECT_EQ(json["totalNodes"], 2);
  EXPECT_EQ(json["totalStreams"], 1);
  EXPECT_EQ(json["streams"].size(), 1);

  if (json["streams"].size() > 0) {
    EXPECT_EQ(json["streams"][0]["streamIdxInGraph"], 0);
    EXPECT_TRUE(json["streams"][0].contains("nodeCount"));
    EXPECT_TRUE(json["streams"][0].contains("nodes"));
    ASSERT_EQ(json["streams"][0]["nodes"].size(), 2);
    EXPECT_TRUE(json["streams"][0]["nodes"][0].contains("taskInfo"));
    EXPECT_FALSE(json["streams"][0]["nodes"][0].contains("tasks"));
  }
}

TEST_F(SuperKernelGraphToJsonTest, GraphWithMixedNodeTypesToJson) {
  SuperKernelGraph graph(nullptr);

  auto kernelNode = CreateKernelNode(10);
  AddKernelInfoToNode(kernelNode.get());
  kernelNode->SetNodeType(SkNodeType::NODE_KERNEL);
  kernelNode->streamIdxInGraph = 0;
  kernelNode->streamId = 1;
  graph.graphMap[10] = std::move(kernelNode);

  auto memoryNode = CreateMemoryNode(11, ACL_MODEL_RI_TASK_EVENT_RECORD);
  memoryNode->SetNodeType(SkNodeType::NODE_NOTIFY);
  memoryNode->streamIdxInGraph = 0;
  memoryNode->streamId = 1;
  memoryNode->nodeInfos.syncInfos.eventId = 0x1234;
  graph.graphMap[11] = std::move(memoryNode);

  auto waitNode = CreateMemoryNode(12, ACL_MODEL_RI_TASK_EVENT_WAIT);
  waitNode->SetNodeType(SkNodeType::NODE_WAIT);
  waitNode->streamIdxInGraph = 0;
  waitNode->streamId = 1;
  waitNode->nodeInfos.syncInfos.eventId = 0x5678;
  graph.graphMap[12] = std::move(waitNode);

  auto defaultNode = std::make_unique<SuperKernelDefaultNode>(nullptr, ACL_MODEL_RI_TASK_DEFAULT, 0, 0,
                                                              INVALID_STREAM_ID, INVALID_TASK_ID);
  defaultNode->SetNodeId(13);
  defaultNode->SetNodeType(SkNodeType::NODE_DEFAULT);
  defaultNode->streamIdxInGraph = 0;
  defaultNode->streamId = 1;
  graph.graphMap[13] = std::move(defaultNode);

  graph.streams.push_back(reinterpret_cast<aclrtStream>(0x200));
  graph.headNodes.push_back(10);
  graph.nodeSizeInStream.push_back(4);

  Json json = graph.ToJson();
  EXPECT_EQ(json["totalNodes"], 4);
  EXPECT_EQ(json["totalStreams"], 1);
}

TEST_F(SuperKernelGraphToJsonTest, GraphWithMultipleStreamsToJson) {
  SuperKernelGraph graph(nullptr);

  for (int i = 0; i < 3; ++i) {
    auto node = CreateKernelNode(100 + i);
    AddKernelInfoToNode(node.get());
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->streamIdxInGraph = i;
    node->streamId = i * 10;
    graph.graphMap[100 + i] = std::move(node);

    graph.streams.push_back(reinterpret_cast<aclrtStream>(0x1000 + i));
    graph.headNodes.push_back(100 + i);
    graph.nodeSizeInStream.push_back(1);
  }

  Json json = graph.ToJson();
  EXPECT_EQ(json["totalStreams"], 3);
  EXPECT_EQ(json["totalNodes"], 3);
  EXPECT_EQ(json["streams"].size(), 3);
}

TEST_F(SuperKernelGraphToJsonTest, GraphWithOptionsToJson) {
  SuperKernelOptionsManager opts;
  opts.AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1));
  opts.AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 4));

  SuperKernelGraph graph(nullptr, opts);

  Json json = graph.ToJson();
  EXPECT_TRUE(json.contains("options"));
  EXPECT_TRUE(json["options"].contains("numOptions"));
}

TEST_F(SuperKernelGraphToJsonTest, GraphWithMemoryWriteAndWaitNodesToJson) {
  SuperKernelGraph graph(nullptr);

  auto writeNode = CreateMemoryNode(10, ACL_MODEL_RI_TASK_VALUE_WRITE);
  writeNode->SetNodeType(SkNodeType::NODE_MEMORY_WRITE);
  writeNode->streamIdxInGraph = 0;
  writeNode->streamId = 1;
  writeNode->nodeInfos.syncInfos.eventId = 0x1111;
  writeNode->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x2222);
  writeNode->nodeInfos.syncInfos.memoryValue = 0x3333;
  writeNode->nodeInfos.syncInfos.memoryWaitFlag = 5;
  graph.graphMap[10] = std::move(writeNode);

  auto waitNode = CreateMemoryNode(11, ACL_MODEL_RI_TASK_VALUE_WAIT);
  waitNode->SetNodeType(SkNodeType::NODE_MEMORY_WAIT);
  waitNode->streamIdxInGraph = 0;
  waitNode->streamId = 1;
  waitNode->nodeInfos.syncInfos.eventId = 0x4444;
  waitNode->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x5555);
  waitNode->nodeInfos.syncInfos.memoryValue = 0x6666;
  waitNode->nodeInfos.syncInfos.memoryWaitFlag = 3;
  waitNode->nodeInfos.syncInfos.correspondingNotifyNodeId = 10;
  graph.graphMap[11] = std::move(waitNode);

  graph.streams.push_back(reinterpret_cast<aclrtStream>(0x300));
  graph.headNodes.push_back(10);
  graph.nodeSizeInStream.push_back(2);

  Json json = graph.ToJson();
  EXPECT_EQ(json["totalNodes"], 2);
  EXPECT_EQ(json["totalStreams"], 1);
}

// ==================== Node ToJson Tests ====================

class NodeToJsonTest : public SkDumpJsonTest {};

TEST_F(NodeToJsonTest, KernelNodeToJson) {
  SuperKernelGraph graph(nullptr);
  graph.scopeIdxToName[0] = "(none)";

  auto node = CreateKernelNode(10);
  AddKernelInfoToNode(node.get());
  node->SetNodeType(SkNodeType::NODE_KERNEL);
  node->streamIdxInGraph = 0;
  node->streamId = 5;
  node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  node->nodeInfos.kernelInfos.kernelTypeInt = 1;
  node->nodeInfos.kernelInfos.numBlocks = 32;
  node->nodeInfos.kernelInfos.funcName = "test_kernel_func";
  node->nodeInfos.kernelInfos.vecNum = 8;
  node->nodeInfos.kernelInfos.cubeNum = 16;
  graph.graphMap[10] = std::move(node);

  Json kernelJson = SuperKernelKernelNodeToJson(static_cast<const SuperKernelKernelNode *>(graph.graphMap[10].get()));

  EXPECT_TRUE(kernelJson.contains("taskId"));
  EXPECT_TRUE(kernelJson.contains("streamId"));
  EXPECT_TRUE(kernelJson.contains("taskType"));
  EXPECT_TRUE(kernelJson.contains("taskTypeInt"));
  EXPECT_TRUE(kernelJson.contains("kernelParams"));
}

TEST_F(NodeToJsonTest, MemoryNodeToJson) {
  auto writeNode = CreateMemoryNode(20, ACL_MODEL_RI_TASK_VALUE_WRITE);
  writeNode->SetNodeType(SkNodeType::NODE_MEMORY_WRITE);
  writeNode->nodeInfos.syncInfos.eventId = 0x1234;
  writeNode->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x5678);
  writeNode->nodeInfos.syncInfos.memoryValue = 0x9abc;
  writeNode->nodeInfos.syncInfos.memoryWaitFlag = 7;

  Json memoryJson = SuperKernelMemoryNodeToJson(writeNode.get());

  EXPECT_TRUE(memoryJson.contains("taskId"));
  EXPECT_TRUE(memoryJson.contains("streamId"));
  EXPECT_TRUE(memoryJson.contains("taskType"));
  EXPECT_TRUE(memoryJson.contains("taskTypeInt"));
  EXPECT_TRUE(memoryJson.contains("valueWriteParams"));
  EXPECT_TRUE(memoryJson["valueWriteParams"].contains("devAddr"));
}

TEST_F(NodeToJsonTest, DefaultNodeToJson) {
  auto defaultNode = std::make_unique<SuperKernelDefaultNode>(nullptr, ACL_MODEL_RI_TASK_DEFAULT, 0, 0,
                                                              INVALID_STREAM_ID, INVALID_TASK_ID);
  defaultNode->SetNodeId(30);
  defaultNode->SetNodeType(SkNodeType::NODE_DEFAULT);

  Json defaultJson = SuperKernelDefaultNodeToJson(defaultNode.get());

  EXPECT_TRUE(defaultJson.contains("taskId"));
  EXPECT_TRUE(defaultJson.contains("streamId"));
  EXPECT_TRUE(defaultJson.contains("taskType"));
  EXPECT_TRUE(defaultJson.contains("taskTypeInt"));
}

TEST_F(NodeToJsonTest, NotifyAndWaitNodesToJson) {
  auto notifyNode = CreateMemoryNode(40, ACL_MODEL_RI_TASK_EVENT_RECORD);
  notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
  notifyNode->nodeInfos.syncInfos.eventId = 0x1111;
  notifyNode->nodeInfos.syncInfos.correspondingWaitNodeIds = {41, 42};

  Json notifyJson = SuperKernelMemoryNodeToJson(notifyNode.get());
  EXPECT_TRUE(notifyJson.contains("eventParams"));
  EXPECT_TRUE(notifyJson["eventParams"].contains("eventId"));

  auto waitNode = CreateMemoryNode(41, ACL_MODEL_RI_TASK_EVENT_WAIT);
  waitNode->SetNodeType(SkNodeType::NODE_WAIT);
  waitNode->nodeInfos.syncInfos.eventId = 0x2222;
  waitNode->nodeInfos.syncInfos.correspondingNotifyNodeId = 40;

  Json waitJson = SuperKernelMemoryNodeToJson(waitNode.get());
  EXPECT_TRUE(waitJson.contains("eventParams"));
  EXPECT_TRUE(waitJson["eventParams"].contains("eventId"));
}

TEST_F(NodeToJsonTest, UpdatedNodeToJson) {
  auto kernelNode = CreateKernelNode(50);
  AddKernelInfoToNode(kernelNode.get());
  kernelNode->SetNodeType(SkNodeType::NODE_KERNEL);
  kernelNode->SetUpdate(true);
  kernelNode->taskParams.type = ACL_MODEL_RI_TASK_KERNEL;
  kernelNode->taskParams.kernelTaskParams.numBlocks = 64;
  kernelNode->taskParams.kernelTaskParams.argsSize = 128;
  kernelNode->taskParams.kernelTaskParams.isHostArgs = 1;

  Json updatedJson = SuperKernelKernelNodeToJson(kernelNode.get());
  EXPECT_TRUE(updatedJson.contains("kernelParams"));
  EXPECT_EQ(updatedJson["kernelParams"]["argsSize"], 128);
  EXPECT_EQ(updatedJson["kernelParams"]["isHostArgs"], 1);
}
