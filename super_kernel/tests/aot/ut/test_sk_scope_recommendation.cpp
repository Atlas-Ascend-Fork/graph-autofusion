/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#include <memory>
#include <algorithm>
#include <string>
#include <vector>

#define private public
#define protected public
#include "sk_common.h"
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_scope_recommendation.h"
#include "sk_scope_split.h"

namespace {

aclError FakeGetDeviceCoreNums(int64_t &cubeNum, int64_t &vecNum) {
  cubeNum = 32;
  vecNum = 32;
  return ACL_SUCCESS;
}

class ScopeRecommendationTest : public testing::Test {
 protected:
  void SetUp() override {
    GlobalMockObject::verify();
    MOCKER(GetDeviceCoreNums).stubs().will(invoke(FakeGetDeviceCoreNums));
    savedCube_ = LockDetector::deviceRealCubeNum;
    savedVector_ = LockDetector::deviceRealVecNum;
    LockDetector::deviceRealCubeNum = 32;
    LockDetector::deviceRealVecNum = 32;
  }

  void TearDown() override {
    LockDetector::deviceRealCubeNum = savedCube_;
    LockDetector::deviceRealVecNum = savedVector_;
    GlobalMockObject::verify();
  }

  SuperKernelBaseNode *AddKernel(uint64_t nodeId, uint32_t stream, uint32_t cube, uint32_t vec = 0) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, stream, INVALID_STREAM_ID,
                                                        INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetIsFusible(true);
    node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
    node->nodeInfos.kernelInfos.numBlocks = 1;
    node->nodeInfos.kernelInfos.cubeNum = cube;
    node->nodeInfos.kernelInfos.vecNum = vec;
    auto *result = node.get();
    graph_.graphMap[nodeId] = std::move(node);
    return result;
  }

  SuperKernelBaseNode *AddScopeMarker(uint64_t nodeId, uint32_t stream, bool begin, const std::string &name) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, stream, INVALID_STREAM_ID,
                                                        INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetIsFusible(true);
    node->SetIsScopeNode(true);
    node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
    node->nodeInfos.kernelInfos.numBlocks = 1;
    node->nodeInfos.kernelInfos.cubeNum = 1;
    node->scopeName = name;
    node->isScopeBegin = begin;
    node->isScopeEnd = !begin;
    auto *result = node.get();
    graph_.graphMap[nodeId] = std::move(node);
    return result;
  }

  void SetStream(uint32_t stream, const std::vector<uint64_t> &nodeIds) {
    if (graph_.streams.size() <= stream) {
      graph_.streams.resize(static_cast<size_t>(stream) + 1U);
    }
    if (graph_.headNodes.size() <= stream) {
      graph_.headNodes.resize(static_cast<size_t>(stream) + 1U, INVALID_TASK_ID);
    }
    graph_.headNodes[stream] = nodeIds.front();
    for (size_t index = 0; index < nodeIds.size(); ++index) {
      auto *node = graph_.GetNodeById(nodeIds[index]);
      node->streamIdxInGraph = stream;
      node->nodeIdxInStream = index;
      node->SetPreNodeId(index == 0 ? INVALID_TASK_ID : nodeIds[index - 1]);
      node->SetNextNodeId(index + 1 == nodeIds.size() ? INVALID_TASK_ID : nodeIds[index + 1]);
    }
  }

  SuperKernelGraph graph_;
  int64_t savedCube_ = 0;
  int64_t savedVector_ = 0;
};

std::unique_ptr<SuperKernelOptionsManager> MakeRecommendationOptions(const char *rawValue = nullptr) {
  auto options = std::make_unique<SuperKernelOptionsManager>();
  if (rawValue == nullptr) {
    options->ParseOptions(nullptr);
  } else {
    std::string value = rawValue;
    aclskOption option{};
    option.optionType = aclskOptionType::OPT_EXTEND_OPTION;
    option.optExtend.value = value.data();
    const aclskOptions input{&option, 1};
    options->ParseOptions(&input);
  }
  return options;
}

std::vector<std::vector<uint64_t>> ScopeNodeIds(SuperKernelScopeSplitter &splitter) {
  std::vector<std::vector<uint64_t>> result;
  for (const auto &scope : splitter.GetScopeInfos()) {
    std::vector<uint64_t> nodeIds;
    for (const auto *node : scope.GetNodes()) {
      nodeIds.push_back(node->GetNodeId());
    }
    std::sort(nodeIds.begin(), nodeIds.end());
    result.push_back(std::move(nodeIds));
  }
  std::sort(result.begin(), result.end());
  return result;
}

TEST_F(ScopeRecommendationTest, KeepsWholeStreamSegmentWhenResourcesFit) {
  AddKernel(1, 0, 8, 4);
  AddKernel(2, 0, 4, 8);
  SetStream(0, {1, 2});

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  const uint64_t candidate = recommendation.GetCandidateId(1);
  EXPECT_NE(candidate, ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(2), candidate);
}

TEST_F(ScopeRecommendationTest, UsesOnlyAdjacentEqualCoreRunAfterWholeSegmentExceedsCapacity) {
  AddKernel(1, 0, 16);
  AddKernel(2, 0, 16);
  AddKernel(3, 0, 32);
  AddKernel(10, 1, 16);
  SetStream(0, {1, 2, 3});
  SetStream(1, {10});

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  const uint64_t runCandidate = recommendation.GetCandidateId(1);
  EXPECT_NE(runCandidate, ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(2), runCandidate);
  EXPECT_EQ(recommendation.GetCandidateId(3), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(10), ScopeRecommendation::INVALID_CANDIDATE_ID);
}

TEST_F(ScopeRecommendationTest, KeepsIntermediateNonKernelNodesInsideEqualCoreRun) {
  AddKernel(90, 0, 8);
  AddKernel(5, 0, 0)->SetNodeType(SkNodeType::NODE_DEFAULT);
  AddKernel(70, 0, 8);
  AddKernel(10, 0, 32);
  AddKernel(30, 1, 16);
  SetStream(0, {90, 5, 70, 10});
  SetStream(1, {30});

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  const auto candidate = recommendation.GetCandidateId(90);
  ASSERT_NE(candidate, ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(5), candidate);
  EXPECT_EQ(recommendation.GetCandidateId(70), candidate);
  EXPECT_EQ(recommendation.GetCandidateId(10), ScopeRecommendation::INVALID_CANDIDATE_ID);
}

TEST_F(ScopeRecommendationTest, TakesIndependentResourceMaximaWithinExternalInterval) {
  AddKernel(9999, 0, 8, 8);
  SetStream(0, {9999});
  AddKernel(1000, 1, 24, 0);
  AddKernel(997, 1, 0, 24);
  SetStream(1, {1000, 997});

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_NE(recommendation.GetCandidateId(9999), ScopeRecommendation::INVALID_CANDIDATE_ID);
  graph_.GetNodeById(997)->nodeInfos.kernelInfos.vecNum = 25;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_EQ(recommendation.GetCandidateId(9999), ScopeRecommendation::INVALID_CANDIDATE_ID);
}

TEST_F(ScopeRecommendationTest, EvaluatesSameExternalSetWithDifferentCubeAndVectorBudgets) {
  AddKernel(1, 0, 4, 20);
  AddKernel(2, 0, 4, 20);
  AddKernel(3, 0, 20, 4);
  AddKernel(4, 0, 20, 4);
  AddKernel(10, 1, 16, 8);
  SetStream(0, {1, 2, 3, 4});
  SetStream(1, {10});

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  ASSERT_NE(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(1), recommendation.GetCandidateId(2));
  EXPECT_EQ(recommendation.GetCandidateId(3), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(4), ScopeRecommendation::INVALID_CANDIDATE_ID);
}

struct RecommendationOptionCase {
  const char *name;
  const char *value;
  bool enabled;
};

class ScopeRecommendationOptionTest : public ScopeRecommendationTest,
                                      public testing::WithParamInterface<RecommendationOptionCase> {};

TEST_P(ScopeRecommendationOptionTest, ParsesOptionAndRestrictsInitialScopesOnlyWhenEnabled) {
  AddKernel(1, 0, 16);
  AddKernel(2, 0, 16);
  AddKernel(3, 0, 32);
  AddKernel(10, 1, 16);
  SetStream(0, {1, 2, 3});
  SetStream(1, {10});

  auto options = MakeRecommendationOptions(GetParam().value);
  SuperKernelScopeSplitter splitter(graph_, *options);
  ASSERT_TRUE(splitter.SplitGraph());
  const std::vector<std::vector<uint64_t>> expected = GetParam().enabled
                                                          ? std::vector<std::vector<uint64_t>>{{1, 2}}
                                                          : std::vector<std::vector<uint64_t>>{{1, 2, 3, 10}};
  EXPECT_EQ(ScopeNodeIds(splitter), expected);
  // Recommendation filters membership without changing the existing fusibility state.
  for (const uint64_t nodeId : {1U, 2U, 3U, 10U}) {
    EXPECT_TRUE(graph_.GetNodeById(nodeId)->IsFusible());
  }
}

INSTANTIATE_TEST_SUITE_P(OptionValues, ScopeRecommendationOptionTest,
                         testing::Values(RecommendationOptionCase{"Absent", nullptr, false},
                                         RecommendationOptionCase{"Empty", "", false},
                                         RecommendationOptionCase{"Unrelated", "other_option=1", false},
                                         RecommendationOptionCase{"Zero", "recommended_fusion_scopes=0", false},
                                         RecommendationOptionCase{"False", "recommended_fusion_scopes=false", false},
                                         RecommendationOptionCase{"One", "recommended_fusion_scopes=1", true},
                                         RecommendationOptionCase{"True", "recommended_fusion_scopes=true", true}),
                         [](const testing::TestParamInfo<RecommendationOptionCase> &info) { return info.param.name; });

TEST_F(ScopeRecommendationTest, KeepsAdjacentCandidatesSeparateInSplitter) {
  AddKernel(1, 0, 8);
  AddKernel(2, 0, 8);
  AddKernel(3, 0, 16);
  AddKernel(4, 0, 16);
  AddKernel(5, 0, 32);
  AddKernel(10, 1, 16);
  SetStream(0, {1, 2, 3, 4, 5});
  SetStream(1, {10});

  auto options = MakeRecommendationOptions("recommended_fusion_scopes=1");
  SuperKernelScopeSplitter splitter(graph_, *options);
  ASSERT_TRUE(splitter.SplitGraph());
  EXPECT_EQ(ScopeNodeIds(splitter), (std::vector<std::vector<uint64_t>>{{1, 2}, {3, 4}}));
}

TEST_F(ScopeRecommendationTest, RecommendedCandidateDoesNotOverrideUnfusibleKernel) {
  AddKernel(1, 0, 8);
  auto *unfusible = AddKernel(2, 0, 8);
  unfusible->SetIsFusible(false);
  AddKernel(3, 0, 8);
  SetStream(0, {1, 2, 3});

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  ASSERT_NE(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(1), recommendation.GetCandidateId(2));
  EXPECT_FALSE(unfusible->IsFusible());

  auto options = MakeRecommendationOptions("recommended_fusion_scopes=1");
  SuperKernelScopeSplitter splitter(graph_, *options);
  ASSERT_TRUE(splitter.SplitGraph());
  EXPECT_EQ(ScopeNodeIds(splitter), (std::vector<std::vector<uint64_t>>{{1}, {3}}));
  EXPECT_FALSE(unfusible->IsFusible());
}

TEST_F(ScopeRecommendationTest, PerOpMaxCoreModeIgnoresRecommendationOption) {
  AddKernel(1, 0, 16);
  AddKernel(2, 0, 16);
  AddKernel(3, 0, 32);
  AddKernel(10, 1, 16);
  SetStream(0, {1, 2, 3});
  SetStream(1, {10});

  auto options = MakeRecommendationOptions("recommended_fusion_scopes=1");
  options->GetOption(aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM)->SetValue(1U);
  SuperKernelScopeSplitter splitter(graph_, *options);
  ASSERT_TRUE(splitter.SplitGraph());
  EXPECT_EQ(ScopeNodeIds(splitter), (std::vector<std::vector<uint64_t>>{{1}, {2}, {3}, {10}}));
  for (const auto &scope : splitter.GetScopeInfos()) {
    EXPECT_EQ(scope.GetScopeCoreInfo().type, SkKernelType::MIX_AIC_1_2);
    EXPECT_EQ(scope.GetScopeCoreInfo().numBlocks, 16U);
  }
}

TEST_F(ScopeRecommendationTest, LimitsRecommendationToMarkedRangeIncludingPlaceholder) {
  AddKernel(1, 0, 4);
  AddScopeMarker(2, 0, true, "user_scope");
  AddKernel(3, 0, 8);
  AddKernel(4, 0, 8);
  auto *placeholder = static_cast<SuperKernelKernelNode *>(AddKernel(40, 0, 1));
  placeholder->SetIsScopeNode(true);
  placeholder->isPlaceholder = true;
  AddScopeMarker(50, 0, false, "user_scope");
  AddKernel(60, 0, 4);
  SetStream(0, {1, 2, 3, 4, 40, 50, 60});
  graph_.scopeNameToIdx["user_scope"] = 0;
  ASSERT_TRUE(graph_.UpdateNodeScopeBitFlags());

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_EQ(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_NE(recommendation.GetCandidateId(2), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(2), recommendation.GetCandidateId(3));
  EXPECT_EQ(recommendation.GetCandidateId(3), recommendation.GetCandidateId(4));
  EXPECT_EQ(recommendation.GetCandidateId(4), recommendation.GetCandidateId(40));
  EXPECT_EQ(recommendation.GetCandidateId(40), recommendation.GetCandidateId(50));
  EXPECT_EQ(recommendation.GetCandidateId(60), ScopeRecommendation::INVALID_CANDIDATE_ID);

  auto enabledOptions = MakeRecommendationOptions("recommended_fusion_scopes=1");
  SuperKernelScopeSplitter splitter(graph_, *enabledOptions);
  ASSERT_TRUE(splitter.SplitGraph());
  EXPECT_EQ(ScopeNodeIds(splitter), (std::vector<std::vector<uint64_t>>{{2, 3, 4, 40, 50}}));
}

TEST_F(ScopeRecommendationTest, KeepsNestedScopeFlagRangesSeparate) {
  AddScopeMarker(1, 0, true, "outer");
  AddKernel(2, 0, 4);
  AddScopeMarker(3, 0, true, "inner");
  AddKernel(4, 0, 4);
  AddScopeMarker(5, 0, false, "inner");
  AddKernel(6, 0, 4);
  AddScopeMarker(7, 0, false, "outer");
  SetStream(0, {1, 2, 3, 4, 5, 6, 7});
  graph_.scopeNameToIdx["outer"] = 0;
  graph_.scopeNameToIdx["inner"] = 1;
  ASSERT_TRUE(graph_.UpdateNodeScopeBitFlags());
  const auto innerFlags = graph_.GetNodeById(4)->GetScopeBitFlags();
  ASSERT_EQ(innerFlags.count(), 2U);

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  ASSERT_NE(recommendation.GetCandidateId(2), ScopeRecommendation::INVALID_CANDIDATE_ID);
  ASSERT_NE(recommendation.GetCandidateId(4), ScopeRecommendation::INVALID_CANDIDATE_ID);
  ASSERT_NE(recommendation.GetCandidateId(6), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_NE(recommendation.GetCandidateId(2), recommendation.GetCandidateId(4));
  EXPECT_NE(recommendation.GetCandidateId(4), recommendation.GetCandidateId(6));
  EXPECT_NE(recommendation.GetCandidateId(2), recommendation.GetCandidateId(6));
  EXPECT_EQ(graph_.GetNodeById(4)->GetScopeBitFlags(), innerFlags);

  auto options = MakeRecommendationOptions("recommended_fusion_scopes=1");
  SuperKernelScopeSplitter splitter(graph_, *options);
  ASSERT_TRUE(splitter.SplitGraph());
  EXPECT_EQ(ScopeNodeIds(splitter), (std::vector<std::vector<uint64_t>>{{1, 2}, {3, 4, 5}, {6, 7}}));
  EXPECT_EQ(graph_.GetNodeById(4)->GetScopeBitFlags(), innerFlags);
}

TEST_F(ScopeRecommendationTest, KeepsRepeatedSameNameScopesSeparate) {
  AddScopeMarker(1, 0, true, "user_scope");
  AddKernel(2, 0, 4);
  AddScopeMarker(3, 0, false, "user_scope");
  AddKernel(4, 0, 4);
  AddScopeMarker(5, 0, true, "user_scope");
  AddKernel(6, 0, 4);
  AddScopeMarker(7, 0, false, "user_scope");
  SetStream(0, {1, 2, 3, 4, 5, 6, 7});
  graph_.scopeNameToIdx["user_scope"] = 0;
  ASSERT_TRUE(graph_.UpdateNodeScopeBitFlags());
  ASSERT_EQ(graph_.GetNodeById(2)->GetScopeBitFlags(), graph_.GetNodeById(6)->GetScopeBitFlags());

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  ASSERT_NE(recommendation.GetCandidateId(2), ScopeRecommendation::INVALID_CANDIDATE_ID);
  ASSERT_NE(recommendation.GetCandidateId(6), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_NE(recommendation.GetCandidateId(2), recommendation.GetCandidateId(6));
  EXPECT_EQ(recommendation.GetCandidateId(4), ScopeRecommendation::INVALID_CANDIDATE_ID);

  auto options = MakeRecommendationOptions("recommended_fusion_scopes=1");
  SuperKernelScopeSplitter splitter(graph_, *options);
  ASSERT_TRUE(splitter.SplitGraph());
  EXPECT_EQ(ScopeNodeIds(splitter), (std::vector<std::vector<uint64_t>>{{1, 2, 3}, {5, 6, 7}}));
}

TEST_F(ScopeRecommendationTest, CountsExternalResourcesAndRecommendsEqualCoreRunOnlyInsideUserScope) {
  AddKernel(1, 0, 32);
  AddScopeMarker(2, 0, true, "user_scope");
  AddKernel(3, 0, 8);
  AddKernel(4, 0, 8);
  AddKernel(5, 0, 32);
  AddScopeMarker(6, 0, false, "user_scope");
  AddKernel(7, 0, 32);
  AddKernel(10, 1, 16);
  SetStream(0, {1, 2, 3, 4, 5, 6, 7});
  SetStream(1, {10});
  graph_.scopeNameToIdx["user_scope"] = 0;
  ASSERT_TRUE(graph_.UpdateNodeScopeBitFlags());
  ASSERT_FALSE(graph_.GetNodeById(10)->IsFusible());

  // Even an unfusible out-of-scope kernel consumes resources: 32 + 16 > 32.
  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  ASSERT_NE(recommendation.GetCandidateId(3), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(3), recommendation.GetCandidateId(4));
  for (const uint64_t nodeId : {1U, 2U, 5U, 6U, 7U, 10U}) {
    EXPECT_EQ(recommendation.GetCandidateId(nodeId), ScopeRecommendation::INVALID_CANDIDATE_ID);
  }
  EXPECT_TRUE(graph_.GetNodeById(5)->IsFusible());
  EXPECT_FALSE(graph_.GetNodeById(10)->IsFusible());

  auto options = MakeRecommendationOptions("recommended_fusion_scopes=1");
  SuperKernelScopeSplitter splitter(graph_, *options);
  ASSERT_TRUE(splitter.SplitGraph());
  EXPECT_EQ(ScopeNodeIds(splitter), (std::vector<std::vector<uint64_t>>{{3, 4}}));
}

class ScopeRecommendationSyncTest : public ScopeRecommendationTest {
 protected:
  SuperKernelBaseNode *AddSync(uint64_t nodeId, uint32_t stream, bool signal) {
    const auto taskType = signal ? ACL_MODEL_RI_TASK_EVENT_RECORD : ACL_MODEL_RI_TASK_EVENT_WAIT;
    auto node =
        std::make_unique<SuperKernelMemoryNode>(nullptr, taskType, 0, stream, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNodeType(signal ? SkNodeType::NODE_NOTIFY : SkNodeType::NODE_WAIT);
    auto *result = node.get();
    graph_.graphMap[nodeId] = std::move(node);
    return result;
  }

  void Link(uint64_t signalId, uint64_t waitId) {
    graph_.GetNodeById(signalId)->SetCorrespondingWaitNodeIds({waitId});
    graph_.GetNodeById(waitId)->SetCorrespondingNotifyNodeId(signalId);
  }

  void Unlink(uint64_t signalId, uint64_t waitId) {
    graph_.GetNodeById(signalId)->SetCorrespondingWaitNodeIds({});
    graph_.GetNodeById(waitId)->SetCorrespondingNotifyNodeId(INVALID_TASK_ID);
  }
};

TEST_F(ScopeRecommendationSyncTest, AssignsCandidateIdsByStreamOrderRegardlessOfHeadTraversalOrder) {
  AddKernel(90, 0, 4, 4);
  AddSync(80, 0, true);
  AddKernel(70, 0, 4, 4);
  AddSync(10, 2, false);
  AddKernel(30, 2, 4, 4);
  AddKernel(5, 5, 4, 4);
  AddKernel(3, 5, 4, 4);
  SetStream(0, {90, 80, 70});
  SetStream(2, {10, 30});
  SetStream(5, {5, 3});
  Link(80, 10);

  std::vector<uint64_t> heads{5, 10, 90};
  do {
    graph_.headNodes = heads;
    ScopeRecommendation recommendation;
    ASSERT_TRUE(recommendation.Analyze(graph_));
    EXPECT_EQ(recommendation.GetCandidateId(90), 0U);
    EXPECT_EQ(recommendation.GetCandidateId(70), 1U);
    EXPECT_EQ(recommendation.GetCandidateId(30), 2U);
    EXPECT_EQ(recommendation.GetCandidateId(5), 3U);
    EXPECT_EQ(recommendation.GetCandidateId(3), 3U);
    EXPECT_EQ(recommendation.GetCandidateId(80), ScopeRecommendation::INVALID_CANDIDATE_ID);
    EXPECT_EQ(recommendation.GetCandidateId(10), ScopeRecommendation::INVALID_CANDIDATE_ID);
  } while (std::next_permutation(heads.begin(), heads.end()));
}

TEST_F(ScopeRecommendationSyncTest, ExcludesOrderedKernelFromParallelResources) {
  // stream 0: K1(32/32) -> signal2
  // stream 1: wait10 -> K11(32/32), with signal2 -> wait10.
  AddKernel(1, 0, 32, 32);
  AddSync(2, 0, true);
  AddSync(10, 1, false);
  AddKernel(11, 1, 32, 32);
  SetStream(0, {1, 2});
  SetStream(1, {10, 11});
  Link(2, 10);

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_NE(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_NE(recommendation.GetCandidateId(11), ScopeRecommendation::INVALID_CANDIDATE_ID);

  // Without the dependency, 32 + 32 exceeds both capacities.
  Unlink(2, 10);
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_EQ(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(11), ScopeRecommendation::INVALID_CANDIDATE_ID);
}

TEST_F(ScopeRecommendationSyncTest, SplitsCandidatesAtBothSynchronizationBoundaries) {
  // Each stream has kernels on both sides of its synchronization node.
  AddKernel(1, 0, 8, 8);
  AddSync(2, 0, true);
  AddKernel(3, 0, 8, 8);
  AddKernel(10, 1, 8, 8);
  AddSync(11, 1, false);
  AddKernel(12, 1, 8, 8);
  SetStream(0, {1, 2, 3});
  SetStream(1, {10, 11, 12});
  Link(2, 11);

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  for (const uint64_t kernelId : {1U, 3U, 10U, 12U}) {
    EXPECT_NE(recommendation.GetCandidateId(kernelId), ScopeRecommendation::INVALID_CANDIDATE_ID);
  }
  EXPECT_NE(recommendation.GetCandidateId(1), recommendation.GetCandidateId(3));
  EXPECT_NE(recommendation.GetCandidateId(10), recommendation.GetCandidateId(12));
  EXPECT_EQ(recommendation.GetCandidateId(2), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(11), ScopeRecommendation::INVALID_CANDIDATE_ID);
}

TEST_F(ScopeRecommendationSyncTest, TakesIndependentCubeAndVectorMaximaOfOrderedExternalStreams) {
  // Target K1-K2 overlaps K10 and K20, but K10 -> K20.
  // Total C/V = (8, 8) + max((24, 12), (12, 24)) = (32, 32).
  AddKernel(1, 0, 8, 8);
  AddKernel(2, 0, 4, 4);
  AddKernel(10, 1, 24, 12);
  AddSync(11, 1, true);
  AddSync(19, 2, false);
  AddKernel(20, 2, 12, 24);
  SetStream(0, {1, 2});
  SetStream(1, {10, 11});
  SetStream(2, {19, 20});
  Link(11, 19);

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  ASSERT_NE(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(1), recommendation.GetCandidateId(2));

  // Without synchronization, total C/V = (44, 44).
  Unlink(11, 19);
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_EQ(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(2), ScopeRecommendation::INVALID_CANDIDATE_ID);
}

TEST_F(ScopeRecommendationSyncTest, FollowsTransitiveDependenciesAcrossThreeExternalStreams) {
  // Target K1 overlaps the external chain K10 -> K20 -> K30.
  AddKernel(1, 0, 8, 8);
  AddKernel(10, 1, 24, 12);
  AddSync(11, 1, true);
  AddSync(19, 2, false);
  AddKernel(20, 2, 20, 16);
  AddSync(21, 2, true);
  AddSync(29, 3, false);
  AddKernel(30, 3, 12, 24);
  SetStream(0, {1});
  SetStream(1, {10, 11});
  SetStream(2, {19, 20, 21});
  SetStream(3, {29, 30});
  Link(11, 19);
  Link(21, 29);

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_NE(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);

  // K30 becomes independent of K10/K20 when the second link is removed.
  Unlink(21, 29);
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_EQ(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);
}

TEST_F(ScopeRecommendationSyncTest, SumsIndependentSuccessorsOfOneSignal) {
  // External fork: K10 -> {K20, K30}. K20 and K30 may run together.
  AddKernel(1, 0, 8, 8);
  AddKernel(10, 1, 24, 16);
  AddSync(11, 1, true);
  AddSync(19, 2, false);
  AddKernel(20, 2, 16, 8);
  AddSync(29, 3, false);
  auto *branchKernel = AddKernel(30, 3, 12, 16);
  SetStream(0, {1});
  SetStream(1, {10, 11});
  SetStream(2, {19, 20});
  SetStream(3, {29, 30});
  Link(11, 19);
  graph_.GetNodeById(11)->SetCorrespondingWaitNodeIds({19, 29});
  graph_.GetNodeById(29)->SetCorrespondingNotifyNodeId(11);

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  // C = 8 + max(24, 16 + 12) = 36: do not treat the branches as ordered.
  EXPECT_EQ(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);

  branchKernel->nodeInfos.kernelInfos.cubeNum = 8;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  // C/V = (8, 8) + max((24, 16), (24, 24)) = (32, 32).
  EXPECT_NE(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);
}

TEST_F(ScopeRecommendationSyncTest, PreservesDependencyThroughZeroWeightKernelInResourceProfile) {
  AddKernel(1, 0, 4, 4);
  AddKernel(10, 1, 20, 12);
  AddSync(11, 1, true);
  AddSync(19, 2, false);
  AddKernel(20, 2, 0, 0);
  AddSync(21, 2, true);
  AddSync(29, 3, false);
  AddKernel(30, 3, 12, 20);
  auto *independent = AddKernel(40, 4, 8, 8);
  SetStream(0, {1});
  SetStream(1, {10, 11});
  SetStream(2, {19, 20, 21});
  SetStream(3, {29, 30});
  SetStream(4, {40});
  Link(11, 19);
  Link(21, 29);

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_NE(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);
  independent->nodeInfos.kernelInfos.cubeNum = 9;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_EQ(recommendation.GetCandidateId(1), ScopeRecommendation::INVALID_CANDIDATE_ID);
}

TEST_F(ScopeRecommendationSyncTest, ReusesExactProfilesWithoutTreatingBoundsAsExactWeights) {
  AddKernel(1, 0, 16, 4);
  AddKernel(2, 0, 16, 4);
  AddKernel(3, 0, 8, 16);
  AddKernel(4, 0, 8, 16);
  AddKernel(5, 0, 8, 8);
  AddKernel(6, 0, 8, 8);
  AddKernel(10, 1, 20, 12);
  AddSync(11, 1, true);
  AddSync(19, 2, false);
  AddKernel(20, 2, 12, 20);
  SetStream(0, {1, 2, 3, 4, 5, 6});
  SetStream(1, {10, 11});
  SetStream(2, {19, 20});
  Link(11, 19);

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  for (const uint64_t nodeId : {1U, 2U, 3U, 4U}) {
    EXPECT_EQ(recommendation.GetCandidateId(nodeId), ScopeRecommendation::INVALID_CANDIDATE_ID);
  }
  ASSERT_NE(recommendation.GetCandidateId(5), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(5), recommendation.GetCandidateId(6));
}

TEST_F(ScopeRecommendationSyncTest, PreservesDependenciesAcrossMoreThan64Intervals) {
  AddKernel(99999, 0, 8, 8);
  SetStream(0, {99999});
  for (uint32_t stream = 1; stream <= 30; ++stream) {
    const uint64_t base = 10000 - stream * 100;
    AddSync(base, stream, false);
    AddKernel(base + 1, stream, 24, 0);
    AddKernel(base + 2, stream, 0, 24);
    AddSync(base + 3, stream, true);
    SetStream(stream, {base, base + 1, base + 2, base + 3});
    if (stream > 1) {
      Link(base + 103, base);
    }
  }

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_NE(recommendation.GetCandidateId(99999), ScopeRecommendation::INVALID_CANDIDATE_ID);

  Unlink(8503, 8400);
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_EQ(recommendation.GetCandidateId(99999), ScopeRecommendation::INVALID_CANDIDATE_ID);
}

TEST_F(ScopeRecommendationSyncTest, KeepsOutOfScopeSynchronizationOnlyPath) {
  AddScopeMarker(1, 0, true, "user_scope");
  AddKernel(2, 0, 8, 8);
  AddScopeMarker(3, 0, false, "user_scope");
  AddKernel(10, 1, 24, 12);
  AddSync(11, 1, true);
  AddSync(19, 2, false);
  AddKernel(20, 2, 0)->SetNodeType(SkNodeType::NODE_DEFAULT);
  AddSync(21, 2, true);
  AddSync(29, 3, false);
  AddKernel(30, 3, 12, 24);
  SetStream(0, {1, 2, 3});
  SetStream(1, {10, 11});
  SetStream(2, {19, 20, 21});
  SetStream(3, {29, 30});
  Link(11, 19);
  Link(21, 29);
  graph_.scopeNameToIdx["user_scope"] = 0;
  ASSERT_TRUE(graph_.UpdateNodeScopeBitFlags());

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_NE(recommendation.GetCandidateId(2), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(10), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(30), ScopeRecommendation::INVALID_CANDIDATE_ID);

  Unlink(21, 29);
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_EQ(recommendation.GetCandidateId(2), ScopeRecommendation::INVALID_CANDIDATE_ID);
}

TEST_F(ScopeRecommendationSyncTest, UsesEachScopeBudgetWithinOneDependencyInterval) {
  AddScopeMarker(1, 0, true, "user_scope");
  AddKernel(2, 0, 8, 8);
  AddKernel(3, 0, 16, 4);
  AddKernel(4, 0, 4, 16);
  AddKernel(5, 0, 32, 32);
  for (const uint64_t nodeId : {1U, 2U, 3U, 4U}) {
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(nodeId <= 2 ? 0 : nodeId - 2);
    graph_.GetNodeById(nodeId)->SetScopeBitFlags(flags);
  }
  AddKernel(10, 1, 20, 12);
  AddSync(11, 1, true);
  AddSync(19, 2, false);
  AddKernel(20, 2, 12, 20);
  SetStream(0, {1, 2, 3, 4, 5});
  SetStream(1, {10, 11});
  SetStream(2, {19, 20});
  Link(11, 19);

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  ASSERT_NE(recommendation.GetCandidateId(2), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_EQ(recommendation.GetCandidateId(1), recommendation.GetCandidateId(2));
  for (const uint64_t nodeId : {3U, 4U, 5U, 10U, 20U}) {
    EXPECT_EQ(recommendation.GetCandidateId(nodeId), ScopeRecommendation::INVALID_CANDIDATE_ID);
  }
}

TEST_F(ScopeRecommendationSyncTest, KeepsSameStreamWaitGroupInOneCandidate) {
  AddKernel(1, 0, 8, 8);
  AddSync(2, 0, true);
  AddKernel(3, 0, 8, 8);
  AddSync(4, 0, false);
  AddKernel(5, 0, 8, 8);
  AddSync(6, 0, false);
  AddKernel(7, 0, 8, 8);
  SetStream(0, {1, 2, 3, 4, 5, 6, 7});
  graph_.GetNodeById(2)->SetCorrespondingWaitNodeIds({4, 6});
  graph_.GetNodeById(4)->SetCorrespondingNotifyNodeId(2);
  graph_.GetNodeById(6)->SetCorrespondingNotifyNodeId(2);

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  const auto candidate = recommendation.GetCandidateId(1);
  ASSERT_NE(candidate, ScopeRecommendation::INVALID_CANDIDATE_ID);
  for (const uint64_t nodeId : {2U, 3U, 4U, 5U, 6U, 7U}) {
    EXPECT_EQ(recommendation.GetCandidateId(nodeId), candidate);
  }
}

TEST_F(ScopeRecommendationSyncTest, KeepsMixedStreamWaitGroupAsBoundaries) {
  AddKernel(1, 0, 8, 8);
  AddSync(2, 0, true);
  AddKernel(3, 0, 8, 8);
  AddSync(4, 0, false);
  AddKernel(5, 0, 8, 8);
  AddSync(10, 1, false);
  AddKernel(11, 1, 8, 8);
  SetStream(0, {1, 2, 3, 4, 5});
  SetStream(1, {10, 11});
  graph_.GetNodeById(2)->SetCorrespondingWaitNodeIds({4, 10});
  graph_.GetNodeById(4)->SetCorrespondingNotifyNodeId(2);
  graph_.GetNodeById(10)->SetCorrespondingNotifyNodeId(2);

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_NE(recommendation.GetCandidateId(1), recommendation.GetCandidateId(3));
  EXPECT_NE(recommendation.GetCandidateId(3), recommendation.GetCandidateId(5));
  for (const uint64_t nodeId : {2U, 4U, 10U}) {
    EXPECT_EQ(recommendation.GetCandidateId(nodeId), ScopeRecommendation::INVALID_CANDIDATE_ID);
  }
}

TEST_F(ScopeRecommendationSyncTest, KeepsUnpairedSynchronizationAsBoundaries) {
  AddKernel(1, 0, 8, 8);
  AddSync(2, 0, true);
  AddKernel(3, 0, 8, 8);
  AddSync(4, 0, false);
  AddKernel(5, 0, 8, 8);
  SetStream(0, {1, 2, 3, 4, 5});

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_NE(recommendation.GetCandidateId(1), recommendation.GetCandidateId(3));
  EXPECT_NE(recommendation.GetCandidateId(3), recommendation.GetCandidateId(5));
  for (const uint64_t nodeId : {2U, 4U}) {
    EXPECT_EQ(recommendation.GetCandidateId(nodeId), ScopeRecommendation::INVALID_CANDIDATE_ID);
  }
}

class ScopeRecommendationMemorySyncTest : public ScopeRecommendationTest {
 protected:
  void SetUp() override {
    ScopeRecommendationTest::SetUp();
    options_ = MakeRecommendationOptions("recommended_fusion_scopes=1");
    aclskOption option{};
    option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
    option.aggressiveOpts.valueBreakerBypass = ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT;
    options_->SetOptOptionValue(&option);
    graph_.opts_ = options_.get();
  }

  SuperKernelBaseNode *AddMemorySync(uint64_t nodeId, uint32_t stream, bool signal, uint64_t value) {
    const auto taskType = signal ? ACL_MODEL_RI_TASK_VALUE_WRITE : ACL_MODEL_RI_TASK_VALUE_WAIT;
    auto node =
        std::make_unique<SuperKernelMemoryNode>(nullptr, taskType, 0, stream, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNodeType(signal ? SkNodeType::NODE_MEMORY_WRITE : SkNodeType::NODE_MEMORY_WAIT);
    node->nodeInfos.syncInfos.eventId = reinterpret_cast<uintptr_t>(&syncValue_);
    node->nodeInfos.syncInfos.addrValue = &syncValue_;
    node->nodeInfos.syncInfos.memoryValue = value;
    node->nodeInfos.syncInfos.memoryWaitFlag = static_cast<uint32_t>(SkMemoryWaitFlag::EQ);
    auto *result = node.get();
    EXPECT_TRUE(graph_.AddNode(std::move(node)));
    return result;
  }

  uint64_t syncValue_ = 0;
  std::unique_ptr<SuperKernelOptionsManager> options_;
};

TEST_F(ScopeRecommendationMemorySyncTest, KeepsConvertedSameStreamPairInOneCandidateAndScope) {
  AddKernel(1, 0, 8, 8);
  auto *write = AddMemorySync(2, 0, true, 1);
  AddKernel(3, 0, 8, 8);
  auto *wait = AddMemorySync(4, 0, false, 1);
  AddKernel(5, 0, 8, 8);
  SetStream(0, {1, 2, 3, 4, 5});
  ASSERT_TRUE(graph_.PostProcessMemoryNode());
  graph_.BuildEventNodeAssociations();
  ASSERT_EQ(write->GetNodeType(), SkNodeType::NODE_NOTIFY);
  ASSERT_EQ(wait->GetNodeType(), SkNodeType::NODE_WAIT);
  ASSERT_TRUE(write->IsFusible());
  ASSERT_TRUE(wait->IsFusible());

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  const auto candidate = recommendation.GetCandidateId(1);
  ASSERT_NE(candidate, ScopeRecommendation::INVALID_CANDIDATE_ID);
  for (const uint64_t nodeId : {2U, 3U, 4U, 5U}) {
    EXPECT_EQ(recommendation.GetCandidateId(nodeId), candidate);
  }
  SuperKernelScopeSplitter splitter(graph_, *options_);
  ASSERT_TRUE(splitter.SplitGraph());
  EXPECT_EQ(ScopeNodeIds(splitter), (std::vector<std::vector<uint64_t>>{{1, 2, 3, 4, 5}}));
}

TEST_F(ScopeRecommendationMemorySyncTest, KeepsConvertedCrossStreamPairAsBoundaries) {
  AddKernel(1, 0, 8, 8);
  auto *write = AddMemorySync(2, 0, true, 1);
  AddKernel(3, 0, 8, 8);
  auto *wait = AddMemorySync(10, 1, false, 1);
  AddKernel(11, 1, 8, 8);
  SetStream(0, {1, 2, 3});
  SetStream(1, {10, 11});
  ASSERT_TRUE(graph_.PostProcessMemoryNode());
  graph_.BuildEventNodeAssociations();
  ASSERT_TRUE(write->IsFusible());
  ASSERT_TRUE(wait->IsFusible());

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_NE(recommendation.GetCandidateId(1), recommendation.GetCandidateId(3));
  EXPECT_NE(recommendation.GetCandidateId(3), recommendation.GetCandidateId(11));
  for (const uint64_t nodeId : {2U, 10U}) {
    EXPECT_EQ(recommendation.GetCandidateId(nodeId), ScopeRecommendation::INVALID_CANDIDATE_ID);
  }
}

TEST_F(ScopeRecommendationMemorySyncTest, KeepsUnpairedWaitAndResetBehaviorAfterConversion) {
  AddKernel(1, 0, 8, 8);
  auto *wait = AddMemorySync(2, 0, false, 1);
  AddKernel(3, 0, 8, 8);
  auto *reset = AddMemorySync(4, 0, true, 0);
  AddKernel(5, 0, 8, 8);
  SetStream(0, {1, 2, 3, 4, 5});
  ASSERT_TRUE(graph_.PostProcessMemoryNode());
  graph_.BuildEventNodeAssociations();
  ASSERT_EQ(wait->GetNodeType(), SkNodeType::NODE_WAIT);
  ASSERT_EQ(reset->GetNodeType(), SkNodeType::NODE_RESET);
  ASSERT_FALSE(wait->IsFusible());
  ASSERT_FALSE(reset->IsFusible());

  ScopeRecommendation recommendation;
  ASSERT_TRUE(recommendation.Analyze(graph_));
  EXPECT_EQ(recommendation.GetCandidateId(2), ScopeRecommendation::INVALID_CANDIDATE_ID);
  EXPECT_NE(recommendation.GetCandidateId(1), recommendation.GetCandidateId(3));
  EXPECT_EQ(recommendation.GetCandidateId(3), recommendation.GetCandidateId(4));
  EXPECT_EQ(recommendation.GetCandidateId(3), recommendation.GetCandidateId(5));
  SuperKernelScopeSplitter splitter(graph_, *options_);
  ASSERT_TRUE(splitter.SplitGraph());
  EXPECT_EQ(ScopeNodeIds(splitter), (std::vector<std::vector<uint64_t>>{{1}, {3}, {5}}));
}

}  // namespace
