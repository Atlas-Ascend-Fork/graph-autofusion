/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_map>
#include <vector>

#include "sk_scope_verify.h"
#include "sk_graph.h"
#include "sk_lock_detector.h"
#include "sk_log.h"
#include "sk_options_manager.h"
#include "sk_verify_graph.h"
#include "sk_verify_scope_split.h"

namespace {

const char *ScopeVerifySplitTypeToString(aclskScopeVerifySplitType splitType) {
  switch (splitType) {
    case ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE:
      return "ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE";
    case ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE:
      return "ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE";
    default:
      return "UNKNOWN";
  }
}

const char *ScopeVerifySplitReasonToString(aclskScopeVerifySplitReason splitReason) {
  switch (splitReason) {
    case ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED:
      return "ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED";
    case ACLSK_SCOPE_VERIFY_SYNCALL_OP_DROP:
      return "ACLSK_SCOPE_VERIFY_SYNCALL_OP_DROP";
    default:
      return "UNKNOWN";
  }
}

class ScopeVerifyRunner {
 public:
  ScopeVerifyRunner(const aclskScopeVerifyGraphInfo *verifyGraph, size_t maxSplitResultCount,
                    aclskScopeVerifySplitResult *splitResults, size_t *realSplitResultCount)
      : verifyGraph_(verifyGraph),
        maxSplitResultCount_(maxSplitResultCount),
        splitResults_(splitResults),
        realSplitResultCount_(realSplitResultCount) {}

  aclError Run() {
    // realSplitResultCount is always reset before running verification, so callers can rely on it after failure.
    if (realSplitResultCount_ == nullptr) {
      SK_DLOGE("aclskScopeVerify realSplitResultCount is null");
      return ACL_ERROR_INVALID_PARAM;
    }
    *realSplitResultCount_ = 0;
    if (maxSplitResultCount_ > 0 && splitResults_ == nullptr) {
      SK_DLOGE("aclskScopeVerify splitResults is null, maxSplitResultCount=%zu", maxSplitResultCount_);
      return ACL_ERROR_INVALID_PARAM;
    }
    splitRecords_.clear();

    // Adapt GE scope verify graph to the internal SuperKernelGraph and initial scope groups.
    ScopeVerifyGraphBuilder builder;
    aclError ret = builder.Build(verifyGraph_, graph_, scopes_, verifyNodesById_);
    if (ret != ACL_SUCCESS) {
      SK_DLOGE("aclskScopeVerify ScopeVerifyGraphBuilder failed, ret=%d", ret);
      return ret;
    }

    // Match the original SuperKernel scope refinement order: run full-core ScheMode first,
    // then deadlock on refined scopes.
    if (!RunScopeVerifyScheModeSplitPass(graph_, scopes_, splitRecords_)) {
      SK_DLOGE("aclskScopeVerify ScheModeKernelSplitPass failed");
      return ACL_ERROR_FAILURE;
    }

    ret = LockDetector::GetDeviceCores();
    if (ret != ACL_SUCCESS) {
      SK_DLOGE("aclskScopeVerify GetDeviceCores failed, ret=%d", ret);
      return ret;
    }

    SuperKernelOptionsManager opts;
    if (!RunScopeVerifyDeadlockRefinePass(graph_, opts, scopes_, splitRecords_)) {
      SK_DLOGE("aclskScopeVerify DeadlockRefinePass failed");
      return ACL_ERROR_FAILURE;
    }

    // Map internal split records back to caller-owned aclskScopeVerifyNodeInfo pointers.
    return FillScopeVerifySplitResults();
  }

 private:
  aclError FillScopeVerifySplitResults() {
    *realSplitResultCount_ = splitRecords_.size();
    if (splitRecords_.size() > maxSplitResultCount_) {
      SK_DLOGE("aclskScopeVerify splitResults buffer is too small, maxSplitResultCount=%zu, realSplitResultCount=%zu",
               maxSplitResultCount_, splitRecords_.size());
      return ACL_ERROR_INVALID_PARAM;
    }
    if (splitRecords_.empty()) {
      SK_DLOGI("aclskScopeVerify split result: No risk of Superkernel fusion caused by deadlock or Syncall");
      return ACL_SUCCESS;
    }
    for (size_t i = 0; i < splitRecords_.size(); ++i) {
      auto it = verifyNodesById_.find(splitRecords_[i].nodeId);
      if (it == verifyNodesById_.end()) {
        SK_DLOGE("aclskScopeVerify split node %lu is not found in verify node map", splitRecords_[i].nodeId);
        return ACL_ERROR_FAILURE;
      }
      splitResults_[i].splitNode = it->second;
      splitResults_[i].splitType = splitRecords_[i].splitType;
      splitResults_[i].splitReason = splitRecords_[i].splitReason;
      splitResults_[i].extendType = 0;
      splitResults_[i].extendInfo = nullptr;
      SK_DLOGI("aclskScopeVerify split result[%zu]: taskId=%ld, scopeId=%d, splitType=%s, splitReason=%s", i,
               splitResults_[i].splitNode->taskId, splitResults_[i].splitNode->scopeId,
               ScopeVerifySplitTypeToString(splitResults_[i].splitType),
               ScopeVerifySplitReasonToString(splitResults_[i].splitReason));
    }
    return ACL_SUCCESS;
  }

  const aclskScopeVerifyGraphInfo *verifyGraph_ = nullptr;
  size_t maxSplitResultCount_ = 0;
  aclskScopeVerifySplitResult *splitResults_ = nullptr;
  size_t *realSplitResultCount_ = nullptr;
  SuperKernelGraph graph_;
  std::vector<SuperKernelScopeInfo> scopes_;
  std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById_;
  std::vector<ScopeVerifySplitRecord> splitRecords_;
};

}  // namespace

void LogScopeVerifyInput(const aclskScopeVerifyGraphInfo *verifyGraph, size_t maxSplitResultCount) {
  if (verifyGraph == nullptr || verifyGraph->nodes == nullptr) {
    return;
  }
  SK_DLOGI("aclskScopeVerify input: nodeCount=%zu, maxSplitResultCount=%zu", verifyGraph->nodeCount,
           maxSplitResultCount);
  for (size_t i = 0; i < verifyGraph->nodeCount; ++i) {
    const aclskScopeVerifyNodeInfo &node = verifyGraph->nodes[i];
    SK_DLOGI(
        "aclskScopeVerify input node[%zu]: taskId=%ld, streamId=%ld, eventId=%ld, scopeId=%d, taskType=%d, "
        "kernelType=%d, numBlocks=%u, taskRatio={%u,%u}, scheMode=%d, flag=0x%x, coreLimit={%d,%d}",
        i, node.taskId, node.streamId, node.eventId, node.scopeId, static_cast<int32_t>(node.taskType),
        static_cast<int32_t>(node.kernelType), node.numBlocks, node.taskRatio[0], node.taskRatio[1], node.scheMode,
        node.flag, node.coreLimit[0], node.coreLimit[1]);
  }
}

aclError RunScopeVerify(const aclskScopeVerifyGraphInfo *verifyGraph, size_t maxSplitResultCount,
                        aclskScopeVerifySplitResult *splitResults, size_t *realSplitResultCount) {
  ScopeVerifyRunner runner(verifyGraph, maxSplitResultCount, splitResults, realSplitResultCount);
  return runner.Run();
}
