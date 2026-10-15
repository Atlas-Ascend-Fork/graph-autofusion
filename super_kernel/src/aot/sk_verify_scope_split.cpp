/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_verify_scope_split.h"
#include "sk_node.h"
#include "sk_options_manager.h"
#include "sk_scope_split.h"

namespace {

class ScopeVerifyScheModeSplitPass : public ScheModeKernelSplitPass {
 public:
  ScopeVerifyScheModeSplitPass(SuperKernelGraph &inputGraph, std::vector<ScopeVerifySplitRecord> &splitRecords)
      : ScheModeKernelSplitPass(inputGraph), splitRecords_(splitRecords) {}

 protected:
  void RecordScheModeSplitResult(SuperKernelBaseNode *splitNode) override {
    if (splitNode == nullptr) {
      return;
    }
    splitRecords_.push_back(ScopeVerifySplitRecord{splitNode->GetNodeId(), ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE,
                                                   ACLSK_SCOPE_VERIFY_SYNCALL_OP_DROP});
  }

 private:
  std::vector<ScopeVerifySplitRecord> &splitRecords_;
};

class ScopeVerifyDeadlockRefinePass : public DeadlockRefinePass {
 public:
  ScopeVerifyDeadlockRefinePass(SuperKernelGraph &inputGraph, SuperKernelOptionsManager &opts,
                                std::vector<ScopeVerifySplitRecord> &splitRecords)
      : DeadlockRefinePass(inputGraph, opts), splitRecords_(splitRecords) {}

 protected:
  void RecordDeadlockSplitResult(SuperKernelBaseNode *deadlockNode, SuperKernelBaseNode *deadlockWaitNode) override {
    (void)deadlockNode;
    if (deadlockWaitNode == nullptr) {
      return;
    }
    splitRecords_.push_back(ScopeVerifySplitRecord{deadlockWaitNode->GetNodeId(), ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE,
                                                   ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED});
  }

 private:
  std::vector<ScopeVerifySplitRecord> &splitRecords_;
};

}  // namespace

bool RunScopeVerifyScheModeSplitPass(SuperKernelGraph &graph, std::vector<SuperKernelScopeInfo> &scopes,
                                     std::vector<ScopeVerifySplitRecord> &splitRecords) {
  ScopeVerifyScheModeSplitPass scheModePass(graph, splitRecords);
  return scheModePass.Run(scopes);
}

bool RunScopeVerifyDeadlockRefinePass(SuperKernelGraph &graph, SuperKernelOptionsManager &opts,
                                      std::vector<SuperKernelScopeInfo> &scopes,
                                      std::vector<ScopeVerifySplitRecord> &splitRecords) {
  ScopeVerifyDeadlockRefinePass deadlockPass(graph, opts, splitRecords);
  return deadlockPass.Run(scopes);
}
