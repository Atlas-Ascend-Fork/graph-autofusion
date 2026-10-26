/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_OPTIMIZER_H__
#define __SK_OPTIMIZER_H__

#include "sk_types.h"
#include "sk_options_manager.h"
#include "sk_candidate_heap.h"
#include "sk_scope_split.h"
#include "sk_scope_postprocess.h"
#include "sk_graph.h"
#include "sk_scope_info.h"

class SkTaskBuilder;

/**
 * @brief Super Kernel 优化器 - 核心编排类 (Facade Pattern)
 *
 * 职责：
 * 1. 解析运行时选项为 OptimizeOptions
 * 2. 从模型中捕获任务信息
 * 3. 使用 SkTaskBuilder 构建优化任务
 * 4. 通过 Update 持久化参数到 RTS buffer，支持 replay
 */
class SuperKernelOptimizer {
 public:
  SuperKernelOptimizer(SuperKernelOptionsManager &opts) : opts(opts) {}
  virtual ~SuperKernelOptimizer() = default;
  bool Process(SuperKernelGraph &graph);

  /*!
   * @brief Get the processed scope infos after fusion
   * @return Const reference to the vector of scope infos
   * @note Only valid after Process() has been called successfully
   */
  const std::vector<SuperKernelScopeInfo> &GetScopeInfos() const {
    return processedScopeInfos_;
  }

  /*!
   * @brief Add task queue JSON for a scope
   * @param scopeId Scope ID
   * @param taskQueueJson JSON object for the task queue
   */
  void AddScopeTaskQueueJson(uint16_t scopeId, Json taskQueueJson) {
    taskQueueJsons_[std::to_string(scopeId)] = taskQueueJson;
  }

 private:
  SuperKernelOptionsManager &opts;
  bool ShouldReorderWaitNodesForTaskBuild() const;
  std::vector<SuperKernelBaseNode *> ReorderWaitNodesForTaskBuild(
      const std::vector<SuperKernelBaseNode *> &taskNodes) const;
  void ConnectLastKernelsToFirstEvent(const std::vector<SuperKernelBaseNode *> &taskNodes,
                                      SuperKernelBaseNode *firstEvent) const;
  bool ExtractIntraStreamDependencies(const std::vector<SuperKernelBaseNode *> &taskNodes) const;
  std::vector<SuperKernelScopeInfo> processedScopeInfos_;
  std::unordered_map<std::string, Json> taskQueueJsons_;  // scopeId -> task queue json
  bool Schedule(SuperKernelScopeInfo &scopeInfo, SuperKernelGraph &graph, SkTaskBuilder &builder);
  bool Update(SuperKernelScopeInfo &scopeInfo, SuperKernelGraph &graph, const SkLaunchInfo &launchInfo);
};

#endif  // __SK_OPTIMIZER_H__
