/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_BUFFER_ALLOCATE_BUF_QUE_ALLOCATOR_H_
#define OPTIMIZE_BUFFER_ALLOCATE_BUF_QUE_ALLOCATOR_H_

#include <map>
#include "ascir.h"
#include "schedule_result.h"
#include "ascgen_log.h"
#include "platform/base_platform.h"
#include "tensor_mem_defs.h"

namespace optimize {
class BufQueAllocator {
 public:
  Status AllocBufQue(::ascir::FusedScheduledResult &fused_scheduled_result);
  Status PrepareImplGraphMemoryPlan(::ascir::FusedScheduledResult &fused_scheduled_result);
  Status CollectFusedIoNodes(::ascir::FusedScheduledResult &fused_scheduled_result);

 private:
  Status AllocBufQueForSingleImplGraph(af::AscGraph &impl_graph, size_t max_que_num,
                                       bool is_reduce_mem_reuse = false) const;
  Status ProcessSingleImplGraph(af::AscGraph &impl_graph, BasePlatform &platform, size_t max_que_num,
                                bool is_reduce_mem_reuse);
  Status AllocateForIoNodes(::ascir::FusedScheduledResult &fused_scheduled_result);
  Status AllocateForIoNodes(const af::AscGraph &impl_graph);
  Status CollectIoNodes(const af::AscGraph &impl_graph);
  void AppendCollectedIoNodes(::ascir::FusedScheduledResult &fused_scheduled_result) const;
  Status SetOutputTensorAttr(const af::AscGraph &impl_graph) const;
  static void SetGlobalMemInfo(const af::AscTensor &tensor, int64_t tensor_id);
  static bool ShouldReplaceRepresentative(const ascir::NodeView &current, const ascir::NodeView &candidate);
  void InitTensorReuseInfoAndLifeTime(const ascir::NodeView &node, const af::AscTensor *output, TensorInfo &tensor_info,
                                      bool is_reduce_mem_reuse, bool is_cube_none_db) const;
  void InitTensorReuseInfo(const ascir::NodeView &node, const af::AscTensor *output, TensorInfo &tensor_info,
                           bool is_reduce_mem_reuse, bool is_node_cached) const;
  static void InitTensorLifeTime(const ascir::NodeView &node, const af::AscTensor *output, TensorInfo &tensor_info,
                                 bool is_node_cached, bool is_cube_none_db);
  static Status InitTensorMemInfo(af::AscGraph &graph, const af::AscTensor *output, TensorInfo &tensor_info);
  Status InitTensorInfo(af::AscGraph &graph, TensorInfoMap &tensor_attr_to_tensor_info, bool is_reduce_mem_reuse) const;
  static Status InitNodeTmpBuffInfo(af::AscGraph &graph, TmpBuffInfoMap &node_attr_to_tensor_info);
  static void AllocateReuseId(const af::AscGraph &graph, TensorInfoMap &tensor_attr_to_tensor_info);
  static TensorInfo *FindBestInplaceSource(const af::AscNodePtr &node, const TensorInfo &output_info,
                                           TensorInfoMap &tensor_attr_to_tensor_info);
  static void InitGroupId(const af::AscGraph &graph, TensorInfoMap &tensor_attr_to_tensor_info);
  static bool IsTensorUsedByOtherUnit(const af::AscNodePtr &node, const af::AscTensor *output);
  Status AllocateWithinGroup(af::AscGraph &graph, size_t &total_vecin_nums, size_t &total_vecout_nums,
                             bool is_reduce_mem_reuse = false) const;
  static Status ShortenVecinLifetime(af::AscGraph &graph, size_t max_que_num);
  static Status ShortenVecoutLifetime(af::AscGraph &graph, size_t max_que_num);
  static Status GetAndSetNodeTempBuffer(const af::AscNodePtr &node);
  static Status TopoSortByLoadPriority(af::AscGraph &graph);
  static Status TopoSortByCubeLoadPriority(af::AscGraph &graph);
  static Status MarkUnreusableTensors(const af::AscGraph &graph);

  int64_t prev_tensor_id_ = 0;
  ascir::CubeTemplateType cube_type{ascir::CubeTemplateType::kDefault};
  std::map<std::string, std::map<int64_t, int64_t>> node_type_to_index_to_tensor_id_;
  std::map<std::string, std::map<int64_t, ::ascir::NodeView>> node_type_to_index_to_node_;
  // workspace 根据name来表达是否同一块
  std::map<std::string, int64_t> workspace_name_to_tensor_id_;
};
}  // namespace optimize

#endif  // OPTIMIZE_BUFFER_ALLOCATE_BUF_QUE_ALLOCATOR_H_
