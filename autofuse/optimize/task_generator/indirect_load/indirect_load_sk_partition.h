/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_SK_PARTITION_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_SK_PARTITION_H

#include "task_generator/indirect_load/indirect_load_types.h"

namespace optimize {
namespace indirect_load_detail {
// SK 分区：workspace 边界插入、SK 图分区、Norm 边界化与 SK 图遍历（自 indirect_load_schedule_case_generator.cpp
// 纯搬迁，语义不变）。

af::Status CopyBoundaryTensorAttr(const af::AscNodePtr &src_node, size_t src_output_idx,
                                  const af::AscNodePtr &dst_node);

af::Status CopyWorkspaceTensorAttr(const af::AscNodePtr &boundary_node, const af::AscNodePtr &workspace_node);

af::Status InsertWorkspaceBoundary(af::AscGraph &graph, const af::AscNodePtr &src_node, size_t src_output_idx,
                                   const af::AscNodePtr &dst_node, size_t dst_input_idx,
                                   const std::string &boundary_name, bool align_store, bool align_load);

af::Status PartitionSkGraph(af::AscGraph &graph, const af::AscNodePtr &indirect_load, bool align_input_path,
                            bool align_index_path, bool skip_input_boundaries = false);

af::Status BuildSkPartitionOrder(const ascir::ImplGraph &graph, const af::AscNodePtr &indirect_load,
                                 std::vector<af::AscNodePtr> &node_order);

af::Status ApplySkGraphPass(af::AscGraph &graph, const af::AscNodePtr &indirect_load, bool &is_candidate_legal);

// [SK段SIMT化] 重定向的形态约束：IL 输出的下游闭包（DAG，LayerNorm 的
// x-mean 为菱形双消费）全部由 Cast/Reduce/Broadcast/Elementwise/Scalar/Store/
// Output 构成、且至少一个 Reduce 时允许重定向——即 Norm 类复合区域结构本身。
// 这是 R 非全载需要分段归约的目标形态：IL 段 SIMT gather 写 output workspace，
// 消费子图（单/多 Reduce 的 Norm DAG）独立调度，R 轴分 tile 由其自身 tiling 承载。
// 闭包含 IL/Transpose 等特殊算子或控制边时维持原 SK 行为不重定向。
bool IsNormRegionDag(const af::AscNodePtr &indirect_load);

// [SK段SIMT化·复合Norm] Norm 链分段化。通用调度对含 Reduce 的连通图按 R 轴切
// loop，Reduce/多消费者节点输出的下游若与其同分量，跨 loop 引用（codegen
// ConnectApiCallInputs 在单 loop 的 tensor_calls 中找不到生产者）。且连通性分区
// 为双向遍历（上下游都走），LayerNorm 的 x-mean 菱形（diff 被方差链与输出链双
// 消费）使单一 Reduce 边界无法切断分量。规则：对每个 Reduce 节点、每个多消费
// 者计算节点的输出，逐消费者独立插 workspace 边界（每边一份，彻底切断连通分
// 量）——每个子图数据流自足（输入均为 Load），R 切分不再产生跨循环依赖，与独
// 立 LayerNorm 的统计段/消费段分离形态对齐。单消费者非 Reduce 节点不切（同段
// 内常规流水）。
af::Status PartitionNormReduceBoundaries(af::AscGraph &graph, const af::AscNodePtr &indirect_load);

}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // {guard}
