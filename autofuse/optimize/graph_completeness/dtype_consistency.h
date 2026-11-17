/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef OPTIMIZE_GRAPH_COMPLETENESS_DTYPE_CONSISTENCY_H
#define OPTIMIZE_GRAPH_COMPLETENESS_DTYPE_CONSISTENCY_H

#include <vector>
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace optimize {
// Actual dtype requirements of nodes
struct NodeDtypeRequirement {
  af::AscNodePtr node;
  std::vector<af::DataType> input_dtypes;
  std::vector<af::DataType> output_dtypes;
  std::vector<std::vector<af::DataType>> output_cast_paths;
};

class DtypeConsistency {
 public:
  // Ensure dtype consistency for the graph: insert necessary Cast nodes and remove redundant ones
  static af::Status EnsureDtypeConsistency(af::AscGraph &graph);

 private:
  // Collect dtype requirements for all nodes
  static af::Status CollectDtypeRequirements(af::AscGraph &graph, std::vector<NodeDtypeRequirement> &requirements);

  // Process output dtype: directly modify the dtype of node's output tensor
  static af::Status ProcessOutputDtype(const NodeDtypeRequirement &req);

  // Apply ASCIR codegen output cast paths when direct execution-to-output conversion cannot preserve semantics.
  static af::Status ProcessOutputCastPath(af::AscGraph &graph, const NodeDtypeRequirement &req);

  // Process input dtype: insert Cast when dtype does not match
  static af::Status ProcessInputDtype(af::AscGraph &graph, const NodeDtypeRequirement &req);

  // Check if cast conversion is supported
  static af::Status CheckCastSupported(af::DataType src_dtype, af::DataType dst_dtype, const af::AscNodePtr &node,
                                       size_t input_idx);

  // Try to merge with upstream cast, return whether merge succeeded
  static bool TryMergeWithUpstreamCast(af::AscGraph &graph, const af::AscNodePtr &upstream_cast,
                                       const af::AscNodePtr &downstream_node, size_t input_idx,
                                       af::DataType target_dtype);

  // Merge cast when there's only one downstream consumer
  static bool MergeCastWithSingleConsumer(const af::AscNodePtr &upstream_cast, const af::AscNodePtr &downstream_node,
                                          size_t input_idx, af::DataType target_dtype);

  // Merge cast when there are multiple downstream consumers
  static bool MergeCastWithMultipleConsumers(af::AscGraph &graph, const af::AscNodePtr &upstream_cast,
                                             const af::AscNodePtr &downstream_node, size_t input_idx,
                                             af::DataType target_dtype);

  // Insert a new cast node
  static af::Status InsertCastNode(af::AscGraph &graph, const af::AscNodePtr &src_node, const af::AscNodePtr &dst_node,
                                   size_t input_idx, af::DataType target_dtype);

  static af::Status ApplyDtypeConversions(af::AscGraph &graph, const std::vector<NodeDtypeRequirement> &requirements);

  // Remove redundant Cast operators
  static af::Status CancelRedundantCast(af::AscGraph &graph);

  // Merge multiple identical dtype Cast nodes from the same upstream into one
  static af::Status DoCastCSE(af::AscGraph &graph);

  // Remove Cast(A->A) redundancy
  static af::Status CancelIdentityCast(af::AscGraph &graph);
};

}  // namespace optimize
#endif  // OPTIMIZE_GRAPH_COMPLETENESS_DTYPE_CONSISTENCY_H
