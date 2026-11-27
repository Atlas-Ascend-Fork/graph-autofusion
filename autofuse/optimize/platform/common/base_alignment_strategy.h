/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_COMMON_BASE_ALIGNMENT_STRATEGY_H
#define OPTIMIZE_PLATFORM_COMMON_BASE_ALIGNMENT_STRATEGY_H

#include <cstdint>
#include <queue>
#include "ascir.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "optimize/schedule_utils.h"
#include "graph/utils/graph_utils.h"
#include "symbolizer/symbolic_utils.h"

namespace optimize {
constexpr uint32_t kAlignWidth = 32U;

enum class AlignmentType : uint32_t {
  kNotAligned = 0U,  // 不需要对齐
  kAligned,          // 尾轴对齐
  kDiscontinuous,    // 尾轴非连续
  kFixedNotAligned,  // 固定非对齐
  kInvalid,          // 无效值
};

struct TensorAlignState {
  AlignmentType align_type = AlignmentType::kNotAligned;
  bool conflict_with_output = false;
};

using AlignInferFunc = std::function<af::Status(const af::AscNodePtr &)>;

class BaseAlignmentStrategy {
 public:
  virtual ~BaseAlignmentStrategy() = default;
  explicit BaseAlignmentStrategy() = default;
  BaseAlignmentStrategy(const BaseAlignmentStrategy &) = delete;
  BaseAlignmentStrategy &operator=(const BaseAlignmentStrategy &) = delete;
  BaseAlignmentStrategy(BaseAlignmentStrategy &&) = delete;
  BaseAlignmentStrategy &operator=(BaseAlignmentStrategy &&) = delete;

  // 只允许load出现尾轴非连续
  af::Status AlignVectorizedStrides(ascir::ImplGraph &impl_graph);
  static af::Status SetVectorizedStridesForTensor(const af::NodePtr &node, af::AscTensorAttr &output_attr,
                                                  const AlignmentType align_type);
  static uint32_t GetAlignWidth() {
    return align_width_;
  }
  virtual af::Status ModifyVectorizedStrides([[maybe_unused]] ascir::ImplGraph &impl_graph) {
    return af::SUCCESS;
  }

 protected:
  using NodeProcessor = af::Status (BaseAlignmentStrategy::*)(ascir::ImplGraph &, const af::AscNodePtr &, bool &);

  static const char *AlignmentTypeToString(AlignmentType type);

  virtual AlignmentType GetDefaultAlignmentType() = 0;

  virtual void InitAlignmentInferFunc();
  virtual af::Status DefaultAlignmentInferFunc(const af::AscNodePtr &node);
  virtual af::Status BroadcastAlignmentInferFunc(const af::AscNodePtr &node);
  virtual af::Status ConcatAlignmentInferFunc(const af::AscNodePtr &node);
  virtual af::Status EleWiseAlignmentInferFunc(const af::AscNodePtr &node);
  virtual af::Status LoadAlignmentInferFunc(const af::AscNodePtr &node);
  virtual af::Status StoreAlignmentInferFunc(const af::AscNodePtr &node);
  virtual af::Status ReduceAlignmentInferFunc(const af::AscNodePtr &node);
  virtual af::Status SplitAlignmentInferFunc(const af::AscNodePtr &node);

  virtual af::Status SetAlignWidth(const ascir::ImplGraph &impl_graph);
  af::Status ForEachNode(ascir::ImplGraph &graph, NodeProcessor processor);
  af::Status InferAlignmentForOneNode(ascir::ImplGraph &impl_graph, const af::AscNodePtr &node, bool &changed);
  // 当前tensor的对齐行为只会出现在尾轴,如果没有新的对齐行为或者类型,该函数不应该修改
  af::Status SetVectorizedStridesForOneNode(ascir::ImplGraph &impl_graph, const af::AscNodePtr &node, bool &changed);
  // 反向推导对齐逻辑
  virtual af::Status BackPropagateAlignment(const af::AscNodePtr &node,
                                            AlignmentType aligned_type = AlignmentType::kAligned);
  void SetAlignInfoForNodeInputs(AlignmentType aligned_type, af::AscNode *node, std::set<af::Node *> &visited_nodes,
                                 std::queue<af::Node *> &node_queue);
  bool SetAlignInfoForNodeOutputs(AlignmentType aligned_type, af::AscNode *node, std::set<af::Node *> &visited_nodes,
                                  std::queue<af::Node *> &node_queue);

  af::Status AddRemovePadForOneNode(ascir::ImplGraph &impl_graph, const af::AscNodePtr &node, bool &inserted);
  af::Status AddPadForCompactReduce(ascir::ImplGraph &impl_graph, const af::AscNodePtr &node, bool &inserted);
  bool FindCompactReduceInput(const af::AscNodePtr &node, af::InDataAnchorPtr &reduce_input,
                              af::OutDataAnchorPtr &reduce_output, size_t start_index) const;
  af::Status InsertPadForCompactReduce(ascir::ImplGraph &impl_graph, const af::InDataAnchorPtr &reduce_input,
                                       const af::OutDataAnchorPtr &reduce_output, bool &inserted);
  af::Status CheckIsNoNeedPad(const af::AscNodePtr &node, af::AscTensorAttr &out_attr, bool &is_no_need_pad) const;
  af::Status AddPadForAlignmentConflictOneNode(ascir::ImplGraph &impl_graph, const af::AscNodePtr &node,
                                               bool &inserted);
  // 多输入elewise,有一个fix,需要向上传递fix状态,防止输入链路上被后续节点刷成align
  af::Status BackPropagateFixUnAlignType(const af::AscNodePtr &node);

  std::unordered_map<const af::AscTensorAttr *, TensorAlignState> tensor_to_align_type_;
  std::map<af::ComputeType, AlignInferFunc> compute_type_to_infer_func_;
  inline static uint32_t align_width_ = 32U;
};

bool IsLoadNeedAlignForReduce(const af::AscNodePtr &node);
bool IsLoadNeedAlign(const af::AscNodePtr &node_load);
bool IsTailAxisTranspose(const af::AscTensorAttr &attr);
bool IsTailAxisTransposeV2(const af::AscNodePtr &node_load);
}  // namespace optimize
#endif  // OPTIMIZE_PLATFORM_COMMON_BASE_ALIGNMENT_STRATEGY_H
