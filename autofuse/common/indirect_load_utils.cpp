/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indirect_load_utils.h"

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "common_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "schedule_result.h"
#include "utils/extern_math_util.h"

namespace ascgen_utils::indirect_load {
namespace {
constexpr char kTemplateOuterAxisAttr[] = "af.internal.indirect_load.outer_axis";
constexpr char kTemplateInnerAxisAttr[] = "af.internal.indirect_load.inner_axis";
constexpr char kTemplateInputInnerAxisAttr[] = "af.internal.indirect_load.input_inner_axis";
constexpr char kTemplateIndexInnerAxisAttr[] = "af.internal.indirect_load.index_inner_axis";
constexpr char kTemplateTileOuterAxisAttr[] = "af.internal.indirect_load.tile_outer_axis";
constexpr char kTemplateTileInnerAxisAttr[] = "af.internal.indirect_load.tile_inner_axis";
constexpr char kTemplateVectorizedAxesAttr[] = "af.internal.indirect_load.vectorized_axes";
constexpr char kTemplateSyntheticOuterAttr[] = "af.internal.indirect_load.synthetic_outer";
constexpr char kImplementationAttr[] = "af.internal.indirect_load.implementation";
constexpr char kSimdInputInnerAxisHintAttr[] = "af.internal.indirect_load.simd_input_inner_axis";
constexpr char kSimtGroupMemberAttr[] = "af.internal.indirect_load.simt_group_member";
bool IsValidLogicalTensorView(const LogicalTensorView &view) {
  return !view.axis_ids.empty() && view.axis_ids.size() == view.sizes.size() &&
         view.axis_ids.size() == view.strides.size();
}

enum class TensorDimKind : int64_t { kIllegal, kZeroStride, kRegular };
TensorDimKind ClassifyTensorDim(const af::Expression &size, const af::Expression &stride) {
  if (af::SymbolicUtils::StaticCheckLe(size, af::sym::kSymbolZero) == af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckLt(stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
    return TensorDimKind::kIllegal;
  }
  if (af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
    return TensorDimKind::kZeroStride;
  }
  return TensorDimKind::kRegular;
}

bool TryClassifyDynamicShapeLayout(const LogicalTensorView &logical, IndirectLoadTensorLayout &layout) {
  const bool has_dynamic_shape = std::any_of(logical.sizes.begin(), logical.sizes.end(),
                                             [](const af::Expression &size) { return !size.IsConstExpr(); });
  if (!has_dynamic_shape) {
    return false;
  }
  // A dynamic outer dimension does not prevent proving a compact zero-stride
  // view when all non-broadcast dimensions are contiguous. Preserve this
  // producer-side layout so Broadcast and its source use the same tensor view.
  bool has_zero_stride = false;
  af::Expression physical_span = af::sym::kSymbolOne;
  for (size_t dim = 0UL; dim < logical.sizes.size(); ++dim) {
    const auto dim_kind = ClassifyTensorDim(logical.sizes[dim], logical.strides[dim]);
    if (dim_kind == TensorDimKind::kIllegal) {
      return false;
    }
    if (dim_kind == TensorDimKind::kZeroStride) {
      layout.physical_repeats[dim] = af::sym::kSymbolOne;
      has_zero_stride = true;
    }
  }
  if (has_zero_stride) {
    physical_span = af::sym::kSymbolOne;
    for (size_t index = logical.sizes.size(); index > 0UL; --index) {
      const size_t dim = index - 1UL;
      if (ClassifyTensorDim(logical.sizes[dim], logical.strides[dim]) == TensorDimKind::kZeroStride) {
        continue;
      }
      if (af::SymbolicUtils::StaticCheckEq(logical.strides[dim], physical_span) != af::TriBool::kTrue) {
        layout.kind = IndirectLoadLayoutKind::kStrided;
        return true;
      }
      physical_span = physical_span + (logical.sizes[dim] - af::sym::kSymbolOne) * logical.strides[dim];
    }
    layout.kind = IndirectLoadLayoutKind::kZeroStrideCompact;
    return true;
  }
  // 无零 stride 的动态视图与静态流程使用同一符号连续性判据证明 dense：证明成立时
  // dense facade（RegisterGather/GatherApi）的 kernel 参数全部为运行时表达式，
  // 动态 shape 可安全使用向量化 Gather；证明不成立（维间存在不可排除的物理间隙）
  // 时保持 kStrided，由 codegen 运行时推导 offsets。
  physical_span = af::sym::kSymbolOne;
  bool has_physical_gap = false;
  for (size_t index = logical.sizes.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    if (af::SymbolicUtils::StaticCheckLt(logical.strides[dim], physical_span) != af::TriBool::kFalse) {
      layout.kind = IndirectLoadLayoutKind::kStrided;
      return true;
    }
    has_physical_gap =
        has_physical_gap || af::SymbolicUtils::StaticCheckEq(logical.strides[dim], physical_span) != af::TriBool::kTrue;
    physical_span = physical_span + (logical.sizes[dim] - af::sym::kSymbolOne) * logical.strides[dim];
  }
  layout.kind = has_physical_gap ? IndirectLoadLayoutKind::kStrided : IndirectLoadLayoutKind::kDense;
  return true;
}

// 转置稠密视图检测：存在轴排列使行主序稠密链精确衔接（每层 stride 等于已覆盖跨度）
// 时，视图物理非重叠且完整覆盖底层数据，仅逻辑轴序非行主序，判 kStrided（与动态
// 分支对动态形状统一判 kStrided 的语义一致；kDense 要求轴序与物理序一致，不适用）。
// 链按 stride 值唯一吸收：歧义只会出现在重叠视图（链必断，正确拒绝）；退化轴
// （size==1）单元素，任意位置入链。stride 互质的混洗（如 [2,3] strides=[3,2]）
// 每维跨步、不在倍数链内，维持拒绝。
bool TryClassifyTransposedDenseLayout(const LogicalTensorView &logical, IndirectLoadTensorLayout &layout) {
  const size_t rank = logical.sizes.size();
  std::vector<bool> used(rank, false);
  af::Expression expected = af::sym::kSymbolOne;
  for (;;) {
    // 非退化轴优先入链并推进已覆盖跨度；退化轴随后扫尾（不改变链）。
    bool progressed = false;
    for (size_t dim = 0UL; dim < rank; ++dim) {
      if (used[dim] || af::SymbolicUtils::StaticCheckEq(logical.sizes[dim], af::ops::One) == af::TriBool::kTrue) {
        continue;
      }
      if (af::SymbolicUtils::StaticCheckEq(logical.strides[dim], expected) != af::TriBool::kTrue) {
        continue;
      }
      used[dim] = true;
      expected = af::sym::Mul(expected, logical.sizes[dim]);
      progressed = true;
    }
    if (!progressed) {
      break;
    }
  }
  for (size_t dim = 0UL; dim < rank; ++dim) {
    if (used[dim] || af::SymbolicUtils::StaticCheckEq(logical.sizes[dim], af::ops::One) != af::TriBool::kTrue) {
      continue;
    }
    used[dim] = true;  // 退化轴（size==1）单元素，任意正 stride 均非重叠。
  }
  if (!std::all_of(used.begin(), used.end(), [](bool dim_used) { return dim_used; })) {
    return false;
  }
  layout.kind = IndirectLoadLayoutKind::kStrided;
  return true;
}

bool IsValidTensorLayout(const IndirectLoadTensorLayout &layout) {
  return IsValidLogicalTensorView(layout) && layout.kind != IndirectLoadLayoutKind::kUnsupported &&
         layout.physical_repeats.size() == layout.sizes.size();
}

TemplateAxes ReadTemplateAxes(const af::AscNodePtr &node) {
  TemplateAxes axes;
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return axes;
  }
  const auto op_desc = node->GetOpDesc();
  axes.outer_axis = op_desc->TryGetExtAttr(kTemplateOuterAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.inner_axis = op_desc->TryGetExtAttr(kTemplateInnerAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.input_inner_axis = op_desc->TryGetExtAttr(kTemplateInputInnerAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.index_inner_axis = op_desc->TryGetExtAttr(kTemplateIndexInnerAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.tile_outer_axis = op_desc->TryGetExtAttr(kTemplateTileOuterAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.tile_inner_axis = op_desc->TryGetExtAttr(kTemplateTileInnerAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.vectorized_axes = op_desc->TryGetExtAttr(kTemplateVectorizedAxesAttr, std::vector<af::AxisId>{});
  axes.synthetic_outer = op_desc->TryGetExtAttr(kTemplateSyntheticOuterAttr, false);
  return axes;
}

TemplateRole GetAnnotatedTemplateRole(const af::AscNodePtr &node) {
  if (node == nullptr) {
    return TemplateRole::kNone;
  }
  return static_cast<TemplateRole>(::ascir::GetTemplateRoleOrDefault(*node, static_cast<int64_t>(TemplateRole::kNone)));
}

TemplateBehavior GetBehavior(TemplateRole role) {
  TemplateBehavior behavior;
  switch (role) {
    case TemplateRole::kSimdInputPre:
    case TemplateRole::kSimdInputPreStridedUbPath:
    case TemplateRole::kSimdIndexPre:
      behavior.excludes_tiling_group = true;
      behavior.preserves_vectorized_axis = true;
      break;
    case TemplateRole::kSimtInputBoundary:
    case TemplateRole::kSimtDirectGmBoundary:
    case TemplateRole::kSimtInlineTransform:
      behavior.skips_main_schedule_tiling = true;
      behavior.skips_api_emit = true;
      behavior.uses_direct_gm_pipeline = true;
      behavior.preserves_vectorized_axis = true;
      break;
    case TemplateRole::kSimtFanoutBranch:
      behavior.skips_main_schedule_tiling = true;
      behavior.skips_api_emit = true;
      behavior.skips_ub_lifecycle = true;
      behavior.uses_direct_gm_pipeline = true;
      behavior.preserves_vectorized_axis = true;
      break;
    case TemplateRole::kSimtOp:
      behavior.uses_direct_gm_pipeline = true;
      behavior.skips_ub_lifecycle = true;
      behavior.preserves_vectorized_axis = true;
      break;
    case TemplateRole::kSkInputBoundary:
      behavior.skips_main_schedule_tiling = true;
      behavior.skips_api_emit = true;
      behavior.preserves_vectorized_axis = true;
      break;
    default:
      break;
  }
  return behavior;
}

struct PostReduceChain {
  af::AscNodePtr input_producer;
  af::AscNodePtr reduce;
};

PostReduceChain FindPostReduceChain(const af::AscNodePtr &node) {
  if (node == nullptr) {
    return {};
  }
  std::vector<af::AscNodePtr> pending;
  std::unordered_set<af::AscNode *> visited;
  for (const auto &out_node : node->GetOutDataNodes()) {
    const auto out_asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
    if (out_asc_node != nullptr && visited.emplace(out_asc_node.get()).second) {
      pending.emplace_back(out_asc_node);
    }
  }

  af::AscNodePtr reduce;
  for (size_t index = 0UL; index < pending.size(); ++index) {
    const auto &current = pending[index];
    if (current->attr.api.compute_type == af::ComputeType::kComputeReduce) {
      if (reduce == nullptr) {
        reduce = current;
      }
      // 不在 Reduce 处截断：继续遍历 Reduce 下游的 Broadcast/Elementwise，
      // 使串行 R1→Broadcast→Elementwise→R2 仍由现有 post-reduce lowering
      // 看到首个 Reduce；后续 Reduce 由普通调度链执行，不新增专用 Norm API。
    }
    for (const auto &out_node : current->GetOutDataNodes()) {
      const auto out_asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (out_asc_node != nullptr && visited.emplace(out_asc_node.get()).second) {
        pending.emplace_back(out_asc_node);
      }
    }
  }
  if (reduce == nullptr) {
    return {};
  }
  return {GetInputProducer(reduce, 0UL), reduce};
}
}  // namespace

TemplateRole GetTemplateRole(const af::AscNodePtr &node) {
  return GetAnnotatedTemplateRole(node);
}

TemplateBehavior GetTemplateBehavior(const af::AscNodePtr &node) {
  const TemplateRole role = GetTemplateRole(node);
  TemplateBehavior behavior = GetBehavior(role);
  if (role == TemplateRole::kSimtOp && GetPostReduceConsumer(node) != nullptr) {
    behavior = {};
    behavior.excludes_tiling_group = true;
  }
  behavior.skips_input_lifecycle = node != nullptr && af::ops::IsOps<af::ascir_op::IndirectLoad>(node) &&
                                   ::ascir::GetTemplateIdOrDefault(*node) == ::ascir::TemplateId::kIndirectLoadSimt;
  return behavior;
}

af::AscNodePtr GetOnlyOutputConsumer(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOutDataNodesSize() != 1UL) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<af::AscNode>(*node->GetOutDataNodes().begin());
}

ascir::TensorId FindSkippedChainResultTensor(const af::AscNodePtr &root) {
  if (root == nullptr || root->outputs().empty()) {
    return af::kIdNone;
  }
  std::vector<af::AscNodePtr> pending{root};
  std::unordered_set<af::AscNode *> visited;
  ascir::TensorId result = root->outputs()[0]->attr.mem.tensor_id;
  for (size_t i = 0UL; i < pending.size(); ++i) {
    const auto &node = pending[i];
    if (node == nullptr || !visited.emplace(node.get()).second || node->outputs().empty()) {
      continue;
    }
    result = node->outputs()[0]->attr.mem.tensor_id;
    bool has_skipped_consumer = false;
    for (const auto &out_node : node->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (consumer == nullptr) {
        continue;
      }
      const auto behavior = GetTemplateBehavior(consumer);
      if (behavior.skips_api_emit) {
        has_skipped_consumer = true;
        pending.emplace_back(consumer);
      }
    }
    if (!has_skipped_consumer) {
      return result;
    }
  }
  return result;
}

af::AscNodePtr GetPostReduceConsumer(const af::AscNodePtr &node) {
  return FindPostReduceChain(node).reduce;
}

af::AscNodePtr GetPostReduceInputProducer(const af::AscNodePtr &node) {
  return FindPostReduceChain(node).input_producer;
}

bool ShouldSkipTpipeTensorCollection(const af::AscNodePtr &node) {
  const TemplateBehavior behavior = GetTemplateBehavior(node);
  const af::AscNodePtr consumer = GetOnlyOutputConsumer(node);
  if (!(behavior.skips_api_emit || behavior.skips_ub_lifecycle)) {
    return false;
  }
  // 豁免1：kSimtInlineTransform 的唯一消费者是 Reduce，其输出需进入 tpipe。
  if (GetTemplateRole(node) == TemplateRole::kSimtInlineTransform && consumer != nullptr &&
      consumer->attr.api.compute_type == af::ComputeType::kComputeReduce) {
    return false;
  }
  // 豁免2：post-Reduce 输出重定向的目标节点（post-Reduce 输入生产者链末端），
  // 其输出 tensor 会被 IndirectLoad 的 outputs[0].id 重定向引用，
  // RegisterApiCallOutputs 需要从 tpipe 取到该 tensor，必须收集。
  const auto owner_graph = node->GetOwnerComputeGraph();
  if (owner_graph != nullptr) {
    for (const auto &graph_node : owner_graph->GetDirectNode()) {
      const auto candidate = std::dynamic_pointer_cast<af::AscNode>(graph_node);
      if (candidate == nullptr || !af::ops::IsOps<af::ascir_op::IndirectLoad>(candidate)) {
        continue;
      }
      const auto post_reduce_producer = GetPostReduceInputProducer(candidate);
      if (post_reduce_producer != nullptr && post_reduce_producer->GetName() == node->GetName()) {
        return false;
      }
    }
  }
  return true;
}

af::Status InheritTemplateRoleIfIL(af::AscGraph &graph, const std::string &vf_node_name, const af::AscNodePtr &src) {
  GE_ASSERT_NOTNULL(src);
  if (GetTemplateRole(src) == TemplateRole::kNone) {
    return af::SUCCESS;
  }
  auto vf_node = graph.FindNode(vf_node_name.c_str());
  GE_ASSERT_NOTNULL(vf_node, "IndirectLoad: cannot find new VectorFunc node %s.", vf_node_name.c_str());
  GE_ASSERT_SUCCESS(::ascir::SetTemplateRole(vf_node, ::ascir::GetTemplateRoleOrDefault(*src)));
  return af::SUCCESS;
}

af::Status SetTemplateRole(const af::AscNodePtr &node, TemplateRole role) {
  return ::ascir::SetTemplateRole(node, static_cast<int64_t>(role));
}

af::Status SetTemplateAxes(const af::AscNodePtr &node, const TemplateAxes &axes) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateOuterAxisAttr, static_cast<int64_t>(axes.outer_axis)),
                 "Set IndirectLoad outer axis failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateInnerAxisAttr, static_cast<int64_t>(axes.inner_axis)),
                 "Set IndirectLoad inner axis failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateInputInnerAxisAttr, static_cast<int64_t>(axes.input_inner_axis)),
                 "Set IndirectLoad input inner axis failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateIndexInnerAxisAttr, static_cast<int64_t>(axes.index_inner_axis)),
                 "Set IndirectLoad index inner axis failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateTileOuterAxisAttr, static_cast<int64_t>(axes.tile_outer_axis)),
                 "Set IndirectLoad tile outer axis failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateTileInnerAxisAttr, static_cast<int64_t>(axes.tile_inner_axis)),
                 "Set IndirectLoad tile inner axis failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateVectorizedAxesAttr, axes.vectorized_axes),
                 "Set IndirectLoad vectorized axes failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateSyntheticOuterAttr, axes.synthetic_outer),
                 "Set IndirectLoad synthetic outer failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

af::Status GetTemplateAxes(const af::AscNodePtr &node, TemplateAxes &axes) {
  GE_ASSERT_NOTNULL(node);
  axes = ReadTemplateAxes(node);
  GE_ASSERT_TRUE(axes.outer_axis != af::kIdNone, "IndirectLoad template axes are missing, node = %s",
                 node->GetNamePtr());
  return af::SUCCESS;
}

af::Status SetTemplateLogicalView(const af::AscNodePtr &node, const TemplateLogicalView &view) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(
      IsValidTensorLayout(view.input) && IsValidTensorLayout(view.index) && IsValidLogicalTensorView(view.output),
      "IndirectLoad logical view is invalid, node = %s", node->GetNamePtr());
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateLogicalViewAttr, view), "Set IndirectLoad logical view failed, node = %s",
                 node->GetNamePtr());
  return af::SUCCESS;
}

af::Status GetTemplateLogicalView(const af::AscNodePtr &node, TemplateLogicalView &view) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  view = op_desc->TryGetExtAttr(kTemplateLogicalViewAttr, TemplateLogicalView{});
  GE_ASSERT_TRUE(
      IsValidTensorLayout(view.input) && IsValidTensorLayout(view.index) && IsValidLogicalTensorView(view.output),
      "IndirectLoad logical view is missing or invalid, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

af::Status SetIndirectLoadAccessInfo(const af::AscNodePtr &node, const IndirectLoadAccessInfo &info) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kAccessInfoAttr, info), "Set IndirectLoad access info failed, node = %s",
                 node->GetNamePtr());
  return af::SUCCESS;
}

af::Status GetIndirectLoadAccessInfo(const af::AscNodePtr &node, IndirectLoadAccessInfo &info) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  info = op_desc->TryGetExtAttr(kAccessInfoAttr, IndirectLoadAccessInfo{});
  return af::SUCCESS;
}

af::Status SetLoweringMetadata(const af::AscNodePtr &node, const IndirectLoadLoweringMetadata &metadata) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(metadata.version == 1U && metadata.axis >= 0L && metadata.outer_axis != af::kIdNone,
                 "IndirectLoad lowering metadata is invalid, node = %s", node->GetNamePtr());
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kLoweringMetadataAttr, metadata),
                 "Set IndirectLoad lowering metadata failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

af::Status GetLoweringMetadata(const af::AscNodePtr &node, IndirectLoadLoweringMetadata &metadata) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  metadata = op_desc->TryGetExtAttr(kLoweringMetadataAttr, IndirectLoadLoweringMetadata{});
  GE_ASSERT_TRUE(metadata.version == 1U && metadata.axis >= 0L && metadata.outer_axis != af::kIdNone,
                 "IndirectLoad lowering metadata is missing or invalid, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

bool HasLoweringMetadata(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return false;
  }
  const auto metadata = node->GetOpDesc()->TryGetExtAttr(kLoweringMetadataAttr, IndirectLoadLoweringMetadata{});
  return metadata.version == 1U && metadata.axis >= 0L && metadata.outer_axis != af::kIdNone;
}

namespace {
af::Expression ProductFrom(const LogicalTensorView &view, size_t begin) {
  af::Expression product = af::sym::kSymbolOne;
  for (size_t index = begin; index < view.sizes.size(); ++index) {
    product = product * view.sizes[index];
  }
  return product;
}

// Verifies that the suffix after axis_index is contiguous.  When skip_zero_stride is set,
// zero-stride (broadcast) dimensions are ignored by the stride chain.  has_payload reports
// whether any suffix dimension carries data; axis_stride receives the dense stride of the
// axis dimension on success.
bool IsContiguousSuffixImpl(const LogicalTensorView &view, size_t axis_index, bool skip_zero_stride, bool &has_payload,
                            af::Expression &axis_stride) {
  has_payload = false;
  if (axis_index >= view.sizes.size()) {
    return false;
  }
  af::Expression expected_stride = af::sym::kSymbolOne;
  for (size_t index = view.sizes.size(); index > axis_index + 1UL; --index) {
    const size_t dim = index - 1UL;
    if (skip_zero_stride &&
        af::SymbolicUtils::StaticCheckEq(view.strides[dim], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    if (af::SymbolicUtils::StaticCheckEq(view.strides[dim], expected_stride) != af::TriBool::kTrue) {
      return false;
    }
    has_payload = true;
    expected_stride = expected_stride + (view.sizes[dim] - af::sym::kSymbolOne) * view.strides[dim];
  }
  axis_stride = expected_stride;
  return true;
}

bool IsContiguousSuffix(const LogicalTensorView &view, size_t axis_index) {
  bool has_payload = false;
  af::Expression axis_stride;
  return IsContiguousSuffixImpl(view, axis_index, false, has_payload, axis_stride);
}

bool IsContiguousPayloadSuffix(const LogicalTensorView &view, size_t axis_index) {
  bool has_payload = false;
  af::Expression axis_stride;
  return IsContiguousSuffixImpl(view, axis_index, true, has_payload, axis_stride);
}

bool HasZeroStrideOnInputPayload(const LogicalTensorView &input, const LogicalTensorView &index, size_t axis_index) {
  bool has_payload = false;
  af::Expression axis_stride;
  if (input.sizes.size() != index.sizes.size() ||
      !IsContiguousSuffixImpl(input, axis_index, true, has_payload, axis_stride)) {
    return false;
  }
  if (!has_payload) {
    return false;
  }
  for (size_t dim = axis_index + 1UL; dim < input.sizes.size(); ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(input.strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(index.strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

bool IsPayloadInvariantView(const af::AscTensorAttr &source_attr, const LogicalTensorView &input,
                            const LogicalTensorView &index, size_t axis_index) {
  if (source_attr.axis.size() != source_attr.repeats.size() || source_attr.axis.size() != source_attr.strides.size()) {
    return false;
  }
  for (size_t dim = axis_index + 1UL; dim < index.axis_ids.size(); ++dim) {
    // Only dimensions that carry the gathered payload need an invariant index.
    // Other dimensions may legitimately enumerate independent lookups (for
    // example the sequence axis in [B, D, S] embedding views).
    if (af::SymbolicUtils::StaticCheckEq(input.strides[dim], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    const auto source_axis = std::find(source_attr.axis.begin(), source_attr.axis.end(), index.axis_ids[dim]);
    if (source_axis == source_attr.axis.end()) {
      return false;
    }
    const size_t source_dim = static_cast<size_t>(std::distance(source_attr.axis.begin(), source_axis));
    const auto &source_size = source_attr.repeats[source_dim];
    const auto &source_stride = source_attr.strides[source_dim];
    if (af::SymbolicUtils::StaticCheckEq(source_stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    if (af::SymbolicUtils::StaticCheckGt(source_size, af::sym::kSymbolOne) != af::TriBool::kFalse) {
      return false;
    }
  }
  return true;
}

bool IsIndexPayloadInvariantFromSources(const af::AscNodePtr &indirect_load, const LogicalTensorView &input,
                                        const LogicalTensorView &index, size_t axis_index) {
  const af::AscNodePtr root = GetInputProducer(indirect_load, kIndexTensorIndex);
  if (root == nullptr) {
    return false;
  }
  std::vector<af::AscNodePtr> pending{root};
  std::unordered_set<const af::AscNode *> visited;
  bool saw_source = false;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (node == nullptr || !visited.emplace(node.get()).second) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Scalar>(node) || af::ops::IsOps<af::ascir_op::ScalarData>(node)) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Load>(node)) {
      const auto outputs = node->outputs();
      if (outputs.empty() || outputs.front() == nullptr ||
          !IsPayloadInvariantView(outputs.front()->attr, input, index, axis_index)) {
        return false;
      }
      saw_source = true;
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Data>(node)) {
      const auto outputs = node->outputs();
      if (!outputs.empty() && outputs.front() != nullptr && !outputs.front()->attr.axis.empty()) {
        if (!IsPayloadInvariantView(outputs.front()->attr, input, index, axis_index)) {
          return false;
        }
        saw_source = true;
      }
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::IndirectLoad>(node)) {
      return false;
    }
    for (size_t input_index = 0UL; input_index < node->inputs.Size(); ++input_index) {
      pending.emplace_back(GetInputProducer(node, input_index));
    }
  }
  return saw_source;
}

bool IsSimdEmbeddingAccess(const TemplateLogicalView &logical_view, const IndirectLoadAccessInfo &info) {
  const size_t rank = logical_view.input.sizes.size();
  if (info.kind != IndirectLoadAccessInfo::Kind::kEmbeddingLike || rank < 2UL ||
      logical_view.index.sizes.size() != rank || logical_view.output.sizes.size() != rank || info.axis < 0L ||
      static_cast<size_t>(info.axis + 1L) >= rank) {
    return false;
  }
  const auto &input = logical_view.input;
  const auto &index = logical_view.index;
  const auto &output = logical_view.output;
  const size_t axis = static_cast<size_t>(info.axis);
  for (size_t dim = 0UL; dim < rank; ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(index.sizes[dim], output.sizes[dim]) != af::TriBool::kTrue) {
      return false;
    }
  }
  bool has_payload = false;
  af::Expression input_axis_stride;
  if (!IsContiguousSuffixImpl(input, axis, false, has_payload, input_axis_stride)) {
    return false;
  }
  for (size_t dim = axis + 1UL; dim < rank; ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(input.sizes[dim], output.sizes[dim]) != af::TriBool::kTrue) {
      return false;
    }
  }
  if (af::SymbolicUtils::StaticCheckEq(input.strides[axis], input_axis_stride) != af::TriBool::kTrue) {
    return false;
  }
  for (size_t dim = 0UL; dim < axis; ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(input.strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(input.sizes[dim], output.sizes[dim]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return af::SymbolicUtils::StaticCheckGt(input.strides[axis], af::sym::kSymbolZero) == af::TriBool::kTrue;
}
}  // namespace

af::Status AnalyzeIndirectLoadAccess(const af::AscNodePtr &node, const TemplateLogicalView &logical_view,
                                     IndirectLoadAccessInfo &info) {
  GE_ASSERT_NOTNULL(node);
  const auto &input = logical_view.input;
  const auto &index = logical_view.index;
  const auto &output = logical_view.output;
  GE_ASSERT_TRUE(input.sizes.size() == index.sizes.size() && input.sizes.size() == output.sizes.size(),
                 "IndirectLoad access analysis rank mismatch.");
  const size_t rank = input.sizes.size();
  const auto *ir_attr = node->attr.ir_attr == nullptr
                            ? nullptr
                            : node->attr.ir_attr->DownCastTo<af::ascir_op::IndirectLoad::AscIndirectLoadIrAttrDef>();
  GE_ASSERT_NOTNULL(ir_attr, "IndirectLoad access analysis axis attribute is missing.");
  int64_t axis = 0L;
  GE_ASSERT_SUCCESS(ir_attr->GetAxis(axis));
  GE_ASSERT_TRUE(axis >= -static_cast<int64_t>(rank) && axis < static_cast<int64_t>(rank),
                 "IndirectLoad access analysis axis is invalid.");
  const size_t axis_index = static_cast<size_t>(axis < 0L ? axis + static_cast<int64_t>(rank) : axis);

  info = {};
  info.axis = static_cast<int64_t>(axis_index);
  const af::Expression input_inner_span = ProductFrom(input, axis_index + 1UL);
  const auto indirect_load_inputs = node->inputs();
  GE_ASSERT_TRUE(indirect_load_inputs.size() > kInputTensorIndex, "IndirectLoad access analysis input is missing.");
  const af::Expression dtype_bytes =
      af::Symbol(af::GetSizeByDataType(indirect_load_inputs[kInputTensorIndex]->attr.dtype));
  info.input_slice_bytes = input_inner_span * dtype_bytes;
  info.index_varying_extent = af::sym::kSymbolOne;
  for (size_t dim = 0UL; dim < rank; ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(index.strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue) {
      info.index_varying_extent = info.index_varying_extent * index.sizes[dim];
    }
  }
  // The index is invariant on the payload axes when proven from IL's view or from source
  // Load/Data views reached through the x2 producer subgraph (pointwise outputs can be
  // dense even when their values repeat along a payload axis).
  const bool index_has_zero_stride_on_inner_axes = HasZeroStrideOnInputPayload(input, index, axis_index) ||
                                                   IsIndexPayloadInvariantFromSources(node, input, index, axis_index);
  const bool input_axis_stride_is_positive =
      af::SymbolicUtils::StaticCheckGt(input.strides[axis_index], af::sym::kSymbolZero) == af::TriBool::kTrue;
  // An Embedding-like access must have a non-empty payload suffix after the
  // gather axis.  Without this guard, a last-axis gather vacuously satisfies
  // the zero-stride/contiguous-payload checks and is misclassified as
  // Embedding-like, although it has no payload to copy as a unit.  The suffix
  // must also carry real data: when every suffix dimension of the input is
  // zero-stride (broadcast), the payload-invariance proof is vacuous and the
  // index may legitimately vary along the inner axes, which the Embedding
  // address policy cannot express.
  bool input_suffix_has_payload = false;
  af::Expression input_axis_dense_stride;
  IsContiguousSuffixImpl(input, axis_index, true, input_suffix_has_payload, input_axis_dense_stride);
  const bool has_payload_suffix = axis_index + 1UL < rank && input_suffix_has_payload;
  const bool embedding_like = has_payload_suffix && index_has_zero_stride_on_inner_axes &&
                              IsContiguousPayloadSuffix(input, axis_index) && IsContiguousSuffix(output, axis_index) &&
                              input_axis_stride_is_positive;
  info.kind = embedding_like ? IndirectLoadAccessInfo::Kind::kEmbeddingLike : IndirectLoadAccessInfo::Kind::kGeneric;
  info.can_use_simt_structured = embedding_like;
  info.can_use_simd_embedding = IsSimdEmbeddingAccess(logical_view, info);
  // [SK窗口可行性] 源头收敛：视图层证据（inner 轴零 stride）或生产链回溯的源头
  // 视图证据。源头收敛时 gather 引用成组（按行/列），SK 行窗口只需装载被引用
  // 分片；源头不收敛（逐元素独立引用）时 SK 窗口必须覆盖 axis 维全部，退化为
  // 全量。尾轴 gather（无 payload 后缀）等 kind 为 generic 但源头收敛的形态
  // 由此字段正确区分。
  info.index_source_converged =
      index_has_zero_stride_on_inner_axes || IsIndexPayloadInvariantFromSources(node, input, index, axis_index);
  return af::SUCCESS;
}

namespace {
using NodePath = std::vector<af::AscNodePtr>;
using NodeSet = std::unordered_set<const af::AscNode *>;

bool IsSimtDirectGmBoundary(const af::AscNodePtr &node) {
  return af::ops::IsOps<af::ascir_op::Load>(node) ||
         (af::ops::IsOps<af::ascir_op::Nddma>(node) && GetTemplateBehavior(node).uses_direct_gm_pipeline);
}

bool SimtLoadUsesZeroOffset(const af::AscNodePtr &node) {
  if (node == nullptr || node->outputs().empty()) {
    return false;
  }
  const auto &attr = node->outputs()[0]->attr;
  const bool all_zero_strides =
      !attr.strides.empty() && std::all_of(attr.strides.begin(), attr.strides.end(), [](const af::Expression &stride) {
        return af::SymbolicUtils::StaticCheckEq(stride, af::ops::Zero) == af::TriBool::kTrue;
      });
  const bool single_element_shape =
      !attr.repeats.empty() && std::all_of(attr.repeats.begin(), attr.repeats.end(), [](const af::Expression &size) {
        return af::SymbolicUtils::StaticCheckEq(size, af::ops::One) == af::TriBool::kTrue;
      });
  return all_zero_strides || single_element_shape;
}

bool SimtLoadViewsMatch(const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
  if (lhs == nullptr || rhs == nullptr || lhs->outputs().empty() || rhs->outputs().empty()) {
    return false;
  }
  const auto &lhs_attr = lhs->outputs()[0]->attr;
  const auto &rhs_attr = rhs->outputs()[0]->attr;
  if (lhs_attr.repeats.empty() || lhs_attr.strides.empty() || lhs_attr.repeats.size() != rhs_attr.repeats.size() ||
      lhs_attr.strides.size() != rhs_attr.strides.size()) {
    return false;
  }
  for (size_t dim = 0UL; dim < lhs_attr.repeats.size(); ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(lhs_attr.repeats[dim], rhs_attr.repeats[dim]) != af::TriBool::kTrue ||
        af::SymbolicUtils::StaticCheckEq(lhs_attr.strides[dim], rhs_attr.strides[dim]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

af::Status GetStableGraphNodes(const af::AscNodePtr &node, NodePath &nodes) {
  const auto owner_graph = node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph, "IndirectLoad node has no owner graph.");
  for (const auto &graph_node : owner_graph->GetDirectNode()) {
    const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(graph_node);
    GE_ASSERT_NOTNULL(asc_node, "IndirectLoad graph contains invalid node.");
    nodes.emplace_back(asc_node);
  }
  return af::SUCCESS;
}

af::Status CollectSimtLoweringBackwardNodes(const af::AscNodePtr &root, const af::AscNodePtr &indirect_load,
                                            NodeSet &nodes) {
  NodePath pending = {root};
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr current = pending[cursor];
    if (current == nullptr || current == indirect_load || !nodes.emplace(current.get()).second) {
      continue;
    }
    // Nddma 无论是否带 SIMT direct-GM 角色，本质都是 GM 搬运节点，标量求值器不能展开其内部：
    // 复合 Norm 普通调度侧输入经 NddmaTemplate 合并 Load→Broadcast 后即为无角色的队列搬运
    // Nddma，越过它会走到 Data 图输入端子。Data 按定义无生产者，同样是合法的 walk 终止边界。
    if (IsSimtDirectGmBoundary(current) || af::ops::IsOps<af::ascir_op::Nddma>(current) ||
        af::ops::IsOps<af::ascir_op::Data>(current) || af::ops::IsOps<af::ascir_op::Scalar>(current) ||
        af::ops::IsOps<af::ascir_op::ScalarData>(current) || af::ops::IsOps<af::ascir_op::Arange>(current)) {
      continue;
    }
    GE_ASSERT_TRUE(current->inputs.Size() > 0UL, "IndirectLoad SIMT node[%s] has no input.", current->GetNamePtr());
    for (size_t i = 0UL; i < current->inputs.Size(); ++i) {
      const auto producer = GetInputProducer(current, i);
      GE_ASSERT_NOTNULL(producer, "IndirectLoad SIMT node[%s] input[%zu] has no producer.", current->GetNamePtr(), i);
      pending.emplace_back(producer);
    }
  }
  return af::SUCCESS;
}

size_t GetProducerOutputIndex(const af::AscNodePtr &consumer) {
  const auto input_anchor = consumer == nullptr ? nullptr : consumer->GetInDataAnchor(0UL);
  const auto peer_anchor = input_anchor == nullptr ? nullptr : input_anchor->GetPeerOutAnchor();
  return peer_anchor == nullptr ? 0UL : static_cast<size_t>(peer_anchor->GetIdx());
}

struct SimtOutputChainBuild {
  SimtOutputChainMetadata metadata;
  NodeSet nodes;
};

af::Status CollectSimtOutputDescendants(const af::AscNodePtr &indirect_load, NodeSet &descendants) {
  NodePath pending;
  for (const auto &out_node : indirect_load->GetOutDataNodes()) {
    const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
    GE_ASSERT_NOTNULL(consumer, "IndirectLoad SIMT output successor is invalid.");
    pending.emplace_back(consumer);
  }
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const auto &node = pending[cursor];
    if (!descendants.emplace(node.get()).second || af::ops::IsOps<af::ascir_op::Store>(node)) {
      continue;
    }
    for (const auto &out_node : node->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      GE_ASSERT_NOTNULL(consumer, "IndirectLoad SIMT output successor is invalid.");
      pending.emplace_back(consumer);
    }
  }
  return af::SUCCESS;
}

af::Status CollectSimtOutputChainBuilds(const NodePath &graph_nodes, const af::AscNodePtr &indirect_load,
                                        std::vector<SimtOutputChainBuild> &chains) {
  NodeSet descendants;
  GE_ASSERT_SUCCESS(CollectSimtOutputDescendants(indirect_load, descendants));
  std::unordered_map<const af::AscNode *, size_t> cached_node_sets;
  for (const af::AscNodePtr &node : graph_nodes) {
    if (descendants.count(node.get()) == 0UL || !af::ops::IsOps<af::ascir_op::Store>(node)) {
      continue;
    }
    const auto producer = GetInputProducer(node, 0UL);
    GE_ASSERT_NOTNULL(producer, "IndirectLoad SIMT output terminal[%s] has no producer.", node->GetNamePtr());
    const size_t producer_output_index = GetProducerOutputIndex(node);
    GE_ASSERT_TRUE(producer_output_index < producer->outputs().size(),
                   "IndirectLoad SIMT terminal[%s] producer output index[%zu] is invalid.", node->GetNamePtr(),
                   producer_output_index);
    SimtOutputChainBuild chain;
    const auto cached = cached_node_sets.find(producer.get());
    if (cached == cached_node_sets.end()) {
      GE_ASSERT_SUCCESS(CollectSimtLoweringBackwardNodes(producer, indirect_load, chain.nodes));
      cached_node_sets.emplace(producer.get(), chains.size());
    } else {
      chain.nodes = chains[cached->second].nodes;
    }
    chain.metadata.result_tensor_id = producer->outputs()[producer_output_index]->attr.mem.tensor_id;
    chain.metadata.target_tensor_id = node->outputs()[0]->attr.mem.tensor_id;
    chain.metadata.dtype = producer->outputs()[producer_output_index]->attr.dtype;
    const auto &target_attr = node->outputs()[0]->attr;
    chain.metadata.target_sizes = target_attr.repeats;
    chain.metadata.target_strides = target_attr.strides;
    af::Expression store_offset = af::ops::Zero;
    if (node->attr.ir_attr != nullptr) {
      (void)node->attr.ir_attr->GetAttrValue("offset", store_offset);
    }
    chain.metadata.target_offset = store_offset;
    chains.emplace_back(std::move(chain));
  }
  return af::SUCCESS;
}

// [index 链外部 store] index 生产链尾部张量除喂给 IndirectLoad 的 index 输入外，
// 还可能存在区域外直连 store（调度侧已校验唯一合法外部消费形态并将其纳入融合
// 区域）。该 store 收为额外输出链：nodes 恒为空，链值在 codegen 侧由 Index()
// 求值器结果注入；与值侧收集器按 target 去重，避免合流 store 双链双写。
af::Status CollectSimtIndexChainStoreBuilds(const NodePath &graph_nodes, const af::AscNodePtr &index_root,
                                            ascir::TensorId index_result_tensor_id,
                                            std::vector<SimtOutputChainBuild> &chains) {
  if (index_root == nullptr) {
    return af::SUCCESS;
  }
  std::unordered_set<ascir::TensorId> collected_targets;
  for (const auto &chain : chains) {
    collected_targets.emplace(chain.metadata.target_tensor_id);
  }
  for (const af::AscNodePtr &node : graph_nodes) {
    if (!af::ops::IsOps<af::ascir_op::Store>(node)) {
      continue;
    }
    const auto producer = GetInputProducer(node, 0UL);
    if (producer != index_root) {
      continue;
    }
    const size_t producer_output_index = GetProducerOutputIndex(node);
    GE_ASSERT_TRUE(producer_output_index < producer->outputs().size(),
                   "IndirectLoad SIMT index terminal[%s] producer output index[%zu] is invalid.", node->GetNamePtr(),
                   producer_output_index);
    const ascir::TensorId result_tensor_id = producer->outputs()[producer_output_index]->attr.mem.tensor_id;
    const ascir::TensorId target_tensor_id = node->outputs()[0]->attr.mem.tensor_id;
    if (result_tensor_id != index_result_tensor_id || collected_targets.count(target_tensor_id) != 0UL) {
      continue;
    }
    SimtOutputChainBuild chain;
    chain.metadata.result_tensor_id = result_tensor_id;
    chain.metadata.target_tensor_id = target_tensor_id;
    chain.metadata.dtype = producer->outputs()[producer_output_index]->attr.dtype;
    chains.emplace_back(std::move(chain));
    collected_targets.emplace(target_tensor_id);
  }
  return af::SUCCESS;
}

af::Status ValidateSimtLoweringNode(const af::AscNodePtr &node) {
  if (IsSimtDirectGmBoundary(node) || af::ops::IsOps<af::ascir_op::Scalar>(node) ||
      af::ops::IsOps<af::ascir_op::ScalarData>(node) || af::ops::IsOps<af::ascir_op::Arange>(node) ||
      af::ops::IsOps<af::ascir_op::Store>(node) || af::ops::IsOps<af::ascir_op::Transpose>(node)) {
    return af::SUCCESS;
  }
  const auto role = GetTemplateRole(node);
  GE_ASSERT_TRUE(role == TemplateRole::kSimtInlineTransform || role == TemplateRole::kSimtFanoutBranch,
                 "IndirectLoad SIMT node[%s] has no scalar evaluator role.", node->GetNamePtr());
  GE_ASSERT_TRUE(!af::ops::IsOps<af::ascir_op::VectorFunc>(node),
                 "IndirectLoad SIMT transform must use scalar emission, node:%s", node->GetNamePtr());
  return af::SUCCESS;
}

bool TryBuildStaticSpan(const af::Expression &size_expr, uint64_t stride, uint64_t &span) {
  int64_t size = 0L;
  if (!size_expr.GetConstValue(size) || size < 0L) {
    return false;
  }
  return !ge::MulOverflow(static_cast<uint64_t>(size), stride, span);
}

bool TryAccumulateStaticOffset(const af::Expression &size_expr, const af::Expression &stride_expr, uint64_t &offset) {
  int64_t size = 0L;
  int64_t stride = 0L;
  if (!size_expr.GetConstValue(size) || size <= 0L || !stride_expr.GetConstValue(stride) || stride < 0L) {
    return false;
  }
  uint64_t dim_offset = 0U;
  return !ge::MulOverflow(static_cast<uint64_t>(size - 1L), static_cast<uint64_t>(stride), dim_offset) &&
         !ge::AddOverflow(offset, dim_offset, offset);
}

bool TryGetStaticSpans(const LogicalTensorView &input, const LogicalTensorView &output, size_t axis,
                       SimtPolicyMetadata &policy) {
  uint64_t inner = 1U;
  for (size_t i = axis + 1U; i < output.sizes.size(); ++i) {
    if (!TryBuildStaticSpan(output.sizes[i], inner, inner)) {
      return false;
    }
  }
  int64_t input_stride = 0L;
  if (!input.strides[axis].GetConstValue(input_stride) || input_stride < 0L ||
      !TryBuildStaticSpan(output.sizes[axis], inner, policy.output_axis_span) ||
      !TryBuildStaticSpan(input.sizes[axis], static_cast<uint64_t>(input_stride), policy.input_axis_span)) {
    return false;
  }
  policy.inner_span = inner;
  policy.input_axis_stride = static_cast<uint64_t>(input_stride);
  return true;
}

bool TryGetMaxElementOffset(const LogicalTensorView &tensor, uint64_t &max_offset) {
  max_offset = 0U;
  for (size_t i = 0U; i < tensor.sizes.size(); ++i) {
    if (!TryAccumulateStaticOffset(tensor.sizes[i], tensor.strides[i], max_offset)) {
      return false;
    }
  }
  return true;
}

bool IsDense(const LogicalTensorView &tensor) {
  af::Expression expected_stride = af::ops::One;
  for (size_t i = tensor.sizes.size(); i > 0U; --i) {
    const size_t dim = i - 1U;
    if (af::SymbolicUtils::StaticCheckEq(tensor.strides[dim], expected_stride) != af::TriBool::kTrue) {
      return false;
    }
    expected_stride = af::sym::Mul(expected_stride, tensor.sizes[dim]);
  }
  return true;
}

bool IsStructuredSimt(const LogicalTensorView &input, const LogicalTensorView &index, const LogicalTensorView &output,
                      size_t axis) {
  if (!IsDense(input) || !IsDense(index) || !IsDense(output) || index.sizes.size() != output.sizes.size() ||
      input.sizes.size() != output.sizes.size()) {
    return false;
  }
  for (size_t i = 0U; i < output.sizes.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(index.sizes[i], output.sizes[i]) != af::TriBool::kTrue ||
        (i != axis && af::SymbolicUtils::StaticCheckEq(input.sizes[i], output.sizes[i]) != af::TriBool::kTrue)) {
      return false;
    }
  }
  return true;
}

bool IsPowerOfTwo(uint64_t value) {
  return value != 0U && (value & (value - 1U)) == 0U;
}

bool CanUseUint32Offsets(const LogicalTensorView &input, const LogicalTensorView &index,
                         const LogicalTensorView &output) {
  uint64_t input_max = 0U;
  uint64_t index_max = 0U;
  uint64_t output_max = 0U;
  const uint64_t limit = std::numeric_limits<uint32_t>::max();
  return TryGetMaxElementOffset(input, input_max) && input_max <= limit && TryGetMaxElementOffset(index, index_max) &&
         index_max <= limit && TryGetMaxElementOffset(output, output_max) && output_max <= limit;
}

// Decomposed store addressing only engages when the target view mirrors the logical output view
// (rank and sizes equal); every other target view keeps legacy linear addressing (offset +
// output_index), whose extent the dense logical output view already bounds. Merged (ApplyMerge)
// or split store views therefore only need the static offset check, while mirroring views must
// prove their full static span fits the offset width.
bool StoreTargetsFitUint32(const std::vector<SimtOutputChainMetadata> &chains, const LogicalTensorView &output) {
  const uint64_t limit = std::numeric_limits<uint32_t>::max();
  uint64_t output_max = 0U;
  if (!TryGetMaxElementOffset(output, output_max)) {
    return false;
  }
  for (const SimtOutputChainMetadata &chain : chains) {
    if (chain.local_target) {
      continue;
    }
    int64_t offset = 0L;
    if (!chain.target_offset.GetConstValue(offset) || offset < 0L) {
      return false;
    }
    uint64_t span = static_cast<uint64_t>(offset);
    if (span > limit) {
      return false;
    }
    if (chain.target_sizes.size() != output.sizes.size() || chain.target_strides.size() != output.sizes.size() ||
        chain.target_sizes != output.sizes) {
      if (output_max + span > limit) {
        return false;
      }
      continue;
    }
    for (size_t dim = 0UL; dim < chain.target_sizes.size(); ++dim) {
      if (!TryAccumulateStaticOffset(chain.target_sizes[dim], chain.target_strides[dim], span) || span > limit) {
        return false;
      }
    }
  }
  return true;
}

bool CanUseUint32Divisors(const LogicalTensorView &index) {
  for (const af::Expression &size_expr : index.sizes) {
    int64_t size = 0L;
    if (!size_expr.GetConstValue(size) || size < 0L || size > static_cast<int64_t>(INT32_MAX)) {
      return false;
    }
  }
  return true;
}

uint64_t BuildNonZeroStrideMask(const std::vector<af::Expression> &strides) {
  uint64_t mask = 0U;
  for (size_t dim = 0U; dim < strides.size() && dim < 64UL; ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue) {
      mask |= 1ULL << dim;
    }
  }
  return mask;
}

bool SimtPhysicalStridesEqual(const std::vector<af::Expression> &lhs, const std::vector<af::Expression> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return std::equal(lhs.begin(), lhs.end(), rhs.begin(), [](const af::Expression &left, const af::Expression &right) {
    return af::SymbolicUtils::StaticCheckEq(left, right) == af::TriBool::kTrue;
  });
}

bool SimtHasZeroStride(const std::vector<af::Expression> &strides) {
  return std::any_of(strides.begin(), strides.end(), [](const af::Expression &stride) {
    return af::SymbolicUtils::StaticCheckEq(stride, af::ops::Zero) == af::TriBool::kTrue;
  });
}

bool ApplySimtIndexPhysicalStrides(const NodePath &index_nodes, LogicalTensorView &index, bool &mixed_views) {
  mixed_views = false;
  const std::vector<af::Expression> *canonical_strides = nullptr;
  bool has_zero_stride = false;
  for (const af::AscNodePtr &node : index_nodes) {
    if (!IsSimtDirectGmBoundary(node) || node->outputs().empty() ||
        node->outputs()[0]->attr.strides.size() != index.strides.size()) {
      continue;
    }
    const auto &strides = node->outputs()[0]->attr.strides;
    if (canonical_strides == nullptr) {
      canonical_strides = &strides;
    } else if (!SimtPhysicalStridesEqual(*canonical_strides, strides)) {
      mixed_views = true;
    }
    has_zero_stride = has_zero_stride || SimtHasZeroStride(strides);
  }
  if (canonical_strides == nullptr) {
    for (const af::AscNodePtr &node : index_nodes) {
      const std::vector<af::Expression> *physical_strides = nullptr;
      if (af::ops::IsOps<af::ascir_op::Broadcast>(node) && !node->inputs().empty() &&
          node->inputs()[0]->attr.strides.size() == index.strides.size()) {
        physical_strides = &node->inputs()[0]->attr.strides;
      } else if (!node->outputs().empty() && node->outputs()[0]->attr.strides.size() == index.strides.size()) {
        physical_strides = &node->outputs()[0]->attr.strides;
      }
      if (physical_strides != nullptr && SimtHasZeroStride(*physical_strides)) {
        canonical_strides = physical_strides;
        has_zero_stride = true;
        break;
      }
    }
  }
  if (canonical_strides != nullptr && has_zero_stride && !mixed_views) {
    index.strides = *canonical_strides;
  }
  return has_zero_stride || mixed_views;
}

void AppendRuntimeParams(std::vector<af::Expression> &target, const std::vector<af::Expression> &source) {
  target.insert(target.end(), source.begin(), source.end());
}

af::Status BuildSimtPolicyMetadata(const TemplateLogicalView &logical_view, const NodePath &index_nodes, size_t axis,
                                   bool embedding_structured, bool &mixed_index_views, bool store_targets_fit_uint32,
                                   SimtPolicyMetadata &metadata, LogicalTensorView &policy_index_view) {
  const size_t rank = logical_view.input.sizes.size();
  GE_ASSERT_TRUE(rank < 64UL, "IndirectLoad SIMT rank must be smaller than 64.");
  LogicalTensorView input = logical_view.input;
  LogicalTensorView index = logical_view.index;
  const auto &output = logical_view.output;
  const bool index_broadcast_strided = ApplySimtIndexPhysicalStrides(index_nodes, index, mixed_index_views);
  policy_index_view = index;
  const bool strided = logical_view.input.kind != IndirectLoadLayoutKind::kDense ||
                       logical_view.index.kind != IndirectLoadLayoutKind::kDense || index_broadcast_strided;
  af::Expression inner_span = af::sym::kSymbolOne;
  for (size_t i = axis + 1U; i < output.sizes.size(); ++i) {
    inner_span = af::sym::Mul(inner_span, output.sizes[i]);
  }
  const af::Expression output_axis_span = af::sym::Mul(output.sizes[axis], inner_span);
  const af::Expression input_axis_stride = input.strides[axis];
  const af::Expression input_axis_span = af::sym::Mul(input.sizes[axis], input_axis_stride);
  const bool structured = IsStructuredSimt(input, index, output, axis);
  const bool static_spans = TryGetStaticSpans(input, output, axis, metadata);
  if (embedding_structured) {
    metadata.policy = SimtAddressPolicy::kEmbedding;
    metadata.input_stride_mask = BuildNonZeroStrideMask(input.strides);
    metadata.index_stride_mask = BuildNonZeroStrideMask(index.strides);
    if (rank == 2UL && axis == 0UL) {
      metadata.input_stride_mask = 0U;
      metadata.index_stride_mask = 0U;
    }
  } else if (strided) {
    metadata.policy = SimtAddressPolicy::kStrided;
    metadata.input_stride_mask = BuildNonZeroStrideMask(input.strides);
    metadata.index_stride_mask = BuildNonZeroStrideMask(index.strides);
  } else if (structured) {
    const bool static_power_of_two =
        static_spans && IsPowerOfTwo(metadata.inner_span) && IsPowerOfTwo(metadata.output_axis_span);
    const bool static_inner = static_spans && IsPowerOfTwo(metadata.inner_span);
    const bool magic_divisors_supported =
        !static_spans || (metadata.inner_span <= static_cast<uint64_t>(INT64_MAX) &&
                          metadata.output_axis_span <= static_cast<uint64_t>(INT64_MAX));
    metadata.policy = static_power_of_two        ? SimtAddressPolicy::kStaticPowerOfTwo
                      : static_inner             ? SimtAddressPolicy::kStaticInner
                      : magic_divisors_supported ? SimtAddressPolicy::kStructuredMagic
                                                 : SimtAddressPolicy::kRecursive;
  }
  const bool structured_magic_supported = metadata.policy != SimtAddressPolicy::kStructuredMagic ||
                                          (static_spans && metadata.inner_span <= static_cast<uint64_t>(INT32_MAX) &&
                                           metadata.output_axis_span <= static_cast<uint64_t>(INT32_MAX));
  const bool static_inner_magic_supported = metadata.policy != SimtAddressPolicy::kStaticInner ||
                                            metadata.output_axis_span <= static_cast<uint64_t>(INT32_MAX);
  const bool general_magic_supported =
      (metadata.policy != SimtAddressPolicy::kRecursive && metadata.policy != SimtAddressPolicy::kStrided &&
       metadata.policy != SimtAddressPolicy::kEmbedding) ||
      CanUseUint32Divisors(index);
  if (CanUseUint32Offsets(input, index, output) && store_targets_fit_uint32 && structured_magic_supported &&
      static_inner_magic_supported && general_magic_supported) {
    metadata.offset_width = SimtOffsetWidth::kUint32;
  }
  switch (metadata.policy) {
    case SimtAddressPolicy::kStaticPowerOfTwo:
      break;
    case SimtAddressPolicy::kStaticInner:
      metadata.runtime_params.emplace_back(output_axis_span);
      break;
    case SimtAddressPolicy::kStructuredMagic:
      metadata.runtime_params = {inner_span, output_axis_span, input_axis_stride, input_axis_span};
      break;
    case SimtAddressPolicy::kEmbedding:
      AppendRuntimeParams(metadata.runtime_params, index.sizes);
      AppendRuntimeParams(metadata.runtime_params, input.strides);
      AppendRuntimeParams(metadata.runtime_params, index.strides);
      break;
    case SimtAddressPolicy::kRecursive:
      AppendRuntimeParams(metadata.runtime_params, index.sizes);
      AppendRuntimeParams(metadata.runtime_params, input.strides);
      break;
    case SimtAddressPolicy::kStrided:
      AppendRuntimeParams(metadata.runtime_params, index.sizes);
      AppendRuntimeParams(metadata.runtime_params, input.strides);
      AppendRuntimeParams(metadata.runtime_params, index.strides);
      break;
  }
  return af::SUCCESS;
}

LogicalTensorView GetNodeOutputView(const af::AscNodePtr &node) {
  const auto &attr = node->outputs()[0]->attr;
  return {attr.axis, attr.repeats, attr.strides};
}

void AppendSimtGmTensorMetadata(const af::AscNodePtr &node, SimtLoweringMetadata &metadata,
                                std::unordered_set<ascir::TensorId> &seen_tensor_ids) {
  const auto output = node->outputs()[0];
  if (!seen_tensor_ids.emplace(output->attr.mem.tensor_id).second) {
    return;
  }
  if (IsSimtDirectGmBoundary(node)) {
    metadata.gm_tensors.push_back(
        {output->attr.mem.tensor_id, node->inputs()[0]->attr.mem.tensor_id, output->attr.dtype, false});
  } else {
    metadata.gm_tensors.push_back({output->attr.mem.tensor_id, output->attr.mem.tensor_id, output->attr.dtype, true});
  }
}

af::Status ValidateSimtOutputPlan(const SimtLoweringMetadata &metadata) {
  GE_ASSERT_TRUE(!metadata.output_chains.empty(), "IndirectLoad SIMT output chains are empty.");
  const size_t local_target_count =
      static_cast<size_t>(std::count_if(metadata.output_chains.begin(), metadata.output_chains.end(),
                                        [](const SimtOutputChainMetadata &chain) { return chain.local_target; }));
  GE_ASSERT_TRUE(local_target_count <= 1UL, "IndirectLoad SIMT supports at most one local output target.");
  // 单 gather：post Reduce 与 local target 必须成对出现；多 gather 组员：value 直接写 UB 局部
  // 目标（归并/输出链为普通向量代码），允许 local target 不伴随 post Reduce。
  GE_ASSERT_TRUE(!metadata.has_post_reduce || local_target_count == 1UL,
                 "IndirectLoad SIMT post Reduce requires a local output target.");
  if (metadata.has_post_reduce) {
    const auto &local_chain = metadata.output_chains.back();
    GE_ASSERT_TRUE(local_chain.local_target && local_chain.result_tensor_id == metadata.output_result_tensor_id &&
                       local_chain.target_tensor_id == metadata.output_result_tensor_id &&
                       local_chain.node_names == metadata.output_node_names,
                   "IndirectLoad SIMT local output must be the synthetic post Reduce chain.");
  }
  return af::SUCCESS;
}

af::Status BuildSimtLoweringMetadata(const af::AscNodePtr &indirect_load, IndirectLoadLoweringMetadata &metadata) {
  NodePath graph_nodes;
  GE_ASSERT_SUCCESS(GetStableGraphNodes(indirect_load, graph_nodes));
  auto &simt = metadata.simt;
  // 多 gather 组员：SIMT 计划仅含本路 index 链与 input 读取；共享归并/输出链不进入计划
  // （普通向量代码在全部组员 invocation 之后执行），value 直接写 UB 局部目标。
  const bool is_group_member = IsSimtGroupMember(indirect_load);
  const af::AscNodePtr post_reduce = is_group_member ? nullptr : GetPostReduceConsumer(indirect_load);
  simt.has_post_reduce = post_reduce != nullptr;
  const auto inputs = indirect_load->inputs();
  GE_ASSERT_TRUE(inputs.size() == 2UL, "Invalid IndirectLoad SIMT input number:%zu.", inputs.size());
  simt.index_result_tensor_id = inputs[kIndexTensorIndex]->attr.mem.tensor_id;
  simt.index_dtype = inputs[kIndexTensorIndex]->attr.dtype;
  simt.value_tensor_id = indirect_load->outputs()[0]->attr.mem.tensor_id;

  const auto index_root = GetInputProducer(indirect_load, kIndexTensorIndex);
  GE_ASSERT_NOTNULL(index_root, "IndirectLoad SIMT index input has no producer.");
  NodeSet index_set;
  GE_ASSERT_SUCCESS(CollectSimtLoweringBackwardNodes(index_root, indirect_load, index_set));
  NodeSet output_set;
  af::AscNodePtr output_root;
  if (simt.has_post_reduce) {
    output_root = GetPostReduceInputProducer(indirect_load);
    GE_ASSERT_NOTNULL(output_root, "IndirectLoad post Reduce input has no producer.");
    GE_ASSERT_SUCCESS(CollectSimtLoweringBackwardNodes(output_root, indirect_load, output_set));
  }

  std::vector<SimtOutputChainBuild> chain_builds;
  if (is_group_member) {
    // 组员：仅收 index 生产链尾部的外部 store 链（per-IL，无共享）；不收集 IL 前向输出链。
    GE_ASSERT_SUCCESS(
        CollectSimtIndexChainStoreBuilds(graph_nodes, index_root, simt.index_result_tensor_id, chain_builds));
  } else {
    GE_ASSERT_SUCCESS(CollectSimtOutputChainBuilds(graph_nodes, indirect_load, chain_builds));
    GE_ASSERT_SUCCESS(
        CollectSimtIndexChainStoreBuilds(graph_nodes, index_root, simt.index_result_tensor_id, chain_builds));
  }
  std::vector<bool> keep_chain(chain_builds.size(), true);
  for (size_t chain_index = 0UL; chain_index < chain_builds.size(); ++chain_index) {
    const bool contains_reduce =
        simt.has_post_reduce && std::any_of(chain_builds[chain_index].nodes.begin(),
                                            chain_builds[chain_index].nodes.end(), [](const af::AscNode *node) {
                                              return node != nullptr &&
                                                     node->attr.api.compute_type == af::ComputeType::kComputeReduce;
                                            });
    keep_chain[chain_index] = !contains_reduce;
  }

  NodePath index_nodes;
  NodePath output_load_nodes;
  std::unordered_set<ascir::TensorId> seen_gm_tensors;
  for (const af::AscNodePtr &node : graph_nodes) {
    const bool in_index = index_set.count(node.get()) != 0UL;
    const bool in_output_region = output_set.count(node.get()) != 0UL;
    bool in_output_chain = false;
    for (size_t chain_index = 0UL; chain_index < chain_builds.size(); ++chain_index) {
      if (keep_chain[chain_index] && chain_builds[chain_index].nodes.count(node.get()) != 0UL) {
        chain_builds[chain_index].metadata.node_names.emplace_back(node->GetName());
        in_output_chain = true;
      }
    }
    if (!in_index && !in_output_region && !in_output_chain) {
      continue;
    }
    GE_ASSERT_SUCCESS(ValidateSimtLoweringNode(node));
    if (in_index) {
      index_nodes.emplace_back(node);
      simt.index_node_names.emplace_back(node->GetName());
    }
    if (in_output_region || in_output_chain) {
      simt.output_node_names.emplace_back(node->GetName());
    }
    if ((in_output_region || in_output_chain) && IsSimtDirectGmBoundary(node)) {
      output_load_nodes.emplace_back(node);
    }
    if (IsSimtDirectGmBoundary(node) || af::ops::IsOps<af::ascir_op::ScalarData>(node)) {
      AppendSimtGmTensorMetadata(node, simt, seen_gm_tensors);
    }
  }
  for (size_t chain_index = 0UL; chain_index < chain_builds.size(); ++chain_index) {
    if (keep_chain[chain_index]) {
      simt.output_chains.emplace_back(std::move(chain_builds[chain_index].metadata));
    }
  }
  if (simt.has_post_reduce) {
    GE_ASSERT_TRUE(!output_root->outputs().empty(), "IndirectLoad post Reduce input producer has no output.");
    simt.output_result_tensor_id = output_root->outputs()[0]->attr.mem.tensor_id;
    simt.output_dtype = output_root->outputs()[0]->attr.dtype;
    simt.output_chains.push_back({simt.output_node_names,
                                  simt.output_result_tensor_id,
                                  simt.output_result_tensor_id,
                                  simt.output_dtype,
                                  true,
                                  {},
                                  {},
                                  af::ops::Zero});
  } else if (is_group_member) {
    // 组员输出 = value 直写 UB 局部目标：空变换链，evaluator 的 value 种子直接落 target
    // （与 post Reduce 合成链同构，见 GenerateSimtOutputsEvaluator 的 values[value_tensor_id]）。
    simt.output_result_tensor_id = simt.value_tensor_id;
    simt.output_dtype = indirect_load->outputs()[0]->attr.dtype;
    simt.output_chains.push_back({std::vector<std::string>{},
                                  simt.value_tensor_id,
                                  simt.value_tensor_id,
                                  simt.output_dtype,
                                  true,
                                  {},
                                  {},
                                  af::ops::Zero});
  } else {
    GE_ASSERT_TRUE(!simt.output_chains.empty(), "IndirectLoad SIMT output chains are empty.");
    if (simt.output_chains.size() == 1UL) {
      simt.output_result_tensor_id = simt.output_chains[0].result_tensor_id;
      simt.output_node_names = simt.output_chains[0].node_names;
    }
    simt.output_dtype = simt.output_chains.front().dtype;
  }

  NodePath index_load_nodes;
  for (const auto &node : index_nodes) {
    if (IsSimtDirectGmBoundary(node)) {
      index_load_nodes.emplace_back(node);
    }
  }
  std::unordered_map<std::string, SimtLoadAddressSource> output_sources;
  for (const auto &node : output_load_nodes) {
    auto source = SimtLoadAddressSource::kOutputOffset;
    if (SimtLoadUsesZeroOffset(node)) {
      source = SimtLoadAddressSource::kZeroOffset;
    } else if (std::any_of(index_load_nodes.begin(), index_load_nodes.end(), [&node](const af::AscNodePtr &index_load) {
                 return SimtLoadViewsMatch(node, index_load);
               })) {
      source = SimtLoadAddressSource::kIndexOffset;
    }
    output_sources.emplace(node->GetName(), source);
  }

  bool mixed_index_views = false;
  const bool store_targets_fit_uint32 = StoreTargetsFitUint32(simt.output_chains, metadata.logical_view.output);
  LogicalTensorView policy_index_view;
  GE_ASSERT_SUCCESS(BuildSimtPolicyMetadata(metadata.logical_view, index_nodes, static_cast<size_t>(metadata.axis),
                                            metadata.access_info.can_use_simt_structured, mixed_index_views,
                                            store_targets_fit_uint32, simt.policy, policy_index_view));
  for (const auto &node : index_load_nodes) {
    const LogicalTensorView current_view = GetNodeOutputView(node);
    SimtLoadMetadata load_meta;
    load_meta.node_name = node->GetName();
    load_meta.address_source =
        SimtLoadUsesZeroOffset(node) ? SimtLoadAddressSource::kZeroOffset : SimtLoadAddressSource::kOutputOffset;
    // 唯一 index Load 通常可直接使用按其物理 view 计算的 index_offset。对于 dense policy
    // 下的非 dense 物理 view，或多个 index Load，仍必须按各自 view 做 logical offset
    // 坐标折叠，避免丢失行 stride 或将主 Load 的偏移用于 side-input。
    const bool policy_index_dense = IsDense(policy_index_view);
    load_meta.use_logical_offset =
        index_load_nodes.size() != 1UL || mixed_index_views ||
        (policy_index_dense && !IsDenseEquivalentView(current_view, metadata.logical_view.output));
    load_meta.physical_view = current_view;
    // [行级广播 side-input] 尾轴零贡献（stride==0 且 size==1）的广播形态：记原始
    // 视图（此时已过 NormalizeTemplateAxes 的轴改写，若视图已符号化则再取一次不可
    // 恢复——改写前的语义在 CompletePreservedVectorizedViews 之前才完整，但该函数
    // 在 metadata 构建前运行；此处能取到的是改写后视图，行级广播判定按其结构特征
    // （尾轴零贡献 + 其余轴稠密）识别，坐标重建不依赖具体尺寸值——见 codegen 兜底）。
    // 改写后视图无法可靠判定（split 破坏结构特征）——读 generator 在视图改写前
    // 写入的节点 attr 标记。
    load_meta.is_row_broadcast = ascir::IsRowBroadcastLoad(*node);
    load_meta.original_view = load_meta.physical_view;
    simt.index_loads.push_back(std::move(load_meta));
  }
  // Every GM side load inside the SIMT region is addressed with the full logical output_index,
  // including post-Reduce regions, so its physical view must always provide the coordinate
  // folding; a raw output_index is only valid for dense-matching views.
  for (const auto &node : output_load_nodes) {
    SimtLoadMetadata output_load_meta;
    output_load_meta.node_name = node->GetName();
    output_load_meta.address_source = output_sources.at(node->GetName());
    output_load_meta.use_logical_offset = true;
    output_load_meta.physical_view = GetNodeOutputView(node);
    output_load_meta.original_view = output_load_meta.physical_view;
    // [行级广播] 输出侧 GM load 同样读 attr 标记（SIMT body 内联融合算子的
    // side-input 归属 output evaluator，报错栈走 simt_output_loads_）。
    output_load_meta.is_row_broadcast = ascir::IsRowBroadcastLoad(*node);
    simt.output_loads.push_back(std::move(output_load_meta));
  }
  return af::SUCCESS;
}
}  // namespace

// [SK窗口可行性] 现场判定 index 源头收敛性（不依赖候选流程构建的 AccessInfo attr——
// attr 写在先跑的 SIMD/SIMT 候选图副本上，SK 候选副本从原图拷贝时为空，跨副本读取
// 会把收敛形态误判为非收敛而误触发重定向）。视图直接取 IL 两个输入生产者的输出
// 视图，与 HasZeroStrideOnInputPayload / IsIndexPayloadInvariantFromSources 同源。
bool IsIndexSourceConverged(const af::AscNodePtr &indirect_load) {
  if (indirect_load == nullptr) {
    return false;
  }
  const auto input_producer = GetInputProducer(indirect_load, kInputTensorIndex);
  const auto index_producer = GetInputProducer(indirect_load, kIndexTensorIndex);
  if (input_producer == nullptr || index_producer == nullptr) {
    return false;
  }
  const auto input = GetNodeOutputView(input_producer);
  const auto index = GetNodeOutputView(index_producer);
  int64_t axis = -1L;
  // 注意 GetAttrValue 返回 graphStatus（成功为 0），不能以 !status 判定失败。
  if (indirect_load->attr.ir_attr == nullptr || indirect_load->attr.ir_attr->GetAttrValue("axis", axis) != 0 ||
      axis < 0L) {
    return false;
  }
  const size_t axis_index = static_cast<size_t>(axis);
  return HasZeroStrideOnInputPayload(input, index, axis_index) ||
         IsIndexPayloadInvariantFromSources(indirect_load, input, index, axis_index);
}

af::Status FinalizeLoweringMetadata(const af::AscNodePtr &node, bool &is_supported) {
  is_supported = false;
  GE_ASSERT_NOTNULL(node);
  const auto template_id = ::ascir::GetTemplateIdOrDefault(*node);
  GE_ASSERT_TRUE(
      template_id == ::ascir::TemplateId::kIndirectLoadSimd || template_id == ::ascir::TemplateId::kIndirectLoadSimt,
      "IndirectLoad lowering metadata only supports SIMD/SIMT, node = %s", node->GetNamePtr());
  IndirectLoadLoweringMetadata metadata;
  GE_ASSERT_SUCCESS(GetTemplateLogicalView(node, metadata.logical_view));
  GE_ASSERT_SUCCESS(AnalyzeIndirectLoadAccess(node, metadata.logical_view, metadata.access_info));
  TemplateAxes axes;
  GE_ASSERT_SUCCESS(GetTemplateAxes(node, axes));
  metadata.axis = metadata.access_info.axis;
  metadata.outer_axis = axes.outer_axis;
  Implementation implementation;
  GE_ASSERT_SUCCESS(GetImplementation(node, implementation));
  if (template_id == ::ascir::TemplateId::kIndirectLoadSimd) {
    // [padded 输出窗口强制 strided] 通用对齐把输出视图尾轴 pad 到对齐块（如 [32行,
    // 4有效+4空洞] 的行 stride 8）时，dense facade（RegGather）线性稠密写出与视图
    // 布局不符——下游（Reduce 按 first×行跨度读取、VF 按视图 strides 访问）会把
    // 稠密数据按 strided 行解释，数值错乱。strided facade 按输出视图 strides 逐行
    // 写出，与 padded 视图自洽；行内 pad 区由 ReduceInit 的 OptImpl（inner_r 非对齐
    // 路径）清中性值。判定：输出视图物理跨度（Σ(size-1)*stride+1）大于逻辑元素积
    // （Πsize）即存在 pad。
    // [padded 判定源] 必须用输出 tensor 的 vectorized_strides（通用对齐改写的向量化
    // 视图，Tiler::TensorActualSize 的 actual_size 公式同源）而非 logical_view 的
    // strides（图原生视图，可能仍为稠密 (4,1)，而对齐已把 vectorized_strides 改写
    // 为 (8,1)——生产 gather+sum 即此：判定读 logical 层会漏判）。
    const auto &simd_out_attr = node->outputs()[0]->attr;
    af::Expression out_span = af::sym::kSymbolOne;
    af::Expression out_product = af::sym::kSymbolOne;
    bool out_view_valid = simd_out_attr.vectorized_axis.size() == simd_out_attr.vectorized_strides.size() &&
                          !simd_out_attr.vectorized_axis.empty();
    if (out_view_valid) {
      for (size_t dim = 0UL; dim < simd_out_attr.vectorized_axis.size(); ++dim) {
        const auto axis_it =
            std::find(simd_out_attr.axis.begin(), simd_out_attr.axis.end(), simd_out_attr.vectorized_axis[dim]);
        if (axis_it == simd_out_attr.axis.end()) {
          out_view_valid = false;
          break;
        }
        const size_t axis_pos = static_cast<size_t>(std::distance(simd_out_attr.axis.begin(), axis_it));
        const auto &vec_stride = simd_out_attr.vectorized_strides[dim];
        if (af::SymbolicUtils::StaticCheckEq(vec_stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
          continue;
        }
        out_span = out_span + (simd_out_attr.repeats[axis_pos] - af::sym::kSymbolOne) * vec_stride;
        out_product = out_product * simd_out_attr.repeats[axis_pos];
      }
    }
    const bool out_padded = out_view_valid && simd_out_attr.vectorized_axis.size() > 1UL &&
                            af::SymbolicUtils::StaticCheckGt(out_span, out_product) == af::TriBool::kTrue;
    GELOGI("[IndirectLoad] SIMD fallback pre-check at lowering: node[%s] span[%s] product[%s] padded[%d].",
           node->GetNamePtr(), out_span.Str().get(), out_product.Str().get(), static_cast<int>(out_padded));
    const bool strided = metadata.logical_view.input.kind != IndirectLoadLayoutKind::kDense ||
                         metadata.logical_view.index.kind != IndirectLoadLayoutKind::kDense || out_padded;
    metadata.simd.fallback = strided                                        ? SimdFallback::kStrided
                             : implementation == Implementation::kGatherApi ? SimdFallback::kGatherApi
                                                                            : SimdFallback::kRegisterGather;
    metadata.simd.try_embedding =
        implementation == Implementation::kDefault && metadata.access_info.can_use_simd_embedding;
  } else {
    if (metadata.logical_view.input.sizes.size() >= 64UL) {
      GELOGI("[IndirectLoad] Reject SIMT lowering metadata for node[%s]: rank[%zu] must be smaller than 64.",
             node->GetNamePtr(), metadata.logical_view.input.sizes.size());
      return af::SUCCESS;
    }
    GE_ASSERT_SUCCESS(BuildSimtLoweringMetadata(node, metadata));
    GE_ASSERT_SUCCESS(ValidateSimtOutputPlan(metadata.simt));
    // [SIMT index 闭包预计算] 调度期判定并 stash scratch 需求：CalcTmpBufSize（impl
    // 注册库）只读 ext attr，不跨库回调本库符号；codegen 期用同一 builder 再判一次
    // 决定是否发射（两处图副本结构漂移时以 tmp 缺失/白分配为安全方向）。
    SimtIndexPrepassPlan prepass_plan;
    GE_ASSERT_SUCCESS(BuildSimtIndexPrepassPlan(*node, prepass_plan));
    GE_ASSERT_SUCCESS(SetSimtIndexPrepassTmpDemand(node, prepass_plan));
  }
  GE_ASSERT_SUCCESS(SetIndirectLoadAccessInfo(node, metadata.access_info));
  GE_ASSERT_SUCCESS(SetLoweringMetadata(node, metadata));
  is_supported = true;
  return af::SUCCESS;
}

bool IsDenseEquivalentView(const LogicalTensorView &load_view, const LogicalTensorView &output_view) {
  if (load_view.sizes.size() != load_view.strides.size() || output_view.sizes.size() != output_view.strides.size()) {
    return false;
  }
  af::Expression load_total = af::sym::kSymbolOne;
  af::Expression expected_stride = af::sym::kSymbolOne;
  bool has_dense_tail = false;
  for (size_t rev = 0UL; rev < load_view.sizes.size(); ++rev) {
    const size_t dim = load_view.sizes.size() - 1UL - rev;
    const bool unit_size = af::SymbolicUtils::StaticCheckEq(load_view.sizes[dim], af::ops::One) == af::TriBool::kTrue;
    const bool zero_stride =
        af::SymbolicUtils::StaticCheckEq(load_view.strides[dim], af::sym::kSymbolZero) == af::TriBool::kTrue;
    if (unit_size || zero_stride) {
      continue;  // 退化轴不参与连续性与计数
    }
    if (!has_dense_tail) {
      // 最右侧有效轴必须 stride==1
      if (af::SymbolicUtils::StaticCheckEq(load_view.strides[dim], af::ops::One) != af::TriBool::kTrue) {
        return false;
      }
      has_dense_tail = true;
    } else if (af::SymbolicUtils::StaticCheckEq(load_view.strides[dim], expected_stride) != af::TriBool::kTrue) {
      return false;
    }
    expected_stride = af::sym::Mul(load_view.sizes[dim], expected_stride);
    load_total = af::sym::Mul(load_view.sizes[dim], load_total);
  }
  af::Expression output_total = af::sym::kSymbolOne;
  for (const auto &size : output_view.sizes) {
    output_total = af::sym::Mul(size, output_total);
  }
  return af::SymbolicUtils::StaticCheckEq(load_total, output_total) == af::TriBool::kTrue;
}

af::Status ClassifyIndirectLoadLayout(const LogicalTensorView &logical, IndirectLoadTensorLayout &layout,
                                      bool allow_non_overlapping_zero_stride) {
  GE_ASSERT_TRUE(IsValidLogicalTensorView(logical), "IndirectLoad input layout rank is invalid.");
  static_cast<LogicalTensorView &>(layout) = logical;
  layout.kind = IndirectLoadLayoutKind::kUnsupported;
  layout.physical_repeats = logical.sizes;

  if (TryClassifyDynamicShapeLayout(logical, layout)) {
    return af::SUCCESS;
  }
  af::Expression physical_span = af::sym::kSymbolOne;
  bool has_zero_stride = false;
  bool has_physical_gap = false;
  for (size_t index = logical.sizes.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    const auto dim_kind = ClassifyTensorDim(logical.sizes[dim], logical.strides[dim]);
    if (dim_kind == TensorDimKind::kIllegal) {
      return af::SUCCESS;
    }
    if (dim_kind == TensorDimKind::kZeroStride) {
      layout.physical_repeats[dim] = af::sym::kSymbolOne;
      has_zero_stride = true;
      continue;
    }
    if (af::SymbolicUtils::StaticCheckLt(logical.strides[dim], physical_span) != af::TriBool::kFalse) {
      // 行主序单调性证明失败，尝试按转置稠密（倍数链）证明非重叠；证不出维持 kUnsupported。
      static_cast<void>(TryClassifyTransposedDenseLayout(logical, layout));
      return af::SUCCESS;
    }
    has_physical_gap =
        has_physical_gap || af::SymbolicUtils::StaticCheckEq(logical.strides[dim], physical_span) != af::TriBool::kTrue;
    physical_span = physical_span + (logical.sizes[dim] - af::sym::kSymbolOne) * logical.strides[dim];
  }
  if (has_zero_stride && has_physical_gap) {
    if (allow_non_overlapping_zero_stride) {
      layout.kind = IndirectLoadLayoutKind::kStrided;
      layout.physical_repeats = logical.sizes;
    }
    return af::SUCCESS;
  }
  layout.kind = has_zero_stride
                    ? IndirectLoadLayoutKind::kZeroStrideCompact
                    : (has_physical_gap ? IndirectLoadLayoutKind::kStrided : IndirectLoadLayoutKind::kDense);
  return af::SUCCESS;
}

af::Status ValidateIndirectLoadOutputLayout(const LogicalTensorView &output) {
  GE_ASSERT_TRUE(IsValidLogicalTensorView(output), "IndirectLoad output layout rank is invalid.");
  af::Expression expected_stride = af::sym::kSymbolOne;
  for (size_t index = output.sizes.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    if (af::SymbolicUtils::StaticCheckEq(output.sizes[dim], af::sym::kSymbolOne) == af::TriBool::kTrue) {
      continue;
    }
    GE_ASSERT_TRUE(af::SymbolicUtils::StaticCheckEq(output.strides[dim], expected_stride) == af::TriBool::kTrue,
                   "IndirectLoad output must use a dense contiguous layout.");
    expected_stride = af::sym::Mul(expected_stride, output.sizes[dim]);
  }
  return af::SUCCESS;
}

// Classifies the layout of an IndirectLoad output Store target.  The supported
// non-dense form mirrors a ConcatKernel slice view: a contiguous suffix starting
// at the innermost dim, arbitrary positive static strides on the prefix dims, and
// a provably non-negative static offset.
// Every stride (including dims skipped below) must be provably positive.
af::Status ClassifyIndirectLoadStoreTarget(const std::vector<af::Expression> &sizes,
                                           const std::vector<af::Expression> &strides, const af::Expression &offset,
                                           IndirectLoadStoreKind &kind) {
  GE_ASSERT_TRUE(!sizes.empty() && sizes.size() == strides.size(),
                 "IndirectLoad store target rank is invalid, sizes:%zu, strides:%zu.", sizes.size(), strides.size());
  kind = IndirectLoadStoreKind::kUnsupported;
  if (af::SymbolicUtils::StaticCheckGe(offset, af::sym::kSymbolZero) != af::TriBool::kTrue) {
    return af::SUCCESS;
  }
  for (const af::Expression &stride : strides) {
    if (af::SymbolicUtils::StaticCheckGt(stride, af::sym::kSymbolZero) != af::TriBool::kTrue) {
      return af::SUCCESS;
    }
  }
  af::Expression expected_stride = af::sym::kSymbolOne;
  bool has_contiguous_suffix = false;
  for (size_t index = sizes.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    if (af::SymbolicUtils::StaticCheckEq(sizes[dim], af::sym::kSymbolOne) == af::TriBool::kTrue) {
      continue;
    }
    if (!has_contiguous_suffix) {
      if (af::SymbolicUtils::StaticCheckEq(strides[dim], expected_stride) != af::TriBool::kTrue) {
        return af::SUCCESS;
      }
      has_contiguous_suffix = true;
      expected_stride = af::sym::Mul(expected_stride, sizes[dim]);
      continue;
    }
    if (af::SymbolicUtils::StaticCheckEq(strides[dim], expected_stride) == af::TriBool::kTrue) {
      expected_stride = af::sym::Mul(expected_stride, sizes[dim]);
      continue;
    }
    kind = IndirectLoadStoreKind::kSliceStrided;
    return af::SUCCESS;
  }
  kind = IndirectLoadStoreKind::kDense;
  return af::SUCCESS;
}

af::Status SetImplementation(const af::AscNodePtr &node, Implementation implementation) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kImplementationAttr, static_cast<int64_t>(implementation)),
                 "Set IndirectLoad implementation failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

af::Status GetImplementation(const af::AscNodePtr &node, Implementation &implementation) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  implementation = static_cast<Implementation>(op_desc->TryGetExtAttr(kImplementationAttr, -1L));
  GE_ASSERT_TRUE(implementation == Implementation::kDefault || implementation == Implementation::kGatherApi,
                 "IndirectLoad implementation is missing or invalid, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

bool ShouldApplyInputInnerVectorization(const af::AscNodePtr &node) {
  const TemplateRole role = GetTemplateRole(node);
  return role == TemplateRole::kSimdInputPre || role == TemplateRole::kSimdInputPreStridedUbPath ||
         role == TemplateRole::kSkInputBoundary;
}

af::AscNodePtr GetInputProducer(const af::AscNodePtr &node, size_t input_index) {
  auto input_anchor = node == nullptr ? nullptr : node->GetInDataAnchor(input_index);
  if (input_anchor == nullptr || input_anchor->GetPeerOutAnchor() == nullptr) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<af::AscNode>(input_anchor->GetPeerOutAnchor()->GetOwnerNode());
}

namespace {
std::vector<af::AscNodePtr> CollectIndirectLoadNodes(const af::AscGraph &graph) {
  std::vector<af::AscNodePtr> nodes;
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::IndirectLoad>(node)) {
      nodes.push_back(node);
    }
  }
  return nodes;
}
}  // namespace

af::AscNodePtr FindIndirectLoadNode(const af::AscGraph &graph) {
  const auto nodes = CollectIndirectLoadNodes(graph);
  return nodes.empty() ? nullptr : nodes.front();
}

std::vector<af::AscNodePtr> GetAllIndirectLoadNodes(const af::AscGraph &graph) {
  return CollectIndirectLoadNodes(graph);
}

af::Status SetSimdInputInnerAxis(const af::AscNodePtr &node, af::AxisId axis) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kSimdInputInnerAxisHintAttr, static_cast<int64_t>(axis)),
                 "Set IndirectLoad SIMD input inner axis hint failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

af::AxisId GetSimdInputInnerAxis(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return af::kIdNone;
  }
  return node->GetOpDesc()->TryGetExtAttr(kSimdInputInnerAxisHintAttr, static_cast<int64_t>(af::kIdNone));
}

af::Status SetSimtGroupMember(const af::AscNodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kSimtGroupMemberAttr, true),
                 "Set IndirectLoad SIMT group member flag failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

bool IsSimtGroupMember(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return false;
  }
  return node->GetOpDesc()->TryGetExtAttr(kSimtGroupMemberAttr, false);
}

constexpr char kSimtFusedPrimaryAttr[] = "af.internal.indirect_load.simt_fused_primary";
constexpr char kSimtFusedMembersAttr[] = "af.internal.indirect_load.simt_fused_members";

af::Status SetSimtFusedPrimary(const af::AscNodePtr &node, const std::vector<std::string> &member_names) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kSimtFusedPrimaryAttr, true),
                 "Set IndirectLoad SIMT fused primary flag failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kSimtFusedMembersAttr, member_names),
                 "Set IndirectLoad SIMT fused member names failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

bool IsSimtFusedPrimary(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return false;
  }
  return node->GetOpDesc()->TryGetExtAttr(kSimtFusedPrimaryAttr, false);
}

std::vector<std::string> GetSimtFusedMemberNames(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return {};
  }
  return node->GetOpDesc()->TryGetExtAttr(kSimtFusedMembersAttr, std::vector<std::string>{});
}

constexpr char kSimtFusedMergeInfoAttr[] = "af.internal.indirect_load.simt_fused_merge_info";

af::Status SetSimtFusedMergeInfo(const af::AscNodePtr &node, const SimtFusedMergeInfo &info) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kSimtFusedMergeInfoAttr, info),
                 "Set IndirectLoad SIMT fused merge info failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

SimtFusedMergeInfo GetSimtFusedMergeInfo(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return {};
  }
  return node->GetOpDesc()->TryGetExtAttr(kSimtFusedMergeInfoAttr, SimtFusedMergeInfo{});
}

constexpr char kSimtFusedMemberEmittedAttr[] = "af.internal.indirect_load.simt_fused_member_emitted";

af::Status SetSimtFusedMemberEmitted(const af::AscNodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kSimtFusedMemberEmittedAttr, true),
                 "Set IndirectLoad SIMT fused member emitted flag failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

bool IsSimtFusedMemberEmitted(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return false;
  }
  return node->GetOpDesc()->TryGetExtAttr(kSimtFusedMemberEmittedAttr, false);
}

bool IsSimtDirectGmLoadNode(const af::AscNodePtr &node) {
  return IsSimtDirectGmBoundary(node);
}

af::Status BuildSimtMergeSideLoadMetadata(const af::AscNodePtr &load, SimtLoadMetadata &metadata) {
  metadata = SimtLoadMetadata{};
  GE_ASSERT_NOTNULL(load);
  GE_ASSERT_TRUE(!load->outputs().empty() && load->outputs()[0] != nullptr,
                 "IndirectLoad SIMT merge side load[%s] has no output.", load->GetNamePtr());
  // 与单 gather 输出侧 load 同构的分类（kZeroOffset / kOutputOffset）。视图匹配成员 index
  // load 的 ② 形态无需特判：use_logical_offset 恒为 true，寻址按自身物理视图从 output_index
  // 坐标折叠——分类值在折叠路径下不被消费（单 gather 输出侧 load 同此），不存在 N 路
  // index_offset 歧义。
  if (SimtLoadUsesZeroOffset(load)) {
    metadata.address_source = SimtLoadAddressSource::kZeroOffset;
  } else {
    metadata.address_source = SimtLoadAddressSource::kOutputOffset;
  }
  metadata.node_name = load->GetName();
  // 输出侧 load 恒走逻辑偏移坐标折叠（与单 gather 输出侧一致）：物理 view 承载 strided/
  // 行广播/② 形态的正确寻址，行广播标记读改写前写入的节点 attr（跨漂移稳定）。
  metadata.use_logical_offset = true;
  metadata.physical_view = GetNodeOutputView(load);
  metadata.original_view = metadata.physical_view;
  metadata.is_row_broadcast = ascir::IsRowBroadcastLoad(*load);
  return af::SUCCESS;
}

af::Status ValidateSingleIndirectLoadNode(const af::AscGraph &graph, af::AscNodePtr &node) {
  node = nullptr;
  const auto nodes = CollectIndirectLoadNodes(graph);
  GE_ASSERT_TRUE(nodes.size() <= 1UL,
                 "[IndirectLoad] Graph[%s] contains multiple IndirectLoad nodes, first[%s], next[%s].",
                 graph.GetName().c_str(), nodes.empty() ? "<null>" : nodes[0]->GetNamePtr(),
                 nodes.size() < 2UL ? "<null>" : nodes[1]->GetNamePtr());
  node = nodes.empty() ? nullptr : nodes.front();
  if (node != nullptr) {
    GELOGD("[IndirectLoad] Graph[%s] found IndirectLoad node[%s].", graph.GetName().c_str(), node->GetNamePtr());
  }
  return af::SUCCESS;
}

bool GetPrebuiltYTilingCase(const af::AscGraph &graph, af::AxisId &tile_id,
                            std::pair<af::AxisPtr, af::AxisPtr> &tiling) {
  tile_id = af::kIdNone;
  tiling = {nullptr, nullptr};

  const af::AscNodePtr indirect_load = FindIndirectLoadNode(graph);
  if (indirect_load == nullptr) {
    return false;
  }

  const TemplateAxes axes = ReadTemplateAxes(indirect_load);
  if (axes.outer_axis == af::kIdNone) {
    // 模板候选全部被拒时节点无模板标注：节点存在不等于模板接管，此时不存在可预建
    // 的 tile 轴。返回 false 让调用方走通用 tiling case 生成，避免 kIdNone 毒化
    // PruneTilingCase 的轴查找（axis_id:-1 断言）。
    return false;
  }
  tile_id = axes.outer_axis;
  const auto all_axes = graph.GetAllAxis();
  if (axes.tile_outer_axis >= 0L && static_cast<size_t>(axes.tile_outer_axis) < all_axes.size()) {
    tiling.first = all_axes[static_cast<size_t>(axes.tile_outer_axis)];
  }
  if (axes.tile_inner_axis >= 0L && static_cast<size_t>(axes.tile_inner_axis) < all_axes.size()) {
    tiling.second = all_axes[static_cast<size_t>(axes.tile_inner_axis)];
  }
  return true;
}

// ===== [SIMT index 闭包预计算] =====
// 预计算可行性判定与发射计划构建。判定为纯图结构推导，调度期（tmp 尺寸规划，
// IndirectLoadAscIrCodegenImplV2::CalcTmpBufSize）与 codegen 期（pre-pass 发射）共
// 用同一位点，避免两处判定位漂移导致 scratch 缺口或白分配。
namespace {
// 值域区间：双精度端点 + 未知标记。无穷端点仅可能来自 Ln(0) 类退化输入（引用
// int64 语义在该路径同样产生整型极值哨兵，最终由 Select 语义丢弃——与逐元素
// 路径同一契约），有限端点必须落入 int32 才能写入索引表。
struct PrepassRange {
  double lo = 0.0;
  double hi = 0.0;
  bool unknown = false;
  bool has_inf = false;
};

PrepassRange MakeScalarRange(double value) {
  return {value, value, false, false};
}

// int32 索引表值域的校验端点（double 承载，按 fp32 槽重放约束取值）：下界 -2^31
// （INT32_MIN，fp32 可精确表示）；上界 2^31-128 是不超过 INT32_MAX 的最大可表示
// fp32——fp32 在 [2^30, 2^31) 的尾数粒度为 128，更大的值会舍入到 2^31 后溢出。
constexpr double kPrepassInt32DomainLo = -2147483648.0;
constexpr double kPrepassInt32DomainHi = 2147483520.0;
// fp32 逐位精确整数界限：|v| ≤ 2^24 时整数在 24 位尾数（含隐含位）下精确。
constexpr double kPrepassFp32ExactIntLimit = 16777216.0;

// 标量字面量解析为 double：要求整串消费（与 att 侧 stod 判定同模式）；不可解析
// 或超出 double 表示域时按数值不可用处理，由调用方按各自口径回退。
bool ParsePrepassScalarDouble(const std::string &text, double &value) {
  try {
    size_t parsed = 0UL;
    value = std::stod(text, &parsed);
    return parsed == text.size();
  } catch (...) {
    return false;
  }
}

bool RangeFitsInt32(const PrepassRange &range) {
  return !range.unknown && range.lo >= kPrepassInt32DomainLo && range.hi <= kPrepassInt32DomainHi;
}

// int 值物化为 fp32 的精确性检查点：int64 算术在 fp32 槽内重放必须逐位可表示。
bool RangeIsExactFp32(const PrepassRange &range) {
  return !range.unknown && !range.has_inf && range.lo >= -kPrepassFp32ExactIntLimit &&
         range.hi <= kPrepassFp32ExactIntLimit;
}

PrepassRange RangeBinary(const PrepassRange &lhs, const PrepassRange &rhs, bool is_mul) {
  if (lhs.unknown || rhs.unknown) {
    return {0.0, 0.0, true, false};
  }
  if (is_mul) {
    const double products[4] = {lhs.lo * rhs.lo, lhs.lo * rhs.hi, lhs.hi * rhs.lo, lhs.hi * rhs.hi};
    return {*std::min_element(products, products + 4), *std::max_element(products, products + 4), false,
            lhs.has_inf || rhs.has_inf};
  }
  return {lhs.lo + rhs.lo, lhs.hi + rhs.hi, false, lhs.has_inf || rhs.has_inf};
}

bool IsPrepassFloatDtype(af::DataType dtype) {
  return dtype == af::DT_FLOAT;
}

bool IsPrepassIntDtype(af::DataType dtype) {
  return dtype == af::DT_BOOL || dtype == af::DT_INT8 || dtype == af::DT_INT16 || dtype == af::DT_INT32 ||
         dtype == af::DT_INT64 || dtype == af::DT_UINT8 || dtype == af::DT_UINT16 || dtype == af::DT_UINT32 ||
         dtype == af::DT_UINT64;
}

// 预计算 scratch 槽位需求上限：4B 槽（fp32 值、int32 ramp、截断往返、Where 标量
// 分支物化均计入峰值）与 uint8 比较 mask 槽。超限形态回退逐元素路径。
constexpr size_t kPrepassMaxFourByteSlots = 6UL;
constexpr size_t kPrepassMaxMaskSlots = 2UL;
constexpr size_t kPrepassInvalidSlot = static_cast<size_t>(-1);

// 沿透传/同域 Cast 回溯 Where 掩码的比较来源；非比较来源不可作掩码。
size_t ResolvePrepassMaskOrigin(const std::vector<SimtIndexPrepassNodePlan> &nodes, size_t node_index) {
  size_t cursor = node_index;
  for (size_t hop = 0UL; hop < nodes.size() && cursor < nodes.size(); ++hop) {
    const auto &node = nodes[cursor];
    if (node.kind == SimtIndexPrepassNodeKind::kOpCompare) {
      return cursor;
    }
    if ((node.kind == SimtIndexPrepassNodeKind::kPassthrough || node.kind == SimtIndexPrepassNodeKind::kCastNoop) &&
        !node.operands.empty() && node.operands.front().is_node) {
      cursor = node.operands.front().node_index;
      continue;
    }
    break;
  }
  return nodes.size();
}

bool PrepassNodeOwnsFourByteSlot(SimtIndexPrepassNodeKind kind) {
  return kind == SimtIndexPrepassNodeKind::kLeafArangeInnerRamp ||
         kind == SimtIndexPrepassNodeKind::kLeafArangeRowConstant || kind == SimtIndexPrepassNodeKind::kOpBinary ||
         kind == SimtIndexPrepassNodeKind::kOpUnary || kind == SimtIndexPrepassNodeKind::kOpWhere ||
         kind == SimtIndexPrepassNodeKind::kCastTruncToInt;
}
}  // namespace

af::Status BuildSimtIndexPrepassPlan(const af::AscNode &indirect_load, SimtIndexPrepassPlan &plan) {
  plan = SimtIndexPrepassPlan{};
  // 按节点引用归位 owner 图内的 AscNodePtr（锚点遍历与闭包收集基于指针接口）。
  const auto owner_graph = indirect_load.GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph, "IndirectLoad node[%s] has no owner graph.", indirect_load.GetNamePtr());
  af::AscNodePtr indirect_load_ptr;
  for (const auto &candidate : owner_graph->GetAllNodes()) {
    const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(candidate);
    if (asc_node != nullptr && asc_node.get() == std::addressof(indirect_load)) {
      indirect_load_ptr = asc_node;
      break;
    }
  }
  GE_ASSERT_NOTNULL(indirect_load_ptr, "IndirectLoad node[%s] is absent from its owner graph.",
                    indirect_load.GetNamePtr());
  if (!af::ops::IsOps<af::ascir_op::IndirectLoad>(indirect_load_ptr)) {
    return af::SUCCESS;
  }
  const auto il_inputs = indirect_load_ptr->inputs();
  if (il_inputs.size() != 2UL || il_inputs[kInputTensorIndex] == nullptr) {
    return af::SUCCESS;
  }
  // 索引表复用 value tensor 存储（int32 视图）：值 dtype 小于 4B 时表宽超出存储。
  if (af::GetSizeByDataType(il_inputs[kInputTensorIndex]->attr.dtype) < 4) {
    return af::SUCCESS;
  }
  TemplateLogicalView view;
  GE_ASSERT_SUCCESS(GetTemplateLogicalView(indirect_load_ptr, view));
  if (view.output.sizes.empty() || view.output.strides.empty() || view.output.axis_ids.empty() ||
      view.output.sizes.size() != view.output.strides.size() ||
      view.output.sizes.size() != view.output.axis_ids.size()) {
    return af::SUCCESS;
  }
  // 尾轴尺寸需编译期可知（Arange 值域与行常量坐标推导依赖常量边界）。
  int64_t inner_const = 0L;
  if (!view.output.sizes.back().GetConstValue(inner_const) || inner_const <= 0L) {
    return af::SUCCESS;
  }
  const af::AscNodePtr index_root = GetInputProducer(indirect_load_ptr, kIndexTensorIndex);
  if (index_root == nullptr) {
    return af::SUCCESS;
  }
  NodeSet index_set;
  GE_ASSERT_SUCCESS(CollectSimtLoweringBackwardNodes(index_root, indirect_load_ptr, index_set));
  NodePath graph_nodes;
  GE_ASSERT_SUCCESS(GetStableGraphNodes(indirect_load_ptr, graph_nodes));
  std::vector<af::AscNodePtr> chain;
  for (const af::AscNodePtr &node : graph_nodes) {
    if (node != nullptr && index_set.count(node.get()) != 0UL) {
      chain.emplace_back(node);
    }
  }
  if (chain.empty()) {
    return af::SUCCESS;
  }

  auto reject = [&plan](const char *reason, const af::AscNodePtr &node) {
    GELOGI("[IndirectLoad] SIMT index prepass is ineligible (%s), node[%s, %s].", reason,
           node == nullptr ? "<null>" : node->GetTypePtr(), node == nullptr ? "<null>" : node->GetNamePtr());
    plan = SimtIndexPrepassPlan{};
    return af::SUCCESS;
  };

  plan.nodes.resize(chain.size());
  std::vector<PrepassRange> ranges(chain.size());
  // 死亡期下界为自身：值至少存活到自身节点完成；被消费时用 max 抬升。若初始化为
  // chain.size()，max 更新永远无法下调，所有槽将被视为活到链尾（峰值虚高）。
  std::vector<size_t> last_consumer(chain.size());
  for (size_t i = 0UL; i < chain.size(); ++i) {
    last_consumer[i] = i;
  }
  // 操作数解析按生产者节点指针而非 tensor_id：调度期图的 tensor_id 尚未赋值（全部
  // 解析会命中同一默认值），而 SIMT 元数据收集同样基于节点可达性，指针语义与其
  // 一致；链外单输入视图节点（折叠 broadcast 中继）沿输入上溯到链内等价值。
  std::map<const af::AscNode *, size_t> producer_index;
  for (size_t i = 0UL; i < chain.size(); ++i) {
    producer_index.emplace(chain[i].get(), i);
  }
  auto is_view_relay = [](const af::AscNodePtr &n) {
    return n != nullptr && n->inputs.Size() == 1UL &&
           (af::ops::IsOps<af::ascir_op::Store>(n) || af::ops::IsOps<af::ascir_op::Transpose>(n) ||
            af::ops::IsOps<af::ascir_op::Broadcast>(n));
  };
  // 解析 node 的第 input_pos 个输入：返回 0=链内节点 / 1=链外常量标量 / -1=不可解。
  auto resolve_input = [&producer_index, &is_view_relay](const af::AscNodePtr &node, size_t input_pos,
                                                         SimtIndexPrepassOperand &operand) -> int {
    af::AscNodePtr cur = GetInputProducer(node, input_pos);
    for (size_t hops = 0UL; cur != nullptr && hops < 8UL; ++hops) {
      const auto found = producer_index.find(cur.get());
      if (found != producer_index.end()) {
        operand = SimtIndexPrepassOperand{true, found->second, ""};
        return 0;
      }
      if (af::ops::IsOps<af::ascir_op::Scalar>(cur)) {
        std::string value;
        std::string dtype_name;
        if (cur->attr.ir_attr == nullptr || cur->outputs().empty() ||
            cur->attr.ir_attr->GetAttrValue("value", value) != ge::GRAPH_SUCCESS ||
            ascgen_utils::DtypeName(cur->outputs()[0]->attr.dtype, dtype_name) != af::SUCCESS) {
          return -1;
        }
        std::string processed;
        if (ascgen_utils::ScalarValuePreProcess(value, dtype_name, processed) != af::SUCCESS) {
          return -1;
        }
        operand = SimtIndexPrepassOperand{false, 0UL, processed};
        return 1;
      }
      if (!is_view_relay(cur)) {
        return -1;
      }
      cur = GetInputProducer(cur, 0UL);
    }
    return -1;
  };
  for (size_t i = 0UL; i < chain.size(); ++i) {
    const af::AscNodePtr node = chain[i];
    auto &node_plan = plan.nodes[i];
    if (af::ops::IsOps<af::ascir_op::Store>(node) || af::ops::IsOps<af::ascir_op::Transpose>(node) ||
        af::ops::IsOps<af::ascir_op::Broadcast>(node)) {
      // Broadcast 为纯视图复制（零步长轴扩展），逐元素值与输入一致：SIMT 标量发射
      // 同样直接透传输入表达式，此处按透传处理保持两路径一致。
      if (node->inputs.Size() != 1UL || node->inputs().empty()) {
        return reject("passthrough arity", node);
      }
      SimtIndexPrepassOperand relay;
      if (resolve_input(node, 0UL, relay) != 0) {
        return reject("passthrough input is external", node);
      }
      node_plan.kind = SimtIndexPrepassNodeKind::kPassthrough;
      node_plan.operands.emplace_back(relay);
      node_plan.slot = plan.nodes[relay.node_index].slot;
      ranges[i] = ranges[relay.node_index];
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Scalar>(node)) {
      std::string value;
      if (node->attr.ir_attr == nullptr || node->outputs().empty() ||
          node->attr.ir_attr->GetAttrValue("value", value) != ge::GRAPH_SUCCESS) {
        return reject("scalar value attr", node);
      }
      std::string dtype_name;
      GE_ASSERT_SUCCESS(ascgen_utils::DtypeName(node->outputs()[0]->attr.dtype, dtype_name));
      std::string processed;
      GE_ASSERT_SUCCESS(ascgen_utils::ScalarValuePreProcess(value, dtype_name, processed));
      double numeric = 0.0;
      if (!ParsePrepassScalarDouble(processed, numeric)) {
        return reject("scalar value is not numeric", node);
      }
      if (!IsPrepassIntDtype(node->outputs()[0]->attr.dtype) && !IsPrepassFloatDtype(node->outputs()[0]->attr.dtype)) {
        return reject("scalar dtype is not prepass-representable", node);
      }
      node_plan.kind = SimtIndexPrepassNodeKind::kLeafScalar;
      node_plan.operands.emplace_back(SimtIndexPrepassOperand{false, 0UL, processed});
      ranges[i] = MakeScalarRange(numeric);
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::ScalarData>(node) || IsSimtDirectGmBoundary(node)) {
      // 运行期标量与 GM 装载：index 闭包不再是输出坐标的纯函数。
      return reject("index chain is data dependent", node);
    }
    if (af::ops::IsOps<af::ascir_op::Arange>(node)) {
      const auto &out_attr = node->outputs()[0]->attr;
      if (out_attr.axis.size() != out_attr.strides.size() || out_attr.strides.empty()) {
        return reject("arange view metadata", node);
      }
      size_t varying_pos = out_attr.strides.size();
      for (size_t pos = 0UL; pos < out_attr.strides.size(); ++pos) {
        if (af::SymbolicUtils::StaticCheckNe(out_attr.strides[pos], af::sym::kSymbolZero) == af::TriBool::kTrue) {
          varying_pos = pos;
          break;
        }
      }
      if (varying_pos == out_attr.strides.size()) {
        return reject("arange has no varying axis", node);
      }
      const auto axis_iter =
          std::find(view.output.axis_ids.begin(), view.output.axis_ids.end(), out_attr.axis[varying_pos]);
      if (axis_iter == view.output.axis_ids.end()) {
        return reject("arange varying axis is absent from output view", node);
      }
      const size_t logical_pos = static_cast<size_t>(std::distance(view.output.axis_ids.begin(), axis_iter));
      af::Expression base = af::ops::Zero;
      af::Expression step = af::ops::One;
      int64_t base_const = 0L;
      int64_t step_const = 1L;
      int64_t size_const = 0L;
      if (node->attr.ir_attr == nullptr || node->attr.ir_attr->GetAttrValue("base", base) != ge::GRAPH_SUCCESS ||
          node->attr.ir_attr->GetAttrValue("step", step) != ge::GRAPH_SUCCESS || !base.GetConstValue(base_const) ||
          !step.GetConstValue(step_const) || !view.output.sizes[logical_pos].GetConstValue(size_const) ||
          size_const <= 0L) {
        return reject("arange base/step/size is not compile-time", node);
      }
      node_plan.arange.view_stride = view.output.strides[logical_pos];
      node_plan.arange.view_size = view.output.sizes[logical_pos];
      node_plan.arange.base = base;
      node_plan.arange.step = step;
      const bool unit_stride =
          af::SymbolicUtils::StaticCheckEq(view.output.strides[logical_pos], af::sym::kSymbolOne) == af::TriBool::kTrue;
      if (unit_stride && size_const == inner_const) {
        // 尾轴 ramp：tile 基址按整行对齐，坐标退化为列号 ramp [0, inner)。
        node_plan.kind = SimtIndexPrepassNodeKind::kLeafArangeInnerRamp;
      } else {
        // 行常量：stride 整除 inner（行对齐边界）时行内坐标恒定。
        int64_t stride_const = 0L;
        const bool stride_divides =
            view.output.strides[logical_pos].GetConstValue(stride_const) && stride_const % inner_const == 0L;
        const bool stride_equals_inner =
            af::SymbolicUtils::StaticCheckEq(view.output.strides[logical_pos], view.output.sizes.back()) ==
            af::TriBool::kTrue;
        if (!stride_divides && !stride_equals_inner) {
          return reject("arange leaf shape is neither inner ramp nor row constant", node);
        }
        node_plan.kind = SimtIndexPrepassNodeKind::kLeafArangeRowConstant;
      }
      const double lo = static_cast<double>(base_const);
      const double hi = static_cast<double>(base_const) + static_cast<double>(size_const - 1L) * step_const;
      ranges[i] = {std::min(lo, hi), std::max(lo, hi), false, false};
    } else {
      // 通用算子：解析操作数（链内节点、链内/链外常量标量）。
      std::vector<SimtIndexPrepassOperand> operands;
      bool operand_ok = true;
      for (size_t input_pos = 0UL; input_pos < node->inputs.Size(); ++input_pos) {
        SimtIndexPrepassOperand operand;
        const int resolved = resolve_input(node, input_pos, operand);
        if (resolved < 0) {
          const auto producer = GetInputProducer(node, input_pos);
          if (producer != nullptr &&
              (af::ops::IsOps<af::ascir_op::ScalarData>(producer) || IsSimtDirectGmBoundary(producer))) {
            // 运行期标量与 GM 装载：index 闭包不再是输出坐标的纯函数。
            return reject("index chain is data dependent", node);
          }
          operand_ok = false;
          break;
        }
        operands.emplace_back(operand);
      }
      if (!operand_ok) {
        return reject("operand is not chain value or constant", node);
      }
      // Where/Select 掩码语义定位：输入端口序不保证 cond 在首位，按“唯一来源为
      // 比较节点的输入”识别掩码并换至 operands[0]（其余两支保持相对顺序，与
      // [cond, 真值, 假值] 规范序一致）。
      const std::string type = node->GetType();
      if (type == "Where" || type == "Select") {
        size_t mask_pos = operands.size();
        for (size_t pos = 0UL; pos < operands.size(); ++pos) {
          if (operands[pos].is_node &&
              ResolvePrepassMaskOrigin(plan.nodes, operands[pos].node_index) < plan.nodes.size()) {
            if (mask_pos != operands.size()) {
              return reject("where has multiple compare inputs", node);
            }
            mask_pos = pos;
          }
        }
        if (mask_pos == operands.size()) {
          return reject("where mask does not originate from a compare", node);
        }
        std::swap(operands[0UL], operands[mask_pos]);
      }
      node_plan.operands = operands;
      const af::DataType out_dtype = node->outputs().empty() ? af::DT_UNDEFINED : node->outputs()[0]->attr.dtype;
      auto operand_range = [&ranges](const SimtIndexPrepassOperand &operand) -> PrepassRange {
        if (operand.is_node) {
          return ranges[operand.node_index];
        }
        double numeric = 0.0;
        if (!ParsePrepassScalarDouble(operand.scalar_value, numeric)) {
          return {0.0, 0.0, true, false};
        }
        return MakeScalarRange(numeric);
      };
      if (af::ops::IsOps<af::ascir_op::Cast>(node)) {
        if (node->inputs.Size() != 1UL || node->inputs().empty() || node->outputs().empty() ||
            !operands.front().is_node) {
          return reject("cast arity", node);
        }
        const af::DataType in_dtype = node->inputs()[0]->attr.dtype;
        const bool in_float = IsPrepassFloatDtype(in_dtype);
        const bool out_float = IsPrepassFloatDtype(out_dtype);
        if ((!in_float && !IsPrepassIntDtype(in_dtype)) || (!out_float && !IsPrepassIntDtype(out_dtype))) {
          return reject("cast dtype is not prepass-representable", node);
        }
        if (in_float && !out_float) {
          // float→int 截断（CAST_ZERO）：经 int32 槽往返两段 Cast 还原截断语义。
          node_plan.kind = SimtIndexPrepassNodeKind::kCastTruncToInt;
          ranges[i] = ranges[operands.front().node_index];
        } else {
          if (!in_float && out_float && !RangeIsExactFp32(ranges[operands.front().node_index])) {
            // int 值物化 fp32 必须逐位精确，否则预计算改变取值。
            return reject("integer value range is not fp32 exact", node);
          }
          node_plan.kind = SimtIndexPrepassNodeKind::kCastNoop;
          ranges[i] = ranges[operands.front().node_index];
          node_plan.slot = plan.nodes[operands.front().node_index].slot;
        }
        continue;
      }
      if (!IsPrepassIntDtype(out_dtype) && !IsPrepassFloatDtype(out_dtype)) {
        return reject("node dtype is not prepass representable", node);
      }
      if (af::ops::IsOps<af::ascir_op::Abs>(node) || af::ops::IsOps<af::ascir_op::Ln>(node)) {
        if (operands.size() != 1UL || !operands.front().is_node) {
          return reject("unary arity", node);
        }
        node_plan.kind = SimtIndexPrepassNodeKind::kOpUnary;
        node_plan.op_code = af::ops::IsOps<af::ascir_op::Abs>(node) ? 0UL : 1UL;
        const PrepassRange input = ranges[operands.front().node_index];
        if (node_plan.op_code == 0UL) {
          ranges[i] = {0.0, std::max(std::fabs(input.lo), std::fabs(input.hi)), input.unknown, input.has_inf};
        } else {
          if (input.unknown || input.lo < 0.0) {
            return reject("ln input range includes negatives", node);
          }
          PrepassRange result{0.0, 0.0, false, false};
          result.lo = input.lo <= 0.0 ? -std::numeric_limits<double>::infinity() : std::log(input.lo);
          result.hi = input.hi <= 0.0 ? -std::numeric_limits<double>::infinity() : std::log(input.hi);
          result.has_inf = input.lo <= 0.0 || input.has_inf;
          ranges[i] = result;
        }
      } else if (type == "Add" || type == "Sub" || type == "Mul" || type == "Minimum" || type == "Maximum" ||
                 type == "Max") {
        if (operands.size() != 2UL) {
          return reject("binary arity", node);
        }
        node_plan.kind = SimtIndexPrepassNodeKind::kOpBinary;
        node_plan.op_code = type == "Add"       ? 0UL
                            : type == "Sub"     ? 1UL
                            : type == "Mul"     ? 2UL
                            : type == "Minimum" ? 3UL
                                                : 4UL;
        if (node_plan.op_code == 1UL && !operands[0].is_node && operands[1].is_node) {
          // Sub 的被减数为标量：*s 变体无法表达（tensor − scalar），标量物化又引入
          // 额外临时槽，保守回退逐元素路径。
          return reject("sub scalar minuend is not supported", node);
        }
        const PrepassRange lhs = operand_range(operands[0]);
        const PrepassRange rhs = operand_range(operands[1]);
        if (node_plan.op_code == 2UL) {
          ranges[i] = RangeBinary(lhs, rhs, true);
        } else if (node_plan.op_code == 1UL) {
          if (lhs.unknown || rhs.unknown) {
            ranges[i] = {0.0, 0.0, true, false};
          } else {
            ranges[i] = {lhs.lo - rhs.hi, lhs.hi - rhs.lo, false, lhs.has_inf || rhs.has_inf};
          }
        } else if (node_plan.op_code == 0UL) {
          ranges[i] = RangeBinary(lhs, rhs, false);
        } else {
          ranges[i] = lhs;
          if (node_plan.op_code == 3UL) {
            ranges[i].hi = std::min(lhs.hi, rhs.hi);
          } else {
            ranges[i].lo = std::max(lhs.lo, rhs.lo);
          }
          ranges[i].unknown = lhs.unknown || rhs.unknown;
        }
      } else if (type == "Gt" || type == "Ge" || type == "Lt" || type == "Le" || type == "Eq" || type == "Ne") {
        if (operands.size() != 2UL) {
          return reject("compare arity", node);
        }
        node_plan.kind = SimtIndexPrepassNodeKind::kOpCompare;
        node_plan.op_code = type == "Gt"   ? 0UL
                            : type == "Ge" ? 1UL
                            : type == "Lt" ? 2UL
                            : type == "Le" ? 3UL
                            : type == "Eq" ? 4UL
                                           : 5UL;
        ranges[i] = {0.0, 1.0, false, false};
      } else if (type == "Where" || type == "Select") {
        if (operands.size() != 3UL || !operands[0].is_node) {
          return reject("where arity", node);
        }
        node_plan.kind = SimtIndexPrepassNodeKind::kOpWhere;
        const PrepassRange branch0 = operand_range(operands[1]);
        const PrepassRange branch1 = operand_range(operands[2]);
        ranges[i] = {std::min(branch0.lo, branch1.lo), std::max(branch0.hi, branch1.hi),
                     branch0.unknown || branch1.unknown, branch0.has_inf || branch1.has_inf};
      } else {
        return reject("node type is not prepass supported", node);
      }
    }
  }

  // ===== 阶段 B：透传归并与死亡期推导 =====
  // 透传/同域 Cast 值穿透：槽位持有者沿链归并（effective_owner），操作数死亡期按
  // 持有者统计（消费透传输出即消费底层持有者的值）。
  std::vector<size_t> effective_owner(chain.size(), 0UL);
  for (size_t i = 0UL; i < chain.size(); ++i) {
    const auto &current = plan.nodes[i];
    if ((current.kind == SimtIndexPrepassNodeKind::kPassthrough ||
         current.kind == SimtIndexPrepassNodeKind::kCastNoop) &&
        !current.operands.empty() && current.operands.front().is_node) {
      effective_owner[i] = effective_owner[current.operands.front().node_index];
    } else {
      effective_owner[i] = i;
    }
  }
  for (size_t i = 0UL; i < chain.size(); ++i) {
    for (const auto &operand : plan.nodes[i].operands) {
      if (!operand.is_node) {
        continue;
      }
      const size_t owner = effective_owner[operand.node_index];
      last_consumer[owner] = std::max(last_consumer[owner], i);
    }
  }
  // 操作数预解析（发射语义）：沿 effective_owner 归并到值持有者——持有者是标量叶
  // 则直接内联字面量（透传 broadcast 包裹的常量在此摊平），否则改指槽持有者节
  // 点。发射端从此只见到“有效槽读取”或“字面量”两种操作数，无需再理解透传。
  for (size_t i = 0UL; i < chain.size(); ++i) {
    for (auto &operand : plan.nodes[i].operands) {
      if (!operand.is_node) {
        continue;
      }
      const size_t owner = effective_owner[operand.node_index];
      if (plan.nodes[owner].kind == SimtIndexPrepassNodeKind::kLeafScalar) {
        operand.is_node = false;
        operand.node_index = 0UL;
        operand.scalar_value = plan.nodes[owner].operands.front().scalar_value;
      } else {
        operand.node_index = owner;
      }
    }
  }

  // ===== 阶段 C：槽位分配（拓扑序单遍） =====
  // 死亡槽（持有者 last_consumer < 当前节点）先释放；节点内临时槽（int32 ramp /
  // 截断往返 / Where 标量分支物化）获取即释放（记录保留供发射期复用）；输出槽优
  // 先原地复用死亡操作数槽。峰值 = 任意时刻占用槽数的最大值。
  std::vector<size_t> free_slots;
  for (size_t slot = kPrepassMaxFourByteSlots; slot > 0UL; --slot) {
    free_slots.push_back(slot - 1UL);
  }
  std::vector<size_t> free_mask_slots;
  for (size_t slot = kPrepassMaxMaskSlots; slot > 0UL; --slot) {
    free_mask_slots.push_back(slot - 1UL);
  }
  size_t peak_slots = 0UL;
  size_t peak_masks = 0UL;
  // slot_holder[s] = 当前占用槽 s 的链节点：原地复用只转移持有关系，不回写旧持有
  // 者的 slot 字段（发射端仍按旧持有者的 slot 读取操作数值，复用节点在同槽先读后
  // 写）；释放仅当“当前持有者”死亡，避免重复释放。
  std::vector<size_t> slot_holder(kPrepassMaxFourByteSlots, chain.size());
  std::vector<bool> mask_owned(kPrepassMaxMaskSlots, false);
  for (size_t i = 0UL; i < chain.size(); ++i) {
    auto &node_plan = plan.nodes[i];
    for (size_t owner = 0UL; owner < i; ++owner) {
      if (effective_owner[owner] != owner || last_consumer[owner] >= i) {
        continue;
      }
      const auto &owner_plan = plan.nodes[owner];
      if (PrepassNodeOwnsFourByteSlot(owner_plan.kind) && owner_plan.slot != kPrepassInvalidSlot &&
          slot_holder[owner_plan.slot] == owner) {
        slot_holder[owner_plan.slot] = chain.size();
        free_slots.push_back(owner_plan.slot);
      } else if (owner_plan.kind == SimtIndexPrepassNodeKind::kOpCompare && owner_plan.slot != kPrepassInvalidSlot &&
                 mask_owned[owner_plan.slot]) {
        mask_owned[owner_plan.slot] = false;
        free_mask_slots.push_back(owner_plan.slot);
      }
    }
    size_t temp_count = 0UL;
    if (node_plan.kind == SimtIndexPrepassNodeKind::kLeafArangeInnerRamp ||
        node_plan.kind == SimtIndexPrepassNodeKind::kCastTruncToInt) {
      temp_count = 1UL;
    } else if (node_plan.kind == SimtIndexPrepassNodeKind::kOpWhere) {
      for (size_t branch_pos = 1UL; branch_pos < node_plan.operands.size(); ++branch_pos) {
        if (!node_plan.operands[branch_pos].is_node) {
          ++temp_count;
        }
      }
    }
    for (size_t temp = 0UL; temp < temp_count; ++temp) {
      if (free_slots.empty()) {
        return reject("four byte slot demand exceeds limit", chain[i]);
      }
      node_plan.temp_slots.push_back(free_slots.back());
      free_slots.pop_back();
      peak_slots = std::max(peak_slots, kPrepassMaxFourByteSlots - free_slots.size());
    }
    if (node_plan.kind == SimtIndexPrepassNodeKind::kOpCompare) {
      if (free_mask_slots.empty()) {
        return reject("mask slot demand exceeds limit", chain[i]);
      }
      node_plan.slot = free_mask_slots.back();
      free_mask_slots.pop_back();
      mask_owned[node_plan.slot] = true;
      peak_masks = std::max(peak_masks, kPrepassMaxMaskSlots - free_mask_slots.size());
    } else if (PrepassNodeOwnsFourByteSlot(node_plan.kind)) {
      size_t dying_input_slot = kPrepassInvalidSlot;
      for (const auto &operand : node_plan.operands) {
        if (!operand.is_node) {
          continue;
        }
        const size_t owner = effective_owner[operand.node_index];
        if (last_consumer[owner] == i && PrepassNodeOwnsFourByteSlot(plan.nodes[owner].kind) &&
            plan.nodes[owner].slot != kPrepassInvalidSlot) {
          dying_input_slot = plan.nodes[owner].slot;
          break;
        }
      }
      if (dying_input_slot != kPrepassInvalidSlot) {
        // 原地复用：持有关系转移给本节点；旧持有者的 slot 字段保留（发射端读取操
        // 作数仍指向该槽，本节点发射在同槽先读后写）。
        node_plan.slot = dying_input_slot;
        slot_holder[dying_input_slot] = i;
      } else {
        if (free_slots.empty()) {
          return reject("four byte slot demand exceeds limit", chain[i]);
        }
        node_plan.slot = free_slots.back();
        free_slots.pop_back();
        slot_holder[node_plan.slot] = i;
        peak_slots = std::max(peak_slots, kPrepassMaxFourByteSlots - free_slots.size());
      }
    }
    for (size_t temp : node_plan.temp_slots) {
      free_slots.push_back(temp);  // 临时槽归还（temp_slots 记录保留供发射期使用）。
    }
  }

  // 比较结果只允许作为 Where/Select 掩码消费（掩码是 uint8，不能进入算术域）。
  // 掩码经语义定位后位于 operands[0]，此处按操作数而非输入端口判断。
  for (size_t i = 0UL; i < chain.size(); ++i) {
    const auto &node_plan = plan.nodes[i];
    for (size_t pos = 0UL; pos < node_plan.operands.size(); ++pos) {
      if (!node_plan.operands[pos].is_node) {
        continue;
      }
      if (pos == 0UL && node_plan.kind == SimtIndexPrepassNodeKind::kOpWhere) {
        continue;
      }
      if (ResolvePrepassMaskOrigin(plan.nodes, node_plan.operands[pos].node_index) < plan.nodes.size()) {
        return reject("compare result is consumed outside a where mask", chain[i]);
      }
    }
  }
  // ===== 阶段 E：结果解析与终局校验 =====
  // 结果节点（index 输入的透传归并终点）的区间必须落入 int32 索引表。
  SimtIndexPrepassOperand index_operand;
  if (resolve_input(indirect_load_ptr, kIndexTensorIndex, index_operand) != 0) {
    return reject("index result is not produced inside the chain", indirect_load_ptr);
  }
  const size_t result_index = effective_owner[index_operand.node_index];
  if (!PrepassNodeOwnsFourByteSlot(plan.nodes[result_index].kind) ||
      plan.nodes[result_index].slot == kPrepassInvalidSlot) {
    return reject("index result has no prepass slot", chain[result_index]);
  }
  // Ln(0) 等哨兵 lane 会把区间端点推到 ±inf：这些 lane 的取值在参考路径同样铸成
  // 未定义 int64 索引（图契约默认不出现，或被 Where 掩码丢弃），按“无效应 lane”
  // 剪除无穷端点后仅校验有限区间。
  PrepassRange finite_range = ranges[result_index];
  if (std::isinf(finite_range.lo) && finite_range.lo < 0.0) {
    finite_range.lo = kPrepassInt32DomainLo;
  }
  if (std::isinf(finite_range.hi) && finite_range.hi > 0.0) {
    finite_range.hi = kPrepassInt32DomainHi;
  }
  finite_range.has_inf = false;
  if (!RangeFitsInt32(finite_range)) {
    return reject("index result range does not fit int32", chain[result_index]);
  }
  plan.result_slot = plan.nodes[result_index].slot;
  plan.four_byte_slot_count = peak_slots;
  plan.mask_slot_count = peak_masks;
  plan.inner_size = view.output.sizes.back();
  plan.eligible = true;
  GELOGI("[IndirectLoad] SIMT index prepass is eligible for node[%s]: chain[%zu], fp32 slots[%zu], masks[%zu].",
         indirect_load_ptr->GetNamePtr(), chain.size(), plan.four_byte_slot_count, plan.mask_slot_count);
  return af::SUCCESS;
}

}  // namespace ascgen_utils::indirect_load
