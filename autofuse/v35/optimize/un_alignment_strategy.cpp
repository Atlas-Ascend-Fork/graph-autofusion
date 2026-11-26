/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "un_alignment_strategy.h"

#include <stack>
#include "ascir_utils.h"
#include "common_utils.h"
#include "graph_utils.h"
#include "indirect_load_utils.h"
#include "tensor_layout_utils.h"
#include "platform/v1/alignment_strategy.h"

namespace optimize {
namespace {
constexpr uint32_t kAlignBytes = 32U;

bool IsStaticUnalignedStride(const af::Expression &stride, int64_t dtype_size) {
  if (dtype_size <= 0) {
    return false;
  }
  int64_t stride_val = 0;
  if (!stride.IsConstExpr() || !stride.GetConstValue(stride_val)) {
    return false;
  }
  return (stride_val * dtype_size) % static_cast<int64_t>(kAlignBytes) != 0;
}

bool ArangeStoreNeedsDiscontinuousAlignment(const af::AscTensorAttr &attr) {
  for (auto id = attr.vectorized_axis.rbegin(); id != attr.vectorized_axis.rend(); ++id) {
    const auto iter = std::find(attr.axis.begin(), attr.axis.end(), *id);
    if (iter == attr.axis.end()) {
      return false;
    }
    const size_t index = static_cast<size_t>(std::distance(attr.axis.begin(), iter));
    if (af::SymbolicUtils::StaticCheckEq(attr.repeats[index], af::sym::kSymbolOne) == af::TriBool::kTrue ||
        af::SymbolicUtils::StaticCheckEq(attr.strides[index], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    return af::SymbolicUtils::StaticCheckNe(attr.strides[index], af::sym::kSymbolOne) == af::TriBool::kTrue;
  }
  return false;
}

af::Status NormalizeStoreSingletonStrides(const af::AscNodePtr &node) {
  GE_ASSERT_TRUE(node->outputs().size() == 1UL);
  auto &attr = node->outputs[0].attr;
  GE_ASSERT_TRUE(attr.vectorized_axis.size() == attr.vectorized_strides.size());
  GE_ASSERT_TRUE(attr.axis.size() == attr.repeats.size());
  af::Expression inner_span = af::sym::kSymbolOne;
  for (size_t index = attr.vectorized_axis.size(); index > 0UL; --index) {
    const size_t vector_index = index - 1UL;
    const auto axis_iter = std::find(attr.axis.begin(), attr.axis.end(), attr.vectorized_axis[vector_index]);
    GE_ASSERT_TRUE(axis_iter != attr.axis.end());
    const size_t axis_index = static_cast<size_t>(std::distance(attr.axis.begin(), axis_iter));
    auto &stride = attr.vectorized_strides[vector_index];
    if (af::SymbolicUtils::StaticCheckEq(attr.repeats[axis_index], af::sym::kSymbolOne) == af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
      stride = inner_span;
    }
    if (af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) != af::TriBool::kTrue) {
      inner_span = af::sym::Mul(stride, attr.repeats[axis_index]);
    }
  }
  return af::SUCCESS;
}

// 紧凑连续布局静态校验：向量化轴规模自洽、dtype 有效、尾轴 stride==1 且 repeat>1、次内维
// stride==尾轴 repeat（紧凑连续）且行距静态非 32B 对齐；校验通过时输出 dtype 字节数与尾轴 stride。
bool IsCompactUnalignedRowStride(const af::AscTensorAttr &attr, int64_t &dtype_size, af::Expression &tail_stride) {
  const auto vec_stride_size = attr.vectorized_strides.size();
  if (vec_stride_size < 2UL || attr.vectorized_axis.size() != vec_stride_size ||
      attr.axis.size() != attr.repeats.size()) {
    return false;
  }
  dtype_size = static_cast<int64_t>(af::GetSizeByDataType(attr.dtype));
  if (dtype_size <= 0) {
    return false;
  }
  tail_stride = attr.vectorized_strides[vec_stride_size - 1UL];
  if (af::SymbolicUtils::StaticCheckEq(tail_stride, af::sym::kSymbolOne) != af::TriBool::kTrue) {
    return false;
  }
  auto tail_axis_iter = std::find(attr.axis.begin(), attr.axis.end(), attr.vectorized_axis[vec_stride_size - 1UL]);
  if (tail_axis_iter == attr.axis.end()) {
    return false;
  }
  const auto tail_axis_index = static_cast<size_t>(std::distance(attr.axis.begin(), tail_axis_iter));
  const auto &tail_repeat = attr.repeats[tail_axis_index];
  if (af::SymbolicUtils::StaticCheckEq(tail_repeat, af::sym::kSymbolOne) == af::TriBool::kTrue) {
    return false;
  }
  const auto &penultimate_stride = attr.vectorized_strides[vec_stride_size - 2UL];
  if (af::SymbolicUtils::StaticCheckEq(penultimate_stride, af::sym::Mul(tail_repeat, tail_stride)) !=
      af::TriBool::kTrue) {
    return false;
  }
  return IsStaticUnalignedStride(penultimate_stride, dtype_size);
}

// 对单个 tensor 的向量化 stride 做 32B 行距垫整：紧凑连续布局（见 IsCompactUnalignedRowStride）
// 且次内维行距静态非 32B 对齐时，自尾侧重建——次内维垫整到 32B/dtype 对齐粒度，外层轴按垫整后
// 行距逐层累积，广播零 stride 轴保持不变；非紧凑跳变（batch/slice 间隙）或符号无法静态判定时
// 保持原样。返回是否发生垫整。
bool PadCompactVectorizedRowStrides(af::AscTensorAttr &attr) {
  int64_t dtype_size = 0;
  af::Expression tail_stride;
  if (!IsCompactUnalignedRowStride(attr, dtype_size, tail_stride)) {
    return false;
  }
  const auto vec_stride_size = attr.vectorized_strides.size();
  std::vector<af::Expression> padded_strides = attr.vectorized_strides;
  af::Expression size_product = tail_stride;
  af::Expression orig_product = tail_stride;
  const auto align_factor = kAlignBytes / static_cast<uint32_t>(dtype_size);
  for (int64_t index = static_cast<int64_t>(vec_stride_size) - 1; index >= 0; --index) {
    auto &stride = padded_strides.at(static_cast<size_t>(index));
    if (af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    auto axis_iter = std::find(attr.axis.begin(), attr.axis.end(), attr.vectorized_axis[static_cast<size_t>(index)]);
    if (axis_iter == attr.axis.end()) {
      return false;
    }
    const auto axis_index = static_cast<size_t>(std::distance(attr.axis.begin(), axis_iter));
    const auto &repeat = attr.repeats[axis_index];
    if (index == static_cast<int64_t>(vec_stride_size) - 1) {
      stride = size_product;
      size_product = af::sym::Mul(size_product, repeat);
    } else if (index == static_cast<int64_t>(vec_stride_size) - 2) {
      stride = af::sym::Align(size_product, align_factor);
      size_product = af::sym::Mul(stride, repeat);
    } else {
      // 外层轴仅当原 stride 为原始布局的内层累计跨距（紧致连续）或其对齐形式时，才可按垫整后
      // 跨距重写（垫整后行距已对齐，跨距乘积天然对齐，两种形式的重写值数值一致）。
      const bool dense = af::SymbolicUtils::StaticCheckEq(stride, orig_product) == af::TriBool::kTrue;
      const bool aligned_dense =
          af::SymbolicUtils::StaticCheckEq(stride, af::sym::Align(orig_product, align_factor)) == af::TriBool::kTrue;
      if (!dense && !aligned_dense) {
        return false;
      }
      stride = size_product;
      size_product = af::sym::Mul(size_product, repeat);
    }
    orig_product = af::sym::Mul(orig_product, repeat);
  }
  attr.vectorized_strides = padded_strides;
  return true;
}

// IndirectLoad SIMT 后置 Reduce 链行距垫整：kSimtOp 角色 value 写 UB local target 后由后置
// Reduce 消费，模板 Store 与 Reduce 按 SizeAlign(inner, 32/sizeof(dtype)) 的对齐行距读写；
// 而尾轴 reduce 的对齐推导（Reduce 输出尾轴 stride==0 判 kNotAligned）沿链回传后生成紧凑
// vectorized_strides，buffer size 与 tiling 预算按紧凑 span 计算，行数 > 1 时 Store/Reduce
// 读写越过 buffer 末端。链上成员（IL 输出至 Reduce 输入的各 tensor）统一垫整到同一对齐行距。
void PadSimtPostReduceChainRowStrides(af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    const auto indirect_load = std::dynamic_pointer_cast<af::AscNode>(node);
    if (indirect_load == nullptr || !af::ops::IsOps<af::ascir_op::IndirectLoad>(indirect_load) ||
        ascgen_utils::indirect_load::GetTemplateRole(indirect_load) !=
            ascgen_utils::indirect_load::TemplateRole::kSimtOp) {
      continue;
    }
    const auto reduce = ascgen_utils::indirect_load::GetPostReduceConsumer(indirect_load);
    if (reduce == nullptr) {
      // 非后置 Reduce 形态没有行对齐写回契约，保持紧凑布局
      continue;
    }
    std::vector<af::AscNodePtr> pending{indirect_load};
    std::set<af::Node *> visited;
    for (size_t index = 0UL; index < pending.size(); ++index) {
      const auto &chain_node = pending[index];
      if (chain_node == nullptr || !visited.insert(chain_node.get()).second) {
        continue;
      }
      for (const auto &output : chain_node->outputs()) {
        if (output != nullptr) {
          (void)PadCompactVectorizedRowStrides(output->attr);
        }
      }
      // Reduce 输出视图无紧凑行形态，天然不满足垫整判据；不越过 Reduce 继续传播
      if (chain_node.get() == reduce.get()) {
        continue;
      }
      for (const auto &next : chain_node->GetOutDataNodes()) {
        const auto next_node = std::dynamic_pointer_cast<af::AscNode>(next);
        if (next_node != nullptr) {
          pending.emplace_back(next_node);
        }
      }
    }
  }
}
}  // namespace
AlignmentType UnAlignmentStrategy::GetDefaultAlignmentType() {
  return AlignmentType::kNotAligned;
}

af::Status UnAlignmentStrategy::LoadAlignmentInferFunc(const af::AscNodePtr &node) {
  const auto &output_attr = node->outputs[0].attr;
  if (!af::ops::IsOps<af::ascir_op::Load>(node)) {
    GELOGD("Node[%s] is continuous loading, input tensor does not need to be aligned.", node->GetNamePtr());
    // vectorized_axis连续则可以连续搬运
    tensor_to_align_type_[&output_attr] = {AlignmentType::kNotAligned};
    return af::SUCCESS;
  }

  if (IsLoadNeedAlignForReduce(node)) {
    // 符合该条件的load节点会转为nddma，不必对齐
    GELOGD("Node[%s] is discontinuous loading, input tensor needs to be fixed not aligned.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kNotAligned};
    return af::SUCCESS;
  }
  ascgen_utils::DiscontinuityInfo info;
  GE_ASSERT_SUCCESS(ascgen_utils::TensorLayoutUtils::AnalyzeLoadDiscontinuity(output_attr, info),
                    "Failed to analyze discontinuity info for node:[%s].", node->GetNamePtr());
  if (!info.has_multiple_discontinuities) {
    // 走compat模式搬运
    tensor_to_align_type_[&output_attr] = {AlignmentType::kNotAligned};
  } else if (info.is_tail_axis_discontinuous) {
    // vectorized_axis在尾轴非连续, 且不能走compat模式, 需要转nddma
    GELOGD("Node[%s] is tail axis discontinuous loading, need to use nddma.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kNotAligned};
  } else {
    // vectorized_axis 在gm上非连续,需要尾轴对齐搬运
    GELOGD("Node[%s] is discontinuous loading, input tensor needs to be aligned.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kAligned};
  }
  return af::SUCCESS;
}

af::Status UnAlignmentStrategy::StoreAlignmentInferFunc(const af::AscNodePtr &node) {
  const auto &output_attr = node->outputs[0].attr;
  AlignmentType input_align = tensor_to_align_type_[&node->inputs[0].attr].align_type;
  tensor_to_align_type_[&output_attr] = {input_align};
  auto owner_graph = node->GetOwnerComputeGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph);
  auto graph_attr = owner_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  const auto &axis = graph_attr->axis;
  size_t tile_inner_axis_size = 0UL;
  for (auto axis_id : node->inputs[0].attr.vectorized_axis) {
    auto iter =
        std::find_if(axis.begin(), axis.end(), [axis_id](const af::AxisPtr &axis) { return axis->id == axis_id; });
    if (iter != axis.end() && (*iter)->type == af::Axis::kAxisTypeTileInner) {
      ++tile_inner_axis_size;
    }
  }
  const auto input_node = std::dynamic_pointer_cast<af::AscNode>(node->inputs[0].anchor.GetOwnerNode());
  const bool is_arange_input = input_node != nullptr && af::ops::IsOps<af::ascir_op::Arange>(input_node);

  if (ScheduleUtils::IsNeedDiscontinuousAligned(output_attr) ||
      (is_arange_input && ArangeStoreNeedsDiscontinuousAlignment(output_attr))) {
    GELOGD("Node[%s] is last axis discontinuous writing, input tensor needs to be aligned.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kDiscontinuous};
    GE_ASSERT_SUCCESS(BackPropagateAlignment(node, AlignmentType::kDiscontinuous));
  } else if (!ScheduleUtils::IsVectorizedAxisContinuousInGM(output_attr) &&
             (tile_inner_axis_size > 1UL || is_arange_input)) {
    GELOGD("Node[%s] is discontinuous writing, input tensor needs to be aligned.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kAligned};
    GE_ASSERT_SUCCESS(BackPropagateAlignment(node, AlignmentType::kAligned));
  }
  return af::SUCCESS;
}

bool IsTailBroadcastNode(af::AscNode &node) {
  // 判断节点尾轴是否广播的节点
  if (node.attr.api.compute_type != af::ComputeType::kComputeBroadcast) {
    return false;
  }

  const auto &in_nodes = node.GetInDataNodes();
  // scalar广播
  if (ScheduleUtils::IsScalarLikeNode(in_nodes.at(0UL))) {
    GELOGD("Input of Broadcast[%s] is Scalar[%s], support.", node.GetNamePtr(), in_nodes.at(0UL)->GetNamePtr());
    return true;
  }
  auto inputs = node.inputs();
  auto outputs = node.outputs();
  GE_ASSERT(!inputs.empty());
  GE_ASSERT(!outputs.empty());
  GE_ASSERT_NOTNULL(inputs[0]);
  GE_ASSERT_NOTNULL(outputs[0]);

  const auto &in_attr = inputs[0]->attr;
  std::vector<af::Expression> in_vec_repeats;
  GE_ASSERT(ScheduleUtils::GetVectorRepeats(in_attr.repeats, in_attr.axis, in_attr.vectorized_axis, in_vec_repeats) ==
            af::SUCCESS);

  const auto &out_attr = outputs[0]->attr;
  std::vector<af::Expression> out_vec_repeats;
  GE_ASSERT(ScheduleUtils::GetVectorRepeats(out_attr.repeats, out_attr.axis, out_attr.vectorized_axis,
                                            out_vec_repeats) == af::SUCCESS);

  // 输入输出vectorized_strides大小一致，且非空
  GE_ASSERT(in_vec_repeats.size() == out_vec_repeats.size());
  GE_ASSERT(!in_vec_repeats.empty());
  auto vec_size = in_vec_repeats.size();
  auto in_last_dim_size = in_vec_repeats[vec_size - 1];
  auto out_last_dim_size = out_vec_repeats[vec_size - 1];
  auto is_tail_brc = ascgen_utils::ExpressEq(in_last_dim_size, af::Symbol(1)) &&
                     !ascgen_utils::ExpressEq(in_last_dim_size, out_last_dim_size);
  return is_tail_brc;
}

// 识别尾轴广播源的"逐元素 32B 槽位"布局，返回其最内层有效数据轴下标。
// 用意：这种布局由对齐策略对退化尾轴 pad 产生（最内层有效轴 stride 恒为
// Align(1, 32/dtype_size)），是唯一一类"BroadcastExtend 会错读、且 RemovePad 恰好能修"的
// 非连续签名——BroadcastExtend 只按 src_shape 稠密反推源 stride，会把槽位间隙当地址；
// 而 RemovePadApiCall 不读 descriptor 的 stride，按 SizeAlign(尾轴 extent, 32/dtype_size)
// 重算行距，与本布局的产生公式一致，归一化才正确。其它任意跨步（如转置写 20B）连
// RemovePad 自己都修不了，因此只认精确等值，不做宽泛的 stride != 1 判定。
// 其内侧的轴必须全部是静态 repeat==1 的退化广播尾（发射侧按 1 处理，不占行数）；
// repeat>1 的零 stride 中间轴会打破下方行数上界检查与发射口径的一致性，视为不命中。
// 返回值 >= 0 为命中的数据轴下标；-1 为不命中。out_vec_repeats 带回各向量化轴的 repeat，
// out_vec_axis 带回各向量化轴 id（供行数上界证明沿切分链回溯原轴）。
int64_t Get32BSlotInnerAxis(const af::AscNodePtr &node, std::vector<af::Expression> &out_vec_repeats,
                            std::vector<int64_t> &out_vec_axis) {
  const auto inputs = node->inputs();
  if (inputs.empty()) {
    GELOGW("[%s] has no data input, skip 32B slot detection.", node->GetNamePtr());
    return -1;
  }
  const auto &in_attr = inputs[0]->attr;
  const auto &vec_strides = in_attr.vectorized_strides;
  if (vec_strides.empty() || in_attr.vectorized_axis.size() != vec_strides.size()) {
    return -1;
  }
  const int64_t dtype_size = af::GetSizeByDataType(in_attr.dtype);
  const auto align_width = static_cast<int64_t>(BaseAlignmentStrategy::GetAlignWidth());
  if (dtype_size == 0 || dtype_size < 0 || dtype_size > align_width) {
    return -1;
  }
  const auto align_factor = static_cast<uint32_t>(align_width / dtype_size);

  if (ScheduleUtils::GetVectorRepeats(in_attr.repeats, in_attr.axis, in_attr.vectorized_axis, out_vec_repeats) !=
          af::SUCCESS ||
      out_vec_repeats.size() != vec_strides.size()) {
    return -1;
  }
  out_vec_axis = in_attr.vectorized_axis;

  // 从尾向前跳过退化广播尾，定位最内层有效数据轴
  int64_t inner_index = -1;
  for (int64_t i = static_cast<int64_t>(vec_strides.size()) - 1; i >= 0; --i) {
    if (af::SymbolicUtils::StaticCheckEq(vec_strides[static_cast<size_t>(i)], af::sym::kSymbolZero) ==
        af::TriBool::kTrue) {
      if (af::SymbolicUtils::StaticCheckEq(out_vec_repeats[static_cast<size_t>(i)], af::sym::kSymbolOne) !=
          af::TriBool::kTrue) {
        return -1;
      }
      continue;
    }
    inner_index = i;
    break;
  }
  if (inner_index < 0) {
    return -1;
  }
  if (af::SymbolicUtils::StaticCheckEq(vec_strides[static_cast<size_t>(inner_index)], af::Symbol(align_factor)) !=
      af::TriBool::kTrue) {
    return -1;
  }
  return inner_index;
}

// 用意：符号 tile 轴（如 nddma 模板留在向量化尾段里的 z0t）的运行时值有静态上界——
// tiling 切分不变式保证同一原轴的各段 repeat >= 1 且乘积等于原轴 size，因此 tile 轴
// 运行时值 <= 原轴静态 size。沿 Axis::from 切分链回溯到 Original 轴取其 size 常量即得上界。
// 回溯不到静态常量（动态 shape 原轴、Merged 多来源轴等）时返回 false 保守放弃。
bool GetTileAxisStaticUpperBound(af::AscGraph &graph, const int64_t axis_id, uint64_t &upper_bound) {
  constexpr int32_t kMaxSplitChainDepth = 16;  // 防御性上限：正常切分链远短于此
  const af::Axis *axis = graph.FindAxis(axis_id);
  for (int32_t depth = 0; axis != nullptr && depth < kMaxSplitChainDepth; ++depth) {
    if (axis->from.empty()) {
      break;
    }
    if (axis->from.size() != 1UL) {
      return false;  // Merged 等多来源轴：单链回溯证不出上界
    }
    axis = graph.FindAxis(axis->from.front());
  }
  if (axis == nullptr) {
    return false;
  }
  int64_t bound = 0;
  if (!axis->size.GetConstValue(bound) || bound <= 0) {
    return false;
  }
  upper_bound = static_cast<uint64_t>(bound);
  return true;
}

// 用意：GatherMask 的 repeatTimes 为 uint16，非最内合并行数必须有静态可证的上界。
// 静态常量段直接取常量；符号 tile 段取其原轴静态 size 作上界（nddma 模板的行数段
// 形如 z0t*3，靠原轴上界才证得 100*3 << 65535）。任一段证不出上界或乘积越界时返回
// false 保守跳过，避免插桩后 GatherMask 静默截断。
bool IsGatherMaskRowBoundProvable(af::AscGraph &graph, const std::vector<af::Expression> &vec_repeats,
                                  const std::vector<int64_t> &vec_axis, const int64_t inner_axis) {
  constexpr uint64_t kMaxRemovePadRepeatTimes = 65535UL;
  uint64_t merged_rows = 1UL;
  for (int64_t i = 0; i <= inner_axis; ++i) {
    uint64_t upper_bound = 0UL;
    int64_t repeat = 0;
    if (vec_repeats[static_cast<size_t>(i)].GetConstValue(repeat) && repeat > 0) {
      upper_bound = static_cast<uint64_t>(repeat);
    } else if (!GetTileAxisStaticUpperBound(graph, vec_axis[static_cast<size_t>(i)], upper_bound)) {
      return false;
    }
    merged_rows *= upper_bound;
    if (merged_rows > kMaxRemovePadRepeatTimes) {
      return false;
    }
  }
  return true;
}

// 用意：尾轴广播（src 尾轴 size==1）的源为逐元素 32B 槽位布局且 GatherMask 行数上界可证时，
// 需要插入 RemovePad 把源归一化成紧凑布局——否则 BroadcastExtend 按 src_shape 稠密反推
// 读源，会把槽位间隙当地址、错读源数据。判据只读广播输入 tensor 的布局，与生产者类型无关。
bool TailBrcSrcNeedsRemovePad(af::AscGraph &graph, const af::AscNodePtr &node) {
  std::vector<af::Expression> in_vec_repeats;
  std::vector<int64_t> in_vec_axis;
  const int64_t inner_axis = Get32BSlotInnerAxis(node, in_vec_repeats, in_vec_axis);
  if (inner_axis < 0) {
    return false;
  }
  return IsGatherMaskRowBoundProvable(graph, in_vec_repeats, in_vec_axis, inner_axis);
}

// 在广播输入边上插入 RemovePad（只重接该消费边，生产者的其它消费边不动），
// 输出紧凑布局由 SetVectorizedStridesForTensor(kNotAligned) 生成。
af::Status InsertRemovePadBeforeBroadcast(af::AscGraph &graph, const af::AscNodePtr &broadcast_node) {
  const auto producer = std::dynamic_pointer_cast<af::AscNode>(broadcast_node->GetInDataNodes().at(0UL));
  if (producer == nullptr || ScheduleUtils::IsBuffer(producer) || af::ops::IsOps<af::ascir_op::RemovePad>(producer)) {
    return af::SUCCESS;
  }
  const auto in_anchor = broadcast_node->GetInDataAnchor(0UL);
  const auto out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(out_anchor);
  const int32_t out_idx = out_anchor->GetIdx();
  GE_ASSERT_TRUE(out_idx >= 0);

  // 按消费边插入：名字需含 out_idx 和消费者名，避免同一生产者扇出多个尾轴广播时重名。
  const std::string node_name =
      producer->GetName() + "_remove_pad_" + std::to_string(out_idx) + "_brc_" + broadcast_node->GetName();
  af::ascir_op::RemovePad remove_pad_op(node_name.c_str());
  const auto remove_pad_node = graph.AddNode(remove_pad_op);
  GE_ASSERT_NOTNULL(remove_pad_node);
  remove_pad_node->attr = producer->attr;
  remove_pad_node->outputs[0].attr = producer->outputs[static_cast<uint32_t>(out_idx)].attr;
  GE_ASSERT_SUCCESS(BaseAlignmentStrategy::SetVectorizedStridesForTensor(
      remove_pad_node, remove_pad_node->outputs[0].attr, AlignmentType::kNotAligned));
  remove_pad_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  remove_pad_node->attr.api.type = af::ApiType::kAPITypeCompute;
  remove_pad_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  GE_ASSERT_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(out_anchor, in_anchor, remove_pad_node->GetOutDataAnchor(0)));
  GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(out_anchor, remove_pad_node->GetInDataAnchor(0)));
  return af::SUCCESS;
}

af::Status UnAlignmentStrategy::SetAlignInfoForTailBrcNodes(AlignmentType aligned_type, af::AscNode *node,
                                                            std::set<af::Node *> &visited_nodes,
                                                            std::queue<af::Node *> &node_queue) {
  for (const auto &output : node->outputs()) {
    auto &align_info = tensor_to_align_type_[&output->attr];
    if (align_info.align_type == aligned_type) {
      continue;
    }
    if (align_info.align_type == AlignmentType::kFixedNotAligned) {
      align_info.conflict_with_output = true;
      continue;
    }
    GELOGD("Node [%s]'s align type need to be changed.", node->GetNamePtr());
    align_info.align_type = aligned_type;

    for (const auto &peer_in : output->anchor.GetPeerInDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(peer_in);
      auto asc_node = std::dynamic_pointer_cast<af::AscNode>(peer_in->GetOwnerNode());
      if (ScheduleUtils::IsBuffer(asc_node)) {
        continue;
      }
      if (ScheduleUtils::IsReduce(asc_node)) {
        // Reduce changes the tensor layout, so input alignment must not propagate to its output.
        GELOGI("Stop alignment propagation from node [%s] to Reduce node [%s].", node->GetNamePtr(),
               asc_node->GetNamePtr());
        continue;
      }
      if (visited_nodes.insert(asc_node.get()).second) {
        node_queue.push(asc_node.get());
      }
    }
  }
  visited_nodes.insert(node);
  return af::SUCCESS;
}

af::Status UnAlignmentStrategy::BackPropagateAlignment(const af::AscNodePtr &node, AlignmentType aligned_type) {
  std::set<af::Node *> visited_nodes;
  std::queue<af::Node *> node_queue;
  visited_nodes.emplace(node.get());
  SetAlignInfoForNodeInputs(aligned_type, node.get(), visited_nodes, node_queue);
  while (!node_queue.empty()) {
    const auto curr_node = dynamic_cast<af::AscNode *>(node_queue.front());
    node_queue.pop();
    GE_ASSERT_NOTNULL(curr_node);
    if (IsTailBroadcastNode(*curr_node)) {
      GE_ASSERT(SetAlignInfoForTailBrcNodes(aligned_type, curr_node, visited_nodes, node_queue) == af::SUCCESS);
      continue;
    }

    bool alignment_changed = SetAlignInfoForNodeOutputs(aligned_type, curr_node, visited_nodes, node_queue);
    if (alignment_changed) {
      SetAlignInfoForNodeInputs(aligned_type, curr_node, visited_nodes, node_queue);
    }
  }
  return af::SUCCESS;
}

af::Status UnAlignmentStrategy::ConcatAlignmentInferFunc(const af::AscNodePtr &node) {
  bool any_input_aligned = false;
  for (const auto &input : node->inputs()) {
    auto alignment_iter = tensor_to_align_type_.find(&input->attr);
    if (alignment_iter == tensor_to_align_type_.end()) {
      continue;
    }
    const AlignmentType input_alignment = alignment_iter->second.align_type;
    if (input_alignment == AlignmentType::kAligned || input_alignment == AlignmentType::kDiscontinuous) {
      any_input_aligned = true;
      break;
    }
  }
  const auto &output_attr = node->outputs[0].attr;
  if (any_input_aligned) {
    size_t concat_dim = 0UL;
    bool unused;
    GE_ASSERT_SUCCESS(ScheduleUtils::ResolveDiffDim(node, concat_dim, unused));
    if (concat_dim != output_attr.repeats.size() - 1UL) {
      GELOGI("%s concat_dim is not the last dim, and inputs are aligned, use default infer func", node->GetNamePtr());
      return DefaultAlignmentInferFunc(node);
    }
  }
  tensor_to_align_type_[&output_attr] = {AlignmentType::kFixedNotAligned};
  return af::SUCCESS;
}

/**
 * 功能：将尾轴transpose的load节点替换为nddma节点；
 * 与naddma_template的区别：这里仅处理load，而nddma模板则处理load + brc或load + cast
 */
Status GenLoadToGenNddmaNode(const af::AscNodePtr &node_load) {
  GE_CHECK_NOTNULL(node_load);
  GE_CHECK_NOTNULL(node_load->GetOpDesc());

  node_load->GetOpDesc()->SetType("Nddma");
  node_load->attr.type = "Nddma";
  return af::SUCCESS;
}

Status PadNddmaOutputVectorizedStrides(const af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    // 垫整仅作用于 UB 侧视图，GM 侧边界节点保持原样
    if (af::ops::IsOps<af::ascir_op::Data>(node) || af::ops::IsOps<af::ascir_op::Scalar>(node) ||
        af::ops::IsOps<af::ascir_op::ScalarData>(node) || af::ops::IsOps<af::ascir_op::Output>(node) ||
        af::ops::IsOps<af::ascir_op::Workspace>(node) || node->outputs().empty()) {
      continue;
    }
    (void)PadCompactVectorizedRowStrides(node->outputs[0].attr);
  }
  return af::SUCCESS;
}
Status UnAlignmentStrategy::GetCurrentNodeContinuousTailAxisNum(const af::AscNodePtr &node,
                                                                uint32_t &continuous_tail_axis_num,
                                                                uint32_t &discontinuous_axis_num) {
  continuous_tail_axis_num = 0;
  uint32_t current_discontinuous_axis_num = 0;
  auto &output_attr = node->outputs[0].attr;
  const auto &output_vec_axis = output_attr.vectorized_axis;
  af::Expression inner_repeat = af::sym::kSymbolOne;
  af::Expression inner_stride = af::sym::kSymbolOne;
  for (auto axis_it = output_vec_axis.rbegin(); axis_it != output_vec_axis.rend(); axis_it++) {
    const auto axis = *axis_it;
    auto axis_tensor_iter = std::find(output_attr.axis.begin(), output_attr.axis.end(), axis);
    GE_ASSERT_TRUE(axis_tensor_iter != output_attr.axis.end(),
                   "Cannot find vectorized axis [%ld] in [%s]'s output tensor.", axis, node->GetNamePtr());

    const int64_t axis_index = std::distance(output_attr.axis.begin(), axis_tensor_iter);
    const auto &stride = output_attr.strides[axis_index];
    const auto &repeat = output_attr.repeats[axis_index];
    if (af::SymbolicUtils::StaticCheckEq(stride, inner_repeat * inner_stride) == af::TriBool::kTrue) {
      continuous_tail_axis_num++;
      inner_repeat = repeat;
      inner_stride = stride;
    } else if (af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
      // 向量化轴的stride为0继续合轴，不更新inner_repeat和inner_stride
      continuous_tail_axis_num++;
    } else {
      // 出现第二根非连续轴的时候退出，第一根不连续轴用Compact模式处理
      if (current_discontinuous_axis_num < discontinuous_axis_num) {
        inner_repeat = repeat;
        inner_stride = stride;
        continuous_tail_axis_num++;
        current_discontinuous_axis_num++;
      } else {
        break;
      }
    }
  }
  return af::SUCCESS;
}

Status UnAlignmentStrategy::GetNodeContinuousTailAxisNumByStore(const af::AscNodePtr &node,
                                                                uint32_t &continuous_tail_axis_num,
                                                                uint32_t &discontinuous_axis_num) {
  continuous_tail_axis_num = UINT32_MAX;
  if (af::ops::IsOps<af::ascir_op::Store>(node)) {
    GE_ASSERT_SUCCESS(GetCurrentNodeContinuousTailAxisNum(node, continuous_tail_axis_num, discontinuous_axis_num));
    return af::SUCCESS;
  }
  for (size_t out_idx = 0U; out_idx < node->GetAllOutDataAnchorsSize(); out_idx++) {
    auto out_anchor = node->GetOutDataAnchor(out_idx);
    GE_CHECK_NOTNULL(out_anchor, "Node [%s] out_anchor[%zu] is null", node->GetNamePtr(), out_idx);
    for (size_t in_idx = 0U; in_idx < out_anchor->GetPeerInDataAnchors().size(); in_idx++) {
      auto in_anchor = out_anchor->GetPeerInDataAnchors().at(in_idx);
      GE_CHECK_NOTNULL(in_anchor, "Node [%s] out_anchor[%zu] in_anchor[%zu] is null", node->GetNamePtr(), out_idx,
                       in_idx);
      const auto &out_node = std::dynamic_pointer_cast<af::AscNode>(in_anchor->GetOwnerNode());
      GE_CHECK_NOTNULL(out_node, "Node [%s] out_anchor[%zu] in_anchor[%zu] owner_node is null", node->GetNamePtr(),
                       out_idx, in_idx);
      if (af::ops::IsOps<af::ascir_op::Store>(out_node)) {
        uint32_t store_continuous_axis_num = 0;
        GE_ASSERT_SUCCESS(
            GetCurrentNodeContinuousTailAxisNum(out_node, store_continuous_axis_num, discontinuous_axis_num));
        continuous_tail_axis_num = std::min(continuous_tail_axis_num, store_continuous_axis_num);
      } else {
        uint32_t out_node_continuous_axis_num = 0;
        GE_ASSERT_SUCCESS(
            GetNodeContinuousTailAxisNumByStore(out_node, out_node_continuous_axis_num, discontinuous_axis_num));
        continuous_tail_axis_num = std::min(continuous_tail_axis_num, out_node_continuous_axis_num);
      }
    }
  }
  return af::SUCCESS;
}

Status UnAlignmentStrategy::GetNodeContinuousTailAxisNumByLoad(const af::AscNodePtr &node,
                                                               uint32_t &continuous_tail_axis_num,
                                                               uint32_t &discontinuous_axis_num) {
  continuous_tail_axis_num = UINT32_MAX;
  if (af::ops::IsOps<af::ascir_op::Load>(node)) {
    GE_ASSERT_SUCCESS(GetCurrentNodeContinuousTailAxisNum(node, continuous_tail_axis_num, discontinuous_axis_num));
    return af::SUCCESS;
  }
  for (size_t in_idx = 0U; in_idx < node->GetAllInDataAnchorsSize(); in_idx++) {
    auto in_anchor = node->GetInDataAnchor(in_idx);
    GE_CHECK_NOTNULL(in_anchor, "Node [%s] in_anchor[%zu] is null", node->GetNamePtr(), in_idx);
    auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor, "Node [%s] in_anchor[%zu] peer_out_anchor is null", node->GetNamePtr(), in_idx);
    const auto &in_node = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
    GE_CHECK_NOTNULL(in_node, "Node [%s] in_anchor[%zu] peer_out_anchor owner_node is null", node->GetNamePtr(),
                     in_idx);
    if (af::ops::IsOps<af::ascir_op::Load>(in_node)) {
      uint32_t load_continuous_axis_num = 0;
      GE_ASSERT_SUCCESS(GetCurrentNodeContinuousTailAxisNum(in_node, load_continuous_axis_num, discontinuous_axis_num));
      continuous_tail_axis_num = std::min(continuous_tail_axis_num, load_continuous_axis_num);
    } else if (af::ops::IsOps<af::ascir_op::Broadcast>(node)) {
      // transpose模板中出现的brc节点的输入一定是scalar，向量化轴的对齐按照brc节点本身标注的stride信息进行。
      uint32_t load_continuous_axis_num = 0;
      GE_ASSERT_SUCCESS(GetCurrentNodeContinuousTailAxisNum(node, load_continuous_axis_num, discontinuous_axis_num));
      continuous_tail_axis_num = std::min(continuous_tail_axis_num, load_continuous_axis_num);
    } else {
      uint32_t in_node_continuous_axis_num = 0;
      GE_ASSERT_SUCCESS(
          GetNodeContinuousTailAxisNumByLoad(in_node, in_node_continuous_axis_num, discontinuous_axis_num));
      continuous_tail_axis_num = std::min(continuous_tail_axis_num, in_node_continuous_axis_num);
    }
  }
  return af::SUCCESS;
}

Status UnAlignmentStrategy::CollectTransposePreNodes(const af::AscGraph &graph,
                                                     std::set<af::AscNodePtr> &transpose_pre_nodes) {
  transpose_pre_nodes.clear();
  // 1. 找到所有 Transpose 节点
  std::vector<af::AscNodePtr> transpose_nodes;
  for (const auto &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Transpose>(node)) {
      transpose_nodes.push_back(node);
    }
  }
  if (transpose_nodes.empty()) {
    return af::SUCCESS;
  }
  // 2. 对每个 Transpose 节点，向上遍历收集所有前序节点
  for (const auto &transpose_node : transpose_nodes) {
    std::stack<af::AscNodePtr> node_stack;
    std::set<af::AscNodePtr> visited;
    node_stack.push(transpose_node);
    visited.insert(transpose_node);
    while (!node_stack.empty()) {
      auto current_node = node_stack.top();
      node_stack.pop();
      // 将当前节点加入前序节点集合
      if (!af::ops::IsOps<af::ascir_op::Transpose>(current_node)) {
        transpose_pre_nodes.insert(current_node);
      }
      // 向上遍历输入节点
      for (size_t in_idx = 0U; in_idx < current_node->GetAllInDataAnchorsSize(); in_idx++) {
        auto in_anchor = current_node->GetInDataAnchor(in_idx);
        GE_CHECK_NOTNULL(in_anchor, "Node [%s] in_anchor[%zu] is null", current_node->GetNamePtr(), in_idx);
        auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
        GE_CHECK_NOTNULL(peer_out_anchor, "Node [%s] in_anchor[%zu] peer_out_anchor is null",
                         current_node->GetNamePtr(), in_idx);
        const auto &in_node = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
        GE_CHECK_NOTNULL(in_node, "Node [%s] in_anchor[%zu] peer_out_anchor owner_node is null",
                         current_node->GetNamePtr(), in_idx);
        if (visited.find(in_node) != visited.end()) {
          continue;
        }
        // 遇到 Data/Scalar/ScalarData/Workspace 节点停止
        if (af::ops::IsOps<af::ascir_op::Data>(in_node) || af::ops::IsOps<af::ascir_op::Scalar>(in_node) ||
            af::ops::IsOps<af::ascir_op::ScalarData>(in_node) || af::ops::IsOps<af::ascir_op::Workspace>(in_node)) {
          continue;
        }
        visited.insert(in_node);
        node_stack.push(in_node);
      }
    }
  }
  return af::SUCCESS;
}

Status UnAlignmentStrategy::UpdateOutputVectorizedStrides(const af::AscNodePtr &node, uint32_t continuous_tail_axis_num,
                                                          uint32_t align_width) {
  auto &output_attr = node->outputs[0].attr;
  const auto &output_vec_axis = output_attr.vectorized_axis;
  ascir::SizeExpr size_product = af::sym::kSymbolOne;
  GE_ASSERT_TRUE(output_vec_axis.size() >= continuous_tail_axis_num,
                 "Node [%s] Output vectorized axis size is less than continuous axis num.", node->GetNamePtr());
  const auto dtype_size = af::GetSizeByDataType(output_attr.dtype);
  GE_ASSERT_TRUE(dtype_size > 0, "Node [%s] output tensor dtype is invalid.", node->GetNamePtr());
  const uint32_t align_factor = align_width / static_cast<uint32_t>(dtype_size);

  for (size_t i = output_vec_axis.size(); i > 0; i--) {
    auto index = i - 1;
    const auto axis = output_vec_axis[index];
    auto axis_tensor_iter = std::find(output_attr.axis.begin(), output_attr.axis.end(), axis);
    GE_ASSERT_TRUE(axis_tensor_iter != output_attr.axis.end(),
                   "Node [%s] Cannot find vectorized axis [%ld] in [%s]'s output tensor.", axis, node->GetNamePtr());

    const int64_t axis_index = std::distance(output_attr.axis.begin(), axis_tensor_iter);
    const auto &repeat = output_attr.repeats[axis_index];
    // 向量化轴的stride为0则不做处理，保留原stride
    if (af::SymbolicUtils::StaticCheckEq(output_attr.vectorized_strides[index], af::sym::kSymbolZero) !=
        af::TriBool::kTrue) {
      output_attr.vectorized_strides[index] = size_product;
      size_product = size_product * repeat;
    }
    if (index == (output_vec_axis.size() - continuous_tail_axis_num)) {
      size_product = af::sym::Align(size_product, align_factor);
    }
  }
  return af::SUCCESS;
}

bool IsGraphHasBroadcast(const af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Broadcast>(node)) {
      return true;
    }
  }
  return false;
}
Status UnAlignmentStrategy::ModifyTransposeFusionVectorizedStrides(af::AscGraph &graph, uint32_t align_width) {
  if (IsGraphHasBroadcast(graph)) {  // 当前broadcast api在出现不对齐轴的时候默认按照尾轴对齐，因此暂时不做优化。
    return af::SUCCESS;
  }
  // 收集 Transpose 前序节点
  std::set<af::AscNodePtr> transpose_pre_nodes;
  GE_ASSERT_SUCCESS(CollectTransposePreNodes(graph, transpose_pre_nodes));
  // ub-transpose模板：Compact模式拷贝，遇到第二根不连续轴时完成连续轴收集
  uint32_t discontinuous_axis_num = transpose_pre_nodes.empty() ? 0 : 1;

  for (const auto &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Data>(node) || af::ops::IsOps<af::ascir_op::Scalar>(node) ||
        af::ops::IsOps<af::ascir_op::ScalarData>(node) || af::ops::IsOps<af::ascir_op::Output>(node) ||
        af::ops::IsOps<af::ascir_op::Workspace>(node)) {
      continue;
    }
    uint32_t continuous_tail_axis_num = 0;
    // 根据节点类型选择不同的连续轴计算策略
    if (!transpose_pre_nodes.empty() && transpose_pre_nodes.find(node) != transpose_pre_nodes.end()) {
      // Transpose前序节点：按 Load 轴判断
      GE_ASSERT_SUCCESS(GetNodeContinuousTailAxisNumByLoad(node, continuous_tail_axis_num, discontinuous_axis_num));
    } else {
      // Transpose及后续节点或无Transpose：按 Store 轴判断
      GE_ASSERT_SUCCESS(GetNodeContinuousTailAxisNumByStore(node, continuous_tail_axis_num, discontinuous_axis_num));
      // ub-Transpose模板的Transpose输出侧最多处理3根连续轴，超过3根连续轴时按3根连续轴处理。
      if (!transpose_pre_nodes.empty() && continuous_tail_axis_num > 3U) {
        continuous_tail_axis_num = 3U;
      }
    }
    // 如果是ub-Transpose模板，则可以走Compact模式搬运，出现第二根不连续轴时退出，第一根不连续轴用Compact模式处理。
    // 实际上nddma模板也可以走Compact模式，但是codegen目前只能判断显式Transpose节点，暂时不放开。
    if (continuous_tail_axis_num <= 1 || continuous_tail_axis_num == UINT32_MAX) {
      continue;
    }
    for (const auto &output : node->outputs()) {
      auto &output_attr = output->attr;
      if (output_attr.vectorized_axis.empty()) {
        continue;
      }
      GE_ASSERT_SUCCESS(UpdateOutputVectorizedStrides(node, continuous_tail_axis_num, align_width));
    }
  }
  return af::SUCCESS;
}

Status UnAlignmentStrategy::ModifyIndirectLoadVectorizedStrides(ascir::ImplGraph &impl_graph) {
  bool has_transpose = false;
  bool has_nddma_load = false;
  for (const auto &node : impl_graph.GetAllNodes()) {
    has_transpose = has_transpose || node->attr.api.compute_type == af::ComputeType::kComputeTranspose;
    has_nddma_load = has_nddma_load || (ScheduleUtils::IsLoad(node) && node->attr.type == "Nddma");
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
    if (role != ascgen_utils::indirect_load::TemplateRole::kStridedUbPath &&
        role != ascgen_utils::indirect_load::TemplateRole::kSimdInputPreStridedUbPath) {
      continue;
    }
    for (const auto &output : node->outputs()) {
      if (!output->attr.vectorized_axis.empty()) {
        GE_ASSERT_SUCCESS(
            BaseAlignmentStrategy::SetVectorizedStridesForTensor(node, output->attr, AlignmentType::kAligned));
      }
    }
  }
  // SIMT 后置 Reduce 链的 Store/Reduce 按对齐行距读写 UB local target，链上成员需同步垫整
  PadSimtPostReduceChainRowStrides(impl_graph);
  // Only the direct IL->Store boundary inherits the transposed/NDDMA layout.
  // Other IL stores (for example, stores after elementwise or reduce nodes) use
  // their own layout and must not be normalized here.
  if (has_transpose || has_nddma_load) {
    for (const auto &node : impl_graph.GetAllNodes()) {
      if (!af::ops::IsOps<af::ascir_op::Store>(node) || node->inputs.Size() != 1UL || node->outputs().size() != 1UL) {
        continue;
      }
      const auto producer = ascgen_utils::indirect_load::GetInputProducer(node, 0UL);
      if (af::ops::IsOps<af::ascir_op::IndirectLoad>(producer)) {
        GE_ASSERT_SUCCESS(NormalizeStoreSingletonStrides(node));
      }
    }
  }
  return af::SUCCESS;
}

af::Status UnAlignmentStrategy::AddRemovePadForTailBroadcastSrc(af::AscGraph &impl_graph) {
  std::vector<af::AscNodePtr> broadcast_nodes;
  for (const auto &ge_node : impl_graph.GetAllNodes()) {
    const auto node = std::dynamic_pointer_cast<af::AscNode>(ge_node);
    if (node == nullptr || node->attr.api.compute_type != af::ComputeType::kComputeBroadcast) {
      continue;
    }
    if (!IsTailBroadcastNode(*node)) {
      continue;
    }
    const auto &in_nodes = node->GetInDataNodes();
    if (ScheduleUtils::IsScalarLikeNode(in_nodes.at(0UL))) {
      continue;
    }
    if (!TailBrcSrcNeedsRemovePad(impl_graph, node)) {
      continue;
    }
    broadcast_nodes.push_back(node);
  }
  for (const auto &node : broadcast_nodes) {
    GE_ASSERT_SUCCESS(InsertRemovePadBeforeBroadcast(impl_graph, node));
  }
  return af::SUCCESS;
}

Status UnAlignmentStrategy::ModifyVectorizedStrides(af::AscGraph &impl_graph) {
  if (ascgen_utils::indirect_load::FindIndirectLoadNode(impl_graph) != nullptr) {
    GE_ASSERT_SUCCESS(ModifyIndirectLoadVectorizedStrides(impl_graph));
  } else {
    bool has_transpose = false;
    for (auto node : impl_graph.GetAllNodes()) {
      if (af::ops::IsOps<af::ascir_op::Transpose>(node)) {
        has_transpose = true;
        break;
      }
    }
    if (has_transpose) {
      GE_ASSERT_SUCCESS(ModifyTransposeFusionVectorizedStrides(impl_graph, BaseAlignmentStrategy::GetAlignWidth()));
    }
  }
  return AddRemovePadForTailBroadcastSrc(impl_graph);
}
}  // namespace optimize
