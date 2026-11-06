/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_SCHEDULE_UTILS_H_
#define OPTIMIZE_SCHEDULE_UTILS_H_

#include <unordered_map>

#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "asc_graph_utils.h"
#include "ascir_ops_utils.h"
#include "ascgen_log.h"
#include "ascir.h"
#include "ascir_ops.h"
#include "common/platform_context.h"
#include "schedule_result.h"

namespace optimize {
// 按loop_axis连通区域划分出的for循环节点分组，loop_groups中的下标即for循环编号
struct LoopGroup {
  int64_t loop_axis = af::kIdNone;  // 组内统一的循环轴ID
  std::vector<af::Node *> nodes;    // 组内节点，首个节点为该组的扩散种子
};

class ScheduleUtils {
 public:
  static af::ComputeType GetComputeType(const af::AscNodePtr &node) {
    return node->attr.api.compute_type;
  }

  // 后端默认采用逆dfs的拓扑排序方式
  static Status TopologicalSorting(af::AscGraph &graph, bool use_rdfs_v2 = false);

  // 判断图是否满足按循环轴分组的前置条件：
  // 1. 所有reduce节点的loop_axis均已赋值（保证该处理在AutoScheduler之后才生效）；
  // 2. 全图有效loop_axis种类数 > 1（-1不计入，单一循环轴无需分组）。
  // reduce节点数量 > 1 的检查与条件2重复，非必须，仅用于控制影响范围。
  static bool IsNeedLoopGrouping(const af::AscGraph &graph);

  // 按loop_axis连通区域将图中节点划分为多个for循环分组，供TopologicalSorting使用：
  // 同循环节点在拓扑排序中相邻排列，避免被其它循环轴的节点穿插而硬拆成多个循环。
  // 以当前拓扑序的第一个reduce节点为初始种子，从种子沿输入/输出方向BFS扩散，
  // loop_axis相同或为-1的连通节点归入同一分组（-1节点并入当前组并继续扩散）；
  // loop_axis不同的有效节点作为新分组种子继续扩散（种子记录发现轮次与loop_axis），
  // 同一轮扩散中发现的同轴种子归入同一分组，不同轮发现的同轴区域拆分为不同分组。
  // 分组编号从0开始，即loop_groups的下标。
  // 规则约束：如下
  // 1. 同一分组内有效节点的loop_axis必然相同，loop_axis为-1的节点按所在组的轴参与排序；
  //    同轮发现的同轴种子（即使互不连通）归入同一分组，不同轮发现的同轴区域拆分为不同编号的分组；
  // 2. loop_axis为-1的节点（如Data/Output等Buffer节点）并入当前分组并继续扩散，不作为新分组种子，
  //    无法从首个reduce扩散到达的节点不会出现在分组结果中；
  // 3. 调用方需保证存在loop_axis已赋值的reduce节点（建议先经IsNeedLoopGrouping检查）。
  // 输出：loop_groups存储各分组及其轴，node_to_group记录节点到分组编号的映射，未入组节点不在其中。
  static Status BuildLoopGroups(const af::AscGraph &graph, std::vector<LoopGroup> &loop_groups,
                                std::unordered_map<af::Node *, size_t> &node_to_group);

  static bool IsElewise(const af::AscNodePtr &node) {
    return node->attr.api.compute_type == af::ComputeType::kComputeElewise;
  }

  static bool IsBroadcast(const af::AscNodePtr &node) {
    return node->attr.api.compute_type == af::ComputeType::kComputeBroadcast;
  }

  static bool IsReduce(const af::AscNodePtr &node) {
    return node->attr.api.compute_type == af::ComputeType::kComputeReduce;
  }

  static bool IsTranspose(const af::AscNodePtr &node) {
    return node->attr.api.compute_type == af::ComputeType::kComputeTranspose;
  }

  static bool IsLoad(const af::AscNodePtr &node) {
    return node->attr.api.compute_type == af::ComputeType::kComputeLoad;
  }

  static bool IsStore(const af::AscNodePtr &node) {
    return node->attr.api.compute_type == af::ComputeType::kComputeStore;
  }

  static bool IsConcat(const af::AscNodePtr &node) {
    return node->attr.api.compute_type == af::ComputeType::kComputeConcat;
  }

  static bool IsSplit(const af::AscNodePtr &node) {
    return node->attr.api.compute_type == af::ComputeType::kComputeSplit;
  }

  static bool IsGather(const af::AscNodePtr &node) {
    return node->attr.api.compute_type == af::ComputeType::kComputeGather;
  }

  static bool IsGatherLikeLoad(const af::AscNodePtr &node) {
    const auto template_id = ::ascir::GetTemplateIdOrDefault(*node);
    return af::ops::IsOps<af::ascir_op::Gather>(node) || template_id == ascir::TemplateId::kIndirectLoadSimd ||
           template_id == ascir::TemplateId::kIndirectLoadSK;
  }

  static bool IsBuffer(const af::AscNodePtr &node) {
    return node->attr.api.type == af::ApiType::kAPITypeBuffer;
  }

  static bool IsCompute(const af::AscNodePtr &node) {
    return node->attr.api.type == af::ApiType::kAPITypeCompute;
  }

  static bool IsCube(const af::AscNodePtr &node) {
    return node->attr.api.compute_type == af::ComputeType::kComputeCube;
  }

  static bool HasComputeType(const ascir::ImplGraph &impl_graph, const af::ComputeType compute_type);

  static bool IsIOBuffer(const af::NodePtr &node) {
    // IndexExpr 是无输入标量表达式节点(sched.axis 为空), 与 Scalar 同属不参与
    // 全 reduce 展维的标量路径, 需一并跳过, 否则空 axis 触发展维断言。
    return af::ops::IsOps<af::ascir_op::Scalar>(node) || af::ops::IsOps<af::ascir_op::IndexExpr>(node) ||
           IsDataInput(node) || af::ops::IsOps<af::ascir_op::Output>(node);
  }

  static bool IsDataInput(const af::NodePtr &node) {
    return af::ops::IsOps<af::ascir_op::Data>(node) || af::ops::IsOps<af::ascir_op::ScalarData>(node);
  }

  static bool IsDataInput(const af::Node *const node) {
    return af::ops::IsOps<af::ascir_op::Data>(node) || af::ops::IsOps<af::ascir_op::ScalarData>(node);
  }

  // IndexExpr 是标量表达式节点(输出无 axis/repeats), 与 Scalar/ScalarData 同属
  // 标量语义; 标量路径的调度/缓存判定需将其一并识别, 否则以标量为输入的节点
  // 会因空 axis 与输出轴数不一致而触发断言。
  static bool IsScalarLikeNode(const af::NodePtr &node) {
    return af::ops::IsOps<af::ascir_op::Scalar>(node) || af::ops::IsOps<af::ascir_op::ScalarData>(node) ||
           af::ops::IsOps<af::ascir_op::IndexExpr>(node);
  }

  static bool IsScalarLikeNode(const af::Node *const node) {
    return af::ops::IsOps<af::ascir_op::Scalar>(node) || af::ops::IsOps<af::ascir_op::ScalarData>(node) ||
           af::ops::IsOps<af::ascir_op::IndexExpr>(node);
  }

  static bool IsConstantScalar(const af::Node *const node) {
    return af::ops::IsOps<af::ascir_op::Scalar>(node) || af::ops::IsOps<af::ascir_op::IndexExpr>(node);
  }

  static bool IsRemovePad(const af::NodePtr &node) {
    return af::ops::IsOps<af::ascir_op::RemovePad>(node);
  }

  template <typename T>
  static bool IsNodeSupportDataType(const af::DataType data_type) {
    std::string npu_arch;
    GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
    std::vector exp_dtypes{data_type};
    if (T::InferDataType({data_type}, exp_dtypes, npu_arch) != af::SUCCESS) {
      GELOGD("%s does not support dtype=%s", T::Type, af::TypeUtils::DataTypeToSerialString(data_type).c_str());
      return false;
    }
    return true;
  }

  // 获取 npu_arch 并调用 InferDataType
  template <typename OpType>
  static Status CallAscirInferDataType(const std::vector<af::DataType> &input_dtypes,
                                       std::vector<af::DataType> &expect_output_dtypes) {
    std::string npu_arch;
    GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
    return OpType::InferDataType(input_dtypes, expect_output_dtypes, npu_arch);
  }

  static bool GetGatherParams(af::AscGraph &graph, int64_t &attr_axis, int64_t &params_size) {
    bool has_gather = false;
    for (const auto &node : graph.GetAllNodes()) {
      if (!IsGather(node)) {
        continue;
      }
      has_gather = true;
      auto input_axis = node->inputs[1].attr.axis;
      GE_ASSERT_TRUE(!input_axis.empty(), "input attr is invalid.");
      auto output_axis = node->outputs[0].attr.axis;
      for (auto iter = output_axis.begin(); iter != output_axis.end(); ++iter) {
        auto axis = graph.FindAxis(*iter);
        GE_ASSERT_NOTNULL(axis);
        if (axis->id == input_axis[0] ||
            std::find(axis->from.begin(), axis->from.end(), input_axis[0]) != axis->from.end()) {
          attr_axis = std::distance(output_axis.begin(), iter);
          GELOGD("Axis index:[%ld], axis_id:[%ld].", attr_axis, axis->id);
        }
      }
      params_size = node->inputs[0].attr.repeats.size();
      break;
    }
    return has_gather;
  }

  /**
   * 从ComputeGraph上获取AscGraphAttr属性，如果graph或者获取到的ascGraphAttr是空，则抛出异常。
   */
  static af::AscGraphAttr *GetOrCreateGraphAttrsGroup(const af::ComputeGraphPtr &graph) {
    GE_CHECK_NOTNULL_EXEC(graph, return nullptr;);
    auto attr = graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
    GE_CHECK_NOTNULL_EXEC(attr, return nullptr;);
    return attr;
  }

  /**
   * 给定node，根据其所属Graph的类型，获取循环轴。
   * 若是HintGraph，则获取图属性中的axis信息；
   * 否则ImplGraph，则直接从node的属性中获取axis信息
   */
  static Status GetLoopAxis(const af::AscNode &node, std::vector<int64_t> &axes) {
    auto attr = GetOrCreateGraphAttrsGroup(node.GetOwnerComputeGraph());
    GE_CHECK_NOTNULL(attr, "Get ascgraph type failed, attr is null.");
    if (attr->type == af::AscGraphType::kImplGraph) {
      GELOGD("GetLoopAxis Impl graph, node.name=%s", node.GetNamePtr());
      axes.insert(axes.end(), node.attr.sched.axis.begin(), node.attr.sched.axis.end());
      return af::SUCCESS;
    }
    axes.clear();
    axes.reserve(attr->axis.size());
    for (const auto &graph_axis : attr->axis) {
      GELOGD("[GetLoopAxis] node %s[%s] axis.id=%ld, axis.size=%s", node.GetTypePtr(), node.GetNamePtr(),
             graph_axis->id, graph_axis->size.Str().get());
      axes.push_back(graph_axis->id);
    }
    GELOGD("[GetLoopAxis] attr.axis.size=%zu, ids.size=%zu", attr->axis.size(), axes.size());
    return af::SUCCESS;
  }

  /**
   * 从HintGraph的图属性中，获取repeats，直接获取每个轴的size即可。
   */
  static Status GetLoopRepeats(const af::AscNode &node, std::vector<ascir::SizeExpr> &repeats) {
    auto attr = GetOrCreateGraphAttrsGroup(node.GetOwnerComputeGraph());
    GE_CHECK_NOTNULL(attr, "Get ascgraph type failed, attr is null.");
    repeats.clear();
    repeats.reserve(attr->axis.size());
    for (const auto &axis : attr->axis) {
      GELOGD("[GetLoopRepeats] node %s[%s] axis.id=%ld, axis.size=%s", node.GetTypePtr(), node.GetNamePtr(), axis->id,
             axis->size.Str().get());
      repeats.push_back(axis->size);
    }
    GELOGD("[GetLoopRepeats] attr.axis.size=%zu, repeats.size=%zu", attr->axis.size(), repeats.size());
    return af::SUCCESS;
  }

  /**
   * 从HintGraph的图属性中，获取strides信息。需要根据repeat从后往前计算出来，尾轴的stride=1，然后依次累乘repeat
   */
  static Status GetReduceInputStrides(af::AscNode &node, std::vector<ascir::SizeExpr> &strides) {
    auto attr = GetOrCreateGraphAttrsGroup(node.GetOwnerComputeGraph());
    GE_CHECK_NOTNULL(attr, "Get ascgraph type failed, attr is null.");
    if (attr->type == af::AscGraphType::kImplGraph) {
      strides = node.inputs[0].attr.strides;
      return af::SUCCESS;
    }
    strides.resize(attr->axis.size());
    ascir::SizeExpr basic_stride = af::Symbol(1);
    for (int64_t i = static_cast<int64_t>(attr->axis.size()) - 1; i >= 0; --i) {
      if (af::SymbolicUtils::StaticCheckEq(attr->axis[i]->size, af::sym::kSymbolOne) == af::TriBool::kTrue) {
        strides[i] = af::Symbol(0);
      } else {
        strides[i] = basic_stride;
        basic_stride = basic_stride * attr->axis[i]->size;
      }
      GELOGD("[GetLoopStrides] node %s[%s] axis.id=%ld, axis.strides=%s", node.GetTypePtr(), node.GetNamePtr(),
             attr->axis[i]->id, strides[i].Str().get());
    }
    GELOGD("[GetLoopStrides] attr.axis.size=%zu", attr->axis.size());
    return af::SUCCESS;
  }

  static std::vector<af::AxisId> CalcReduceAxes(const std::vector<af::Expression> &src_strides,
                                                const std::vector<af::Expression> &dst_strides,
                                                const std::vector<ascir::AxisId> &axes);

  template <typename T>
  static Status GetNodeIrAttrValue(const af::NodePtr &node, const string &attr_name, T &attr_value) {
    auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
    GE_ASSERT_NOTNULL(asc_node);
    GE_ASSERT_NOTNULL(asc_node->attr.ir_attr);
    asc_node->attr.ir_attr->GetAttrValue(attr_name, attr_value);  // 允许获取失败
    return af::SUCCESS;
  }

  static Status GetNodeIrAttrIndex(const af::NodePtr &node, int64_t &index) {
    return GetNodeIrAttrValue(node, "index", index);
  }

  static Status GetNodeIrAttrOffset(const af::NodePtr &node, af::Expression &offset) {
    auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
    GE_ASSERT_NOTNULL(asc_node);
    GE_ASSERT_NOTNULL(asc_node->attr.ir_attr);
    return asc_node->attr.ir_attr->GetAttrValue("offset", offset);
  }

  static bool IsAxisStrideAllZero(const std::vector<ascir::AxisId> &origin_ids,
                                  const std::vector<ascir::SizeExpr> &axis_strides,
                                  const std::vector<ascir::AxisId> &axis_ids) {
    for (uint64_t i = 0; i < axis_ids.size(); i++) {
      for (uint64_t j = 0; j < origin_ids.size(); j++) {
        if (axis_ids[i] != origin_ids[j]) {
          continue;
        }
        if (af::SymbolicUtils::StaticCheckEq(axis_strides[j], af::ops::Zero) == af::TriBool::kTrue) {
          return true;
        }
      }
    }
    return false;
  }

  static bool IsBroadcastNeedMemUnique(const ascir::NodeView &node, const std::vector<ascir::AxisId> &axis_ids) {
    if (IsBuffer(node) || node->GetInDataNodesSize() == 0) {
      return false;
    }
    if (IsLoad(node)) {
      auto output = node->outputs[0];
      bool is_all_zero = IsAxisStrideAllZero(output.attr.axis, output.attr.strides, axis_ids);
      return is_all_zero;
    }

    for (const auto &in : node->inputs()) {
      GE_ASSERT_NOTNULL(in);
      auto prev_node = std::dynamic_pointer_cast<af::AscNode>(in->anchor.GetOwnerNode());
      if (IsBroadcastNeedMemUnique(prev_node, axis_ids)) {
        return true;
      }
    }
    return false;
  }

  static bool IsNextNodeRemovePad(const ascir::NodeView &node);

  static bool IsContinuesBroadcast(const std::vector<af::Expression> &in_repeats,
                                   const std::vector<af::Expression> &out_repeats);

  static bool IsContinuesStrides(const std::vector<af::Expression> &repeats,
                                 const std::vector<af::Expression> &strides);

  static bool IsContinuesVecStrides(const ascir::NodeView &node);

  static bool IsVectorizedAxisContinuousInGM(const af::AscTensorAttr &output_tensor);

  // 尾轴在GM上连续: 从尾轴向前跳过stride为0的广播轴, 第一个非零stride为1
  static bool IsTailAxisContinuousInGM(const af::AscTensorAttr &output_tensor);

  static bool IsLastAxisSliceLoad(const af::AscNodePtr &node);

  static bool NotNeedAlignVectorStride(const af::AscGraph &graph);

  static bool IsIntervalBroadcast(const std::vector<af::Expression> &in_repeats,
                                  const std::vector<af::Expression> &out_repeats);

  static bool GetTailAxisDataSize(const af::AscNodePtr &node, uint32_t &size);
  // 获取节点输出尾轴的原始大小表达式：若尾轴为切分轴（Tile/Block Inner/Outer），
  // 沿 Axis::from 回溯至原始轴（或合并轴），返回其 size 表达式（输入变量或常量）。
  // 返回 false 表示尾轴缺失或轴 id 无效，调用方应回退到 repeats 尾轴。
  static bool GetOriginTailDimExpr(const af::AscGraph &graph, const af::AscNodePtr &node,
                                   af::Expression &tail_dim_expr);
  static bool IsTailAxisLessThan(const af::AscNodePtr &node, const uint32_t value);
  static bool IsTailAxisAlignedBy(const af::AscNodePtr &node, const uint32_t align_bytes = 32);

  static bool IsStaticShape(const ascir::NodeView &node);

  static bool IsStaticGraph(const af::AscGraph &graph);

  static Status GetNonBrcInputTensor(const ascir::NodeView &node, const size_t index,
                                     std::unique_ptr<af::AscTensor> &tensor);

  static bool IsComputeNodes(const ascir::NodeView &node) {
    for (auto &out_node : node->GetOutDataNodes()) {
      auto cur_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (cur_node->attr.api.type == af::ApiType::kAPITypeCompute && !IsBroadcast(cur_node)) {
        return true;
      }
    }
    return false;
  }

  static Status RemoveUnusedAxes(af::AscGraph &graph);

  static void NormalizeAxisIds(const af::AscGraph &graph);

  // 将 graph.sched.axis 作为统一输入轴下发到各节点，保留节点级 loop_axis 和 exec_condition。
  // graph.sched.loop_axis 当前为预留字段，若配置则记录错误并重置为 kIdNone（-1）。
  static Status ApplyGraphSchedAxisToNodes(const af::AscGraph &graph);

  static Status GetVectorRepeats(const std::vector<af::Expression> &repeats, const std::vector<int64_t> &axis,
                                 const std::vector<int64_t> &vector_axis, std::vector<af::Expression> &vector_repeats);
  static Status GetNodeInputVectorRepeats(const ascir::NodeView &node, std::vector<af::Expression> &vector_repeats);
  static Status GetNodeOutVectorRepeats(const ascir::NodeView &node, std::vector<af::Expression> &vec_repeats);
  static Status GetVectorAxisStrides(const af::AscTensorAttr &attr, std::vector<af::Expression> &vec_strides);
  static Status GetConcatDim(const af::AscNodePtr &node, size_t &concat_dim);

  static std::string AxesToString(const std::vector<af::AxisPtr> &axes);
  static bool IsNormStruct(const ascir::ImplGraph &implGraph);
  static bool IsReduceArFullLoad(const ascir::ImplGraph &implGraph);
  static bool HasSameInput(const af::AscNodePtr &node);
  static bool IsLastAxisReduce(const ascir::ImplGraph &impl_graph);
  static bool IsReduceOnTailAxis(const af::AscNodePtr &node);
  static bool IsScalarBroadcastNode(const ascir::NodeView &node);
  static bool IsScalarBrc(const af::AscNodePtr &node);
  static Status SwapInputIndex(const ascir::NodeView &node, const int32_t idx1, const int32_t idx2);
  static Status GetInputForTranspose(af::AscNode &node, std::vector<ascir::AxisId> &input_axis);
  template <typename T>
  static af::AscNodePtr FindFirstNodeOfType(const af::AscGraph &graph) {
    for (const auto &node : graph.GetAllNodes()) {
      if (af::ops::IsOps<T>(node)) {
        return node;
      }
    }
    return nullptr;
  }
  static Status RemoveNode(const ascir::ImplGraph &impl_graph, const af::AscNodePtr &node,
                           const af::OutDataAnchorPtr &pre_out_anchor);
  static Status RemoveNodeDst(const ascir::ImplGraph &impl_graph, const af::AscNodePtr &node,
                              const af::InDataAnchorPtr &next_in_anchor);
  static bool FindContinuesBroadcastNode(const ascir::NodeView &node, vector<af::AscNodePtr> &continues_brc_nodes);

  static Status AddRemovePadAfter(af::AscGraph &graph, const af::AscNodePtr &node, af::AscNodePtr &remove_pad_node,
                                  const int32_t idx = 0);
  static bool IsOutNodeWithMultiInputs(const af::AscNodePtr &node);
  static Status ResolveDiffDim(const af::AscNodePtr &node, size_t &diff_dim, bool &is_first_dim);
  static Status RecalculateStridesFromRepeats(const std::vector<af::Expression> &repeats,
                                              std::vector<af::Expression> &strides);
  static bool IsNeedDiscontinuousAligned(const af::AscTensorAttr &attr);
  static Status ClearAllSizeVar(const af::AscGraph &graph);
  // 判断节点的Micro API是否支持Scalar输入，用于scalar_broadcast优化
  static bool IsMicroApiSupportsScalarInput(const af::AscNodePtr &node);
  static void GenerateStrides(const std::vector<ge::Expression> &repeats, std::vector<ge::Expression> &strides);
};
}  // namespace optimize

#endif  // OPTIMIZE_SCHEDULE_UTILS_H_
