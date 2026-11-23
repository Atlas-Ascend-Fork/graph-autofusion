/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/ub_expr/asc_graph_ub_expr_builder.h"

#include <algorithm>
#include <set>

#include "ascir_ops.h"
#include "common/checker.h"
#include "common_utils.h"
#include "graph/types_af.h"
#include "indirect_load_utils.h"
#include "schedule_result.h"

namespace ascir {
namespace {
constexpr int64_t kMinTmpBufferSize = 8 * 1024;
// [边界化] 边界化列分块/逐 tile 调用时单块最大 8192 个元素（fp32 为 32KB），无显式 tmp
// 的节点复用该兜底缓冲，8KB 会越界导致结果错误。仅在含边界化 IL 的图上提升，其他图
// 维持 8KB，保证非边界化场景的 UB 估算逐字节不变。
constexpr int64_t kBoundaryFusionMinTmpBufferSize = 32 * 1024;
constexpr int64_t kBlockAlignBytes = 32;
// GetTmpBuffer 会为 tmp 公式套上 v1 repeat cap 壳: Min(size, 255 * 256 + 32)。
// CV UBFuse 的 epilogue 内联公式按原始需求分配 tmp, host UB 约束须与之一致, 否则
// 求解器会选出使 tmp 越界的 stage 尺寸。数值须与 default_reg_func.cpp 的 cap 保持一致。
constexpr int64_t kMaxV1RepeatTmpBytes = 255LL * 256LL + 32LL;

UbExpr StripV1RepeatCap(UbExpr size) {
  if (!size.IsValid()) {
    return size;
  }
  auto args = size.GetArgs();
  if (args.size() != 2UL) {
    return size;
  }
  for (size_t i = 0UL; i < args.size(); i++) {
    int64_t cap_value = 0;
    if (args[i].IsConstExpr() && args[i].GetConstValue(cap_value) && cap_value == kMaxV1RepeatTmpBytes) {
      return args[1UL - i];
    }
  }
  return size;
}

std::string MakeQueueName(int64_t id) {
  return "q" + std::to_string(id) + "_size";
}

std::string MakeBufferName(int64_t id) {
  return "b" + std::to_string(id) + "_size";
}

std::string GetQueueName(const af::AscTensorAttr &attr) {
  return attr.que.name.empty() ? MakeQueueName(attr.que.id) : attr.que.name;
}

std::string GetBufferName(const af::AscTensorAttr &attr) {
  return attr.buf.name.empty() ? MakeBufferName(attr.buf.id) : attr.buf.name;
}

uint32_t GetTypeSize(ge::DataType dtype) {
  uint32_t type_size = 0U;
  if (!ge::TypeUtils::GetDataTypeLength(dtype, type_size)) {
    return 1U;
  }
  return type_size > 0U ? type_size : 1U;
}

UbExpr AlignToBlock(const UbExpr &size) {
  return af::sym::Mul(af::Symbol(kBlockAlignBytes), af::sym::Ceiling(af::sym::Div(size, af::Symbol(kBlockAlignBytes))));
}

bool HasPrintableExpr(const UbExpr &expr) {
  if (!expr.IsValid()) {
    return false;
  }
  const auto expr_str = expr.Str();
  return expr_str != nullptr && expr_str[0] != '\0';
}

UbExpr TensorVectorizedElementSize(const af::AscGraph &graph, const af::AscTensorAttr &attr) {
  if (attr.vectorized_axis.empty()) {
    return af::sym::kSymbolOne;
  }
  for (size_t i = 0UL; i < attr.vectorized_axis.size(); ++i) {
    if (i >= attr.vectorized_strides.size()) {
      continue;
    }
    const auto &stride = attr.vectorized_strides[i];
    if (stride == af::sym::kSymbolZero) {
      continue;
    }
    if (attr.axis.empty() && i < attr.repeats.size()) {
      return stride == af::sym::kSymbolOne ? attr.repeats[i] : attr.repeats[i] * stride;
    }
    auto axis_iter = std::find(attr.axis.cbegin(), attr.axis.cend(), attr.vectorized_axis[i]);
    if (axis_iter == attr.axis.cend()) {
      GELOGW("[AscGraphUbExprBuilder] cannot find vectorized axis %ld in tensor attr of graph %s",
             attr.vectorized_axis[i], graph.GetName().c_str());
      return {};
    }
    const auto axis_index = static_cast<size_t>(std::distance(attr.axis.cbegin(), axis_iter));
    if (axis_index >= attr.repeats.size()) {
      GELOGW("[AscGraphUbExprBuilder] vectorized axis %ld repeat index %zu is invalid in graph %s",
             attr.vectorized_axis[i], axis_index, graph.GetName().c_str());
      return {};
    }
    return stride == af::sym::kSymbolOne ? attr.repeats[axis_index] : attr.repeats[axis_index] * stride;
  }
  return af::sym::kSymbolOne;
}

UbExpr TensorBytes(const af::AscGraph &graph, const af::AscTensorAttr &attr) {
  return AlignToBlock(TensorVectorizedElementSize(graph, attr) * af::Symbol(GetTypeSize(attr.dtype)));
}

void AppendUniqueVar(const UbExpr &var, std::vector<UbExpr> &vars) {
  if (!HasPrintableExpr(var)) {
    return;
  }
  const auto iter = std::find_if(vars.cbegin(), vars.cend(), [&var](const auto &item) { return item == var; });
  if (iter == vars.cend()) {
    vars.emplace_back(var);
  }
}

void CollectVars(const UbExpr &expr, std::vector<UbExpr> &vars) {
  if (!HasPrintableExpr(expr)) {
    return;
  }
  for (const auto &var : expr.FreeSymbols()) {
    AppendUniqueVar(var, vars);
  }
}

void AppendMax(UbExpr &expr, const UbExpr &item) {
  if (!HasPrintableExpr(item)) {
    return;
  }
  expr = HasPrintableExpr(expr) ? af::sym::Max(expr, item) : item;
}

void AppendAdd(UbExpr &expr, const UbExpr &item) {
  if (!HasPrintableExpr(item)) {
    return;
  }
  expr = HasPrintableExpr(expr) ? expr + item : item;
}

struct ContainerState {
  std::string name;
  UbExpr normal_max;
  uint32_t buf_num = 0U;
  std::map<int64_t, UbExpr> share_group_bytes;
};

void AddContainerTensor(const af::AscGraph &graph, const af::AscTensorAttr &attr, ContainerState &state) {
  const auto bytes = TensorBytes(graph, attr);
  AppendMax(state.normal_max, bytes);
  if (attr.mem.reuse_id == af::kIdNone) {
    return;
  }
  AppendAdd(state.share_group_bytes[attr.mem.reuse_id], bytes);
}

void AddQueueTensor(const af::AscGraph &graph, const af::AscTensorAttr &attr, ContainerState &state) {
  if (state.name.empty()) {
    state.name = GetQueueName(attr);
  }
  AddContainerTensor(graph, attr, state);
  state.buf_num = std::max(state.buf_num, static_cast<uint32_t>(std::max<int64_t>(attr.que.buf_num, 0)));
}

void AddBufferTensor(const af::AscGraph &graph, const af::AscTensorAttr &attr, ContainerState &state) {
  if (state.name.empty()) {
    state.name = GetBufferName(attr);
  }
  AddContainerTensor(graph, attr, state);
}

UbExpr ContainerSlotBytes(const ContainerState &state) {
  UbExpr size = state.normal_max;
  for (const auto &item : state.share_group_bytes) {
    AppendMax(size, item.second);
  }
  return size;
}

UbExpr QueueTotalBytes(const ContainerState &state, const UbExpr &slot_bytes) {
  if (!slot_bytes.IsValid()) {
    return {};
  }
  const uint32_t buf_num = state.buf_num == 0U ? 1U : state.buf_num;
  return buf_num == 1U ? slot_bytes : slot_bytes * af::Symbol(buf_num);
}

bool IsUbAlloc(const af::AscTensorAttr &attr) {
  return attr.mem.hardware == af::MemHardware::kMemHardwareUB &&
         (attr.mem.alloc_type == af::AllocType::kAllocTypeQueue ||
          attr.mem.alloc_type == af::AllocType::kAllocTypeBuffer);
}

bool IsUbTmpBuffer(const af::TmpBuffer &tmp_buffer) {
  return tmp_buffer.id != af::kIdNone;
}

void AddContainer(UbExprContext &context, const std::string &name, const UbExpr &expr, const UbExpr &total_expr) {
  if (!HasPrintableExpr(expr) || !HasPrintableExpr(total_expr)) {
    return;
  }
  const UbExpr symbol = af::Symbol(name.c_str());
  context.container_expr[symbol] = expr;
  context.container_names[symbol] = name;
  AppendAdd(context.ub_expr, total_expr);
  CollectVars(expr, context.ub_related_vars);
}

void AddContainer(UbExprContext &context, const std::string &name, const UbExpr &expr) {
  const UbExpr symbol = af::Symbol(name.c_str());
  AddContainer(context, name, expr, symbol);
}

bool IsConstZero(const UbExpr &expr) {
  int64_t value = 0;
  return HasPrintableExpr(expr) && expr.GetConstValue(value) && value == 0;
}

void AddBuiltinTmpBuffer(const af::AscGraph &graph, UbExprContext &context) {
  const auto builtin_tmp_buffer = ascgen_utils::CalcExtraTmpBufForAscGraph(graph);
  if (IsConstZero(builtin_tmp_buffer)) {
    return;
  }
  AppendAdd(context.ub_expr, builtin_tmp_buffer);
  CollectVars(builtin_tmp_buffer, context.ub_related_vars);
}

UbExpr CalcReservedUbSize(const af::AscGraph &graph) {
  UbExpr reserved_ub_size = af::Symbol(ascgen_utils::CalcReservedTmpBufSizeForAscGraph(graph));
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    const int64_t simt_dcache_size = ::ascir::GetDcacheSize(*node);
    if (simt_dcache_size > 0) {
      reserved_ub_size = reserved_ub_size + af::Symbol(simt_dcache_size);
      break;
    }
  }
  return reserved_ub_size;
}

void AddReservedUb(const af::AscGraph &graph, UbExprContext &context) {
  const auto reserved_ub_size = CalcReservedUbSize(graph);
  if (IsConstZero(reserved_ub_size)) {
    return;
  }
  AppendAdd(context.ub_expr, reserved_ub_size);
}

bool IsDynamicSizeVar(const af::SizeVarPtr &size_var) {
  return size_var != nullptr && HasPrintableExpr(size_var->expr) && !size_var->expr.IsConstExpr();
}

void FillSizeVars(const af::AscGraph &graph, UbExprContext &context) {
  for (const auto &size_var : graph.GetAllSizeVar()) {
    if (IsDynamicSizeVar(size_var)) {
      AppendUniqueVar(size_var->expr, context.dynamic_size_vars);
    }
  }
}

void FillTileVars(const af::AscGraph &graph, UbExprContext &context) {
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis == nullptr || !HasPrintableExpr(axis->size)) {
      continue;
    }
    if (axis->type == af::Axis::kAxisTypeTileInner) {
      context.var_min_values[axis->size] = af::sym::kSymbolOne;
      AppendUniqueVar(axis->size, context.ub_related_vars);
      continue;
    }
    if (axis->type == af::Axis::kAxisTypeOriginal && !axis->size.IsConstExpr()) {
      AppendUniqueVar(axis->size, context.dynamic_size_vars);
    }
  }
}

void AddTmpBuffer(const af::TmpBuffer &tmp_buffer, bool use_raw_tmp_size,
                  std::map<int64_t, UbExpr> &node_tmp_buffer_bytes) {
  if (!IsUbTmpBuffer(tmp_buffer) || !HasPrintableExpr(tmp_buffer.buf_desc.size)) {
    return;
  }
  const UbExpr size = use_raw_tmp_size ? StripV1RepeatCap(tmp_buffer.buf_desc.size) : tmp_buffer.buf_desc.size;
  AppendAdd(node_tmp_buffer_bytes[tmp_buffer.id], size);
}

void MergeNodeTmpBuffers(const std::map<int64_t, UbExpr> &node_tmp_buffer_bytes,
                         std::map<int64_t, ContainerState> &buffer_bytes, int64_t min_tmp_buffer_size) {
  for (const auto &item : node_tmp_buffer_bytes) {
    AppendMax(buffer_bytes[item.first].normal_max, af::sym::Max(item.second, af::Symbol(min_tmp_buffer_size)));
  }
}

// [列分块] 把节点的全部输入 producer 加入闭包（已收集则跳过）。
void CollectProducerClosure(af::AscNode *node, std::set<const af::Node *> &collected) {
  std::vector<const af::AscNode *> pending;
  for (const auto &input : node->inputs()) {
    if (input == nullptr || input->anchor.GetOwnerNode() == nullptr) {
      continue;
    }
    auto *producer = dynamic_cast<af::AscNode *>(input->anchor.GetOwnerNode().get());
    if (producer != nullptr && collected.insert(producer).second) {
      pending.push_back(producer);
    }
  }
  while (!pending.empty()) {
    auto *cur = const_cast<af::AscNode *>(pending.back());
    pending.pop_back();
    for (const auto &input : cur->inputs()) {
      if (input == nullptr || input->anchor.GetOwnerNode() == nullptr) {
        continue;
      }
      auto *producer = dynamic_cast<af::AscNode *>(input->anchor.GetOwnerNode().get());
      if (producer != nullptr && collected.insert(producer).second) {
        pending.push_back(producer);
      }
    }
  }
}

// direct-GM 节点仅在作为 SIMT post-Reduce local target 时计入 UB（其余由 GM 直读）。
bool IsSimtLocalTargetInGraph(const af::AscGraph &graph, const af::AscNodePtr &node) {
  for (const auto &candidate : graph.GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::IndirectLoad>(candidate)) {
      continue;
    }
    const auto local_target = ascgen_utils::indirect_load::GetSimtPostReduceLocalProducer(candidate);
    if (local_target != nullptr && local_target->GetName() == node->GetName()) {
      return true;
    }
  }
  return false;
}

// [列分块] 收集列分块 IL 的输入链节点（input 表/索引走 GM 直读，不驻留 UB）。仅当
// IndirectLoad 带 column_tile_inner 标记（boundary fusion 列分块场景）时生效，其他图
// 该集合为空，行为零变化。has_boundary_fusion_il 标记图中是否存在边界化 IL，用于决定
// 无显式 tmp 节点的兜底缓冲尺寸。
std::set<const af::Node *> CollectColumnTileInputNodes(const af::AscGraph &graph, bool &has_boundary_fusion_il) {
  std::set<const af::Node *> column_tile_input_nodes;
  has_boundary_fusion_il = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (node == nullptr || !af::ops::IsOps<af::ascir_op::IndirectLoad>(node)) {
      continue;
    }
    if (ascgen_utils::indirect_load::IsSimtBoundaryFusion(node)) {
      has_boundary_fusion_il = true;
    }
    if (ascgen_utils::indirect_load::GetSimtColumnTileInner(node) <= 0L) {
      continue;
    }
    has_boundary_fusion_il = true;
    CollectProducerClosure(node.get(), column_tile_input_nodes);
  }
  return column_tile_input_nodes;
}

void AddTensor(const af::AscGraph &graph, const af::AscTensorAttr &attr, std::map<int64_t, ContainerState> &queue_bytes,
               std::map<int64_t, ContainerState> &buffer_bytes) {
  if (!IsUbAlloc(attr)) {
    return;
  }
  if (attr.mem.alloc_type == af::AllocType::kAllocTypeQueue && attr.que.id != af::kIdNone) {
    AddQueueTensor(graph, attr, queue_bytes[attr.que.id]);
    return;
  }
  if (attr.mem.alloc_type == af::AllocType::kAllocTypeBuffer && attr.buf.id != af::kIdNone) {
    AddBufferTensor(graph, attr, buffer_bytes[attr.buf.id]);
  }
}

}  // namespace

af::Status AscGraphUbExprBuilder::Build(const af::AscGraph &graph, UbExprContext &context,
                                        bool use_raw_tmp_size) const {
  context = UbExprContext{};
  context.graph_name = graph.GetName();
  context.tiling_case_id = graph.GetTilingKey();
  context.use_raw_tmp_size = use_raw_tmp_size;
  FillSizeVars(graph, context);
  FillTileVars(graph, context);

  // [列分块] 收集列分块 IL 的输入链节点（input 表/索引走 GM 直读，不驻留 UB）。
  // 仅当 IndirectLoad 带 column_tile_inner 标记（boundary fusion 列分块场景）时生效，
  // 其他图/场景该集合为空，行为零变化。输入链回溯到 Data（GM）为止。
  bool has_boundary_fusion_il = false;
  const auto column_tile_input_nodes = CollectColumnTileInputNodes(graph, has_boundary_fusion_il);

  std::map<int64_t, ContainerState> queue_bytes;
  std::map<int64_t, ContainerState> buffer_bytes;
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (column_tile_input_nodes.count(node.get()) != 0UL) {
      continue;  // [列分块] IL 输入链走 GM 直读，不驻留 UB
    }
    if (ascgen_utils::indirect_load::GetTemplateBehavior(node).uses_direct_gm_pipeline &&
        !IsSimtLocalTargetInGraph(graph, node)) {
      continue;
    }
    for (const auto &output : node->outputs()) {
      AddTensor(graph, output->attr, queue_bytes, buffer_bytes);
    }
    std::map<int64_t, UbExpr> node_tmp_buffer_bytes;
    for (const auto &tmp_buffer : node->attr.tmp_buffers) {
      AddTmpBuffer(tmp_buffer, use_raw_tmp_size, node_tmp_buffer_bytes);
    }
    MergeNodeTmpBuffers(node_tmp_buffer_bytes, buffer_bytes,
                        has_boundary_fusion_il ? kBoundaryFusionMinTmpBufferSize : kMinTmpBufferSize);
  }

  for (const auto &item : queue_bytes) {
    const auto name = item.second.name.empty() ? MakeQueueName(item.first) : item.second.name;
    const auto symbol = af::Symbol(name.c_str());
    const auto slot_bytes = ContainerSlotBytes(item.second);
    AddContainer(context, name, slot_bytes, QueueTotalBytes(item.second, symbol));
  }
  for (const auto &item : buffer_bytes) {
    const auto name = item.second.name.empty() ? MakeBufferName(item.first) : item.second.name;
    AddContainer(context, name, ContainerSlotBytes(item.second));
  }
  AddBuiltinTmpBuffer(graph, context);
  AddReservedUb(graph, context);
  return af::SUCCESS;
}

}  // namespace ascir
