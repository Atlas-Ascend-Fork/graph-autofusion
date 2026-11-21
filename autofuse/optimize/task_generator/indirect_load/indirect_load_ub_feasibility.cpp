/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load/indirect_load_ub_feasibility.h"
#include "task_generator/indirect_load/indirect_load_view_layout.h"
#include "task_generator/indirect_load/indirect_load_input_path.h"
#include "task_generator/indirect_load/indirect_load_output_analysis.h"
#include "task_generator/indirect_load/indirect_load_template_plan.h"
#include "task_generator/indirect_load/indirect_load_template_annotation.h"
#include "task_generator/indirect_load/indirect_load_axis_normalize.h"
#include "task_generator/indirect_load/indirect_load_score_generator.h"
#include "task_generator/indirect_load/indirect_load_sk_partition.h"
#include "task_generator/indirect_load/indirect_load_simt_emission.h"
#include "task_generator/indirect_load/indirect_load_group_candidate.h"

#include "task_generator/indirect_load/indirect_load_graph_utils.h"

#include "v35/ascir/ascir_codegen_v2.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"

namespace optimize {
namespace indirect_load_detail {
bool TryGetIndirectLoadPlatformUbSize(int64_t &ub_size) {
  ge::PlatformInfo platform_info;
  if (!ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(platform_info)) {
    return false;
  }
  ub_size = platform_info.ub_size;
  return ub_size > 0;
}

bool TryGetTensorLogicalBytes(const af::AscTensor &tensor, int64_t &bytes) {
  int64_t count = 1;
  for (const auto &repeat : tensor.attr.repeats) {
    int64_t value = 0;
    if (!repeat.GetConstValue(value) || value <= 0) {
      return false;  // 动态 shape 保守放行
    }
    count *= value;
  }
  bytes = count * static_cast<int64_t>(af::GetSizeByDataType(tensor.attr.dtype));
  return true;
}

// [SK段SIMT化] SK 行窗口对当前图形态是否退化为全量不可行：
// IndirectLoadSk 设备 API 的窗口 = axis 起完整 payload 后缀 × 窗口行数，index 需
// 全量驻留该段 UB。index 源头不收敛（逐元素独立引用，沿 gather 轴枚举任意行号）
// 时窗口行数必须覆盖 axis 维全部 → 窗口 = input 全量 + index 全量，超出单核 UB
// 即该形态的 SK 不可行（调度期必然被 StaticUbTemplateFilter 淘汰）。
// 源头收敛（引用按行/列成组）的形态窗口只需被引用分片，不受此限。
bool IsSkWindowUbInfeasible(const af::AscNodePtr &indirect_load) {
  // 现场判定收敛性：AccessInfo attr 写在先跑的 SIMD/SIMT 候选图副本上，SK 候选
  // 副本拷贝自原图时为空，跨副本读取会把收敛形态误判为非收敛而误触发重定向
  // （原版 user_layernorm：收敛 index + 大表被误重定向，产生 dtype 错位的 kernel）。
  if (ascgen_utils::indirect_load::IsIndexSourceConverged(indirect_load)) {
    return false;  // 引用成组：窗口按分片装载，不按全量判定
  }
  int64_t ub_size = 0;
  if (!TryGetIndirectLoadPlatformUbSize(ub_size)) {
    return false;  // 平台信息不可得时保守走原 SK 路径
  }
  const auto &inputs = indirect_load->inputs();
  int64_t input_bytes = 0;
  int64_t index_bytes = 0;
  if (!TryGetTensorLogicalBytes(*inputs[ascgen_utils::indirect_load::kInputTensorIndex], input_bytes) ||
      !TryGetTensorLogicalBytes(*inputs[ascgen_utils::indirect_load::kIndexTensorIndex], index_bytes)) {
    return false;  // 动态 shape 保守走原 SK 路径
  }
  if (input_bytes + index_bytes > ub_size) {
    GELOGI("[IndirectLoad] SK window (generic access) bytes %ld (input %ld + index %ld) exceeds platform UB %ld.",
           input_bytes + index_bytes, input_bytes, index_bytes, ub_size);
    return true;
  }
  return false;
}

// 统计 Norm 闭包内尾轴仍为完整 R 的中间张量，作为 fused SIMT 的附加 R 域因子。
// Reduce 统计量尾轴折叠后自然排除，Store/Output 不重复计入基础输出双缓冲。
int64_t CountRDomainChainTensors(const af::AscNodePtr &indirect_load, int64_t tail) {
  int64_t count = 0;
  std::vector<af::AscNodePtr> pending;
  std::unordered_set<const af::AscNode *> visited;
  for (const auto &consumer : indirect_load->GetOutDataNodes()) {
    pending.emplace_back(std::dynamic_pointer_cast<af::AscNode>(consumer));
  }
  for (size_t cursor = 0UL; cursor < pending.size() && cursor < 256UL; ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (node == nullptr || !visited.emplace(node.get()).second) {
      continue;
    }
    if (!af::ops::IsOps<af::ascir_op::Store>(node) && !af::ops::IsOps<af::ascir_op::Output>(node)) {
      const auto node_outputs = node->outputs();
      int64_t node_tail = 0;
      if (!node_outputs.empty() && node_outputs[0] != nullptr && !node_outputs[0]->attr.repeats.empty() &&
          node_outputs[0]->attr.repeats.back().GetConstValue(node_tail) && node_tail == tail) {
        ++count;
      }
    }
    for (const auto &consumer : node->GetOutDataNodes()) {
      pending.emplace_back(std::dynamic_pointer_cast<af::AscNode>(consumer));
    }
  }
  return count;
}

// [SK段SIMT化] SIMT 候选自身的 UB 可行性（硬下界）：输出 tile 双缓冲（2R）+
// Reduce 输入完整 R 域（R）+ 预留（32KB），R=IL 输出尾轴尺寸×dtype；softmax
// 形态（原始 Pattern 链；窗口不可行的 SK 候选不规范化，见 Generate）fused 还需
// SoftmaxAR tmp 的 1 个 R 域（生产实测 [4,8,16000] fp32 需 288800 ≈ 4R+32K）。
// 下界漏算该 tmp 会把 R 落在 3R+32K ≤ UB < 4R+32K 区间的 softmax 图误判为
// SIMT 可行——重定向不触发而 SK 窗口又不可行，全候选被 UB 过滤。其余复合 Norm
// 链按闭包内尾轴满 R 的中间张量数量扩展 R 域因子。SIMT 可行时 SK 候选无需重定向——
// 重定向是为 SK 行窗口不可行且无其他活路时提供的分段替代；SIMT 可用的图（如
// softmax 专用形态带 Scalar 标量链/多输入，R 可全载）走原 SIMT 内联路径，分段形态
// 对其不支持（生产 9.28：SIMT 可行却重定向，边界化误伤标量输入链）。
bool IsSimtUbInfeasible(const af::AscGraph &graph, const af::AscNodePtr &indirect_load) {
  int64_t ub_size = 0;
  if (!TryGetIndirectLoadPlatformUbSize(ub_size)) {
    return false;
  }
  // outputs() 每次调用重建内部快照，先前返回的引用会悬垂——一次性拷贝所需数据。
  const auto outputs = indirect_load->outputs();
  if (outputs.empty() || outputs[0] == nullptr || outputs[0]->attr.repeats.empty()) {
    return false;
  }
  const auto output_repeats = outputs[0]->attr.repeats;
  const auto output_dtype = outputs[0]->attr.dtype;
  int64_t tail = 0;
  if (!output_repeats.back().GetConstValue(tail) || tail <= 0) {
    return false;  // 动态 shape 保守不重定向
  }
  const int64_t r_bytes = tail * static_cast<int64_t>(af::GetSizeByDataType(output_dtype));
  softmax_pattern::MatchResult softmax_match;
  int64_t r_domain_factor = kSimtRDomainUbFactor;
  if (softmax_pattern::FindDirectPostSoftmaxPattern(graph, indirect_load, softmax_match)) {
    // softmax Pattern 的 fused 候选已规范化为单节点（链中间张量不存在），仅需补
    // SoftmaxAR tmp 的 1 个 R 域。
    r_domain_factor += 1;
  } else {
    // 复合 Norm 链：fused SIMT 把闭包内每个尾轴满 R 的中间张量各自物化为独立
    // UB buffer，逐个计入下界。
    r_domain_factor += CountRDomainChainTensors(indirect_load, tail);
  }
  return r_domain_factor * r_bytes + kIndirectLoadUbReservedBytes > ub_size;
}

// 组 SIMD 硬下界：共循环流水要求全部成员的 input 切片同时驻留 UB，最小 tile
// （gather 轴 1 行）时的下界 = Σ 成员切片字节 + 预留。基于原始图成员的 tensor
// 属性计算（gather 轴后缀 repeats × dtype），不依赖候选图的改写后视图。
bool IsGroupSimdUbDoomed(const std::vector<af::AscNodePtr> &members) {
  int64_t ub_size = 0L;
  if (!TryGetIndirectLoadPlatformUbSize(ub_size)) {
    return false;
  }
  int64_t slice_floor_sum = 0L;
  for (const af::AscNodePtr &member : members) {
    const size_t axis_index = GetIndirectLoadAxisIndex(member);
    if (axis_index == kIndirectLoadInvalidAxisIndex) {
      return false;
    }
    const auto inputs = member->inputs();
    if (inputs.size() <= ascgen_utils::indirect_load::kInputTensorIndex) {
      return false;
    }
    const auto &input_attr = inputs[ascgen_utils::indirect_load::kInputTensorIndex]->attr;
    int64_t slice_bytes = static_cast<int64_t>(af::GetSizeByDataType(input_attr.dtype));
    for (size_t dim = axis_index + 1UL; dim < input_attr.repeats.size(); ++dim) {
      int64_t dim_size = 0L;
      if (!input_attr.repeats[dim].GetConstValue(dim_size) || dim_size <= 0L) {
        return false;  // 动态 shape 保守放行
      }
      slice_bytes *= dim_size;
    }
    slice_floor_sum += slice_bytes;
  }
  return slice_floor_sum + kIndirectLoadUbReservedBytes > ub_size;
}

// [组 SIMT N 块 value tile UB 下界·唯一实现] 成员 value 写 UB 的形态（L3/L4）共用：
// 静态尾轴可算时下界 = N × r_bytes + 预留，超平台 UB 即放不下。平台信息不可得或
// 动态 R（尾轴非常量）保守放行（交由 tiling 期资源校验淘汰、SK 兜底接管）。
// 调用方一（生成期预检 IsGroupSimtL4UbDoomed）：形态 2 必然 L3/L4，直接判；
// 调用方二（ApplyGraphPassGroup 计划落定后）：非 L2（!register_merge）统一判——
// 覆盖形态 1 压力降级 L3 与 L4（组员形态 value 仍写 N 块 UB）。两处禁止各持副本，
// 常量/判据变更必须同步。
bool IsGroupSimtNValueTileUbDoomed(const std::vector<af::AscNodePtr> &members) {
  int64_t ub_size = 0L;
  if (!TryGetIndirectLoadPlatformUbSize(ub_size)) {
    return false;
  }
  const auto outputs = members.front()->outputs();
  if (outputs.empty() || outputs[0] == nullptr || outputs[0]->attr.repeats.empty()) {
    return false;
  }
  int64_t tail = 0L;
  if (!outputs[0]->attr.repeats.back().GetConstValue(tail) || tail <= 0L) {
    return false;  // 动态 shape 保守放行
  }
  const int64_t r_bytes = tail * static_cast<int64_t>(af::GetSizeByDataType(outputs[0]->attr.dtype));
  return static_cast<int64_t>(members.size()) * r_bytes + kIndirectLoadUbReservedBytes > ub_size;
}

// [组 SIMT UB 预检·形态化] 与寄存器常数解耦：按"是否需要落 N 块 UB"判——共享
// post-Reduce 终端（形态 2）必然 L3/L4（成员 N 份 value tile 驻留下界 = N×R + 预留），
// 该下界超平台 UB 即跳过 SIMT 候选；无共享 Reduce（形态 1）L2 可能成立（无 N 块
// 需求），预检放行，L2→L3 压力降级的 UB 可行性由计划落定后的统一补检与 tiling 期
// 资源校验淘汰、SK 兜底接管。
bool IsGroupSimtL4UbDoomed(const std::vector<af::AscNodePtr> &members) {
  if (ascgen_utils::indirect_load::GetPostReduceConsumer(members.front()) == nullptr) {
    return false;  // 形态 1：L2 无 N 块 UB 需求，不在预检跳过
  }
  return IsGroupSimtNValueTileUbDoomed(members);
}

// [编译期 UB 绝对下限硬门] 平台常量级问题（与 shape 无关）：任何形态都放不下
// N×32B 最小行 + 32KB 预留即拒绝组融合候选（SIMD/SIMT 一并跳过，SK 兜底按其自身
// 窗口可行性另行判定）——避免产出必然 tiling 失败的候选。ub_size 不可得（0/未知）
// 时不判，交给运行期 solver。
bool IsGroupUbAbsoluteDoomed(const std::vector<af::AscNodePtr> &members) {
  int64_t ub_size = 0L;
  if (!TryGetIndirectLoadPlatformUbSize(ub_size) || ub_size <= 0L) {
    return false;
  }
  return kIndirectLoadUbReservedBytes + static_cast<int64_t>(members.size()) * 32L > ub_size;
}
}  // namespace indirect_load_detail
}  // namespace optimize
