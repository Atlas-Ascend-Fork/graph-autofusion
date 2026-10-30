/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.huawei.com
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPTIMIZE_PLATFORM_COMMON_GRAPH_PASS_SOFTMAX_PATTERN_FUSION_UTILS_H
#define OPTIMIZE_PLATFORM_COMMON_GRAPH_PASS_SOFTMAX_PATTERN_FUSION_UTILS_H

#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace optimize {
namespace softmax_pattern {

// 稳定 Softmax Pattern 的完整匹配结果。input_anchor 是 Pattern 的原始输入
// （同时供给 Max 和 Sub，即原始输入回流点），替换后成为专用 Softmax 节点的输入。
struct MatchResult {
  af::OutDataAnchorPtr input_anchor;
  af::AscNodePtr max_node;
  af::AscNodePtr max_broadcast_node;
  af::AscNodePtr sub_node;
  af::AscNodePtr exp_node;
  af::AscNodePtr sum_node;
  af::AscNodePtr sum_broadcast_node;
  af::AscNodePtr true_div_node;
};

// 完整稳定 Softmax 匹配：结构（TrueDiv/Exp/Sum/Sub/Max/Broadcast 链）+
// dtype 支持 + 尾轴归约 + 布局一致 + 消费者闭合。与全图 RunPass 使用同一规则，
// 任何一处不满足都返回 false，不做宽松匹配。
bool MatchStableStructure(const af::AscNodePtr &true_div_node, MatchResult &pattern);
bool MatchStableDedicated(const af::AscNodePtr &true_div_node, MatchResult &pattern);
bool MatchStable(const af::AscNodePtr &true_div_node, MatchResult &pattern);

// 将命中的 Pattern 替换为专用 Softmax 节点并移除原节点。
af::Status ReplaceWithSoftmax(af::AscGraph &graph, const MatchResult &pattern);

// 定位以指定 IndirectLoad 输出为原始输入的直接后置稳定 Softmax Pattern（仅匹配，
// 不改图，未命中返回 false）。供需要感知 softmax 形态但不做替换的调用方（如 SIMT
// UB 下界模型）与 NormalizeDirectPostSoftmax 复用同一匹配规则。
bool FindDirectPostSoftmaxPattern(const af::AscGraph &graph, const af::AscNodePtr &indirect_load, MatchResult &pattern);

// 在候选图副本内查找以指定 IndirectLoad 输出为原始输入的稳定 Softmax Pattern
// 并逐一替换。仅处理 input_anchor 直接来自该 IndirectLoad 的 Pattern：
// 中间 Elementwise 链形态作为后续扩展，当前不做宽松回溯。
// 返回是否发生替换；替换失败属于候选级失败，调用方丢弃副本即可。
af::Status NormalizeDirectPostSoftmax(af::AscGraph &graph, const af::AscNodePtr &indirect_load, bool &changed);

}  // namespace softmax_pattern
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_COMMON_GRAPH_PASS_SOFTMAX_PATTERN_FUSION_UTILS_H
