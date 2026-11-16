/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_NODE_PARTITION_UTILS_H_
#define ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_NODE_PARTITION_UTILS_H_

#include "ascir.h"
#include "ascgen_log.h"
#include <string>
#include <vector>

namespace optimize {
// 判定 pre_node 到 broadcast_node 是否为存在首轴扩展的 Broadcast(首轴 B 轴扩展, 其余轴不限)。
// 仅做结构判定, 不限制首轴大小(常量/动态轴均可识别), 供不限大小的 Broadcast 切分场景识别使用
bool IsFirstAxisBroadcast(const af::AscNodePtr &pre_node, const af::AscNodePtr &broadcast_node);

// 判定 pre_node 到 broadcast_node 是否为仅首轴扩展的 Broadcast(首轴 B 轴扩展, 其余轴不变)。
// 仅做结构判定, 不限制首轴大小(常量/动态轴均可识别)
bool IsOnlyFirstAxisBroadcast(const af::AscNodePtr &pre_node, const af::AscNodePtr &broadcast_node);

// 克隆节点的调度属性与全部输出 tensor 属性
Status DoCopyAscNodeTensorAttr(const af::AscNodePtr &src_node, af::AscNodePtr &dst_node);

// 克隆节点的调度属性与指定源输出(按索引)的 tensor 属性到 dst 的首输出
Status DoCopyAscNodeOutputTensorAttr(const af::AscNodePtr &src_node, af::AscNodePtr &dst_node,
                                     const uint32_t src_output_index);

// 克隆 workspace 节点的 dtype(与数据源输出对齐)
Status DoCopyWorkspaceTensorAttr(const af::AscNodePtr &src_node, af::AscNodePtr &workspace_node);

// 在 src_node -> dst_node 的数据边处建立切分:
// 1. src 为 Load/Scalar 类/Data 输入时复制输入节点, 断开原边后由副本供 dst;
// 2. 其余节点建立 workspace 链 src -> store -> ws_pre ⫽ ws_post -> load -> dst,
//    ws_pre 与 ws_post 同名标识同一 workspace 缓冲, 连通性分组时两侧自然断开。
// partition_flag 置位表示发生切分, node_order 记录切分源节点(连通性分组种子, 与汇点种子配合使用)。
Status PartitionByNode(af::AscNodePtr &src_node, af::AscNodePtr &dst_node, ascir::ImplGraph &impl_graph,
                       bool &partition_flag, std::vector<af::AscNodePtr> &node_order);
}  // namespace optimize

#endif  // ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_NODE_PARTITION_UTILS_H_
