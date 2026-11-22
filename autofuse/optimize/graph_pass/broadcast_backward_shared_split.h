/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPTIMIZE_GRAPH_PASS_BROADCAST_BACKWARD_SHARED_SPLIT_H
#define OPTIMIZE_GRAPH_PASS_BROADCAST_BACKWARD_SHARED_SPLIT_H

#include "base_graph_pass.h"

namespace optimize {
namespace broadcast_backward_shared_split {

Status SplitSharedBroadcastBranches(af::AscGraph &graph);
// 消费者共享拆分：多消费 Broadcast 链与"Brc 直连多消费 Cast"（Cast 输出视图会被
// partial backward 公共轴塌缩殃及全部消费分支）均按消费者整链拆分为单链，后移收
// 益交由主循环逐链判定。
Status SplitSharedBroadcastConsumers(af::AscGraph &graph);

// SplitSharedBroadcastConsumers 的反向动作：后移主循环结束后，把仍从同一 source
// 出发、逐节点视图/dtype/sched 完全一致且整链未能后移的纯单消费广播链合并回一条
// 共享链（链尾恢复多消费；链尾直连的等价 Cast 一并合并去重），并将后移改写后幸存
// 的拆分克隆名（Broadcast/Cast）回退为原始链名（原名已不存在时），消除"拆分后无
// 后移收益"的重复广播/重复 Cast 与命名漂移。
Status MergeSharedBroadcastConsumers(af::AscGraph &graph);

}  // namespace broadcast_backward_shared_split
}  // namespace optimize

#endif  // OPTIMIZE_GRAPH_PASS_BROADCAST_BACKWARD_SHARED_SPLIT_H
