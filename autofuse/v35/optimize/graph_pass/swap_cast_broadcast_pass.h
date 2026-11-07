/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V2_GRAPH_PASS_SWAP_CAST_BROADCAST_PASS_H
#define OPTIMIZE_PLATFORM_V2_GRAPH_PASS_SWAP_CAST_BROADCAST_PASS_H

#include "optimize/graph_pass/base_graph_pass.h"

namespace optimize {
// 将 load -> cast(低byte向高byte转换) -> broadcast 在预处理阶段交换为
// load -> broadcast -> cast: 广播改在窄dtype上复制、cast后移做宽化, 调度期对齐pass
// 按broadcast实际输出dtype(窄侧)原生生成向量化stride, load->broadcast可直接融合为
// Nddma并继承已对齐的stride。cast逐元素、broadcast纯复制, 两者可交换; 广播的其它
// 消费者改接cast输出后取值不变。
//
// 与SwapCastBrcAndGenNddma(模板期交换)的关系:
// 模板期交换在对齐与缓冲计账之后执行: NDDMA输出dtype变为cast输入侧, 但brc的向量化
// stride是按cast输出侧dtype对齐生成的, 交换后窄侧复用宽侧对齐会不足(如fp32下stride
// 72元素=288B已对齐, fp16下同一stride仅144B), 只能靠gap判据加模板期重排补救。本pass
// 把交换前移到调度之前, 该问题在源头消失; 生效后load->cast->brc(低转高)模式到达模板
// 前已被交换, SwapCastBrcAndGenNddma不再命中该场景, 仅作为兜底保留。
// 手工注释：如果想完全复原原先的场景，可在nddma模板里把原图再swap回来。
//
// 为什么仅限cast由低byte向高byte转换:
// 1. 低->高(如fp16->fp32)是唯一会产生上述对齐不足的方向, 是本pass要解决的对象;
// 2. 高->低(如fp32->fp16)留在模板期交换不会遇到对齐问题: brc的stride按窄侧(2B)已
//    32B对齐, NDDMA换成宽侧(4B)后同一stride字节数翻倍, 天然仍满足32B对齐, 无需本pass
//    提前交换; 若未来高->低场景出于性能等原因需要交换, 可在非nddma模板中处理, 不经过
//    "NDDMA跨dtype继承stride"的路径, 不会引入本问题。
//
// 排除场景: 广播结果被IndirectLoad消费时不交换——间接加载模板按load->cast->brc的
// 链形状识别index语义(地址计算协议), 交换index广播链会破坏其取数正确性。
//
// 时序依赖: 必须注册在BroadcastBackwardPass(公共广播后移)之后, 避免两个pass对同一
// broadcast的改写竞争; 在调度之前执行, 保证对齐pass看到的是交换后的最终结构。
class SwapCastBroadcastPass final : public BaseGraphPass {
 public:
  SwapCastBroadcastPass() = default;
  Status RunPass(af::AscGraph &graph) override;
  ~SwapCastBroadcastPass() override = default;
};
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_V2_GRAPH_PASS_SWAP_CAST_BROADCAST_PASS_H
