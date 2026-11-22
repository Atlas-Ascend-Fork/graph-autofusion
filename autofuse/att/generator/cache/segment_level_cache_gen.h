/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_SEGMENT_LEVEL_CACHE_GEN_H_
#define ATT_SEGMENT_LEVEL_CACHE_GEN_H_

#include <map>
#include <string>
#include "base/model_info.h"
#include "generator_config.h"
#include "ge_common_af/ge_api_types_af.h"
#include "common/code_printer.h"

namespace att {
namespace cache {
// 动态shape分段缓存的group扁平槽位键：(asc_graph_id, impl_graph_id, group_id)
using SegmentGroupSlotKey = std::array<size_t, 3UL>;

/**
 * @brief 动态shape分段缓存代码生成器
 * 生成按shape特征分类（3个布尔公式→8种分段）的模板选择结果缓存：
 * 公式①(R轴)：∏(R轴原始轴大小×dtype) < R轴阈值(默认256)，无R轴为true；
 * 公式②(最内轴)：所有load/store最内轴 原始轴大小×dtype < 最内轴阈值(默认128)，全部满足为true；
 * 公式③(分核)：∏(bind_multicore轴的原始轴大小) < 可用核数(tiling前的block_dim)。
 * 三个布尔值组合为key(0..7)，每key一个槽：首填后命中复用，命中case求解失败失效自愈重填。
 */
class SegmentLevelCacheGen {
 public:
  /**
   * @brief 读取分段缓存公式阈值（优先取TilingScheduleConfigTable覆写，默认256/128）
   */
  static void GetThresholds(const TilingModelInfo &tiling_model_info, uint32_t &r_axis_threshold,
                            uint32_t &innermost_threshold);

  /**
   * @brief 构建 (asc_graph_id, impl_graph_id, group_id) → 扁平槽位编号 的映射
   *        按三元组升序枚举，保证类型生成与使用侧编号一致
   */
  static std::map<SegmentGroupSlotKey, size_t> BuildGroupSlotMap(const TilingModelInfo &tiling_model_info);

  /**
   * @brief 生成分段缓存类型定义（SegmentGroupCase/SegmentCacheEntry/SegmentCacheContext）
   *        输出到kState公共头，供跨翻译单元的函数签名引用
   */
  static af::Status GenSegmentCacheTypes(ge::CodePrinter &code_printer, size_t asc_graph_num, size_t group_slot_num);

  /**
   * @brief 生成分段缓存线程本地存储与查找/保存/失效辅助函数
   *        输出到owner翻译单元的函数体（单group为其group TU，多group为tail TU）
   */
  static af::Status GenSegmentCacheStorage(ge::CodePrinter &code_printer, uint32_t r_axis_threshold,
                                           uint32_t innermost_threshold);

  /**
   * @brief 生成ComputeSegmentKey函数（3公式，短路链防整数溢出）
   * @param code_printer 代码打印器（函数体）
   * @param tiling_model_info Tiling模型信息（用于收集三类原始轴并映射运行时accessor）
   * @param config 生成器配置
   */
  static af::Status GenKeyComputeFunc(ge::CodePrinter &code_printer, const TilingModelInfo &tiling_model_info,
                                      const TilingCodeGenConfig &config);
};
}  // namespace cache
}  // namespace att

#endif  // ATT_SEGMENT_LEVEL_CACHE_GEN_H_
