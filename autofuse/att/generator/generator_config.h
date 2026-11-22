/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_GENERATOR_CONFIG_H_
#define ATT_GENERATOR_CONFIG_H_
#include <string>
#include <sstream>
#include "autofuse_config/auto_fuse_config_utils.h"
#include "base/base_types.h"
#include "autofuse_config/auto_fuse_config.h"
namespace att {
enum class TilingImplType {
  HIGH_PERF,
  MAX,
  AXES_REORDER,
  UNKNOWN,
};

struct TilingCodeGenConfig {
  TilingImplType type;
  std::string path;
  std::string op_name;
  std::string tiling_data_type_name{"TilingData"};
  bool gen_extra_infos{false};
  bool gen_tiling_data{true};
  bool high_precision{true};
  bool enable_small_shape_strategy{false};
  bool enable_multicore_ub_tradeoff{false};
  bool enable_autofuse_pgo{false};
  int64_t pgo_step_max{16};
  bool enable_score_func{false};
  bool is_autofuse{false};
  bool is_inductor_scene{false};
  bool is_cube{false};
  // ub多核权衡策略里ub和多核的阈值
  bool do_variable_replace{true};
  // 临时配置，用于控制变量替换是否开关
  double ub_threshold{0.2};
  double corenum_threshold{0.4};
  bool cache_enabled_at_compile_time{false};  // 编译态缓存开关（默认关闭）
  bool enable_tiling_segment_cache{false};    // 动态shape分段缓存开关（att_accuracy_level=0时开启）
  ge::ForceTilingCaseResult force_tiling_case;
  int64_t force_schedule_result{-1L};
  std::string force_template_op_name;
  std::string Debug() const {
    std::stringstream ss;
    ss << "TilingCodeGenConfig[type(" << static_cast<int32_t>(type) << ")"
       << ", path(" << path << ")"
       << ", tiling_data_type_name(" << tiling_data_type_name << ")"
       << ", gen_extra_infos(" << gen_extra_infos << ")"
       << ", gen_tiling_data(" << gen_tiling_data << ")"
       << ", high_precision(" << high_precision << ")"
       << ", ub_threshold(" << ub_threshold << ")"
       << ", corenum_threshold(" << corenum_threshold << ")"
       << ", enable_small_shape_strategy(" << enable_small_shape_strategy << ")"
       << ", enable_multicore_ub_tradeoff(" << enable_multicore_ub_tradeoff << ")"
       << ", enable_autofuse_pgo(" << enable_autofuse_pgo << ")"
       << ", enable_score_func(" << enable_score_func << ")"
       << ", enable_tiling_segment_cache(" << enable_tiling_segment_cache << ")"
       << ", force_schedule_result(" << force_schedule_result << ")"
       << ", force_tiling_case(" << force_tiling_case.Debug() << ")"
       << "]";
    return ss.str();
  }
};
}  // namespace att
#endif  // ATT_GENERATOR_CONFIG_H
