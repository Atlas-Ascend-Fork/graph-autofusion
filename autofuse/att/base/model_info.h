/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_MODEL_INFO_H
#define ATT_MODEL_INFO_H
#include <memory>
#include <map>
#include "base/arch_param.h"
#include "base/base_types.h"
#include "base/perf_breakdown.h"
#include "schedule_result.h"
#include "util/ternary_op.h"

namespace att {
// NO_TAIL用于modelifno等式约束表达父轴大小要整除子轴，no_tail对应的表达式应为div
const std::string kFatherToChildNoTail = "NO_TAIL";
// NORMAL 用于modelinfo不等式约束中表达父轴大于子轴， normal对应的表达式应为sub
const std::string kFatherToChildLarger = "NORMAL";
enum class kModelInfoLevel : int32_t {
  K_SCHEDULE_RESULT_LEVEL = 0,
  K_SCHEDULE_GROUP_LEVEL,
  K_INVALID_SCHEDULE_LEVEL,
};

struct SymInfo {
  SymInfo() = default;
  explicit SymInfo(const Expr &e) : symbol_expr(e) {}
  virtual ~SymInfo() = default;
  Expr symbol_expr;
  uint32_t prompt_align{1u};
  uint32_t data_type_size{4U};
  std::pair<int64_t, int64_t> value_range = {-1, -1};
};
using SymInfoPtr = std::shared_ptr<SymInfo>;

struct SymVarInfo : public SymInfo {
  explicit SymVarInfo(const Expr &e) : SymInfo(e) {}
  ~SymVarInfo() override = default;
  Expr align = af::Symbol(1);
  std::vector<HardwareDef> related_scope;
  Expr max_value;
};
using SymVarInfoPtr = std::shared_ptr<SymVarInfo>;

struct SymConstInfo : public SymInfo {
  explicit SymConstInfo(const Expr &e) : SymInfo(e) {}
  ~SymConstInfo() override = default;
  uint32_t const_value{0u};
};
using SymConstInfoPtr = std::shared_ptr<SymConstInfo>;

struct AttAxis {
  std::string name;  // 轴的名称
  AxisPosition
      axis_pos;  // 切分轴的位置。origin原始轴，对应求解问题的输入，inner轴，对应待求解变量，outter轴可由origin和inner轴推导
  bool bind_multicore;               // 是否绑多核
  bool is_last;                      // 原始轴的最内切分轴，用于设定轴的默认初始值
  bool is_node_innerest_dim;         // 是否是某个node的最内轴，用于决定轴的搜索优先级
  bool is_concat_outer_dim;          // 是否是concat node的concat dim外轴
  bool is_concat_inner_dim;          // 是否是concat node的concat dim尾轴
  size_t order{0UL};                 // 轴排序的优先级，值越小优先级越高，默认为0
  SymInfoPtr size;                   // 用于表达轴的size
  std::vector<AttAxis *> orig_axis;  // 原始轴的信息
  std::vector<AttAxis *> from_axis;  // 父轴的信息

  // 分核轴类型标记（用于Store冲突检测）
  bool is_reduce_split_axis{false};     // 该轴是否是Reduce分核轴
  bool is_broadcast_split_axis{false};  // 该轴是否是Broadcast分核轴
};

using AttAxisPtr = std::shared_ptr<AttAxis>;

struct ATTConfig {
  std::vector<std::string> config_names;
  std::map<std::string, std::string> config_value;
};

struct Optional {
  std::string optional_name;
  std::string data_type;
  std::string min_value;
  std::string max_value;
};

struct InputTensor {
  int32_t data_type;
};

struct ScheduleGroupIdent {
  size_t asc_graph_id{0L};   // AscGraph的ID
  size_t impl_graph_id{0L};  // ImplGraph的ID
  size_t group_id{0L};       // ScheduleGroup的ID
  bool operator<(const ScheduleGroupIdent &other) const {
    if (impl_graph_id < other.impl_graph_id) {
      return true;
    } else if (impl_graph_id > other.impl_graph_id) {
      return false;
    }
    // 如果 impl_graph_id 相等，则比较 group_id
    return group_id < other.group_id;
  }
  bool operator==(const ScheduleGroupIdent &other) const {
    return (impl_graph_id == other.impl_graph_id) && (group_id == other.group_id);
  }
  bool operator!=(const ScheduleGroupIdent &other) const {
    return (impl_graph_id != other.impl_graph_id) || (group_id != other.group_id);
  }
  [[nodiscard]] std::string GetGroupPrefix() const {
    return "AscGraph" + std::to_string(asc_graph_id) + "ScheduleResult" + std::to_string(impl_graph_id) + "G" +
           std::to_string(group_id);
  }
  // 小写加短下划线风格：snake_case
  [[nodiscard]] std::string GetGroupPrefixSnakeCase() const {
    return "asc_graph" + std::to_string(asc_graph_id) + "_schedule_result" + std::to_string(impl_graph_id) + "_g" +
           std::to_string(group_id);
  }
  [[nodiscard]] std::string GetItemPrefix() const {
    return "graph" + std::to_string(asc_graph_id) + "_result" + std::to_string(impl_graph_id) + "_g" +
           std::to_string(group_id);
  }
};

struct ReuseScheduleGroupInfo {
  std::vector<std::string> reuse_input_axes;   // 复用的schedule group内所有输入轴名称
  std::vector<std::string> reuse_search_axes;  // 复用的schedule group内所有求解轴名称
  std::vector<uint32_t> tiling_keys;           // 复用的schedule group内对应的tiling key
};
struct ReuseScheduleGroup {
  ScheduleGroupIdent reuse_group_ident;  // 复用的schedule group信息
  ReuseScheduleGroupInfo info;
  std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>
      schedule_group_to_info;  // 所有schedule group对应的轴名称，映射的轴与reuse_axes对应
  bool IsReuseGroup(const ScheduleGroupIdent &schedule_group_ident) const {
    if (reuse_group_ident == schedule_group_ident) {
      return false;
    }
    const auto &iter = schedule_group_to_info.find(schedule_group_ident);
    if (iter == schedule_group_to_info.end()) {
      return false;
    }
    return (info.reuse_search_axes.size() == iter->second.reuse_search_axes.size()) &&
           (info.reuse_input_axes.size() == iter->second.reuse_input_axes.size()) &&
           (info.tiling_keys.size() == iter->second.tiling_keys.size());
  }
};
using ReuseScheduleGroupPtr = std::shared_ptr<ReuseScheduleGroup>;

enum class TilingScheduleConfigPriority : int32_t {
  kDefaultPriority = 0,
  kHeavyOpPriority = 1,
};

struct TradeOffConfig {
  bool is_enable = false;                // 是否使能 multicore-ub tradeoff
  Expr ub_ratio{af::Symbol(0.1)};        // UB 阈值（Expr 类型）
  Expr core_num_ratio{af::Symbol(0.8)};  // 核数比例（Expr 类型）

  [[nodiscard]] std::string DebugString() const {
    return "is_enable: " + std::to_string(is_enable) + ", ub_ratio: " + Str(ub_ratio) +
           ", core_num_ratio: " + Str(core_num_ratio);
  }
};

// Model 级别的 Tiling 调度配置
struct TilingScheduleConfig {
  // 多核UB权衡配置
  TradeOffConfig trade_off_config;

  // CacheLine 大小（字节）
  uint32_t cache_line_size{arch_param::kDefaultCacheLineSize};

  // Vector length 大小（字节）
  uint32_t vector_len_size{arch_param::kDefaultVectorLenSize};

  // 是否启用惩罚配置（用于日志区分）
  bool is_penalty_config{false};

  [[nodiscard]] std::string DebugString() const {
    return "trade_off_config: {" + trade_off_config.DebugString() +
           "}, cache_line_size: " + std::to_string(cache_line_size) +
           ", vector_len_size: " + std::to_string(vector_len_size) +
           ", is_penalty_config: " + std::to_string(is_penalty_config);
  }
};

enum class CacheLineDirection : uint8_t {
  kUnknown = 0,
  kGmToUb = 1,
  kUbToGm = 2,
};

// Contract: In non-PGO group-parallel path, GenGetTilingbyCaseId generates
// tiling_data.set_tiling_key(tiling_case_id), so runtime get_tiling_key()
// returns the case_id used at codegen time. This allows mapping final_key -> helper.
// Trace: GenGetScheduleResult -> group-parallel branch -> GenGroupParallelFirstTiling
//   (uses tiling_case_id) -> DoGroupTiling stores group tiling_key via
//   group_var.get_tiling_key(). GenGetTilingbyCaseId calls GenerateInputParamsAndTiling
//   which emits tiling_data.set_tiling_key(tiling_case_id).
struct CacheLineConfig {
  CacheLineConfig() = default;
  CacheLineConfig(const std::string &name, const Expr &expr, uint32_t size,
                  CacheLineDirection cache_line_direction = CacheLineDirection::kUnknown,
                  const Expr &solver_expr = Expr())
      : node_name(name),
        cache_line_expr(expr),
        cache_line_size(size),
        direction(cache_line_direction),
        solver_cache_line_expr(solver_expr) {}

  std::string node_name;
  // Single transfer bytes (repeats.back() * data_type_size), used by conflict detection
  Expr cache_line_expr;
  uint32_t cache_line_size;
  CacheLineDirection direction{CacheLineDirection::kUnknown};
  // Total block bytes (accumulate(input_dims) * data_type_size), used by solver UB constraint.
  // Empty when total bytes is a compile-time constant (solver should skip).
  Expr solver_cache_line_expr;
  std::string ToString() const {
    return "node: " + node_name + ", expr: " + Str(cache_line_expr) + ", size: " + std::to_string(cache_line_size);
  }
  [[nodiscard]] bool IsCacheLineConflictCandidate() const {
    return direction == CacheLineDirection::kGmToUb || direction == CacheLineDirection::kUbToGm;
  }
};

struct RuntimeReorderRule {
  Expr condition_axis;
  Expr compare_axis;
  uint32_t condition_threshold{0U};
  uint32_t compare_threshold{0U};
  std::vector<Expr> preferred_order;
};

class TilingScheduleConfigTable {
 public:
  virtual ~TilingScheduleConfigTable() = default;
  [[nodiscard]] virtual bool IsEnableBlockLoopAutoTune() const = 0;
  [[nodiscard]] virtual bool IsEnableCacheLineCheck() const = 0;
  [[nodiscard]] virtual TradeOffConfig GetTradeOffConfig() const = 0;
  [[nodiscard]] virtual TilingScheduleConfigPriority GetConfigPriority() const {
    return TilingScheduleConfigPriority::kDefaultPriority;
  }
  // ub利用率大于该值时，性能公式在模板选择是才会生效
  [[nodiscard]] virtual double GetUbThresholdPerfValEffect() const = 0;
  // 模板比较时，相差超过该值时，才会使用性能公式进行比较，否则直接比较ub利用率
  [[nodiscard]] virtual double GetPerfEffectVal() const {
    constexpr double kDefaultPerfEffectVal = 5000.0;
    return kDefaultPerfEffectVal;
  }

  // 新增：获取 Model 级别的 Tiling 调度配置
  [[nodiscard]] virtual TilingScheduleConfig GetModelTilingScheduleConfig() const = 0;

  // 新增：获取 CacheLine 大小
  [[nodiscard]] virtual uint32_t GetCacheLineSize() const = 0;

  // 新增：获取 Vector length 大小
  [[nodiscard]] virtual uint32_t GetVectorLenSize() const {
    return arch_param::kDefaultVectorLenSize;
  }

  // 新增：是否启用Reduce分核Store地址冲突惩罚功能
  [[nodiscard]] virtual bool IsCoreNumThresholdPenaltyEnable() const = 0;
};

struct TilingCaseIdent {
  ScheduleGroupIdent schedule_group_ident;
  uint32_t tiling_case_id;
  std::string sub_case_tag;
};

struct ModelInfo {
  uint32_t tiling_case_id;
  std::string graph_name;
  std::string score_func;
  std::string semantic_name;
  std::string sub_case_tag;                    // Reduce切R，优先切R轴的模板
  std::map<int64_t, Expr> workspace_size_map;  // 用于描述每个tensor_id的workspace大小
  std::map<HardwareDef, Expr> hardware_cons;   // 用于描述硬件约束
  Expr reserved_ub_size{CreateExpr(0)};
  std::map<std::string, std::vector<std::pair<Expr, Expr>>>
      eq_exprs;  // 用于描述等式约束,切分轴之间的整除约束key值为NO_TAIL
  std::map<std::string, std::vector<Expr>> leq_exprs;              // 用于描述不等式约束
  std::map<std::string, NodeApiTilingCode> node_name_to_api_code;  // 用于定义高阶API的代码
  std::map<std::string, std::pair<std::string, std::string>>
      tiling_api_name_to_vars;       // 工具场景高阶API使用, API名,高阶API变量名,高阶API变量类型
  std::map<PipeType, Expr> objects;  // 用于描述目标表达式
  std::vector<AttAxisPtr> arg_list;  // 用于描述轴以及轴size的信息, owner att axis ptr
  std::map<std::string, Expr> container_exprs;
  std::map<std::string, Expr> tensor_exprs;
  Expr head_cost{CreateExpr(0)};                           // 用于描述多核头开销
  ScheduleGroupIdent schedule_group_ident;                 // 标记graph的schedule group信息
  ReuseScheduleGroupPtr reuse_schedule_group;              // 标记reuse group信息
  ExprExprMap variable_expr_map;                           // 用于记录tensor的表达式
  std::map<Expr, std::string, ExprCmp> variable_name_map;  // 用于记录tensor的名称
  std::map<Expr, TernaryOp, ExprCmp> ternary_op_map;       // 用于记录三目运算符的名称
  std::vector<PerfBreakdownGroup> perf_breakdowns;         // 用于记录性能公式的语义化拆解
  uint32_t output_size;
  std::vector<af::AscNodePtr> input_nodes;    // 获取输入schedule_results[0].input_nodes
  std::vector<af::AscNodePtr> output_nodes;   // 获取输入出schedule_results[0].output_nodes
  bool enable_group_parallel{false};          // 使能group并行
  std::vector<Expr> sizes{};                  // 图原始Sizes信息
  vector<CacheLineConfig> cache_line_config;  // ub->gm/gm->ub节点的cache配置信息
  const TilingScheduleConfigTable *tiling_schedule_config_table{nullptr};
  TilingScheduleConfig tiling_schedule_config;            // Model 级别的 Tiling 调度配置
  bool is_enable_equal_order_tiling{false};               // 使能等order tiling算法
  std::vector<RuntimeReorderRule> runtime_reorder_rules;  // 运行时按shape调整单模板内轴优先级
};

using TilingModelInfo = std::vector<ModelInfo>;
using GroupsTilingModelInfo = std::map<size_t, TilingModelInfo>;
using TensorIdSet = std::map<size_t, std::map<size_t, std::set<int64_t>>>;
// score_funcs: {level, {asc_graph_id, {impl_graph_id, score_func}}
using ScoreFuncs = std::map<kModelInfoLevel, std::map<size_t, std::map<size_t, std::string>>>;
using EnableGroupParallels = std::map<size_t, std::map<size_t, bool>>;
using VarRelations =
    std::map<size_t, std::map<size_t, std::map<size_t, std::map<size_t, std::map<std::string, af::Expression>>>>>;
// schedule result id->{score_func, group_model_infos}
struct ParsedScheduleResult {
  size_t asc_graph_id{0UL};
  size_t impl_graph_id{0UL};
  std::string score_func;
  std::map<size_t, std::map<size_t, std::map<std::string, af::Expression>>> var_relations;
  GroupsTilingModelInfo groups_tiling_model_info;
  bool enable_group_parallel{false};
};
using FusedParsedScheduleResult = std::map<size_t, std::map<size_t, ParsedScheduleResult>>;
}  // namespace att
#endif
