/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ARG_LIST_REORDER_H_
#define ARG_LIST_REORDER_H_

#include <vector>
#include <map>
#include <set>
#include <queue>
#include <list>
#include "parser/tuning_space.h"
#include "base/model_info.h"
#include "common/util/mem_utils.h"
#include "util/att_utils.h"

namespace att {
// 表征轴优先级的图结构， a->b表示a的优先级高于b
class ArgPriorityGraph {
 public:
  // 构造函数初始化
  explicit ArgPriorityGraph() = default;
  explicit ArgPriorityGraph(size_t n)
      : vertex_count_(n),
        adj_list_(n + 1),  // 顶点编号从1开始
        in_degree_(n + 1, 0) {}

  // 检查是否会创建环（模拟添加边后检查）
  bool WillCreateCycle(size_t from, size_t to) {
    std::vector<bool> visited(vertex_count_ + 1, false);
    std::queue<size_t> bfs_queue;

    // 模拟添加边后的状态，从to出发检测是否能回到from
    bfs_queue.push(to);
    while (!bfs_queue.empty()) {
      size_t current = bfs_queue.front();
      bfs_queue.pop();

      for (size_t neighbor : adj_list_[current]) {
        if (neighbor == from) {
          return true;  // 发现环
        }
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          bfs_queue.push(neighbor);
        }
      }
    }
    return false;
  }

  // 添加有向边
  bool AddEdge(size_t from, size_t to) {
    // 参数检查
    if (from == 0u || from > vertex_count_ || to == 0u || to > vertex_count_) {
      GELOGE(af::FAILED, "node id invalid");
      return false;
    }

    // 检查边是否已存在
    auto &edges = adj_list_[from];
    if (std::find(edges.begin(), edges.end(), to) != edges.end()) {
      return true;  // 边已存在，直接返回
    }

    // 成环检测
    if (WillCreateCycle(from, to)) {
      GELOGI("Create Cycle from %zu to %zu, refuse to add edge", from, to);
      return true;  // 添加会导致环，拒绝操作
    }

    // 执行添加
    edges.push_back(to);
    in_degree_[to]++;
    return true;
  }

  // 拓扑排序主入口
  std::vector<size_t> TopologicalSort(bool use_bfs = true) {
    return use_bfs ? KahnAlgorithm() : DfsPostOrder();
  }

 private:
  // BFS拓扑排序（时间复杂度O(V+E)）
  std::vector<size_t> KahnAlgorithm() {
    std::vector<size_t> result;
    std::queue<size_t> zero_in_degree_queue_;
    auto temp_in_degree_ = in_degree_;  // 临时副本避免修改原数据

    // 初始化入度为0的队列
    for (size_t i = 1; i <= vertex_count_; ++i) {
      if (temp_in_degree_[i] == 0) {
        zero_in_degree_queue_.push(i);
      }
    }

    while (!zero_in_degree_queue_.empty()) {
      size_t current = zero_in_degree_queue_.front();
      zero_in_degree_queue_.pop();
      result.push_back(current);

      // 处理邻接顶点
      for (size_t neighbor : adj_list_[current]) {
        if (--temp_in_degree_[neighbor] == 0) {
          zero_in_degree_queue_.push(neighbor);
        }
      }
    }

    // 环检测
    if (result.size() != vertex_count_) {
      GELOGE(af::FAILED, "has cycle, failed to toposort");
      return {};
    }
    return result;
  }

  // DFS拓扑排序（时间复杂度O(V+E)）
  std::vector<size_t> DfsPostOrder() {
    std::vector<size_t> result;
    std::vector<bool> visited(vertex_count_ + 1, false);
    std::vector<bool> recursion_stack_(vertex_count_ + 1, false);

    for (size_t i = 1; i <= vertex_count_; ++i) {
      if (!visited[i] && HasCycleDfs(i, visited, recursion_stack_, result)) {
        GELOGE(af::FAILED, "has cycle, failed to toposort");
        return {};
      }
    }
    std::reverse(result.begin(), result.end());
    return result;
  }

  // DFS递归辅助函数
  bool HasCycleDfs(size_t node, std::vector<bool> &visited, std::vector<bool> &recursion_stack_,
                   std::vector<size_t> &result) {
    if (recursion_stack_[node]) {
      return true;  // 发现环
    }
    if (visited[node]) {
      return false;
    }
    visited[node] = true;
    recursion_stack_[node] = true;

    for (size_t neighbor : adj_list_[node]) {
      if (HasCycleDfs(neighbor, visited, recursion_stack_, result)) {
        return true;
      }
    }
    recursion_stack_[node] = false;
    result.push_back(node);  // 后序位置记录节点
    return false;
  }

  size_t vertex_count_;                      // 顶点总数
  std::vector<std::list<size_t>> adj_list_;  // 邻接表
  std::vector<size_t> in_degree_;            // 顶点入度统计
};

using ArgPriorityGraphPtr = std::shared_ptr<ArgPriorityGraph>;

class ArgListReorder {
 public:
  explicit ArgListReorder(const TuningSpacePtr &tuning_space) : tuning_space_(tuning_space) {}
  ~ArgListReorder() = default;
  // 排序ArgList入口函数
  af::Status SortArgList(vector<AttAxisPtr> &arg_list, vector<AttAxisPtr> &tiling_R_arg_list,
                         std::vector<RuntimeReorderRule> *runtime_reorder_rules = nullptr);

 private:
  // 轴属性枚举
  enum class AxisProperty : uint8_t { kReduce, kBroadcast };
  enum class ReduceTailTilePolicy : uint8_t { kFallback, kEqual, kPreferTail, kKeepDefault };
  enum class PriorityPolicy : uint8_t { kReduceFirst, kInnermostFirst, kNonReduceFirst };

  struct ReduceTailTileInfo {
    std::string reduce_axis_name;
    std::string tail_axis_name;
    Expr reduce_repeat;
    Expr tail_repeat;
    uint32_t data_type_size{0U};
  };

  // 轴分类结果结构体
  struct AxisCategories {
    std::vector<std::string> reduce_arg_names;
    std::vector<std::string> non_reduce_arg_names;
    std::vector<std::string> broadcast_arg_names;
    std::vector<std::string> non_broadcast_arg_names;
    std::vector<std::string> innermost_dim_arg_names;
    std::vector<std::string> non_innermost_dim_arg_names;
  };

  // 核心函数保持原名
  af::Status BuildArgListPriorityGraph(const vector<AttAxisPtr> &arg_list, PriorityPolicy policy);
  af::Status InitArgListPriorityGraph(const vector<AttAxisPtr> &arg_list);
  af::Status BuildTilingRArgList(const vector<AttAxisPtr> &arg_list, const std::vector<size_t> &canonical_order,
                                 vector<AttAxisPtr> &tiling_R_arg_list);

  // 变量名改为下划线
  std::map<std::string, size_t> axis_name_2_id_map_;
  std::map<std::string, bool> reduce_map_;
  std::map<std::string, bool> broadcast_map_;
  std::map<std::string, bool> innermost_dim_map_;
  std::set<std::string> load_store_inner_most_dims_;  // 搬运类节点Tile切分轴的order值
  TuningSpacePtr tuning_space_;
  ArgPriorityGraphPtr graph_;
  bool tiling_R_ = false;
  bool prefer_reduce_tile_ = false;
  bool has_dynamic_reduce_tile_reorder_ = false;
  RuntimeReorderRule dynamic_reduce_tile_reorder_rule_;
  std::set<std::string> equal_order_reduce_tail_axes_;

  // 成员函数保持原名
  bool CheckReduce(const SubAxis *dim, const Expr &repeat, const Expr &stride,
                   const std::vector<TensorPtr> &output_tensors);
  bool CheckBroadcast(const SubAxis *dim, const Expr &repeat, const Expr &stride,
                      const std::vector<TensorPtr> &output_tensors);
  bool CheckAxisProperty(const SubAxis *dim, const Expr &repeat, const Expr &stride,
                         const std::vector<TensorPtr> &output_tensors, AxisProperty property);
  void FindSpecialArgs();
  af::Status AddEdgeGroups(const std::vector<std::string> &from_axes_group,
                           const std::vector<std::string> &to_axes_group);
  bool IsReduceAxisBlockSplit(const std::vector<SubAxisPtr> &all_axes,
                              const std::set<std::string> &reduce_axis_ori_axes_set) const;
  bool IsReduceAxisTileSplit(const std::set<std::string> &reduce_axis_ori_axes_set) const;
  uint32_t GetCacheLineSize() const;
  uint32_t GetVectorLenSize() const;
  bool GetReduceAxisDataTypeSize(const NodeInfo &node, const SubAxis *axis,
                                 const std::set<std::string> &reduce_axis_ori_axes_set, uint32_t &data_type_size) const;
  const SubAxis *FindReduceTileAxis(const std::set<std::string> &reduce_axis_ori_axes_set) const;
  bool IsReduceInputTensor(const TensorPtr &tensor, const SubAxis *reduce_axis,
                           const std::set<std::string> &reduce_axis_ori_axes_set) const;
  bool CollectReduceTailAxis(const TensorPtr &tensor, const SubAxis *reduce_axis,
                             const std::set<std::string> &reduce_axis_ori_axes_set, const SubAxis *&tail_axis,
                             uint32_t &data_type_size) const;
  bool IsReduceTailTileAxis(const SubAxis *tail_axis) const;
  bool TryGetReduceTailTileInfo(const NodeInfo &node, const std::set<std::string> &reduce_axis_ori_axes_set,
                                ReduceTailTileInfo &tile_info) const;
  ReduceTailTilePolicy ClassifyReduceTailTilePolicy(const ReduceTailTileInfo &tile_info) const;
  bool TryBuildReduceTileRuntimeReorderRule(const NodeInfo &node, const std::set<std::string> &reduce_axis_ori_axes_set,
                                            RuntimeReorderRule &rule) const;
  bool HasSmallTailLargeReduceTile(const NodeInfo &node, const std::set<std::string> &reduce_axis_ori_axes_set,
                                   RuntimeReorderRule &rule) const;
  void RecordReduceTileTemplateSelection(const NodeInfo &node, const std::set<std::string> &reduce_axis_ori_axes_set);
  void SaveReduceAxisOrig(const SubAxis *reduce_axis, std::set<std::string> &reduce_axis_ori_axes_set) const;
  AxisCategories CategorizeAxesByProperty(const vector<AttAxisPtr> &arg_list);
  af::Status ApplyPriorityRules(PriorityPolicy policy, const AxisCategories &categories);
  std::vector<AttAxisPtr> GetNewArgList(const std::vector<size_t> &topo_order,
                                        const std::vector<AttAxisPtr> &arg_list) const;
  std::vector<Expr> GetTileSplitOrder(const std::vector<size_t> &topo_order,
                                      const std::vector<AttAxisPtr> &arg_list) const;
  bool SetRuntimePreferredOrder(const std::vector<AttAxisPtr> &canonical_arg_list,
                                const std::vector<AttAxisPtr> &source_arg_list,
                                const std::vector<size_t> &preferred_topo_order, RuntimeReorderRule &rule) const;
  void SetAxesSameOrder(const std::vector<AttAxisPtr> &arg_list, const std::set<std::string> &axis_names) const;
  void MakeSureLoadStoreInnerestSameOrder(const std::vector<AttAxisPtr> &arg_list) const;
  bool HandleProperty(const SubAxis *dim, att::ArgListReorder::AxisProperty property, bool is_reduce,
                      bool is_broadcast);
  void RecordSpecialArgs(const NodeInfo &node, const TensorPtr &tensor, size_t id,
                         const std::vector<TensorPtr> &output_tensors, std::set<std::string> &reduce_axis_ori_axes_set);
};

}  // namespace att
#endif
