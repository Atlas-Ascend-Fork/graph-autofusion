/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_READABLE_DUMP_H_
#define INC_GRAPH_UTILS_READABLE_DUMP_H_

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <google/protobuf/text_format.h>

#include "graph_metadef/graph/debug/ge_util.h"
#include "common/checker.h"
#include "graph/node.h"

namespace af {
static constexpr const char *const kIndentZero = "";   // 0个空格
static constexpr const char *const kIndentTwo = "  ";  // 2个空格
static constexpr const char *const kNetOutput = "NetOutput";

class ReadableDump {
 private:
  class OutputHandler;

 public:
  ReadableDump() = delete;
  ReadableDump(const ReadableDump &) = delete;
  ReadableDump &operator=(const ReadableDump &) = delete;
  ~ReadableDump() = delete;

  /**
   * @brief 生成Readable Dump主函数
   * @param readable_ss 字符串流
   * @param graph Dump图
   */
  static Status GenReadableDump(std::stringstream &readable_ss, const ComputeGraphPtr &graph);

 private:
  struct DumpContext {
    std::set<std::string> visited_subgraph_instances;
    ComputeGraphPtr root_graph;
  };

  /**
   * @brief 生成可读的图结构
   * @param readable_ss 字符串流
   * @param graph 计算图
   * @param ctx dump上下文
   */
  static Status GenReadableDump(std::stringstream &readable_ss, const ComputeGraphPtr &graph, DumpContext &ctx);

  class OutputHandler {
   public:
    OutputHandler() = default;
    ~OutputHandler() = default;

    std::unordered_map<std::string, std::shared_ptr<std::vector<std::string>>> &GetNodeToOutputsMap() {
      return node_to_outputs_;
    }
    std::string GetOutputRet() {
      if (index_ == 0) {
        index_++;
        return "ret";
      }
      return "ret_" + std::to_string(index_++);
    }
    void GenNodeToOutputsMap(const af::ComputeGraphPtr &graph) {
      for (const auto &node : graph->GetDirectNode()) {
        const auto op_desc = node->GetOpDesc();
        if (op_desc == nullptr) {
          continue;
        }
        if (op_desc->GetType() != kNetOutput) {
          std::shared_ptr<std::vector<std::string>> output_rets = ComGraphMakeShared<std::vector<std::string>>();
          if (output_rets == nullptr) {
            REPORT_INNER_ERR_MSG("E18888", "Failed to initialize output vector");
            GELOGE(GRAPH_FAILED, "[OutputHandler][GenNodeToOutputsMap] failed to initialize output vector");
            return;
          }
          if (node->GetAllOutDataAnchorsPtr().size() <= 1) {
            output_rets->emplace_back(node->GetName());
          } else {
            for (const auto output_idx : node->GetAllOutDataAnchorsPtr()) {
              if (output_idx != nullptr) {
                output_rets->emplace_back(GetOutputRet());
              }
            }
          }

          node_to_outputs_.emplace(node->GetName(), output_rets);
        }
      }
    }

   private:
    int32_t index_ = 0;
    std::unordered_map<std::string, std::shared_ptr<std::vector<std::string>>> node_to_outputs_{};
  };

  /**
   * @brief 生成节点Readable Dump
   * @param readable_ss 字符串流
   * @param output_handler 节点输出处理器
   * @param node Dump节点
   * @param subgraphs_to_dump 收集到的子图列表
   * @param ctx dump上下文
   */
  static void GenNodeDump(std::stringstream &readable_ss, OutputHandler &output_handler, const Node *node,
                          std::vector<ComputeGraphPtr> &subgraphs_to_dump, DumpContext &ctx);

  /**
   * @brief 获取实例名称
   * @param name 实例名
   * @param indent 前空行
   * @return Readable Dump节点名或输出名
   */
  static std::string GetInstanceName(const std::string &name, const std::string &indent = kIndentTwo);

  /**
   * @brief 获取节点出度
   * @param node 节点
   * @return 节点出度字符串
   */
  static std::string GetNodeOutDegree(const Node *node);

  /**
   * @brief 获取节点IR类型
   * @param node 节点
   * @return 节点IR类型字符串
   */
  static std::string GetNodeType(const Node *node);

  /**
   * @brief 获取入参实例名称
   * @param node 节点
   * @param input_index 输入锚点索引
   * @param output_handler 输出处理器
   * @return 入参实例名称，如果获取失败返回空字符串
   */
  static std::string GetInputInstanceName(const Node *node, size_t input_index, OutputHandler &output_handler);

  /**
   * @brief 追加入参实例到字符串流
   * @param ss 字符串流
   * @param first 是否为第一个入参
   * @param param_name 参数IR名称
   * @param instance_name 实例名称
   */
  static void AppendInputInstance(std::stringstream &ss, bool &first, const std::string &param_name,
                                  const std::string &instance_name);

  /**
   * @brief 获取节点入参，带 IR 参数名称
   *
   * @param node 节点
   * @param output_handler 输出处理器
   * @return 带参数名称的入参实例字符串，格式：param1=%instance1, param2=%instance2, ...
   */
  static std::string GetNodeInputInstanceWithIr(const Node *node, OutputHandler &output_handler);

  /**
   * @brief 获取节点入参，带参数名称
   *
   * @param node 节点
   * @param output_handler 输出处理器
   * @return 带参数名称的入参实例字符串，格式：param1=%instance1, param2=%instance2, ...
   */
  static std::string GetNodeInputInstance(const Node *node, OutputHandler &output_handler);

  /**
   * @brief 生成节点的可读入参信息
   * @param readable_ss 字符串流
   * @param node 节点
   * @param output_handler 节点输出处理器
   */
  static void GenNodeInputs(std::stringstream &readable_ss, const Node *node, OutputHandler &output_handler);

  static std::string GetAttrValueStr(const OpDescPtr &op_desc, const std::string &attr_name, const AnyValue &attr_value,
                                     const std::string &av_type);

  /**
   * @brief 获取 IR 子图索引到 desc 索引范围的映射
   * @param node 节点
   * @return IR 子图索引到 desc 索引范围的映射
   *         key: IR 子图索引
   *         value: (start_index, count) - desc 中的起始索引和数量
   */
  static std::map<size_t, std::pair<size_t, size_t>> GetIrGraphDescRange(const Node *node);

  /**
   * @brief 收集子图到 subgraphs_to_dump
   * @param subgraphs_to_dump 收集到的子图列表
   * @param instance_name 子图实例名称
   * @param ctx dump上下文
   */
  static void CollectSubgraphIfNeeded(std::vector<ComputeGraphPtr> &subgraphs_to_dump, const std::string &instance_name,
                                      DumpContext &ctx);

  /**
   * @brief 追加子图属性到字符串流
   * @param ss 字符串流
   * @param first 是否为第一个属性
   * @param param_name 参数IR名称
   * @param instance_name 子图实例名称
   */
  static void AppendSubgraphAttr(std::stringstream &ss, bool &first, const std::string &param_name,
                                 const std::string &instance_name);

  /**
   * @brief 获取节点的子图属性信息，带子图 IR 名称
   * @param node 节点
   * @param subgraphs_to_dump 收集到的子图列表
   * @param ctx dump上下文
   * @return 子图属性字符串，格式：ir_name1: %instance1, ir_name2: %instance2, ...
   *         如果 IR 定义为空，返回空字符串
   */
  static std::string GetSubgraphAttrsWithIr(const Node *node, std::vector<ComputeGraphPtr> &subgraphs_to_dump,
                                            DumpContext &ctx);

  /**
   * @brief 获取节点的子图属性信息，同时收集子图用于后续展开
   * @param node 节点
   * @param subgraphs_to_dump 收集到的子图列表
   * @param ctx dump上下文
   * @return 子图属性字符串
   *         优先使用 IR 定义中的子图名称，如果失败则使用（_graph_0, _graph_1, ...）
   */
  static std::string GetSubgraphAttrs(const Node *node, std::vector<ComputeGraphPtr> &subgraphs_to_dump,
                                      DumpContext &ctx);

  /**
   * @brief 添加节点属性信息
   * @param attr_contents 字符串流
   * @param node 节点
   */
  static void AppendNodeAttrs(std::stringstream &attr_contents, const Node *node);

  /**
   * @brief 生成节点属性信息，同时收集子图用于后续展开
   * @param readable_ss 字符串流
   * @param node 节点
   * @param subgraphs_to_dump 收集到的子图列表
   * @param ctx dump上下文
   */
  static void GenNodeAttrs(std::stringstream &readable_ss, const Node *node,
                           std::vector<ComputeGraphPtr> &subgraphs_to_dump, DumpContext &ctx);

  /**
   * @brief 获取节点输出实例的出度信息
   * @param node 节点
   * @param index 节点输出下标
   * @return 节点输出实例的出度字符串
   */
  static std::string GetOutputOutDegree(const Node *node, int32_t index);

  /**
   * @brief 当节点包含多数出时，生成多数出信息，输出实例名称按照'ret', 'ret_1', 'ret_2'递增
   * @param readable_ss 字符串流
   * @param node Dump节点
   * @param output_handler 输出处理器
   */
  static void GenMultipleOutputsIfNeeded(std::stringstream &readable_ss, const Node *node,
                                         OutputHandler &output_handler);

  /**
   * @brief 获取图输出实例字符串
   * @param net_output NetOutput节点
   * @param output_handler 节点输出处理器
   * @return 图输出实例字符串
   */
  static std::string GetGraphOutputInstance(const Node *net_output, OutputHandler &output_handler);

  /**
   * @brief 获取图的输出实例信息
   * @param graph_output_ss 字符串流
   * @param net_output 图的NetOutput节点
   * @param output_handler 节点输出处理器
   */
  static void GenGraphOutput(std::stringstream &graph_output_ss, const Node *net_output, OutputHandler &output_handler);
};
}  // namespace af

#endif  // INC_GRAPH_UTILS_READABLE_DUMP_H_
