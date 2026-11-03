/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_TUNING_SPACE_H_
#define PARSER_TUNING_SPACE_H_

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <sstream>
#include "graph/node.h"
#include "base/model_info.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "ascendc_ir.h"
#include "ascir_node_param/ascir_node_param.h"

namespace att {
class TilingScheduleConfigTable;
const std::unordered_map<AxisPosition, std::string> AxisType2Str = {
    {AxisPosition::OUTER, "OUTER"},   {AxisPosition::INNER, "INNER"},    {AxisPosition::ORIGIN, "ORIGIN"},
    {AxisPosition::MERGED, "MERGED"}, {AxisPosition::POSERR, "INVALID"},
};

const std::unordered_map<HardwareDef, std::string> HardwareType2Str = {
    {HardwareDef::GM, "GM"},           {HardwareDef::L1, "L1"},
    {HardwareDef::L2, "L2"},           {HardwareDef::L0A, "L0A"},
    {HardwareDef::L0B, "L0B"},         {HardwareDef::L0C, "L0C"},
    {HardwareDef::UB, "UB"},           {HardwareDef::BTBUF, "BTBUF"},
    {HardwareDef::CORENUM, "CORENUM"}, {HardwareDef::HARDWAREERR, "INVALID"},
};

const std::unordered_map<PipeType, std::string> PipeType2Str = {
    {PipeType::AIC_MTE1, "AIC_MTE1"},       {PipeType::AIC_MTE2, "AIC_MTE2"},
    {PipeType::AIC_FIXPIPE, "AIC_FIXPIPE"}, {PipeType::AIC_MAC, "AIC_MAC"},
    {PipeType::AIV_MTE2, "AIV_MTE2"},       {PipeType::AIV_MTE3, "AIV_MTE3"},
    {PipeType::AIV_VEC, "AIV_VEC"},         {PipeType::AICORE_MTE1, "AICORE_MTE1"},
    {PipeType::AICORE_MTE2, "AICORE_MTE2"}, {PipeType::AICORE_MTE3, "AICORE_MTE3"},
    {PipeType::AICORE_CUBE, "AICORE_CUBE"}, {PipeType::AICORE_VEC, "AICORE_VEC"},
    {PipeType::PIPE_NONE, "INVALID"},
};

struct SubAxis;
using SubAxisPtr = std::unique_ptr<SubAxis>;

struct SubAxis {
  std::string ToString() const {
    std::stringstream ss;
    ss << "name: " << name << ", is_bind_multi_core: " << is_bind_multi_core << ", is_split: " << is_split
       << ", is_last: " << is_last << ", enable_pad: " << enable_pad
       << ", is_node_innerest_dim: " << is_node_innerest_dim << ", align: " << align
       << ", axis_type: " << AxisType2Str.at(axis_type);
    ss << ", repeat: " << ((repeat.IsValid()) ? Str(repeat) : "");
    ss << ", orig_axis_name: ";
    for (const auto &n : orig_axis_name) {
      ss << n << ",";
    }
    ss << " parent_axis_name: ";
    for (auto &axis : parent_axis) {
      ss << axis->name << ",";
    }
    return ss.str();
  }

  // 轴基本信息
  std::string name;
  int64_t id{-1L};                    // ASCIR原始axis id，用于根据ScheduleAttr::loop_axis_id回查轴
  AxisPosition axis_type{};           // 轴类型
  bool is_bind_multi_core = false;    // 该轴是否block切分，是否要技术block inner
  bool enable_tail = false;           // 是否能做不对齐tail
  bool is_split = false;              // outer轴允许split，目前未被使用
  bool enable_pad = false;            // 是否允许对轴做pad
  bool is_last = false;               // 原始轴的最内切分轴，用于设定轴的默认初始值
  bool is_node_innerest_dim = false;  // 是否是某个node的最内轴，用于决定轴的搜索优先级
  bool is_concat_vec_axis = false;    // 是否是concat node的vectorized轴
  uint32_t data_type_size = 4;        // 轴的数据类型大小，默认为4(fp32)
  Expr align = af::Symbol(1);         // 轴对齐要求
  Expr repeat;                        // 轴大小
  std::pair<int64_t, int64_t> value_range = {-1, -1};

  // 轴关联信息
  std::vector<std::string> orig_axis_name;  // 原始轴信息
  // 上面两个信息是否和下面的有重复定义
  std::vector<SubAxis *> orig_axis;
  std::vector<SubAxis *> parent_axis;

  std::string basic;

  // 分核轴类型标记（用于Store冲突检测）
  bool is_reduce_split_axis{false};     // 该轴是否是Reduce分核轴
  bool is_broadcast_split_axis{false};  // 该轴是否是Broadcast分核轴
};

struct Tensor {
  std::string ToString() {
    std::stringstream ss;
    ss << "name: " << name << ", datasize: " << data_type_size << ", resource_id: " << resource_id;
    ss << ", axis {";
    for (auto &axis : dim_info) {
      ss << axis->name << ", ";
    }
    ss << "} ";
    return ss.str() + GetRepeat() + GetStride();
  }

  std::string GetStride() const {
    std::stringstream ss;
    ss << ", stride: {";
    for (auto &tensor_size : stride) {
      std::string size = (tensor_size.IsValid()) ? Str(tensor_size) : "";
      ss << size << ", ";
    }
    ss << "}";
    ss << ", ori_stride: {";
    for (auto &tensor_size : ori_stride) {
      std::string size = (tensor_size.IsValid()) ? Str(tensor_size) : "";
      ss << size << ", ";
    }
    ss << "}";
    ss << ", gm_stride: {";
    for (auto &tensor_size : gm_stride) {
      std::string size = (tensor_size.IsValid()) ? Str(tensor_size) : "";
      ss << size << ", ";
    }
    ss << "}";
    return ss.str();
  }

  std::string GetRepeat() const {
    std::stringstream ss;
    ss << ", repeat: {";
    for (auto &tensor_size : repeat) {
      std::string size = (tensor_size.IsValid()) ? Str(tensor_size) : "";
      ss << size << ", ";
    }
    ss << "}";
    ss << ", ori_repeat: {";
    for (auto &tensor_size : ori_repeat) {
      std::string size = (tensor_size.IsValid()) ? Str(tensor_size) : "";
      ss << size << ", ";
    }
    ss << "}";
    return ss.str();
  }

  std::string name;                     // tensor name = node_name + "_output_" + out_id;
  uint32_t data_type_size;              // 数据类型的大小
  int32_t resource_id = -1;             // 来源于哪一个container,对应container的containerId
  af::AscNode *owner_node{nullptr};     // 所属的node_name
  std::string node_type;                // 就是对应node的type
  std::string data_type;                // 数据类型
  std::vector<SubAxis *> dim_info;      // 内存大小
  std::vector<Expr> repeat;             // tensor向量轴的repeat
  std::vector<Expr> stride;             // tensor向量轴的stride
  std::vector<Expr> gm_stride;          // global tensor的stride
  std::vector<SubAxis *> ori_dim_info;  // tensor 非向量轴且是block inner的轴
  std::vector<Expr> ori_repeat;         // ori_dim_info repeat
  std::vector<Expr> ori_stride;         // ori_dim_info stride, stride可能是0
  std::vector<int32_t> orig_idx;        // 向量轴对应原始轴index（位置信息）
  HardwareDef loc = HardwareDef::GM;    // tensor 物理位置
};
using TensorPtr = std::shared_ptr<Tensor>;

struct NodeInfo {
  std::string name;  // ascendc api name, 也是图node name
  std::string
      node_type;  // Data、Store、Workspace、api,根据此type可以从已注册的api接口中获取内部buffer以及api的性能公式
  std::string node_unit;             // node unit信息，用来构造缺省值
  std::string trans_config;          // 预留
  std::vector<TensorPtr> inputs;     // node 输入tensor
  std::vector<TensorPtr> outputs;    // node 输出tensor
  std::vector<SubAxis *> loop_axes;  // node loop size
  uint32_t depth = 1U;               // node 输出tensor mem队列深度最大值
  af::AscNodePtr node_ptr;
  std::set<std::string> from_data;  // 隶属的Data节点名称
  std::vector<NodeInfo> sub_nodes_infos;
  ascir_param::VectorFuncNodeParams vector_func_params;
  ascir_param::ReduceNodeParams reduce_specific_params;
  ascir_param::CastNodeParams cast_node_params;
  ascir_param::BroadcastNodeParams broadcast_node_params;
  ascir_param::CompareNodeParams compare_node_params;
  ascir_param::WhereNodeParams where_node_params;
  ascir_param::UnaryBitWidthChangeNodeParams unary_bitwidth_change_node_params;
  ascir_param::TransposeNodeParams transpose_node_params;
  af::ExecuteCondition exec_condition{af::ExecuteCondition::kNoCache};
  // schedule 级 Codegen 路径门禁。kUBFuse 将 NDDMA 描述固定为 2D，因此 raw 1D 必须使用 legacy 模型。
  bool is_cv_ub_fusion{false};
  std::vector<ascir_param::CacheExecutionParams> cache_executions;
  std::string DebugString() const {
    std::stringstream ss;
    ss << "NodeInfo {" << name << ", " << node_type << ", " << node_unit;
    ss << ", input size=" << inputs.size() << ", output size=" << outputs.size();
    ss << ", loop_axes size=" << loop_axes.size() << ", from_data=";
    for (const auto &data : from_data) {
      ss << data << ", ";
    }
    ss << "sub_nodes_infos size=" << sub_nodes_infos.size();
    ss << ", exec_condition = " << static_cast<int32_t>(exec_condition);
    ss << " }";
    return ss.str();
  }
};

struct Container {
  explicit Container(const std::string &name) : name(name) {}
  std::vector<std::vector<TensorPtr>> GetCoTensors() const {
    return coexist_tensors;
  }
  virtual int64_t GetBufferNum() const = 0;
  std::string name;
  int32_t reuse_id{0};
  Expr align;
  std::vector<TensorPtr> allocated_tensors;  // queue或者buf分配了哪些tensor
  std::vector<HardwareDef> buf_location;     // queue或者buf涉及哪些硬件
  std::vector<std::vector<TensorPtr>>
      coexist_tensors;  // coexist_tensors表示tensor共存且位于同一scope,比如tbuf做pingpong、tqueue中两个tensor同时存在
  af::AllocType alloc_type{af::AllocType::kAllocTypeInvalid};
  int64_t container_id{0};
};
using ContainerPtr = std::shared_ptr<Container>;

struct Queue : public Container {
  explicit Queue(const std::string &name) : Container(name) {}
  int64_t GetBufferNum() const override {
    return buffer_num;
  }
  int64_t buffer_num = 1L;
};

struct Buf : public Container {
  explicit Buf(const std::string &name) : Container(name) {}
  int64_t GetBufferNum() const override {
    return buffer_num;
  }
  int64_t buffer_num = 1L;
};

struct GlobalCache : public Container {
  explicit GlobalCache(const std::string &name) : Container(name) {}
  int64_t GetBufferNum() const override {
    return buffer_num;
  }
  int64_t buffer_num = 1L;
};

struct TuningSpace {
  std::vector<ContainerPtr> containers;         // queue和buf所有信息
  std::vector<ContainerPtr> global_containers;  // 所有gm上的信息
  std::vector<SubAxisPtr> sub_axes;             // 所有轴信息
  std::vector<NodeInfo> node_infos;             // 所有ascir api信息
  std::map<int64_t, Expr> workspace_size_map;
  std::vector<std::vector<SubAxis *>> block_dims;                   // block outer轴大小
  std::map<const SubAxis *, std::set<HardwareDef>> related_scopes;  // 向量轴涉及多少mem 类型
  std::map<int64_t, Expr> tmp_buffer;                               // 临时空间
  std::map<std::string, uint32_t> reserve_ub;                       // 预留空间
  Expr builtin_tmp_buffer;  // kernel内部申请的tmp buffer（这部分会在计算UB Size的时候使用，但不需要申请TilingData）
  vector<CacheLineConfig> *cache_line_config{nullptr};
  const TilingScheduleConfigTable *tiling_schedule_config_table{nullptr};
  const af::AscGraph *asc_graph{nullptr};
  uint32_t penalty_cache_line_size{0};  // Reduce核数惩罚粒度，0表示使用配置的Cache Line
};
using TuningSpacePtr = std::shared_ptr<TuningSpace>;
}  // namespace att

#endif  // PARSER_TUNING_SPACE_H_
