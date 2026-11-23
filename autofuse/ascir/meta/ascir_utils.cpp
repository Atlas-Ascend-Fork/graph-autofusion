/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascir_utils.h"

#include <sstream>
#include <fstream>
#include <iomanip>
#include <queue>
#include <set>
#include <algorithm>
#include <optional>
#include <cstdint>
#include <unistd.h>
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "graph/utils/type_utils.h"
#include "graph_utils.h"
#include "mmpa/mmpa_api.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "ascend_graph_code_dumper.h"
#include "asc_graph_dumper_context.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph_metadef/graph/utils/readable_dump.h"
#include "graph_metadef/graph/model.h"
#include "graph/utils/ge_ir_utils.h"
#include "common_utils.h"
#include "ascir_ops_utils.h"
#include "ascir_ops.h"
#include "ascendc_graph_txt_dumper.h"

namespace {
constexpr int32_t DUMP_ID_WIDTH = 5;
constexpr size_t MAX_DUMP_FILE_NAME_LENGTH = 255U;
constexpr size_t DUMP_FILE_NAME_HASH_LENGTH = 16U;
constexpr uint64_t FNV_OFFSET_BASIS = 1469598103934665603ULL;
constexpr uint64_t FNV_PRIME = 1099511628211ULL;
std::string SanitizeFileName(const std::string &name) {
  std::string result;
  result.reserve(name.length());
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.' || c == ' ') {
      result += c;
    } else {
      result += '_';
    }
  }
  return result;
}

std::string FormatDumpIndex(uint64_t index) {
  std::ostringstream ss;
  ss << std::setw(DUMP_ID_WIDTH) << std::setfill('0') << index;
  return ss.str();
}

uint64_t GetStableDumpFileHash(const std::string &name) {
  uint64_t hash = FNV_OFFSET_BASIS;
  for (const char c : name) {
    hash = (hash ^ static_cast<unsigned char>(c)) * FNV_PRIME;
  }
  return hash;
}

std::string LimitDumpFileName(const std::string &name) {
  if (name.size() <= MAX_DUMP_FILE_NAME_LENGTH) {
    return name;
  }

  const size_t hash_text_length = MAX_DUMP_FILE_NAME_LENGTH - DUMP_FILE_NAME_HASH_LENGTH - 2U;
  const size_t head_length = hash_text_length / 2U;
  const size_t tail_length = hash_text_length - head_length;
  const uint64_t hash = GetStableDumpFileHash(name);
  std::ostringstream ss;
  ss << name.substr(0U, head_length) << "_" << std::hex << std::setw(DUMP_FILE_NAME_HASH_LENGTH) << std::setfill('0')
     << hash << "_" << name.substr(name.size() - tail_length);
  return ss.str();
}

bool GetConcatDim(const af::AscNode &node, size_t &concat_dim) {
  auto node_inputs = node.inputs;
  auto node_outputs = node.outputs;
  const auto &input_repeats = node_inputs[0].attr.repeats;
  const auto &output_repeats = node_outputs[0].attr.repeats;
  GE_CHK_BOOL_RET_SPECIAL_STATUS(input_repeats.size() != output_repeats.size(), false,
                                 "[%s] input_repeats.size() = %zu, mismatches output_repeats.size() = %zu",
                                 node.GetNamePtr(), input_repeats.size(), output_repeats.size());
  for (size_t i = 0U; i < input_repeats.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != af::TriBool::kTrue) {
      concat_dim = i;
      return true;
    }
  }
  return false;
}

bool IsStoreWithoutStride(const af::AscNode &node) {
  std::set<af::Node *> visited_nodes;
  std::queue<af::NodePtr> next_nodes;
  std::vector<af::AscNodePtr> store_nodes;
  for (const auto &out_data_node : node.GetOutDataNodes()) {
    if (visited_nodes.emplace(out_data_node.get()).second) {
      next_nodes.push(out_data_node);
    }
  }
  while (!next_nodes.empty()) {
    auto &top = next_nodes.front();
    auto asc_node = std::dynamic_pointer_cast<af::AscNode>(top);
    GE_ASSERT_NOTNULL(asc_node);
    if (asc_node->attr.api.compute_type == af::ComputeType::kComputeStore) {
      store_nodes.emplace_back(asc_node);
    } else {
      for (const auto &out_node : top->GetOutDataNodes()) {
        if (visited_nodes.emplace(out_node.get()).second) {
          next_nodes.emplace(out_node);
        }
      }
    }
    next_nodes.pop();
  }

  for (auto &peer_node : store_nodes) {
    const auto &output_tensor_attr = peer_node->outputs[0].attr;
    GELOGI("%s output_repeat = %s, output_stride = %s", peer_node->GetNamePtr(),
           af::ToString(output_tensor_attr.repeats).c_str(), af::ToString(output_tensor_attr.strides).c_str());
    size_t concat_dim;
    GE_WARN_ASSERT(GetConcatDim(node, concat_dim));
    GE_WARN_ASSERT((concat_dim > 0) && (concat_dim < output_tensor_attr.repeats.size()),
                   "concat_dim out of range, concat_dim = %zu, valid range is (0, %zu), repeats = %s, ", concat_dim,
                   output_tensor_attr.repeats.size(), af::ToString(output_tensor_attr.repeats).c_str());
    af::Expression elt_num = output_tensor_attr.repeats[concat_dim];
    for (size_t i = concat_dim + 1UL; i < output_tensor_attr.repeats.size(); ++i) {
      elt_num = elt_num * output_tensor_attr.repeats[i];
    }
    if (af::SymbolicUtils::StaticCheckEq(elt_num, output_tensor_attr.strides[concat_dim - 1U]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

bool GetConcatDimAndColSizes(const af::AscNode &node, size_t &concat_dim, std::vector<int64_t> &src_col_sizes,
                             int64_t &dst_col_size) {
  concat_dim = std::numeric_limits<size_t>::max();
  GE_WARN_ASSERT(GetConcatDim(node, concat_dim));

  auto node_inputs = node.inputs;
  auto node_outputs = node.outputs;
  const auto &input_repeats = node_inputs[0].attr.repeats;
  const auto &output_repeats = node_outputs[0].attr.repeats;
  GE_WARN_ASSERT(concat_dim < input_repeats.size());
  int64_t concat_dim_stride = 1;
  for (size_t i = concat_dim + 1; i < input_repeats.size(); ++i) {
    const auto &dim_size_expr = input_repeats[i];
    GE_CHK_BOOL_RET_SPECIAL_STATUS((!dim_size_expr.IsConstExpr()), false,
                                   "[%s] dynamic dim after concat dim, inputs = %s, outputs = %s", node.GetNamePtr(),
                                   af::ToString(input_repeats).c_str(), af::ToString(output_repeats).c_str());
    int64_t dim_size = -1;
    (void)dim_size_expr.GetConstValue(dim_size);
    concat_dim_stride *= dim_size;
  }
  GELOGD("[%s] inputs = %s, output = %s, concat_dim = %u, concat_dim_stride = %ld", node.GetNamePtr(),
         af::ToString(input_repeats).c_str(), af::ToString(output_repeats).c_str(), concat_dim, concat_dim_stride);
  for (uint32_t i = 0U; i < node_inputs.Size(); ++i) {
    const auto &dim_size_expr = node_inputs[i].attr.repeats[concat_dim];
    GE_CHK_BOOL_RET_SPECIAL_STATUS((!dim_size_expr.IsConstExpr()), false,
                                   "[%s] input[%u] concat dim = %s, not a static dim", node.GetNamePtr(), i,
                                   dim_size_expr.Str().get());
    int64_t dim_size = -1;
    (void)dim_size_expr.GetConstValue(dim_size);
    src_col_sizes.emplace_back(dim_size * concat_dim_stride);
  }

  const auto &output_dim_size_expr = node_outputs[0].attr.repeats[concat_dim];
  GE_CHK_BOOL_RET_SPECIAL_STATUS((!output_dim_size_expr.IsConstExpr()), false,
                                 "[%s] output concat dim = %s, not a static dim", node.GetNamePtr(),
                                 output_dim_size_expr.Str().get());
  int64_t output_dim_size = -1L;
  (void)output_dim_size_expr.GetConstValue(output_dim_size);
  dst_col_size = output_dim_size * concat_dim_stride;
  GELOGD("src_col_sizes = %s, dst_col_size = %ld", af::ToString(src_col_sizes).c_str(), dst_col_size);
  return true;
}

// Dump 配置结构体
struct DumpConfig {
  bool enabled = false;
  std::string debug_dir;
};

// 去除字符串两端空白
std::string TrimParam(const std::string &s) {
  const size_t start = s.find_first_not_of(" \t");
  if (start == std::string::npos) return "";
  const size_t end = s.find_last_not_of(" \t");
  return s.substr(start, end - start + 1);
}

// 解析 debug_dir 参数值，去除引号
std::string ParseDebugDirValue(const std::string &value, bool has_space) {
  std::string dir = value;
  if (has_space) {
    // 去除可能的引号和空格
    dir.erase(0, dir.find_first_not_of(" \t\"'"));
    dir.erase(dir.find_last_not_of(" \t\"'") + 1);
  } else {
    // 去除可能的引号
    if (!dir.empty() && (dir.front() == '"' || dir.front() == '\'')) {
      dir.erase(0, 1);
    }
    if (!dir.empty() && (dir.back() == '"' || dir.back() == '\'')) {
      dir.pop_back();
    }
  }
  return dir;
}

// 尝试从参数中解析 debug_dir
bool TryParseDebugDir(const std::string &param, std::string &debug_dir) {
  const char *patterns[] = {"debug_dir=", "debug_dir ="};
  for (size_t i = 0; i < 2; ++i) {
    size_t pos = param.find(patterns[i]);
    if (pos != std::string::npos) {
      std::string dir = ParseDebugDirValue(param.substr(pos + strlen(patterns[i])), i == 1);
      if (!dir.empty()) {
        debug_dir = dir;
        GELOGI("[DumpGraph] Found debug_dir in AUTOFUSE_DFX_FLAGS: %s", dir.c_str());
        return true;
      }
    }
  }
  return false;
}

// 检查参数是否包含 codegen_compile_debug=true
bool IsCodegenCompileDebugEnabled(const std::string &param) {
  return param.find("codegen_compile_debug=true") != std::string::npos ||
         param.find("codegen_compile_debug = true") != std::string::npos;
}

// 解析 AUTOFUSE_DFX_FLAGS 环境变量
DumpConfig ParseDfxFlags(const char *dfx_flags) {
  DumpConfig cfg;
  if (dfx_flags == nullptr || strlen(dfx_flags) == 0) {
    return cfg;
  }

  std::string flags(dfx_flags);
  std::stringstream ss(flags);
  std::string param;

  while (std::getline(ss, param, ';')) {
    param = TrimParam(param);
    if (param.empty()) continue;

    if (!cfg.enabled && IsCodegenCompileDebugEnabled(param)) {
      cfg.enabled = true;
    }
    if (cfg.debug_dir.empty() && TryParseDebugDir(param, cfg.debug_dir)) {
      continue;
    }
  }
  return cfg;
}

DumpConfig GetDumpConfig() {
  return ParseDfxFlags(std::getenv("AUTOFUSE_DFX_FLAGS"));
}

std::string GetCodegenCompileDebugDir() {
  return GetDumpConfig().debug_dir;
}

// 当前 fused_graph 名称，用于按图分目录 dump
thread_local std::string g_current_fused_graph_name;

// 当前 fused_graph 的文件编号计数器（txt）
thread_local uint64_t g_current_fused_graph_dump_index = 0UL;

// 当前 fused_graph 的文件编号计数器（onnx）
thread_local uint64_t g_current_onnx_dump_index = 0UL;

// 已缓存的 pid 目录路径（不含 graph 子目录）
thread_local std::string g_cached_pid_dir;

// 已创建的 graph 子目录集合
thread_local std::set<std::string> g_created_graph_dirs;
}  // namespace

namespace ascir::utils {
static std::string DtypeToStr(af::DataType dtype) {
  switch (dtype) {
    case af::DT_FLOAT:
      return "float32";
    case af::DT_FLOAT16:
      return "float16";
    case af::DT_INT8:
      return "int8_t";
    case af::DT_INT32:
      return "int32_t";
    case af::DT_UINT8:
      return "uint8_t";
    case af::DT_INT16:
      return "int16_t";
    case af::DT_UINT16:
      return "uint16_t";
    case af::DT_UINT32:
      return "uint32_t";
    case af::DT_INT64:
      return "int64_t";
    case af::DT_UINT64:
      return "uint64_t";
    default:
      return af::TypeUtils::DataTypeToSerialString(dtype);
  }
}

// 构建 axis_id 到 axis_name 的映射（供 ascir_utils.cpp 内部使用）
static std::map<af::AxisId, std::string> GetAxisIdToName(const std::vector<af::AxisPtr> &axes) {
  std::map<af::AxisId, std::string> axis_id_to_name;
  for (auto &axis : axes) {
    axis_id_to_name[axis->id] = axis->name;
  }
  return axis_id_to_name;
}

// 构建基础路径（debug_dir 或当前目录）
static std::string BuildBasePath(const std::string &debug_dir) {
  if (!debug_dir.empty()) {
    return debug_dir + (debug_dir.back() == '/' ? "" : "/");
  }
  return "./";
}

static bool TryCreateDir(const std::string &dir_path) {
  if (mmAccess2(dir_path.c_str(), M_F_OK) == EN_OK) {
    return true;
  }
  if (af::CreateDir(dir_path) == 0) {
    return true;
  }
  GELOGW("[DumpGraph][CreateDir] Create dir failed, path:%s", dir_path.c_str());
  return false;
}

static bool InitDumpDirectories() {
  if (!g_cached_pid_dir.empty()) {
    return true;
  }

  const std::string base_path = BuildBasePath(GetCodegenCompileDebugDir());
  const std::string autofuse_dir = base_path + "autofuse_compile_debug/";
  if (!TryCreateDir(autofuse_dir)) {
    return false;
  }

  const std::string pid_dir = autofuse_dir + "ascgen_dump_pid_" + std::to_string(mmGetPid()) + "/";
  if (!TryCreateDir(pid_dir)) {
    return false;
  }

  g_cached_pid_dir = pid_dir;
  return true;
}

static std::string GetDumpGraphPrefixAndCreateDir() {
  if (!InitDumpDirectories()) {
    return "";
  }

  // 如果没有设置 fused_graph_name，直接返回 pid 目录
  if (g_current_fused_graph_name.empty()) {
    return g_cached_pid_dir;
  }

  // 创建 graph 子目录
  const std::string graph_dir = g_cached_pid_dir + g_current_fused_graph_name + "/";
  if (g_created_graph_dirs.count(graph_dir) > 0) {
    return graph_dir;
  }

  if (TryCreateDir(graph_dir)) {
    g_created_graph_dirs.insert(graph_dir);
    return graph_dir;
  }
  return g_cached_pid_dir;
}

static std::string ExecConditionToStr(af::ExecuteCondition condition) {
  static const std::map<af::ExecuteCondition, std::string> kTypeToStr = {
      {af::ExecuteCondition::kNoCache, "no_cache"},
      {af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis, "cache_block_split_fused_brc_axis"},
      {af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis, "cache_block_split_origin_brc_axis"},
      {af::ExecuteCondition::kConditionInvalid, "invalid"}};

  auto it = kTypeToStr.find(condition);
  if (it != kTypeToStr.end()) {
    return it->second;
  }
  return "unknown";
}

static std::string ComputeUnitToStr(af::ComputeUnit compute_unit) {
  const char *kTypeName[] = {
      [static_cast<int32_t>(af::ComputeUnit::kUnitNone)] = "None",
      [static_cast<int32_t>(af::ComputeUnit::kUnitMTE1)] = "MTE1",
      [static_cast<int32_t>(af::ComputeUnit::kUnitMTE2)] = "MTE2",
      [static_cast<int32_t>(af::ComputeUnit::kUnitMTE3)] = "MTE3",
      [static_cast<int32_t>(af::ComputeUnit::kUnitScalar)] = "Scalar",
      [static_cast<int32_t>(af::ComputeUnit::kUnitVector)] = "Vector",
      [static_cast<int32_t>(af::ComputeUnit::kUnitCube)] = "Cube",
  };

  if (static_cast<size_t>(compute_unit) >= sizeof(kTypeName) / sizeof(kTypeName[0])) {
    return "unknown";
  }
  return kTypeName[static_cast<size_t>(compute_unit)];
}

static std::string MemHardwareToStr(af::MemHardware mem_hardware) {
  const char *kTypeName[] = {
      [static_cast<int32_t>(af::MemHardware::kMemHardwareGM)] = "GM",
      [static_cast<int32_t>(af::MemHardware::kMemHardwareUB)] = "UB",
  };

  if (static_cast<size_t>(mem_hardware) >= sizeof(kTypeName) / sizeof(kTypeName[0])) {
    return "unknown";
  }
  return kTypeName[static_cast<size_t>(mem_hardware)];
}

static std::string PositionToStr(af::Position position) {
  static const std::map<af::Position, std::string> gPositionToStr = {
      {af::Position::kPositionGM, "TPosition::GM"},
      {af::Position::kPositionVecIn, "TPosition::VECIN"},
      {af::Position::kPositionVecCalc, "TPosition::VECCALC"},
      {af::Position::kPositionVecOut, "TPosition::VECOUT"}};
  auto it = gPositionToStr.find(position);
  if (it != gPositionToStr.end()) {
    return it->second;
  }
  return "unknown";
}

static std::stringstream &GraphNameStr(std::stringstream &ss, const ascir::Graph &graph) {
  ss << "Graph: " << graph.GetName() << std::endl;
  return ss;
}

static std::stringstream &GraphSizeStr(std::stringstream &ss, const ascir::Graph &graph) {
  ss << "Sizes:" << std::endl;
  auto all_size_var = graph.GetAllSizeVar();
  for (const auto &size_var : all_size_var) {
    if (size_var->expr.GetExprType() == af::ExprType::kExprVariable) {
      ss << "  " << size_var->expr.Str().get() << ": VAR" << std::endl;
    } else if (size_var->expr.GetExprType() <= af::ExprType::kExprConstantRation) {
      int64_t val;
      size_var->expr.GetConstValue(val);
      ss << "  " << size_var->expr.Str().get() << ": CONST(" << val << ")" << std::endl;
    } else {
      //
    }
  }

  return ss;
}

static std::stringstream &GraphAxisStr(std::stringstream &ss, const ascir::Graph &graph) {
  static const std::map<af::Axis::Type, std::string> kTypeToStr = {
      {af::Axis::Type::kAxisTypeOriginal, "ORIGINAL"},   {af::Axis::Type::kAxisTypeBlockOuter, "BLOCK_OUT"},
      {af::Axis::Type::kAxisTypeBlockInner, "BLOCK_IN"}, {af::Axis::Type::kAxisTypeTileOuter, "TILE_OUT"},
      {af::Axis::Type::kAxisTypeTileInner, "TILE_IN"},   {af::Axis::Type::kAxisTypeMerged, "MERGED"},
  };

  ss << "Axis:" << std::endl;
  auto all_axis = graph.GetAllAxis();
  std::map<af::AxisId, std::string> axis_id_to_name = GetAxisIdToName(all_axis);
  for (auto &axis : all_axis) {
    ss << "  " << axis->name << "(" << axis->id << ") : ";
    auto iter = kTypeToStr.find(axis->type);
    if (iter != kTypeToStr.end()) {
      ss << iter->second;
    } else {
      ss << "UNKNOWN";
    }
    ss << ", size:" << af::SymbolicUtils::ToString(axis->size) << ", ";

    if (!axis->from.empty()) {
      ss << ", from: {";
      for (auto from_axis : axis->from) {
        ss << axis_id_to_name[from_axis] << ", ";
      }
      ss << "}";
    }

    ss << std::endl;
  }
  return ss;
}

static std::stringstream &NodeAttrStr(std::stringstream &ss, const ascir::Graph &graph, ascir::NodeView &node,
                                      bool verbose = false) {
  auto &ir_attr = node->GetOpDesc()->GetAttrsGroup<af::AscNodeAttr>()->ir_attr;
  if (ir_attr != nullptr) {
    ascendc_ir::proto::AscIrAttrDef asc_ir_attr_def;
    (void)ir_attr->Serialize(asc_ir_attr_def);
    if (!asc_ir_attr_def.attr().empty()) {
      ss << "    .ir_attr =  {";
      for (const auto &pair : asc_ir_attr_def.attr()) {
        ss << "." << pair.first << " = " << pair.second.ShortDebugString();
      }
      ss << "}" << std::endl;
    }
  }

  auto all_axis = graph.GetAllAxis();
  std::map<af::AxisId, std::string> axis_id_to_name = GetAxisIdToName(all_axis);

  const bool is_buf = (node->attr.api.type == af::ApiType::kAPITypeBuffer);

  if (!is_buf) {
    ss << "    .axis = "
       << "{";
    for (auto axis_id : node->attr.sched.axis) {
      ss << axis_id_to_name[axis_id] << ", ";
    }
    ss << "}" << std::endl;
  }

  // Node sched exec_condition (只显示非默认值)
  if (node->attr.sched.exec_condition != af::ExecuteCondition::kNoCache) {
    ss << "    .exec_condition = " << ExecConditionToStr(node->attr.sched.exec_condition) << std::endl;
  }

  if (verbose && !is_buf) {
    const auto loop_axis = node->attr.sched.loop_axis;
    if ((loop_axis >= 0) && (loop_axis < static_cast<int64_t>(all_axis.size()))) {
      ss << "    .loop_axis = " << axis_id_to_name[loop_axis] << std::endl;
    } else {
      ss << "    .loop_axis = " << std::to_string(loop_axis) << std::endl;
    }
  }

  if (verbose && !is_buf) {
    ss << "    .api.unit = " << ComputeUnitToStr(node->attr.api.unit) << std::endl;
    const auto &tmp_buffers = node->attr.tmp_buffers;
    if (!tmp_buffers.empty()) {
      ss << "    .tmp_buf = {";
      for (size_t i = 0; i < tmp_buffers.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "{buf_id=" << tmp_buffers[i].id << ", size=" << af::SymbolicUtils::ToString(tmp_buffers[i].buf_desc.size)
           << "}";
      }
      ss << "}" << std::endl;
    }
  }

  return ss;
}

static std::stringstream &NodeInputStr(std::stringstream &ss, const std::vector<std::string> &input_names) {
  bool all_nil = true;
  for (const auto &name : input_names) {
    if (name != "nil") {
      all_nil = false;
      break;
    }
  }
  if (all_nil) {
    return ss;
  }

  if (input_names.size() == 1) {
    ss << "    .x = " << input_names[0] << std::endl;
  } else {
    ss << "    .x = {";
    for (size_t i = 0; i < input_names.size(); ++i) {
      if (i > 0) ss << ", ";
      if (input_names[i] != "nil") {
        ss << input_names[i];
      }
    }
    ss << "}" << std::endl;
  }

  return ss;
}

// 输出形状信息 (axis, repeats, strides)
static std::stringstream &OutputShapeStr(std::stringstream &ss, const ascir::Graph &graph,
                                         const af::AscTensorAttr &output_attr) {
  // 输出 axis 列表
  if (!output_attr.axis.empty()) {
    auto all_axis = graph.GetAllAxis();
    std::map<af::AxisId, std::string> axis_id_to_name = GetAxisIdToName(all_axis);
    ss << "        .axis = {";
    for (size_t i = 0; i < output_attr.axis.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << axis_id_to_name[output_attr.axis[i]];
    }
    ss << "}" << std::endl;
  }

  // 输出 repeats
  if (!output_attr.repeats.empty()) {
    ss << "        .repeats = (";
    for (size_t i = 0; i < output_attr.repeats.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << af::SymbolicUtils::ToString(output_attr.repeats[i]);
    }
    ss << ")" << std::endl;
  }

  // 输出 strides
  if (!output_attr.strides.empty()) {
    ss << "        .strides = (";
    for (size_t i = 0; i < output_attr.strides.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << af::SymbolicUtils::ToString(output_attr.strides[i]);
    }
    ss << ")" << std::endl;
  }

  return ss;
}

// 输出 vectorized 信息
static std::stringstream &OutputVectorizedStr(std::stringstream &ss, const ascir::Graph &graph,
                                              const af::AscTensorAttr &output_attr) {
  if (!output_attr.vectorized_axis.empty()) {
    auto all_axis = graph.GetAllAxis();
    std::map<af::AxisId, std::string> axis_id_to_name = GetAxisIdToName(all_axis);

    ss << "        .vectorized_strides = {";
    for (size_t i = 0; i < output_attr.vectorized_axis.size(); ++i) {
      if (i > 0) ss << ", ";
      auto axis_name = axis_id_to_name[output_attr.vectorized_axis[i]];
      ss << axis_name << ":";
      if (i < output_attr.vectorized_strides.size()) {
        ss << af::SymbolicUtils::ToString(output_attr.vectorized_strides[i]);
      }
    }
    ss << "}" << std::endl;
  }
  return ss;
}

// 输出 Queue 类型的内存信息
static std::stringstream &OutputQueueMemStr(std::stringstream &ss, const af::AscTensorAttr &output_attr,
                                            const std::string &pos_str) {
  const auto &que = output_attr.que;
  ss << MemHardwareToStr(output_attr.mem.hardware) << "[";
  if (output_attr.mem.tensor_id != af::kIdNone) {
    ss << "tensor_id=" << output_attr.mem.tensor_id << ", ";
  }
  ss << "que_id=" << que.id;
  if (output_attr.mem.reuse_id >= 0) {
    ss << ", reuse_id=" << output_attr.mem.reuse_id;
  }
  ss << ", depth=" << que.depth << ", pos=" << pos_str << "]";
  return ss;
}

// 输出 Buffer 类型的内存信息
static std::stringstream &OutputBufferMemStr(std::stringstream &ss, const af::AscTensorAttr &output_attr,
                                             const std::string &pos_str) {
  ss << MemHardwareToStr(output_attr.mem.hardware) << "[";
  if (output_attr.mem.tensor_id != af::kIdNone) {
    ss << "tensor_id=" << output_attr.mem.tensor_id << ", ";
  }
  ss << "buf_id=" << output_attr.buf.id;
  if (output_attr.mem.reuse_id >= 0) {
    ss << ", reuse_id=" << output_attr.mem.reuse_id;
  }
  ss << ", pos=" << pos_str << "]";
  return ss;
}

// 输出普通类型的内存信息
static std::stringstream &OutputNormalMemStr(std::stringstream &ss, const af::AscTensorAttr &output_attr,
                                             const std::string &pos_str) {
  ss << MemHardwareToStr(output_attr.mem.hardware);
  if (output_attr.mem.tensor_id != af::kIdNone) {
    ss << "[tensor_id=" << output_attr.mem.tensor_id << ", pos=" << pos_str << "]";
  } else {
    ss << "[pos=" << pos_str << "]";
  }
  return ss;
}

// 输出内存信息
static std::stringstream &OutputMemStr(std::stringstream &ss, const af::AscTensorAttr &output_attr, bool verbose) {
  if (!verbose && (output_attr.mem.alloc_type != af::AllocType::kAllocTypeQueue) &&
      (output_attr.mem.alloc_type != af::AllocType::kAllocTypeBuffer)) {
    return ss;
  }

  ss << "        .mem = ";
  // 获取 position 字符串
  std::string pos_str;
  switch (output_attr.mem.position) {
    case af::Position::kPositionVecIn:
      pos_str = "VECIN";
      break;
    case af::Position::kPositionVecCalc:
      pos_str = "VECCALC";
      break;
    case af::Position::kPositionVecOut:
      pos_str = "VECOUT";
      break;
    case af::Position::kPositionGM:
      pos_str = "GM";
      break;
    default:
      pos_str = PositionToStr(output_attr.mem.position);
      break;
  }

  if (output_attr.mem.alloc_type == af::AllocType::kAllocTypeQueue) {
    OutputQueueMemStr(ss, output_attr, pos_str);
  } else if (output_attr.mem.alloc_type == af::AllocType::kAllocTypeBuffer) {
    OutputBufferMemStr(ss, output_attr, pos_str);
  } else {
    OutputNormalMemStr(ss, output_attr, pos_str);
  }
  ss << std::endl;
  return ss;
}

// 输出节点信息
static std::stringstream &NodeOutputStr(std::stringstream &ss, const ascir::Graph &graph, af::AscNode &node,
                                        af::AscTensorAttr &output_attr, size_t output_idx, bool verbose) {
  auto output_name = node.GetOpDesc()->GetOutputNameByIndex(output_idx);
  auto dtype = node.GetOpDesc()->GetOutputDesc(output_idx).GetDataType();

  ss << "    ." << output_name << ": " << DtypeToStr(dtype) << std::endl;

  OutputShapeStr(ss, graph, output_attr);
  OutputVectorizedStr(ss, graph, output_attr);
  OutputMemStr(ss, output_attr, verbose);

  return ss;
}

static void DumpGraphText(const Graph &graph, const string &suffix, const uint32_t graph_id, const bool verbose,
                          const std::string &prefix) {
  // 判断是否是子图
  const bool is_subgraph = (suffix.find("_Subgraph_") != std::string::npos);
  // 使用新的 MLIR 风格格式
  auto dump_asc_graph = DebugStrNew(graph, verbose, is_subgraph);

  std::string base_name = "ascgraph_" + FormatDumpIndex(g_current_fused_graph_dump_index) + "_" + graph.GetName() +
                          "_" + suffix + "_" + std::to_string(graph_id) + ".txt";
  std::string file_name = prefix + LimitDumpFileName(SanitizeFileName(base_name));
  std::ofstream f_stream(file_name);
  if (f_stream.is_open()) {
    f_stream << dump_asc_graph << std::endl;
    f_stream.close();
  }
}

// 文件名格式: {prefix}/ge_onnx_{index:05d}_{graph_name}_{suffix}.pbtxt
static void DumpComputeGraphImpl(const af::ComputeGraphPtr &compute_graph, const std::string &suffix,
                                 const std::string &prefix) {
  if (compute_graph == nullptr) {
    return;
  }
  std::string base_name = "ge_onnx_" + FormatDumpIndex(g_current_onnx_dump_index) + "_" + compute_graph->GetName() +
                          "_" + suffix + ".pbtxt";
  std::string file_name = prefix + LimitDumpFileName(SanitizeFileName(base_name));
  af::Model model("GE", "");
  model.SetGraph(compute_graph);
  af::onnx::ModelProto model_proto;
  if (!af::OnnxUtils::ConvertGeModelToModelProto(model, model_proto)) {
    GELOGW("[DumpGraph] ConvertGeModelToModelProto failed for graph: %s", compute_graph->GetName().c_str());
    return;
  }

  af::GraphUtils::WriteProtoToTextFile(model_proto, file_name.c_str());
  ++g_current_onnx_dump_index;
}

void DumpComputeGraph(const af::ComputeGraphPtr &compute_graph, const std::string &suffix, bool always_dump) {
  // 使用统一的 AUTOFUSE_DFX_FLAGS 环境变量检查
  if (!always_dump && !ascir::utils::IsCodegenCompileEnabled()) {
    return;
  }
  std::string prefix = GetDumpGraphPrefixAndCreateDir();
  if (prefix.empty()) {
    return;
  }
  DumpComputeGraphImpl(compute_graph, suffix, prefix);
}

static void DumpGraphImpl(const ascir::Graph &graph, const std::string &suffix, const uint32_t graph_id,
                          const bool verbose) {
  std::string prefix = GetDumpGraphPrefixAndCreateDir();
  if (prefix.empty()) {
    return;
  }
  // dump txt
  DumpGraphText(graph, suffix, graph_id, verbose, prefix);
  std::vector<af::AscGraph> subgraphs;
  (void)graph.GetAllSubGraphs(subgraphs);
  for (auto &subgraph : subgraphs) {
    DumpGraphText(subgraph, "_Subgraph_", graph_id, verbose, prefix);
  }

  // dump onnx
  const auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  DumpComputeGraphImpl(compute_graph, suffix, prefix);

  ++g_current_fused_graph_dump_index;
}

void DumpGraph(const ascir::Graph &graph, const std::string &suffix, const uint32_t graph_id, const bool verbose) {
  if (!ascir::utils::IsCodegenCompileEnabled()) {
    // 环境变量没开启时, 捕获图对象为异常退出时维测服务
    AscGraphDumperContext::GetThreadLocalCtx().AddWatchGraph(suffix, graph);
    return;
  }
  DumpGraphImpl(graph, suffix, graph_id, verbose);
}

void AlwaysDumpGraph(const Graph &graph, const string &suffix, const uint32_t graph_id, const bool verbose) {
  // AlwaysDumpGraph 用于在没开 codegen_compile_debug=true时，当异常时强制 dump
  if (ascir::utils::IsCodegenCompileEnabled()) {
    // 正常流程已经 dump，这里不需要再 dump
    return;
  }
  DumpGraphImpl(graph, suffix, graph_id, verbose);
}

void DumpImplGraphs(const std::vector<ascir::Graph> &graphs, const std::string &suffix) {
  if (!ascir::utils::IsCodegenCompileEnabled()) {
    return;
  }
  for (size_t i = 0UL; i < graphs.size(); ++i) {
    DumpGraph(graphs[i], std::to_string(i) + "_" + suffix, i, true);
  }
}

void DumpPyCode(const af::AscGraph &graph) {
  if (!ascir::utils::IsCodegenCompileEnabled()) {
    return;
  }
  const std::string prefix = GetDumpGraphPrefixAndCreateDir();
  if (prefix.empty()) {
    return;
  }
  const std::string base_name = "py_code_" + graph.GetName() + ".py";
  std::string file_name = prefix + LimitDumpFileName(SanitizeFileName(base_name));
  af::ascir::PythonCodeDumper dumper;
  (void)dumper.Dump(graph, file_name);
}

std::string IdentifierToStr(ascir::Identifier id) {
  if (id == af::kIdNone) {
    return "nil";
  } else {
    return std::to_string(id);
  }
}

std::string DebugStr(const ascir::Graph &graph, bool verbose) {
  std::stringstream ss;

  GraphNameStr(ss, graph);
  GraphSizeStr(ss, graph);
  GraphAxisStr(ss, graph);

  ss << "Nodes:" << std::endl;
  size_t idx = 0UL;
  for (auto node : graph.GetAllNodes()) {
    ss << "  " << node->GetName() << ": " << node->GetType() << " (" << idx++ << ")" << std::endl;

    NodeAttrStr(ss, graph, node, verbose);

    // Node inputs
    auto input_names = dumper::CollectInputNames(graph, node);
    NodeInputStr(ss, input_names);

    // Node outputs
    for (size_t i = 0UL; i < node->outputs().size(); i++) {
      NodeOutputStr(ss, graph, *node, node->outputs[i].attr, i, verbose);
    }
  }

  return ss.str();
}

std::string DebugHintGraphStr(const ascir::HintGraph &graph) {
  return DebugStr(graph, false);
}

std::string DebugImplGraphStr(const ascir::ImplGraph &graph) {
  return DebugStr(graph, true);
}

std::string DebugStrNew(const ascir::Graph &graph, bool verbose, bool is_subgraph) {
  return ascir::dumper::DumpGraphText(graph, verbose, is_subgraph);
}

void DumpScheduleResult(const ascir::FusedScheduledResult &fused_scheduled_result, const std::string &suffix,
                        uint32_t graph_id, bool verbose) {
  for (const auto &results : fused_scheduled_result.node_idx_to_scheduled_results) {
    for (const auto &result : results) {
      for (const auto &schedule_group : result.schedule_groups) {
        for (const auto &impl_graph : schedule_group.impl_graphs) {
          ascir::utils::DumpGraph(impl_graph, suffix, graph_id, verbose);
        }
      }
    }
  }
}

bool UseSmallTailConcatApi(const af::AscNode &node, bool *output_need_align) {
  bool force_small_tail = false;
  (void)af::AttrUtils::GetBool(node.GetOpDesc(), "_concat_small_tail", force_small_tail);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(force_small_tail, true, "[%s] marked use small tail kernel", node.GetNamePtr());
  auto node_inputs = node.inputs;
  GE_CHK_BOOL_RET_SPECIAL_STATUS(node_inputs.Size() <= 1U, false, "input num = %u, do not use small tail concat api",
                                 node_inputs.Size());
  const auto dtype_size = GetSizeByDataType(node_inputs[0].attr.dtype);
  GE_WARN_ASSERT(dtype_size > 0);  // 其实下一判断可以确保非0, 然而静态检查识别不了
  GE_CHK_BOOL_RET_SPECIAL_STATUS(dtype_size != sizeof(uint16_t) && dtype_size != sizeof(uint32_t), false,
                                 "[%s] only support dtype size = 2 or 4, but = %d", node.GetNamePtr(), dtype_size);
  const int32_t kAlignSize = 32 / dtype_size;
  std::vector<int64_t> src_col_sizes;
  int64_t dst_col_size = 0;
  size_t concat_dim;
  GE_CHK_BOOL_RET_SPECIAL_STATUS(!GetConcatDimAndColSizes(node, concat_dim, src_col_sizes, dst_col_size), false,
                                 "do not use small tail concat api");
  size_t aligned_cnt = 0;
  int64_t gcd = 16;
  for (const auto src_col_size : src_col_sizes) {
    aligned_cnt += (src_col_size % kAlignSize == 0) ? 1 : 0;
    gcd = ascgen_utils::Gcd(gcd, src_col_size);
  }
  // 全对齐, 使用全对齐的api性能更好
  GE_CHK_BOOL_RET_SPECIAL_STATUS(aligned_cnt == node_inputs.Size(), false, "[%s] inputs are all aligned",
                                 node.GetNamePtr());
  constexpr int64_t kSrcMaxSrcColSize = 64;
  constexpr uint32_t kMaxDstColSize = 96;  // 最大96K的tmp buffer, 只支持96
  for (size_t i = 0U; i < src_col_sizes.size(); ++i) {
    GE_CHK_BOOL_RET_SPECIAL_STATUS((gcd == 0) || (src_col_sizes[i] / gcd > kSrcMaxSrcColSize), false,
                                   "[%s] input[%zu] col_size = %ld, gcd = %ld, not a small dim", node.GetNamePtr(), i,
                                   src_col_sizes[i], gcd);
  }
  GE_CHK_BOOL_RET_SPECIAL_STATUS((gcd == 0) || (dst_col_size / gcd) > kMaxDstColSize, false,
                                 "[%s] output col_size = %ld, gcd = %ld, not a small dim", node.GetNamePtr(),
                                 dst_col_size, gcd);
  if (!IsStoreWithoutStride(node)) {
    bool concat_last_dim = concat_dim == (node_inputs[0].attr.repeats.size() - 1UL);
    GE_CHK_BOOL_RET_SPECIAL_STATUS((!concat_last_dim) && (dst_col_size % kAlignSize != 0), false,
                                   "[%s] Concat on non tail dim, output is not aligned and need store with strides",
                                   node.GetNamePtr());
    if (output_need_align != nullptr) {
      *output_need_align = (dst_col_size % kAlignSize != 0);
    }
  }
  GELOGI("[%s] will use small tail kernel", node.GetNamePtr());
  return true;
}

bool IsConcatAllInputsAligned(const af::AscNode &node) {
  constexpr int32_t kAlignment = 32;
  auto node_inputs = node.inputs;
  auto concat_dim = std::numeric_limits<size_t>::max();
  GE_WARN_ASSERT(GetConcatDim(node, concat_dim));
  GE_WARN_ASSERT(concat_dim < node_inputs[0].attr.repeats.size());

  const auto dtype_size = GetSizeByDataType(node_inputs[0].attr.dtype);
  GE_WARN_ASSERT(dtype_size > 0);
  for (size_t i = 0UL; i < node_inputs.Size(); ++i) {
    const auto &input_repeats = node_inputs[i].attr.repeats;
    af::Expression size = af::Symbol(dtype_size);
    for (size_t j = concat_dim; j < input_repeats.size(); ++j) {
      size = size * input_repeats[j];
    }
    if (af::SymbolicUtils::StaticCheckEq(af::sym::Mod(size, af::Symbol(kAlignment)), af::ops::Zero) !=
        af::TriBool::kTrue) {
      GELOGI("input[%zu] size = %s, is not aligned", i, af::SymbolicUtils::ToString(size).c_str());
      return false;
    }
  }
  GELOGI("[%s] All inputs are aligned", node.GetNamePtr());
  return true;
}

af::TriBool AreConcatInputShapesEqual(const af::AscNodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  auto node_inputs = node->inputs;
  af::TriBool is_equal = af::TriBool::kTrue;
  if (node_inputs.Size() > 1) {
    auto concat_dim = std::numeric_limits<size_t>::max();
    GE_WARN_ASSERT(GetConcatDim(*node, concat_dim));
    GE_WARN_ASSERT(concat_dim < node_inputs[0].attr.repeats.size());
    const auto &first_concat_dim_size = node_inputs[0].attr.repeats[concat_dim];
    for (uint32_t i = 1U; i < node_inputs.Size(); ++i) {
      const auto &concat_dim_size = node_inputs[i].attr.repeats[concat_dim];
      // unknown时在运行时确定
      const auto cmp_ret = af::SymbolicUtils::StaticCheckEq(first_concat_dim_size, concat_dim_size);
      if (cmp_ret == af::TriBool::kFalse) {
        GELOGD("src_cols[0] = %s, src_cols[%u] = %s, shapes are different", first_concat_dim_size.Str().get(), i,
               concat_dim_size.Str().get());
        is_equal = af::TriBool::kFalse;
        break;
      } else if (cmp_ret == af::TriBool::kUnknown) {
        GELOGD("src_cols[0] = %s, src_cols[%u] = %s, compare result is unknown", first_concat_dim_size.Str().get(), i,
               concat_dim_size.Str().get());
        is_equal = af::TriBool::kUnknown;
      } else {
        // equal, do nothing
      }
    }
  }
  return is_equal;
}

bool AreAllInputDistinct(const af::NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  std::set<const af::Node *> distinct_nodes;
  for (const auto &in_node : node->GetInDataNodes()) {
    if (!distinct_nodes.emplace(in_node.get()).second) {
      GELOGD("%s: multiple inputs share same input: %s", node->GetNamePtr(), in_node->GetNamePtr());
      return false;
    }
  }
  return true;
}

bool AreAllInputsFromPosition(const af::AscNodePtr &node, Position position) {
  GE_ASSERT_NOTNULL(node);
  for (uint32_t i = 0U; i < node->inputs.Size(); ++i) {
    const auto &input = node->inputs[i];
    if (input.attr.mem.position != position) {
      GELOGD("%s: input[%u] not from position: %d", node->GetNamePtr(), i, static_cast<int32_t>(position));
      return false;
    }
  }
  return true;
}

std::string SetCurrentFusedGraphName(const std::string &name) {
  std::string prev_name = std::move(g_current_fused_graph_name);

  // 如果名称为空，直接清空
  if (name.empty()) {
    g_current_fused_graph_name.clear();
    g_current_fused_graph_dump_index = 0UL;
    g_current_onnx_dump_index = 0UL;
    GELOGI("[DumpGraph] Clear fused_graph_name");
    return prev_name;
  }

  // 如果名称发生变化，重置计数器
  // 对 name 进行 sanitize 以确保可以用于目录名
  std::string safe_name = SanitizeFileName(name);
  if (prev_name != safe_name) {
    g_current_fused_graph_name = safe_name;
    g_current_fused_graph_dump_index = 0UL;
    g_current_onnx_dump_index = 0UL;
    GELOGI("[DumpGraph] Set fused_graph_name to: %s (original: %s), reset dump index to 0", safe_name.c_str(),
           name.c_str());
  }
  return prev_name;
}

void ResetDumpConfig() {
  // 清除目录缓存
  g_cached_pid_dir.clear();
  g_created_graph_dirs.clear();
  g_current_fused_graph_name.clear();
  g_current_fused_graph_dump_index = 0UL;
  g_current_onnx_dump_index = 0UL;
}

bool IsCodegenCompileEnabled() {
  return GetDumpConfig().enabled;
}

std::string GetDumpFilePrefix() {
  if (!ascir::utils::IsCodegenCompileEnabled()) {
    return "";
  }
  return GetDumpGraphPrefixAndCreateDir();
}
}  // namespace ascir::utils
