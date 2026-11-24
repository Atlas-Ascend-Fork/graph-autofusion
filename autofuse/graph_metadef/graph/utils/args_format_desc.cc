/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/args_format_desc.h"
#include <cstring>
#include <functional>
#include <map>
#include <sstream>
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"

namespace af {
constexpr size_t kMaxDimNum = 25UL;
constexpr size_t kMaxWorkspaceNum = 16UL;
constexpr int32_t kDecimalCarry = 10;
constexpr int32_t kAsciiZero = 48;
constexpr int32_t kDigitFormatCnt = 1;
constexpr int32_t kAmbiguousIrIdx = -1;

using ParseFunc =
    std::function<graphStatus(const OpDescPtr &, const std::string &, const AddrType type, std::vector<ArgDesc> &)>;
using GetArgsSize = std::function<graphStatus(const OpDescPtr &, const ArgDesc &, size_t &)>;
using SerializeFunc = std::function<void(std::stringstream &ss, const std::string &pattern, const ArgDesc &arg_desc)>;

struct PatternHandler {
  ParseFunc parse;
  GetArgsSize getArgsSize;
  SerializeFunc serialize;
  AddrType type;
};

graphStatus FindSkSubNode(const OpDescPtr &sk_op, const int32_t id, NodePtr &sub_node) {
  GE_ASSERT_NOTNULL(sk_op);
  std::shared_ptr<af::ComputeGraph> sub_graph = nullptr;
  sub_graph = sk_op->TryGetExtAttr("_sk_sub_graph", sub_graph);
  GE_ASSERT_NOTNULL(sub_graph);
  for (const auto &node : sub_graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetOpDesc()->GetId() == static_cast<int64_t>(id)) {
      sub_node = node;
      GELOGI("find %d sub node %s from sk node %s", id, node->GetNamePtr(), sk_op->GetNamePtr());
      return ge::GRAPH_SUCCESS;
    }
  }
  GELOGE(ge::GRAPH_FAILED, "can not find %d sub node from sk node %s", id, sk_op->GetNamePtr());
  return ge::GRAPH_FAILED;
}

static std::set<AddrType> kNeedEasyParserTypes{
    AddrType::INPUT,          AddrType::OUTPUT,          AddrType::INPUT_DESC, AddrType::OUTPUT_DESC,
    AddrType::INPUT_INSTANCE, AddrType::OUTPUT_INSTANCE, AddrType::WORKSPACE};

static graphStatus DefaultCalcSize(const OpDescPtr &op_desc, const ArgDesc &arg_desc, size_t &size) {
  (void)op_desc;
  (void)arg_desc;
  size += sizeof(uintptr_t);
  return ge::GRAPH_SUCCESS;
}

static graphStatus DefaultParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                                 std::vector<ArgDesc> &args_desc) {
  (void)op_desc;
  (void)pattern_str;
  args_desc.push_back({type, kAmbiguousIrIdx, false, {0}});
  return ge::GRAPH_SUCCESS;
}

static graphStatus PlaceholderParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                                     std::vector<ArgDesc> &args_desc) {
  (void)op_desc;
  auto width = ArgsFormatWidth::BIT64;
  if (pattern_str == ".32b") {
    width = ArgsFormatWidth::BIT32;
  } else if (!pattern_str.empty()) {
    GELOGE(ge::PARAM_INVALID, "Args format [%s] match failed, it may be unsupported.", pattern_str.c_str());
    return ge::GRAPH_FAILED;
  }
  args_desc.push_back({type, static_cast<int32_t>(width), false, {0}});
  return ge::GRAPH_SUCCESS;
}

static void PlaceholderSerializer(std::stringstream &ss, const std::string &pattern, const ArgDesc &arg_desc) {
  ss << pattern;
  if (arg_desc.ir_idx == static_cast<int32_t>(ArgsFormatWidth::BIT32)) {
    ss << ".32b";
  }
}

static void DefaultSerializer(std::stringstream &ss, const std::string &pattern, const ArgDesc &arg_desc) {
  (void)arg_desc;
  ss << pattern;
}

static void FftsTilingSerializer(std::stringstream &ss, const std::string &pattern, const ArgDesc &arg_desc) {
  ss << pattern;
  if (arg_desc.ir_idx == 0) {
    ss << ".non_tail";
  } else {
    ss << ".tail";
  }
}

static void ArrayLikeSerializer(std::stringstream &ss, const std::string &pattern, const ArgDesc &arg_desc) {
  ss << pattern;
  if (arg_desc.ir_idx >= 0) {
    ss << std::to_string(arg_desc.ir_idx);
    if (!arg_desc.folded) {
      ss << '*';
    }
  } else {
    ss << '*';
  }
}

static graphStatus WorkspaceCalcSize(const OpDescPtr &op_desc, const ArgDesc &arg_desc, size_t &size) {
  (void)op_desc;
  if (arg_desc.ir_idx == kAmbiguousIrIdx) {
    size += sizeof(uintptr_t) * kMaxWorkspaceNum;
  } else {
    size += sizeof(uintptr_t);
  }
  return ge::GRAPH_SUCCESS;
}

static graphStatus EventAddrParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                                   std::vector<ArgDesc> &args_desc) {
  (void)op_desc;
  int32_t ir_idx = 0;
  const std::string prefix = "event_addr";
  for (size_t i = prefix.size(); i < pattern_str.size(); ++i) {
    if (isdigit(pattern_str[i])) {
      ir_idx = ir_idx * kDecimalCarry + static_cast<int32_t>(pattern_str[i]) - kAsciiZero;
    }
  }
  args_desc.push_back({type, ir_idx, false, {0}});
  return ge::GRAPH_SUCCESS;
}

static void EventAddrSerializer(std::stringstream &ss, const std::string &pattern, const ArgDesc &arg_desc) {
  ss << pattern << arg_desc.ir_idx << "*";
}

static graphStatus WorkspaceParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                                   std::vector<ArgDesc> &args_desc) {
  (void)op_desc;
  if (pattern_str == "ws*") {
    args_desc.push_back({type, kAmbiguousIrIdx, false, {0}});
    return ge::GRAPH_SUCCESS;
  }
  int32_t ir_idx = 0;
  for (size_t i = 2UL; i < pattern_str.size(); ++i) {
    if (isdigit(pattern_str[i])) {
      ir_idx = ir_idx * kDecimalCarry + static_cast<int32_t>(pattern_str[i]) - kAsciiZero;
    }
  }
  args_desc.push_back({type, ir_idx, false, {0}});
  return ge::GRAPH_SUCCESS;
}

static graphStatus InputCalcSize(const OpDescPtr &op_desc, const ArgDesc &arg_desc, size_t &size) {
  const auto &ir_inputs = op_desc->GetIrInputs();
  size_t count = 0UL;
  if (arg_desc.ir_idx >= 0) {
    // 非通配符场景
    GE_ASSERT((static_cast<size_t>(arg_desc.ir_idx) < ir_inputs.size()), "ir_index [%d] is out of range [0, %zu)",
              arg_desc.ir_idx, ir_inputs.size());
    if (ir_inputs[arg_desc.ir_idx].second == IrInputType::kIrInputDynamic) {
      if (arg_desc.folded) {
        ++count;  // pointer to addr
      }
      int32_t dyn_num = 0;
      for (auto &iter : op_desc->GetAllInputName()) {
        if (iter.first == ir_inputs[arg_desc.ir_idx].first + std::to_string(dyn_num)) {
          ++dyn_num;
          ++count;  // real input_addr
        }
      }
    } else {
      ++count;
    }
  } else {
    // 通配符场景，非动态输入默认展开, 动态输入按照i0形式折叠
    for (const auto &ir_input : ir_inputs) {
      ++count;
      if (ir_input.second == IrInputType::kIrInputDynamic) {
        int32_t dyn_num = 0;
        for (auto &iter : op_desc->GetAllInputName()) {
          if (iter.first == ir_input.first + std::to_string(dyn_num)) {
            ++count;  // real input addr
            ++dyn_num;
          }
        }
      }
    }
  }
  size += count * sizeof(uintptr_t);

  return ge::GRAPH_SUCCESS;
}

static graphStatus InputInstanceParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                                       std::vector<ArgDesc> &arg_descs) {
  GE_ASSERT_NOTNULL(op_desc);
  const size_t valid_input_size = op_desc->GetInputsSize();
  if (pattern_str == "i_instance*") {
    // 为了方便加载时使用，通配符场景解析后默认展开成多个
    for (size_t i = 0UL; i < valid_input_size; ++i) {
      arg_descs.push_back({type, static_cast<int32_t>(i), false, {0}});
    }
  } else {
    int32_t ir_idx{0};
    GE_ASSERT_TRUE(sscanf_s(pattern_str.c_str(), "i_instance%d", &ir_idx) == kDigitFormatCnt,
                   "Arg format [%s] is invalid", pattern_str.c_str());
    GE_ASSERT(static_cast<size_t>(ir_idx) < valid_input_size, "ir index [%d] is out of range [0, %zu)", ir_idx,
              valid_input_size);
    arg_descs.push_back({type, ir_idx, false, {0}});
  }
  return af::SUCCESS;
}

static graphStatus InputInstanceCalcSize(const OpDescPtr &op_desc, const ArgDesc &arg_desc, size_t &size) {
  size_t count = 1UL;
  if (arg_desc.ir_idx < 0) {
    count = op_desc->GetInputsSize();
  }
  if (arg_desc.folded) {
    count *= 2UL;
  }
  size += count * sizeof(uintptr_t);

  return ge::GRAPH_SUCCESS;
}

static graphStatus OutputInstanceCalcSize(const OpDescPtr &op_desc, const ArgDesc &arg_desc, size_t &size) {
  size_t count = 1UL;
  if (arg_desc.ir_idx < 0) {
    count = op_desc->GetOutputsSize();
  }
  if (arg_desc.folded) {
    count *= 2UL;
  }
  size += count * sizeof(uintptr_t);
  return ge::GRAPH_SUCCESS;
}

static graphStatus OutputInstanceParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                                        std::vector<ArgDesc> &arg_descs) {
  GE_ASSERT_NOTNULL(op_desc);
  const size_t valid_output_size = op_desc->GetOutputsSize();
  if (pattern_str == "o_instance*") {
    // 为了方便加载时使用，通配符场景解析后默认展开成多个
    for (size_t i = 0UL; i < valid_output_size; ++i) {
      arg_descs.push_back({type, static_cast<int32_t>(i), false, {0}});
    }
  } else {
    int32_t ir_idx{0};
    GE_ASSERT_TRUE(sscanf_s(pattern_str.c_str(), "o_instance%d", &ir_idx) == kDigitFormatCnt,
                   "Arg format [%s] is invalid", pattern_str.c_str());
    GE_ASSERT(static_cast<size_t>(ir_idx) < valid_output_size, "ir index [%d] is out of range [0, %zu)", ir_idx,
              valid_output_size);
    arg_descs.push_back({type, ir_idx, false, {0}});
  }
  return af::SUCCESS;
}

static graphStatus InputParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                               std::vector<ArgDesc> &arg_descs) {
  GE_ASSERT_NOTNULL(op_desc);
  const auto &ir_inputs = op_desc->GetIrInputs();
  if (pattern_str == "i*") {
    // 为了方便加载时使用，通配符场景解析后默认展开成多个
    for (size_t i = 0UL; i < ir_inputs.size(); ++i) {
      bool folded{false};
      if (ir_inputs[i].second == IrInputType::kIrInputDynamic) {
        folded = true;
      }
      arg_descs.push_back({type, static_cast<int32_t>(i), folded, {0}});
    }
  } else {
    int32_t ir_idx{0};
    bool has_idx{false};
    for (size_t i = 1UL; i < pattern_str.size(); ++i) {
      if (isdigit(pattern_str[i])) {
        ir_idx = ir_idx * kDecimalCarry + static_cast<int32_t>(pattern_str[i]) - kAsciiZero;
        has_idx = true;
      }
    }
    GE_ASSERT(has_idx, "Arg format [%s] is invalid", pattern_str.c_str());
    GE_ASSERT(static_cast<size_t>(ir_idx) < ir_inputs.size(), "ir index [%d] is out of range [0, %zu)", ir_idx,
              ir_inputs.size());

    bool folded{false};
    if (ir_inputs[static_cast<size_t>(ir_idx)].second == IrInputType::kIrInputDynamic &&
        pattern_str[pattern_str.length() - 1UL] != '*') {
      folded = true;
    }
    arg_descs.push_back({type, ir_idx, folded, {0}});
  }

  return ge::GRAPH_SUCCESS;
}

static graphStatus OutputCalcSize(const OpDescPtr &op_desc, const ArgDesc &arg_desc, size_t &size) {
  const auto &ir_outputs = op_desc->GetIrOutputs();
  size_t count = 0UL;
  if (arg_desc.ir_idx >= 0) {
    // 非通配符场景
    GE_ASSERT((static_cast<size_t>(arg_desc.ir_idx) < ir_outputs.size()), "ir_index [%d] is out of range [0, %zu)",
              arg_desc.ir_idx, ir_outputs.size());
    if (ir_outputs[arg_desc.ir_idx].second == IrOutputType::kIrOutputDynamic) {
      if (arg_desc.folded) {
        count++;  // pointer to addr
      }
      int32_t dyn_num = 0;
      for (auto &iter : op_desc->GetAllOutputName()) {
        if (iter.first == ir_outputs[arg_desc.ir_idx].first + std::to_string(dyn_num)) {
          ++count;  // real input_addr
          ++dyn_num;
        }
      }
    } else {
      count++;
    }
  } else {
    // 通配符场景，非动态输入默认展开, 动态输入按照i0形式折叠
    for (const auto &ir_output : ir_outputs) {
      count++;
      if (ir_output.second == IrOutputType::kIrOutputDynamic) {
        int32_t dyn_num = 0;
        for (auto &iter : op_desc->GetAllOutputName()) {
          if (iter.first == ir_output.first + std::to_string(dyn_num)) {
            ++count;  // real input addr
            ++dyn_num;
          }
        }
      }
    }
  }
  size += count * sizeof(uintptr_t);

  return ge::GRAPH_SUCCESS;
}

static graphStatus OutputParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                                std::vector<ArgDesc> &arg_descs) {
  GE_ASSERT_NOTNULL(op_desc);
  const auto &ir_outputs = op_desc->GetIrOutputs();
  if (pattern_str == "o*") {
    // 为了方便加载时使用，通配符场景解析后默认展开成多个
    for (size_t i = 0UL; i < ir_outputs.size(); ++i) {
      bool folded{false};
      if (ir_outputs[i].second == IrOutputType::kIrOutputDynamic) {
        folded = true;
      }
      arg_descs.push_back({type, static_cast<int32_t>(i), folded, {0}});
    }
  } else {
    int32_t ir_idx{0};
    bool has_idx{false};
    for (size_t i = 1UL; i < pattern_str.size(); ++i) {
      if (isdigit(pattern_str[i])) {
        ir_idx = ir_idx * kDecimalCarry + static_cast<int32_t>(pattern_str[i]) - kAsciiZero;
        has_idx = true;
      }
    }
    GE_ASSERT(has_idx, "Op[%s] arg format [%s] is invalid", op_desc->GetNamePtr(), pattern_str.c_str());
    GE_ASSERT(static_cast<size_t>(ir_idx) < ir_outputs.size(), "Op[%s] ir index [%d] is out of range [0, %zu)",
              op_desc->GetNamePtr(), ir_idx, ir_outputs.size());
    bool folded{false};
    if (ir_outputs[static_cast<size_t>(ir_idx)].second == IrOutputType::kIrOutputDynamic &&
        pattern_str[pattern_str.length() - 1UL] != '*') {
      folded = true;
    }
    arg_descs.push_back({type, ir_idx, folded, {0}});
  }
  return ge::GRAPH_SUCCESS;
}

static graphStatus InputDescCalcSize(const OpDescPtr &op_desc, const ArgDesc &arg_desc, size_t &size) {
  const auto &ir_inputs = op_desc->GetIrInputs();
  GE_ASSERT((arg_desc.ir_idx >= 0 && static_cast<size_t>(arg_desc.ir_idx) < ir_inputs.size()),
            "ir_index [%d] is out of range [0, %zu)", arg_desc.ir_idx, ir_inputs.size());
  auto ir_name = ir_inputs[static_cast<size_t>(arg_desc.ir_idx)].first;
  if (arg_desc.folded) {
    size += sizeof(uintptr_t);  // pointer to desc
  }
  size += sizeof(uintptr_t);  // offset to addr
  size_t dyn_num = 0UL;
  for (auto &iter : op_desc->GetAllInputName()) {
    if (iter.first == ir_name + std::to_string(dyn_num)) {
      const auto &input_desc = op_desc->GetInputDesc(iter.second);
      size += sizeof(uintptr_t) * 2UL;  // dims_info + addr
      if (input_desc.GetShape().IsUnknownDimNum()) {
        size += sizeof(uintptr_t) * kMaxDimNum;
      } else if (input_desc.GetShape().IsScalar()) {
        size += sizeof(uintptr_t);
      } else {
        size += sizeof(uintptr_t) * input_desc.GetShape().GetDimNum();
      }
      ++dyn_num;
    }
  }
  return ge::GRAPH_SUCCESS;
}

static graphStatus OutputDescCalcSize(const OpDescPtr &op_desc, const ArgDesc &arg_desc, size_t &size) {
  const auto &ir_outputs = op_desc->GetIrOutputs();
  GE_ASSERT((arg_desc.ir_idx >= 0 && static_cast<size_t>(arg_desc.ir_idx) < ir_outputs.size()),
            "ir_index [%d] is out of range [0, %zu)", arg_desc.ir_idx, ir_outputs.size());
  auto ir_name = ir_outputs[static_cast<size_t>(arg_desc.ir_idx)].first;

  if (arg_desc.folded) {
    size += sizeof(uintptr_t);  // pointer to desc
  }
  size += sizeof(uintptr_t);  // offset to addr
  size_t dyn_num = 0UL;
  for (auto &iter : op_desc->GetAllOutputName()) {
    if (iter.first == ir_name + std::to_string(dyn_num)) {
      const auto &output_desc = op_desc->GetOutputDesc(iter.second);
      size += sizeof(uintptr_t) * 2UL;  // dims_info + addr
      if (output_desc.GetShape().IsUnknownDimNum()) {
        size += sizeof(uintptr_t) * kMaxDimNum;
      } else if (output_desc.GetShape().IsScalar()) {
        size += sizeof(uintptr_t);
      } else {
        size += sizeof(uintptr_t) * output_desc.GetShape().GetDimNum();
      }
      ++dyn_num;
    }
  }
  return ge::GRAPH_SUCCESS;
}

static graphStatus IODescParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                                std::vector<ArgDesc> &arg_descs) {
  (void)op_desc;
  bool folded{true};
  if (pattern_str[pattern_str.length() - 1] == '*') {
    folded = false;
  }
  int32_t ir_idx{0};
  bool has_idx{false};
  for (size_t i = 6UL; i < pattern_str.size(); ++i) {  // start after i_desc/o_desc
    if (isdigit(pattern_str[i])) {
      ir_idx = ir_idx * kDecimalCarry + static_cast<int32_t>(pattern_str[i]) - kAsciiZero;
      has_idx = true;
    }
  }
  GE_ASSERT(has_idx, "Dynamic input/output should have a concrete ir idx.");
  arg_descs.push_back({type, ir_idx, folded, {0}});
  return ge::GRAPH_SUCCESS;
}

static graphStatus HiddenInputParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                                     std::vector<ArgDesc> &arg_descs) {
  (void)op_desc;
  ArgDesc arg = {type, kAmbiguousIrIdx, false, {0}};
  if (sscanf_s(pattern_str.c_str(), "hi.hcom%d*", &arg.ir_idx) == kDigitFormatCnt) {
    *reinterpret_cast<uint32_t *>(arg.reserved) = static_cast<uint32_t>(HiddenInputsType::HCOM);
    arg_descs.emplace_back(arg);
    return ge::GRAPH_SUCCESS;
  }
  if (sscanf_s(pattern_str.c_str(), "hi.tilefwk%d*", &arg.ir_idx) == kDigitFormatCnt) {
    *reinterpret_cast<uint32_t *>(arg.reserved) = static_cast<uint32_t>(HiddenInputsType::TILEFWK);
    arg_descs.emplace_back(arg);
    return ge::GRAPH_SUCCESS;
  }
  GELOGE(ge::GRAPH_FAILED, "Hidden input type [%s] is unsupported.", pattern_str.c_str());
  return ge::GRAPH_FAILED;
}

static void HiddenInputSerializer(std::stringstream &ss, const std::string &pattern, const ArgDesc &arg_desc) {
  if (*reinterpret_cast<const uint32_t *>(arg_desc.reserved) == static_cast<uint32_t>(HiddenInputsType::HCOM)) {
    ss << pattern << ".hcom" << arg_desc.ir_idx << "*";
  }
  if (*reinterpret_cast<const uint32_t *>(arg_desc.reserved) == static_cast<uint32_t>(HiddenInputsType::TILEFWK)) {
    ss << pattern << ".tilefwk" << arg_desc.ir_idx << "*";
  }
  return;
}

static graphStatus TilingContextParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                                       std::vector<ArgDesc> &arg_descs) {
  (void)op_desc;
  static const std::map<std::string, TilingContextSubType> pattern_to_subtype{
      {"tiling_context", TilingContextSubType::TILING_CONTEXT},
      {"tiling_context.tiling_data", TilingContextSubType::TILING_DATA},
      {"tiling_context.tiling_key", TilingContextSubType::TILING_KEY},
      {"tiling_context.block_dim", TilingContextSubType::BLOCK_DIM},
  };
  const auto iter = pattern_to_subtype.find(pattern_str);
  GE_ASSERT_TRUE(iter != pattern_to_subtype.end(), "pattern [%s] is unsupported.", pattern_str.c_str());
  arg_descs.push_back({type, static_cast<int32_t>(iter->second), false, {0}});
  return ge::GRAPH_SUCCESS;
}

static void TilingContextSerializer(std::stringstream &ss, const std::string &pattern, const ArgDesc &arg_desc) {
  ss << pattern;
  const TilingContextSubType sub_type = static_cast<TilingContextSubType>(arg_desc.ir_idx);
  switch (sub_type) {
    case TilingContextSubType::TILING_DATA:
      ss << ".tiling_data";
      break;
    case TilingContextSubType::TILING_KEY:
      ss << ".tiling_key";
      break;
    case TilingContextSubType::BLOCK_DIM:
      ss << ".block_dim";
      break;
    default:
      break;
  }
}

static graphStatus CustomValueParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                                     std::vector<ArgDesc> &arg_descs) {
  (void)op_desc;
  auto width = ArgsFormatWidth::BIT64;
  uint64_t payload;
  if (sscanf_s(pattern_str.c_str(), "#.32b%lu", &payload) == kDigitFormatCnt) {
    width = ArgsFormatWidth::BIT32;
  } else if (sscanf_s(pattern_str.c_str(), "#%lu", &payload) != kDigitFormatCnt) {
    GELOGE(ge::GRAPH_FAILED, "Unsupported custom value format: [%s]", pattern_str.c_str());
    return ge::GRAPH_FAILED;
  }
  ArgDesc arg = {type, static_cast<int32_t>(width), false, {0}};
  *reinterpret_cast<uint64_t *>(arg.reserved) = payload;
  arg_descs.emplace_back(arg);
  return ge::GRAPH_SUCCESS;
}

static void CustomValueSerializer(std::stringstream &ss, const std::string &pattern, const ArgDesc &arg_desc) {
  ss << pattern;
  if (arg_desc.ir_idx == static_cast<int32_t>(ArgsFormatWidth::BIT32)) {
    ss << ".32b";
  }
  ss << *reinterpret_cast<const uint64_t *>(arg_desc.reserved);
}

static graphStatus VariableWidthCalcSize(const OpDescPtr &, const ArgDesc &arg_desc, size_t &size) {
  GE_ASSERT(arg_desc.addr_type == AddrType::PLACEHOLDER || arg_desc.addr_type == AddrType::CUSTOM_VALUE);
  auto width = static_cast<ArgsFormatWidth>(arg_desc.ir_idx);
  switch (width) {
    case ArgsFormatWidth::BIT64:
      size += sizeof(uint64_t);
      break;
    case ArgsFormatWidth::BIT32:
      size += sizeof(uint32_t);
      break;
    default:
      GELOGE(ge::PARAM_INVALID, "Encountering undefined ArgsFormatWidth: %d", static_cast<int32_t>(width));
      return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

struct PatternCmp {
  bool operator()(const std::string &lhs, const std::string &rhs) const {
    if (lhs.size() != rhs.size()) {
      return lhs.size() > rhs.size();
    } else {
      return rhs.compare(lhs) > 0;
    }
  };
};

static const std::map<std::string, PatternHandler, PatternCmp> kSkPatternToHandler = {
    {"i", {InputParser, InputCalcSize, ArrayLikeSerializer, AddrType::INPUT}},
    {"o", {OutputParser, OutputCalcSize, ArrayLikeSerializer, AddrType::OUTPUT}},
    {"ws", {WorkspaceParser, WorkspaceCalcSize, ArrayLikeSerializer, AddrType::WORKSPACE}},
    {"t", {DefaultParser, DefaultCalcSize, DefaultSerializer, AddrType::TILING}},
    {"i_desc", {IODescParser, InputDescCalcSize, ArrayLikeSerializer, AddrType::INPUT_DESC}},
    {"o_desc", {IODescParser, OutputDescCalcSize, ArrayLikeSerializer, AddrType::OUTPUT_DESC}},
    {"ffts_addr", {DefaultParser, DefaultCalcSize, DefaultSerializer, AddrType::FFTS_ADDR}},
    {"overflow_addr", {DefaultParser, DefaultCalcSize, DefaultSerializer, AddrType::OVERFLOW_ADDR}},
    {"t_ffts", {DefaultParser, DefaultCalcSize, FftsTilingSerializer, AddrType::TILING_FFTS}},
    {"hi", {HiddenInputParser, DefaultCalcSize, HiddenInputSerializer, AddrType::HIDDEN_INPUT}},
    {"*op_type", {DefaultParser, DefaultCalcSize, DefaultSerializer, AddrType::OP_TYPE}},
    {"tiling_context", {TilingContextParser, DefaultCalcSize, TilingContextSerializer, AddrType::TILING_CONTEXT}},
    {"", {PlaceholderParser, VariableWidthCalcSize, PlaceholderSerializer, AddrType::PLACEHOLDER}},
    {"#", {CustomValueParser, VariableWidthCalcSize, CustomValueSerializer, AddrType::CUSTOM_VALUE}},
    {"i_instance", {InputInstanceParser, InputInstanceCalcSize, ArrayLikeSerializer, AddrType::INPUT_INSTANCE}},
    {"o_instance", {OutputInstanceParser, OutputInstanceCalcSize, ArrayLikeSerializer, AddrType::OUTPUT_INSTANCE}},
    {"event_addr", {EventAddrParser, DefaultCalcSize, EventAddrSerializer, AddrType::EVENT_ADDR}},
};

static graphStatus ConvertArgDescNormal2Sk(const ArgDesc &normal_arg_desc, int32_t op_id, ArgDesc &sk_arg_desc) {
  GE_ASSERT_TRUE(normal_arg_desc.addr_type != AddrType::CUSTOM_VALUE);
  SkArgDescV2 sk_arg_desc_tmp{};
  sk_arg_desc_tmp.addr_type = AddrType::SUPER_KERNEL_SUB_NODE;
  sk_arg_desc_tmp.ir_idx = op_id;
  if (normal_arg_desc.addr_type != AddrType::HIDDEN_INPUT) {
    sk_arg_desc_tmp.reserved = normal_arg_desc.folded;
  } else {
    sk_arg_desc_tmp.reserved = *reinterpret_cast<const uint32_t *>(normal_arg_desc.reserved);
  }
  sk_arg_desc_tmp.sub_addr_type = normal_arg_desc.addr_type;
  sk_arg_desc_tmp.sub_idx = normal_arg_desc.ir_idx;
  sk_arg_desc = *reinterpret_cast<ArgDesc *>(&sk_arg_desc_tmp);
  return ge::GRAPH_SUCCESS;
}

static graphStatus ConvertArgDescSk2Normal(const ArgDesc &sk_arg_desc, ArgDesc &arg_desc, int32_t &sub_op_id) {
  if (sk_arg_desc.addr_type != AddrType::SUPER_KERNEL_SUB_NODE) {
    arg_desc = sk_arg_desc;
    sub_op_id = INT32_MAX;
    return ge::GRAPH_SUCCESS;
  }
  ArgDesc tmp_arg_desc{};
  const SkArgDesc *sk_arg_desc_tmp = reinterpret_cast<const SkArgDesc *>(&sk_arg_desc);
  sub_op_id = sk_arg_desc_tmp->ir_idx;
  if (sk_arg_desc_tmp->sub_addr_type != AddrType::HIDDEN_INPUT) {
    tmp_arg_desc.addr_type = sk_arg_desc_tmp->sub_addr_type;
    tmp_arg_desc.ir_idx = sk_arg_desc_tmp->sub_idx;
    tmp_arg_desc.folded = sk_arg_desc_tmp->folded;
  } else {
    const SkArgDescV2 *sk_arg_desc_v2_tmp = reinterpret_cast<const SkArgDescV2 *>(&sk_arg_desc);
    tmp_arg_desc.addr_type = sk_arg_desc_v2_tmp->sub_addr_type;
    tmp_arg_desc.ir_idx = sk_arg_desc_v2_tmp->sub_idx;
    tmp_arg_desc.folded = false;
    *reinterpret_cast<uint32_t *>(tmp_arg_desc.reserved) = sk_arg_desc_v2_tmp->reserved;
  }
  arg_desc = tmp_arg_desc;
  return ge::GRAPH_SUCCESS;
}

static graphStatus SknParser(const OpDescPtr &op_desc, const std::string &pattern_str, const AddrType type,
                             std::vector<ArgDesc> &arg_descs) {
  GELOGD("get pattern %s, type %d", pattern_str.c_str(), type);
  const std::string skn_str = "skn";
  GE_ASSERT_TRUE(pattern_str.substr(0, skn_str.length()) == skn_str);
  int32_t sub_idx{0};
  bool has_idx{false};
  size_t i = skn_str.length();
  for (; i < pattern_str.size(); ++i) {  // start after skn
    if (isdigit(pattern_str[i])) {
      sub_idx = sub_idx * kDecimalCarry + static_cast<int32_t>(pattern_str[i]) - kAsciiZero;
      has_idx = true;
    } else {
      break;
    }
  }
  GE_ASSERT(has_idx, "skn should have a concrete sub idx.");
  NodePtr sub_node;
  GE_ASSERT_SUCCESS(FindSkSubNode(op_desc, sub_idx, sub_node));
  std::vector<ArgDesc> sub_arg_descs;
  std::string sub_pattern_str = pattern_str.substr(i);
  GELOGD("get sub_pattern_str %s, type %d", sub_pattern_str.c_str(), type);
  for (const auto &iter : kSkPatternToHandler) {
    if (strncmp(sub_pattern_str.c_str(), iter.first.c_str(), iter.first.length()) == 0) {
      GE_ASSERT_SUCCESS(iter.second.parse(sub_node->GetOpDesc(), sub_pattern_str, iter.second.type, sub_arg_descs));
      break;
    }
  }
  GE_ASSERT_TRUE(sub_arg_descs.size() == 1);
  ArgDesc tmp_sk_desc{};
  GE_ASSERT_GRAPH_SUCCESS(ConvertArgDescNormal2Sk(sub_arg_descs[0], sub_idx, tmp_sk_desc));
  GELOGD("get sub_pattern_str %s, sub_type %d, sub id %d", sub_pattern_str.c_str(), sub_arg_descs[0].addr_type,
         sub_arg_descs[0].ir_idx);

  arg_descs.emplace_back(tmp_sk_desc);
  return ge::GRAPH_SUCCESS;
}

static Status SknSerializer(std::stringstream &ss, const std::string &pattern, const ArgDesc &sk_arg_desc) {
  ArgDesc tmp_arg_desc{};
  int32_t sub_op_id = 0;
  GE_ASSERT_GRAPH_SUCCESS(ConvertArgDescSk2Normal(sk_arg_desc, tmp_arg_desc, sub_op_id));
  ss << pattern << sub_op_id;
  bool founded = false;
  for (const auto &iter : kSkPatternToHandler) {
    if (iter.second.type == tmp_arg_desc.addr_type) {
      iter.second.serialize(ss, iter.first, tmp_arg_desc);
      founded = true;
      break;
    }
  }
  GE_ASSERT_TRUE(founded, "find %d no serialize func", tmp_arg_desc.addr_type);
  return ge::GRAPH_SUCCESS;
}

static graphStatus SknCalcSize(const OpDescPtr &op_desc, const ArgDesc &sk_arg_desc, size_t &size) {
  ArgDesc tmp_arg_desc{};
  int32_t sub_op_id = 0;
  GE_ASSERT_GRAPH_SUCCESS(ConvertArgDescSk2Normal(sk_arg_desc, tmp_arg_desc, sub_op_id));
  NodePtr sub_node;
  GE_ASSERT_SUCCESS(FindSkSubNode(op_desc, sub_op_id, sub_node));

  bool founded = false;
  for (const auto &iter : kSkPatternToHandler) {
    if (iter.second.type == tmp_arg_desc.addr_type) {
      GE_ASSERT_SUCCESS(iter.second.getArgsSize(sub_node->GetOpDesc(), tmp_arg_desc, size));
      founded = true;
      break;
    }
  }
  GE_ASSERT_TRUE(founded, "find %d no serialize func", tmp_arg_desc.addr_type);
  return ge::GRAPH_SUCCESS;
}

static const std::map<std::string, PatternHandler, PatternCmp> kPatternToHandler = {
    {"i", {InputParser, InputCalcSize, ArrayLikeSerializer, AddrType::INPUT}},
    {"o", {OutputParser, OutputCalcSize, ArrayLikeSerializer, AddrType::OUTPUT}},
    {"ws", {WorkspaceParser, WorkspaceCalcSize, ArrayLikeSerializer, AddrType::WORKSPACE}},
    {"t", {DefaultParser, DefaultCalcSize, DefaultSerializer, AddrType::TILING}},
    {"i_desc", {IODescParser, InputDescCalcSize, ArrayLikeSerializer, AddrType::INPUT_DESC}},
    {"o_desc", {IODescParser, OutputDescCalcSize, ArrayLikeSerializer, AddrType::OUTPUT_DESC}},
    {"ffts_addr", {DefaultParser, DefaultCalcSize, DefaultSerializer, AddrType::FFTS_ADDR}},
    {"overflow_addr", {DefaultParser, DefaultCalcSize, DefaultSerializer, AddrType::OVERFLOW_ADDR}},
    {"t_ffts", {DefaultParser, DefaultCalcSize, FftsTilingSerializer, AddrType::TILING_FFTS}},
    {"hi", {HiddenInputParser, DefaultCalcSize, HiddenInputSerializer, AddrType::HIDDEN_INPUT}},
    {"*op_type", {DefaultParser, DefaultCalcSize, DefaultSerializer, AddrType::OP_TYPE}},
    {"tiling_context", {TilingContextParser, DefaultCalcSize, TilingContextSerializer, AddrType::TILING_CONTEXT}},
    {"", {PlaceholderParser, VariableWidthCalcSize, PlaceholderSerializer, AddrType::PLACEHOLDER}},
    {"#", {CustomValueParser, VariableWidthCalcSize, CustomValueSerializer, AddrType::CUSTOM_VALUE}},
    {"i_instance", {InputInstanceParser, InputInstanceCalcSize, ArrayLikeSerializer, AddrType::INPUT_INSTANCE}},
    {"o_instance", {OutputInstanceParser, OutputInstanceCalcSize, ArrayLikeSerializer, AddrType::OUTPUT_INSTANCE}},
    {"event_addr", {EventAddrParser, DefaultCalcSize, EventAddrSerializer, AddrType::EVENT_ADDR}},
    {"skn", {SknParser, SknCalcSize, SknSerializer, AddrType::SUPER_KERNEL_SUB_NODE}},
};

void ArgsFormatDesc::Append(AddrType type, int32_t ir_idx, bool folded) {
  int32_t idx = (type == AddrType::HIDDEN_INPUT ? 0 : ir_idx);
  arg_descs_.push_back({type, idx, folded, {0}});
}

void ArgsFormatDesc::AppendTilingContext(TilingContextSubType sub_type) {
  arg_descs_.push_back({AddrType::TILING_CONTEXT, static_cast<int32_t>(sub_type), false, {0}});
}

void ArgsFormatDesc::AppendPlaceholder(ArgsFormatWidth width) {
  arg_descs_.push_back({AddrType::PLACEHOLDER, static_cast<int32_t>(width), false, {0}});
}

void ArgsFormatDesc::AppendCustomValue(uint64_t value, ArgsFormatWidth width) {
  ArgDesc arg = {AddrType::CUSTOM_VALUE, static_cast<int32_t>(width), false, {0}};
  *reinterpret_cast<uint64_t *>(arg.reserved) = value;
  arg_descs_.push_back(arg);
}

std::string ArgsFormatDesc::ToString() const {
  return Serialize(arg_descs_);
}

graphStatus ArgsFormatDesc::GetArgsSize(const OpDescPtr &op_desc, size_t &args_size) const {
  GE_ASSERT_NOTNULL(op_desc);
  size_t total_size{0UL};
  for (const auto &arg_desc : arg_descs_) {
    for (const auto &iter : kPatternToHandler) {
      if (iter.second.type == arg_desc.addr_type) {
        GE_ASSERT_SUCCESS(iter.second.getArgsSize(op_desc, arg_desc, total_size));
      }
    }
  }
  args_size = total_size;
  return ge::GRAPH_SUCCESS;
}

graphStatus ArgsFormatDesc::GetArgSize(const OpDescPtr &op_desc, const ArgDesc arg_desc, size_t &arg_size) {
  GE_ASSERT_NOTNULL(op_desc);
  for (const auto &iter : kPatternToHandler) {
    if (iter.second.type == arg_desc.addr_type) {
      GE_ASSERT_SUCCESS(iter.second.getArgsSize(op_desc, arg_desc, arg_size));
      return ge::GRAPH_SUCCESS;
    }
  }
  GELOGE(ge::GRAPH_PARAM_INVALID, "arg_desc type [%d] is unsupported.", static_cast<int32_t>(arg_desc.addr_type));
  return ge::GRAPH_PARAM_INVALID;
}

static graphStatus EasyParser(const std::string &pattern_str, const AddrType type, const std::string &prefix_str,
                              std::vector<ArgDesc> &arg_descs) {
  if (prefix_str + "*" == pattern_str) {
    // i*, o*,ws*
    arg_descs.push_back({type, kAmbiguousIrIdx, false, {0U}});
    return ge::GRAPH_SUCCESS;
  }
  // 处理：i0, o0，ws0， i0*, o0*等场景
  int32_t ir_idx{kAmbiguousIrIdx};
  std::string scan_str = prefix_str + "%d";
  GE_ASSERT_TRUE(sscanf_s(pattern_str.c_str(), scan_str.c_str(), &ir_idx) == kDigitFormatCnt,
                 "Args format {%s} is invalid.", pattern_str.c_str());
  const bool folded = pattern_str[pattern_str.length() - 1UL] != '*';
  arg_descs.push_back({type, ir_idx, folded, {0U}});
  return ge::GRAPH_SUCCESS;
}

static graphStatus SingleParser(const std::string &pattern_str, const OpDescPtr &op_desc, bool easy_mode,
                                std::vector<ArgDesc> &arg_descs, bool &parsed) {
  for (const auto &iter : kPatternToHandler) {
    if (strncmp(pattern_str.c_str(), iter.first.c_str(), iter.first.length()) == 0) {
      if (easy_mode && kNeedEasyParserTypes.count(iter.second.type) > 0UL) {
        GE_ASSERT_SUCCESS(EasyParser(pattern_str, iter.second.type, iter.first, arg_descs));
      } else {
        GE_ASSERT_SUCCESS(iter.second.parse(op_desc, pattern_str, iter.second.type, arg_descs));
      }
      parsed = true;
      break;
    }
  }
  return ge::GRAPH_SUCCESS;
}

// Compatible with the old offline model.
graphStatus ArgsFormatDesc::Parse(const OpDescPtr &op_desc, const std::string &str, std::vector<ArgDesc> &arg_descs) {
  return ArgsFormatDesc::Parse(op_desc, str, arg_descs, false);
}

graphStatus ArgsFormatDesc::Parse(const OpDescPtr &op_desc, const std::string &str, std::vector<ArgDesc> &arg_descs,
                                  const bool easy_mode) {
  arg_descs.clear();
  size_t start_idx = 0UL;
  while (start_idx < str.size()) {
    GE_ASSERT(str[start_idx] == '{', "SyntaxError: argsformat should be surrounded by '{','}'");
    size_t end_idx = start_idx + 1UL;
    bool parsed{false};
    while (end_idx < str.size()) {
      if (str[end_idx] == '}') {
        std::string pattern_str = str.substr(start_idx + 1, end_idx - start_idx - 1);
        GE_ASSERT_SUCCESS(SingleParser(pattern_str, op_desc, easy_mode, arg_descs, parsed),
                          "args format [%s] parse failed.", pattern_str.c_str());
        start_idx = end_idx + 1UL;
        break;
      }
      ++end_idx;
    }
    GE_ASSERT(parsed, "SyntaxError: argsformat should be surrounded by '{','}'");
  }
  return ge::GRAPH_SUCCESS;
}

std::string ArgsFormatDesc::Serialize(const std::vector<ArgDesc> &arg_descs) {
  std::stringstream ss;
  for (const auto &arg_desc : arg_descs) {
    for (const auto &iter : kPatternToHandler) {
      if (iter.second.type == arg_desc.addr_type) {
        ss << '{';
        iter.second.serialize(ss, iter.first, arg_desc);
        ss << '}';
      }
    }
  }
  return ss.str();
}

void ArgsFormatDesc::Clear() {
  arg_descs_.clear();
}

graphStatus ArgsFormatDesc::ConvertArgDescSkToNormal(const ArgDesc &sk_arg_desc, ArgDesc &arg_desc,
                                                     int32_t &sub_op_id) {
  return ConvertArgDescSk2Normal(sk_arg_desc, arg_desc, sub_op_id);
}

graphStatus ArgsFormatDesc::ConvertToSuperKernelArgFormat(const NodePtr &sk_node, const NodePtr &sub_node,
                                                          const std::string &sub_node_arg_format,
                                                          std::string &sk_arg_format) {
  GE_ASSERT_NOTNULL(sk_node);
  GE_ASSERT_NOTNULL(sub_node);
  auto sk_opdesc = sk_node->GetOpDesc();
  GE_ASSERT_NOTNULL(sk_opdesc);
  auto sub_op = sub_node->GetOpDesc();
  GE_ASSERT_NOTNULL(sub_op);
  GELOGI("current sub_op %s arg format %s, sk %s arg format %s", sub_node->GetNamePtr(), sub_node_arg_format.c_str(),
         sk_node->GetNamePtr(), sk_arg_format.c_str());

  std::vector<ArgDesc> cur_op_arg_descs;
  ArgsFormatDesc::Parse(sub_op, sub_node_arg_format, cur_op_arg_descs, false);
  std::vector<ArgDesc> append_sk_arg_descs;
  for (auto &arg_desc : cur_op_arg_descs) {
    ArgDesc tmp_arg_desc{};
    GE_ASSERT_GRAPH_SUCCESS(ConvertArgDescNormal2Sk(arg_desc, sub_op->GetId(), tmp_arg_desc));
    append_sk_arg_descs.emplace_back(tmp_arg_desc);
  }
  sk_arg_format += ArgsFormatDesc::Serialize(append_sk_arg_descs);

  const size_t max_log_string_len = 800U;
  size_t index = 0U;
  while (index < sk_arg_format.length()) {
    GELOGI("%s", sk_arg_format.substr(index, max_log_string_len).c_str());
    index += max_log_string_len;
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace af
