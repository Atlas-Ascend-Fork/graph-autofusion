/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ir_definitions_recover.h"
#include <algorithm>
#include <cinttypes>
#include <ostream>
#include <sstream>
#include "graph/operator_factory_af.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_op_types.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/recover_ir_utils.h"
using IrDefinition = af::RecoverIrUtils::IrDefinition;
namespace {
std::string IrAttrNamesToString(const std::vector<std::string> &attr_names) {
  std::ostringstream oss;
  bool first = true;
  for (const auto &attr : attr_names) {
    if (first) {
      first = false;
    } else {
      oss << ", ";
    }
    oss << attr;
  }
  return oss.str();
}

// 将 IrInputType 转换为友好的字符串
std::string IrInputTypeToString(af::IrInputType type) {
  switch (type) {
    case af::kIrInputRequired:
      return "Required";
    case af::kIrInputOptional:
      return "Optional";
    case af::kIrInputDynamic:
      return "Dynamic";
    default:
      return "Unknown(" + std::to_string(static_cast<int>(type)) + ")";
  }
}

// 将 IrOutputType 转换为友好的字符串
std::string IrOutputTypeToString(af::IrOutputType type) {
  switch (type) {
    case af::kIrOutputRequired:
      return "Required";
    case af::kIrOutputDynamic:
      return "Dynamic";
    default:
      return "Unknown(" + std::to_string(static_cast<int>(type)) + ")";
  }
}
template <typename IrType>
std::string IrTypeToString(const IrType &ir_type) {
  return std::to_string(ir_type);
}
template <>
std::string IrTypeToString<af::IrInputType>(const af::IrInputType &ir_type) {
  return IrInputTypeToString(ir_type);
}
template <>
std::string IrTypeToString<af::IrOutputType>(const af::IrOutputType &ir_type) {
  return IrOutputTypeToString(ir_type);
}
// 通用模板函数
template <typename IrDef>
std::string IrDefsToString(const IrDef &ir_defs) {
  std::ostringstream oss;
  bool first = true;
  for (const auto &pair : ir_defs) {
    if (first) {
      first = false;
    } else {
      oss << ", ";
    }
    oss << "[" << pair.first << ", " << IrTypeToString(pair.second) << "]";
  }
  return oss.str();
}

template <typename IrDef, typename IrType>
af::graphStatus AppendIrDefs(const af::OpDescPtr &op_desc, const IrDef &ir_ins, const IrDef &ir_defs,
                             const af::RecoverIrUtils::IrDefAppender<IrType> appender,
                             const std::string &ir_type_name) {
  if (ir_defs.size() < ir_ins.size()) {
    GELOGE(af::FAILED,
           "In the current running version, the number of operator[%s][%s] %s has been reduced, "
           "ir_def.%s size[%zu] is less than ir_%s_in_node size[%zu], ir_def.%s is [%s], "
           "ir_%s_in_node is [%s]",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), ir_type_name.c_str(), ir_type_name.c_str(),
           ir_defs.size(), ir_type_name.c_str(), ir_ins.size(), ir_type_name.c_str(),
           IrDefsToString<IrDef>(ir_defs).c_str(), ir_type_name.c_str(), IrDefsToString<IrDef>(ir_ins).c_str());
    return af::FAILED;
  }
  // 当前运行版本中，算子输入/输出个数在后面增加了，需要添加到node中，或者 ir_ins 为空，全部拷贝到node中
  for (size_t i = ir_ins.size(); i < ir_defs.size(); ++i) {
    appender(op_desc, ir_defs[i].first, ir_defs[i].second);
    GELOGD("Append ir %s:[%s, %s] for node[%s(%s)]", ir_type_name.c_str(), ir_defs[i].first.c_str(),
           IrTypeToString(ir_defs[i].second).c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
  }
  return af::GRAPH_SUCCESS;
}

bool IsRequiredAttr(const af::OpDescPtr &desc, const std::string &attr_name) {
  const auto &required_attrs = desc->GetRequiredAttrWithType();
  return required_attrs.find(attr_name) != required_attrs.end();
}

bool IsInputConnected(const af::OpDescPtr &desc, const std::string &ir_input_name) {
  return desc->MutableInputDesc(ir_input_name) != nullptr;
}

// 验证前N个元素的顺序兼容性（公共函数，不区分前后向兼容，只检查不修改）
template <typename IrDef>
bool ValidateIrOrderCompatibility(const IrDef &node_ir_defs, const IrDef &compatible_ir_defs) {
  const size_t min_size = std::min(node_ir_defs.size(), compatible_ir_defs.size());
  for (size_t i = 0U; i < min_size; ++i) {
    if (node_ir_defs[i] != compatible_ir_defs[i]) {
      GELOGE(af::GRAPH_FAILED,
             "ValidateIrOrderCompatibility failed: index is %zu, node ir defs size is %zu, "
             "compatible ir defs size is %zu",
             i, node_ir_defs.size(), compatible_ir_defs.size());
      return false;
    }
  }
  return true;
}

// 验证输入、输出的顺序兼容性（公共函数，不区分前后向兼容，只检查不修改）
af::graphStatus ValidateIrInputOutputOrderCompatibility(const af::OpDescPtr &desc,
                                                        const af::RecoverIrUtils::InputIrDefs &ir_inputs_in_node,
                                                        const af::RecoverIrUtils::OutputIrDefs &ir_outputs_in_node,
                                                        const IrDefinition &ir_def) {
  // 验证输入顺序兼容性
  GE_ASSERT_TRUE(ValidateIrOrderCompatibility(ir_inputs_in_node, ir_def.inputs),
                 "Compatibility failed: operator[%s][%s] input order or type has changed. "
                 "ir_inputs_in_node are [%s], ir_def.inputs are [%s]",
                 desc->GetName().c_str(), desc->GetType().c_str(),
                 IrDefsToString<af::RecoverIrUtils::InputIrDefs>(ir_inputs_in_node).c_str(),
                 IrDefsToString<af::RecoverIrUtils::InputIrDefs>(ir_def.inputs).c_str());

  // 验证输出顺序兼容性, 直构图IR输出个数如果大于兼容图IR输出个数，则直接判定为不兼容
  GE_ASSERT_TRUE(ir_outputs_in_node.size() <= ir_def.outputs.size(),
                 "Compatibility failed: operator[%s][%s] output size has changed. "
                 "ir_outputs_in_node size is %zu, ir_def.outputs size is %zu",
                 desc->GetName().c_str(), desc->GetType().c_str(), ir_outputs_in_node.size(), ir_def.outputs.size());
  GE_ASSERT_TRUE(ValidateIrOrderCompatibility(ir_outputs_in_node, ir_def.outputs),
                 "Compatibility failed: operator[%s][%s] output order or type has changed. "
                 "ir_outputs_in_node is [%s], ir_def.outputs is [%s]",
                 desc->GetName().c_str(), desc->GetType().c_str(),
                 IrDefsToString<af::RecoverIrUtils::OutputIrDefs>(ir_outputs_in_node).c_str(),
                 IrDefsToString<af::RecoverIrUtils::OutputIrDefs>(ir_def.outputs).c_str());

  return af::GRAPH_SUCCESS;
}

}  // namespace
namespace af {
// 处理前向兼容的输入：检查并删除多余的未使用可选输入
graphStatus RecoverIrUtils::ProcessForwardCompatInputs(const af::OpDescPtr &desc, const InputIrDefs &ir_inputs_in_node,
                                                       const IrDefinition &ir_def) {
  // 检查直构图IR中存在但兼容图IR中不存在的可选输入（从ir_def.inputs.size()开始都是新增的）
  for (size_t i = ir_def.inputs.size(); i < ir_inputs_in_node.size(); ++i) {
    // 保存输入名副本，因为RemoveIrInput会删除字符串，导致引用失效
    const std::string ir_input_name = ir_inputs_in_node[i].first;
    const af::IrInputType ir_input_type = ir_inputs_in_node[i].second;

    // 该输入在兼容图IR中不存在，检查是否为可选输入
    GE_ASSERT_TRUE(ir_input_type == af::kIrInputOptional,
                   "Forward compatibility failed: operator[%s][%s] has required input[%s] "
                   "(type: %s) that does not exist in the compatible IR version. "
                   "This is an incompatible change.",
                   desc->GetName().c_str(), desc->GetType().c_str(), ir_input_name.c_str(),
                   IrInputTypeToString(ir_input_type).c_str());

    // 可选输入，检查是否已连边
    GE_ASSERT_TRUE(!IsInputConnected(desc, ir_input_name),
                   "Forward compatibility failed: operator[%s][%s] uses optional input[%s] "
                   "that does not exist in the compatible IR version. "
                   "The input is connected in the node but not supported by the runtime environment.",
                   desc->GetName().c_str(), desc->GetType().c_str(), ir_input_name.c_str());

    // 未连边的可选输入，直接删除（RecoverIrUtils是OpDesc的友元类，可以访问impl_）
    desc->impl_->MutableIRMeta().RemoveIrInput(ir_input_name);
    GELOGD("Forward compatibility: removed unused optional input[%s] from node[%s(%s)]", ir_input_name.c_str(),
           desc->GetName().c_str(), desc->GetType().c_str());
  }

  return af::GRAPH_SUCCESS;
}

graphStatus RecoverIrUtils::RecoverIrAttrNames(const af::OpDescPtr &desc, IrDefinition &ir_def) {
  const auto &ir_attr_names_in_node = desc->GetIrAttrNames();
  GE_ASSERT_TRUE(ValidateIrOrderCompatibility(ir_attr_names_in_node, ir_def.attr_names),
                 "Compatibility failed: operator[%s][%s] attribute order has changed. "
                 "ir_attr_names_in_node is [%s], ir_def.attr_names is [%s]",
                 desc->GetName().c_str(), desc->GetType().c_str(), IrAttrNamesToString(ir_attr_names_in_node).c_str(),
                 IrAttrNamesToString(ir_def.attr_names).c_str());

  // 根据策略处理
  if (ir_def.strategy == CompatibilityStrategy::kForward) {
    // 前向兼容场景：直构图IR版本 > 兼容图IR版本
    // 检查直构图IR中存在但兼容图IR中不存在的属性（从ir_def.attr_names.size()开始都是新增的）
    for (size_t i = ir_def.attr_names.size(); i < ir_attr_names_in_node.size(); ++i) {
      // 保存属性名副本，因为RemoveIrAttrName会删除字符串，导致引用失效
      const std::string attr_name = ir_attr_names_in_node[i];
      // 可选属性且属性值未配置，直接删除
      // 前端ES构图可以保证可选属性非默认值时才配置到图上
      GE_ASSERT_TRUE(!IsRequiredAttr(desc, attr_name),
                     "Forward compatibility failed: operator[%s][%s] has required attribute[%s] "
                     "that does not exist in the compatible IR version. This is an incompatible change.",
                     desc->GetName().c_str(), desc->GetType().c_str(), attr_name.c_str());
      GE_ASSERT(!desc->HasAttr(attr_name),
                "Forward compatibility failed: operator[%s][%s] has optional attribute[%s] "
                "but is configured as non-default value in the node. This is an incompatible change.",
                desc->GetName().c_str(), desc->GetType().c_str(), attr_name.c_str());
      desc->impl_->MutableIRMeta().RemoveIrAttrName(attr_name);
      GELOGD("Forward compatibility: removed optional attribute[%s] from node[%s(%s)]", attr_name.c_str(),
             desc->GetName().c_str(), desc->GetType().c_str());
    }
    return af::GRAPH_SUCCESS;
  }

  if (ir_def.strategy == CompatibilityStrategy::kBackward) {
    // 向后兼容场景：直构图IR版本 < 兼容图IR版本
    // 当前运行版本中，算子属性在后面增加了，需要拷贝到node中，或者 ir_attr_names_in_node 为空，全部拷贝到node中
    for (size_t i = ir_attr_names_in_node.size(); i < ir_def.attr_names.size(); ++i) {
      desc->AppendIrAttrName(ir_def.attr_names[i]);
      GELOGD("Append ir attr name:%s for desc[%s(%s), is_required_attr:%d]", ir_def.attr_names[i].c_str(),
             desc->GetName().c_str(), desc->GetType().c_str(), ir_def.is_required_attr[i]);
    }
    return af::GRAPH_SUCCESS;
  }

  // kNone: 无差异，直接返回成功
  return af::GRAPH_SUCCESS;
}

void RecoverIrUtils::InitIrDefinitionsIfNeed(const string &op_type, IrDefinition &ir_def) {
  if (!ir_def.inited) {
    auto op = af::OperatorFactory::CreateOperator("temp", op_type.c_str());
    op.BreakConnect();
    auto op_desc = af::OpDescUtils::GetOpDescFromOperator(op);
    if (op_desc == nullptr) {
      GELOGW("Failed to construct operator from type %s", op_type.c_str());
      ir_def.has_ir_definition = false;
      ir_def.inited = true;
      return;
    }
    ir_def.attr_names = op_desc->GetIrAttrNames();
    ir_def.is_required_attr.resize(ir_def.attr_names.size(), 0);
    for (size_t i = 0; i < ir_def.attr_names.size(); ++i) {
      ir_def.is_required_attr[i] = IsRequiredAttr(op_desc, ir_def.attr_names[i]);
    }
    ir_def.inputs = op_desc->GetIrInputs();
    ir_def.outputs = op_desc->GetIrOutputs();
    ir_def.attr_value = af::AttrUtils::GetAllAttrs(op_desc);
    ir_def.has_ir_definition = true;
    ir_def.inited = true;
    ir_def.op_desc = op_desc;
  }
}

graphStatus RecoverIrUtils::RecoverIrAttrDefaultValue(const af::OpDescPtr &desc, const string &op_type,
                                                      IrDefinition &ir_def) {
  const auto node_all_attrs = af::AttrUtils::GetAllAttrs(desc);
  for (const auto &name : ir_def.attr_names) {
    if (node_all_attrs.find(name) != node_all_attrs.cend()) {
      continue;
    }
    const std::map<std::string, af::AnyValue>::const_iterator iter = ir_def.attr_value.find(name);
    if (iter == ir_def.attr_value.cend()) {
      GELOGI(
          "node[%s(%s)] missing attr name[%s], and can not find default value for the attr,"
          " it may be REQUIRED_ATTR.",
          desc->GetName().c_str(), op_type.c_str(), name.c_str());
      continue;
    }
    GELOGD("node[%s(%s)] missing attr name[%s], set default value.", desc->GetName().c_str(), op_type.c_str(),
           name.c_str());
    GE_ASSERT_GRAPH_SUCCESS(desc->AttrHolder::SetAttr(name, iter->second));
  }
  return af::GRAPH_SUCCESS;
}

CompatibilityStrategy RecoverIrUtils::DeriveCompatibilityStrategy(const af::OpDescPtr &desc,
                                                                  const IrDefinition &ir_def) {
  const int64_t attr_diff =
      static_cast<int64_t>(desc->GetIrAttrNames().size()) - static_cast<int64_t>(ir_def.attr_names.size());
  const int64_t input_diff =
      static_cast<int64_t>(desc->GetIrInputs().size()) - static_cast<int64_t>(ir_def.inputs.size());
  if ((attr_diff > 0) && (input_diff > 0)) {
    return CompatibilityStrategy::kForward;
  }
  if ((attr_diff < 0) && (input_diff < 0)) {
    return CompatibilityStrategy::kBackward;
  }
  // 方向不一致 -> 推导失败
  if (((attr_diff > 0) && (input_diff < 0)) || ((attr_diff < 0) && (input_diff > 0))) {
    GELOGE(af::GRAPH_FAILED,
           "Compatibility strategy derivation failed: operator[%s][%s] has inconsistent compatibility direction. "
           "Node has %zu attributes and %zu inputs, compatible IR has %zu attributes and %zu inputs. "
           "Attributes suggest %s compatibility, but inputs suggest %s compatibility. This is an incompatible change.",
           desc->GetName().c_str(), desc->GetType().c_str(), desc->GetIrAttrNames().size(), desc->GetIrInputs().size(),
           ir_def.attr_names.size(), ir_def.inputs.size(), (attr_diff > 0 ? "forward" : "backward"),
           (input_diff > 0 ? "forward" : "backward"));
    return CompatibilityStrategy::kFailed;
  }
  // 都相等 -> 无差异
  if ((attr_diff == 0) && (input_diff == 0)) {
    return CompatibilityStrategy::kNone;
  }
  // 一个为0，另一个不为0 -> 根据非零方向判断
  return ((attr_diff > 0) || (input_diff > 0)) ? CompatibilityStrategy::kForward : CompatibilityStrategy::kBackward;
}

graphStatus RecoverIrUtils::RecoverOpDescIrDefinition(const af::OpDescPtr &desc, const string &op_type,
                                                      IrDefinition &ir_def) {
  if ((desc->GetType() == af::NETOUTPUT) || af::OpTypeUtils::IsDataNode(desc->GetType())) {
    return af::GRAPH_SUCCESS;
  }
  InitIrDefinitionsIfNeed(op_type, ir_def);

  if (!ir_def.has_ir_definition) {
    GELOGI("Op type:%s has no registered IR, maybe no need to recover.", op_type.c_str());
    return af::GRAPH_SUCCESS;
  }

  ir_def.strategy = DeriveCompatibilityStrategy(desc, ir_def);
  if (ir_def.strategy == CompatibilityStrategy::kFailed) {
    return af::GRAPH_FAILED;
  }

  // ir_attr_names
  GE_ASSERT_GRAPH_SUCCESS(RecoverIrAttrNames(desc, ir_def), "%s %s recover ir attr names failed.", desc->GetNamePtr(),
                          desc->GetTypePtr());
  // ir input and output
  GE_ASSERT_GRAPH_SUCCESS(RecoverIrInputAndOutput(desc, ir_def), "%s %s recover ir input and output failed.",
                          desc->GetNamePtr(), desc->GetTypePtr());
  // sym store
  desc->ShareDtypeSymbolsFrom(*ir_def.op_desc);
  // attr
  GE_ASSERT_GRAPH_SUCCESS(RecoverIrAttrDefaultValue(desc, op_type, ir_def),
                          "%s %s recover ir attr default value failed.", desc->GetNamePtr(), desc->GetTypePtr());
  return af::GRAPH_SUCCESS;
}

graphStatus RecoverIrUtils::RecoverIrInputAndOutput(const OpDescPtr &desc, IrDefinition &ir_def) {
  const auto &ir_inputs_in_node = desc->GetIrInputs();
  const auto &ir_outputs_in_node = desc->GetIrOutputs();
  GE_ASSERT_GRAPH_SUCCESS(ValidateIrInputOutputOrderCompatibility(desc, ir_inputs_in_node, ir_outputs_in_node, ir_def),
                          "%s %s validate ir input output order compatibility failed.", desc->GetNamePtr(),
                          desc->GetTypePtr());

  // 处理输入：根据策略处理
  if (ir_def.strategy == CompatibilityStrategy::kForward) {
    // 前向兼容场景：直构图IR版本 > 兼容图IR版本
    GE_ASSERT_GRAPH_SUCCESS(RecoverIrUtils::ProcessForwardCompatInputs(desc, ir_inputs_in_node, ir_def));
  } else if (ir_def.strategy == CompatibilityStrategy::kBackward) {
    // 向后兼容场景：直构图IR版本 < 兼容图IR版本
    auto input_appender = [](const af::OpDescPtr &op_desc, const std::string &ir_name,
                             const af::IrInputType ir_type) -> void { op_desc->AppendIrInput(ir_name, ir_type); };
    if (AppendIrDefs<InputIrDefs, af::IrInputType>(desc, desc->GetIrInputs(), ir_def.inputs, input_appender, "input") !=
        af::GRAPH_SUCCESS) {
      GELOGE(af::GRAPH_FAILED, "recover ir inputs failed.");
      return af::GRAPH_FAILED;
    }
  }
  // kNone: 输入无差异，无需处理

  // 处理输出：输出单独处理，总是向后兼容（如果兼容图有更多输出，则添加）
  auto output_appender = [](const af::OpDescPtr &op_desc, const std::string &ir_name,
                            const af::IrOutputType ir_type) -> void { op_desc->AppendIrOutput(ir_name, ir_type); };
  if (AppendIrDefs<OutputIrDefs, af::IrOutputType>(desc, desc->GetIrOutputs(), ir_def.outputs, output_appender,
                                                   "output") != af::GRAPH_SUCCESS) {
    GELOGE(af::GRAPH_FAILED, "recover ir outputs failed.");
    return af::GRAPH_FAILED;
  }

  return af::GRAPH_SUCCESS;
}

static graphStatus RecoverNodeIrDefinitions(const af::NodePtr &node, std::string &op_type, IrDefinition &ir_def) {
  return RecoverIrUtils::RecoverOpDescIrDefinition(node->GetOpDesc(), op_type, ir_def);
}
graphStatus RecoverIrUtils::RecoverOpDescIrDefinition(const af::OpDescPtr &desc, const std::string &op_type) {
  std::string specified_type = op_type.empty() ? desc->GetType() : op_type;
  IrDefinition ir_def;
  ir_def.inited = false;
  return RecoverIrUtils::RecoverOpDescIrDefinition(desc, specified_type, ir_def);
}

af::graphStatus RecoverIrUtils::RecoverIrDefinitions(const af::ComputeGraphPtr &graph,
                                                     const vector<std::string> &attr_names) {
  GELOGD("Start to recover all ir definitions for graph:%s.", graph->GetName().c_str());
  std::map<std::string, IrDefinition> op_type_to_ir_def;
  for (const auto &node : graph->GetAllNodes()) {
    std::string op_type = af::NodeUtils::GetNodeType(node);
    auto &ir_def = op_type_to_ir_def[op_type];
    if (RecoverNodeIrDefinitions(node, op_type, ir_def) != af::GRAPH_SUCCESS) {
      GELOGE(af::GRAPH_FAILED, "[Recover][NodeIrDefinitions] failed, node[%s], type[%s]", node->GetName().c_str(),
             node->GetType().c_str());
      return af::GRAPH_FAILED;
    }
    for (const auto &attr_name : attr_names) {
      af::ComputeGraphPtr graph_ptr = nullptr;
      (void)af::AttrUtils::GetGraph(node->GetOpDesc(), attr_name, graph_ptr);
      if (graph_ptr == nullptr) {
        continue;
      }
      if (RecoverIrDefinitions(graph_ptr) != af::GRAPH_SUCCESS) {
        GELOGE(af::GRAPH_FAILED, "[Recover][IrDefinitions] failed, graph[%s]", graph_ptr->GetName().c_str());
        return af::GRAPH_FAILED;
      }
      (void)af::AttrUtils::SetGraph(node->GetOpDesc(), attr_name, graph_ptr);
      GELOGD("Success to recover definitions for graph:%s with node:%s and attr:%s.", graph->GetName().c_str(),
             node->GetName().c_str(), attr_name.c_str());
    }
  }
  GELOGD("Success to recover all ir definitions for graph:%s.", graph->GetName().c_str());
  return af::GRAPH_SUCCESS;
}

// TODO if all depended is replace, this 2 function will be deleted
af::graphStatus RecoverIrDefinitions(const af::ComputeGraphPtr &graph, const vector<std::string> &attr_names) {
  return RecoverIrUtils::RecoverIrDefinitions(graph, attr_names);
}

af::graphStatus RecoverOpDescIrDefinition(const af::OpDescPtr &desc, const std::string &op_type) {
  return RecoverIrUtils::RecoverOpDescIrDefinition(desc, op_type);
}

bool CheckIrSpec(const af::OpDescPtr &desc) {
  std::string op_type = desc->GetType();
  IrDefinition ir_def;
  ir_def.inited = false;
  RecoverIrUtils::InitIrDefinitionsIfNeed(op_type, ir_def);
  bool ir_input_include_dynamic = false;
  bool ir_output_include_dynamic = false;
  for (auto &ir_def_input : ir_def.inputs) {
    if ((ir_def_input.second == kIrInputDynamic) || (ir_def_input.second == kIrInputOptional)) {
      ir_input_include_dynamic = true;
      break;
    }
  }
  for (auto &ir_def_output : ir_def.outputs) {
    if (ir_def_output.second == kIrOutputDynamic) {
      ir_output_include_dynamic = true;
      break;
    }
  }
  size_t input_num = desc->GetInputsSize();
  size_t output_num = desc->GetOutputsSize();
  GELOGD("Node:%s check input num is %d and ir input num is %d, output num is %d and ir output num is %d",
         desc->GetName().c_str(), input_num, ir_def.inputs.size(), output_num, ir_def.outputs.size());
  if (((input_num != ir_def.inputs.size()) && !ir_input_include_dynamic) ||
      ((output_num != ir_def.outputs.size()) && !ir_output_include_dynamic)) {
    GELOGI("Node:%s inputs/outputs num has changed, compatibility check fail", desc->GetName().c_str());
    return false;
  }
  // attr
  const auto node_all_attrs = af::AttrUtils::GetAllAttrs(desc);
  for (const auto &name : ir_def.attr_names) {
    if (node_all_attrs.find(name) != node_all_attrs.cend()) {
      continue;
    }
    const std::map<std::string, af::AnyValue>::const_iterator iter = ir_def.attr_value.find(name);
    if (iter == ir_def.attr_value.cend()) {
      GELOGI(
          "node[%s(%s)] missing attr name[%s], and can not find default value for the attr,"
          " it may be REQUIRED_ATTR.",
          desc->GetName().c_str(), op_type.c_str(), name.c_str());
      return false;
    }
  }
  return true;
}
}  // namespace af
