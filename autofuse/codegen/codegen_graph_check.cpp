/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_kernel.h"
#include "common/ge_common/debug/log.h"
#include "ascir_ops.h"
#include "common/platform_context.h"
#include "common_utils.h"
#include "indirect_load_utils.h"

namespace codegen {

static std::string VectorToStr(const std::vector<ge::DataType> &vec) {
  std::string result = "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    std::string dtype_name;
    Tensor::DtypeName(vec[i], dtype_name);
    result += dtype_name;
    if (i < vec.size() - 1) {
      result += ", ";
    }
  }
  result += "]";
  return result;
}

bool ProcessRequiredInput(const af::AscNodePtr &node, size_t index, size_t count,
                          std::vector<ge::DataType> &input_dtypes) {
  GE_ASSERT_EQ(count, 1U);
  GE_ASSERT_TRUE(static_cast<uint32_t>(index) < node->inputs.Size());
  const auto &tensor = node->inputs[index];
  input_dtypes.push_back(tensor.attr.dtype);
  return true;
}

bool ProcessDynamicInput(const af::AscNodePtr &node, size_t index, size_t count,
                         std::vector<ge::DataType> &input_dtypes) {
  std::set<ge::DataType> unique_dtypes;
  for (size_t i = index; i < index + count; ++i) {
    GE_ASSERT_TRUE(static_cast<uint32_t>(i) < node->inputs.Size());
    unique_dtypes.insert(node->inputs[i].attr.dtype);
  }
  GE_ASSERT_TRUE(unique_dtypes.size() == 1U, "%s dynamic_input should have uniform dtypes",
                 node->GetOpDesc()->GetNamePtr());
  input_dtypes.push_back(*unique_dtypes.begin());
  return true;
}

bool CollectInputDtypesForOutput(const ascir::NodeView &node, std::vector<ge::DataType> &input_dtypes) {
  std::set<ge::DataType> unique_dtypes;
  for (const auto input : node->inputs()) {
    unique_dtypes.insert(input->attr.dtype);
  }
  GE_ASSERT_TRUE(unique_dtypes.size() == 1U, "%s %s should have uniform dtypes", node->GetNamePtr(),
                 node->GetTypePtr());
  input_dtypes.push_back(*unique_dtypes.begin());
  return true;
}

bool CollectInputDtypesForWorkspace(const ascir::NodeView &node, std::vector<ge::DataType> &input_dtypes) {
  std::set<ge::DataType> unique_dtypes;
  if (node->inputs().size() != 0) {
    for (const auto input : node->inputs()) {
      unique_dtypes.insert(input->attr.dtype);
    }
  } else {
    unique_dtypes.insert(node->outputs()[0]->attr.dtype);
  }

  GE_ASSERT_TRUE(unique_dtypes.size() == 1U, "%s %s should have uniform dtypes", node->GetNamePtr(),
                 node->GetTypePtr());
  input_dtypes.push_back(*unique_dtypes.begin());
  return true;
}

bool CollectInputDtypes(const ascir::NodeView &node, std::vector<ge::DataType> &input_dtypes) {
  if (node->GetType() == af::ascir_op::Output::Type) {
    // Output因为前面做了一个可变ir的操作，即ir是必选输入，但是实际行为支持是动态输入或者必选两种，因此特殊处理一下
    return CollectInputDtypesForOutput(node, input_dtypes);
  }
  if (node->GetType() == af::ascir_op::Workspace::Type) {
    // Workspace连接两张子图时，后一张子图的输入是没有显示指定的，因此输入数据的类型按照输出数据类型特殊处理一下
    return CollectInputDtypesForWorkspace(node, input_dtypes);
  }
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc, "op_desc is nullptr!");

  const auto &ir_inputs = op_desc->GetIrInputs();
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  GE_ASSERT_GRAPH_SUCCESS(af::OpDescUtils::GetIrInputRawDescRange(op_desc, ir_input_2_range),
                          "op %s %s has invalid ir desc", op_desc->GetNamePtr(), op_desc->GetTypePtr());

  size_t index = 0;
  for (size_t ir_input_index = 0; ir_input_index < ir_inputs.size(); ++ir_input_index) {
    const auto &range_iter = ir_input_2_range.find(ir_input_index);
    GE_ASSERT_TRUE(range_iter != ir_input_2_range.end(), "Invalid ir_input_index: %zu", ir_input_index);

    const auto &start_and_count = range_iter->second;
    const auto count = start_and_count.second;
    const auto &ir_input_type = ir_inputs[ir_input_index].second;

    switch (ir_input_type) {
      case af::IrInputType::kIrInputRequired:
        GE_ASSERT_TRUE(ProcessRequiredInput(node, index, count, input_dtypes), "ProcessRequiredInput failed, node = %s",
                       node->GetNamePtr());
        break;
      case af::IrInputType::kIrInputDynamic:
        GE_ASSERT_TRUE(ProcessDynamicInput(node, index, count, input_dtypes), "ProcessDynamicInput failed, node = %s",
                       node->GetNamePtr());
        break;
      default:
        GELOGE(af::FAILED, "%s %s unsupported input type %ld at ir index %zu", op_desc->GetNamePtr(),
               op_desc->GetTypePtr(), static_cast<int64_t>(ir_input_type), ir_input_index);
        return false;
    }
    index += count;
  }
  return true;
}

bool CollectOutputDtypes(const ascir::NodeView &node, std::vector<ge::DataType> &output_dtypes) {
  // 由于目前schedule在某些场景下会丢失Output节点输出tensor的数据类型，这里暂时按照输入tensor的数据类型收集，schedule解决后删除.
  if (node->GetType() == af::ascir_op::Output::Type) {
    output_dtypes.emplace_back(node->inputs()[0]->attr.dtype);
    return true;
  }

  // ArgMaxMultiRPhase1/ArgmaxWithValue/ArgMaxMultiRPhase2WithValue有两个输出（value和index），dtype不同，直接收集所有输出
  if (node->GetType() == "ArgMaxMultiRPhase1" || node->GetType() == "ArgmaxWithValue" ||
      node->GetType() == "ArgMaxMultiRPhase2WithValue" || node->GetType() == af::ascir_op::Frexp::Type) {
    for (auto output : node->outputs()) {
      if (output->attr.dtype == ge::DT_UNDEFINED) {
        return true;
      }
      output_dtypes.push_back(output->attr.dtype);
    }
    return true;
  }

  // 其他节点：检查所有输出的dtype是否一致
  std::set<ge::DataType> unique_dtypes;
  for (auto output : node->outputs()) {
    if (output->attr.dtype == ge::DT_UNDEFINED) {
      return true;
    }
    unique_dtypes.insert(output->attr.dtype);
  }
  GE_ASSERT_TRUE(unique_dtypes.size() == 1U, "%s dynamic_input should have uniform dtypes",
                 node->GetOpDesc()->GetNamePtr());
  output_dtypes.push_back(*unique_dtypes.begin());
  return true;
}

Status IsDataTypeSupported(const ascir::ImplGraph &graph) {
  std::set<string> ignore_node_type = {"Broadcast"};
  for (const auto &node : graph.GetAllNodes()) {
    // 对于动态输入和动态输出的节点，不进行类型检测
    const auto &ir_inputs = node->GetOpDescBarePtr()->GetIrInputs();
    const auto &ir_outputs = node->GetOpDescBarePtr()->GetIrOutputs();
    if (ir_inputs.size() != 0 && ir_inputs[0].second == af::IrInputType::kIrInputDynamic && ir_outputs.size() != 0 &&
        ir_outputs[0].second == af::IrOutputType::kIrOutputDynamic) {
      continue;
    }
    std::vector<ge::DataType> input_dtypes;
    std::vector<ge::DataType> output_dtypes;
    GE_ASSERT_TRUE(CollectInputDtypes(node, input_dtypes), "Collect input dtypes failed, node = %s",
                   node->GetNamePtr());
    GE_ASSERT_TRUE(CollectOutputDtypes(node, output_dtypes), "Collect output dtypes failed, node = %s",
                   node->GetNamePtr());
    // 一些api暂不支持int64输入，但是有一些存量st，因此临时屏蔽这些api的int64类型检测，ascir支持后放开.
    if ((ignore_node_type.find(node->GetType()) != ignore_node_type.end() &&
         std::find(input_dtypes.begin(), input_dtypes.end(), ge::DT_INT64) != input_dtypes.end())) {
      continue;
    }
    std::string npu_arch;
    GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
    if (af::ascir::CommonInferDtype(node->GetType(), input_dtypes, output_dtypes, npu_arch) != af::SUCCESS) {
      GELOGE(af::FAILED, "ASCIR(%s) does not support dtypes(input dtype:%s, output dtype:%s), node:%s",
             node->GetTypePtr(), VectorToStr(input_dtypes).c_str(), VectorToStr(output_dtypes).c_str(),
             node->GetNamePtr());
      return af::FAILED;
    }
  }
  return af::SUCCESS;
}

Status IsRepeatStrideValid(const ascir::ImplGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Scalar" || node->GetType() == "Data" || node->GetType() == "Output" ||
        node->GetType() == "Workspace") {
      continue;
    }
    for (const auto &out : node->outputs()) {
      GE_ASSERT_TRUE(out->attr.axis.size() == out->attr.repeats.size(),
                     "Node(%s) output tensor axis size %d "
                     "does not match repeat size %d, which is invalid.",
                     node->GetNamePtr(), out->attr.axis.size(), out->attr.repeats.size());
      GE_ASSERT_TRUE(out->attr.axis.size() == out->attr.strides.size(),
                     "Node(%s) output tensor axis size %d "
                     "does not match stride size %d, which is invalid.",
                     node->GetNamePtr(), out->attr.axis.size(), out->attr.strides.size());
      GE_ASSERT_TRUE(out->attr.vectorized_axis.size() == out->attr.vectorized_strides.size(),
                     "Node(%s) output tensor "
                     "vectorized_axis size %d does not match vectorized_strides size %d, which is invalid.",
                     node->GetNamePtr(), out->attr.vectorized_axis.size(), out->attr.vectorized_strides.size());
    }
  }
  return af::SUCCESS;
}

Status IsGraphNodeValid(const ascir::ImplGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (ascgen_utils::indirect_load::GetTemplateBehavior(node).skips_api_emit) {
      continue;
    }
    auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
    GE_ASSERT_NOTNULL(impl, "GetAscIrCodegenImpl of node %s[%s] is null", node->GetTypePtr(), node->GetNamePtr());
    GE_ASSERT_TRUE(impl->IsNodeValid(*node), "Node %s[%s] is invalid", node->GetTypePtr(), node->GetNamePtr());
  }
  return af::SUCCESS;
}

Status CheckSingleGraphValidity(const ascir::ImplGraph &graph) {
  GE_ASSERT_SUCCESS(IsDataTypeSupported(graph), "Graph: %s check dtype failed", graph.GetName().c_str());
  // matmul模板不走正常schedule流程，暂不做后续校验。
  if (ascgen_utils::IsCubeType(graph)) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(IsRepeatStrideValid(graph), "Graph: %s check repeat stride failed", graph.GetName().c_str());
  GE_ASSERT_SUCCESS(IsGraphNodeValid(graph), "Graph: %s check graph node failed", graph.GetName().c_str());
  return af::SUCCESS;
}

Status CheckGraphValidity(const ascir::ImplGraph &graph) {
  GE_ASSERT_SUCCESS(CheckSingleGraphValidity(graph), "CheckSingleGraphValidity failed");
  std::vector<af::AscGraph> sub_graphs;
  GE_ASSERT_SUCCESS(graph.GetAllSubGraphs(sub_graphs), "Graph: %s get sub graph failed", graph.GetName().c_str());
  for (auto sub_graph : sub_graphs) {
    GE_ASSERT_SUCCESS(CheckSingleGraphValidity(sub_graph), "SubGraph: %s check validity failed",
                      sub_graph.GetName().c_str());
  }
  return af::SUCCESS;
}
}  // namespace codegen
