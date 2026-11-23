/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascend_graph_code_dumper.h"
#include "ascendc_ir/utils/asc_graph_utils.h"

namespace af {
namespace ascir {
namespace {

static const std::map<ge::DataType, std::string> ge_dtype_2_python_type = {
    {ge::DT_FLOAT, "ascir.dtypes.float32"},
    {ge::DT_FLOAT16, "ascir.dtypes.float16"},
    {ge::DT_INT8, "ascir.dtypes.int8"},
    {ge::DT_INT32, "ascir.dtypes.int32"},
    {ge::DT_UINT8, "ascir.dtypes.uint8"},
    {ge::DT_INT16, "ascir.dtypes.int16"},
    {ge::DT_UINT16, "ascir.dtypes.uint16"},
    {ge::DT_UINT32, "ascir.dtypes.uint32"},
    {ge::DT_INT64, "ascir.dtypes.int64"},
    {ge::DT_UINT64, "ascir.dtypes.uint64"},
    {ge::DT_DOUBLE, "ascir.dtypes.double"},
    {ge::DT_BOOL, "ascir.dtypes.bool"},
    {ge::DT_STRING, "ascir.dtypes.string"},
    {ge::DT_DUAL_SUB_INT8, "ascir.dtypes.dual_sub_int8"},
    {ge::DT_DUAL_SUB_UINT8, "ascir.dtypes.dual_sub_uint8"},
    {ge::DT_COMPLEX64, "ascir.dtypes.complex64"},
    {ge::DT_COMPLEX128, "ascir.dtypes.complex128"},
    {ge::DT_QINT8, "ascir.dtypes.qint8"},
    {ge::DT_QINT16, "ascir.dtypes.qint16"},
    {ge::DT_QINT32, "ascir.dtypes.qint32"},
    {ge::DT_QUINT8, "ascir.dtypes.quint8"},
    {ge::DT_QUINT16, "ascir.dtypes.quint16"},
    {ge::DT_RESOURCE, "ascir.dtypes.resource"},
    {ge::DT_STRING_REF, "ascir.dtypes.string_ref"},
    {ge::DT_DUAL, "ascir.dtypes.dual"},
    {ge::DT_VARIANT, "ascir.dtypes.variant"},
    {ge::DT_BF16, "ascir.dtypes.bf16"},
    {ge::DT_UNDEFINED, "ascir.dtypes.undefined"},
    {ge::DT_INT4, "ascir.dtypes.int4"},
    {ge::DT_UINT1, "ascir.dtypes.uint1"},
    {ge::DT_INT2, "ascir.dtypes.int2"},
    {ge::DT_UINT2, "ascir.dtypes.uint2"},
    {ge::DT_COMPLEX32, "ascir.dtypes.complex32"},
    {ge::DT_HIFLOAT8, "ascir.dtypes.hifloat8"},
    {ge::DT_FLOAT8_E5M2, "ascir.dtypes.float8_e5m2"},
    {ge::DT_FLOAT8_E4M3FN, "ascir.dtypes.float8_e4m3fn"},
    {ge::DT_FLOAT8_E8M0, "ascir.dtypes.float8_e8m0"},
    {ge::DT_FLOAT6_E3M2, "ascir.dtypes.float6_e3m2"},
    {ge::DT_FLOAT6_E2M3, "ascir.dtypes.float6_e2m3"},
    {ge::DT_FLOAT4_E2M1, "ascir.dtypes.float4_e2m1"},
    {ge::DT_FLOAT4_E1M2, "ascir.dtypes.float4_e1m2"},
};

void GeneratePythonHeader(std::ofstream &output_file, const std::string &graph_type) {
  output_file << "# Python code to construct " << graph_type << "\n";
  output_file << "from autofuse.pyautofuse import ascir\n";
  output_file << "from autofuse.pyautofuse import Autofuser, AutofuserOptions\n\n";
}

void GeneratePythonFooter(std::ofstream &output_file) {
  output_file << "fuser = Autofuser(AutofuserOptions())\n";
  output_file << "schedule_results = fuser.schedule(graph)\n";
  output_file << "tiling_def, host_impl, device_impl = fuser.codegen(schedule_results)\n";
}

void FloatHandle(const af::AscNodeAttr *asc_node_attr, const std::string &name, std::string &value_string) {
  float value;
  GE_CHK_BOOL_EXEC(asc_node_attr != nullptr, return, "asc_node_attr is nullptr");
  auto &ir_attr = asc_node_attr->ir_attr;
  GE_CHK_BOOL_EXEC(ir_attr != nullptr, return, "asc_node_attr->ir_attr is nullptr");
  if (ir_attr->GetAttrValue(name, value) == GRAPH_FAILED) {
    return;
  }
  value_string = std::to_string(value);
}

void Int64Handle(const af::AscNodeAttr *asc_node_attr, const std::string &name, std::string &value_string) {
  int64_t value;
  GE_CHK_BOOL_EXEC(asc_node_attr != nullptr, return, "asc_node_attr is nullptr");
  auto &ir_attr = asc_node_attr->ir_attr;
  GE_CHK_BOOL_EXEC(ir_attr != nullptr, return, "asc_node_attr->ir_attr is nullptr");
  if (ir_attr->GetAttrValue(name, value) == GRAPH_FAILED) {
    return;
  }
  value_string = std::to_string(value);
}

void StringHandle(const af::AscNodeAttr *asc_node_attr, const std::string &name, std::string &value_string) {
  std::string value;
  GE_CHK_BOOL_EXEC(asc_node_attr != nullptr, return, "asc_node_attr is nullptr");
  auto &ir_attr = asc_node_attr->ir_attr;
  GE_CHK_BOOL_EXEC(ir_attr != nullptr, return, "asc_node_attr->ir_attr is nullptr");
  if (ir_attr->GetAttrValue(name, value) == GRAPH_FAILED) {
    return;
  }
  value_string = "'" + value + "'";
}

void ExpressionHandle(const af::AscNodeAttr *asc_node_attr, const std::string &name, std::string &value_string) {
  af::Expression value;
  GE_CHK_BOOL_EXEC(asc_node_attr != nullptr, return, "asc_node_attr is nullptr");
  auto &ir_attr = asc_node_attr->ir_attr;
  GE_CHK_BOOL_EXEC(ir_attr != nullptr, return, "asc_node_attr->ir_attr is nullptr");
  if (ir_attr->GetAttrValue(name, value) == GRAPH_FAILED) {
    return;
  }
  value_string = value.Str().get();
}

using handle_ptr = void (*)(const af::AscNodeAttr *asc_node_attr, const std::string &name, std::string &value_string);
const std::unordered_map<std::string, handle_ptr> IrAttrHandleMap = {
    {"float", FloatHandle}, {"int64_t", Int64Handle}, {"std::string", StringHandle}, {"Expression", ExpressionHandle}};

bool IsNodeWithIrInputs(const af::NodePtr &node) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  return !op_desc->GetIrInputs().empty();
}

bool IsNodeWithIrOutputs(const af::NodePtr &node) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  return !op_desc->GetIrOutputs().empty();
}

std::string GetOutputName(const af::NodePtr &src_node, uint32_t idx) {
  if (!IsNodeWithIrOutputs(src_node) && (src_node->GetType() == "AscGraph" || src_node->GetType() == "AscBackend")) {
    return "y[" + std::to_string(idx) + "]";
  }
  const auto &op_desc = src_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &ir_outputs = op_desc->GetIrOutputs();
  std::map<size_t, std::pair<size_t, size_t>> ir_output_2_ranges;
  GE_ASSERT_GRAPH_SUCCESS(af::OpDescUtils::GetIrOutputDescRange(op_desc, ir_output_2_ranges));

  for (const auto &ir_output_2_range : ir_output_2_ranges) {
    if (idx >= ir_output_2_range.second.first &&
        idx < ir_output_2_range.second.first + ir_output_2_range.second.second) {
      GE_ASSERT_TRUE(ir_output_2_range.first < ir_outputs.size());
      if (ir_outputs.at(ir_output_2_range.first).second == af::IrOutputType::kIrOutputDynamic) {
        return ir_outputs.at(ir_output_2_range.first).first + "[" +
               std::to_string(idx - ir_output_2_range.second.first) + "]";
      }
    }
  }

  const auto &idx2name = src_node->GetOpDesc()->GetAllOutputIndexToName();
  auto out_name_iter = idx2name.find(idx);
  GE_ASSERT_TRUE(out_name_iter != idx2name.end());
  return out_name_iter->second;
}

bool GetDynamicOutputCount(const af::OpDescPtr &op_desc, uint32_t &dynamic_output_count) {
  GE_ASSERT_NOTNULL(op_desc);
  const auto &ir_outputs = op_desc->GetIrOutputs();
  if (ir_outputs.size() != 1U || ir_outputs[0U].second != af::IrOutputType::kIrOutputDynamic) {
    return false;
  }

  std::map<size_t, std::pair<size_t, size_t>> ir_output_2_ranges;
  GE_ASSERT_GRAPH_SUCCESS(af::OpDescUtils::GetIrOutputDescRange(op_desc, ir_output_2_ranges));
  const auto range_iter = ir_output_2_ranges.find(0U);
  GE_ASSERT_TRUE(range_iter != ir_output_2_ranges.end());
  dynamic_output_count = static_cast<uint32_t>(range_iter->second.second);
  return true;
}

std::string GetPythonNodeNameByOriginName(const std::string &origin_name,
                                          const std::shared_ptr<NameGenerator> &name_generator) {
  const auto &name_mapping_info = name_generator->GetNameMapping();
  const auto &iter = name_mapping_info.find(origin_name);
  if (iter == name_mapping_info.end()) {
    GELOGW("%s has not been added to name map, the topo may be wrong", origin_name.c_str());
    return "";
  }
  return iter->second;
}

std::string GenerateDataTypeCode(ge::DataType dtype) {
  auto iter = ge_dtype_2_python_type.find(dtype);
  GE_WARN_ASSERT(iter != ge_dtype_2_python_type.end(), "DataType [%s] is not supported by python now",
                 TypeUtils::DataTypeToSerialString(dtype).c_str());
  return iter->second;
}

std::string GenerateAxisCode(const std::vector<int64_t> &axis, const std::vector<af::AxisPtr> &axis_ptrs) {
  std::string axis_code = "[";
  for (size_t i = 0; i < axis.size(); ++i) {
    GE_ASSERT_TRUE(axis[i] >= 0);
    GE_ASSERT_TRUE(static_cast<size_t>(axis[i]) < axis_ptrs.size());
    axis_code += axis_ptrs[axis[i]]->name;
    if (i < axis.size() - 1) {
      axis_code += ", ";
    }
  }
  axis_code += "]";
  return axis_code;
}

std::string GenerateAxisRepeatCode(const std::vector<af::Expression> &repeats) {
  std::string axis_repeat_code = "[";
  for (size_t i = 0; i < repeats.size(); ++i) {
    axis_repeat_code += repeats[i].Str().get();
    if (i < repeats.size() - 1) {
      axis_repeat_code += ", ";
    }
  }
  axis_repeat_code += "]";
  return axis_repeat_code;
}

std::string GenerateAxisStrideCode(const std::vector<af::Expression> &strides) {
  std::string axis_strides_code = "[";
  for (size_t i = 0; i < strides.size(); ++i) {
    axis_strides_code += strides[i].Str().get();
    if (i < strides.size() - 1) {
      axis_strides_code += ", ";
    }
  }
  axis_strides_code += "]";
  return axis_strides_code;
}
}  // namespace

void PythonCodeDumper::GenerateInputCode(const std::string &op_name, const std::string &input_name,
                                         const af::NodePtr &src_node, uint32_t out_idx, std::ostream &output_file) {
  std::string out_name = GetOutputName(src_node, out_idx);
  output_file << op_name << "." << input_name << " = "
              << GetPythonNodeNameByOriginName(src_node->GetName(), name_generator_) << "." << out_name << "\n";
}

Status PythonCodeDumper::GenerateDynamicInputCode(
    const af::Node::Vistor<std::pair<af::NodePtr, af::OutDataAnchorPtr>> &src_nodes, size_t start_index, size_t count,
    const std::string &op_name, const std::string &input_name, std::ostream &output_file) {
  std::string dynamic_inputs_code = "[";
  for (size_t i = start_index; i < start_index + count; ++i) {
    GE_ASSERT_TRUE(i < src_nodes.size());
    const auto &src_node = src_nodes.at(i).first;
    uint32_t out_idx = src_nodes.at(i).second->GetIdx();
    std::string out_name = GetOutputName(src_node, out_idx);
    dynamic_inputs_code += GetPythonNodeNameByOriginName(src_node->GetName(), name_generator_) + "." + out_name;
    if (i < start_index + count - 1) {
      dynamic_inputs_code += ", ";
    }
  }
  dynamic_inputs_code += "]";
  output_file << op_name << "." << input_name << " = " << dynamic_inputs_code << "\n";
  return SUCCESS;
}

void PythonCodeDumper::GenerateHeader(std::ofstream &output_file) {
  GeneratePythonHeader(output_file, "AscGraph");
}

Status PythonCodeDumper::GenerateNodeCode(const af::NodePtr &node, std::ostream &output_file) {
  GE_ASSERT_NOTNULL(node);
  GELOGD("Start to gen node code for %s %s", node->GetNamePtr(), node->GetTypePtr());
  node_name_of_python_ = name_generator_->GenerateUniqueName(*node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  uint32_t dynamic_output_count = 0U;
  const auto has_dynamic_output = GetDynamicOutputCount(op_desc, dynamic_output_count);
  if (node->GetInDataNodesSize() == 0U) {
    output_file << node_name_of_python_ << " = ascir.ops." << node->GetType() << "(" << "\"" << node->GetName() << "\"";
    if (has_dynamic_output) {
      output_file << ", " << dynamic_output_count;
    }
    output_file << ", graph)" << std::endl;
  } else {
    // 有数据输入的节点，不需要graph的入参，通过连边时加入graph中
    output_file << node_name_of_python_ << " = ascir.ops." << node->GetType() << "(" << "\"" << node->GetName() << "\"";
    if (has_dynamic_output) {
      output_file << ", " << dynamic_output_count;
    }
    output_file << ")" << std::endl;
  }
  auto &&node_attr_group = op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  GE_ASSERT_NOTNULL(node_attr_group);
  if (!node_attr_group->sched.axis.empty()) {
    std::string axis_code;
    axis_code.push_back('[');
    for (size_t i = 0U; i < node_attr_group->sched.axis.size(); ++i) {
      auto one_axis = node_attr_group->sched.axis[i];
      GE_ASSERT_TRUE(one_axis >= 0);
      GE_ASSERT_TRUE(static_cast<size_t>(one_axis) < asis_ptrs.size());
      axis_code += asis_ptrs[one_axis]->name;
      if (i < node_attr_group->sched.axis.size() - 1) {
        axis_code += ", ";
      }
    }
    axis_code.push_back(']');
    output_file << node_name_of_python_ << ".attr.sched.axis = " << axis_code << std::endl;
  }
  return SUCCESS;
}

Status PythonCodeDumper::GenerateDataEdgeCode(
    const af::Node::Vistor<std::pair<af::NodePtr, af::OutDataAnchorPtr>> &src_nodes, const af::NodePtr &dst_node,
    std::ostream &output_file) {
  const auto &op_desc = dst_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  if (src_nodes.empty()) {
    GELOGD("[%s:%s] has no input.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return SUCCESS;
  }
  GELOGD("Start to add input for node [%s:%s]", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  const auto &ir_inputs = op_desc->GetIrInputs();
  size_t ir_input_index = 0U;
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  GE_ASSERT_GRAPH_SUCCESS(af::OpDescUtils::GetIrInputRawDescRange(op_desc, ir_input_2_range));
  if (dst_node->GetType() == "Output" && src_nodes.size() > 1) {
    return GenerateDynamicInputCode(src_nodes, 0, src_nodes.size(), node_name_of_python_, ir_inputs[0].first,
                                    output_file);
  }
  for (size_t index = 0; index < src_nodes.size(); ++ir_input_index) {
    const auto &ir_input_2_range_iter = ir_input_2_range.find(ir_input_index);
    GE_ASSERT_TRUE(ir_input_2_range_iter != ir_input_2_range.end());
    GELOGI("ir input:%zu with range [%zu, %zu)", ir_input_index, ir_input_2_range_iter->second.first,
           ir_input_2_range_iter->second.first + ir_input_2_range_iter->second.second);
    GE_ASSERT_TRUE(ir_input_index < ir_inputs.size());
    const auto &ir_input_name_2_input_type = ir_inputs[ir_input_index];
    const auto &ir_input_type = ir_input_name_2_input_type.second;
    const auto &input_name = ir_input_name_2_input_type.first;
    if (ir_input_type == af::IrInputType::kIrInputRequired) {
      GE_ASSERT_EQ(ir_input_2_range_iter->second.second, 1U);
      const auto &src_node = src_nodes.at(index).first;
      uint32_t out_idx = src_nodes.at(index).second->GetIdx();
      GenerateInputCode(node_name_of_python_, input_name, src_node, out_idx, output_file);
      ++index;
    } else if (ir_input_type == af::IrInputType::kIrInputDynamic) {
      GE_ASSERT_EQ(index, ir_input_2_range_iter->second.first);
      GE_ASSERT_TRUE(ir_input_2_range_iter->second.second > 0U);
      GE_ASSERT_SUCCESS(GenerateDynamicInputCode(src_nodes, index, ir_input_2_range_iter->second.second,
                                                 node_name_of_python_, input_name, output_file));
      index += ir_input_2_range_iter->second.second;
    } else {
      GE_ASSERT_TRUE(ir_input_type == af::IrInputType::kIrInputOptional);
      if (ir_input_2_range_iter->second.second == 0U) {
        GELOGI("  optional input[%zu] has no input nodes.", ir_input_index);
      } else {
        GE_ASSERT_EQ(1U, ir_input_2_range_iter->second.second);
        const auto &src_node = src_nodes.at(index).first;
        uint32_t out_idx = src_nodes.at(index).second->GetIdx();
        GenerateInputCode(node_name_of_python_, input_name, src_node, out_idx, output_file);
        ++index;
      }
    }
  }
  return SUCCESS;
}

void PythonCodeDumper::GenerateGraphInstance(const af::AscGraph &asc_graph, std::ostream &output_file) {
  output_file << "graph = ascir.HintGraph(" << "\"" << asc_graph.GetName() << "\"" << ")\n";
  for (const auto &size_var : asc_graph.GetAllSizeVar()) {
    if (!size_var->expr.IsConstExpr()) {
      output_file << size_var->expr.Str().get() << " = graph.create_size(" << "\"" << size_var->expr.Str().get() << "\""
                  << ")\n";
    }
  }
  asis_ptrs = asc_graph.GetAllAxis();
  for (const auto &axis : asis_ptrs) {
    output_file << axis->name << " = " << "" << "graph.create_axis(" << "\"" << axis->name << "\"" << ", "
                << axis->size.Str().get() << ")\n";
  }
  const auto compute_graph = af::AscGraphUtils::GetComputeGraph(asc_graph);
  if (compute_graph == nullptr) {
    return;
  }
  const auto graph_attr = compute_graph->GetAttrsGroup<af::AscGraphAttr>();
  if (graph_attr != nullptr && !graph_attr->sched.axis.empty()) {
    output_file << "graph.sched.axis = " << GenerateAxisCode(graph_attr->sched.axis, asis_ptrs) << "\n";
  }
}

Status PythonCodeDumper::GenerateTensorCode(const af::NodePtr &node, std::ostream &output_file) {
  GELOGD("Start to gen tensor code for %s %s", node->GetNamePtr(), node->GetTypePtr());
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);

  size_t output_index = 0U;
  for (const auto &tensor_desc : op_desc->GetAllOutputsDescPtr()) {
    const auto out_name = GetOutputName(node, static_cast<uint32_t>(output_index++));
    auto dtype = static_cast<ge::DataType>(tensor_desc->GetDataType());
    auto python_dtype = GenerateDataTypeCode(dtype);
    output_file << node_name_of_python_ << "." << out_name << ".dtype = " << python_dtype << std::endl;
    auto tensor_group_attr = tensor_desc->GetAttrsGroup<af::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_group_attr);
    if (tensor_group_attr->axis.empty()) {
      continue;
    }

    const auto &axis_code = GenerateAxisCode(tensor_group_attr->axis, asis_ptrs);
    output_file << node_name_of_python_ << "." << out_name << ".axis = " << axis_code << std::endl;
    const auto &axis_repeat_code = GenerateAxisRepeatCode(tensor_group_attr->repeats);
    output_file << node_name_of_python_ << "." << out_name << ".size = " << axis_repeat_code << std::endl;
    const auto &axis_stride_code = GenerateAxisStrideCode(tensor_group_attr->strides);
    output_file << node_name_of_python_ << "." << out_name << ".strides = " << axis_stride_code << std::endl;
  }
  return SUCCESS;
}

Status PythonCodeDumper::GenerateIrAttrCode(const af::NodePtr &node, std::ostream &output_file) {
  GE_ASSERT_NOTNULL(node);
  GELOGD("Start to gen node code for %s %s", node->GetNamePtr(), node->GetTypePtr());
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto &&node_attr_group = op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  GE_ASSERT_NOTNULL(node_attr_group);
  auto it = types_to_ascir_.find(node->GetType());
  if (it == types_to_ascir_.end()) {
    GELOGD("%s is not registered.", node->GetType().c_str());
    return SUCCESS;
  }
  for (const auto &attr_def : it->second.GetAttrDefs()) {
    if (IrAttrHandleMap.find(attr_def.asc_ir_type) == IrAttrHandleMap.end()) {
      GELOGW("This ir_attr data type [%s] does not implement the dump function", attr_def.asc_ir_type.c_str());
      continue;
    }
    std::string value;
    IrAttrHandleMap.at(attr_def.asc_ir_type)(node_attr_group, attr_def.name, value);
    if (value.empty()) {
      continue;
    }
    output_file << node_name_of_python_ << ".attr.ir_attr." << attr_def.name << " = " << value << std::endl;
  }
  return SUCCESS;
}

void PythonCodeDumper::GenerateFooter(std::ofstream &output_file) {
  GeneratePythonFooter(output_file);
}

Status PythonCodeDumper::DumpAscGraphNode(const af::AscGraph &graph, std::ostream &output_file) {
  GenerateGraphInstance(graph, output_file);
  for (const auto &node : graph.GetAllNodes()) {
    GELOGD("Start to gen code for %s %s", node->GetNamePtr(), node->GetTypePtr());
    GE_ASSERT_SUCCESS(GenerateNodeCode(node, output_file));
    const auto &input_nodes = node->GetInDataNodesAndAnchors();
    GE_ASSERT_SUCCESS(GenerateDataEdgeCode(input_nodes, node, output_file));
    GE_ASSERT_SUCCESS(GenerateTensorCode(node, output_file));
    GE_ASSERT_SUCCESS(GenerateIrAttrCode(node, output_file));
  }
  return SUCCESS;
}

Status PythonCodeDumper::Dump(const af::AscGraph &graph, const std::string &out_file_path) {
  std::ofstream output_file(out_file_path);
  GE_ASSERT_TRUE(output_file.is_open(), "out_file_path %s is invalid", out_file_path.c_str());
  GenerateHeader(output_file);
  GE_ASSERT_SUCCESS(DumpAscGraphNode(graph, output_file));
  GenerateFooter(output_file);
  output_file.close();
  return SUCCESS;
}

void PythonCodeDumperFused::GenerateHeader(std::ofstream &output_file) {
  GeneratePythonHeader(output_file, "ComputeGraph");
}

void PythonCodeDumperFused::GenerateFooter(std::ofstream &output_file) const {
  GeneratePythonFooter(output_file);
}

Status PythonCodeDumperFused::GenerateDataEdgeCodeWithOutIr(
    const af::Node::Vistor<std::pair<af::NodePtr, af::OutDataAnchorPtr>> &src_nodes, const af::NodePtr &dst_node,
    std::ofstream &output_file) {
  const auto &op_desc = dst_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  if (src_nodes.empty()) {
    GELOGD("[%s:%s] has no input.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return SUCCESS;
  }
  GELOGD("Start to add input for node [%s:%s]", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  GE_ASSERT_TRUE(dst_node->GetType() == "AscGraph" || dst_node->GetType() == "AscBackend");

  std::string dynamic_inputs_code = "[";
  for (size_t index = 0; index < src_nodes.size(); ++index) {
    const auto &src_node = src_nodes.at(index).first;
    uint32_t out_idx = src_nodes.at(index).second->GetIdx();
    std::string out_name = GetOutputName(src_node, out_idx);
    dynamic_inputs_code += GetPythonNodeNameByOriginName(src_node->GetName(), name_generator_) + "." + out_name;
    if (index < src_nodes.size() - 1) {
      dynamic_inputs_code += ", ";
    }
  }
  dynamic_inputs_code += "]";
  output_file << node_name_of_python_ << ".x" << " = " << dynamic_inputs_code << "\n";
  return SUCCESS;
}

Status PythonCodeDumperFused::GenerateDataEdgeCode(
    const af::Node::Vistor<std::pair<af::NodePtr, af::OutDataAnchorPtr>> &src_nodes, const af::NodePtr &dst_node,
    std::ofstream &output_file) {
  if (!IsNodeWithIrInputs(dst_node)) {
    GELOGW("%s has no ir inputs information", dst_node->GetName().c_str());
    return GenerateDataEdgeCodeWithOutIr(src_nodes, dst_node, output_file);
  }
  code_dumper_asc_graph_.GenerateDataEdgeCode(src_nodes, dst_node, output_file);
  return SUCCESS;
}

void PythonCodeDumperFused::GenerateGraphInstance(const af::ComputeGraph &compute_graph,
                                                  std::ofstream &output_file) const {
  output_file << "graph = ascir.FusedGraph(" << "\"" << compute_graph.GetName() << "\"" << ")\n";
}

Status PythonCodeDumperFused::DumpAscGraphNode(const af::NodePtr &node, std::ofstream &output_file) {
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  std::string asc_graph_str = "";
  af::AscGraph asc_graph("");
  GE_ASSERT_TRUE(af::AttrUtils::GetStr(op_desc, "ascgraph", asc_graph_str));
  GE_ASSERT_GRAPH_SUCCESS(af::AscGraphUtils::DeserializeFromReadable(asc_graph_str, asc_graph));

  node_name_of_python_ = name_generator_->GenerateUniqueName(*node);
  output_file << "\ndef Get" << node_name_of_python_ << "():\n";
  std::ostringstream asc_graph_out;
  auto asc_graph_node_dump = PythonCodeDumper(name_generator_);
  GE_ASSERT_GRAPH_SUCCESS(asc_graph_node_dump.DumpAscGraphNode(asc_graph, asc_graph_out));
  std::istringstream asc_graph_in(asc_graph_out.str());
  for (std::string line; std::getline(asc_graph_in, line);) {
    output_file << "    " << line << "\n";
  }
  output_file << "    return graph\n";

  output_file << "\n"
              << node_name_of_python_ << " = ascir.ops." << node->GetType() << "('" << node->GetName() << "', Get"
              << node_name_of_python_ << "(), graph)" << std::endl;

  code_dumper_asc_graph_.node_name_of_python_ = node_name_of_python_;
  const auto &input_nodes = node->GetInDataNodesAndAnchors();
  GenerateDataEdgeCode(input_nodes, node, output_file);
  output_file << std::endl;
  return SUCCESS;
}

Status PythonCodeDumperFused::Dump(const af::ComputeGraph &graph, const std::string &out_file_path) {
  std::ofstream output_file(out_file_path);
  GE_ASSERT_TRUE(output_file.is_open(), "out_file_path %s is invalid", out_file_path.c_str());
  GenerateHeader(output_file);
  GenerateGraphInstance(graph, output_file);
  auto nodes = graph.GetAllNodes();
  for (const auto &node : nodes) {
    if (node->GetType() == "AscGraph" || node->GetType() == "AscBackend") {
      GE_ASSERT_SUCCESS(DumpAscGraphNode(node, output_file));
      continue;
    }
    GELOGD("Start to gen code for %s %s", node->GetNamePtr(), node->GetTypePtr());
    GE_ASSERT_SUCCESS(code_dumper_asc_graph_.GenerateNodeCode(node, output_file));
    const auto &input_nodes = node->GetInDataNodesAndAnchors();
    node_name_of_python_ = code_dumper_asc_graph_.node_name_of_python_;
    GE_ASSERT_SUCCESS(GenerateDataEdgeCode(input_nodes, node, output_file));
    GE_ASSERT_SUCCESS(code_dumper_asc_graph_.GenerateTensorCode(node, output_file));
    GE_ASSERT_SUCCESS(code_dumper_asc_graph_.GenerateIrAttrCode(node, output_file));
  }
  GenerateFooter(output_file);
  output_file.close();
  return SUCCESS;
}

}  // namespace ascir
}  // namespace af
