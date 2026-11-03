/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "vec_func_call.h"

#include <sstream>
#include "ascir_node_param/ascir_node_param.h"
#include "ascir_ops.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/utils/api_call_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"

namespace {
constexpr size_t kVFMaxLoop = 4U;
constexpr size_t kCommaSpaceLength = 2U;  // 逗号和空格的长度 ", "
}  // namespace

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

namespace {
std::vector<ge::Expression> GetVfOutputDims(const VectorizedAxisLoopMergeStatus &merge_info) {
  std::vector<ge::Expression> output_dims;
  const size_t dim_size = merge_info.merge_repeats.size();
  size_t start_idx = dim_size <= kVFMaxLoop ? 0 : dim_size - kVFMaxLoop;
  for (; start_idx < dim_size; ++start_idx) {
    output_dims.emplace_back(merge_info.merge_repeats[start_idx]);
  }
  return output_dims;
}

void ParamPostProcess(std::stringstream &ss) {
  if (ss.str().length() < kCommaSpaceLength) {
    return;
  }
  std::string str = ss.str();
  if (str.substr(str.length() - kCommaSpaceLength) == ", ") {
    str = str.substr(0, str.length() - kCommaSpaceLength);
    ss.str("");
    ss << str;
  }
  return;
}

void CreateTensorAddr(const std::vector<Tensor> &tensors, const std::vector<std::string> &ub_offsets,
                      const std::vector<Tensor> &tensors_scalar, std::stringstream &ss) {
  size_t count = ub_offsets.size();
  string dtype_name;
  for (size_t i = 0; i < tensors.size(); i++) {
    Tensor::DtypeName(tensors[i].dtype, dtype_name);
    std::string ub_offset = count == 0 ? "0" : ub_offsets[i];
    ss << "(__local_mem__ " << dtype_name << " *)" << tensors[i] << "[" << ub_offset << "].GetPhyAddr(), ";
  }
  for (const auto &scalar : tensors_scalar) {
    ss << scalar << ", ";
  }
}

void GetOuterForOffset(const TPipe &tpipe, const std::vector<std::vector<ascir::SizeExpr>> &strides,
                       std::vector<std::string> &outer_offsets) {
  for (size_t i = 0; i < strides.size(); i++) {
    outer_offsets.emplace_back(CalcInnerOffset(tpipe, strides[i]));
  }
}

void CreateSingleStridesParamsInfo(const Tensor &tensor, const std::vector<ascir::SizeExpr> &strides,
                                   std::stringstream &ss) {
  size_t stride_size = strides.size();
  size_t start_idx = stride_size <= kVFMaxLoop ? 0 : stride_size - kVFMaxLoop;
  for (; start_idx < stride_size; start_idx++) {
    ss << "uint32_t " << tensor << "_stride_" << start_idx << ", ";
  }
}

void CreateSingleStridesInfo(const TPipe &tpipe, const std::vector<ascir::SizeExpr> &strides, std::stringstream &ss) {
  size_t stride_size = strides.size();
  size_t start_idx = stride_size <= kVFMaxLoop ? 0 : stride_size - kVFMaxLoop;
  for (; start_idx < stride_size; start_idx++) {
    auto current_stride = strides[start_idx].Simplify();
    ss << tpipe.tiler.Size(strides[start_idx]) << ", ";
  }
}

void GetOuterForStride(const std::vector<std::vector<ascir::SizeExpr>> &origin_strides,
                       std::vector<std::vector<ascir::SizeExpr>> &target_strides) {
  for (size_t i = 0; i < origin_strides.size(); i++) {
    std::vector<ascir::SizeExpr> strides(origin_strides[i].begin(), origin_strides[i].end() - kVFMaxLoop);
    target_strides.emplace_back(strides);
  }
}

// 生成vf函数体时，用于处理vf函数入参中的main scalar
void CreateVFCallDimAndStrideParmas(const TPipe &tpipe, const std::vector<Tensor> &inputs,
                                    const std::vector<Tensor> &inputs_scalar, const std::vector<Tensor> &outputs,
                                    const VectorizedAxisLoopMergeStatus &merge_info, std::stringstream &ss) {
  string dtype_name;
  for (const auto &output : outputs) {
    Tensor::DtypeName(output.dtype, dtype_name);
    ss << "__local_mem__ " << dtype_name << " *" << output << "_addr" << ", ";
  }

  for (const auto &input : inputs) {
    Tensor::DtypeName(input.dtype, dtype_name);
    ss << "__local_mem__ " << dtype_name << " *" << input << "_addr" << ", ";
  }

  for (const auto &in_scalar : inputs_scalar) {
    Tensor::DtypeName(in_scalar.dtype, dtype_name);
    ss << dtype_name << " " << in_scalar << ", ";
  }

  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    ss << GenCvUbFuseVfFuncDimParams() << ", ";
    ParamPostProcess(ss);
    return;
  }

  size_t dim_size = merge_info.merge_repeats_str.size();
  size_t start_idx = dim_size <= kVFMaxLoop ? 0 : dim_size - kVFMaxLoop;
  for (; start_idx < dim_size; start_idx++) {
    ss << "uint32_t output_dims_" << start_idx << ", ";
  }
  // 生成output stride 参数
  for (size_t i = 0; i < merge_info.outputs_strides.size(); i++) {
    CreateSingleStridesParamsInfo(outputs[i], merge_info.outputs_strides[i], ss);
  }
  // 生成input stride 参数
  for (size_t i = 0; i < merge_info.inputs_strides.size(); i++) {
    CreateSingleStridesParamsInfo(inputs[i], merge_info.inputs_strides[i], ss);
  }
  ParamPostProcess(ss);
}

// 生成函数调用入参
void CreateDimAndStrideParmas(const TPipe &tpipe, const VectorizedAxisLoopMergeStatus &merge_info,
                              std::stringstream &ss) {
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    ss << GenCvUbFuseVfCallDimParams() << ", ";
    ParamPostProcess(ss);
    return;
  }

  // 生成输入dims
  size_t dim_size = merge_info.merge_repeats_str.size();
  size_t start_idx = dim_size <= kVFMaxLoop ? 0 : dim_size - kVFMaxLoop;
  for (; start_idx < dim_size; start_idx++) {
    ss << merge_info.merge_repeats_str[start_idx] << ", ";
  }
  // 生成output stride
  for (size_t i = 0; i < merge_info.outputs_strides.size(); i++) {
    CreateSingleStridesInfo(tpipe, merge_info.outputs_strides[i], ss);
  }
  // 生成input stride
  for (size_t i = 0; i < merge_info.inputs_strides.size(); i++) {
    CreateSingleStridesInfo(tpipe, merge_info.inputs_strides[i], ss);
  }
  ParamPostProcess(ss);
}

void CreateVFCall(const TPipe &tpipe, const std::string &vf_call_name, const std::vector<Tensor> &inputs,
                  const std::vector<Tensor> &outputs, const std::vector<std::string> &input_ub_offsets,
                  const std::vector<std::string> &output_ub_offsets, const std::vector<Tensor> &tensors_scalar,
                  const VectorizedAxisLoopMergeStatus &merge_info, const std::vector<ArangeParam> &arange_params,
                  std::stringstream &ss) {
  ss << vf_call_name << "(";
  CreateTensorAddr(outputs, output_ub_offsets, {}, ss);
  CreateTensorAddr(inputs, input_ub_offsets, tensors_scalar, ss);
  CreateDimAndStrideParmas(tpipe, merge_info, ss);
  for (const auto &param : arange_params) {
    ss << ", " << param.offset << ", " << param.base << ", " << param.step;
    for (const auto &stride : param.logical_strides) {
      ss << ", " << stride;
    }
  }
  ss << ");" << std::endl;
}

void CreateOuterForVFCall(const TPipe &tpipe, const std::string &vf_call_name, const std::vector<Tensor> &inputs,
                          const std::vector<Tensor> &outputs, const std::vector<Tensor> &tensors_scalar,
                          const VectorizedAxisLoopMergeStatus &merge_info,
                          const std::vector<ArangeParam> &arange_params, std::stringstream &ss) {
  std::vector<std::string> repeats(merge_info.merge_repeats_str.begin(),
                                   merge_info.merge_repeats_str.end() - kVFMaxLoop);
  std::vector<std::vector<ascir::SizeExpr>> inputs_strides;
  std::vector<std::vector<ascir::SizeExpr>> outputs_strides;
  std::vector<std::string> inputs_ub_offsets;
  std::vector<std::string> outputs_ub_offsets;
  GetOuterForStride(merge_info.inputs_strides, inputs_strides);
  GetOuterForStride(merge_info.outputs_strides, outputs_strides);
  GetOuterForOffset(tpipe, inputs_strides, inputs_ub_offsets);
  GetOuterForOffset(tpipe, outputs_strides, outputs_ub_offsets);
  std::stringstream ss1;
  CreateVFCall(tpipe, vf_call_name, inputs, outputs, inputs_ub_offsets, outputs_ub_offsets, tensors_scalar, merge_info,
               arange_params, ss1);
  CreateComputeNodeOuterFor(repeats, ss1, ss, 0);
}

void GetVFCallFuncBody(const std::string &params, const std::string &vf_body, std::stringstream &ss) {
  ss << "{" << std::endl;
  ss << params << std::endl;
  ss << vf_body << std::endl;
  ss << "}\n" << std::endl;
}

size_t GeOriginLastAxisPos(const Tiler &tiler, const std::vector<ascir::AxisId> &current_axis_ids,
                           const std::vector<std::vector<ascir::AxisId>> &origin_axis_ids) {
  size_t axis_num = current_axis_ids.size();
  if (origin_axis_ids.size() != axis_num) {
    return axis_num - static_cast<size_t>(1);
  }
  const auto &origin_last_axis = origin_axis_ids.back();
  for (size_t i = 0; i < current_axis_ids.size(); i++) {
    const auto &axis = tiler.GetAxis(current_axis_ids[i]);
    if (axis.type == ascir::Axis::Type::kAxisTypeMerged) {
      std::set<ascir::AxisId> current_ids;
      for (const auto &from : axis.from) {
        current_ids.insert(from);
      }
      size_t match_count = 0;
      for (size_t j = 0; j < origin_last_axis.size(); j++) {
        match_count = (current_ids.count(origin_last_axis[j]) != 0) ? match_count + 1 : match_count;
      }
      if (match_count == origin_last_axis.size()) {
        return i;
      }
      continue;
    }
    if (origin_last_axis.size() == 1 && origin_last_axis.back() == current_axis_ids[i]) {
      return i;
    }
  }
  return axis_num - static_cast<size_t>(1);
}

}  // namespace

void VfCall::SetNodeAxisIds(const std::vector<ascir::AxisId> &origin_axis_ids) {
  if (origin_axis_ids.size() < axis_ids_.size()) {
    return;
  }
  axis_ids_ = origin_axis_ids;
}

Status VfCall::ParseAttr(const ascir::NodeView &node) {
  vf_call_name_ = "VFCall" + node->GetName();
  af::AscGraph sub_graph("sub_graph");
  GE_ASSERT_SUCCESS(af::AscGraphUtils::FromComputeGraph(node->GetOwnerComputeGraph(), sub_graph),
                    "Get sub_graph failed, node:%s", node->GetNamePtr());
  return ParseSubGraph(node, sub_graph);
}

bool VfCall::ShouldInitAsMaskReg(const ascir::NodeView &node, const af::AscTensor *output) const {
  // 条件1：生产者是 Compare 算子
  bool is_compare_output =
      (IsOps<Ge>(node) || IsOps<Eq>(node) || IsOps<Le>(node) || IsOps<Ne>(node) || IsOps<Gt>(node) || IsOps<Lt>(node));
  if (!is_compare_output) {
    return false;
  }

  // 条件2：所有消费者都必须是 Where/Select 的 mask 输入（idx=0）
  const auto &peer_inputs = output->anchor.GetPeerInDataAnchors();
  if (peer_inputs.empty()) {
    return false;
  }
  for (const auto &peer_input : peer_inputs) {
    auto output_node = std::dynamic_pointer_cast<af::AscNode>(peer_input->GetOwnerNode());
    // 只要有一个消费者不是 Where/Select 的 mask 输入，就返回 false
    if (!(IsOps<Where>(output_node) || IsOps<Select>(output_node)) || (peer_input->GetIdx() != 0)) {
      return false;
    }
  }

  // 同时满足才返回 true
  return true;
}

Status VfCall::ParseSubGraph(const ascir::NodeView &vf_node, const ascir::ImplGraph &graph) {
  // 从节点上读取sub_graph_name属性
  const std::string *graph_name = af::AttrUtils::GetStr(vf_node->GetOpDescBarePtr(), "sub_graph_name");
  GE_ASSERT_NOTNULL(graph_name, "Get sub graph name failed, vf node:%s", vf_node->GetNamePtr());
  af::AscGraph sub_graph("vf_sub_graph");
  GE_ASSERT_SUCCESS(graph.FindSubGraph(*graph_name, sub_graph), "Get sub_graph failed, vf node:%s, sub_graph_name:%s",
                    vf_node->GetNamePtr(), graph_name->c_str());
  GELOGI("VF node:%s, sub_graph_name:%s", vf_node->GetNamePtr(), graph_name->c_str());

  uint32_t max_dtype_size = 0;
  VFInputMapping input_mapping;
  for (auto node : sub_graph.GetAllNodes()) {
    // subgraph上的Load api直接使用Tpipe上保存的UB tensor, 因此vf子图上Data节点的输出Tensor不必保存在tensor manager中.
    if (IsOps<Output>(node) || IsOps<Data>(node) || IsOps<Scalar>(node) || IsOps<ScalarData>(node) ||
        IsOps<IndexExpr>(node)) {
      if (IsOps<Scalar>(node)) {
        const auto scalar_tensor_id = node->outputs[0].attr.mem.tensor_id;
        if (scalar_tensor_id != af::kIdNone) {
          subgraph_scalar_ids_.emplace_back(scalar_tensor_id);
        }
      }
      if (IsOps<Data>(node) || IsOps<ScalarData>(node)) {
        int64_t boundary_index = 0;
        int64_t root_input_index = 0;
        GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("index", boundary_index),
                                "Get boundary index failed, node:%s", node->GetNamePtr());
        if (!af::AttrUtils::GetInt(node->GetOpDesc(), "vf_root_input_index", root_input_index)) {
          // vf_root_input_index 由 VectorFuncPartitioner 在建边后写入; 缺失说明该子图未经过分区
          // (如手工构造的UT fixture)。回退用 boundary index, 对"按序连接边界"的子图语义等价。
          GELOGW("Node [%s] lacks vf_root_input_index; fall back to boundary index [%ld].", node->GetNamePtr(),
                 boundary_index);
          root_input_index = boundary_index;
        }
        GE_CHK_BOOL_RET_STATUS(
            root_input_index >= 0 && static_cast<uint64_t>(root_input_index) < vf_node->inputs.Size(), af::FAILED,
            "Invalid root input index:%ld for node:%s", root_input_index, node->GetNamePtr());
        GE_CHK_BOOL_RET_STATUS(input_mapping.emplace(boundary_index, static_cast<uint32_t>(root_input_index)).second,
                               af::FAILED, "Duplicate root input mapping for boundary index:%ld", boundary_index);
      }
      continue;
    }
    // broadcast inline场景,子图内会对轴进行重排序,生成vf代码时,需要找到原始最内层轴所在的位置,用于生成UpdateMask动作
    SetNodeAxisIds(node->attr.sched.axis);
    auto desc = node->GetOpDesc();
    for (auto output : node->outputs()) {
      auto output_index = af::ascir::AscTensorUtils::Index(*output);
      auto tensor_name = node->GetName() + "_" + desc->GetOutputNameByIndex(output_index);

      uint32_t dtype_size = 0;
      std::string dtype_name;
      GE_CHK_STATUS_RET(Tensor::DtypeName(output->attr.dtype, dtype_name), "Codegen get data type:%d failed",
                        static_cast<int32_t>(output->attr.dtype));
      dtype_size = af::GetSizeByDataType(output->attr.dtype);
      if (dtype_size > max_dtype_size || this->max_dtype_size_ == "") {
        this->max_dtype_size_ = dtype_name;
        max_dtype_size = dtype_size;
      }
      auto init_as_mask_reg = ShouldInitAsMaskReg(node, output);
      MicroApiTensor tensor(*output, dtype_name, init_as_mask_reg);
      GE_CHK_STATUS_RET(tensor_mgr_.AddTensor(tensor), "Codegen add tensor failed");
    }
  }
  root_loop_.SetMaxDtypeSize(this->max_dtype_size_);
  // Parse for loop
  return root_loop_.ConstructFromNodes(sub_graph.GetAllNodes(), vf_node, input_mapping);
}

Status VfCall::ParseInputOutputInfo(const TPipe &tpipe) const {
  for (const auto &in : inputs) {
    auto tensor_ptr = tpipe.GetTensor(in->id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "Check[Param] tensor_ptr is nullptr");
    if (tensor_ptr->IsConstScalar()) {
      scalar_inputs_.emplace_back(*tensor_ptr);
    } else {
      ub_inputs_.emplace_back(*tensor_ptr);
    }
  }

  for (const auto &out : outputs) {
    auto tensor_ptr = tpipe.GetTensor(out.id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "Check[Param] tensor_ptr is nullptr");
    ub_outputs_.emplace_back(*tensor_ptr);
  }

  // VF子图内消费的常量Scalar副本不是根图边界输入, 其const tensor需单独补入scalar参数。
  for (const auto scalar_id : subgraph_scalar_ids_) {
    bool already_collected = false;
    for (const auto &scalar : scalar_inputs_) {
      already_collected = already_collected || (scalar.id == scalar_id);
    }
    const auto tensor_ptr = already_collected ? nullptr : tpipe.GetTensor(scalar_id);
    if ((tensor_ptr != nullptr) && tensor_ptr->IsConstScalar()) {
      scalar_inputs_.emplace_back(*tensor_ptr);
    }
  }
  // 处理合轴信息
  return af::SUCCESS;
}

void CollectStrideNames(const std::vector<Tensor> &tensors, const std::vector<std::vector<ge::Expression>> &strides,
                        std::vector<std::string> &all_stride_names, std::vector<ge::Expression> &all_strides) {
  for (size_t j = 0; j < strides.size(); j++) {
    size_t stride_size = strides[j].size();
    size_t start_idx = stride_size <= kVFMaxLoop ? 0 : stride_size - kVFMaxLoop;
    for (; start_idx < stride_size; start_idx++) {
      if (strides[j][start_idx].Simplify() == af::ops::One || strides[j][start_idx].Simplify() == af::ops::Zero) {
        continue;
      }
      std::stringstream tensor_name;
      tensor_name << tensors[j];
      const std::string stride_name = tensor_name.str() + "_stride_" + std::to_string(start_idx);
      all_stride_names.push_back(stride_name);
      all_strides.emplace_back(strides[j][start_idx]);
    }
  }
}

void GenerateStridesEqualCheck(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs,
                               const VectorizedAxisLoopMergeStatus &merge_info,
                               std::vector<ge::Expression> &all_strides, std::stringstream &ss) {
  std::vector<std::string> all_stride_names;
  CollectStrideNames(inputs, merge_info.inputs_strides, all_stride_names, all_strides);
  CollectStrideNames(outputs, merge_info.outputs_strides, all_stride_names, all_strides);

  if (all_stride_names.empty()) {
    ss << "  uint32_t strides_align = 0;\n  bool strides_equal = false;\n";
    return;
  }

  ss << "  bool strides_equal = false;\n";
  ss << "  uint32_t strides_align = static_cast<uint32_t>(" << all_stride_names[0] << ");\n";
  if (all_stride_names.size() == 1) {
    return;
  }
  ss << "  if (";
  for (size_t i = 1; i < all_stride_names.size(); i++) {
    ss << all_stride_names[0] << " == " << all_stride_names[i];
    if (i < all_stride_names.size() - 1) {
      ss << " && ";
    }
  }
  ss << ") {\n";
  ss << "    strides_equal = true;\n";
  ss << "  }\n";

  return;
}

void OptimizeMergeParamsAndLoopSize(const std::vector<std::string> &loop_size_vec,
                                    const VectorizedAxisLoopMergeStatus &merge_info, std::stringstream &ss) {
  if (loop_size_vec.size() < MAX_VF_AXIS_MERGE_SIZE) {
    return;
  }

  const auto &loop_size_0 = loop_size_vec[0];
  const auto &loop_size_1 = loop_size_vec[1];

  // A single merged output axis is already flattened. VFLoop still emits the
  // logical axis as an outer loop around the vector loop, so it must be
  // collapsed even when its only stride is dynamic and cannot establish
  // strides_equal.
  const bool is_single_merged_axis = merge_info.merge_axis_ids.size() == 1;
  if (is_single_merged_axis) {
    ss << "  {\n";
  } else {
    ss << "  if (strides_equal) {\n";
  }
  ss << "    " << loop_size_0 << " = 1;\n";
  if (is_single_merged_axis) {
    ss << "    element_count = static_cast<uint32_t>(output_dims_0);\n";
  } else {
    ss << "    element_count = static_cast<uint32_t>(strides_align * output_dims_0);\n";
  }
  ss << "    " << loop_size_1
     << " = static_cast<uint16_t>((element_count + ELEMENT_PER_VECTOR_LENGTH - 1) / ELEMENT_PER_VECTOR_LENGTH);\n";
  ss << "  }\n";
  return;
}

void GenerateVectorFuncParams(const std::string &max_dtype_size, int32_t stride_depth,
                              const std::vector<std::vector<ascir::AxisId>> &merge_axis_ids, std::stringstream &ss) {
  ss << "  constexpr static uint32_t VECTOR_LENGTH = AscendC::GetVecLen();\n";
  ss << "  constexpr static uint32_t SIZE_OF_DTYPE = sizeof(" << max_dtype_size << ");\n";
  ss << "  constexpr static uint32_t ELEMENT_PER_VECTOR_LENGTH = VECTOR_LENGTH / SIZE_OF_DTYPE;\n";
  if (merge_axis_ids.size() == 0) {
    ss << "  uint32_t element_count = 1;\n";
  } else {
    const auto output_dim_index = std::min<size_t>(static_cast<size_t>(stride_depth), merge_axis_ids.size() - 1UL);
    ss << "  uint32_t element_count = static_cast<uint32_t>(output_dims_" << output_dim_index << ");\n";
  }
  ss << "  uint16_t loop_times = static_cast<uint16_t>((element_count + ELEMENT_PER_VECTOR_LENGTH - 1) / "
        "ELEMENT_PER_VECTOR_LENGTH);\n";
  return;
}

void GenerateTensorDefs(const TPipe &tpipe, const TensorManager &tensor_mgr, const VFLoop &root_loop,
                        std::stringstream &vf_body) {
  std::string reg_tensor_def;
  tensor_mgr.GenerateVreg(reg_tensor_def);
  vf_body << reg_tensor_def;

  std::vector<std::string> mask_reg_temp_tensors;
  root_loop.CollectMaskRegTempTensors(tpipe, tensor_mgr, mask_reg_temp_tensors);
  for (const auto &tensor_name : mask_reg_temp_tensors) {
    vf_body << "AscendC::MicroAPI::MaskReg " << tensor_name << "_temp_mask;" << std::endl;
  }
}

void GenerateVfCallFuncHeader(const TPipe &tpipe, const std::string &vf_call_name, const std::vector<Tensor> &inputs,
                              const std::vector<Tensor> &scalar_inputs, const std::vector<Tensor> &outputs,
                              const VectorizedAxisLoopMergeStatus &merge_info,
                              const std::vector<ArangeParam> &arange_params, std::stringstream &ss) {
  ss << "#if defined(__DAV_C310__) || "
        "(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || "
        "__NPU_ARCH__ == 9202))"
     << std::endl;
  ss << "\ninline __simd_vf__ void " << vf_call_name << "(";
  CreateVFCallDimAndStrideParmas(tpipe, inputs, scalar_inputs, outputs, merge_info, ss);
  if (!arange_params.empty()) {
    for (const auto &arange_param : arange_params) {
      ss << ", int64_t arange_offset_" << arange_param.tensor_id << ", int64_t arange_base_" << arange_param.tensor_id
         << ", int64_t arange_step_" << arange_param.tensor_id;
      for (size_t index = 0; index < arange_param.logical_strides.size(); ++index) {
        ss << ", int64_t arange_stride_" << arange_param.tensor_id << "_" << index;
      }
    }
  }
  ss << ")" << std::endl;
}

void GenerateVfCallLoopParams(const TPipe &tpipe, const std::string &max_dtype_size, int32_t stride_depth,
                              const VectorizedAxisLoopMergeStatus &merge_info, std::stringstream &params) {
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    params << "  constexpr static uint32_t VECTOR_LENGTH = AscendC::GetVecLen();\n";
    params << "  constexpr static uint32_t SIZE_OF_DTYPE = sizeof(" << max_dtype_size << ");\n";
    params << "  constexpr static uint32_t ELEMENT_PER_VECTOR_LENGTH = VECTOR_LENGTH / SIZE_OF_DTYPE;\n";
    params << "  uint32_t element_count = static_cast<uint32_t>(curAivN);\n";
    params << "  uint16_t loop_times = static_cast<uint16_t>((element_count + ELEMENT_PER_VECTOR_LENGTH - 1) / "
              "ELEMENT_PER_VECTOR_LENGTH);\n";
    return;
  }
  GenerateVectorFuncParams(max_dtype_size, stride_depth, merge_info.merge_axis_ids, params);
}

void GenerateLocalMemTensorPtrs(const std::vector<Tensor> &outputs, const std::vector<Tensor> &inputs,
                                std::stringstream &vf_body) {
  std::string dtype_name;
  for (const auto &output : outputs) {
    Tensor::DtypeName(output.dtype, dtype_name);
    vf_body << "    " << "__local_mem__ " << dtype_name << " *" << output << " = "
            << "(__local_mem__ " << dtype_name << " *)" << output << "_addr" << ";\n";
  }

  for (const auto &input : inputs) {
    Tensor::DtypeName(input.dtype, dtype_name);
    vf_body << "    " << "__local_mem__ " << dtype_name << " *" << input << " = "
            << "(__local_mem__ " << dtype_name << " *)" << input << "_addr" << ";\n";
  }
}

void GenerateVfCallBodyPreamble(const TPipe &tpipe, const TensorManager &tensor_mgr, const VFLoop &root_loop,
                                const std::vector<Tensor> &outputs, const std::vector<Tensor> &inputs,
                                const std::string &loop_dtype, std::stringstream &vf_body) {
  GenerateLocalMemTensorPtrs(outputs, inputs, vf_body);
  GenerateTensorDefs(tpipe, tensor_mgr, root_loop, vf_body);
  vf_body << "\nAscendC::MicroAPI::MaskReg preg_main = AscendC::MicroAPI::CreateMask<" << loop_dtype
          << ", AscendC::MicroAPI::MaskPattern::ALL>();\n";
  vf_body << "AscendC::MicroAPI::MaskReg preg_vl1 = AscendC::MicroAPI::CreateMask<" << loop_dtype
          << ", AscendC::MicroAPI::MaskPattern::VL1>();\n";
}

af::Status UpdateVectorFuncNodeParams(const af::AscNodePtr &node, const VectorizedAxisLoopMergeStatus &merge_info,
                                      const std::vector<ge::Expression> &all_strides);

Status GenerateVfCallLoopBody(const TPipe &tpipe, const TensorManager &tensor_mgr, const VFLoop &root_loop,
                              int32_t stride_depth, const VectorizedAxisLoopMergeStatus &merge_info,
                              const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs,
                              const af::AscNodePtr &node, std::stringstream &params, std::stringstream &vf_body) {
  std::string loop_body;
  std::string loop_size;
  int32_t only_loop_max_depth = -1;
  std::vector<std::string> loop_size_vec;
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    GE_CHK_STATUS_RET(root_loop.GenerateCvUbFuse(tpipe, tensor_mgr, loop_body, loop_size),
                      "Generate CV UBFuse VectorFunc body failed");
  } else {
    ArangeOffsetMap arange_offsets;
    std::vector<ArangeParam> arange_params;
    root_loop.CollectArangeParams(tpipe, arange_params);
    for (auto &param : arange_params) {
      const auto *tensor = tensor_mgr.GetTensor(param.tensor_id);
      GE_ASSERT_NOTNULL(tensor);
      for (const auto &stride : tensor->axis_strides_) {
        param.logical_strides.push_back(tpipe.tiler.Size(stride));
      }
    }
    for (const auto &param : arange_params) {
      arange_offsets.emplace(param.tensor_id, "arange_offset_" + std::to_string(param.tensor_id));
    }
    GE_CHK_STATUS_RET(root_loop.Generate(tpipe, tensor_mgr, stride_depth, loop_body, loop_size, only_loop_max_depth,
                                         loop_size_vec, arange_offsets),
                      "Generate VectorFunc body failed");
  }
  const bool is_double_loop = tpipe.cv_fusion_type != ascir::CubeTemplateType::kUBFuse &&
                              stride_depth == MAX_VF_AXIS_MERGE_SIZE - 1 &&
                              only_loop_max_depth == MAX_VF_AXIS_MERGE_SIZE - 1;
  std::vector<ge::Expression> all_strides;
  params << std::endl << loop_size << std::endl;
  if (is_double_loop) {  // 假如stride_depth为1即两层循环，那实际上loop里递归了三次，分别是0、1、2，在2里单独处理call
    GenerateStridesEqualCheck(inputs, outputs, merge_info, all_strides, params);
    OptimizeMergeParamsAndLoopSize(loop_size_vec, merge_info, params);
    GE_ASSERT_SUCCESS(UpdateVectorFuncNodeParams(node, merge_info, all_strides));
  }
  vf_body << std::endl << loop_body << std::endl;
  return af::SUCCESS;
}

af::Status UpdateVectorFuncNodeParams(const af::AscNodePtr &node, const VectorizedAxisLoopMergeStatus &merge_info,
                                      const std::vector<ge::Expression> &all_strides) {
  GE_ASSERT_NOTNULL(node);
  auto params = ascir_param::GetAscirNodeParams(node);
  GE_ASSERT_NOTNULL(params);
  ascir_param::VectorFuncNodeParams vector_func_params;
  vector_func_params.is_double_loop = true;
  vector_func_params.all_strides = all_strides;
  vector_func_params.output_dims = GetVfOutputDims(merge_info);
  params->api_name = node->GetType();
  params->status = ascir_param::ParamBuildStatus::kBuilt;
  params->specific_params = vector_func_params;
  return af::SUCCESS;
}

Status VfCall::GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler, std::stringstream &ss) const {
  std::vector<ArangeParam> arange_params;
  root_loop_.CollectArangeParams(tpipe, arange_params);
  for (auto &param : arange_params) {
    const auto *tensor = tensor_mgr_.GetTensor(param.tensor_id);
    GE_ASSERT_NOTNULL(tensor);
    for (const auto &stride : tensor->axis_strides_) {
      param.logical_strides.push_back(tpipe.tiler.Size(stride));
    }
  }
  GE_ASSERT_TRUE(tpipe.cv_fusion_type != ascir::CubeTemplateType::kUBFuse || arange_params.empty(),
                 "Arange is not supported in CV UBFuse VectorFunc.");
  // 收集输入输出信息，由于GenInnerLoopSizeAndActualSize函数中会刷新tiler对象中的actual_sizes字段,
  // 导致生成函数签名和函数调用时，获取到的size信息不一致，因此生成函数签名和函数调用时均需要调用合轴函数
  GE_ASSERT_SUCCESS(ParseInputOutputInfo(tpipe));

  // 处理合轴信息，生成函数体时，需要知道子图内的合轴状态
  VectorizedAxisLoopMergeStatus merge_info;
  bool status = GenerateVectorizedAxisMergeStatus(this->ub_inputs_, this->ub_outputs_, merge_info, tpipe);
  GE_ASSERT_TRUE(status, "GenerateVectorizedAxisMergeStatus failed");

  GenerateVfCallFuncHeader(tpipe, this->vf_call_name_, this->ub_inputs_, this->scalar_inputs_, this->ub_outputs_,
                           merge_info, arange_params, ss);

  // func body
  std::stringstream params;
  std::stringstream vf_body;
  int32_t stride_depth = GeOriginLastAxisPos(tiler, axis_ids_, merge_info.merge_axis_ids);
  // Loop stride and masks follow the widest compute dtype in the subgraph:
  // narrower outputs (e.g. uint8 masks from int32 compares) still use the
  // compute register's lane count, and per-tensor Load/Store keep their own
  // dtype. Basing the loop on the output dtype shrank EPL below the compute
  // register width and produced per-4th-element corruption (device-verified).
  std::string loop_dtype = this->max_dtype_size_;

  //   constexpr static uint32_t VECTOR_LENGTH = AscendC::GetVecLen();
  //   constexpr static uint32_t SIZE_OF_DTYPE = sizeof(half);
  //   constexpr static uint32_t ELEMENT_PER_VECTOR_LENGTH = VECTOR_LENGTH / SIZE_OF_DTYPE;
  //   uint32_t element_count = static_cast<uint32_t>(output_dims_0);
  //   uint16_t loop_times = static_cast<uint16_t>((element_count + ELEMENT_PER_VECTOR_LENGTH - 1) /
  //   ELEMENT_PER_VECTOR_LENGTH);
  GenerateVfCallLoopParams(tpipe, loop_dtype, stride_depth, merge_info, params);
  root_loop_.SetLoopDtype(loop_dtype);
  GenerateVfCallBodyPreamble(tpipe, tensor_mgr_, root_loop_, this->ub_outputs_, this->ub_inputs_, loop_dtype, vf_body);
  GE_ASSERT_SUCCESS(GenerateVfCallLoopBody(tpipe, tensor_mgr_, root_loop_, stride_depth, merge_info, this->ub_inputs_,
                                           this->ub_outputs_, node, params, vf_body));
  GetVFCallFuncBody(params.str(), vf_body.str(), ss);
  ss << "#endif" << std::endl;

  return af::SUCCESS;
}

Status VfCall::Generate(const TPipe &tpipe, [[maybe_unused]] const std::vector<ascir::AxisId> &current_axis,
                        [[maybe_unused]] const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                        [[maybe_unused]] const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                        std::string &result) const {
  // 合轴信息只会在子图内部体现，不会在原图上体现
  // 对于broadcast inline场景, 需要再合轴后对ApiParams中的repeats和strides做排序
  VectorizedAxisLoopMergeStatus merge_info;
  bool status = GenerateVectorizedAxisMergeStatus(this->ub_inputs_, this->ub_outputs_, merge_info, tpipe);
  GE_ASSERT_TRUE(status, "GenerateVectorizedAxisMergeStatus failed");

  std::stringstream ss;
  std::vector<ArangeParam> arange_params;
  root_loop_.CollectArangeParams(tpipe, arange_params);
  for (auto &param : arange_params) {
    const auto *tensor = tensor_mgr_.GetTensor(param.tensor_id);
    GE_ASSERT_NOTNULL(tensor);
    for (const auto &stride : tensor->axis_strides_) {
      param.logical_strides.push_back(tpipe.tiler.Size(stride));
    }
  }
  size_t loop_num = merge_info.merge_repeats_str.size();
  const size_t outer_loop_num = loop_num > kVFMaxLoop ? loop_num - kVFMaxLoop : 0UL;
  for (auto &param : arange_params) {
    const auto *arange_tensor = tensor_mgr_.GetTensor(param.tensor_id);
    GE_ASSERT_NOTNULL(arange_tensor);
    std::stringstream offset;
    offset << (current_axis.empty()
                   ? "0"
                   : tpipe.tiler.Offset(current_axis, arange_tensor->axis_, arange_tensor->axis_strides_));
    for (size_t i = 0UL; i < outer_loop_num; ++i) {
      std::string stride;
      GE_CHK_STATUS_RET(GetArangeLogicalStride(tpipe, arange_tensor, merge_info.merge_axis_ids[i], stride),
                        "Generate Arange outer-for stride failed");
      offset << " + outer_for_" << i << " * " << stride;
    }
    param.offset = offset.str();
  }
  ss << "#if defined(__DAV_C310__) || "
        "(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || "
        "__NPU_ARCH__ == 9202))"
     << std::endl;
  ss << "AscendC::SetCtrlSpr<60, 60>(0);" << std::endl;
  if (loop_num <= kVFMaxLoop) {
    std::vector<std::string> inputs_ub_offsets = {};
    std::vector<std::string> outputs_ub_offsets = {};
    CreateVFCall(tpipe, this->vf_call_name_, this->ub_inputs_, this->ub_outputs_, inputs_ub_offsets, outputs_ub_offsets,
                 this->scalar_inputs_, merge_info, arange_params, ss);
  } else {
    CreateOuterForVFCall(tpipe, this->vf_call_name_, this->ub_inputs_, this->ub_outputs_, this->scalar_inputs_,
                         merge_info, arange_params, ss);
  }
  ss << "#endif" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

VfCall::~VfCall() {
  root_loop_.Destruct();
}

static ApiCallRegister<VfCall> register_vf_api_call("VfCall");
}  // namespace codegen
