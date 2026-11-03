/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "asc_tensor_utils.h"
#include "codegen_kernel.h"
#include "ascir_ops.h"
#include "micro_api_call/micro_api_call_factory.h"
#include "vf_loop.h"
#include "ascir_ops.h"

using namespace af::ops;
using namespace af::ascir_op;
namespace codegen {

namespace {
bool IsStrideZero(const ascir::SizeExpr &stride) {
  return af::SymbolicUtils::StaticCheckEq(stride.Simplify(), af::sym::kSymbolZero) == af::TriBool::kTrue;
}

std::string GetUbAddrOffset(const TPipe &tpipe, const MicroApiTensor *&reg_tensor, const Tensor *&ub_tensor) {
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    return GenCvUbFuseAddrOffset(tpipe, *ub_tensor);
  }

  std::stringstream offset_expr;
  offset_expr << "0";
  for (size_t i = 0; i < reg_tensor->vectorized_strides_.size(); i++) {
    auto current_stride = reg_tensor->vectorized_strides_[i];
    current_stride = current_stride.Simplify();
    auto current_axis_id = reg_tensor->vectorized_axis_[i];
    const auto &axis = tpipe.tiler.GetAxis(current_axis_id);
    bool current_stride_is_one =
        (af::SymbolicUtils::StaticCheckEq(current_stride, af::sym::kSymbolOne) == af::TriBool::kTrue);
    bool current_stride_is_zero =
        (af::SymbolicUtils::StaticCheckEq(current_stride, af::sym::kSymbolZero) == af::TriBool::kTrue);
    if (current_stride_is_one) {
      offset_expr << " + " << axis.Variable::name << " * " << "ELEMENT_PER_VECTOR_LENGTH";
    } else if (!current_stride_is_zero) {
      offset_expr << " + " << axis.Variable::name << " * " << ub_tensor->name << "_stride_" << i;
    }
  }
  return offset_expr.str();
}

Status CollectArangeLogicalAxisIndices(const TPipe &tpipe, const MicroApiTensor *reg_tensor,
                                       const std::vector<ascir::AxisId> &axis_ids,
                                       std::vector<int64_t> &logical_axis_indices) {
  const auto collect = [&tpipe, reg_tensor](ascir::AxisId axis_id, auto &self, std::vector<int64_t> &indices) -> bool {
    const auto direct = std::find(reg_tensor->axis_.begin(), reg_tensor->axis_.end(), axis_id);
    if (direct != reg_tensor->axis_.end()) {
      indices.push_back(std::distance(reg_tensor->axis_.begin(), direct));
      return true;
    }
    const auto &source_axes = tpipe.tiler.GetAxis(axis_id).from;
    if (source_axes.empty()) {
      return false;
    }
    for (const auto source_id : source_axes) {
      if (!self(source_id, self, indices)) {
        return false;
      }
    }
    return true;
  };
  for (const auto axis_id : axis_ids) {
    GE_ASSERT_TRUE(collect(axis_id, collect, logical_axis_indices),
                   "Arange vectorized axis must map to its logical view.");
  }
  GE_ASSERT_TRUE(std::adjacent_find(logical_axis_indices.begin(), logical_axis_indices.end(),
                                    std::greater_equal<int64_t>()) == logical_axis_indices.end(),
                 "Arange merged axis sources must follow logical axis order.");
  return af::SUCCESS;
}

Status GetArangeBlockOffset(const TPipe &tpipe, const MicroApiTensor *reg_tensor,
                            const std::vector<ascir::AxisId> &current_axis, std::string &result) {
  GE_ASSERT_TRUE(reg_tensor->vectorized_axis_.size() == reg_tensor->vectorized_strides_.size(),
                 "Arange vectorized axis and stride sizes must match.");
  size_t lane_axis_index = reg_tensor->vectorized_strides_.size();
  for (size_t i = reg_tensor->vectorized_strides_.size(); i > 0UL; --i) {
    if (!IsStrideZero(reg_tensor->vectorized_strides_[i - 1UL])) {
      lane_axis_index = i - 1UL;
      break;
    }
  }
  std::stringstream offset;
  bool has_offset = false;
  for (size_t i = 0; i < reg_tensor->vectorized_strides_.size(); ++i) {
    if (std::find(current_axis.begin(), current_axis.end(), reg_tensor->vectorized_axis_[i]) == current_axis.end()) {
      continue;
    }
    const auto &axis = tpipe.tiler.GetAxis(reg_tensor->vectorized_axis_[i]);
    if (has_offset) {
      offset << " + ";
    }
    offset << axis.Variable::name << " * ";
    std::vector<int64_t> logical_axis_indices;
    GE_ASSERT_SUCCESS(
        CollectArangeLogicalAxisIndices(tpipe, reg_tensor, {reg_tensor->vectorized_axis_[i]}, logical_axis_indices));
    if (i == lane_axis_index) {
      offset << "ELEMENT_PER_VECTOR_LENGTH";
      has_offset = true;
      continue;
    }
    GE_ASSERT_TRUE(!logical_axis_indices.empty(), "Arange vectorized axis has no logical source.");
    const auto axis_index = static_cast<size_t>(logical_axis_indices.back());
    GE_ASSERT_TRUE(axis_index < reg_tensor->axis_strides_.size(), "Arange logical axis stride is missing.");
    offset << "arange_stride_" << reg_tensor->id_ << "_" << axis_index;
    has_offset = true;
  }
  result = has_offset ? offset.str() : "0";
  return af::SUCCESS;
}

std::string GetOriginPregName(const std::vector<ascir::AxisId> &current_axis, int32_t depth) {
  if (current_axis.empty() || static_cast<int32_t>(current_axis.size()) < depth) {
    return "preg_main";
  }
  return "preg_" + std::to_string(depth);
}

void GetUbStorePreg(const Tensor *&ub_tensor, std::string &preg_name) {
  bool all_zero = true;
  (void)all_zero;
  for (size_t i = 0; i < ub_tensor->vectorized_strides.size(); i++) {
    bool stride_is_zero = (af::SymbolicUtils::StaticCheckEq(ub_tensor->vectorized_strides[i], af::sym::kSymbolZero) ==
                           af::TriBool::kTrue);
    if (!stride_is_zero) {
      all_zero = false;
      return;
    }
  }
  preg_name = "preg_vl1";
}

Status GenerateMicroApiCall(const TPipe &tpipe, const TensorManager &tensor_mgr, const VFLoopBody &body,
                            const std::string &max_dtype_size, std::string &preg_name, std::stringstream &ss) {
  std::string ub_offset = "";
  if (body.call_->GetMicroApiName() == "Load") {
    const MicroApiTensor *reg_tensor_ptr = tensor_mgr.GetTensor(body.call_->GetOutputTensorIdByIndex(0));
    const Tensor *ub_tensor_ptr = tpipe.GetTensor(body.call_->GetInputTensorIdByIndex(0));
    ub_offset = GetUbAddrOffset(tpipe, reg_tensor_ptr, ub_tensor_ptr);
  } else if (body.call_->GetMicroApiName() == "Store") {
    const Tensor *ub_tensor_ptr = tpipe.GetTensor(body.call_->GetOutputTensorIdByIndex(0));
    const MicroApiTensor *reg_tensor_ptr = tensor_mgr.GetTensor(body.call_->GetOutputTensorIdByIndex(1));
    ub_offset = GetUbAddrOffset(tpipe, reg_tensor_ptr, ub_tensor_ptr);
    GetUbStorePreg(ub_tensor_ptr, preg_name);
  }

  std::string micro_api_call_str;
  CallParam param = {preg_name, ub_offset, max_dtype_size, {}};
  GE_CHK_STATUS_RET(body.call_->Generate(tensor_mgr, tpipe, param, micro_api_call_str),
                    "Generate CV UBFuse MicroAPI call failed");
  ss << micro_api_call_str;
  return af::SUCCESS;
}
}  // namespace

Status GetArangeLogicalStride(const TPipe &tpipe, const MicroApiTensor *reg_tensor,
                              const std::vector<ascir::AxisId> &axis_ids, std::string &result) {
  GE_ASSERT_NOTNULL(reg_tensor);
  std::vector<int64_t> logical_axis_indices;
  GE_ASSERT_SUCCESS(CollectArangeLogicalAxisIndices(tpipe, reg_tensor, axis_ids, logical_axis_indices));
  GE_ASSERT_TRUE(!logical_axis_indices.empty(), "Arange vectorized axis has no logical source.");
  const auto axis_index = static_cast<size_t>(logical_axis_indices.back());
  GE_ASSERT_TRUE(axis_index < reg_tensor->axis_strides_.size(), "Arange logical axis stride is missing.");
  result = tpipe.tiler.Size(reg_tensor->axis_strides_[axis_index]);
  return af::SUCCESS;
}

std::string GenCvUbFuseVfFuncDimParams() {
  return "uint32_t curAivM, uint32_t curAivN, uint32_t curAlignN";
}

std::string GenCvUbFuseVfCallDimParams() {
  return "curAivM, curAivN, curAlignN";
}

std::string GenCvUbFuseRowStride(const TPipe &tpipe, const Tensor &ub_tensor) {
  if (ub_tensor.id == tpipe.cube_output_tensor_id) {
    return "curAlignN";
  }
  std::string dtype_name;
  if (Tensor::DtypeName(ub_tensor.dtype, dtype_name) != af::SUCCESS) {
    return "curAivN";
  }
  return "KernelUtils::BlkAlign<" + dtype_name + ">(curAivN)";
}

std::string GenCvUbFuseAddrOffset(const TPipe &tpipe, const Tensor &ub_tensor) {
  bool enable_m_offset = true;
  bool enable_n_offset = true;
  const auto &strides = ub_tensor.vectorized_strides;
  if (strides.size() >= 2U) {
    enable_m_offset = !IsStrideZero(strides[strides.size() - 2U]);
    enable_n_offset = !IsStrideZero(strides[strides.size() - 1U]);
  } else if (strides.size() == 1U && IsStrideZero(strides[0])) {
    enable_m_offset = false;
    enable_n_offset = false;
  }

  std::stringstream offset_expr;
  offset_expr << "0";
  if (enable_m_offset) {
    offset_expr << " + cv_m * " << GenCvUbFuseRowStride(tpipe, ub_tensor);
  }
  if (enable_n_offset) {
    offset_expr << " + cv_n * ELEMENT_PER_VECTOR_LENGTH";
  }
  return offset_expr.str();
}

VFLoop::VFLoop(const ascir::AxisId axis) {
  axis_id_ = axis;
  parent_ = nullptr;
}

/********************************** 子图图解析阶段调用 ***********************************/
void VFLoop::AddLoop(VFLoop *loop) {
  loop->parent_ = this;
  loop->SetMaxDtypeSize(this->max_dtype_size_);
  VFLoopBody tmp;
  tmp.type_ = LoopType::LOOP;
  tmp.loop_ = loop;
  this->bodys_.emplace_back(tmp);
}

void VFLoop::AddCall(MicroApiCall *call) {
  VFLoopBody tmp;
  tmp.type_ = LoopType::CALL;
  tmp.call_ = call;
  this->bodys_.emplace_back(tmp);
}

/* 图解析阶段调用 */
Status VFLoop::ConstructFromNodes(ascir::NodeViewVisitorConst nodes, const ascir::NodeView &vf_node,
                                  const VFInputMapping &input_mapping) {
  auto current_loop = this;
  std::vector<ascir::AxisId> current_axis;
  std::map<ascir::TensorId, MicroApiCall *> tensor_calls;
  for (auto node : nodes) {
    // Loop enter or create
    GELOGI("node:%s, ComputeUnit:%u\r\n", node->GetNamePtr(), static_cast<uint32_t>(node->attr.api.unit));
    if (node->attr.api.unit != af::ComputeUnit::kUnitNone) {
      auto node_axis = node->attr.sched.axis;
      auto node_loop_axis = node->attr.sched.loop_axis;
      int32_t loop_distance;
      GE_CHK_STATUS_RET(LoopAxisDistance(current_axis, node_axis, node_loop_axis, loop_distance),
                        "Codegen get loop axis distance failed");
      while (loop_distance != 0) {
        if (loop_distance > 0) {
          auto axis = node_axis[current_axis.size()];
          current_axis.push_back(axis);
          current_loop->AddLoop(new VFLoop(axis));
          current_loop = current_loop->bodys_.back().loop_;
        } else {
          current_axis.pop_back();
          current_loop = current_loop->parent_;
        }

        GE_CHK_STATUS_RET(LoopAxisDistance(current_axis, node_axis, node_loop_axis, loop_distance),
                          "Codegen get loop axis distance failed");
      }
    }

    // Add call
    auto call = CreateMicroApiCallObject(node);
    GE_ASSERT_NOTNULL(call, "Create api call object failed, ascir type:%s", node->GetTypePtr());
    current_loop->AddCall(call);
    GE_CHK_STATUS_RET(call->Init(node), "ApiCall Init failed, ascir type:%s", node->GetTypePtr());

    for (auto in : node->inputs()) {
      if (in == nullptr) {
        call->AddInput(af::kIdNone, TensorType::UB_TENSOR);
        continue;
      }

      auto in_call = tensor_calls.find(in->attr.mem.tensor_id);
      GE_CHK_BOOL_RET_STATUS(
          in_call != tensor_calls.end(), af::FAILED,
          "Codegen node[%s] no API call found for input tensor id[%ld], it may be a topological order error",
          node->GetNamePtr(), in->attr.mem.tensor_id);
      // Load和Store api需要使用UB
      // tensor信息，所以这里需要保存vf_node上的tensor_id，在LoadApiCall中通过Tpipe获取对应Tensor.
      auto data_node = std::dynamic_pointer_cast<af::AscNode>(in->anchor.GetOwnerNode());
      GE_CHK_BOOL_RET_STATUS(data_node != nullptr, af::FAILED, "Codegen node[%s] data_node is nullptr",
                             node->GetNamePtr());
      if (IsOps<Scalar>(data_node) || IsOps<IndexExpr>(data_node)) {
        call->AddInput(data_node->outputs[0].attr.mem.tensor_id, TensorType::UB_TENSOR);
      } else if (IsOps<Data>(data_node) || IsOps<ScalarData>(data_node)) {
        int64_t index = 0;
        GE_CHK_BOOL_RET_STATUS(data_node->attr.ir_attr != nullptr, af::FAILED,
                               "Codegen node[%s] data_node->attr.ir_attr is nullptr", node->GetNamePtr());
        GE_CHK_GRAPH_STATUS_RET(data_node->attr.ir_attr->GetAttrValue("index", index),
                                "Get Data index failed, node:%s, index:%ld", data_node->GetNamePtr(), index);
        const auto mapping_iter = input_mapping.find(index);
        GE_CHK_BOOL_RET_STATUS(mapping_iter != input_mapping.end(), af::FAILED,
                               "Codegen data node[%s] has no root input mapping for boundary index:%ld",
                               data_node->GetNamePtr(), index);
        const uint32_t root_input_index = mapping_iter->second;
        GE_CHK_BOOL_RET_STATUS(root_input_index < vf_node->inputs.Size(), af::FAILED,
                               "Codegen data node[%s] has invalid root input index:%u", data_node->GetNamePtr(),
                               root_input_index);
        call->AddInput(vf_node->inputs[root_input_index].attr.mem.tensor_id, TensorType::UB_TENSOR);
      } else {
        call->AddInput(in->attr.mem.tensor_id);  // 默认为REG_TENSOR
      }
    }

    if (IsOps<Output>(node)) {
      continue;
    }

    for (auto out : node->outputs()) {
      tensor_calls.insert({out->attr.mem.tensor_id, call});
      auto peer_anchors = out->anchor.GetPeerInDataAnchors();
      GE_CHK_BOOL_RET_STATUS(!peer_anchors.empty(), af::FAILED, "Codegen node[%s] output has no peer input anchor",
                             node->GetNamePtr());
      auto peer_input = peer_anchors.at(0);
      auto output_node = std::dynamic_pointer_cast<af::AscNode>(peer_input->GetOwnerNode());
      GE_CHK_BOOL_RET_STATUS(output_node != nullptr, af::FAILED, "Codegen node[%s] output_node is nullptr",
                             node->GetNamePtr());
      if (IsOps<Output>(output_node)) {
        int64_t index;
        GE_CHK_BOOL_RET_STATUS(output_node->attr.ir_attr != nullptr, af::FAILED,
                               "Codegen node[%s] output_node->attr.ir_attr is nullptr", node->GetNamePtr());
        GE_CHK_GRAPH_STATUS_RET(output_node->attr.ir_attr->GetAttrValue("index", index),
                                "Get Output index failed, node:%s, index:%ld", output_node->GetNamePtr(), index);
        call->AddOutput(vf_node->outputs[index].attr.mem.tensor_id, TensorType::UB_TENSOR);
      }
      call->AddOutput(out->attr.mem.tensor_id);  // 默认为REG_TENSOR
    }
  }
  return af::SUCCESS;
}

void VFLoop::SetMaxDtypeSize(std::string dtype) {
  this->max_dtype_size_ = dtype;
  this->SetLoopDtype(dtype);
}

void VFLoop::SetLoopDtype(std::string dtype) const {
  this->loop_dtype_ = dtype;
  for (const auto &body : this->bodys_) {
    if (body.type_ == LoopType::LOOP) {
      body.loop_->SetLoopDtype(dtype);
    }
  }
}

void VFLoop::Destruct() {
  for (auto body : this->bodys_) {
    if (body.type_ == LoopType::LOOP) {
      body.loop_->Destruct();
      delete body.loop_;
    } else if (body.type_ == LoopType::CALL) {
      delete body.call_;
    }
  }
}

/********************************** 生成阶段调用 ***********************************/
Status VFLoop::Generate(const TPipe &tpipe, const TensorManager &tensor_mgr, int32_t depth, std::string &result,
                        std::string &loop_size_result, int32_t &only_loop_max_depth,
                        std::vector<std::string> &loop_size_vec, const ArangeOffsetMap &arange_offsets) const {
  std::vector<ascir::AxisId> current_axis;
  std::stringstream ss;
  std::stringstream loop_size_ss;
  std::set<std::string> emitted_declarations;
  GE_CHK_STATUS_RET(this->GenerateLoop(tpipe, tensor_mgr, depth, current_axis, ss, loop_size_ss, only_loop_max_depth,
                                       loop_size_vec, emitted_declarations, arange_offsets),
                    "Generate loop failed");
  result = ss.str();
  loop_size_result = loop_size_ss.str();
  return af::SUCCESS;
}

Status VFLoop::GenerateCvUbFuse(const TPipe &tpipe, const TensorManager &tensor_mgr, std::string &result,
                                std::string &loop_size_result) const {
  std::stringstream ss;
  std::stringstream loop_size_ss;
  loop_size_ss << "  uint16_t cv_m_loop_size = static_cast<uint16_t>(curAivM);\n";
  loop_size_ss << "  uint16_t cv_n_loop_size = loop_times;\n";

  ss << "for (uint16_t cv_m = 0; cv_m < cv_m_loop_size; cv_m++) {" << std::endl;
  ss << "  AscendC::MicroAPI::MaskReg preg_0;" << std::endl;
  ss << "  for (uint16_t cv_n = 0; cv_n < cv_n_loop_size; cv_n++) {" << std::endl;
  ss << "    uint32_t sreg_0 = curAivN - cv_n * ELEMENT_PER_VECTOR_LENGTH;" << std::endl;
  ss << "    preg_0 = AscendC::MicroAPI::UpdateMask<" << this->max_dtype_size_ << ">(sreg_0);\n";
  std::vector<ascir::AxisId> current_axis = {af::kIdNone};
  GE_CHK_STATUS_RET(GenerateCvUbFuseBody(tpipe, tensor_mgr, current_axis, ss), "Generate CV UBFuse body failed");
  ss << "  }" << std::endl;
  ss << "}" << std::endl;

  result = ss.str();
  loop_size_result = loop_size_ss.str();
  return af::SUCCESS;
}

Status VFLoop::GenerateLoop(const TPipe &tpipe, const TensorManager &tensor_mgr, int32_t depth,
                            std::vector<ascir::AxisId> &current_axis, std::stringstream &ss,
                            std::stringstream &loop_size_ss, int32_t &only_loop_max_depth,
                            std::vector<std::string> &loop_size_vec, std::set<std::string> &emitted_declarations,
                            const ArangeOffsetMap &arange_offsets) const {
  if (this->axis_id_ == af::kIdNone) {
    GE_CHK_STATUS_RET(this->GenerateBody(tpipe, tensor_mgr, depth, current_axis, ss, loop_size_ss, only_loop_max_depth,
                                         loop_size_vec, emitted_declarations, arange_offsets),
                      "Codegen generate body failed when axis id is none");
    return af::SUCCESS;
  }

  const auto &axis = tpipe.tiler.GetAxis(this->axis_id_);
  int32_t current_depth = static_cast<int32_t>(current_axis.size());
  const bool is_innermost_loop = current_depth == depth;
  const bool emit_loop_size = emitted_declarations.insert(axis.loop_size.Str()).second;
  bool emit_depth_declarations = false;
  if (is_innermost_loop) {
    emit_depth_declarations = emitted_declarations.insert("depth_" + std::to_string(current_depth)).second;
  }
  if (emit_loop_size) {
    const std::string loop_size_value =
        is_innermost_loop ? "loop_times" : "static_cast<uint16_t>(output_dims_" + std::to_string(current_depth) + ")";
    loop_size_ss << "  uint16_t " << axis.loop_size.Str() << " = " << loop_size_value << ";\n";
  }
  if (is_innermost_loop && emit_depth_declarations) {
    ss << "  uint32_t sreg_" << current_depth << " = element_count;\n";
    ss << "  AscendC::MicroAPI::MaskReg preg_" << current_depth << ";\n";
  }
  loop_size_vec.push_back(axis.loop_size.Str());
  current_axis.push_back(this->axis_id_);
  ss << "for (" << "uint16_t " << axis.Variable::name << " = 0; " << axis << " < " << axis.loop_size.Str() << "; "
     << axis << "++) "
     << "{" << std::endl;
  if (is_innermost_loop) {
    ss << "    preg_" << current_depth << " = " << "AscendC::MicroAPI::UpdateMask<" << this->loop_dtype_ << ">("
       << "sreg_" << current_depth << ");\n";
  }
  GE_CHK_STATUS_RET(this->GenerateBody(tpipe, tensor_mgr, depth, current_axis, ss, loop_size_ss, only_loop_max_depth,
                                       loop_size_vec, emitted_declarations, arange_offsets),
                    "Codegen generate body failed for normal loop");
  ss << "}" << std::endl;

  current_axis.pop_back();
  return af::SUCCESS;
}

Status VFLoop::GenerateBody(const TPipe &tpipe, const TensorManager &tensor_mgr, int32_t depth,
                            std::vector<ascir::AxisId> &current_axis, std::stringstream &ss,
                            std::stringstream &loop_size_ss, int32_t &only_loop_max_depth,
                            std::vector<std::string> &loop_size_vec, std::set<std::string> &emitted_declarations,
                            const ArangeOffsetMap &arange_offsets) const {
  bool has_loop = false;
  bool has_call = false;
  for (const auto &body : this->bodys_) {
    if (body.type_ == LoopType::LOOP) {
      GE_CHK_STATUS_RET(
          body.loop_->GenerateLoop(tpipe, tensor_mgr, depth, current_axis, ss, loop_size_ss, only_loop_max_depth,
                                   loop_size_vec, emitted_declarations, arange_offsets),
          "Generate loop for body failed");
      has_loop = true;
    } else if (body.type_ == LoopType::CALL) {
      if (body.call_->unit == ge::ComputeUnit::kUnitNone) {
        continue;
      }
      std::string preg_name = GetOriginPregName(current_axis, depth);
      std::string ub_offset = "";
      if (body.call_->GetMicroApiName() == "Load") {
        const MicroApiTensor *reg_tensor_ptr = tensor_mgr.GetTensor(body.call_->GetOutputTensorIdByIndex(0));
        const Tensor *ub_tensor_ptr = tpipe.GetTensor(body.call_->GetInputTensorIdByIndex(0));
        ub_offset = GetUbAddrOffset(tpipe, reg_tensor_ptr, ub_tensor_ptr);
      } else if (body.call_->GetMicroApiName() == "Store") {
        const Tensor *ub_tensor_ptr = tpipe.GetTensor(body.call_->GetOutputTensorIdByIndex(0));
        const MicroApiTensor *reg_tensor_ptr = tensor_mgr.GetTensor(body.call_->GetOutputTensorIdByIndex(1));
        ub_offset = GetUbAddrOffset(tpipe, reg_tensor_ptr, ub_tensor_ptr);
        GetUbStorePreg(ub_tensor_ptr, preg_name);
      } else if (body.call_->GetMicroApiName() == "Arange") {
        const auto *reg_tensor_ptr = tensor_mgr.GetTensor(body.call_->GetOutputTensorIdByIndex(0));
        GE_ASSERT_NOTNULL(reg_tensor_ptr);
        GE_CHK_STATUS_RET(GetArangeBlockOffset(tpipe, reg_tensor_ptr, current_axis, ub_offset),
                          "Generate Arange block offset failed");
        const auto tensor_id = body.call_->GetOutputTensorIdByIndex(0);
        const auto external_offset = arange_offsets.find(tensor_id);
        if (external_offset != arange_offsets.end() && external_offset->second != "0") {
          ub_offset = external_offset->second + " + (" + ub_offset + ")";
        }
      }
      std::string micro_api_call_str;
      CallParam param = {preg_name, ub_offset, this->max_dtype_size_, {}};
      if (body.call_->HasArangeParam()) {
        const auto tensor_id = body.call_->GetOutputTensorIdByIndex(0);
        param.arange.valid = true;
        param.arange.base = "arange_base_" + std::to_string(tensor_id);
        param.arange.step = "arange_step_" + std::to_string(tensor_id);
      }
      GE_CHK_STATUS_RET(body.call_->Generate(tensor_mgr, tpipe, param, micro_api_call_str),
                        "Generate micro api call failed");
      ss << micro_api_call_str;
      has_call = true;
    }
  }
  if (has_loop && !has_call) {
    only_loop_max_depth = std::max(only_loop_max_depth, static_cast<int32_t>(current_axis.size()));
  }
  return af::SUCCESS;
}

void VFLoop::CollectArangeParams(const TPipe &tpipe, std::vector<ArangeParam> &params) const {
  for (const auto &body : bodys_) {
    if (body.type_ == LoopType::CALL && body.call_->HasArangeParam()) {
      std::string base;
      std::string step;
      body.call_->GetArangeParams(tpipe, base, step);
      params.push_back({body.call_->GetOutputTensorIdByIndex(0), std::move(base), std::move(step), "0", {}});
    } else if (body.type_ == LoopType::LOOP) {
      body.loop_->CollectArangeParams(tpipe, params);
    }
  }
}

Status VFLoop::GenerateCvUbFuseBody(const TPipe &tpipe, const TensorManager &tensor_mgr,
                                    std::vector<ascir::AxisId> &current_axis, std::stringstream &ss) const {
  for (const auto &body : this->bodys_) {
    if (body.type_ == LoopType::LOOP) {
      GE_CHK_STATUS_RET(body.loop_->GenerateCvUbFuseBody(tpipe, tensor_mgr, current_axis, ss),
                        "Generate CV UBFuse loop body failed");
      continue;
    }
    if (body.type_ != LoopType::CALL || body.call_->unit == ge::ComputeUnit::kUnitNone) {
      continue;
    }

    std::string preg_name = GetOriginPregName(current_axis, 0);
    GE_CHK_STATUS_RET(GenerateMicroApiCall(tpipe, tensor_mgr, body, this->max_dtype_size_, preg_name, ss),
                      "Generate CV UBFuse MicroAPI call failed");
  }
  return af::SUCCESS;
}

void VFLoop::CollectMaskRegTempTensors(const TPipe &tpipe, const TensorManager &tensor_mgr,
                                       std::vector<std::string> &temp_tensors) const {
  for (const auto &body : this->bodys_) {
    if (body.type_ == LoopType::LOOP) {
      body.loop_->CollectMaskRegTempTensors(tpipe, tensor_mgr, temp_tensors);
    } else if (body.type_ == LoopType::CALL) {
      std::string api_name = body.call_->GetMicroApiName();
      std::string tensor_name;

      // Compare 输出不是 MaskReg：需要临时 MaskReg 用于转换
      if (api_name == "GE" || api_name == "EQ" || api_name == "NE" || api_name == "LE" || api_name == "LT" ||
          api_name == "GT") {
        const MicroApiTensor *output_tensor = tensor_mgr.GetTensor(body.call_->GetOutputTensorIdByIndex(0));
        if (output_tensor && !output_tensor->init_as_mask_reg_) {
          tensor_name = output_tensor->name;
        }
      }
      // Where/Select mask 输入不是 MaskReg：需要临时 MaskReg 用于转换
      if (api_name == "Select") {
        const MicroApiTensor *input_tensor = tensor_mgr.GetTensor(body.call_->GetInputTensorIdByIndex(0));
        if (input_tensor && !input_tensor->init_as_mask_reg_) {
          tensor_name = input_tensor->name;
        }
      }

      // 添加前检查是否已存在，避免重复添加同一个 tensor
      if (!tensor_name.empty() &&
          std::find(temp_tensors.begin(), temp_tensors.end(), tensor_name) == temp_tensors.end()) {
        temp_tensors.push_back(tensor_name);
      }
    }
  }
}
}  // namespace codegen
