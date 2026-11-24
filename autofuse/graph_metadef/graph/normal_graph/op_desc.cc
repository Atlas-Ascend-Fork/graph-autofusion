/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/op_desc.h"

#include "base/err_msg.h"
#include "graph/common_error_codes.h"
#include "graph/operator_factory_impl.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/transformer_utils.h"
#include "graph/utils/node_utils.h"
#include "common/util/mem_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "common/util/trace_manager/trace_manager.h"
#include "common/checker.h"
#include "graph/utils/op_desc_utils.h"

namespace {
using std::make_pair;
using std::shared_ptr;

void AddDynamicNameIndex(const std::map<std::string, uint32_t> &dynamic_names_indexes, size_t insert_index,
                         std::map<std::string, uint32_t> &name_indexes) {
  // Update index in input_name_idx
  for (auto &name_index : name_indexes) {
    if (name_index.second >= (insert_index)) {
      name_index.second += dynamic_names_indexes.size();
    }
  }
  name_indexes.insert(dynamic_names_indexes.cbegin(), dynamic_names_indexes.cend());
}
}  // namespace

namespace ge {
using af::ComGraphMakeShared;
TensorType::TensorType(DataType dt) {
  tensor_type_impl_ = ComGraphMakeShared<TensorTypeImpl>();
  if (tensor_type_impl_ != nullptr) {
    tensor_type_impl_->GetMutableDateTypeSet().emplace(dt);
  }
}

TensorType::TensorType(const std::initializer_list<DataType> &initial_types) {
  tensor_type_impl_ = ComGraphMakeShared<TensorTypeImpl>();
  if (tensor_type_impl_ != nullptr) {
    tensor_type_impl_->GetMutableDateTypeSet() = initial_types;
  }
}
}  // namespace ge

namespace af {

static const GeTensorDesc &InvalidGeTensorDesc() {
  const static GeTensorDesc kGlobalInvalidGeTensorDesc;
  return kGlobalInvalidGeTensorDesc;
}
const std::string ATTR_NAME_ID = "id";

const std::string ATTR_NAME_STREAM_ID = "stream_id";

const std::string ATTR_NAME_INPUT_NAME = "input_name";

const std::string ATTR_NAME_SRC_NAME = "src_name";

const std::string ATTR_NAME_SRC_INDEX = "src_index";

const std::string ATTR_NAME_INPUT = "input";

const std::string ATTR_NAME_INPUT_DESC = "input_desc";

const std::string ATTR_NAME_OUTPUT_DESC = "output_desc";

const std::string ATTR_NAME_DST_NAME = "dst_name";

const std::string ATTR_NAME_DST_INDEX = "dst_index";

const std::string ATTR_NAME_WORKSPACE = "workspace";

const std::string ATTR_NAME_WORKSPACE_BYTES = "workspace_bytes";

const std::string ATTR_NAME_IS_INPUT_CONST = "is_input_const";

const std::string ATTR_NAME_OP_KERNEL_LIB_NAME = "_ge_attr_op_kernel_lib_name";

OpDescImpl::OpDescImpl() {
  meta_data_.has_out_attr_ = true;
}

OpDescImpl::OpDescImpl(const std::string &name, const std::string &type) : meta_data_(name, type) {
  meta_data_.has_out_attr_ = true;
}

OpDescImpl::OpDescImpl(const OpDescImpl &op_desc_impl) {
  subgraph_instance_names_ = op_desc_impl.subgraph_instance_names_;
  subgraph_names_to_index_ = op_desc_impl.subgraph_names_to_index_;
  for (const auto &input_desc : op_desc_impl.inputs_desc_) {
    inputs_desc_.emplace_back(MakeShared<GeTensorDesc>(*input_desc));
  }
  input_name_idx_ = op_desc_impl.input_name_idx_;
  for (const auto &output_desc : op_desc_impl.outputs_desc_) {
    outputs_desc_.emplace_back(MakeShared<GeTensorDesc>(*output_desc));
  }
  output_name_idx_ = op_desc_impl.output_name_idx_;
  infer_func_ = op_desc_impl.infer_func_;
  infer_format_func_ = op_desc_impl.infer_format_func_;
  infer_value_range_func_ = op_desc_impl.infer_value_range_func_;
  verifier_func_ = op_desc_impl.verifier_func_;
  infer_data_slice_func_ = op_desc_impl.infer_data_slice_func_;
  op_kernel_lib_name_ = op_desc_impl.op_kernel_lib_name_;
  engine_name_ = op_desc_impl.engine_name_;
  meta_data_ = op_desc_impl.meta_data_;
  attrs_ = op_desc_impl.attrs_;
  tiling_func_info_ = op_desc_impl.tiling_func_info_;
  atomic_tiling_func_info_ = op_desc_impl.atomic_tiling_func_info_;
}

OpDescImpl &OpDescImpl::operator=(const OpDescImpl &op_desc_impl) & {
  if (this == &op_desc_impl) {
    return *this;
  }
  subgraph_instance_names_ = op_desc_impl.subgraph_instance_names_;
  subgraph_names_to_index_ = op_desc_impl.subgraph_names_to_index_;
  for (const auto &input_desc : op_desc_impl.inputs_desc_) {
    inputs_desc_.emplace_back(MakeShared<GeTensorDesc>(*input_desc));
  }
  input_name_idx_ = op_desc_impl.input_name_idx_;
  for (const auto &output_desc : op_desc_impl.outputs_desc_) {
    outputs_desc_.emplace_back(MakeShared<GeTensorDesc>(*output_desc));
  }
  output_name_idx_ = op_desc_impl.output_name_idx_;
  infer_func_ = op_desc_impl.infer_func_;
  infer_format_func_ = op_desc_impl.infer_format_func_;
  infer_value_range_func_ = op_desc_impl.infer_value_range_func_;
  verifier_func_ = op_desc_impl.verifier_func_;
  infer_data_slice_func_ = op_desc_impl.infer_data_slice_func_;
  op_kernel_lib_name_ = op_desc_impl.op_kernel_lib_name_;
  engine_name_ = op_desc_impl.engine_name_;
  meta_data_ = op_desc_impl.meta_data_;
  attrs_ = op_desc_impl.attrs_;
  tiling_func_info_ = op_desc_impl.tiling_func_info_;
  atomic_tiling_func_info_ = op_desc_impl.atomic_tiling_func_info_;
  return *this;
}

OpDescImpl::OpDescImpl(const af::proto::OpDef &op_def) : meta_data_(op_def.name(), op_def.type()) {
  // proto deserialize meta_data
  DeSerializeOpDefToMetaData(op_def);
}

void OpDescImpl::DeSerializeOpDefToMetaData(const proto::OpDef &op_def) {
  meta_data_.has_out_attr_ = op_def.has_out_attr();
  meta_data_.id_ = op_def.id();
  meta_data_.stream_id_ = op_def.stream_id();
  meta_data_.inputs_.clear();
  (void)meta_data_.inputs_.insert(meta_data_.inputs_.cend(), op_def.input().cbegin(), op_def.input().cend());
  meta_data_.input_names_.clear();
  (void)meta_data_.input_names_.insert(meta_data_.input_names_.cend(), op_def.input_name().cbegin(),
                                       op_def.input_name().cend());
  meta_data_.src_names_.clear();
  (void)meta_data_.src_names_.insert(meta_data_.src_names_.cend(), op_def.src_name().cbegin(),
                                     op_def.src_name().cend());
  meta_data_.src_indexes_.clear();
  (void)meta_data_.src_indexes_.insert(meta_data_.src_indexes_.cend(), op_def.src_index().cbegin(),
                                       op_def.src_index().cend());
  meta_data_.dst_names_.clear();
  (void)meta_data_.dst_names_.insert(meta_data_.dst_names_.cend(), op_def.dst_name().cbegin(),
                                     op_def.dst_name().cend());
  meta_data_.dst_indexes_.clear();
  (void)meta_data_.dst_indexes_.insert(meta_data_.dst_indexes_.cend(), op_def.dst_index().cbegin(),
                                       op_def.dst_index().cend());
  meta_data_.input_offsets_.clear();
  (void)meta_data_.input_offsets_.insert(meta_data_.input_offsets_.cend(), op_def.input_i().cbegin(),
                                         op_def.input_i().cend());
  meta_data_.output_offsets_.clear();
  (void)meta_data_.output_offsets_.insert(meta_data_.output_offsets_.cend(), op_def.output_i().cbegin(),
                                          op_def.output_i().cend());
  meta_data_.workspaces.clear();
  (void)meta_data_.workspaces.insert(meta_data_.workspaces.cend(), op_def.workspace().cbegin(),
                                     op_def.workspace().cend());
  meta_data_.workspace_bytes_list_.clear();
  (void)meta_data_.workspace_bytes_list_.insert(meta_data_.workspace_bytes_list_.cend(),
                                                op_def.workspace_bytes().cbegin(), op_def.workspace_bytes().cend());
  meta_data_.is_input_consts_.clear();
  (void)meta_data_.is_input_consts_.insert(meta_data_.is_input_consts_.cend(), op_def.is_input_const().cbegin(),
                                           op_def.is_input_const().cend());
  meta_data_.subgraph_names_.clear();
  (void)meta_data_.subgraph_names_.insert(meta_data_.subgraph_names_.cend(), op_def.subgraph_name().cbegin(),
                                          op_def.subgraph_name().cend());
}

void OpDescImpl::SerializeMetaDataToOpDef(proto::OpDef *const op_def) {
  op_def->set_name(meta_data_.name_);
  op_def->set_type(meta_data_.type_);
  op_def->set_has_out_attr(meta_data_.has_out_attr_);
  op_def->set_id(meta_data_.id_);
  op_def->set_stream_id(meta_data_.stream_id_);
  op_def->clear_input();
  for (const auto &input : meta_data_.inputs_) {
    op_def->add_input(input);
  }
  op_def->clear_input_name();
  for (const auto &input_name : meta_data_.input_names_) {
    op_def->add_input_name(input_name);
  }
  op_def->clear_src_name();
  for (const auto &src_name : meta_data_.src_names_) {
    op_def->add_src_name(src_name);
  }
  op_def->clear_src_index();
  for (const auto src_idx : meta_data_.src_indexes_) {
    op_def->add_src_index(src_idx);
  }
  op_def->clear_dst_name();
  for (const auto &dst_name : meta_data_.dst_names_) {
    op_def->add_dst_name(dst_name);
  }
  op_def->clear_dst_index();
  for (const auto dst_idx : meta_data_.dst_indexes_) {
    op_def->add_dst_index(dst_idx);
  }
  op_def->clear_input_i();
  for (const auto input_i : meta_data_.input_offsets_) {
    op_def->add_input_i(input_i);
  }
  op_def->clear_output_i();
  for (const auto output_i : meta_data_.output_offsets_) {
    op_def->add_output_i(output_i);
  }
  op_def->clear_workspace();
  for (const auto workspace : meta_data_.workspaces) {
    op_def->add_workspace(workspace);
  }
  op_def->clear_workspace_bytes();
  for (const auto workspace_bytes : meta_data_.workspace_bytes_list_) {
    op_def->add_workspace_bytes(workspace_bytes);
  }
  op_def->clear_is_input_const();
  for (const auto is_input_const : meta_data_.is_input_consts_) {
    op_def->add_is_input_const(is_input_const);
  }
  op_def->clear_subgraph_name();
  for (const auto &subgraph_name : meta_data_.subgraph_names_) {
    op_def->add_subgraph_name(subgraph_name);
  }
}

string OpDescImpl::GetName() const {
  return meta_data_.name_;
}

const char *OpDescImpl::GetNamePtr() const {
  return meta_data_.name_.c_str();
}

void OpDescImpl::SetName(const std::string &name) {
  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(), "name",
                   "", "", name);
  meta_data_.SetOpName(name);
}

const char *OpDescImpl::GetTypePtr() const {
  return meta_data_.type_.c_str();
}

std::string OpDescImpl::GetType() const {
  return meta_data_.type_;
}

void OpDescImpl::SetType(const std::string &type) {
  if (meta_data_.type_ == type) {
    return;
  }
  meta_data_.type_ = type;

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(), "type",
                   "", "", type);
}

void OpDescImpl::SetIrRelated(const OpDescImpl *r_op_desc) {
  if (r_op_desc != nullptr) {
    this->meta_data_.ir_meta_ = r_op_desc->meta_data_.ir_meta_;
  } else {
    this->meta_data_.ir_meta_ = IRMetaData("");
  }
}

graphStatus OpDescImpl::AddInputDesc(const af::GeTensorDesc &input_desc) {
  const int32_t index = static_cast<int32_t>(inputs_desc_.size());

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(), this->GetName(),
                   "input_desc", "", "", index);
  return AddInputDesc("__input" + std::to_string(index), input_desc);
}

graphStatus OpDescImpl::AddInputDesc(const uint32_t index, const af::GeTensorDesc &input_desc) {
  graphStatus ret = GRAPH_SUCCESS;
  if (index < inputs_desc_.size()) {
    //  InputsDesc[index] is exist, then update it
    ret = UpdateInputDesc(index, input_desc);
  } else {
    //  InputDesc[index] does not exist, then add it
    ret = AddInputDesc(input_desc);
  }
  return ret;
}

graphStatus OpDescImpl::AddInputDesc(const std::string &name, const af::GeTensorDesc &input_desc) {
  if (input_name_idx_.find(name) != input_name_idx_.end()) {
    GELOGI("input %s already exists, update it", name.c_str());
    const graphStatus ret = UpdateInputDesc(name, input_desc);
    return ret;
  } else {
    int32_t index = static_cast<int32_t>(inputs_desc_.size());
    const std::shared_ptr<GeTensorDesc> in_desc = ComGraphMakeShared<GeTensorDesc>(input_desc);
    if (in_desc == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "AddInputDesc failed, as malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] AddInputDesc failed, as malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }
    inputs_desc_.push_back(in_desc);
    (void)input_name_idx_.insert(make_pair(name, index));
    (void)meta_data_.ir_meta_.AddRegisterInputName(name);

    TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(), this->GetName(),
                     "input_desc:" << index, "", "", "input_name:" << name);
    return GRAPH_SUCCESS;
  }
}

graphStatus OpDescImpl::AddInputDescMiddle(const std::string &name, const uint32_t num, const size_t index) {
  std::map<std::string, uint32_t> dynamic_names_indexes;
  for (uint32_t i = 0U; i < num; i++) {
    std::string input_name = name + std::to_string(i);
    GE_CHK_BOOL_EXEC(
        (input_name_idx_.find(input_name) == input_name_idx_.end()),
        REPORT_INNER_ERR_MSG("E18888", "Add input tensor_desc already exists. name[%s]", input_name.c_str());
        GELOGE(af::FAILED, "[Check][Param] Add input tensor_desc already exists. name[%s]", input_name.c_str());
        return GRAPH_FAILED);

    const std::shared_ptr<GeTensorDesc> in_desc = ComGraphMakeShared<GeTensorDesc>(GeTensorDesc());
    if (in_desc == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "AddInputDescMiddle failed, as malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] AddInputDescMiddle failed, as malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }

    if (index > inputs_desc_.size()) {
      REPORT_INNER_ERR_MSG("E18888",
                           "AddInputDescMiddle failed, as param index(%zu) "
                           "is bigger than inputs size(%zu).",
                           index, inputs_desc_.size());
      GELOGE(GRAPH_FAILED,
             "[Check][Param] AddInputDescMiddle failed, as param index(%zu) "
             "is bigger than inputs size(%zu).",
             index, inputs_desc_.size());
      return GRAPH_FAILED;
    }

    auto pos = inputs_desc_.begin();
    std::advance(pos, index + i);
    (void)inputs_desc_.insert(pos, in_desc);

    dynamic_names_indexes.insert(make_pair(input_name, i + index));
    TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(), this->GetName(),
                     "input_desc:" << (i + index), "", "", "input_name:" << input_name);
  }
  AddDynamicNameIndex(dynamic_names_indexes, index, input_name_idx_);
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::AddOutputDescMiddle(const std::string &name, const uint32_t num, const size_t index) {
  std::map<std::string, uint32_t> dynamic_names_indexes;
  for (uint32_t i = 0U; i < num; i++) {
    std::string output_name = name + std::to_string(i);
    GE_CHK_BOOL_EXEC(
        (output_name_idx_.find(output_name) == output_name_idx_.end()),
        REPORT_INNER_ERR_MSG("E18888", "Add output tensor_desc already exists. name[%s]", output_name.c_str());
        return GRAPH_FAILED, "[Check][Param] Add output tensor_desc already exists. name[%s]", output_name.c_str());

    const std::shared_ptr<GeTensorDesc> out_desc = ComGraphMakeShared<GeTensorDesc>(GeTensorDesc());
    if (out_desc == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "AddOutputDescMiddle failed, as malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] AddOutputDescMiddle failed, as malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }

    if (index > outputs_desc_.size()) {
      REPORT_INNER_ERR_MSG("E18888",
                           "AddOutputDescMiddle failed, as param index(%zu) "
                           "is bigger than outputs size(%zu).",
                           index, outputs_desc_.size());
      GELOGE(GRAPH_FAILED,
             "[Check][Param] AddOutputDescMiddle failed, as param index(%zu) "
             "is bigger than outputs size(%zu).",
             index, outputs_desc_.size());
      return GRAPH_FAILED;
    }

    auto pos = outputs_desc_.begin();
    std::advance(pos, index + i);
    (void)outputs_desc_.insert(pos, out_desc);
    (void)dynamic_names_indexes.insert(make_pair(output_name, i + index));
    TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(), this->GetName(),
                     "output_desc:" << (i + index), "", "", output_name);
  }
  AddDynamicNameIndex(dynamic_names_indexes, index, output_name_idx_);
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::AddInputDescForward(const std::string &name, const uint32_t num) {
  std::map<std::string, uint32_t> dynamic_input_name_indexes;
  for (uint32_t i = 0U; i < num; i++) {
    std::string input_name = name + std::to_string(i);
    GE_CHK_BOOL_EXEC(
        (input_name_idx_.find(input_name) == input_name_idx_.end()),
        REPORT_INNER_ERR_MSG("E18888", "Add input tensor_desc already exists. name[%s]", input_name.c_str());
        return GRAPH_FAILED, "[Check][Param] Add input tensor_desc already exists. name[%s]", input_name.c_str());

    const std::shared_ptr<GeTensorDesc> in_desc = ComGraphMakeShared<GeTensorDesc>(GeTensorDesc());
    if (in_desc == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "AddInputDescForward failed, as malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] AddInputDescForward failed, as malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }
    (void)inputs_desc_.insert(inputs_desc_.cbegin(), in_desc);

    dynamic_input_name_indexes.insert(make_pair(input_name, i));
    TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(), this->GetName(),
                     "input_desc:0", "", "", "input_name:" << input_name);
  }
  AddDynamicNameIndex(dynamic_input_name_indexes, 0U, input_name_idx_);
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::AddOutputDescForward(const std::string &name, const uint32_t num) {
  std::map<std::string, uint32_t> output_name_indexes;
  for (uint32_t i = 0U; i < num; i++) {
    const auto index = i;
    std::string output_name = name + std::to_string(index);
    GE_CHK_BOOL_EXEC(
        (output_name_idx_.find(output_name) == output_name_idx_.end()),
        REPORT_INNER_ERR_MSG("E18888", "Add output tensor_desc already exists. name[%s]", output_name.c_str());
        return GRAPH_FAILED, "[Check][Param] Add output tensor_desc already exists. name[%s]", output_name.c_str());

    const std::shared_ptr<GeTensorDesc> in_desc = ComGraphMakeShared<GeTensorDesc>(GeTensorDesc());
    if (in_desc == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "AddOutputDescForward failed, as malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] AddOutputDescForward failed, as malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }

    (void)outputs_desc_.insert(outputs_desc_.cbegin(), in_desc);
    output_name_indexes.insert(make_pair(output_name, index));
    TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(), this->GetName(),
                     "output_desc:0", "", "", "output_name:" << output_name);
  }
  AddDynamicNameIndex(output_name_indexes, 0U, output_name_idx_);
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::AddOptionalInputDesc(const std::string &name, const af::GeTensorDesc &input_desc) {
  if (OpDescImpl::AddInputDesc(name, input_desc) == GRAPH_FAILED) {
    return GRAPH_FAILED;
  }
  (void)meta_data_.ir_meta_.AddRegisterOptionalInputName(name);
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::UpdateInputDesc(const uint32_t index, const af::GeTensorDesc &tensor_Desc) {
  if (index >= inputs_desc_.size()) {
    GELOGW("[UpdateInput][Check] Input index is invalid, index=%u, input_size=%zu", index, inputs_desc_.size());
    return GRAPH_FAILED;
  }

  inputs_desc_[static_cast<uint64_t>(index)] = ComGraphMakeShared<GeTensorDesc>(tensor_Desc);
  if (inputs_desc_[static_cast<uint64_t>(index)] == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "UpdateInputDesc failed, as malloc shared_ptr failed.");
    GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] UpdateInputDesc failed, as malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "input_desc:" << index, "", "", tensor_Desc.GetName());
  return GRAPH_SUCCESS;
}

bool OpDescImpl::OpDescMembersAreEqual(const OpDescImpl &r_op_desc) const {
  return (IsEqual(this->input_name_idx_, r_op_desc.input_name_idx_, "OpDesc.input_name_idx_") &&
          IsEqual(this->output_name_idx_, r_op_desc.output_name_idx_, "OpDesc.output_name_idx_") &&
          IsEqual(this->meta_data_.ir_meta_, r_op_desc.meta_data_.ir_meta_, "OpDesc.ir_mata_") &&
          IsEqual(this->engine_name_, r_op_desc.engine_name_, "OpDesc.engine_name_") &&
          IsEqual(this->op_kernel_lib_name_, r_op_desc.op_kernel_lib_name_, "OpDesc.op_kernel_lib_name_"));
}

bool OpDescImpl::OpDescAttrsAreEqual(const OpDescImpl &r_op_desc) const {
  // 看起来当前的本判等函数没有考虑属性，补一下UT确认一下
  const auto &r_data = r_op_desc.meta_data_;
  return (IsEqual(meta_data_.name_, r_data.name_, "meta_data_.name_") &&
          IsEqual(meta_data_.type_, r_data.type_, "meta_data_.type_") &&
          IsEqual(meta_data_.inputs_, r_data.inputs_, "meta_data_.inputs_") &&
          IsEqual(meta_data_.has_out_attr_, r_data.has_out_attr_, "meta_data_.has_out_attr_") &&
          IsEqual(meta_data_.stream_id_, r_data.stream_id_, "meta_data_.stream_id_") &&
          IsEqual(meta_data_.input_names_, r_data.input_names_, "meta_data_.input_names_") &&
          IsEqual(meta_data_.src_names_, r_data.src_names_, "meta_data_.src_names_") &&
          IsEqual(meta_data_.dst_names_, r_data.dst_names_, "meta_data_.dst_names_") &&
          IsEqual(meta_data_.src_indexes_, r_data.src_indexes_, "meta_data_.src_indexes_") &&
          IsEqual(meta_data_.dst_indexes_, r_data.dst_indexes_, "meta_data_.dst_indexes_") &&
          IsEqual(meta_data_.input_offsets_, r_data.input_offsets_, "meta_data_.input_offsets_") &&
          IsEqual(meta_data_.output_offsets_, r_data.output_offsets_, "meta_data_.output_offsets_") &&
          IsEqual(meta_data_.workspaces, r_data.workspaces, "meta_data_.workspaces") &&
          IsEqual(meta_data_.workspace_bytes_list_, r_data.workspace_bytes_list_, "meta_data_.workspace_bytes_list_") &&
          IsEqual(meta_data_.is_input_consts_, r_data.is_input_consts_, "meta_data_.is_input_consts_"));
}

bool OpDescImpl::OpDescGenTensorDescsAreEqual(const OpDescImpl &r_op_desc) const {
  // 1.Verify inputs and outputs desc size
  const auto inputs_desc_size = this->inputs_desc_.size();
  const auto r_inputs_desc_size = r_op_desc.inputs_desc_.size();
  if (inputs_desc_size != r_inputs_desc_size) {
    REPORT_INNER_ERR_MSG("E18888",
                         "param r_op_desc inputs count(%zu) not equal to %s inputs count(%zu), "
                         "verify failed.",
                         r_inputs_desc_size, this->GetName().c_str(), inputs_desc_size);
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of OpDesc's inputs desc verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }
  const auto outputs_desc_size = this->outputs_desc_.size();
  const auto r_outputs_desc_size = r_op_desc.outputs_desc_.size();
  if (outputs_desc_size != r_outputs_desc_size) {
    REPORT_INNER_ERR_MSG("E18888",
                         "param r_op_desc outputs count(%zu) not equal to %s outputs count(%zu), "
                         "verify failed.",
                         r_outputs_desc_size, this->GetName().c_str(), outputs_desc_size);
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of OpDesc's outputs desc verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }
  // 2.Verify all inputs desc equal
  for (uint32_t i = 0U; i < inputs_desc_size; i++) {
    const auto &in_ge_tensor_desc = this->GetInputDesc(i);
    const auto &r_in_ge_tensor_desc = r_op_desc.GetInputDesc(i);
    // Determine the connection relationship by GeTensorDesc
    if (!(in_ge_tensor_desc == r_in_ge_tensor_desc)) {
      REPORT_INNER_ERR_MSG("E18888",
                           "r_op_desc inputdesc(index:%u) not equal to %s inputdesc(index:%u), "
                           "verify failed.",
                           i, this->GetName().c_str(), i);
      GELOGE(GRAPH_FAILED, "[Check][Param] Link info of OpDesc's inputs desc verify failed, OpDesc name: %s.",
             this->GetName().c_str());
      return false;
    }
  }
  // 3.Verify all outputs desc equal
  for (uint32_t i = 0U; i < outputs_desc_size; i++) {
    const auto &out_ge_tensor_desc = this->GetOutputDesc(i);
    const auto &r_out_ge_tensor_desc = r_op_desc.GetOutputDesc(i);
    if (!(out_ge_tensor_desc == r_out_ge_tensor_desc)) {
      REPORT_INNER_ERR_MSG("E18888",
                           "r_op_desc outputdesc(index:%u) not equal to %s outputdesc(index:%u), "
                           "verify failed.",
                           i, this->GetName().c_str(), i);
      GELOGE(GRAPH_FAILED, "[Check][Param] Link info of OpDesc's outputs desc verify failed, OpDesc name: %s.",
             this->GetName().c_str());
      return false;
    }
  }
  return true;
}

graphStatus OpDescImpl::UpdateInputDesc(const std::string &name, const af::GeTensorDesc &tensor_Desc) {
  const auto it = input_name_idx_.find(name);
  if (it == input_name_idx_.end()) {
    GELOGW("[UpdateInput][Check] Can not find input desc named %s", name.c_str());
    return GRAPH_FAILED;
  }
  if (it->second >= inputs_desc_.size()) {
    REPORT_INNER_ERR_MSG("E18888", "%u is out of range(0, %zu), check invalid", it->second, inputs_desc_.size());
    GELOGE(GRAPH_FAILED, "[Check][Param] [%u] more than size:%zu of inputs_desc_", it->second, inputs_desc_.size());
    return GRAPH_FAILED;
  }

  inputs_desc_[static_cast<uint64_t>(it->second)] = ComGraphMakeShared<GeTensorDesc>(tensor_Desc);
  if (inputs_desc_[static_cast<uint64_t>(it->second)] == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "UpdateInputDesc failed, as malloc shared_ptr failed.");
    GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] UpdateInputDesc failed, as malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "input_desc:" << it->second, "", "", tensor_Desc.GetName());
  return GRAPH_SUCCESS;
}

bool OpDescImpl::InputIsSet(const std::string &name) const {
  const auto it = input_name_idx_.find(name);
  if (it != input_name_idx_.end()) {
    GE_IF_BOOL_EXEC(it->second >= inputs_desc_.size(),
                    REPORT_INNER_ERR_MSG("E18888", "input name(%s) id(%u) is out of range(0, %zu), check invalid",
                                         name.c_str(), it->second, inputs_desc_.size());
                    GELOGE(GRAPH_FAILED, "[Check][Param] it->second is invalid."); return false);
    const auto tensor_desc = inputs_desc_[static_cast<uint64_t>(it->second)];
    GE_IF_BOOL_EXEC(tensor_desc == nullptr,
                    REPORT_INNER_ERR_MSG("E18888", "tensor_desc(index:%u) is null.", it->second);
                    GELOGE(GRAPH_FAILED, "[Check][Param] tensor_desc(index:%u) is null.", it->second); return false);
    const auto dims = tensor_desc->GetShape().GetDims();
    if (dims.size() > 0U) {
      return true;
    }
  }
  return false;
}

const GeTensorDesc &OpDescImpl::GetInputDesc(const uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG(index < inputs_desc_.size(), InvalidGeTensorDesc());
  return *(inputs_desc_[static_cast<uint64_t>(index)].get());
}

const GeTensorDesc &OpDescImpl::GetInputDesc(const std::string &name) const {
  const auto it = input_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != input_name_idx_.end(), InvalidGeTensorDesc());
  GE_CHK_BOOL_RET_STATUS_NOLOG(it->second < inputs_desc_.size(), InvalidGeTensorDesc());
  return *(inputs_desc_[static_cast<uint64_t>(it->second)].get());
}

GeTensorDescPtr OpDescImpl::MutableInputDesc(const uint32_t index) const {
  if (index >= inputs_desc_.size()) {
    GELOGW("[Get][InputDesc] Failed to get input desc [%u]", index);
    return nullptr;
  }
  if (inputs_desc_[static_cast<uint64_t>(index)] == nullptr) {
    return nullptr;
  }
  if (inputs_desc_[static_cast<uint64_t>(index)]->IsValid() != GRAPH_SUCCESS) {
    GELOGD("[Get][InputDesc] Input desc is invalid [%u]", index);
    return nullptr;
  }
  return inputs_desc_[static_cast<uint64_t>(index)];
}

GeTensorDescPtr OpDescImpl::MutableInputDesc(const std::string &name) const {
  auto input_name_idx = GetAllInputName();
  const std::map<std::string, uint32_t>::const_iterator it = input_name_idx.find(name);
  if (it == input_name_idx.cend()) {
    GELOGW("[Get][InputDesc] Failed to get [%s] input desc", name.c_str());
    return nullptr;
  }
  return MutableInputDesc(it->second);
}

OpDesc::Vistor<string> OpDescImpl::GetAllInputNames(const ConstOpDescPtr &op_desc) const {
  std::vector<std::string> names;
  if (input_name_idx_.empty()) {
    return OpDesc::Vistor<string>(op_desc, names);
  }
  for (const std::pair<std::string, uint32_t> input : input_name_idx_) {
    names.push_back(input.first);
  }
  return OpDesc::Vistor<string>(op_desc, names);
}

void OpDescImpl::SetOpKernelLibName(const std::string &name) {
  op_kernel_lib_name_ = name;
}

std::string OpDescImpl::GetOpKernelLibName() const {
  if (!op_kernel_lib_name_.empty()) {
    return op_kernel_lib_name_;
  }
  return "";
}

void OpDescImpl::SetOpEngineName(const std::string &name) {
  engine_name_ = name;
}

std::string OpDescImpl::GetOpEngineName() const {
  return engine_name_;
}

OpDesc::Vistor<GeTensorDesc> OpDescImpl::GetAllInputsDesc(const ConstOpDescPtr &op_desc) const {
  std::vector<GeTensorDesc> temp{};
  for (const auto &it : inputs_desc_) {
    if (it->IsValid() == GRAPH_SUCCESS) {
      temp.push_back(*it);
    } else {
      GELOGW("[Get][InputDesc] This input_desc is invalid, it won't be returned");
      continue;
    }
  }
  return OpDesc::Vistor<GeTensorDesc>(op_desc, temp);
}

OpDesc::Vistor<GeTensorDescPtr> OpDescImpl::GetAllInputsDescPtr(const ConstOpDescPtr &op_desc) const {
  std::vector<GeTensorDescPtr> temp{};
  for (const auto &it : inputs_desc_) {
    if (it->IsValid() == GRAPH_SUCCESS) {
      temp.push_back(it);
    } else {
      GELOGD("[Get][InputDesc] This input_desc is invalid, it won't be returned");
      continue;
    }
  }
  return OpDesc::Vistor<GeTensorDescPtr>(op_desc, temp);
}

size_t OpDescImpl::GetInputsSize() const {
  //  Just return valid inputs size.InValid desc is set in default OPTION_INPUT register.
  size_t size = 0U;
  for (const auto &it : inputs_desc_) {
    if (it->IsValid() == GRAPH_SUCCESS) {
      size++;
    }
  }
  return size;
}

size_t OpDescImpl::GetAllInputsSize() const {
  return inputs_desc_.size();
}

graphStatus OpDescImpl::AddOutputDesc(const af::GeTensorDesc &output_desc) {
  const int32_t index = static_cast<int32_t>(outputs_desc_.size());
  return AddOutputDesc("__output" + std::to_string(index), output_desc);
}

graphStatus OpDescImpl::AddOutputDesc(const std::string &name, const af::GeTensorDesc &output_desc) {
  GE_CHK_BOOL_EXEC((output_name_idx_.find(name) == output_name_idx_.end()),
                   REPORT_INNER_ERR_MSG("E18888", "Output tensor_Desc already exists. name[%s]", name.c_str());
                   return GRAPH_FAILED, "[Check][Param] Output tensor_Desc already exists. name[%s]", name.c_str());
  const int32_t index = static_cast<int32_t>(outputs_desc_.size());

  const std::shared_ptr<GeTensorDesc> tensor = ComGraphMakeShared<GeTensorDesc>(output_desc);
  if (tensor == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "AddOutputDesc failed, as malloc shared_ptr failed.");
    GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] AddOutputDesc failed, as malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }
  outputs_desc_.push_back(tensor);
  (void)output_name_idx_.insert(make_pair(name, index));
  (void)meta_data_.ir_meta_.AddRegisterOutputName(name);

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(), this->GetName(),
                   "output_desc:" << index, "", "", "output_name:" << name);
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::UpdateOutputDesc(const uint32_t index, const af::GeTensorDesc &tensor_Desc) {
  GE_CHK_BOOL_EXEC((index < outputs_desc_.size()),
                   REPORT_INNER_ERR_MSG("E18888", "param index(%u) is out of range(0, %zu), check invalid", index,
                                        outputs_desc_.size());
                   return GRAPH_FAILED, "[Check][Param] The index is invalid. index[%u]", index);
  outputs_desc_[static_cast<uint64_t>(index)] = ComGraphMakeShared<GeTensorDesc>(tensor_Desc);
  if (outputs_desc_[static_cast<uint64_t>(index)] == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "UpdateOutputDesc failed, as malloc shared_ptr failed.");
    GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] UpdateOutputDesc failed, as malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "output_desc:" << index, "", "", tensor_Desc.GetName());
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::UpdateOutputDesc(const std::string &name, const af::GeTensorDesc &tensor_Desc) {
  const auto it = output_name_idx_.find(name);
  if (it == output_name_idx_.end()) {
    GELOGW("[Update][OutputDesc] Can not find the output desc named %s", name.c_str());
    return GRAPH_FAILED;
  }
  GE_IF_BOOL_EXEC(it->second >= outputs_desc_.size(),
                  REPORT_INNER_ERR_MSG("E18888", "output name(%s) idx(%u) is out of range(0, %zu), check invalid",
                                       name.c_str(), it->second, outputs_desc_.size());
                  GELOGE(GRAPH_FAILED, "[Check][Param] it->second is invalid."); return GRAPH_FAILED);
  outputs_desc_[static_cast<uint64_t>(it->second)] = ComGraphMakeShared<GeTensorDesc>(tensor_Desc);
  if (outputs_desc_[static_cast<uint64_t>(it->second)] == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "UpdateOutputDesc failed, as malloc shared_ptr failed.");
    GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] UpdateOutputDesc failed, as malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "output_desc:" << it->second, "", "", tensor_Desc.GetName());
  return GRAPH_SUCCESS;
}

const GeTensorDesc &OpDescImpl::GetOutputDesc(const uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG(static_cast<size_t>(index) < outputs_desc_.size(), InvalidGeTensorDesc());
  return *(outputs_desc_[static_cast<size_t>(index)].get());
}

const GeTensorDesc &OpDescImpl::GetOutputDesc(const std::string &name) const {
  const auto it = output_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != output_name_idx_.end(), InvalidGeTensorDesc());
  GE_CHK_BOOL_RET_STATUS_NOLOG(it->second < outputs_desc_.size(), InvalidGeTensorDesc());
  return *(outputs_desc_[static_cast<uint64_t>(it->second)].get());
}

GeTensorDescPtr OpDescImpl::MutableOutputDesc(const uint32_t index) const {
  if (index < outputs_desc_.size()) {
    return outputs_desc_[static_cast<uint64_t>(index)];
  }
  GELOGW("[Get][OutputDesc] Failed to get output desc [%u], output number [%zu]", index, outputs_desc_.size());
  return nullptr;
}

GeTensorDescPtr OpDescImpl::MutableOutputDesc(const std::string &name) const {
  const auto it = output_name_idx_.find(name);
  if (it == output_name_idx_.end()) {
    GELOGW("[Update][OutputDesc] Can not find the output desc named %s", name.c_str());
    return nullptr;
  }
  return MutableOutputDesc(it->second);
}

uint32_t OpDescImpl::GetAllOutputsDescSize() const {
  return static_cast<uint32_t>(outputs_desc_.size());
}

OpDesc::Vistor<GeTensorDesc> OpDescImpl::GetAllOutputsDesc(const ConstOpDescPtr &op_desc) const {
  std::vector<GeTensorDesc> temp{};
  for (const auto &it : outputs_desc_) {
    temp.push_back(*it);
  }
  return OpDesc::Vistor<GeTensorDesc>(op_desc, temp);
}

OpDesc::Vistor<GeTensorDescPtr> OpDescImpl::GetAllOutputsDescPtr(const ConstOpDescPtr &op_desc) const {
  return OpDesc::Vistor<GeTensorDescPtr>(op_desc, outputs_desc_);
}

size_t OpDescImpl::GetOutputsSize() const {
  return outputs_desc_.size();
}

ConstGeTensorDescPtr OpDescImpl::GetOutputDescPtr(const uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG((index) < static_cast<uint32_t>(outputs_desc_.size()), nullptr);
  return outputs_desc_[static_cast<uint64_t>(index)];
}

ConstGeTensorDescPtr OpDescImpl::GetInputDescPtr(const uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG((index) < static_cast<uint32_t>(inputs_desc_.size()), nullptr);
  if (inputs_desc_[static_cast<uint64_t>(index)] == nullptr) {
    return nullptr;
  }
  if (inputs_desc_[static_cast<uint64_t>(index)]->IsValid() != GRAPH_SUCCESS) {
    GELOGW("[Get][InputDesc] Input desc %u is invalid", index);
    return nullptr;
  } else {
    return inputs_desc_[static_cast<size_t>(index)];
  }
}

ConstGeTensorDescPtr OpDescImpl::GetInputDescPtrDfault(const uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG((index) < static_cast<uint32_t>(inputs_desc_.size()), nullptr);
  return inputs_desc_[static_cast<uint64_t>(index)];
}

ConstGeTensorDescPtr OpDescImpl::GetInputDescPtr(const std::string &name) const {
  const auto it = input_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != input_name_idx_.end(), shared_ptr<const GeTensorDesc>());
  return inputs_desc_[static_cast<uint64_t>(it->second)];
}

graphStatus OpDescImpl::AddDynamicInputDesc(const std::string &name, const uint32_t num, const bool is_push_back) {
  if (is_push_back) {
    for (uint32_t i = 0U; i < num; i++) {
      if (AddInputDesc(name + std::to_string(i), GeTensorDesc()) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    }
  } else {
    if (AddInputDescForward(name, num) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }
  if (meta_data_.ir_meta_.AddRegisterInputName(name) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::AddDynamicInputDescByIndex(const std::string &name, const uint32_t num, const size_t index) {
  if (AddInputDescMiddle(name, num, index) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::AddDynamicOutputDesc(const std::string &name, const uint32_t num, const bool is_push_back) {
  if (is_push_back) {
    for (uint32_t i = 0U; i < num; i++) {
      if (AddOutputDesc(name + std::to_string(i), GeTensorDesc()) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    }
  } else {
    if (AddOutputDescForward(name, num) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }

  if (meta_data_.ir_meta_.AddRegisterOutputName(name) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool OpDescImpl::IsOptionalInput(const uint32_t index) const {
  return meta_data_.ir_meta_.IsOptionalInput(GetInputNameByIndex(index));
}

std::map<std::string, uint32_t> OpDescImpl::GetAllInputName() const {
  return input_name_idx_;
}

std::map<std::string, uint32_t> OpDescImpl::GetAllOutputName() {
  return output_name_idx_;
}

std::map<std::string, uint32_t> &OpDescImpl::MutableAllInputName() {
  return input_name_idx_;
}

std::map<std::string, uint32_t> &OpDescImpl::MutableAllOutputName() {
  return output_name_idx_;
}

bool OpDescImpl::UpdateInputName(std::map<std::string, uint32_t> input_name_idx) {
  const auto &ir_inputs = GetIRMeta().GetIrInputs();
  size_t last_dyn = ir_inputs.size();
  for (size_t i = 0UL; i < ir_inputs.size(); ++i) {
    if (ir_inputs[i].second == kIrInputDynamic) {
      last_dyn = i;
    }
  }

  if (last_dyn + 1UL < ir_inputs.size()) {
    GELOGW(
        "[Update][InputName] Dynamic input in middle is unsupported to update from factory, last_dyn=%zu ir_size=%zu.",
        last_dyn, ir_inputs.size());
    return false;
  }

  // Use inputDesc_.size() to contain the InValid OptionInput.GetInputsSize() will remove default OptionInput name.
  const auto input_map_size = inputs_desc_.size();
  const auto factory_map_size = input_name_idx.size();
  // It indicates that some inputs have no optional name.
  // The redundant optional name of factory needs to be deleted and then assigned
  if (input_map_size < factory_map_size) {
    GELOGI("org_input_name_num=%zu, factory_input_name_num=%zu", input_map_size, factory_map_size);
    for (auto it = input_name_idx.begin(); it != input_name_idx.end();) {
      if (it->second >= input_map_size) {
        it = input_name_idx.erase(it);
      } else {
        ++it;
      }
    }
    if (input_name_idx.size() == input_map_size) {
      GELOGI("UpdateInputName");
      input_name_idx_ = input_name_idx;
    } else {
      GELOGW("[Update][InputName] After update, org_input_name_num=%zu, factory_input_name_num=%zu", input_map_size,
             input_name_idx.size());
      return false;
    }
  } else if (input_map_size == factory_map_size) {
    input_name_idx_ = input_name_idx;
  } else {
    GELOGW(
        "[Update][InputName] factory_input_name_num can not be less than org_input_name_num, exactly "
        "org_input_name_num=%zu, factory_input_name_num=%zu",
        input_map_size, factory_map_size);
    return false;
  }

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "input_name_idx", "", "", "");
  return true;
}

bool OpDescImpl::UpdateOutputName(std::map<std::string, uint32_t> output_name_idx) {
  const size_t output_map_size = GetAllOutputsDescSize();
  const size_t factory_map_size = output_name_idx.size();
  if (output_map_size < factory_map_size) {
    GELOGI("org_output_name_num=%zu, factory_output_name_num=%zu", output_map_size, factory_map_size);
    for (auto it = output_name_idx.begin(); it != output_name_idx.end();) {
      if (it->second >= output_map_size) {
        it = output_name_idx.erase(it);
      } else {
        ++it;
      }
    }
    if (output_name_idx.size() == output_map_size) {
      GELOGI("UpdateOutputName");
      output_name_idx_ = output_name_idx;

      TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                       "output_name_idx", "", "", "");
      return true;
    }
  } else if (output_map_size == factory_map_size) {
    output_name_idx_ = output_name_idx;

    TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                     "output_name_idx", "", "", "");
    return true;
  } else {
    GELOGW(
        "[Update][OutputName] factory_output_name_num can not be less than org_output_name_num, exactly "
        "org_output_name_num=%zu, factory_output_name_num=%zu",
        output_map_size, output_name_idx.size());
    return false;
  }
  GELOGW("[Update][OutputName] After update, org_output_name_num=%zu, factory_output_name_num=%zu", output_map_size,
         factory_map_size);
  return false;
}

void *OpDescImpl::GetTilingFuncInfo() const {
  return tiling_func_info_;
}

void OpDescImpl::SetTilingFuncInfo(void *tiling_func_info) {
  tiling_func_info_ = tiling_func_info;
}

void *OpDescImpl::GetAtomicTilingFuncInfo() const {
  return atomic_tiling_func_info_;
}

void OpDescImpl::SetAtomicTilingFuncInfo(void *atomic_tiling_func_info) {
  atomic_tiling_func_info_ = atomic_tiling_func_info;
}

std::function<graphStatus(Operator &)> OpDescImpl::GetInferFunc() const {
  return infer_func_;
}

std::function<graphStatus(Operator &)> OpDescImpl::GetVerifyFunc() const {
  return verifier_func_;
}

std::function<graphStatus(Operator &)> OpDescImpl::GetInferFormatFunc() const {
  return infer_format_func_;
}

std::function<graphStatus(Operator &)> OpDescImpl::GetInferValueRangeFunc() const {
  return infer_value_range_func_;
}

std::function<graphStatus(Operator &)> OpDescImpl::GetInferDataSliceFunc() const {
  return infer_data_slice_func_;
}

void OpDescImpl::AddInferFunc(const std::function<graphStatus(Operator &)> &func) {
  infer_func_ = func;
}

void OpDescImpl::AddInferFormatFunc(const std::function<graphStatus(Operator &)> &func) {
  infer_format_func_ = func;
}

void OpDescImpl::AddVerifierFunc(const std::function<graphStatus(Operator &)> &func) {
  verifier_func_ = func;
}

void OpDescImpl::AddInferValueRangeFunc(const std::function<graphStatus(Operator &)> &func) {
  infer_value_range_func_ = func;
}

void OpDescImpl::AddInferDataSliceFunc(const std::function<graphStatus(Operator &)> &func) {
  infer_data_slice_func_ = func;
}

graphStatus OpDescImpl::DefaultInferFormat(const ConstOpDescPtr &op_desc) const {
  ge::Format first_none_nd_format = FORMAT_ND;
  const auto input_descs = GetAllInputsDescPtr(op_desc);
  const auto output_descs = GetAllOutputsDescPtr(op_desc);
  // Overall input and output,get the first non-nd format
  for (const auto &input_desc : input_descs) {
    const Format origin_format = input_desc->GetOriginFormat();
    if (origin_format != FORMAT_ND) {
      first_none_nd_format = origin_format;
      break;
    }
  }
  for (const auto &output_desc : output_descs) {
    const Format origin_format = output_desc->GetOriginFormat();
    if (origin_format != FORMAT_ND) {
      first_none_nd_format = origin_format;
      break;
    }
  }
  // Refresh all input output format
  GELOGD("Default infer format.node[%s], first none ND format is:%d", GetName().c_str(), first_none_nd_format);

  for (const auto &input_desc : input_descs) {
    const Format origin_format = input_desc->GetOriginFormat();
    GELOGD("Default infer format[in].node[%s].origin format is:%d", GetName().c_str(), origin_format);
    if (origin_format == FORMAT_ND) {
      input_desc->SetOriginFormat(first_none_nd_format);
      input_desc->SetFormat(first_none_nd_format);
    }
  }
  for (const auto &output_desc : output_descs) {
    const Format origin_format = output_desc->GetOriginFormat();
    GELOGD("Default infer format[out].node[%s].origin format is:%d", GetName().c_str(), origin_format);
    if (origin_format == FORMAT_ND) {
      output_desc->SetOriginFormat(first_none_nd_format);
      output_desc->SetFormat(first_none_nd_format);
    }
  }
  return GRAPH_SUCCESS;
}

std::string OpDescImpl::GetInputNameByIndex(const uint32_t index) const {
  auto it = input_name_idx_.begin();
  for (; it != input_name_idx_.end(); ++it) {
    if (it->second == index) {
      break;
    }
  }
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != input_name_idx_.end(), "");
  return it->first;
}

int32_t OpDescImpl::GetInputIndexByName(const std::string &name) const {
  const auto it_find = input_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it_find != input_name_idx_.end(), -1);
  return static_cast<int32_t>(it_find->second);
}

graphStatus OpDescImpl::GetDynamicInputIndexesByName(const std::string &name, std::vector<int32_t> &indexes) const {
  uint32_t previous_value = UINT32_MAX;  // 初始化为一个不可能的值

  for (size_t i = 0U;; i++) {
    const auto it_find = input_name_idx_.find(name + std::to_string(i));
    if (it_find != input_name_idx_.end()) {
      if (previous_value != UINT32_MAX && it_find->second != previous_value + 1U) {
        // 如果当前的 value 不是前一个 value 加一，返回 GRAPH_FAILED
        GELOGE(GRAPH_FAILED,
               "The dynamic input index is not continuous. input name[%s], current index[%u], "
               "previous index[%u], expected index[%u]",
               name.c_str(), it_find->second, previous_value, previous_value + 1U);
        return GRAPH_FAILED;
      }
      indexes.emplace_back(it_find->second);
      previous_value = it_find->second;
    } else {
      break;
    }
  }
  return GRAPH_SUCCESS;
}

std::string OpDescImpl::GetValidInputNameByIndex(const uint32_t index) const {
  std::map<std::string, uint32_t> valid_input_name_idx{};
  uint32_t j = 0U;
  for (size_t i = 0U; i < GetAllInputsSize(); i++) {
    if (MutableInputDesc(static_cast<uint32_t>(i)) != nullptr) {
      const auto valid_name = GetInputNameByIndex(static_cast<uint32_t>(i));
      GE_CHK_BOOL_RET_STATUS_NOLOG(!valid_name.empty(), "");
      (void)valid_input_name_idx.insert({valid_name, j});
      j++;
    }
  }
  auto it = valid_input_name_idx.begin();
  for (; it != valid_input_name_idx.end(); ++it) {
    if (it->second == index) {
      break;
    }
  }
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != valid_input_name_idx.end(), "");
  return it->first;
}

std::string OpDescImpl::GetOutputNameByIndex(const uint32_t index) const {
  auto it = output_name_idx_.begin();
  for (; it != output_name_idx_.end(); ++it) {
    if (it->second == index) {
      break;
    }
  }
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != output_name_idx_.end(), "");
  return it->first;
}

int32_t OpDescImpl::GetOutputIndexByName(const std::string &name) const {
  const auto it_find = output_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it_find != output_name_idx_.end(), -1);
  return static_cast<int32_t>(it_find->second);
}

graphStatus OpDescImpl::GetDynamicOutputIndexesByName(const std::string &name, std::vector<int32_t> &indexes) const {
  uint32_t previous_value = UINT32_MAX;

  for (size_t i = 0U;; i++) {
    const auto it_find = output_name_idx_.find(name + std::to_string(i));
    if (it_find != output_name_idx_.end()) {
      if (previous_value != UINT32_MAX && it_find->second != previous_value + 1U) {
        GELOGE(GRAPH_FAILED,
               "The dynamic output index is not continuous. output name[%s], current index[%u], "
               "previous index[%u], expected index[%u]",
               name.c_str(), it_find->second, previous_value, previous_value + 1U);
        return GRAPH_FAILED;
      }
      indexes.emplace_back(it_find->second);
      previous_value = it_find->second;
    } else {
      break;
    }
  }
  return GRAPH_SUCCESS;
}

ProtoAttrMap &OpDescImpl::MutableAttrMap() {
  return attrs_;
}

ConstProtoAttrMap &OpDescImpl::GetAttrMap() const {
  return attrs_;
}

void OpDescImpl::SetId(const int64_t id) {
  meta_data_.id_ = id;

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(), "id", "",
                   "", id);
}

int64_t OpDescImpl::GetId() const {
  return meta_data_.id_;
}

void OpDescImpl::SetStreamId(const int64_t stream_id) {
  meta_data_.stream_id_ = stream_id;

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "stream_id", "", "", stream_id);
}

int64_t OpDescImpl::GetStreamId() const {
  return meta_data_.stream_id_;
}

void OpDescImpl::SetInputName(const vector<string> &input_name) {
  meta_data_.input_names_ = input_name;

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "input_name", "", "", "");
}

vector<string> OpDescImpl::GetInputName() const {
  return meta_data_.input_names_;
}

void OpDescImpl::SetSrcName(const vector<string> &src_name) {
  meta_data_.src_names_ = src_name;

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "src_name", "", "", "");
}

vector<string> OpDescImpl::GetSrcName() const {
  return meta_data_.src_names_;
}

void OpDescImpl::SetSrcIndex(const vector<int64_t> &src_index) {
  meta_data_.src_indexes_ = src_index;

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "src_index", "", "", "");
}

vector<int64_t> OpDescImpl::GetSrcIndex() const {
  return meta_data_.src_indexes_;
}

void OpDescImpl::SetInputOffset(const vector<int64_t> &input) {
  meta_data_.input_offsets_ = input;

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "input_offset", "", "", "");
}

vector<int64_t> OpDescImpl::GetInputOffset() const {
  return meta_data_.input_offsets_;
}

void OpDescImpl::SetOutputOffset(const vector<int64_t> &output) {
  meta_data_.output_offsets_ = output;

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "out_offset", "", "", "");
}

vector<int64_t> OpDescImpl::GetOutputOffset() const {
  return meta_data_.output_offsets_;
}

void OpDescImpl::SetDstName(const vector<string> &dst_name) {
  meta_data_.dst_names_ = dst_name;

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "dst_name", "", "", "");
}

vector<string> OpDescImpl::GetDstName() const {
  return meta_data_.dst_names_;
}

void OpDescImpl::SetDstIndex(const vector<int64_t> &dst_index) {
  meta_data_.dst_indexes_ = dst_index;

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "dst_index", "", "", "");
}

void OpDescImpl::SetWorkspace(const vector<int64_t> &workspace) {
  meta_data_.workspaces.assign(workspace.cbegin(), workspace.cend());
}

vector<int64_t> OpDescImpl::GetWorkspace() const {
  vector<int64_t> res(meta_data_.workspaces.size());
  for (size_t i = 0UL; i < meta_data_.workspaces.size(); ++i) {
    res[i] = meta_data_.workspaces[i];
  }
  return res;
}

void OpDescImpl::SetWorkspaceBytes(const vector<int64_t> &workspace_bytes) {
  meta_data_.workspace_bytes_list_.assign(workspace_bytes.cbegin(), workspace_bytes.cend());
}

vector<int64_t> OpDescImpl::GetWorkspaceBytes() const {
  vector<int64_t> res(meta_data_.workspace_bytes_list_.size());
  for (size_t i = 0UL; i < meta_data_.workspace_bytes_list_.size(); ++i) {
    res[i] = meta_data_.workspace_bytes_list_[i];
  }
  return res;
}

void OpDescImpl::SetIsInputConst(const vector<bool> &is_input_const) {
  meta_data_.is_input_consts_ = is_input_const;

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "is_input_const", "", "", "");
}

vector<bool> OpDescImpl::GetIsInputConst() const {
  return meta_data_.is_input_consts_;
}

std::string OpDescImpl::GetSubgraphInstanceName(const size_t index) const {
  if (index >= subgraph_instance_names_.size()) {
    return "";
  }
  return subgraph_instance_names_.at(index);
}

const std::vector<std::string> &OpDescImpl::GetSubgraphInstanceNames() const {
  return subgraph_instance_names_;
}

void OpDescImpl::RemoveSubgraphInstanceName(const std::string &name) {
  for (auto iter = subgraph_instance_names_.begin(); iter != subgraph_instance_names_.end(); ++iter) {
    if ((*iter) == name) {
      *iter = "";

      TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "delete", TraceManager::GetOutGraphName(), this->GetName(),
                       "subgraph_instance_name", "", "", name);
      return;
    }
  }
}

graphStatus OpDescImpl::AddSubgraphName(const std::string &name) {
  GELOGI("Add subgraph name is %s", name.c_str());
  const std::map<std::string, uint32_t>::const_iterator iter = subgraph_names_to_index_.find(name);
  if (iter != subgraph_names_to_index_.cend()) {
    GELOGW("[Add][Subgraph] Subgraph name %s exists, index %u", name.c_str(), iter->second);
    return GRAPH_FAILED;
  }
  const auto size = subgraph_names_to_index_.size();
  subgraph_names_to_index_[name] = size;
  subgraph_instance_names_.resize(size + 1U);

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(), this->GetName(),
                   "subgraph_name", "", "", name);
  return GRAPH_SUCCESS;
}

const std::map<std::string, uint32_t> &OpDescImpl::GetSubgraphNameIndexes() const {
  return subgraph_names_to_index_;
}

graphStatus OpDescImpl::SetSubgraphInstanceName(const size_t index, const std::string &name) {
  GELOGI("Add sub graph instance name is %s, index is %zu", name.c_str(), index);
  if (index >= subgraph_instance_names_.size()) {
    REPORT_INNER_ERR_MSG("E18888", "Index %zu exceeds the max instance count %zu", index,
                         subgraph_instance_names_.size());
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Index %zu exceeds the max instance count %zu", index,
           subgraph_instance_names_.size());
    return GRAPH_PARAM_INVALID;
  }
  subgraph_instance_names_[index] = name;

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(), this->GetName(),
                   "subgraph_instance_index:" << index, "", "", "name:" << name);
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::GetSubgraphNameByInstanceName(const std::string &instance_name,
                                                      std::string &subgraph_name) const {
  for (size_t idx = 0U; idx < subgraph_instance_names_.size(); ++idx) {
    if (subgraph_instance_names_[idx] != instance_name) {  // find subgraph index.
      continue;
    }

    for (const auto &name_to_index : subgraph_names_to_index_) {
      if (name_to_index.second != idx) {  // find subgraph name.
        continue;
      }

      subgraph_name = name_to_index.first;
      return GRAPH_SUCCESS;
    }
  }

  return GRAPH_PARAM_INVALID;
}

IRMetaData &OpDescImpl::MutableIRMeta() {
  return meta_data_.ir_meta_;
}
const IRMetaData &OpDescImpl::GetIRMeta() const {
  return meta_data_.ir_meta_;
}

bool OpDescImpl::IsSupportSymbolicInferDataType() const {
  return GetIRMeta().GetIRDataTypeSymbolStore().IsSupportSymbolicInferDtype();
}
graphStatus OpDescImpl::SymbolicInferDataType(const OpDescPtr &op_desc) const {
  return GetIRMeta().GetIRDataTypeSymbolStore().InferDtype(op_desc);
}

size_t OpDescImpl::GetIrInputsSize() const {
  return meta_data_.ir_meta_.GetIrInputs().size();
}

std::map<uint32_t, std::string> OpDescImpl::GetAllOutputIndexToName() {
  std::map<uint32_t, std::string> idx2name;
  for (auto &item : output_name_idx_) {
    GELOGD("[%s:%s] get output name %s with idx %zu", GetNamePtr(), GetTypePtr(), item.first.c_str(), item.second);
    idx2name.emplace(item.second, item.first);
  }
  GE_ASSERT_EQ(idx2name.size(), output_name_idx_.size());
  return idx2name;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc()
    : enable_shared_from_this(), AttrHolder(), impl_(ComGraphMakeSharedAndThrow<OpDescImpl>()) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::~OpDesc() {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc(const std::string &name, const std::string &type)
    : enable_shared_from_this(), AttrHolder(), impl_(ComGraphMakeSharedAndThrow<OpDescImpl>(name, type)) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc(const OpDesc &op_desc)
    : enable_shared_from_this(), AttrHolder(op_desc), impl_(ComGraphMakeSharedAndThrow<OpDescImpl>(*(op_desc.impl_))) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc(OpDesc &&op_desc)
    : enable_shared_from_this(),
      AttrHolder(std::move(op_desc)),
      impl_(ComGraphMakeSharedAndThrow<OpDescImpl>(std::move(*(op_desc.impl_)))) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc(const af::proto::OpDef &op_def)
    : enable_shared_from_this(), AttrHolder(), impl_(ComGraphMakeSharedAndThrow<OpDescImpl>(op_def)) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetName() const {
  return impl_->GetName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const char *OpDesc::GetNamePtr() const {
  return impl_->GetNamePtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetNamePtr(const char_t *name) {
  std::string name_str = name == nullptr ? "" : std::string(name);
  return impl_->SetName(name_str);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetName(const std::string &name) {
  return impl_->SetName(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetType() const {
  return impl_->GetType();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const char *OpDesc::GetTypePtr() const {
  return impl_->GetTypePtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetType(const std::string &type) {
  return impl_->SetType(type);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetIrRelated(const OpDescPtr &op_desc) {
  impl_->SetIrRelated(op_desc != nullptr ? op_desc->impl_.get() : nullptr);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::AddInputDesc(const af::GeTensorDesc &input_desc) {
  return impl_->AddInputDesc(input_desc);
}

graphStatus OpDesc::AddInputDesc(const uint32_t index, const af::GeTensorDesc &input_desc) {
  return impl_->AddInputDesc(index, input_desc);
}

graphStatus OpDesc::AddInputDesc(const std::string &name, const af::GeTensorDesc &input_desc) {
  return impl_->AddInputDesc(name, input_desc);
}

graphStatus OpDesc::AddInputDescMiddle(const std::string &name, const uint32_t num, const size_t index) {
  return impl_->AddInputDescMiddle(name, num, index);
}

graphStatus OpDesc::AddOutputDescMiddle(const std::string &name, const uint32_t num, const size_t index) {
  return impl_->AddOutputDescMiddle(name, num, index);
}

graphStatus OpDesc::AddOutputDescForward(const std::string &name, const uint32_t num) {
  return impl_->AddOutputDescForward(name, num);
}

graphStatus OpDesc::AddOptionalInputDesc(const std::string &name, const af::GeTensorDesc &input_desc) {
  return impl_->AddOptionalInputDesc(name, input_desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDesc::UpdateInputDesc(const uint32_t index, const af::GeTensorDesc &tensor_desc) {
  return impl_->UpdateInputDesc(index, tensor_desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::OpDescMembersAreEqual(const OpDesc &r_op_desc) const {
  return impl_->OpDescMembersAreEqual(*(r_op_desc.impl_));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::OpDescAttrsAreEqual(const OpDesc &r_op_desc) const {
  return impl_->OpDescAttrsAreEqual(*(r_op_desc.impl_));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::OpDescGenTensorDescsAreEqual(
    const OpDesc &r_op_desc) const {
  return impl_->OpDescGenTensorDescsAreEqual(*(r_op_desc.impl_));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::operator==(const OpDesc &r_op_desc) const {
  return (OpDescAttrsAreEqual(r_op_desc) && OpDescMembersAreEqual(r_op_desc) &&
          OpDescGenTensorDescsAreEqual(r_op_desc));
}

graphStatus OpDesc::UpdateInputDesc(const std::string &name, const af::GeTensorDesc &tensor_desc) {
  return impl_->UpdateInputDesc(name, tensor_desc);
}

bool OpDesc::InputIsSet(const std::string &name) const {
  return impl_->InputIsSet(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const GeTensorDesc &OpDesc::GetInputDesc(const uint32_t index) const {
  return impl_->GetInputDesc(index);
}

const GeTensorDesc &OpDesc::GetInputDesc(const std::string &name) const {
  return impl_->GetInputDesc(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDescPtr OpDesc::MutableInputDesc(const uint32_t index) const {
  return impl_->MutableInputDesc(index);
}

GeTensorDescPtr OpDesc::MutableInputDesc(const std::string &name) const {
  return impl_->MutableInputDesc(name);
}

bool OpDesc::IsOptionalInput(const uint32_t index) const {
  return IsOptionalInput(GetInputNameByIndex(index));
}

GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<string> OpDesc::GetAllInputNames() const {
  return impl_->GetAllInputNames(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetOpKernelLibName(const std::string &name) {
  impl_->SetOpKernelLibName(name);
  const auto ret = AttrUtils::SetStr(this, ATTR_NAME_OP_KERNEL_LIB_NAME, name);
  if (!ret) {
    REPORT_INNER_ERR_MSG("E18888", "set %s to op failed.", ATTR_NAME_OP_KERNEL_LIB_NAME.c_str());
    GELOGE(GRAPH_FAILED, "[Set][Str] %s to op failed.", ATTR_NAME_OP_KERNEL_LIB_NAME.c_str());
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetOpKernelLibName() const {
  std::string op_kernel_lib_name = impl_->GetOpKernelLibName();
  if (op_kernel_lib_name.empty()) {
    const std::string *str_ptr = AttrUtils::GetStr(this, ATTR_NAME_OP_KERNEL_LIB_NAME);
    if (str_ptr != nullptr) {
      op_kernel_lib_name = *str_ptr;
    }
  }
  return op_kernel_lib_name;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetOpEngineName(const std::string &name) {
  impl_->SetOpEngineName(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetOpEngineName() const {
  return impl_->GetOpEngineName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OppImplVersion OpDesc::GetOppImplVersion() const {
  int64_t opp_impl_version;
  if (!AttrUtils::GetInt(this, ATTR_NAME_BINARY_SOURCE, opp_impl_version)) {
    return OppImplVersion::kOpp;
  }
  return static_cast<OppImplVersion>(opp_impl_version);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<GeTensorDesc> OpDesc::GetAllInputsDesc() const {
  return impl_->GetAllInputsDesc(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<GeTensorDescPtr> OpDesc::GetAllInputsDescPtr() const {
  return impl_->GetAllInputsDescPtr(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDesc::GetInputsSize() const {
  //  Just return valid inputs size.InValid desc is set in default OPTION_INPUT register.
  return impl_->GetInputsSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDesc::GetAllInputsSize() const {
  return impl_->GetAllInputsSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::AddOutputDesc(const af::GeTensorDesc &output_desc) {
  return impl_->AddOutputDesc(output_desc);
}

graphStatus OpDesc::AddOutputDesc(const std::string &name, const af::GeTensorDesc &output_desc) {
  return impl_->AddOutputDesc(name, output_desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDesc::UpdateOutputDesc(const uint32_t index, const af::GeTensorDesc &tensor_desc) {
  return impl_->UpdateOutputDesc(index, tensor_desc);
}

graphStatus OpDesc::UpdateOutputDesc(const std::string &name, const af::GeTensorDesc &tensor_desc) {
  return impl_->UpdateOutputDesc(name, tensor_desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const GeTensorDesc &OpDesc::GetOutputDesc(const uint32_t index) const {
  return impl_->GetOutputDesc(index);
}

const GeTensorDesc &OpDesc::GetOutputDesc(const std::string &name) const {
  return impl_->GetOutputDesc(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDescPtr OpDesc::MutableOutputDesc(const uint32_t index) const {
  return impl_->MutableOutputDesc(index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDescPtr
OpDesc::MutableOutputDesc(const std::string &name) const {
  return impl_->MutableOutputDesc(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t OpDesc::GetAllOutputsDescSize() const {
  return impl_->GetAllOutputsDescSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<GeTensorDesc> OpDesc::GetAllOutputsDesc() const {
  return impl_->GetAllOutputsDesc(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<GeTensorDescPtr> OpDesc::GetAllOutputsDescPtr() const {
  return impl_->GetAllOutputsDescPtr(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDesc::GetOutputsSize() const {
  return impl_->GetOutputsSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstGeTensorDescPtr
OpDesc::GetOutputDescPtr(const uint32_t index) const {
  return impl_->GetOutputDescPtr(index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstGeTensorDescPtr
OpDesc::GetInputDescPtr(const uint32_t index) const {
  return impl_->GetInputDescPtr(index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstGeTensorDescPtr
OpDesc::GetInputDescPtrDfault(const uint32_t index) const {
  return impl_->GetInputDescPtrDfault(index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstGeTensorDescPtr
OpDesc::GetInputDescPtr(const std::string &name) const {
  return impl_->GetInputDescPtr(name);
}

graphStatus OpDesc::AddRegisterInputName(const std::string &name) {
  return impl_->MutableIRMeta().AddRegisterInputName(name);
}

vector<std::string> OpDesc::GetRegisterInputName() const {
  return impl_->MutableIRMeta().GetRegisterInputName();
}

graphStatus OpDesc::AddDynamicInputDesc(const std::string &name, const uint32_t num, const bool is_push_back) {
  return impl_->AddDynamicInputDesc(name, num, is_push_back);
}

graphStatus OpDesc::AddDynamicInputDescByIndex(const std::string &name, const uint32_t num, const size_t index) {
  if (AddInputDescMiddle(name, num, index) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDesc::AddRegisterOutputName(const std::string &name) {
  return impl_->MutableIRMeta().AddRegisterOutputName(name);
}

vector<std::string> OpDesc::GetRegisterOutputName() const {
  return impl_->MutableIRMeta().GetRegisterOutputName();
}

graphStatus OpDesc::AddDynamicOutputDesc(const std::string &name, const uint32_t num, const bool is_push_back) {
  if (is_push_back) {
    for (uint32_t i = 0U; i < num; i++) {
      if (AddOutputDesc(name + std::to_string(i), GeTensorDesc()) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    }
  } else {
    if (AddOutputDescForward(name, num) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }

  if (AddRegisterOutputName(name) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool OpDesc::IsOptionalInput(const std::string &name) const {
  return impl_->GetIRMeta().IsOptionalInput(name);
}

std::map<std::string, uint32_t> OpDesc::GetAllInputName() const {
  return impl_->GetAllInputName();
}

std::map<std::string, uint32_t> OpDesc::GetAllOutputName() {
  return impl_->GetAllOutputName();
}

std::map<std::string, uint32_t> &OpDesc::MutableAllInputName() {
  return impl_->MutableAllInputName();
}

std::map<std::string, uint32_t> &OpDesc::MutableAllOutputName() {
  return impl_->MutableAllOutputName();
}

bool OpDesc::UpdateInputName(const std::map<std::string, uint32_t> input_name_idx) {
  return impl_->UpdateInputName(input_name_idx);
}

bool OpDesc::UpdateOutputName(const std::map<std::string, uint32_t> output_name_idx) {
  return impl_->UpdateOutputName(output_name_idx);
}

std::function<graphStatus(Operator &)> OpDesc::GetInferFunc() const {
  return impl_->GetInferFunc();
}

void *OpDesc::GetTilingFuncInfo() const {
  return impl_->GetTilingFuncInfo();
}

void OpDesc::SetTilingFuncInfo(void *tiling_func_info) {
  impl_->SetTilingFuncInfo(tiling_func_info);
}

void *OpDesc::GetAtomicTilingFuncInfo() const {
  return impl_->GetAtomicTilingFuncInfo();
}

void OpDesc::SetAtomicTilingFuncInfo(void *atomic_tiling_func_info) {
  impl_->SetAtomicTilingFuncInfo(atomic_tiling_func_info);
}

std::function<graphStatus(Operator &)> OpDesc::GetVerifyFunc() const {
  return impl_->GetVerifyFunc();
}

std::function<graphStatus(Operator &)> OpDesc::GetInferFormatFunc() const {
  return impl_->GetInferFormatFunc();
}

std::function<graphStatus(Operator &)> OpDesc::GetInferDataSliceFunc() const {
  return impl_->GetInferDataSliceFunc();
}

std::function<graphStatus(Operator &)> OpDesc::GetInferValueRangeFunc() const {
  return impl_->GetInferValueRangeFunc();
}

void OpDesc::AddInferFunc(const std::function<graphStatus(Operator &)> &func) {
  impl_->AddInferFunc(func);
}

void OpDesc::AddInferFormatFunc(const std::function<graphStatus(Operator &)> &func) {
  impl_->AddInferFormatFunc(func);
}

void OpDesc::AddVerifierFunc(const std::function<graphStatus(Operator &)> &func) {
  impl_->AddVerifierFunc(func);
}

void OpDesc::AddInferValueRangeFunc(const std::function<graphStatus(Operator &)> &func) {
  impl_->AddInferValueRangeFunc(func);
}

void OpDesc::AddInferDataSliceFunc(const std::function<graphStatus(Operator &)> &func) {
  impl_->AddInferDataSliceFunc(func);
}

graphStatus OpDesc::DefaultInferFormat() {
  return impl_->DefaultInferFormat(shared_from_this());
}

graphStatus OpDesc::CommonVerify() const {
  for (const std::string &iname : GetAllInputNames()) {
    // Checking shape of all inputs
    GE_CHK_BOOL_RET_STATUS_NOLOG(GetInputDescPtr(iname) != nullptr, GRAPH_FAILED);
    const std::vector<int64_t> ishape = GetInputDescPtr(iname)->GetShape().GetDims();
    if (ishape == DUMMY_SHAPE) {
      continue;
    }
    for (const int64_t dim : ishape) {
      GE_ASSERT_TRUE(dim >= UNKNOWN_DIM_NUM,
                     "Op[%s]'s input %s shape contains invalid dimension, it must >= -2, but it is :%" PRId64,
                     GetName().c_str(), iname.c_str(), dim);
    }
  }
  // Check all attributes defined
  const auto &all_attributes = GetAllAttrs();
  for (const auto &name : GetAllAttrNames()) {
    GE_ASSERT_TRUE(all_attributes.find(name) != all_attributes.end(), "operator attribute %s is empty.", name.c_str());
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetInputNameByIndex(const uint32_t index) const {
  return impl_->GetInputNameByIndex(index);
}

int32_t OpDesc::GetInputIndexByName(const std::string &name) const {
  return impl_->GetInputIndexByName(name);
}

graphStatus OpDesc::GetDynamicInputIndexesByName(const std::string &name, std::vector<int32_t> &indexes) const {
  return impl_->GetDynamicInputIndexesByName(name, indexes);
}

std::string OpDesc::GetValidInputNameByIndex(const uint32_t index) const {
  return impl_->GetValidInputNameByIndex(index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetOutputNameByIndex(const uint32_t index) const {
  return impl_->GetOutputNameByIndex(index);
}

int32_t OpDesc::GetOutputIndexByName(const std::string &name) const {
  return impl_->GetOutputIndexByName(name);
}

graphStatus OpDesc::GetDynamicOutputIndexesByName(const std::string &name, std::vector<int32_t> &indexes) const {
  return impl_->GetDynamicOutputIndexesByName(name, indexes);
}

ProtoAttrMap &OpDesc::MutableAttrMap() {
  return impl_->MutableAttrMap();
}

ConstProtoAttrMap &OpDesc::GetAttrMap() const {
  return impl_->GetAttrMap();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetId(const int64_t id) {
  impl_->SetId(id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY int64_t OpDesc::GetId() const {
  return impl_->GetId();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetStreamId(const int64_t stream_id) {
  impl_->SetStreamId(stream_id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY int64_t OpDesc::GetStreamId() const {
  return impl_->GetStreamId();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetAttachedStreamId(const int64_t stream_id) {
  const auto ret = AttrUtils::SetInt(this, ATTR_NAME_ATTACHED_STREAM_ID, stream_id);
  if (!ret) {
    GELOGW("[Set][Attr] %s to op failed.", ATTR_NAME_ATTACHED_STREAM_ID.c_str());
  }

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   ATTR_NAME_ATTACHED_STREAM_ID, "", "", stream_id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY int64_t OpDesc::GetAttachedStreamId() const {
  int64_t attached_stream_id = -1;  // default invalid value
  (void)AttrUtils::GetInt(this, ATTR_NAME_ATTACHED_STREAM_ID, attached_stream_id);
  return attached_stream_id;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetAttachedStreamIds(
    const std::vector<int64_t> &stream_ids) {
  std::vector<NamedAttrs> attached_stream_infos;
  if (!AttrUtils::GetListNamedAttrs(this, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attached_stream_infos)) {
    // 兼容老的方案,后续全部切换到ATTR_NAME_ATTACHED_STREAM_INFO_LIST属性设置上，不再提供Set/GetAttachedStreamId接口
    GELOGW("[Get][Attr] %s to op failed.", ATTR_NAME_ATTACHED_STREAM_INFO_LIST.c_str());
    if (stream_ids.size() == 1U) {
      (void)SetAttachedStreamId(stream_ids[0U]);
    } else {
      GELOGW("SetAttachedStreamId only supports stream size 1 but got %zu", stream_ids.size());
    }
  }
  if (stream_ids.size() != attached_stream_infos.size()) {
    GELOGW("stream_ids size %zu is not equal to attr %s size %zu", stream_ids.size(),
           ATTR_NAME_ATTACHED_STREAM_INFO_LIST.c_str(), attached_stream_infos.size());
    return;
  }
  for (size_t i = 0U; i < stream_ids.size(); ++i) {
    if (!AttrUtils::SetInt(attached_stream_infos[i], ATTR_NAME_ATTACHED_RESOURCE_ID, stream_ids[i])) {
      GELOGW("[Set][Attr] %s failed, index = %zu.", ATTR_NAME_ATTACHED_STREAM_ID.c_str(), i);
    }
    TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                     ATTR_NAME_ATTACHED_RESOURCE_ID, "", "", stream_ids[i]);
  }
  if (!AttrUtils::SetListNamedAttrs(this, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attached_stream_infos)) {
    GELOGW("[Set][Attr] %s failed, op %s", ATTR_NAME_ATTACHED_STREAM_INFO_LIST.c_str(), GetName().c_str());
    return;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<int64_t> OpDesc::GetAttachedStreamIds() const {
  std::vector<NamedAttrs> attached_stream_infos;
  if (!AttrUtils::GetListNamedAttrs(this, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attached_stream_infos)) {
    // 兼容老的方案
    GELOGD("[Get][Attr] %s to op %s failed.", ATTR_NAME_ATTACHED_STREAM_INFO_LIST.c_str(), GetNamePtr());
    if (GetAttachedStreamId() != -1) {
      return {GetAttachedStreamId()};
    }
    return {};
  }
  std::vector<int64_t> stream_ids;
  for (const auto &attached_stream_info : attached_stream_infos) {
    int64_t stream_id = -1;  // 默认值-1
    if (!AttrUtils::GetInt(attached_stream_info, ATTR_NAME_ATTACHED_RESOURCE_ID, stream_id)) {
      GELOGW("[Get][Attr] %s from op %s's attr map %s failed", ATTR_NAME_ATTACHED_RESOURCE_ID.c_str(), GetNamePtr(),
             ATTR_NAME_ATTACHED_STREAM_INFO_LIST.c_str());
    }
    stream_ids.emplace_back(stream_id);
    TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                     ATTR_NAME_ATTACHED_RESOURCE_ID, "", "", stream_ids.back());
  }
  return stream_ids;
}
// 需要兼容已有的GetAttachedStreamId接口，对于新的接口满足：
// 存在某个attached_streams_id不等于-1即可
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::HasValidAttachedStreamId() const {
  const auto &attached_streams_ids = GetAttachedStreamIds();
  return !attached_streams_ids.empty() &&
         std::any_of(attached_streams_ids.begin(), attached_streams_ids.end(),
                     [](const int64_t attached_stream_id) { return attached_stream_id != -1; });
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetInputName(const std::vector<std::string> &input_name) {
  impl_->SetInputName(input_name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<std::string> OpDesc::GetInputName() const {
  return impl_->GetInputName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetSrcName(const std::vector<std::string> &src_name) {
  impl_->SetSrcName(src_name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<std::string> OpDesc::GetSrcName() const {
  return impl_->GetSrcName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetSrcIndex(const std::vector<int64_t> &src_index) {
  impl_->SetSrcIndex(src_index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<int64_t> OpDesc::GetSrcIndex() const {
  return impl_->GetSrcIndex();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetInputOffset(const std::vector<int64_t> &input) {
  impl_->SetInputOffset(input);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<int64_t> OpDesc::GetInputOffset() const {
  return impl_->GetInputOffset();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetOutputOffset(const std::vector<int64_t> &output) {
  impl_->SetOutputOffset(output);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<int64_t> OpDesc::GetOutputOffset() const {
  return impl_->GetOutputOffset();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetDstName(const std::vector<std::string> &dst_name) {
  impl_->SetDstName(dst_name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<std::string> OpDesc::GetDstName() const {
  return impl_->GetDstName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetOpInferDepends(
    const std::vector<std::string> &depend_names) {
  const auto ret = AttrUtils::SetListStr(this, optiling::ATTR_NAME_OP_INFER_DEPENDS, depend_names);
  if (!ret) {
    GELOGE(GRAPH_FAILED, "[Set][Attr] op_infer_depends fail.");
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<std::string> OpDesc::GetOpInferDepends() const {
  std::vector<std::string> depend_names;
  (void)AttrUtils::GetListStr(this, optiling::ATTR_NAME_OP_INFER_DEPENDS, depend_names);
  return depend_names;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetDstIndex(const std::vector<int64_t> &dst_index) {
  impl_->SetDstIndex(dst_index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetWorkspace(const std::vector<int64_t> &workspace) {
  impl_->SetWorkspace(workspace);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<int64_t> OpDesc::GetWorkspace() const {
  return impl_->GetWorkspace();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetWorkspaceBytes(
    const std::vector<int64_t> &workspace_bytes) {
  impl_->SetWorkspaceBytes(workspace_bytes);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<int64_t> OpDesc::GetWorkspaceBytes() const {
  return impl_->GetWorkspaceBytes();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetIsInputConst(const std::vector<bool> &is_input_const) {
  impl_->SetIsInputConst(is_input_const);
  // If comes from ME,which is_input_const exist as attrs, outside no need to check GE_TRAIN flag
  const auto ret = AttrUtils::SetListBool(this, ATTR_NAME_IS_INPUT_CONST, is_input_const);
  if (!ret) {
    GELOGE(GRAPH_FAILED, "[Set][Attr] is_input_const fail.");
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<bool> OpDesc::GetIsInputConst() const {
  return impl_->GetIsInputConst();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetSubgraphInstanceName(const uint32_t index) const {
  return impl_->GetSubgraphInstanceName(static_cast<size_t>(index));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::vector<std::string> &OpDesc::GetSubgraphInstanceNames()
    const {
  return impl_->GetSubgraphInstanceNames();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::RemoveSubgraphInstanceName(const std::string &name) {
  impl_->RemoveSubgraphInstanceName(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::AddSubgraphName(const std::string &name) {
  return impl_->AddSubgraphName(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::map<std::string, uint32_t> &OpDesc::GetSubgraphNameIndexes()
    const {
  return impl_->GetSubgraphNameIndexes();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::SetSubgraphInstanceName(const uint32_t index,
                                                                                           const std::string &name) {
  return impl_->SetSubgraphInstanceName(static_cast<size_t>(index), name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::RegisterSubgraphIrName(const std::string &name,
                                                                                   const SubgraphType type) {
  impl_->MutableIRMeta().RegisterSubgraphIrName(name, type);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::map<std::string, SubgraphType> &OpDesc::GetSubgraphIrNames()
    const {
  return impl_->GetIRMeta().GetSubgraphIrNames();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::vector<std::pair<std::string, SubgraphType>> &
OpDesc::GetOrderedSubgraphIrNames() const {
  return impl_->GetIRMeta().GetOrderedSubgraphIrNames();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY SubgraphType
OpDesc::GetSubgraphTypeByIrName(const std::string &name) const {
  return impl_->GetIRMeta().GetSubgraphTypeByIrName(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDesc::GetSubgraphNameByInstanceName(const std::string &instance_name, std::string &subgraph_name) const {
  return impl_->GetSubgraphNameByInstanceName(instance_name, subgraph_name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::AppendIrAttrName(const std::string &name) {
  return impl_->MutableIRMeta().AppendIrAttrName(name);
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::vector<std::string> &OpDesc::GetIrAttrNames() const {
  return impl_->GetIRMeta().GetIrAttrNames();
}
void OpDesc::AppendIrInput(std::string name, IrInputType input_type) {
  impl_->MutableIRMeta().AppendIrInput(std::move(name), input_type);
}
const std::vector<std::pair<std::string, IrInputType>> &OpDesc::GetIrInputs() const {
  return impl_->GetIRMeta().GetIrInputs();
}

void OpDesc::SetInputDtypeSymbol(const std::string &ir_input, IrInputType type, const std::string &sym_id) {
  impl_->MutableIRMeta().MutableIRDataTypeSymbolStore().SetInputSymbol(ir_input, type, sym_id);
}
void OpDesc::SetOutputDtypeSymbol(const std::string &ir_output, IrOutputType type, const std::string &sym_id) {
  impl_->MutableIRMeta().MutableIRDataTypeSymbolStore().SetOutputSymbol(ir_output, type, sym_id);
}
void OpDesc::DeclareDtypeSymbol(const std::string &sym_id, const TensorType &type) {
  impl_->MutableIRMeta().MutableIRDataTypeSymbolStore().DeclareSymbol(sym_id, type);
}
void OpDesc::DeclareDtypeSymbol(const std::string &sym_id, const ListTensorType &type) {
  impl_->MutableIRMeta().MutableIRDataTypeSymbolStore().DeclareSymbol(sym_id, type);
}
void OpDesc::DeclareDtypeSymbol(const std::string &sym_id, const Promote &type) {
  impl_->MutableIRMeta().MutableIRDataTypeSymbolStore().DeclareSymbol(sym_id, type);
}
void OpDesc::ShareDtypeSymbolsFrom(const OpDesc &src) {
  impl_->MutableIRMeta().MutableIRDataTypeSymbolStore() = src.impl_->GetIRMeta().GetIRDataTypeSymbolStore();
}
bool OpDesc::IsSupportSymbolicInferDataType() const {
  return impl_->IsSupportSymbolicInferDataType();
}
graphStatus OpDesc::SymbolicInferDataType() {
  return impl_->SymbolicInferDataType(shared_from_this());
}

void OpDesc::AppendIrOutput(std::string name, IrOutputType output_type) {
  impl_->MutableIRMeta().AppendIrOutput(name, output_type);
}
const std::vector<std::pair<std::string, IrOutputType>> &OpDesc::GetIrOutputs() const {
  return impl_->GetIRMeta().GetIrOutputs();
}
size_t OpDesc::GetIrInputsSize() const {
  return impl_->GetIrInputsSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddInput(const std::string &name) {
  inputs_.emplace_back(std::make_pair(name, GeTensorDesc()));
  return *this;
}
graphStatus OpDesc::GetPromoteIrInputList(std::vector<std::vector<size_t>> &promote_index_list) {
  return impl_->MutableIRMeta().MutableIRDataTypeSymbolStore().GetPromoteIrInputList(promote_index_list);
}

std::map<uint32_t, std::string> OpDesc::GetAllOutputIndexToName() {
  return impl_->GetAllOutputIndexToName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddInput(const std::string &name,
                                                                                      const GeTensorDesc &tensor) {
  inputs_.emplace_back(std::make_pair(name, tensor));
  return *this;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddDynamicInput(const std::string &name,
                                                                                             const uint32_t num) {
  for (uint32_t i = 0U; i < num; i++) {
    inputs_.emplace_back(std::make_pair(name + std::to_string(i), GeTensorDesc()));
  }
  return *this;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddDynamicInput(
    const std::string &name, const uint32_t num, const GeTensorDesc &tensor) {
  for (uint32_t i = 0U; i < num; i++) {
    inputs_.emplace_back(std::make_pair(name + std::to_string(i), tensor));
  }
  return *this;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddOutput(const std::string &name) {
  outputs_.emplace_back(std::make_pair(name, GeTensorDesc()));
  return *this;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddOutput(const std::string &name,
                                                                                       const GeTensorDesc &tensor) {
  outputs_.emplace_back(std::make_pair(name, tensor));
  return *this;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddDynamicOutput(const std::string &name,
                                                                                              const uint32_t num) {
  for (uint32_t i = 0U; i < num; i++) {
    outputs_.emplace_back(std::make_pair(name + std::to_string(i), GeTensorDesc()));
  }
  return *this;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddDynamicOutput(
    const std::string &name, const uint32_t num, const GeTensorDesc &tensor) {
  for (uint32_t i = 0U; i < num; i++) {
    outputs_.emplace_back(std::make_pair(name + std::to_string(i), tensor));
  }
  return *this;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr OpDescBuilder::Build() {
  const OpDescPtr op_desc = MakeShared<OpDesc>(name_, type_);
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create opdesc failed, name:%s, type:%s.", name_.c_str(), type_.c_str());
    GELOGE(GRAPH_FAILED, "[Create][OpDesc] failed, name:%s, type:%s.", name_.c_str(), type_.c_str());
    return nullptr;
  }
  for (auto &input : inputs_) {
    if (op_desc->AddInputDesc(input.first, input.second) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "AddInputDesc failed, op:%s.", name_.c_str());
      GELOGE(GRAPH_FAILED, "[Add][InputDesc] failed, op:%s.", name_.c_str());
      return nullptr;
    }
  }
  for (auto &output : outputs_) {
    if (op_desc->AddOutputDesc(output.first, output.second) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "AddOutputDesc failed, op:%s", name_.c_str());
      GELOGE(GRAPH_FAILED, "[Add][OutputDesc] failed, op:%s.", name_.c_str());
      return nullptr;
    }
  }
  return op_desc;
}
}  // namespace af
