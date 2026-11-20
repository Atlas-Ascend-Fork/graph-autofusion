/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "graph/ascendc_ir/ascir_registry.h"
namespace af {
namespace ascir {
struct AscIrDefImpl {
 public:
  std::unique_ptr<AscIrAtt> GetAscIrAttImpl(const std::string &soc_version) {
    auto impl = soc_2_impl_.find(soc_version);
    if (impl != soc_2_impl_.end()) {
      return (impl->second.att == nullptr) ? nullptr : impl->second.att();
    }

    auto impl_v2 = soc_2_impl_v2_.find(soc_version);
    if (impl_v2 != soc_2_impl_v2_.end()) {
      return (impl_v2->second.att == nullptr) ? nullptr : impl_v2->second.att();
    }
    return nullptr;
  }
  std::unique_ptr<AscIrCodegen> GetAscIrCodegenImpl(const std::string &soc_version) {
    auto impl = soc_2_impl_.find(soc_version);
    if (impl != soc_2_impl_.end()) {
      return (impl->second.codegen == nullptr) ? nullptr : impl->second.codegen();
    }

    auto impl_v2 = soc_2_impl_v2_.find(soc_version);
    if (impl_v2 != soc_2_impl_v2_.end()) {
      return (impl_v2->second.codegen == nullptr) ? nullptr : impl_v2->second.codegen();
    }

    return nullptr;
  }

  std::string file_path;
  int64_t line{};
  std::string type;
  std::vector<std::pair<std::string, IrInputType>> input_defs;
  std::vector<std::pair<std::string, IrOutputType>> output_defs;
  std::unordered_map<std::string, std::string> input_name_to_sym_name;
  std::unordered_map<std::string, std::string> output_name_to_sym_name;
  std::vector<AscIrAttrDef> attr_defs;

  std::vector<ViewPolicy> output_views_policy;
  std::vector<DtypePolicy> output_dtypes_policy;

  bool start_node{false};
  // TODO 整改后删除
  IRDataTypeSymbolStore dtype_symbol_store;
  std::string comment;
  CalcTmpBufSizeFunc calc_tmp_buf_size_func;
  std::string tiling_data_name;
  std::map<std::string, AscIrImpl> soc_2_impl_;
  std::map<std::string, AscIrImplV2> soc_2_impl_v2_;
  std::map<std::string, IRDataTypeSymbolStore> soc_2_dtype_sym_store_;
  ComputeType compute_type = ComputeType::kComputeInvalid;
};

AscIrDef::AscIrDef() {
  impl_ = std::make_shared<AscIrDefImpl>();
}

bool AscIrDef::IsAttrExisted(const std::string &attr_name) const {
  return std::find_if(impl_->attr_defs.begin(), impl_->attr_defs.end(),
                      [&attr_name](const AscIrAttrDef &asc_ir_attr_def) {
                        return asc_ir_attr_def.name == attr_name;
                      }) != impl_->attr_defs.end();
}

void AscIrDef::Init(const char *type, const char *def_file_path, int64_t line) const {
  impl_->type = type;
  impl_->file_path = def_file_path;
  impl_->line = line;
  impl_->start_node = false;
}

const std::vector<std::pair<std::string, IrInputType>> &AscIrDef::GetInputDefs() const {
  return impl_->input_defs;
}

const std::string &AscIrDef::GetInputIrSymName(const std::string &ir_name) const {
  static const std::string kEmptySymbol;
  const auto iter = impl_->input_name_to_sym_name.find(ir_name);
  return iter == impl_->input_name_to_sym_name.end() ? kEmptySymbol : iter->second;
}

const std::vector<std::pair<std::string, IrOutputType>> &AscIrDef::GetOutputDefs() const {
  return impl_->output_defs;
}

bool AscIrDef::HasDynamicOutput() const {
  for (auto &def : impl_->output_defs) {
    if (def.second == IrOutputType::kIrOutputDynamic) {
      return true;
    }
  }

  return false;
}

void AscIrDef::AppendInput(const std::string &name, IrInputType type) const {
  impl_->input_defs.emplace_back(name, type);
}
void AscIrDef::AppendOutput(const std::string &name, IrOutputType type) const {
  impl_->output_defs.emplace_back(name, type);
}

void AscIrDef::StoreInputIrSymName(const std::string &ir_name, const std::string &sym_name) const {
  impl_->input_name_to_sym_name[ir_name] = sym_name;
}
void AscIrDef::StoreOutputIrSymName(const std::string &ir_name, const std::string &sym_name) const {
  impl_->output_name_to_sym_name[ir_name] = sym_name;
}

const std::string &AscIrDef::GetType() const {
  return impl_->type;
}
void AscIrDef::StartNode() const {
  impl_->start_node = true;
}

bool AscIrDef::IsStartNode() const {
  return impl_->start_node;
}

void AscIrDef::SetAttr(const std::string &name, const std::string &asc_type, const std::string &ge_type) const {
  impl_->attr_defs.emplace_back(AscIrAttrDef{name, asc_type, ge_type});
}

void AscIrDef::SetDtypePolicy(const std::vector<DtypePolicy> &output_dtypes_policy) const {
  impl_->output_dtypes_policy = output_dtypes_policy;
}

const std::vector<DtypePolicy> &AscIrDef::GetOutputDtypePolicy() const {
  return impl_->output_dtypes_policy;
}

void AscIrDef::SetViewPolicy(const std::vector<ViewPolicy> &view_policy) const {
  impl_->output_views_policy = view_policy;
}

const std::vector<ViewPolicy> &AscIrDef::GetViewPolicy() const {
  return impl_->output_views_policy;
}

void AscIrDef::SetApiTilingDataName(const std::string &tiling_data_name) const {
  if (!impl_->tiling_data_name.empty()) {
    GELOGE(af::FAILED, "%s has registered tiling data: %s", impl_->type.c_str(), impl_->tiling_data_name.c_str());
    return;
  }
  impl_->tiling_data_name = tiling_data_name;
}

const string &AscIrDef::GetApiTilingDataName() const {
  return impl_->tiling_data_name;
}

void AscIrDef::SetCalcTmpBufSizeFunc(const std::string &calc_tmp_buf_size_func, CalcTmpBufSizeFuncType type) const {
  if (!impl_->calc_tmp_buf_size_func.func_name.empty()) {
    GELOGE(af::FAILED, "has registered calc_tmp_buf_size_func: %s", impl_->calc_tmp_buf_size_func.func_name.c_str());
    return;
  }
  impl_->calc_tmp_buf_size_func = CalcTmpBufSizeFunc{calc_tmp_buf_size_func, type};
}

const CalcTmpBufSizeFunc &AscIrDef::GetCalcTmpBufSizeFunc() const {
  return impl_->calc_tmp_buf_size_func;
}

const std::vector<AscIrAttrDef> &AscIrDef::GetAttrDefs() const {
  return impl_->attr_defs;
}

std::vector<AscIrAttrDef> &AscIrDef::MutableAttrDefs() const {
  return impl_->attr_defs;
}

void AscIrDef::SetComment(const string &comment) const {
  impl_->comment = comment;
}

const std::string &AscIrDef::GetComment() const {
  return impl_->comment;
}

const std::string &AscIrDef::GetFilePath() const {
  return impl_->file_path;
}

int64_t AscIrDef::GetLine() const {
  return impl_->line;
}

IRDataTypeSymbolStore &AscIrDef::MutableDataTypeSymbolStore() const {
  return impl_->dtype_symbol_store;
}

const IRDataTypeSymbolStore &AscIrDef::GetDataTypeSymbolStore() const {
  return impl_->dtype_symbol_store;
}

const std::map<std::string, IRDataTypeSymbolStore> &AscIrDef::GetSocToDataTypeSymbolStore() const {
  return impl_->soc_2_dtype_sym_store_;
}

void AscIrDef::AddSocImpl(const std::vector<std::string> &soc_versions, const AscIrImpl &impl) const {
  for (const auto &soc : soc_versions) {
    impl_->soc_2_impl_[soc] = impl;
    auto &dtype_sym_store = impl_->soc_2_dtype_sym_store_[soc];
    // reg_sym
    for (const auto &input_def : impl_->input_defs) {
      dtype_sym_store.SetInputSymbol(input_def.first, input_def.second, impl_->input_name_to_sym_name[input_def.first]);
    }
    for (const auto &output_def : impl_->output_defs) {
      dtype_sym_store.SetOutputSymbol(output_def.first, output_def.second,
                                      impl_->output_name_to_sym_name[output_def.first]);
    }
    // bind symbol to dtypes
    for (const auto &iter : impl.support_dtypes) {
      (void)dtype_sym_store.DeclareSymbol(iter.first, iter.second);
    }
  }
}

void AscIrDef::AddSocImplV2(const std::vector<std::string> &soc_versions, const AscIrImplV2 &impl) const {
  for (const auto &soc : soc_versions) {
    impl_->soc_2_impl_v2_[soc] = impl;
    auto &dtype_sym_store = impl_->soc_2_dtype_sym_store_[soc];
    // reg_sym
    for (const auto &input_def : impl_->input_defs) {
      dtype_sym_store.SetInputSymbol(input_def.first, input_def.second, impl_->input_name_to_sym_name[input_def.first]);
    }
    for (const auto &output_def : impl_->output_defs) {
      dtype_sym_store.SetOutputSymbol(output_def.first, output_def.second,
                                      impl_->output_name_to_sym_name[output_def.first]);
    }
    // bind symbol to dtypes
    for (const auto &iter : impl.support_dtypes) {
      (void)dtype_sym_store.DeclareSymbol(iter.first, iter.second);
    }
  }
}

void AscIrDef::AppendSocImpl(const AscIrDef &ir_def) const {
  impl_->soc_2_impl_.insert(ir_def.impl_->soc_2_impl_.begin(), ir_def.impl_->soc_2_impl_.end());
  impl_->soc_2_impl_v2_.insert(ir_def.impl_->soc_2_impl_v2_.begin(), ir_def.impl_->soc_2_impl_v2_.end());
  impl_->soc_2_dtype_sym_store_.insert(ir_def.impl_->soc_2_dtype_sym_store_.begin(),
                                       ir_def.impl_->soc_2_dtype_sym_store_.end());
}

size_t AscIrDef::GetSocImplSize() const {
  return impl_->soc_2_impl_.size() + impl_->soc_2_impl_v2_.size();
}

std::unique_ptr<AscIrAtt> AscIrDef::GetAscIrAttImpl(const std::string &soc_version) {
  return impl_->GetAscIrAttImpl(soc_version);
}
std::unique_ptr<AscIrCodegen> AscIrDef::GetAscIrCodegenImpl(const std::string &soc_version) {
  return impl_->GetAscIrCodegenImpl(soc_version);
}

AscirRegistry &AscirRegistry::GetInstance() {
  static AscirRegistry registry;
  return registry;
}
void AscirRegistry::RegisterAscIr(const std::string &type, const AscIrDef &def) {
  auto iter = types_to_ascir_.find(type);
  if (iter == types_to_ascir_.end()) {
    types_to_ascir_[type] = def;
  } else {
    iter->second.AppendSocImpl(def);
  }
}
const std::unordered_map<std::string, AscIrDef> &AscirRegistry::GetAll() const {
  return types_to_ascir_;
}

std::unique_ptr<AscIrAtt> AscirRegistry::GetIrAttImpl(const std::string &soc_version, const std::string &type) {
  auto iter = types_to_ascir_.find(type);
  return (iter == types_to_ascir_.end()) ? nullptr : types_to_ascir_[type].GetAscIrAttImpl(soc_version);
}
std::unique_ptr<AscIrCodegen> AscirRegistry::GetIrCodegenImpl(const std::string &soc_version, const std::string &type) {
  auto iter = types_to_ascir_.find(type);
  return (iter == types_to_ascir_.end()) ? nullptr : types_to_ascir_[type].GetAscIrCodegenImpl(soc_version);
}

void AscIrDef::SetComputeType(ComputeType compute_type) const {
  impl_->compute_type = compute_type;
}

ComputeType AscIrDef::GetComputeType() const {
  return impl_->compute_type;
}

void AscirRegistry::ClearAll() {
  types_to_ascir_.clear();
};

}  // namespace ascir
}  // namespace af
