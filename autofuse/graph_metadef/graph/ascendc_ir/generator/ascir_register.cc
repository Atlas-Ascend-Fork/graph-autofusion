/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph/ascendc_ir/ascir_register.h"
#include "graph/ascendc_ir/ascir_registry.h"
#include "graph/types_af.h"
namespace af {
namespace ascir {
AscirRegister::AscirRegister(const char *type, const char *def_file_path, int64_t line) : ir_def_{} {
  ir_def_.Init(type, def_file_path, line);
}

AscirRegister &AscirRegister::Inputs(std::vector<ge::AscendString> &&input_names) {
  for (const auto &input_name : input_names) {
    ir_def_.AppendInput(input_name.GetString(), IrInputType::kIrInputRequired);
  }
  return *this;
}

AscirRegister &AscirRegister::DynamicInput(const std::string &input_name) {
  ir_def_.AppendInput(input_name, IrInputType::kIrInputDynamic);
  return *this;
}

AscirRegister &AscirRegister::OptionalInput(const std::string &input_name) {
  ir_def_.AppendInput(input_name, IrInputType::kIrInputOptional);
  return *this;
}

AscirRegister &AscirRegister::OptionalInput(const char_t *input_name, const char_t *datatype_symbol) {
  ir_def_.AppendInput(input_name, IrInputType::kIrInputOptional);
  ir_def_.StoreInputIrSymName(input_name, datatype_symbol);
  ir_def_.MutableDataTypeSymbolStore().SetInputSymbol(input_name, kIrInputOptional, datatype_symbol);
  return *this;
}

AscirRegister &AscirRegister::Outputs(std::vector<ge::AscendString> &&output_names) {
  for (const auto &output_name : output_names) {
    ir_def_.AppendOutput(output_name.GetString(), IrOutputType::kIrOutputRequired);
  }
  return *this;
}

AscirRegister &AscirRegister::DynamicOutput(const std::string &output_name) {
  ir_def_.AppendOutput(output_name, IrOutputType::kIrOutputDynamic);
  return *this;
}

AscirRegister::AscirRegister(const AscirRegister &other) {
  AscirRegistry::GetInstance().RegisterAscIr(other.ir_def_.GetType(), other.ir_def_);
}
AscirRegister &AscirRegister::Attr(std::string name, std::string asc_type, std::string ge_type) {
  if (ir_def_.IsAttrExisted(name)) {
    return *this;
  }
  ir_def_.SetAttr(name, asc_type, ge_type);
  return *this;
}
AscirRegister &AscirRegister::StartNode() {
  ir_def_.StartNode();
  return *this;
}
AscirRegister &AscirRegister::InferDataType(AscIrDef::CodeGenerator infer_data_type_generator) {
  ir_def_.infer_data_type_generator = std::move(infer_data_type_generator);
  return *this;
}
AscirRegister &AscirRegister::InferView(AscIrDef::CodeGenerator infer_view_generator) {
  ir_def_.infer_view_generator = std::move(infer_view_generator);
  return *this;
}

AscirRegister &AscirRegister::Views(const std::vector<ViewPolicy> &views_policy) {
  ir_def_.SetViewPolicy(views_policy);
  return InferView(InferViewByPolicy);
}
AscirRegister &AscirRegister::DataTypes(const std::vector<DtypePolicy> &data_types_policy) {
  ir_def_.SetDtypePolicy(data_types_policy);
  return InferDataType(InferDtypeByPolicy);
}

AscirRegister &AscirRegister::Input(const char_t *input_name, const char_t *datatype_symbol) {
  ir_def_.AppendInput(input_name, IrInputType::kIrInputRequired);
  ir_def_.StoreInputIrSymName(input_name, datatype_symbol);
  ir_def_.MutableDataTypeSymbolStore().SetInputSymbol(input_name, kIrInputRequired, datatype_symbol);
  return *this;
}
AscirRegister &AscirRegister::Output(const char_t *output_name, const char_t *datatype_symbol) {
  ir_def_.AppendOutput(output_name, IrOutputType::kIrOutputRequired);
  ir_def_.StoreOutputIrSymName(output_name, datatype_symbol);
  ir_def_.MutableDataTypeSymbolStore().SetOutputSymbol(output_name, kIrOutputRequired, datatype_symbol);
  return *this;
}

AscirRegister &AscirRegister::DynamicInput(const char_t *input_name, const char_t *datatype_symbol) {
  ir_def_.AppendInput(input_name, IrInputType::kIrInputDynamic);
  ir_def_.StoreInputIrSymName(input_name, datatype_symbol);
  ir_def_.MutableDataTypeSymbolStore().SetInputSymbol(input_name, kIrInputDynamic, datatype_symbol);
  return *this;
}

AscirRegister &AscirRegister::DynamicOutput(const char_t *output_name, const char_t *datatype_symbol) {
  ir_def_.AppendOutput(output_name, IrOutputType::kIrOutputDynamic);
  ir_def_.StoreOutputIrSymName(output_name, datatype_symbol);
  ir_def_.MutableDataTypeSymbolStore().SetOutputSymbol(output_name, kIrOutputDynamic, datatype_symbol);
  return *this;
}

AscirRegister &AscirRegister::SameTmpBufSizeFromFirstInput() {
  ir_def_.SetCalcTmpBufSizeFunc("SameTmpBufSizeWithFirstInput", CalcTmpBufSizeFuncType::CommonType);
  return *this;
}

AscirRegister &AscirRegister::Impl(const std::vector<std::string> &soc_version, const AscIrImpl &impl) {
  ir_def_.AddSocImpl(soc_version, impl);
  return *this;
}

AscirRegister &AscirRegister::Impl(const std::vector<std::string> &soc_version, const AscIrImplV2 &impl) {
  ir_def_.AddSocImplV2(soc_version, impl);
  return *this;
}

size_t AscirRegister::GetSocImplSize() const {
  return ir_def_.GetSocImplSize();
}

template <>
AscirRegister &AscirRegister::Attr<float>(ge::AscendString &&name) {
  return Attr(name.GetString(), "float", "Float");
}

template <>
AscirRegister &AscirRegister::Attr<bool>(ge::AscendString &&name) {
  return Attr(name.GetString(), "bool", "Bool");
}

template <>
AscirRegister &AscirRegister::Attr<ge::DataType>(ge::AscendString &&name) {
  return Attr(name.GetString(), "ge::DataType", "Int");
}
template <>
AscirRegister &AscirRegister::Attr<Tensor>(ge::AscendString &&name) {
  return Attr(name.GetString(), "Tensor", "Tensor");
}
template <>
AscirRegister &AscirRegister::Attr<std::string>(ge::AscendString &&name) {
  return Attr(name.GetString(), "std::string", "String");
}
template <>
AscirRegister &AscirRegister::Attr<int64_t>(ge::AscendString &&name) {
  return Attr(name.GetString(), "int64_t", "Int");
}
template <>
AscirRegister &AscirRegister::Attr<std::vector<int64_t>>(ge::AscendString &&name) {
  return Attr(name.GetString(), "std::vector<int64_t>", "ListInt");
}
template <>
AscirRegister &AscirRegister::Attr<std::vector<std::vector<int64_t>>>(ge::AscendString &&name) {
  return Attr(name.GetString(), "std::vector<std::vector<int64_t>>", "ListListInt");
}
template <>
AscirRegister &AscirRegister::Attr<ge::Format>(ge::AscendString &&name) {
  return Attr(name.GetString(), "ge::Format", "Int");
}
template <>
AscirRegister &AscirRegister::Attr<Expression>(ge::AscendString &&name) {
  return Attr(name.GetString(), "Expression", "Expression");
}

AscirRegister &AscirRegister::ComputeType(af::ComputeType compute_type) {
  ir_def_.SetComputeType(compute_type);
  return *this;
}

AscirRegister &AscirRegister::Comment(const string &comment) {
  ir_def_.SetComment(comment);
  return *this;
}
}  // namespace ascir
}  // namespace af
