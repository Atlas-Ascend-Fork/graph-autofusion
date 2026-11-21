/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_ASCIR_REGISTER_H
#define AUTOFUSE_ASCIR_REGISTER_H
#include <string>
#include <vector>
#include "graph/ascendc_ir/ascir_registry.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "graph/ascend_string_af.h"
#include "graph/symbolizer/symbolic.h"

namespace af {
namespace ascir {
#ifdef AUTOFUSE_USE_GE_METADEF
using char_t = ge::char_t;
#endif
class AscirRegister {
 public:
  AscirRegister() = default;
  AscirRegister(const char *type, const char *def_file_path, int64_t line);
  AscirRegister &Inputs(std::vector<ge::AscendString> &&input_names);
  AscirRegister &Input(const char_t *input_name, const char_t *datatype_symbol);
  AscirRegister &DynamicInput(const std::string &input_name);
  AscirRegister &DynamicInput(const char_t *input_name, const char_t *datatype_symbol);
  AscirRegister &OptionalInput(const std::string &input_name);
  AscirRegister &OptionalInput(const char_t *input_name, const char_t *datatype_symbol);
  AscirRegister &Outputs(std::vector<ge::AscendString> &&output_names);
  AscirRegister &Output(const char_t *output_name, const char_t *datatype_symbol);
  AscirRegister &DynamicOutput(const std::string &output_name);
  AscirRegister &DynamicOutput(const char_t *output_name, const char_t *datatype_symbol);
  AscirRegister &ComputeType(ComputeType compute_type);
  AscirRegister &Comment(const std::string &comment);

  template <class T>
  AscirRegister &Attr(ge::AscendString &&name);

  AscirRegister &InferDataType(AscIrDef::CodeGenerator infer_data_type_generator);
  AscirRegister &UseFirstInputDataType() {
    const auto &output_defs = ir_def_.GetOutputDefs();
    return DataTypes(std::vector<DtypePolicy>(output_defs.size(), DtypePolicy(0U)));
  }
  AscirRegister &UseSecondInputDataType() {
    const auto &output_defs = ir_def_.GetOutputDefs();
    return DataTypes(std::vector<DtypePolicy>(output_defs.size(), DtypePolicy(1U)));
  }

  AscirRegister &InferView(AscIrDef::CodeGenerator infer_view_generator);
  AscirRegister &UseFirstInputView() {
    const auto &output_defs = ir_def_.GetOutputDefs();
    return Views(std::vector<ViewPolicy>(output_defs.size(), ViewPolicy(0)));
  }

  AscirRegister &StartNode();
  AscirRegister &Views(const std::vector<ViewPolicy> &views_policy);
  AscirRegister &DataTypes(const std::vector<DtypePolicy> &data_types_policy);
  AscirRegister(const AscirRegister &other);
  AscirRegister &operator=(const AscirRegister &) = delete;

  AscirRegister(AscirRegister &&) noexcept = delete;
  AscirRegister &operator=(AscirRegister &&) noexcept = delete;

  AscirRegister &SameTmpBufSizeFromFirstInput();

  AscirRegister &Impl(const std::vector<std::string> &soc_version, const AscIrImpl &impl);

  AscirRegister &Impl(const std::vector<std::string> &soc_version, const AscIrImplV2 &impl);

  size_t GetSocImplSize() const;

 private:
  AscirRegister &Attr(std::string name, std::string asc_type, std::string ge_type);

 private:
  AscIrDef ir_def_;
};

#define REG_ASC_IR(type) static auto g_register_##type = af::ascir::AscirRegister(#type, __FILE__, __LINE__)
#define REG_ASC_IR_START_NODE(type) REG_ASC_IR(type).Inputs({}).Outputs({"y"}).StartNode()
#define REG_ASC_IR_START_NODE_WITH_ATTR(type) \
  REG_ASC_IR(type).Inputs({}).Outputs({"y"}).Attr<int64_t>("index").StartNode()

#define EXPAND_CHAIN_CALL(...) #__VA_ARGS__
#define REG_ASC_IR_WITH_COMMENT(type, ...)                                           \
  constexpr const char *comment_##type =                                             \
      R"COMMENT(REG_ASC_IR()COMMENT" #type ")\n" EXPAND_CHAIN_CALL(__VA_ARGS__) ";"; \
  static auto g_register_##type = AscirRegister(#type, __FILE__, __LINE__) __VA_ARGS__.Comment(comment_##type)
#define EXPORT_GENERATOR()
}  // namespace ascir
}  // namespace af

#endif  // AUTOFUSE_ASCIR_REGISTER_H
