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
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <cctype>

#include "graph/debug/ge_op_types.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/ascendc_ir/ascir_registry.h"
#include "graph/utils/type_utils.h"
namespace af {
namespace ascir {
namespace {
bool IsBoolCompatibleOutput(const std::set<DataType> &support_types) {
  return support_types.size() == 2U && support_types.count(DT_BOOL) > 0U && support_types.count(DT_UINT8) > 0U;
}

const char *GetPureFileName(const char *path) {
  const char *name = std::strrchr(path, '/');
  if (name == nullptr) {
    name = path;
  } else {
    ++name;
  }
  return name;
}
std::string CapitalizeFirstLetter(const std::string &input) {
  if (input.empty()) {
    return input;
  }

  std::string result = input;
  if (std::islower(result[0])) {
    result[0] = std::toupper(result[0]);
  }
  return result;
}

void GenIrAttrMemberFuncs(const std::vector<AscIrAttrDef> &attr_defs, std::stringstream &ss) {
  if (attr_defs.empty()) {
    return;
  }
  // 对每个属性生成对应的Set, Get函数
  for (const auto &attr_def : attr_defs) {
    ss << "    graphStatus Get" << CapitalizeFirstLetter(attr_def.name) << "(" << attr_def.asc_ir_type << "&"
       << "  " << attr_def.name << ") const {" << std::endl;
    ss << "      auto attr_value = attr_store_.GetAnyValue(\"" << attr_def.name << "\");" << std::endl;
    ss << "      GE_WARN_ASSERT(attr_value != nullptr);" << std::endl;
    ss << "      return attr_value->GetValue(" << attr_def.name << ");" << std::endl << "    }" << std::endl;

    ss << "    graphStatus Set" << CapitalizeFirstLetter(attr_def.name) << "(" << attr_def.asc_ir_type << " "
       << attr_def.name << ") {" << std::endl;
    ss << "      auto attr_value = attr_store_.GetOrCreateAnyValue(\"" << attr_def.name << "\");" << std::endl;
    ss << "      GE_ASSERT_NOTNULL(attr_value);" << std::endl;
    ss << "      return attr_value->SetValue(" << attr_def.name << ");" << std::endl << "    }" << std::endl;
  }
}

std::string TryGenIrAttrClass(const AscIrDef &def, std::stringstream &ss) {
  const auto &attr_defs = def.GetAttrDefs();
  if (attr_defs.empty()) {
    return ("");
  }
  const std::string &ir_type = def.GetType();
  std::string derived_class_name = std::string("Asc").append(ir_type).append("IrAttrDef");
  // 暂时没啥好的办法，data的类需要先定义好，gen出来的话有点晚了
  if (ir_type == DATA) {
    ss << "  using " << derived_class_name << " = af::" << derived_class_name << ";" << std::endl;
    ss << "  " << derived_class_name << " &ir_attr;" << std::endl;
    return derived_class_name;
  }
  // 生成子类的定义
  ss << "  struct " << derived_class_name << ": public af::AscIrAttrDefBase {" << std::endl;
  ss << "    ~" << derived_class_name << "() override = default;" << std::endl;
  GenIrAttrMemberFuncs(attr_defs, ss);
  ss << "  };" << std::endl;
  // 添加引用成员到上一级类
  ss << "  " << derived_class_name << " &ir_attr;" << std::endl;
  return derived_class_name;
};

std::string DataTypeToSerialString(const DataType data_type);
std::string TensorTypeToCode(const TensorType &tensor_type);
const IRDataTypeSymbolStore &GetFirstDataTypeSymbolStore(const AscIrDef &ir_def);

void GenIrInputAndOutputDef(const AscIrDef &def, std::stringstream &ss) {
  const auto &input_defs = def.GetInputDefs();
  const auto &symbols = GetFirstDataTypeSymbolStore(def).GetNamedSymbols();
  for (const auto &input_def : input_defs) {
    if (input_def.second == IrInputType::kIrInputDynamic) {
      ss << "    this->DynamicInputRegister(\"" << input_def.first << "\", 0U, true);" << std::endl;
    } else if (input_def.second == IrInputType::kIrInputOptional) {
      const auto &symbol_name = def.GetInputIrSymName(input_def.first);
      if (symbol_name.empty()) {
        ss << "    this->OptionalInputRegister(\"" << input_def.first << "\");" << std::endl;
      } else {
        ss << "    this->OptionalInputRegister(\"" << input_def.first << "\", \"" << symbol_name << "\");" << std::endl;
        const auto symbol = symbols.find(symbol_name);
        if ((symbol != symbols.end()) && (symbol->second != nullptr)) {
          ss << "    this->DataTypeRegister(\"" << symbol_name << "\", "
             << TensorTypeToCode(symbol->second->GetTensorType()) << ");" << std::endl;
        }
      }
    } else {
      ss << "    this->InputRegister(\"" << input_def.first << "\");" << std::endl;
    }
  }

  const auto &output_defs = def.GetOutputDefs();
  for (const auto &output_def : output_defs) {
    if (output_def.second == IrOutputType::kIrOutputDynamic) {
      ss << "    this->DynamicOutputRegister(\"" << output_def.first << "\", 0U, true);" << std::endl;
    } else {
      ss << "    this->OutputRegister(\"" << output_def.first << "\");" << std::endl;
    }
  }
}
void GenOutTensorInitDef(const AscIrDef &def, std::stringstream &ss) {
  const auto &output_defs = def.GetOutputDefs();
  for (const auto &output_def : output_defs) {
    if (output_def.second != IrOutputType::kIrOutputDynamic) {
      ss << "    " << output_def.first << ".TryInitTensorAttr();\n";
    }
  }
}
void GenConstructorDef(const AscIrDef &def, const std::string &attr_class, std::stringstream &ss,
                       const bool need_graph = false) {
  const std::string &ir_type = def.GetType();
  if (need_graph) {
    ss << "  inline " << ir_type << "(const char *name, af::AscGraph &graph) : af::Operator";
  } else {
    ss << "  inline " << ir_type << "(const char *name) : af::Operator";
  }
  if (attr_class.empty()) {
    ss << "(name, Type), attr(" << "*af::AscNodeAttr::Create(*this))";
  } else {
    ss << "(name, Type), attr(" << "*af::AscNodeAttr::Create<" << attr_class << ">(*this))";
    ss << ", ir_attr(dynamic_cast<" << attr_class << "&>(*(attr.ir_attr)))";
  }
  const auto &input_defs = def.GetInputDefs();
  for (const auto &input_def : input_defs) {
    ss << "," << std::endl << "    " << input_def.first << "(this)";
  }
  const auto &output_defs = def.GetOutputDefs();
  for (size_t i = 0UL; i < output_defs.size(); ++i) {
    if (output_defs[i].second != IrOutputType::kIrOutputDynamic) {
      ss << "," << std::endl << "    " << output_defs[i].first << "(this, " << i << ")";
    }
  }
  ss << " {" << std::endl;
  GenIrInputAndOutputDef(def, ss);
  GenOutTensorInitDef(def, ss);
  ss << "    this->attr.api.compute_type =  static_cast<af::ComputeType>(" << static_cast<int32_t>(def.GetComputeType())
     << ");" << std::endl;
  if (need_graph) {
    ss << "    graph.AddNode(*this) ;" << std::endl << "  }" << std::endl;
  } else {
    ss << "  }" << std::endl << std::endl;
  }
}

bool IsSocOrderedTensorListStore(const AscIrDef &ir_def) {
  const auto &soc_to_sym_store = ir_def.GetSocToDataTypeSymbolStore();
  if (!soc_to_sym_store.empty()) {
    return soc_to_sym_store.begin()->second.IsSupportOrderedSymbolicInferDtype();
  }
  return false;
}

std::string DataTypeToSerialString(const DataType data_type) {
  auto res = TypeUtils::DataTypeToSerialString(data_type);
  if (res == "DT_BFLOAT16") {  // 历史原因，DT_BF16的string表达是DT_BFLOAT16，所以我们需要特殊处理一下
    return "DT_BF16";
  }
  return res;
}

std::string TensorTypeToCode(const TensorType &tensor_type) {
  std::string s = "{";
  size_t index = 0U;
  for (const auto &dtype : tensor_type.tensor_type_impl_->GetMutableDateTypeSet()) {
    s += DataTypeToSerialString(dtype);
    if (index++ < (tensor_type.tensor_type_impl_->GetMutableDateTypeSet().size() - 1U)) {
      s += ", ";
    }
  }
  s += "}";
  return s;
}

// TODO 兼容新老接口
const IRDataTypeSymbolStore &GetFirstDataTypeSymbolStore(const AscIrDef &ir_def) {
  const auto &soc_to_sym_store = ir_def.GetSocToDataTypeSymbolStore();
  if (!soc_to_sym_store.empty()) {
    return soc_to_sym_store.begin()->second;
  }
  return ir_def.GetDataTypeSymbolStore();
}

class SymbolProcessor {
 public:
  explicit SymbolProcessor(const AscIrDef &def) : def_(def) {}
  Status ProcessSymbol(const std::pair<std::string, SymDtype *> &sym, std::stringstream &ss) {
    // 外部保证非空，不允许注册一个不带dtype的sym
    GE_ASSERT_TRUE(!(sym.second->GetTensorType().tensor_type_impl_->GetMutableDateTypeSet().empty()));
    // 只有输出持有的sym不在这里处理
    if (sym.second->GetIrInputIndexes().empty()) {
      return SUCCESS;
    }

    GenerateInputDtypeUniquenessCheck(sym.second, ss);
    GenerateTypeDefinition(sym, ss);
    GenerateDtypeValidation(sym, ss);
    return SUCCESS;
  }

  Status GenSocVersionCallFunc() {
    for (const auto &iter : def_.GetSocToDataTypeSymbolStore()) {
      for (const auto &sym : iter.second.GetNamedSymbols())
        name_to_soc_to_sym_dtype_[sym.first].emplace(iter.first, iter.second.GetNamedSymbols()[sym.first]);
    }
    return SUCCESS;
  }

  Status ProcessSymbolWithNoCheck(const std::pair<std::string, SymDtype *> &sym, std::stringstream &ss) {
    // 外部保证非空，不允许注册一个不带dtype的sym
    GE_ASSERT_TRUE(!(sym.second->GetTensorType().tensor_type_impl_->GetMutableDateTypeSet().empty()));
    // 只有输出持有的sym不在这里处理
    if (sym.second->GetIrInputIndexes().empty()) {
      return SUCCESS;
    }
    GenerateInputDtypeUniquenessCheck(sym.second, ss);
    return SUCCESS;
  }

 protected:
  void GenerateTypeDefinition(const std::pair<std::string, SymDtype *> &sym, std::stringstream &ss) {
    const std::string tensor_type_obj = "support_dtypes_of_sym_" + sym.first;
    auto iter = name_to_soc_to_sym_dtype_.find(sym.first);
    if (iter != name_to_soc_to_sym_dtype_.end()) {
      ss << "    std::set<ge::DataType> " << tensor_type_obj << ";" << std::endl;
      bool is_first = true;
      for (const auto &soc_to_sym : iter->second) {
        if (is_first) {
          ss << "    if (npu_arch == \"" << soc_to_sym.first << "\") {\n";
          is_first = false;
        } else {
          ss << "    } else if (npu_arch == \"" << soc_to_sym.first << "\") {\n";
        }
        ss << "      " << tensor_type_obj << " = " << TensorTypeToCode(soc_to_sym.second->GetTensorType()) << ";\n";
      }
      ss << "    } else {\n";
      ss << R"(      GELOGE(af::FAILED, "Unknown npu arch: %s", npu_arch.c_str());)" << std::endl;
      ss << "      return af::FAILED;\n";
      ss << "    }\n";
      return;
    }

    ss << "    const static std::set<ge::DataType> " << tensor_type_obj << " = "
       << TensorTypeToCode(sym.second->GetTensorType()) << ";\n";
  }

  static void GenerateDtypeValidation(const std::pair<std::string, SymDtype *> &sym, std::stringstream &ss) {
    // 共sym的已经都校验过一致了，所以这里可以用第一个来校验是否在支持范围内
    auto check_index = sym.second->GetIrInputIndexes().front();
    ss << "    GE_WARN_ASSERT("
       << "support_dtypes_of_sym_" << sym.first << ".find(input_dtypes[" << check_index
       << "]) != support_dtypes_of_sym_" << sym.first << ".end());" << std::endl;
  }

  void GenerateInputDtypeUniquenessCheck(const SymDtype *sym, std::stringstream &ss) {
    auto input_ir_indexes = sym->GetIrInputIndexes();
    // 小于2个，不需要比较
    if (input_ir_indexes.size() < 2U) {
      return;
    }
    for (size_t index = 0U; index < input_ir_indexes.size() - 1U; ++index) {
      ss << "    GE_WARN_ASSERT(" << input_arg_name_ << "[" << input_ir_indexes[index] << "] == " << input_arg_name_
         << "[" << input_ir_indexes[index + 1] << "]);" << std::endl;
    }
  }
  const AscIrDef &def_;
  std::string input_arg_name_{"input_dtypes"};
  std::map<std::string, std::map<string, SymDtype *>> name_to_soc_to_sym_dtype_;
};

static bool HasOnlyOptionalInputs(const AscIrDef &def) {
  const auto &input_defs = def.GetInputDefs();
  return !input_defs.empty() && std::all_of(input_defs.begin(), input_defs.end(), [](const auto &input_def) {
    return input_def.second == IrInputType::kIrInputOptional;
  });
}

static void AppendArgsSizeAssertion(std::stringstream &ss, const AscIrDef &def) {
  const auto &input_defs = def.GetInputDefs();
  const bool all_optional_inputs = HasOnlyOptionalInputs(def);
  if (!all_optional_inputs) {
    ss << "    GE_ASSERT_EQ(input_dtypes.size(), " << input_defs.size() << "U);" << std::endl;
    return;
  }

  ss << "    const auto input_dtype_count = input_dtypes.size();" << std::endl;
  ss << "    if (input_dtype_count != 0U && input_dtype_count != " << input_defs.size() << "U) {" << std::endl;
  ss << "      GELOGE(af::FAILED, \"Expected 0 or %zu input dtypes, but got %zu\", " << input_defs.size()
     << "U, input_dtype_count);" << std::endl;
  ss << "      return af::FAILED;" << std::endl;
  ss << "    }" << std::endl;
}

class OutputHandler {
 public:
  explicit OutputHandler(const AscIrDef &def) : def_(def) {}

  void GenerateOutputInference(std::stringstream &ss, int space_count = 6) {
    bool could_infer = true;
    std::stringstream warning_code;
    size_t out_index = 0U;
    for (const auto out_sym : GetFirstDataTypeSymbolStore(def_).GetOutSymbols()) {
      sym_2_ir_indexs_[out_sym].push_back(out_index);
      if (!(out_sym->GetIrInputIndexes().empty())) {
        // 有输入对应，一定是可以推导的
        ++out_index;
        continue;
      }
      (void)syms_only_of_output_.insert(out_sym);
      auto support_types = out_sym->GetTensorType().tensor_type_impl_->GetMutableDateTypeSet();
      if (support_types.size() > 1U && !IsBoolCompatibleOutput(support_types)) {
        could_infer = false;
        warning_code << std::string(space_count, ' ') << "GELOGW(\"Output ir_index [" << out_index
                     << "] has multi result " << TensorTypeToCode(out_sym->GetTensorType()) << ", cannot infer.\");\n";
      }
      ++out_index;
    }
    if (!could_infer) {
      ss << warning_code.str();
      ss << std::string(space_count, ' ') << "return FAILED;\n";
      return;
    }
    for (const auto out_sym : GetFirstDataTypeSymbolStore(def_).GetOutSymbols()) {
      if (syms_only_of_output_.find(out_sym) == syms_only_of_output_.end()) {
        // 走到这里说明out使用了input的sym,因为前面校验过共sym的输入dtypes是一样的，所以我们用此sym的第一个输入ir的dtype作为输出的dtype"
        ss << std::string(space_count, ' ') << "expect_output_dtypes.push_back(input_dtypes["
           << out_sym->GetIrInputIndexes().front() << "]);" << std::endl;
      } else {
        GenerateCustomTypeInference(out_sym, ss, space_count);
      }
    }
    ss << std::string(space_count, ' ') << "return SUCCESS;" << std::endl;
  }

  void GenerateOutputValidation(std::stringstream &ss) {
    size_t out_index = 0;
    for (const auto out_sym : GetFirstDataTypeSymbolStore(def_).GetOutSymbols()) {
      if (syms_only_of_output_.find(out_sym) == syms_only_of_output_.end()) {
        // 走到这里说明out使用了input的sym,因为前面校验过共sym的输入dtypes是一样的，所以我们用此sym的第一个输入ir的dtype跟out的dtype比对
        ss << "    GE_WARN_ASSERT(input_dtypes[" << out_sym->GetIrInputIndexes().front() << "] == "
           << "expect_output_dtypes[" << out_index << "]);" << std::endl;
      } else {
        GenerateCustomTypeValidation(out_sym, ss);
      }
      ++out_index;
    }
  }

 protected:
  void GenerateCustomTypeInference(const SymDtype *sym, std::stringstream &ss, int space_count) {
    // 走到这里说明out使用了跟所有input不一样的sym, 并且此输出只有一个支持的类型
    const auto &soc_to_sym_store = def_.GetSocToDataTypeSymbolStore();
    const auto &sym_id = sym->Id();
    bool is_first = true;
    for (const auto &soc_store : soc_to_sym_store) {
      const auto &named_syms = soc_store.second.GetNamedSymbols();
      auto it = named_syms.find(sym_id);
      if (it == named_syms.end()) {
        continue;
      }
      auto support_types = it->second->GetTensorType().tensor_type_impl_->GetMutableDateTypeSet();
      if (is_first) {
        ss << std::string(space_count, ' ') << "if (npu_arch == \"" << soc_store.first << "\") {\n";
        is_first = false;
      } else {
        ss << std::string(space_count, ' ') << "} else if (npu_arch == \"" << soc_store.first << "\") {\n";
      }
      const auto default_dtype = IsBoolCompatibleOutput(support_types) ? DT_BOOL : *support_types.begin();
      ss << std::string(space_count + 2U, ' ') << "expect_output_dtypes.push_back("
         << DataTypeToSerialString(default_dtype) << ");\n";
    }
    if (!is_first) {
      ss << std::string(space_count, ' ') << "}\n";
    }
  }

  Status GenerateCustomTypeValidation(SymDtype *sym, std::stringstream &ss) {
    if (!syms_checked_.insert(sym).second) {
      return SUCCESS;
    }
    auto iter = sym_2_ir_indexs_.find(sym);
    GE_ASSERT_TRUE(iter != sym_2_ir_indexs_.end());
    auto indexes_of_this_sym = iter->second;
    // 小于2个，不需要比较
    if (indexes_of_this_sym.size() >= 2U) {
      for (size_t index = 0U; index < indexes_of_this_sym.size() - 1U; ++index) {
        ss << "    GE_WARN_ASSERT(expect_output_dtypes[" << indexes_of_this_sym[index] << "] == expect_output_dtypes["
           << indexes_of_this_sym[index + 1] << "]);" << std::endl;
      }
    }
    const std::string tensor_type_obj = "support_dtypes_of_sym_" + sym->Id();
    const auto &soc_to_sym_store = def_.GetSocToDataTypeSymbolStore();
    const auto &sym_id = sym->Id();
    ss << "    std::set<ge::DataType> " << tensor_type_obj << ";\n";
    bool is_first = true;
    for (const auto &soc_store : soc_to_sym_store) {
      const auto &named_syms = soc_store.second.GetNamedSymbols();
      auto it = named_syms.find(sym_id);
      if (it == named_syms.end()) {
        continue;
      }
      if (is_first) {
        ss << "    if (npu_arch == \"" << soc_store.first << "\") {\n";
        is_first = false;
      } else {
        ss << "    } else if (npu_arch == \"" << soc_store.first << "\") {\n";
      }
      ss << "      " << tensor_type_obj << " = " << TensorTypeToCode(it->second->GetTensorType()) << ";\n";
    }
    if (!is_first) {
      ss << "    }\n";
      // 共sym的已经都校验过一致了，所以这里可以用第一个来校验是否在支持范围内
      auto check_index = indexes_of_this_sym.front();
      ss << "    GE_WARN_ASSERT("
         << "support_dtypes_of_sym_" << sym->Id() << ".find(expect_output_dtypes[" << check_index
         << "]) != support_dtypes_of_sym_" << sym->Id() << ".end());" << std::endl;
    }
    return SUCCESS;
  }

 private:
  const AscIrDef &def_;
  std::set<SymDtype *> syms_only_of_output_{};
  std::map<SymDtype *, std::vector<std::size_t>> sym_2_ir_indexs_{};
  std::set<SymDtype *> syms_checked_{};  // 多个输出可能sym一样
};

class OrderedSymbolProcessor : public SymbolProcessor {
 public:
  explicit OrderedSymbolProcessor(const AscIrDef &def) : SymbolProcessor(def), valid_dtype_nums_of_sym_(0U) {}

  Status PreProcessSymbol(std::stringstream &ss) {
    const auto &symbols = def_.GetDataTypeSymbolStore().GetSymbols();
    GE_ASSERT_TRUE(!symbols.empty());
    GE_ASSERT_SUCCESS(InitializeSymbolAttributes(symbols));
    GE_ASSERT_SUCCESS(ClassifySymbols(symbols, ss));
    return SUCCESS;
  }

  Status InstanceSymbol(std::stringstream &ss) {
    BuildResultMapping();
    ss << GenerateResultContainer();
    return SUCCESS;
  }

  void CheckSymbol(std::stringstream &ss) {
    ss << "\n";
    if (input_syms_.size() > 1U) {
      ss << "    auto iter = results.find(std::vector<ge::DataType>{";
      size_t index{0U};
      for (auto input_sym : input_syms_) {
        ss << "input_dtypes[" << input_sym->GetIrInputIndexes().front() << "]";
        if (index++ < input_syms_.size() - 1U) {
          ss << ", ";
        }
      }
      ss << "});\n";
    } else {
      ss << "    auto iter = results.find(input_dtypes[" << input_syms_.front()->GetIrInputIndexes().front() << "]);\n";
    }
    ss << "    GE_WARN_ASSERT(iter != results.end());\n";
  }

  Status HandleOutput(std::stringstream &ss) {
    ss << "    // 输出外部不指定的时候，生成推导的代码" << std::endl;
    ss << "    if (expect_output_dtypes.empty()) {" << std::endl;
    GenerateOutputInference(ss);
    ss << "    }" << std::endl;
    ss << "    // 输出外部指定，生成校验的代码" << std::endl;
    GenerateOutputValidation(ss);
    return SUCCESS;
  }

 private:
  void GenerateOutputInference(std::stringstream &ss) {
    size_t only_output_index{0U};
    for (const auto out_sym : def_.GetDataTypeSymbolStore().GetOutSymbols()) {
      if (std::find(only_out_syms_.begin(), only_out_syms_.end(), out_sym) == only_out_syms_.end()) {
        // 走到这里说明out使用了input的sym,因为前面校验过共sym的输入dtypes是一样的，所以我们用此sym的第一个输入ir的dtype作为输出的dtype"
        ss << "      expect_output_dtypes.push_back(input_dtypes[" << out_sym->GetIrInputIndexes().front() << "]);"
           << std::endl;
      } else {
        // 走到这里说明out使用了跟所有input不一样的sym, 我们使用输入的type推导输出
        if (container_meta_.output_count == 1U) {
          // std::set<ge::DataType>
          if (container_meta_.has_multiple_solutions) {
            ss << "      GE_WARN_ASSERT(iter->second.size() == 1U);" << std::endl;
            ss << "      expect_output_dtypes.push_back(*(iter->second.begin()));" << std::endl;
            // ge::DataType
          } else {
            ss << "      expect_output_dtypes.push_back(iter->second);" << std::endl;
          }
        } else {
          // std::vector<std::set<ge::DataType>>
          if (container_meta_.has_multiple_solutions) {
            ss << "      GE_WARN_ASSERT(iter->second[" << only_output_index << "].size() == 1U);" << std::endl;
            ss << "      expect_output_dtypes.push_back(*(iter->second[" << only_output_index << "].begin()));"
               << std::endl;
            // std::vector<ge::DataType>>
          } else {
            ss << "      expect_output_dtypes.push_back(iter->second[" << only_output_index << "]));" << std::endl;
          }
          only_output_index++;
        }
      }
    }
    ss << "      return SUCCESS;" << std::endl;
  }

  void GenerateOutputValidation(std::stringstream &ss) {
    size_t only_output_index{0U};
    size_t output_index{0U};
    for (const auto out_sym : def_.GetDataTypeSymbolStore().GetOutSymbols()) {
      if (std::find(only_out_syms_.begin(), only_out_syms_.end(), out_sym) == only_out_syms_.end()) {
        // 走到这里说明out使用了input的sym,因为前面校验过共sym的输入dtypes是一样的，所以我们用此sym的第一个输入ir的dtype跟输出的dtype做校验"
        ss << "    GE_WARN_ASSERT(input_dtypes[" << out_sym->GetIrInputIndexes().front() << "] == "
           << "expect_output_dtypes[" << output_index << "]);" << std::endl;
      } else {
        // 走到这里说明out使用了跟所有input不一样的sym, 我们使用输入的type推导输出
        if (container_meta_.output_count == 1U) {
          // std::set<ge::DataType>
          if (container_meta_.has_multiple_solutions) {
            ss << "    GE_WARN_ASSERT(iter->second.find(expect_output_dtypes[" << output_index
               << "]) != iter->second.end());" << std::endl;
            // ge::DataType
          } else {
            ss << "    GE_WARN_ASSERT(iter->second == "
               << "expect_output_dtypes[" << output_index << "]);" << std::endl;
          }
        } else {
          // std::vector<std::set<ge::DataType>>
          if (container_meta_.has_multiple_solutions) {
            ss << "    GE_WARN_ASSERT(iter->second[" << only_output_index << "].find(expect_output_dtypes["
               << output_index << "]) != iter->second[" << only_output_index << "].end());" << std::endl;
            // std::vector<ge::DataType>>
          } else {
            ss << "    GE_WARN_ASSERT(iter->second[" << only_output_index << "] == "
               << "expect_output_dtypes[" << output_index << "]);" << std::endl;
          }
          only_output_index++;
        }
      }
      output_index++;
    }
  }
  Status InitializeSymbolAttributes(const std::list<std::shared_ptr<SymDtype>> &symbols) {
    const auto &first_sym = symbols.front();
    GE_ASSERT_NOTNULL(first_sym);
    const auto &type_list = first_sym->GetOrderedTensorTypeList();
    valid_dtype_nums_of_sym_ = type_list.GetOrderedDtypes().size();
    return SUCCESS;
  }

  Status ClassifySymbols(const std::list<std::shared_ptr<SymDtype>> &symbols, std::stringstream &ss) {
    for (const auto &sym : symbols) {
      GE_ASSERT_NOTNULL(sym);
      GE_ASSERT_TRUE(sym->IsOrderedList());
      GE_ASSERT_SUCCESS(ValidateDtypeConsistency(sym));
      if (IsOutputOnlySymbol(sym)) {
        only_out_syms_.emplace_back(sym.get());
      } else {
        ProcessInputSymbol(sym.get(), ss);
      }
    }
    return SUCCESS;
  }

  Status ValidateDtypeConsistency(const std::shared_ptr<SymDtype> &sym) const {
    const auto &current_types = sym->GetOrderedTensorTypeList().GetOrderedDtypes();
    GE_ASSERT_TRUE(current_types.size() == valid_dtype_nums_of_sym_);
    return SUCCESS;
  }

  static bool IsOutputOnlySymbol(const std::shared_ptr<SymDtype> &sym) {
    return sym->GetIrInputIndexes().empty();
  }

  void ProcessInputSymbol(SymDtype *sym, std::stringstream &ss) {
    GenerateInputDtypeUniquenessCheck(sym, ss);
    input_syms_.emplace_back(sym);
  }

  // 结果生成相关
  void BuildResultMapping() {
    results2_.clear();
    for (size_t idx = 0U; idx < valid_dtype_nums_of_sym_; ++idx) {
      auto inputs = CollectInputDtypes(idx);
      auto outputs = CollectOutputDtypes(idx);
      results2_.emplace(std::move(inputs), std::move(outputs));
    }
  }

  std::vector<ge::DataType> CollectInputDtypes(size_t index) {
    std::vector<ge::DataType> inputs;
    inputs.reserve(input_syms_.size());

    for (auto sym : input_syms_) {
      inputs.push_back(GetDtypeByIndex(sym, index));
    }
    return inputs;
  }

  std::vector<ge::DataType> CollectOutputDtypes(size_t index) {
    std::vector<ge::DataType> outputs;
    outputs.reserve(only_out_syms_.size());

    for (auto sym : only_out_syms_) {
      outputs.push_back(GetDtypeByIndex(sym, index));
    }
    return outputs;
  }

  static ge::DataType GetDtypeByIndex(const SymDtype *sym, size_t index) {
    const auto &types = sym->GetOrderedTensorTypeList().GetOrderedDtypes();
    GE_ASSERT_TRUE(index < types.size());
    return types[index];
  }

  // 容器生成相关
  std::string GenerateResultContainer() {
    const auto solution_map = BuildSolutionMap();
    const bool has_multiple = CheckMultipleSolutions(solution_map);

    container_meta_ = {.input_count = input_syms_.size(),
                       .output_count = only_out_syms_.size(),
                       .has_multiple_solutions = has_multiple};

    return BuildContainerString(solution_map, container_meta_);
  }

  using SolutionMap = std::map<std::vector<ge::DataType>, std::set<std::vector<ge::DataType>>>;

  SolutionMap BuildSolutionMap() {
    SolutionMap mapping;
    for (const auto &pair : results2_) {
      mapping[pair.first].insert(pair.second);
    }
    return mapping;
  }

  static bool CheckMultipleSolutions(const SolutionMap &mapping) {
    return std::any_of(mapping.begin(), mapping.end(),
                       [](const std::pair<std::vector<ge::DataType>, std::set<std::vector<ge::DataType>>> &pair) {
                         return pair.second.size() > 1;
                       });
  }

  struct ContainerMeta {
    size_t input_count;
    size_t output_count;
    bool has_multiple_solutions;
  };

  std::string BuildContainerString(const SolutionMap &mapping, const ContainerMeta &meta) {
    std::ostringstream oss;
    container_type_ = GetContainerType(meta);
    oss << "    const static " << container_type_ << " results = {\n";
    AppendContainerEntries(oss, mapping, meta);
    oss << "\n    };";

    return oss.str();
  }
  static std::string GetContainerType(const ContainerMeta &meta) {
    std::ostringstream oss;
    oss << "std::map<" << (meta.input_count > 1 ? "std::vector<ge::DataType>" : "ge::DataType") << ", ";

    if (meta.output_count > 1) {
      oss << (meta.has_multiple_solutions ? "std::vector<std::set<ge::DataType>>" : "std::vector<ge::DataType>");
    } else {
      oss << (meta.has_multiple_solutions ? "std::set<ge::DataType>" : "ge::DataType");
    }

    oss << ">";
    return oss.str();
  }

  static void AppendContainerEntries(std::ostream &os, const SolutionMap &mapping, const ContainerMeta &meta) {
    std::vector<std::string> entries;
    entries.reserve(mapping.size());

    for (const auto &inputs_2_outputs : mapping) {
      entries.push_back(BuildEntryString(inputs_2_outputs.first, inputs_2_outputs.second, meta));
    }
    os << "        " << JoinEntries(entries, ",\n        ");
  }

  static std::string BuildEntryString(const std::vector<ge::DataType> &input,
                                      const std::set<std::vector<ge::DataType>> &outputs, const ContainerMeta &meta) {
    return "{" + SerializeVector(input) + ", " + SerializeOutputs(outputs, meta) + "}";
  }

  static std::string SerializeVector(const std::vector<ge::DataType> &vec) {
    if (vec.size() == 1U) {
      return DataTypeToSerialString(vec[0U]);
    }

    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0U; i < vec.size(); ++i) {
      oss << DataTypeToSerialString(vec[i]);
      if (i < vec.size() - 1U) {
        oss << ", ";
      }
    }
    oss << "}";
    return oss.str();
  }

  static std::string SerializeOutputs(const std::set<std::vector<ge::DataType>> &outputs, const ContainerMeta &meta) {
    if (meta.output_count == 1U) {
      return SerializeSingleOutput(outputs, meta.has_multiple_solutions);
    }
    return SerializeMultiOutputs(outputs, meta);
  }

  static std::string SerializeSingleOutput(const std::set<std::vector<ge::DataType>> &outputs, bool multiple) {
    if (!multiple) {
      return DataTypeToSerialString(outputs.begin()->front());
    }

    std::vector<ge::DataType> unique_outputs;
    for (const auto &out : outputs) {
      unique_outputs.push_back(out.front());
    }
    return SerializeSet(unique_outputs);
  }

  static std::string SerializeMultiOutputs(const std::set<std::vector<ge::DataType>> &outputs,
                                           const ContainerMeta &meta) {
    if (!meta.has_multiple_solutions) {
      return SerializeVector(*outputs.begin());
    }

    std::vector<std::set<ge::DataType>> output_sets;
    for (size_t i = 0; i < meta.output_count; ++i) {
      output_sets.emplace_back();
      for (const auto &out : outputs) {
        output_sets.back().insert(out[i]);
      }
    }
    return SerializeSetVector(output_sets);
  }

  static std::string SerializeSet(const std::vector<ge::DataType> &types) {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0U; i < types.size(); ++i) {
      oss << DataTypeToSerialString(types[i]);
      if (i < types.size() - 1U) {
        oss << ", ";
      }
    }
    oss << "}";
    return oss.str();
  }

  static std::string SerializeSetVector(const std::vector<std::set<ge::DataType>> &sets) {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0U; i < sets.size(); ++i) {
      oss << SerializeSet(std::vector<ge::DataType>{sets[i].begin(), sets[i].end()});
      if (i < sets.size() - 1U) {
        oss << ", ";
      }
    }
    oss << "}";
    return oss.str();
  }

  static std::string JoinEntries(const std::vector<std::string> &entries, const std::string &delimiter) {
    std::ostringstream oss;
    for (size_t i = 0U; i < entries.size(); ++i) {
      oss << entries[i];
      if (i < entries.size() - 1U) {
        oss << delimiter;
      }
    }
    return oss.str();
  }

  std::vector<SymDtype *> input_syms_{};
  std::vector<SymDtype *> only_out_syms_{};
  // 每一个对key,value代表一个合法解，
  // key是多个输入的实参dtype，value是多个输出的推导处理的dtype，之所以是multimap因为某个输出可能 有多个解
  std::multimap<std::vector<ge::DataType>, std::vector<ge::DataType>> results2_{};
  size_t valid_dtype_nums_of_sym_;
  // 根据输入输出的个数和输出是否有唯一解需要实例化不同类型的容器
  std::string container_type_{};
  ContainerMeta container_meta_{};
};

class SocOrderedSymbolProcessor : public SymbolProcessor {
 public:
  explicit SocOrderedSymbolProcessor(const AscIrDef &def) : SymbolProcessor(def) {}

  Status PreProcessSymbol(std::stringstream &ss) {
    GE_ASSERT_SUCCESS(InitializeSymbolAttributes());
    const auto &store = GetFirstDataTypeSymbolStore(def_);
    GE_ASSERT_TRUE(!store.GetSymbols().empty());
    GE_ASSERT_SUCCESS(ClassifySymbols(store, ss));
    return SUCCESS;
  }

  Status InstanceSymbol(std::stringstream &ss) {
    BuildResultMapping();
    ss << GenerateResultContainer();
    return SUCCESS;
  }

  void CheckSymbol(std::stringstream &ss) {
    ss << "\n";
    if (input_sym_id_to_ir_idx_.size() > 1U) {
      ss << "    auto iter = results.find(std::vector<ge::DataType>{";
      size_t index{0U};
      for (const auto &input_sym : input_sym_id_to_ir_idx_) {
        ss << "input_dtypes[" << input_sym.second << "]";
        if (index++ < input_sym_id_to_ir_idx_.size() - 1U) {
          ss << ", ";
        }
      }
      ss << "});\n";
    } else {
      ss << "    auto iter = results.find(input_dtypes[" << input_sym_id_to_ir_idx_.begin()->second << "]);\n";
    }
    ss << "    GE_WARN_ASSERT(iter != results.end());\n";
  }

  Status HandleOutput(std::stringstream &ss) {
    ss << "    // 输出外部不指定的时候，生成推导的代码" << std::endl;
    ss << "    if (expect_output_dtypes.empty()) {" << std::endl;
    GenerateOutputInference(ss);
    ss << "    }" << std::endl;
    ss << "    // 输出外部指定，生成校验的代码" << std::endl;
    GenerateOutputValidation(ss);
    return SUCCESS;
  }

 private:
  struct ContainerMeta {
    size_t input_count;
    size_t output_count;
  };

  void GenerateOutputInference(std::stringstream &ss) {
    size_t only_output_index{0U};
    for (const auto &out_sym : GetFirstDataTypeSymbolStore(def_).GetOutSymbols()) {
      auto iter = input_sym_id_to_ir_idx_.find(out_sym->Id());
      if (iter != input_sym_id_to_ir_idx_.end()) {
        // 走到这里说明out使用了input的sym,因为前面校验过共sym的输入dtypes是一样的，所以我们用此sym的第一个输入ir的dtype作为输出的dtype"
        ss << "      expect_output_dtypes.push_back(input_dtypes[" << iter->second << "]);" << std::endl;
      } else {
        // 走到这里说明out使用了跟所有input不一样的sym, 我们使用输入的type推导输出
        if (container_meta_.output_count == 1U) {
          // std::set<ge::DataType>
          ss << "      GE_WARN_ASSERT(iter->second.size() == 1U);" << std::endl;
          ss << "      expect_output_dtypes.push_back(*(iter->second.begin()));" << std::endl;
        } else {
          // std::vector<std::set<ge::DataType>>
          ss << "      GE_WARN_ASSERT(iter->second[" << only_output_index << "].size() == 1U);" << std::endl;
          ss << "      expect_output_dtypes.push_back(*(iter->second[" << only_output_index << "].begin()));"
             << std::endl;
          only_output_index++;
        }
      }
    }
    ss << "      return af::SUCCESS;" << std::endl;
  }

  void GenerateOutputValidation(std::stringstream &ss) {
    size_t only_output_index{0U};
    size_t output_index{0U};
    for (const auto &out_sym : GetFirstDataTypeSymbolStore(def_).GetOutSymbols()) {
      auto iter = input_sym_id_to_ir_idx_.find(out_sym->Id());
      if (iter != input_sym_id_to_ir_idx_.end()) {
        // 走到这里说明out使用了input的sym,因为前面校验过共sym的输入dtypes是一样的，所以我们用此sym的第一个输入ir的dtype跟输出的dtype做校验"
        ss << "    GE_WARN_ASSERT(input_dtypes[" << iter->second << "] == "
           << "expect_output_dtypes[" << output_index << "]);" << std::endl;
      } else {
        // 走到这里说明out使用了跟所有input不一样的sym, 我们使用输入的type推导输出
        if (container_meta_.output_count == 1U) {
          // std::set<ge::DataType>
          ss << "    GE_WARN_ASSERT(iter->second.find(expect_output_dtypes[" << output_index
             << "]) != iter->second.end());" << std::endl;
        } else {
          // std::vector<std::set<ge::DataType>>
          ss << "    GE_WARN_ASSERT(iter->second[" << only_output_index << "].find(expect_output_dtypes["
             << output_index << "]) != iter->second[" << only_output_index << "].end());" << std::endl;
          only_output_index++;
        }
      }
      output_index++;
    }
  }

  Status InitializeSymbolAttributes() {
    for (const auto &soc_store : def_.GetSocToDataTypeSymbolStore()) {
      bool is_first{true};
      for (const auto &name_to_sym : soc_store.second.GetNamedSymbols()) {
        GE_ASSERT_TRUE(name_to_sym.second->IsOrderedList());
        const auto &current_types = name_to_sym.second->GetOrderedTensorTypeList().GetOrderedDtypes();
        if (is_first) {
          soc_name_to_meta_[soc_store.first].valid_dtype_nums_of_sym = current_types.size();
          is_first = false;
        } else {
          GE_ASSERT_TRUE(current_types.size() == soc_name_to_meta_[soc_store.first].valid_dtype_nums_of_sym);
        }
      }
    }
    return SUCCESS;
  }

  Status ClassifySymbols(const IRDataTypeSymbolStore &store, std::stringstream &ss) {
    for (const auto &sym : store.GetSymbols()) {
      GE_ASSERT_NOTNULL(sym);
      if (!sym->GetIrInputIndexes().empty()) {
        GenerateInputDtypeUniquenessCheck(sym.get(), ss);
        input_sym_id_to_ir_idx_.emplace(sym->Id(), sym->GetIrInputIndexes().front());
      } else {
        only_output_sym_ids_.push_back(sym->Id());
      }
    }

    container_meta_ = {.input_count = input_sym_id_to_ir_idx_.size(), .output_count = only_output_sym_ids_.size()};
    return SUCCESS;
  }

  // 结果生成相关
  Status BuildResultMapping() {
    for (const auto &iter : def_.GetSocToDataTypeSymbolStore()) {
      const auto &soc_name = iter.first;
      auto &soc_meta = soc_name_to_meta_[soc_name];
      for (size_t idx = 0U; idx < soc_meta.valid_dtype_nums_of_sym; ++idx) {
        std::vector<ge::DataType> inputs;
        std::vector<ge::DataType> outputs;
        GE_ASSERT_GRAPH_SUCCESS(CollectInputDtypes(iter.second.GetNamedSymbols(), idx, inputs));
        GE_ASSERT_GRAPH_SUCCESS(CollectOutputDtypes(iter.second.GetNamedSymbols(), idx, outputs));
        soc_meta.results2.emplace(std::move(inputs), std::move(outputs));
      }
    }
    return ge::GRAPH_SUCCESS;
  }

  Status CollectInputDtypes(const std::map<std::string, SymDtype *> &name_to_syms, size_t index,
                            std::vector<ge::DataType> &inputs) {
    inputs.reserve(input_sym_id_to_ir_idx_.size());
    for (const auto &sym_id : input_sym_id_to_ir_idx_) {
      auto iter = name_to_syms.find(sym_id.first);
      GE_ASSERT_TRUE((iter != name_to_syms.end()) && (iter->second != nullptr));
      inputs.push_back(GetDtypeByIndex(iter->second, index));
    }
    return af::SUCCESS;
  }

  Status CollectOutputDtypes(const std::map<std::string, SymDtype *> &name_to_syms, size_t index,
                             std::vector<ge::DataType> &outputs) {
    outputs.reserve(only_output_sym_ids_.size());
    for (const auto &sym_id : only_output_sym_ids_) {
      auto iter = name_to_syms.find(sym_id);
      GE_ASSERT_TRUE((iter != name_to_syms.end()) && (iter->second != nullptr));
      outputs.push_back(GetDtypeByIndex(iter->second, index));
    }
    return af::SUCCESS;
  }

  static ge::DataType GetDtypeByIndex(const SymDtype *sym, size_t index) {
    const auto &types = sym->GetOrderedTensorTypeList().GetOrderedDtypes();
    GE_ASSERT_TRUE(index < types.size());
    return types[index];
  }

  // 容器生成相关
  std::string GenerateResultContainer() {
    std::map<std::string, SolutionMap> soc_to_solution_map;
    BuildSolutionMap(soc_to_solution_map);
    ContainerMeta container_meta = {.input_count = input_sym_id_to_ir_idx_.size(),
                                    .output_count = only_output_sym_ids_.size()};

    return BuildContainerString(soc_to_solution_map, container_meta);
  }

  using SolutionMap = std::map<std::vector<ge::DataType>, std::set<std::vector<ge::DataType>>>;

  void BuildSolutionMap(std::map<std::string, SolutionMap> &soc_to_solution_map) {
    for (auto &iter : soc_name_to_meta_) {
      SolutionMap mapping;
      for (const auto &pair : iter.second.results2) {
        mapping[pair.first].insert(pair.second);
      }
      soc_to_solution_map.emplace(iter.first, std::move(mapping));
    }
  }

  std::string BuildContainerString(const std::map<std::string, SolutionMap> &soc_to_solution_map,
                                   const ContainerMeta &meta) {
    std::ostringstream oss;
    container_type_ = GetContainerType(meta);
    oss << "    " << container_type_ << " results;" << std::endl;
    bool is_first = true;
    for (const auto &iter : soc_to_solution_map) {
      if (is_first) {
        oss << "    if (npu_arch == \"" << iter.first << "\") {\n";
        is_first = false;
      } else {
        oss << "    } else if (npu_arch == \"" << iter.first << "\") {\n";
      }
      oss << GenContainerEntries(iter.second, meta) << std::endl;
    }
    oss << "    } else {\n";
    oss << R"(      GELOGE(af::FAILED, "Unknown npu arch: %s", npu_arch.c_str());)" << std::endl;
    oss << "      return af::FAILED;\n";
    oss << "    }\n";
    return oss.str();
  }

  static std::string GetContainerType(const ContainerMeta &meta) {
    std::ostringstream oss;
    oss << "std::map<" << (meta.input_count > 1 ? "std::vector<ge::DataType>" : "ge::DataType") << ", ";

    if (meta.output_count > 1) {
      oss << "std::vector<std::set<ge::DataType>>";
    } else {
      oss << "std::set<ge::DataType>";
    }

    oss << ">";
    return oss.str();
  }

  static std::string GenContainerEntries(const SolutionMap &mapping, const ContainerMeta &meta) {
    std::stringstream ss;
    ss << "       results = {\n";

    std::vector<std::string> entries;
    entries.reserve(mapping.size());
    for (const auto &inputs_2_outputs : mapping) {
      entries.push_back(BuildEntryString(inputs_2_outputs.first, inputs_2_outputs.second, meta));
    }
    ss << "        " << JoinEntries(entries, ",\n        ");
    ss << "\n      };";
    return ss.str();
  }

  static std::string BuildEntryString(const std::vector<ge::DataType> &input,
                                      const std::set<std::vector<ge::DataType>> &outputs, const ContainerMeta &meta) {
    return "{" + SerializeVector(input) + ", " + SerializeOutputs(outputs, meta) + "}";
  }

  static std::string SerializeVector(const std::vector<ge::DataType> &vec) {
    if (vec.size() == 1U) {
      return DataTypeToSerialString(vec[0U]);
    }

    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0U; i < vec.size(); ++i) {
      oss << DataTypeToSerialString(vec[i]);
      if (i < vec.size() - 1U) {
        oss << ", ";
      }
    }
    oss << "}";
    return oss.str();
  }

  static std::string SerializeOutputs(const std::set<std::vector<ge::DataType>> &outputs, const ContainerMeta &meta) {
    if (meta.output_count == 1U) {
      return SerializeSingleOutput(outputs);
    }
    return SerializeMultiOutputs(outputs, meta);
  }

  static std::string SerializeSingleOutput(const std::set<std::vector<ge::DataType>> &outputs) {
    std::vector<ge::DataType> unique_outputs;
    for (const auto &out : outputs) {
      unique_outputs.push_back(out.front());
    }
    return SerializeSet(unique_outputs);
  }

  static std::string SerializeMultiOutputs(const std::set<std::vector<ge::DataType>> &outputs,
                                           const ContainerMeta &meta) {
    std::vector<std::set<ge::DataType>> output_sets;
    for (size_t i = 0; i < meta.output_count; ++i) {
      output_sets.emplace_back();
      for (const auto &out : outputs) {
        output_sets.back().insert(out[i]);
      }
    }
    return SerializeSetVector(output_sets);
  }

  static std::string SerializeSet(const std::vector<ge::DataType> &types) {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0U; i < types.size(); ++i) {
      oss << DataTypeToSerialString(types[i]);
      if (i < types.size() - 1U) {
        oss << ", ";
      }
    }
    oss << "}";
    return oss.str();
  }

  static std::string SerializeSetVector(const std::vector<std::set<ge::DataType>> &sets) {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0U; i < sets.size(); ++i) {
      oss << SerializeSet(std::vector<ge::DataType>{sets[i].begin(), sets[i].end()});
      if (i < sets.size() - 1U) {
        oss << ", ";
      }
    }
    oss << "}";
    return oss.str();
  }

  static std::string JoinEntries(const std::vector<std::string> &entries, const std::string &delimiter) {
    std::ostringstream oss;
    for (size_t i = 0U; i < entries.size(); ++i) {
      oss << entries[i];
      if (i < entries.size() - 1U) {
        oss << delimiter;
      }
    }
    return oss.str();
  }

  struct SocMeta {
    // 每一个对key,value代表一个合法解，
    // key是多个输入的实参dtype，value是多个输出的推导处理的dtype，之所以是multimap因为某个输出可能 有多个解
    std::multimap<std::vector<ge::DataType>, std::vector<ge::DataType>> results2{};
    size_t valid_dtype_nums_of_sym{0UL};
  };

  std::map<std::string, SocMeta> soc_name_to_meta_;
  // 根据输入输出的个数和输出是否有唯一解需要实例化不同类型的容器
  std::string container_type_{};
  std::map<std::string, size_t> input_sym_id_to_ir_idx_;
  std::vector<std::string> only_output_sym_ids_;
  ContainerMeta container_meta_{};
};

class InferDtypeCodeGenerator {
 public:
  explicit InferDtypeCodeGenerator(const AscIrDef &def) : def_(def) {
    is_soc_ordered_dtype_infer_ = IsSocOrderedTensorListStore(def_);
    is_ordered_dtype_infer_ = def_.GetDataTypeSymbolStore().IsSupportOrderedSymbolicInferDtype();
  }
  Status Generate(std::stringstream &ss) {
    GenerateFunctionSignature(ss);
    GenerateArgsSizeAssertion(ss);
    GE_ASSERT_SUCCESS(GenerateSymbolProcessing(ss));
    GenerateOutputHandling(ss);
    GenerateReturnStatement(ss);
    return SUCCESS;
  }

 private:
  static void GenerateFunctionSignature(std::stringstream &ss) {
    ss << R"(  inline static Status InferDataType(const std::vector<DataType>& input_dtypes,
                                     std::vector<DataType>& expect_output_dtypes,
                                     [[maybe_unused]]const std::string& npu_arch) {)"
       << std::endl;
  }

  void GenerateArgsSizeAssertion(std::stringstream &ss) {
    ss << "    // 校验入参容器的元素个数是否合法" << std::endl;
    AppendArgsSizeAssertion(ss, def_);
    ss << "    GE_ASSERT_TRUE(expect_output_dtypes.empty() || expect_output_dtypes.size() == "
       << def_.GetOutputDefs().size() << "U);" << std::endl;
    ss << std::endl;
  }

  Status GenerateSymbolProcessing(std::stringstream &ss) {
    if (is_soc_ordered_dtype_infer_) {
      if (!def_.GetInputDefs().empty() &&
          std::any_of(def_.GetInputDefs().begin(), def_.GetInputDefs().end(),
                      [](const auto &input_def) { return input_def.second == IrInputType::kIrInputOptional; })) {
        GE_ASSERT_TRUE(false, "Ordered dtype inference with optional inputs is not supported");
      }
      SocOrderedSymbolProcessor ordered_symbol_processor(def_);
      GE_ASSERT_SUCCESS(ordered_symbol_processor.PreProcessSymbol(ss));
      GE_ASSERT_SUCCESS(ordered_symbol_processor.InstanceSymbol(ss));
      ordered_symbol_processor.CheckSymbol(ss);
      // output跟sym处理紧密相关，所以放在这里处理
      GE_ASSERT_SUCCESS(ordered_symbol_processor.HandleOutput(ss));
      ss << std::endl;
      return SUCCESS;
    }
    if (is_ordered_dtype_infer_) {
      if (!def_.GetInputDefs().empty() &&
          std::any_of(def_.GetInputDefs().begin(), def_.GetInputDefs().end(),
                      [](const auto &input_def) { return input_def.second == IrInputType::kIrInputOptional; })) {
        GE_ASSERT_TRUE(false, "Ordered dtype inference with optional inputs is not supported");
      }
      OrderedSymbolProcessor ordered_symbol_processor(def_);
      GE_ASSERT_SUCCESS(ordered_symbol_processor.PreProcessSymbol(ss));
      GE_ASSERT_SUCCESS(ordered_symbol_processor.InstanceSymbol(ss));
      ordered_symbol_processor.CheckSymbol(ss);
      // output跟sym处理紧密相关，所以放在这里处理
      GE_ASSERT_SUCCESS(ordered_symbol_processor.HandleOutput(ss));
      ss << std::endl;
      return SUCCESS;
    }
    ss << "    // 校验同sym的输入的dtype是否在注册范围内并且一致" << std::endl;
    const bool has_only_optional_inputs = HasOnlyOptionalInputs(def_);
    if (has_only_optional_inputs) {
      ss << "    if (!input_dtypes.empty()) {" << std::endl;
    }
    SymbolProcessor symbol_processor(def_);
    symbol_processor.GenSocVersionCallFunc();
    for (const auto &sym : GetFirstDataTypeSymbolStore(def_).GetNamedSymbols()) {
      symbol_processor.ProcessSymbol(sym, ss);
    }
    if (has_only_optional_inputs) {
      ss << "    }" << std::endl;
    }
    ss << std::endl;
    return SUCCESS;
  }

  void GenerateOutputHandling(std::stringstream &ss) {
    if (is_ordered_dtype_infer_ || is_soc_ordered_dtype_infer_) {
      return;
    }
    OutputHandler handler(def_);
    ss << "    // 输出外部不指定的时候，生成推导的代码" << std::endl;
    ss << "    if (expect_output_dtypes.empty()) {" << std::endl;
    handler.GenerateOutputInference(ss);
    ss << "    }" << std::endl;
    ss << "    // 输出外部指定，生成校验的代码" << std::endl;
    handler.GenerateOutputValidation(ss);
  }

  static void GenerateReturnStatement(std::stringstream &ss) {
    ss << "    return SUCCESS;" << std::endl;
    ss << "  };" << std::endl;
  };
  const AscIrDef &def_;
  bool is_ordered_dtype_infer_{false};
  bool is_soc_ordered_dtype_infer_{false};
};

class InferDtypeWithNoCheckCodeGenerator {
 public:
  explicit InferDtypeWithNoCheckCodeGenerator(const AscIrDef &def) : def_(def) {
    is_soc_ordered_dtype_infer_ = IsSocOrderedTensorListStore(def_);
    is_ordered_dtype_infer_ = def_.GetDataTypeSymbolStore().IsSupportOrderedSymbolicInferDtype();
  }
  Status Generate(std::stringstream &ss) const {
    GenerateFunctionSignature(ss);
    if (is_ordered_dtype_infer_ || is_soc_ordered_dtype_infer_) {
      ss << "    // 输入输出存在关联, 无法进行推导" << std::endl;
      ss << "    (void)input_dtypes;" << std::endl;
      ss << "    (void)expect_output_dtypes;" << std::endl;
      ss << "    GELOGW(\"Node type %s is not supported to infernocheck for dtype.\", Type);" << std::endl;
      ss << "    return af::FAILED;" << std::endl;
      ss << "  };" << std::endl << std::endl;
      return SUCCESS;
    }
    GenerateArgsSizeAssertion(ss);
    GE_ASSERT_SUCCESS(GenerateSymbolProcessing(ss));
    GenerateOutputHandling(ss);
    GenerateReturnStatement(ss);
    return SUCCESS;
  }

 private:
  static void GenerateFunctionSignature(std::stringstream &ss) {
    ss << R"(  inline static Status InferDataTypeWithNoCheck(const std::vector<DataType>& input_dtypes,
                                                std::vector<DataType>& expect_output_dtypes,
                                                [[maybe_unused]]const std::string& npu_arch = "") {)"
       << std::endl;
  }

  void GenerateArgsSizeAssertion(std::stringstream &ss) const {
    ss << "    // 校验入参容器的元素个数是否合法" << std::endl;
    AppendArgsSizeAssertion(ss, def_);
    ss << "    GE_ASSERT_TRUE(expect_output_dtypes.empty());" << std::endl;

    ss << std::endl;
  }

  Status GenerateSymbolProcessing(std::stringstream &ss) const {
    ss << "    // 校验同sym的输入的dtype是否一致" << std::endl;
    const bool has_only_optional_inputs = HasOnlyOptionalInputs(def_);
    if (has_only_optional_inputs) {
      ss << "    if (!input_dtypes.empty()) {" << std::endl;
    }
    SymbolProcessor symbol_processor(def_);
    for (const auto &sym : GetFirstDataTypeSymbolStore(def_).GetNamedSymbols()) {
      symbol_processor.ProcessSymbolWithNoCheck(sym, ss);
    }
    if (has_only_optional_inputs) {
      ss << "    }" << std::endl;
    }
    ss << std::endl;
    return SUCCESS;
  }

  void GenerateOutputHandling(std::stringstream &ss) const {
    constexpr int space_count = 4;
    OutputHandler handler(def_);

    handler.GenerateOutputInference(ss, space_count);
  }

  static void GenerateReturnStatement(std::stringstream &ss) {
    ss << "  };" << std::endl << std::endl;
  };
  const AscIrDef &def_;
  bool is_ordered_dtype_infer_{false};
  bool is_soc_ordered_dtype_infer_{false};
};

Status GenInferDtypeFuncDef(const AscIrDef &def, std::stringstream &ss) {
  InferDtypeCodeGenerator generator(def);
  return generator.Generate(ss);
}

Status GenInferDtypeWithNoCheckFuncDef(const AscIrDef &def, std::stringstream &ss) {
  InferDtypeWithNoCheckCodeGenerator generator(def);
  return generator.Generate(ss);
}

void GenCopyConstructor(const AscIrDef &def, std::stringstream &ss) {
  const std::string &ir_type = def.GetType();
  ss << "  inline " << ir_type << "& operator=(const " << ir_type << "&) = delete;" << std::endl;
  ss << "  inline " << ir_type << "(" << ir_type << " &&) = delete;" << std::endl;
  ss << "  inline " << ir_type << "(const " << ir_type << " &other)";
  ss << " : af::Operator(other),";
  ss << " attr(other.attr)";
  if (!def.GetAttrDefs().empty()) {
    ss << ", ir_attr(dynamic_cast<Asc" << ir_type << "IrAttrDef&>(*(attr.ir_attr)))";
  }
  const auto &input_defs = def.GetInputDefs();
  for (const auto &input_def : input_defs) {
    ss << "," << std::endl << "    " << input_def.first << "(this)";
  }
  const auto &output_defs = def.GetOutputDefs();
  for (size_t i = 0UL; i < output_defs.size(); ++i) {
    if (output_defs[i].second != IrOutputType::kIrOutputDynamic) {
      ss << "," << std::endl << "    " << output_defs[i].first << "(this, " << i << ")";
    }
  }
  ss << " {" << std::endl;
  for (const auto &output_def : output_defs) {
    if (output_def.second != IrOutputType::kIrOutputDynamic) {
      ss << "    " << output_def.first << ".TryInitTensorAttr();" << std::endl;
    }
  }

  if ((output_defs.size() == 1U) && (output_defs[0].second == IrOutputType::kIrOutputDynamic)) {
    ss << "  InstanceOutputy(other.y.size());" << std::endl;
  }

  ss << "  }" << std::endl;
}

void GenInstanceOutputy(const AscIrDef &def, std::stringstream &ss) {
  const auto &output_defs = def.GetOutputDefs();
  if (output_defs.size() != 1U) {
    return;
  }
  if (output_defs[0].second != IrOutputType::kIrOutputDynamic) {
    return;
  }

  ss << "  void InstanceOutputy(uint32_t num) {" << std::endl;
  ss << "    this->DynamicOutputRegister(\"y\", num);" << std::endl;
  ss << "    for (size_t i = 0U; i < num; i++) {" << std::endl;
  ss << "      this->y.push_back(af::AscOpOutput(this, i));" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;

  return;
}

void GenAscIr(const AscIrDef &def, std::stringstream &ss) {
  const std::string &ir_type = def.GetType();
  ss << "namespace ascir_op {" << std::endl;
  ss << "struct " << ir_type << " : public Operator {" << std::endl;

  const auto &out_defs = def.GetOutputDefs();
  for (const auto &output_def : out_defs) {
    if (output_def.second == IrOutputType::kIrOutputDynamic) {
      ss << "  using af::Operator::DynamicOutputRegister;" << std::endl;
      break;
    }
  }

  ss << "  static constexpr const char *Type = \"" << ir_type << "\";" << std::endl;
  ss << "  af::AscNodeAttr &attr;" << std::endl;
  const auto &ir_attr_class_name = TryGenIrAttrClass(def, ss);
  // generate input output definitions
  const auto &input_defs = def.GetInputDefs();
  for (size_t i = 0UL; i < input_defs.size(); ++i) {
    const auto &input_def = input_defs[i];
    if (input_def.second == IrInputType::kIrInputDynamic) {
      ss << "  af::AscOpDynamicInput<" << i << "> " << input_def.first << ";" << std::endl;
    } else {
      ss << "  af::AscOpInput<" << i << "> " << input_def.first << ";" << std::endl;
    }
  }

  const auto &output_defs = def.GetOutputDefs();
  for (const auto &output_def : output_defs) {
    if (output_def.second == IrOutputType::kIrOutputDynamic) {
      ss << "  std::vector<af::AscOpOutput> " << output_def.first << ";" << std::endl;
    } else {
      ss << "  af::AscOpOutput " << output_def.first << ";" << std::endl;
    }
  }

  // generate constructor func definitions
  if (def.IsStartNode()) {
    GenConstructorDef(def, ir_attr_class_name, ss, true);
  }
  GenConstructorDef(def, ir_attr_class_name, ss);
  (void)GenInferDtypeFuncDef(def, ss);
  (void)GenInferDtypeWithNoCheckFuncDef(def, ss);
  GenCopyConstructor(def, ss);
  GenInstanceOutputy(def, ss);
  ss << "};" << std::endl;
  ss << "}" << std::endl;
}

void GenIrComment(const AscIrDef &def, std::stringstream &ss) {
  const auto &comment = def.GetComment();
  if (!comment.empty()) {
    ss << "/* \n";
    ss << comment << "\n";
    ss << "*/ \n";
  }
}

class FunctionGenerator {
 public:
  explicit FunctionGenerator(const AscIrDef &def) : def_(def) {}
  virtual ~FunctionGenerator() = default;

  virtual void Gen(std::stringstream &ss, const bool has_optional_input) const {
    GenDefinition(ss, has_optional_input);

    GenInstantiation(ss, NeedGraphInInstantiation(has_optional_input));
    ss << std::endl;

    if (GenConnectInputs(ss, has_optional_input)) {
      ss << std::endl;
    }

    if (GenAttrAssignment(ss)) {
      ss << std::endl;
    }

    GenSchedInfo(ss);
    ss << std::endl;

    if (GenOutputsAssignment(ss)) {
      ss << std::endl;
    }
    GenOutputMemInfo(ss);
    GenPaddingAxis(ss);

    // 计算向量化轴，向量化轴的计算顺序:输出View在当前API的Loop轴内侧(不包括Loop轴)的所有轴
    TryGenOutputsVectorizedAxis(ss);

    ss << std::endl;

    GenReturn(ss);
  }

  virtual void GenDefinition(std::stringstream &ss, const bool has_optional_input) const;
  virtual void GenInstantiation(std::stringstream &ss, const bool graph_used_in_instantiation) const;
  virtual bool GenConnectInputs(std::stringstream &ss, const bool has_optional_input) const;
  virtual bool GenAttrAssignment(std::stringstream &ss) const;
  virtual void GenSchedInfo(std::stringstream &ss) const {
    ss << "  SET_SCHED_AXIS_IF_IN_CONTEXT(op);" << std::endl;
  }
  virtual void TryGenOutputsVectorizedAxis(std::stringstream &ss) const {
    const auto &output_defs = def_.GetOutputDefs();
    for (const auto &name : output_defs) {
      ss << "  *op." << name.first << ".vectorized_axis = AxisUtils::GetDefaultVectorizedAxis(*op." << name.first
         << ".axis, op.attr.sched.loop_axis);" << std::endl;
    }
  }
  virtual void GenOutputMemInfo(std::stringstream &ss) const {
    const auto &output_defs = def_.GetOutputDefs();
    for (const auto &name : output_defs) {
      ss << "  op." << name.first << ".mem->tensor_id = "
         << "CodeGenUtils::GenNextTensorId(op);" << std::endl;
    }
  }
  virtual bool GenOutputsAssignment(std::stringstream &ss) const {
    bool generated = false;

    // generate infer data type code
    if (def_.infer_data_type_generator != nullptr) {
      generated = true;
      def_.infer_data_type_generator(def_, ss);
    }
    if (def_.infer_view_generator != nullptr) {
      generated = true;
      def_.infer_view_generator(def_, ss);
    }
    return generated;
  }
  virtual void GenPaddingAxis(std::stringstream &ss) const {
    const auto &output_defs = def_.GetOutputDefs();
    for (const auto &name : output_defs) {
      ss << "  THROW(PadOutputViewToSched(op." << name.first << "));" << std::endl;
    }
  }
  virtual void GenReturn(std::stringstream &ss) const {
    const auto &output_defs = def_.GetOutputDefs();
    if (output_defs.empty()) {
      ss << "  return op;" << std::endl;
    } else if (output_defs.size() == 1U) {
      ss << "  return op." << output_defs[0U].first << ";" << std::endl;
    } else {
      ss << "  return std::make_tuple(";
      for (size_t i = 0; i < output_defs.size(); ++i) {
        if (i == 0) {
          ss << "op." << output_defs[i].first;
        } else {
          ss << " ,op." << output_defs[i].first;
        }
      }
      ss << ");" << std::endl;
    }
    ss << "}" << std::endl;
  }

 protected:
  const AscIrDef &def_;

  static bool NeedConnectByInputArgs(const bool has_optional_input,
                                     const std::pair<std::string, IrInputType> &input_def) {
    return ((has_optional_input || (input_def.second != IrInputType::kIrInputOptional)) &&
            (input_def.second != IrInputType::kIrInputDynamic));
  }

 private:
  bool NeedGraphInInstantiation(const bool has_optional_input) const {
    return def_.IsStartNode() && (def_.GetInputDefs().empty() || !has_optional_input);
  }
};

void ascir::FunctionGenerator::GenDefinition(std::stringstream &ss, const bool has_optional_input) const {
  const std::vector<std::pair<std::string, IrInputType>> *input_defs;
  std::vector<std::pair<std::string, IrInputType>> empty_input_defs;

  if (def_.IsStartNode()) {
    if (has_optional_input) {
      input_defs = &def_.GetInputDefs();
    } else {
      //  由于历史原因，IsStartNode()（例如Data）仍然可能带有输入定义，但是这些输入实际是不连边的。
      //  但是为了最小化修改，当前先不修改Data的定义，后续需要做调整，对与StartNode类型，不定义输入，
      //  或者认为没有输入的op就是start node，在定义IR时不需要再显式指定start node标记
      input_defs = &empty_input_defs;
    }
  } else {
    input_defs = &def_.GetInputDefs();
  }
  auto append_output_types = [&ss](size_t count) {
    for (size_t i = 0; i < count; ++i) {
      if (i != 0) {
        ss << ", ";
      }
      ss << "af::AscOpOutput";
    }
  };
  const auto &output_defs = def_.GetOutputDefs();
  const bool graph_used_in_instantiation = NeedGraphInInstantiation(has_optional_input);
  ss << "inline ";
  if (output_defs.size() > 1U) {
    ss << "std::tuple<";
    append_output_types(output_defs.size());
    ss << "> " << def_.GetType() << "(const char* name";
  } else {
    ss << "af::AscOpOutput " << def_.GetType() << "(const char* name";
  }
  if (graph_used_in_instantiation) {
    ss << ", af::AscGraph &graph";
  } else {
    for (const auto &input_def : *input_defs) {
      if (NeedConnectByInputArgs(has_optional_input, input_def)) {
        ss << ", const af::AscOpOutput &" << input_def.first << "_in";
      }
    }
  }
  const auto &attr_defs = def_.GetAttrDefs();
  for (const auto &attr_def : attr_defs) {
    ss << ", const " << attr_def.asc_ir_type << " &" << attr_def.name;
  }
  ss << ") {" << std::endl;
}
void ascir::FunctionGenerator::GenInstantiation(std::stringstream &ss, const bool graph_used_in_instantiation) const {
  if (graph_used_in_instantiation) {
    ss << "  const auto &op_ptr = std::make_shared<af::ascir_op::" << def_.GetType() << ">(name, graph);" << std::endl;
  } else {
    ss << "  const auto &op_ptr = std::make_shared<af::ascir_op::" << def_.GetType() << ">(name);" << std::endl;
  }
  ss << "  auto &op = *op_ptr;" << std::endl;
  ss << "  const auto &desc = OpDescUtils::GetOpDescFromOperator(op);" << std::endl;
  ss << "  desc->SetExtAttr(RELATED_OP, op_ptr);" << std::endl;
}
bool ascir::FunctionGenerator::GenConnectInputs(std::stringstream &ss, const bool has_optional_input) const {
  const auto &input_defs = def_.GetInputDefs();
  if (def_.IsStartNode()) {
    if (!has_optional_input) {
      // 这里与GenDefinition同理，传统StartNode的输入定义不参与连接，后续删除。
      return false;
    }
    bool connected = false;
    for (const auto &input_def : input_defs) {
      if (NeedConnectByInputArgs(has_optional_input, input_def)) {
        ss << "  op." << input_def.first << " = " << input_def.first << "_in;" << std::endl;
        connected = true;
      }
    }
    return connected;
  }
  if (!input_defs.empty()) {
    for (const auto &input_def : input_defs) {
      if (NeedConnectByInputArgs(has_optional_input, input_def)) {
        ss << "  op." << input_def.first << " = " << input_def.first << "_in;" << std::endl;
      }
    }
  }
  return !input_defs.empty();
}
bool ascir::FunctionGenerator::GenAttrAssignment(std::stringstream &ss) const {
  const auto &attr_defs = def_.GetAttrDefs();
  if (!attr_defs.empty()) {
    for (const auto &attr_def : attr_defs) {
      // 函数命名大驼峰，所以把属性名第一个字符转换成大写字母
      ss << "  op.ir_attr.Set" << CapitalizeFirstLetter(attr_def.name) << "(" << attr_def.name << ");" << std::endl;
    }
  }
  return !attr_defs.empty();
}

class StartNodeFuncGenerator : public FunctionGenerator {
 public:
  explicit StartNodeFuncGenerator(const AscIrDef &def) : FunctionGenerator(def) {}
  void Gen(std::stringstream &ss, const bool has_optional_input) const override {
    if (!def_.IsStartNode() || def_.GetOutputDefs().size() != 1UL) {
      return;
    }
    FunctionGenerator::Gen(ss, has_optional_input);
  }
  void GenDefinition(std::stringstream &ss, const bool has_optional_input) const override {
    // inline ascir::ops::OpType OpType
    ss << "inline "
       << "af::AscOpOutput " << ' ' << def_.GetType() << "(const char *name, af::AscGraph &graph, ge::DataType dt"
       << ", const std::vector<af::AxisId> &axis_ids"
       << ", const std::vector<af::Expression> &repeats"
       << ", const std::vector<af::Expression> &strides";

    if (has_optional_input) {
      for (const auto &input_def : def_.GetInputDefs()) {
        if (NeedConnectByInputArgs(has_optional_input, input_def)) {
          ss << ", const af::AscOpOutput &" << input_def.first << "_in";
        }
      }
    }

    const auto &attr_defs = def_.GetAttrDefs();
    for (const auto &attr_def : attr_defs) {
      ss << ", const " << attr_def.asc_ir_type << " &" << attr_def.name;
    }
    ss << ") {" << std::endl;
  }
  bool GenOutputsAssignment(std::stringstream &ss) const override {
    const auto &output_defs = def_.GetOutputDefs();
    const auto &output_name = output_defs[0].first;
    ss << "  op." << output_name << ".dtype = dt;" << std::endl;
    ss << "  *op." << output_name << ".axis = axis_ids;" << std::endl;
    ss << "  *op." << output_name << ".repeats = repeats;" << std::endl;
    ss << "  *op." << output_name << ".strides = strides;" << std::endl;
    return true;
  }

  void GenInstantiation(std::stringstream &ss, const bool) const override {
    FunctionGenerator::GenInstantiation(ss, true);
  }
};

class StoreNodeFuncGenerator : public FunctionGenerator {
 public:
  explicit StoreNodeFuncGenerator(const AscIrDef &def) : FunctionGenerator(def) {}
  void Gen(std::stringstream &ss, const bool has_optional_input) const override {
    (void)has_optional_input;
    if (def_.GetType() != "Store") {
      return;
    }
    ss << "inline "
       << "void" << ' ' << def_.GetType() << "(const char *name";
    ss << ", const af::AscOpOutput &" << "ub_in";
    ss << ", af::AscOpOutput &gm_output";
    const auto &attr_defs = def_.GetAttrDefs();
    for (const auto &attr_def : attr_defs) {
      ss << ", const " << attr_def.asc_ir_type << " &" << attr_def.name;
    }
    ss << ") {" << std::endl;
    ss << "  auto store_out = Store(name, ub_in";
    for (const auto &attr_def : attr_defs) {
      ss << ", " << attr_def.name;
    }
    ss << ");" << std::endl;
    ss << "  auto &gm_producer = const_cast<af::Operator &>(gm_output.GetOwnerOp());" << std::endl;
    ss << "  auto &store_op = const_cast<af::Operator &>(store_out.GetOwnerOp());" << std::endl;
    ss << "  gm_producer.SetInput(0U, store_op, 0U);" << std::endl;
    ss << "  AddEdgeForNode(store_op, 0U, gm_producer, 0U);" << std::endl;
    ss << "}" << std::endl;
  }
};

class ContiguousStartNodeFuncGenerator : FunctionGenerator {
 public:
  explicit ContiguousStartNodeFuncGenerator(const AscIrDef &def) : FunctionGenerator(def) {}
  void Gen(std::stringstream &ss, const bool has_optional_input) const override {
    if (!def_.IsStartNode() || def_.GetOutputDefs().size() != 1UL) {
      return;
    }
    FunctionGenerator::Gen(ss, has_optional_input);
  }
  void GenDefinition(std::stringstream &ss, const bool has_optional_input) const override {
    ss << "inline "
       << "af::AscOpOutput" << " Contiguous" << def_.GetType()
       << "(const char *name, af::AscGraph &graph, ge::DataType dt"
       << ", const std::vector<af::Axis> &axes";

    if (has_optional_input) {
      for (const auto &input_def : def_.GetInputDefs()) {
        if (NeedConnectByInputArgs(has_optional_input, input_def)) {
          ss << ", const af::AscOpOutput &" << input_def.first << "_in";
        }
      }
    }
    const auto &attr_defs = def_.GetAttrDefs();
    for (const auto &attr_def : attr_defs) {
      ss << ", const " << attr_def.asc_ir_type << " &" << attr_def.name;
    }
    ss << ") {" << std::endl;
  }
  bool GenOutputsAssignment(std::stringstream &ss) const override {
    const auto &output_name = def_.GetOutputDefs()[0].first;
    ss << "  op." << output_name << ".dtype = dt;" << std::endl;
    ss << "  op." << output_name << ".SetContiguousView(axes);" << std::endl;
    return true;
  }

  void GenInstantiation(std::stringstream &ss, const bool) const override {
    FunctionGenerator::GenInstantiation(ss, true);
  }
};

void GetHeaderGuarderFromPath(const char *path, std::stringstream &ss) {
  auto name = GetPureFileName(path);

  ss << "ASCIR_OPS_";

  while (*name != '\0') {
    auto c = toupper(*name++);
    if (c < 'A' || c > 'Z') {
      ss << '_';
    } else {
      ss << static_cast<char>(c);
    }
  }

  ss << '_';
}
}  // namespace

void GenFunc(const AscIrDef &def, std::stringstream &ss) {
  FunctionGenerator(def).Gen(ss, false);
  StartNodeFuncGenerator(def).Gen(ss, false);
  ContiguousStartNodeFuncGenerator(def).Gen(ss, false);
  StoreNodeFuncGenerator(def).Gen(ss, false);
  bool has_optional_input = false;
  const auto &input_defs = def.GetInputDefs();
  for (const auto &input_def : input_defs) {
    if (input_def.second == IrInputType::kIrInputOptional) {
      has_optional_input = true;
      break;
    }
  }
  if (has_optional_input) {
    FunctionGenerator(def).Gen(ss, true);
    StartNodeFuncGenerator(def).Gen(ss, true);
    ContiguousStartNodeFuncGenerator(def).Gen(ss, true);
    StoreNodeFuncGenerator(def).Gen(ss, true);
  }
}

void GenCalcBufFunc(std::stringstream &ss,
                    const std::map<std::tuple<std::string, int64_t, std::string>, AscIrDef> &ordered_keys_to_def) {
  std::stringstream ss_calc_tmp_buff_map;
  std::stringstream ss_calc_tmp_buff;
  for (auto &key_and_def : ordered_keys_to_def) {
    const auto &calc_func = key_and_def.second.GetCalcTmpBufSizeFunc();
    if (calc_func.func_name.empty()) {
      continue;
    }
    if (calc_func.func_type == CalcTmpBufSizeFuncType::CustomizeType) {
      ss_calc_tmp_buff << "extern std::vector<std::unique_ptr<af::TmpBufDesc>> ";
      ss_calc_tmp_buff << calc_func.func_name << "(const AscNode &Node);" << std::endl;
    }
    ss_calc_tmp_buff_map << "    {\"" << key_and_def.second.GetType() << "\", &";
    ss_calc_tmp_buff_map << calc_func.func_name << "}," << std::endl;
  }
  // 没有API注册时不生成CalcBuf函数
  if (ss_calc_tmp_buff_map.str().empty()) {
    return;
  }
  ss << ss_calc_tmp_buff.str();
  ss << "inline std::vector<std::unique_ptr<af::TmpBufDesc>> CalcAscNodeTmpSize(const af::AscNode &node) {"
     << std::endl;
  ss << "  typedef std::vector<std::unique_ptr<af::TmpBufDesc>> (*calc_func_ptr) (const af::AscNode &node);"
     << std::endl;
  ss << "  static const std::unordered_map<std::string, calc_func_ptr> node_calc_tmp_buff_map = {" << std::endl;
  ss << ss_calc_tmp_buff_map.str();
  ss << "  };" << std::endl;
  ss << "  af::AscNodeAttr attr = node.attr;" << std::endl;
  ss << "  if (node_calc_tmp_buff_map.find(attr.type) != node_calc_tmp_buff_map.end()) {" << std::endl;
  ss << "    return node_calc_tmp_buff_map.at(node.attr.type)(node);" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return std::vector<std::unique_ptr<af::TmpBufDesc>>();" << std::endl;
  ss << "}" << std::endl;
}

void GenCommonInferDtypeBaseFunc(
    std::stringstream &ss, const std::map<std::tuple<std::string, int64_t, std::string>, AscIrDef> &ordered_keys_to_def,
    const string &extra_str = "") {
  std::stringstream func_table;
  for (const auto &key_and_def : ordered_keys_to_def) {
    const auto &node_type = key_and_def.second.GetType();
    func_table << "    {\"" << node_type << "\", ";
    func_table << "::af::ascir_op::" << node_type << "::InferDataType" << extra_str << "}," << std::endl;
  }
  if (func_table.str().empty()) {
    return;
  }
  ss << "inline af::Status CommonInferDtype" << extra_str
     << "(const std::string &type, const std::vector<DataType> &input_dtypes,\n"
        "                                   std::vector<DataType> &expect_output_dtypes,\n"
        "                                   const std::string &npu_arch) {"
     << std::endl;
  ss << "  using func = af::Status (*)(const std::vector<DataType> &input_dtypes, \n"
        "                              std::vector<DataType> &expect_output_dtypes,\n"
        "                              const std::string &npu_arch);"
     << std::endl;
  ss << "  static const std::unordered_map<std::string, func> func_table = {" << std::endl;
  ss << func_table.str();
  ss << "  };" << std::endl;
  ss << "  const auto &iter = func_table.find(type);" << std::endl;
  ss << "  if (iter != func_table.end()) {" << std::endl;
  ss << "    return iter->second(input_dtypes, expect_output_dtypes, npu_arch);" << std::endl;
  ss << "  }" << std::endl;
  ss << "  GELOGW(\"Node type %s is not supported to infer for now!\", type.c_str());" << std::endl;
  ss << "  return af::FAILED;" << std::endl;
  ss << "}" << std::endl;
}

void GenCommonInferDtypeFunc(
    std::stringstream &ss,
    const std::map<std::tuple<std::string, int64_t, std::string>, AscIrDef> &ordered_keys_to_def) {
  GenCommonInferDtypeBaseFunc(ss, ordered_keys_to_def);
}

void GenCommonInferDtypeWithNoCheckFunc(
    std::stringstream &ss,
    const std::map<std::tuple<std::string, int64_t, std::string>, AscIrDef> &ordered_keys_to_def) {
  GenCommonInferDtypeBaseFunc(ss, ordered_keys_to_def, "WithNoCheck");
}

void GenAll(std::stringstream &ss) {
  std::stringstream ss_asc_ir;
  std::stringstream ss_ge_ir;

  ss << R"(#include "ascendc_ir/utils/cg_calc_tmp_buff_common_funcs.h")" << std::endl << std::endl;
  ss << R"(#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h")" << std::endl << std::endl;
  ss << R"(#include "ascendc_ir/ascend_reg_ops.h")" << std::endl << std::endl;
  ss << R"(#include "utils/cg_utils.h")" << std::endl << std::endl;
  ss << R"(#include "graph/type/tensor_type_impl.h")" << std::endl << std::endl;
  ss << R"(#include "graph/type/sym_dtype.h")" << std::endl << std::endl;
  ss << "#include <variant>" << std::endl;
  ss << "#include <type_traits>" << std::endl;
  ss << "#include <tuple>" << std::endl << std::endl;

  std::map<std::tuple<std::string, int64_t, std::string>, AscIrDef> ordered_keys_to_def;
  for (const auto &type_and_def : AscirRegistry::GetInstance().GetAll()) {
    ordered_keys_to_def[std::make_tuple(type_and_def.second.GetFilePath(), type_and_def.second.GetLine(),
                                        type_and_def.first)] = type_and_def.second;
  }

  for (const auto &key_and_def : ordered_keys_to_def) {
    ss << "// Defined at " << GetPureFileName(key_and_def.second.GetFilePath().c_str()) << ':'
       << key_and_def.second.GetLine() << std::endl;
    GenIrComment(key_and_def.second, ss);
    ss << "namespace af {" << std::endl;
    GenAscIr(key_and_def.second, ss);
    ss << "}" << std::endl << std::endl;  // namespace af
  }

  ss << "namespace af {" << std::endl;
  ss << "namespace ascir {" << std::endl;
  ss << "namespace cg {" << std::endl;
  for (auto &key_and_def : ordered_keys_to_def) {
    const auto &ascir_def = key_and_def.second;
    // 增加判断, 如果是动态输出, continue;
    if (ascir_def.HasDynamicOutput()) {
      continue;
    }

    GenFunc(key_and_def.second, ss);
    // 如果有node属性配置，重载一个不设置属性的构造函数，把属性变成可选
    if (!ascir_def.GetAttrDefs().empty()) {
      ascir_def.MutableAttrDefs().clear();
      GenFunc(ascir_def, ss);
    }
  }

  ss << "}" << std::endl;  // namespace cg
  GenCalcBufFunc(ss, ordered_keys_to_def);
  GenCommonInferDtypeFunc(ss, ordered_keys_to_def);
  GenCommonInferDtypeWithNoCheckFunc(ss, ordered_keys_to_def);
  ss << "}" << std::endl;               // namespace ascir
  ss << "}" << std::endl << std::endl;  // namespace af
  AscirRegistry::GetInstance().ClearAll();
}

void GenHeaderFileToStream(const char *path, std::stringstream &ss) {
  std::stringstream ss_header_guarder;
  GetHeaderGuarderFromPath(path, ss_header_guarder);
  auto guarder = ss_header_guarder.str();

  ss << "// Generated from asc-ir definition files, "
        "any modification made to this file may be overwritten after compile."
     << std::endl;
  ss << "// If you want to add self-defined asc-ir, please create a separated header file." << std::endl;
  ss << "#ifndef " << guarder << std::endl;
  ss << "#define " << guarder << std::endl << std::endl;

  GenAll(ss);

  ss << "#endif  // " << guarder << std::endl;
}

int GenHeaderFile(const char *path) {
  std::stringstream ss;
  GenHeaderFileToStream(path, ss);
  AscirRegistry::GetInstance().ClearAll();
  std::ofstream fs(path);
  if (!fs) {
    return -1;
  }
  fs << ss.str();
  fs.close();
  return 0;
}
}  // namespace ascir
}  // namespace af
