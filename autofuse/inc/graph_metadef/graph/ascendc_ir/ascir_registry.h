/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_ASCIR_REGISTRY_H
#define AUTOFUSE_ASCIR_REGISTRY_H
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <algorithm>
#include "ascendc_ir/ascendc_ir_check.h"
#include "graph/types_af.h"
#include "op_desc.h"
#include "ir/ir_data_type_symbol_store.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace af {
namespace ascir {
#ifdef AUTOFUSE_USE_GE_METADEF
using OrderedTensorTypeList = ge::OrderedTensorTypeList;
using TensorType = ge::TensorType;
using IrInputType = ge::IrInputType;
using IrOutputType = ge::IrOutputType;
using IRDataTypeSymbolStore = ge::IRDataTypeSymbolStore;
#endif
using ApplyOutputView = std::function<std::string(const std::string &var)>;
struct ViewPolicy {
 public:
  enum ViewType : int64_t {
    kElementWise = 0,
    kReduce,
    kBroadCast,
    kInvalid,
  };
  ViewPolicy(uint32_t element_wise_input_index) : use_input_index(element_wise_input_index) {
    view_type = kElementWise;
  }
  ViewPolicy(uint32_t reduce_input_index, std::string reduce_axis_name)
      : use_input_index(reduce_input_index), reduce_axis_attr_name(std::move(reduce_axis_name)) {
    view_type = kReduce;
  }

  explicit ViewPolicy(std::vector<uint32_t> broad_cast_in_indexs)
      : broad_cast_input_indexs(std::move(broad_cast_in_indexs)) {
    view_type = kBroadCast;
  }

  ViewType view_type{kInvalid};
  uint32_t use_input_index{UINT32_MAX};
  std::string reduce_axis_attr_name;
  std::vector<uint32_t> broad_cast_input_indexs;
};

inline ViewPolicy ReduceView(uint32_t index, const std::string &attr_name) {
  return ViewPolicy(index, attr_name);
}
inline ViewPolicy BroadCastView(const std::vector<uint32_t> &broad_cast_input_indexs) {
  return ViewPolicy(broad_cast_input_indexs);
}

struct DtypePolicy {
  enum PolicyType : int64_t {
    kUseInput = 0,
    kPromptInput,
    kUseDtype,
    kInvalid,
  };

 public:
  DtypePolicy(uint32_t use_in_index) : use_input_index(use_in_index) {
    policy_type = kUseInput;
  };
  DtypePolicy(ge::DataType dtype) : data_type(dtype) {
    policy_type = kUseDtype;
  };
  PolicyType policy_type{kInvalid};
  uint32_t use_input_index{UINT32_MAX};
  ge::DataType data_type{ge::DataType::DT_UNDEFINED};
};

inline DtypePolicy PromptDtype(uint32_t index) {
  auto policy = DtypePolicy(index);
  policy.policy_type = DtypePolicy::kPromptInput;
  return policy;
}
// TODO: c++的类ABI兼容性不好，后面考虑换成C接口实现
struct AscIrAttrDef {
  std::string name;
  std::string asc_ir_type;
  std::string ge_ir_type;
};
enum CalcTmpBufSizeFuncType : int64_t {
  CommonType = 0,
  CustomizeType,
};
struct CalcTmpBufSizeFunc {
  std::string func_name;
  CalcTmpBufSizeFuncType func_type = CalcTmpBufSizeFuncType::CommonType;
  CalcTmpBufSizeFunc() = default;
  CalcTmpBufSizeFunc(std::string name, const CalcTmpBufSizeFuncType type)
      : func_name(std::move(name)), func_type(type) {}
};

class AscIrCodegen {
 public:
  virtual ~AscIrCodegen() = default;
  virtual std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) {
    (void)node;
    return std::vector<std::unique_ptr<TmpBufDesc>>();
  }
  virtual std::string GetApiTilingTypeName() const {
    return "";
  }

  virtual uint32_t GetInstNum() const {
    return 0U;
  }

  // 返回api call类的名称
  virtual std::string GetApiCallName() const {
    return "";
  }

  // 返回api的名称
  virtual std::string GetApiName() const {
    return "";
  }

  // 返回需要加载的头文件
  virtual std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const {
    return std::vector<std::string>();
  }

  virtual bool IsVectorFunctionSupported(const AscNode &node) const {
    (void)node;
    return false;
  }
  virtual bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const {
    (void)is_scalar_list;
    return false;
  }
  virtual bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const {
    (void)is_scalar_list;
    return false;
  }

  virtual bool IsInplaceSupported(const AscNode &node) const {
    (void)node;
    return false;
  }

  virtual bool IsBrcInlineSupported(const AscNode &node) const {
    (void)node;
    return false;
  }

  // 返回需要包含的头文件
  virtual std::vector<std::string> IncludeApiHeaderFiles() const {
    return {};
  }

  // 如果需要插入cast节点，返回cast的目的类型
  virtual std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(const AscNode &node) {
    std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> conversion_dtype;
    AscNodeInputs node_inputs = node.inputs;
    AscNodeOutputs node_outputs = node.outputs;
    for (size_t i = 0; i < node_inputs().size(); i++) {
      conversion_dtype.first.emplace_back(node_inputs[i].attr.dtype);
    }
    for (size_t i = 0; i < node_outputs().size(); i++) {
      conversion_dtype.second.emplace_back(node_outputs[i].attr.dtype);
    }
    return conversion_dtype;
  }

  virtual bool IsNodeValid(const AscNode &node) const {
    (void)node;
    return true;
  }

  // SIMT 实现需预留的 d-cache 字节数，参与动态 UB 预算与 tiling 预留；非 SIMT 实现返回 0
  virtual int64_t GetSimtDcacheSize(const AscNode &node) const {
    (void)node;
    return 0;
  }
};

class AscIrAtt {
 public:
  virtual ~AscIrAtt() = default;
  // 最内轴建议对齐值（默认32B对齐）
  virtual uint32_t GetInnerDimPromptAlignSize() const {
    return 32U;
  }
  // 最外轴建议对齐值(默认为1，表示对外轴无对齐要求)
  virtual uint32_t GetOuterDimPromptAlignSize() const {
    return 1U;
  }
  // 返回ASCIR API接口性能公式函数(不同硬件的ASCIR实现存在差异，性能公式形式存在差异)
  virtual void *GetApiPerf() const = 0;

  // 返回AscendCApi的性能公式(不同硬件的性能公式参数存在差异)
  virtual void *GetAscendCApiPerfTable() const = 0;
};

template <typename T>
std::function<std::unique_ptr<T>()> AscIrImplCreator() {
  return []() { return std::unique_ptr<T>(new T()); };
}

using AscIrAttCreator = std::function<std::unique_ptr<AscIrAtt>()>;
using AscIrCodegenCreator = std::function<std::unique_ptr<AscIrCodegen>()>;

struct AscIrImpl {
  AscIrAttCreator att;
  AscIrCodegenCreator codegen;
  std::vector<std::pair<std::string, OrderedTensorTypeList>> support_dtypes;
};

struct AscIrImplV2 {
  AscIrAttCreator att;
  AscIrCodegenCreator codegen;
  std::vector<std::pair<std::string, TensorType>> support_dtypes;
};

struct AscIrDefImpl;
class AscIrDef {
 public:
  AscIrDef();
  using CodeGenerator = void (*)(const AscIrDef &def, std::stringstream &ss);
  bool IsAttrExisted(const std::string &attr_name) const;

  void Init(const char *type, const char *def_file_path, int64_t line) const;

  const std::vector<std::pair<std::string, IrInputType>> &GetInputDefs() const;
  const std::vector<std::pair<std::string, IrOutputType>> &GetOutputDefs() const;
  bool HasDynamicOutput() const;
  void AppendInput(const string &name, IrInputType type) const;
  void AppendOutput(const string &name, IrOutputType type) const;
  void StoreInputIrSymName(const std::string &ir_name, const std::string &sym_name) const;
  void StoreOutputIrSymName(const std::string &ir_name, const std::string &sym_name) const;
  const std::string &GetType() const;
  void StartNode() const;
  bool IsStartNode() const;
  void SetAttr(const std::string &name, const std::string &asc_type, const std::string &ge_type) const;
  void SetDtypePolicy(const std::vector<DtypePolicy> &output_dtypes_policy) const;
  const std::vector<DtypePolicy> &GetOutputDtypePolicy() const;
  void SetViewPolicy(const std::vector<ViewPolicy> &view_policy) const;
  const std::vector<ViewPolicy> &GetViewPolicy() const;
  void SetApiTilingDataName(const std::string &tiling_data_name) const;
  const string &GetApiTilingDataName() const;
  void SetCalcTmpBufSizeFunc(const std::string &calc_tmp_buf_size_func, CalcTmpBufSizeFuncType type) const;
  const CalcTmpBufSizeFunc &GetCalcTmpBufSizeFunc() const;
  const std::vector<AscIrAttrDef> &GetAttrDefs() const;
  std::vector<AscIrAttrDef> &MutableAttrDefs() const;
  void SetComment(const string &comment) const;
  const string &GetComment() const;
  const std::string &GetFilePath() const;
  int64_t GetLine() const;
  IRDataTypeSymbolStore &MutableDataTypeSymbolStore() const;
  const IRDataTypeSymbolStore &GetDataTypeSymbolStore() const;
  const std::map<std::string, IRDataTypeSymbolStore> &GetSocToDataTypeSymbolStore() const;
  void AddSocImpl(const std::vector<std::string> &soc_versions, const AscIrImpl &impl) const;
  void AddSocImplV2(const std::vector<std::string> &soc_versions, const AscIrImplV2 &impl) const;
  void AppendSocImpl(const AscIrDef &ir_def) const;
  size_t GetSocImplSize() const;
  CodeGenerator infer_data_type_generator{nullptr};
  CodeGenerator infer_view_generator{nullptr};
  std::unique_ptr<AscIrAtt> GetAscIrAttImpl(const std::string &soc_version);
  std::unique_ptr<AscIrCodegen> GetAscIrCodegenImpl(const std::string &soc_version);
  void SetComputeType(ComputeType compute_type) const;
  ComputeType GetComputeType() const;

 private:
  friend class AscirRegister;
  std::shared_ptr<AscIrDefImpl> impl_;
};

inline std::string UpdateViewIfCrossLoop(const std::string &trans_infos, const std::string &input_api_sched_axis,
                                         const std::string &op_attr_sched_axis, const std::string &tie_expression) {
  return "AxisUtils::UpdateViewIfCrossLoop(" + trans_infos + ", " + input_api_sched_axis + ", " + op_attr_sched_axis +
         ", " + "std::move(" + tie_expression + "))";
}

inline void GenChosenInputView(const AscIrDef &def, const uint32_t chosen_input_index, std::stringstream &ss) {
  const auto &input_defs = def.GetInputDefs();
  ss << input_defs[chosen_input_index].first + "_tmp = " << "{*" << input_defs[chosen_input_index].first
     << "_in.axis, *" << input_defs[chosen_input_index].first << "_in.repeats, *"
     << input_defs[chosen_input_index].first << "_in.strides};" << std::endl;
}
template <class Policy>
void GenErrorIfPolicyInvalid(Policy policy, size_t range, std::stringstream &ss) {
  if (policy.use_input_index < range) {
    return;
  }
  ss << "Policy is invalid as use_input_index :" << policy.use_input_index
     << " should be less than input size:" << range << std::endl;
}

inline void DefineChosenInputView(const AscIrDef &def, const ViewPolicy &policy, uint32_t &chosen_input_index,
                                  std::unordered_set<uint32_t> &chosen_input_index_set, std::stringstream &ss) {
  const auto &input_defs = def.GetInputDefs();
  ss << "  // set tmp view to store input view and apply view transform" << std::endl;
  const std::string view_type("View ");
  GenErrorIfPolicyInvalid(policy, input_defs.size(), ss);
  chosen_input_index = policy.use_input_index;
  ss << "  ";
  if (chosen_input_index_set.insert(chosen_input_index).second) {
    ss << view_type;
  }
  GenChosenInputView(def, chosen_input_index, ss);
}

inline void SameDataTypeFromInput(const AscIrDef &def, std::stringstream &ss, const char *input_name) {
  const auto &output_defs = def.GetOutputDefs();
  for (const auto &output_def : output_defs) {
    ss << "  op." << output_def.first << ".dtype = static_cast<ge::DataType>(" << input_name << "_in.dtype);"
       << std::endl;
  }
}

inline void GenerateViewUpdateCode(const AscIrDef &def, const std::pair<size_t, size_t> out_to_chosen_input,
                                   const ApplyOutputView &apply_output_view, std::stringstream &ss,
                                   bool &gen_trans_infos_instance) {
  const auto &input_defs = def.GetInputDefs();
  const auto &output_defs = def.GetOutputDefs();
  const size_t output_index = out_to_chosen_input.first;
  const size_t chosen_input_index = out_to_chosen_input.second;
  if (!gen_trans_infos_instance) {
    ss << "  auto trans_infos = CodeGenUtils::GetOwnerGraphAscAttr(op." << output_defs[output_index].first
       << ".GetOwnerOp())" << "->trans_info_road;" << std::endl;
    gen_trans_infos_instance = true;
  }

  const std::string which_input_api_sched_axis = output_defs[output_index].first + "_in_api_sched_axis";
  ss << "  auto " << which_input_api_sched_axis << " = CodeGenUtils::GetOwnerOpAscAttr("
     << input_defs[chosen_input_index].first << "_in.GetOwnerOp())"
     << "->sched.axis;" << std::endl;
  std::string view = input_defs[chosen_input_index].first + "_tmp";
  ss << "  {" << std::endl << "    const auto &[axes, repeats, strides] = ";
  std::string val =
      UpdateViewIfCrossLoop("trans_infos", which_input_api_sched_axis, "op.attr.sched.axis", view).append(".second");
  // 应用输出的语义变换
  if (!(apply_output_view == nullptr)) {
    ss << apply_output_view(val) << ";" << std::endl;
  } else {
    ss << val << ";" << std::endl;
  }
  ss << "    std::tie(*op." << output_defs[output_index].first << ".axis, *op." << output_defs[output_index].first
     << ".repeats, *op." << output_defs[output_index].first << ".strides) = std::make_tuple(axes, repeats, strides);"
     << std::endl
     << "  }" << std::endl;
}

inline ApplyOutputView GenApplyOutputViewFunc(const AscIrDef &def, const size_t output_index,
                                              uint32_t &chosen_input_index, std::stringstream &ss) {
  (void)chosen_input_index;
  const auto &output_views_policy = def.GetViewPolicy();
  const auto &policy = output_views_policy[output_index];
  ApplyOutputView apply_output_view;
  switch (policy.view_type) {
    case ViewPolicy::kElementWise:
      break;
    case ViewPolicy::kReduce:
      if (!def.IsAttrExisted(policy.reduce_axis_attr_name)) {
        return apply_output_view;
      }
      apply_output_view = [&output_views_policy, output_index](const std::string &var) -> std::string {
        return "AxisUtils::ReduceView(" + var + ", " + output_views_policy[output_index].reduce_axis_attr_name + ")";
      };
      break;
    case ViewPolicy::kBroadCast:  // TODO 广播代码后续支持
    case ViewPolicy::kInvalid:
    default:
      ss << "unsupported policy type: " << policy.view_type << std::endl;
      break;
  }
  return apply_output_view;
}

inline void InferViewByPolicy(const AscIrDef &def, std::stringstream &ss) {
  const auto &output_defs = def.GetOutputDefs();
  const auto &output_views_policy = def.GetViewPolicy();
  if (output_defs.size() != output_views_policy.size()) {
    std::string error_info = std::string("view_policy's size ")
                                 .append(std::to_string(output_views_policy.size()))
                                 .append(" should be equal with output_defs's size ")
                                 .append(std::to_string(output_defs.size()));
    ss << error_info;
    return;
  }
  bool gen_trans_infos_instance = false;
  std::unordered_set<uint32_t> chosen_input_index_set;
  for (size_t output_index = 0U; output_index < output_views_policy.size(); ++output_index) {
    uint32_t chosen_input_index = 0U;
    DefineChosenInputView(def, output_views_policy[output_index], chosen_input_index, chosen_input_index_set, ss);
    GenerateViewUpdateCode(def, std::make_pair(output_index, chosen_input_index),
                           GenApplyOutputViewFunc(def, output_index, chosen_input_index, ss), ss,
                           gen_trans_infos_instance);
  }
}

inline void InferDtypeByPolicy(const AscIrDef &def, std::stringstream &ss) {
  const auto &output_defs = def.GetOutputDefs();
  const auto &output_dtypes_policy = def.GetOutputDtypePolicy();
  if (output_defs.size() != output_dtypes_policy.size()) {
    std::string error_info = std::string("dtype_policy's size ")
                                 .append(std::to_string(output_dtypes_policy.size()))
                                 .append("should be equal with output_defs's size ")
                                 .append(std::to_string(output_defs.size()));
    ss << error_info;
    return;
  }
  const auto &input_defs = def.GetInputDefs();
  for (size_t output_index = 0U; output_index < output_dtypes_policy.size(); ++output_index) {
    const auto &policy = output_dtypes_policy[output_index];
    switch (policy.policy_type) {
      case DtypePolicy::kUseInput:
        GenErrorIfPolicyInvalid(policy, input_defs.size(), ss);
        ss << "  op." << output_defs[output_index].first << ".dtype = static_cast<ge::DataType>("
           << input_defs[policy.use_input_index].first << "_in.dtype);" << std::endl;
        break;
      case DtypePolicy::kPromptInput:
        GenErrorIfPolicyInvalid(policy, input_defs.size(), ss);
        ss << "  op." << output_defs[output_index].first
           << ".dtype = DtypeTransformUtils::Prompt(static_cast<ge::DataType>("
           << input_defs[policy.use_input_index].first << "_in.dtype));" << std::endl;
        break;
      case DtypePolicy::kUseDtype:
        ss << "  op." << output_defs[output_index].first << ".dtype = static_cast<ge::DataType>(" << policy.data_type
           << ");" << std::endl;
        break;
      case DtypePolicy::kInvalid:
      default:
        ss << "unsupported policy type: " << policy.policy_type << std::endl;
    }
  }
}

inline void SameDataTypeFromFirstInput(const AscIrDef &def, std::stringstream &ss) {
  const auto &input_defs = def.GetInputDefs();
  if (!input_defs.empty()) {
    SameDataTypeFromInput(def, ss, input_defs[0].first.c_str());
  }
}
inline void SameDataTypeFromSecondInput(const AscIrDef &def, std::stringstream &ss) {
  const auto &input_defs = def.GetInputDefs();
  if (input_defs.size() > 1U) {
    SameDataTypeFromInput(def, ss, input_defs[1].first.c_str());
  }
}
inline void SameViewFromInput(const AscIrDef &def, std::stringstream &ss, const char *input_name) {
  const auto &output_defs = def.GetOutputDefs();
  for (const auto &output_def : output_defs) {
    ss << "  op." << output_def.first << ".axis = " << input_name << "_in.axis;" << std::endl;
    ss << "  op." << output_def.first << ".repeats = " << input_name << "_in.repeats;" << std::endl;
    ss << "  op." << output_def.first << ".strides = " << input_name << "_in.strides;" << std::endl;
  }
}
inline void SameViewFromFirstInput(const AscIrDef &def, std::stringstream &ss) {
  const auto &input_defs = def.GetInputDefs();
  if (!input_defs.empty()) {
    SameViewFromInput(def, ss, input_defs[0].first.c_str());
  }
}

class AscirRegistry {
 public:
  static AscirRegistry &GetInstance();
  void RegisterAscIr(const std::string &type, const AscIrDef &def);

  const std::unordered_map<std::string, AscIrDef> &GetAll() const;
  std::unique_ptr<AscIrAtt> GetIrAttImpl(const std::string &soc_version, const std::string &type);
  std::unique_ptr<AscIrCodegen> GetIrCodegenImpl(const std::string &soc_version, const std::string &type);
  void ClearAll();

 private:
  std::unordered_map<std::string, AscIrDef> types_to_ascir_;
};
}  // namespace ascir
}  // namespace af
#endif  // AUTOFUSE_ASCIR_REGISTRY_H
