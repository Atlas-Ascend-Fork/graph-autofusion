/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/type/sym_dtype.h"
#include "common/checker.h"
#include "graph/utils/attr_utils.h"
#include "graph/type/tensor_type_impl.h"
#include "graph/types_af.h"
#include "graph/utils/type_utils.h"
#include "op_common/data_type_utils.h"

namespace af {

namespace {
graphStatus GetDtypeFromAttr(const OpDesc &op, const std::string &attr, DataType &dtype) {
  GELOGI("Trying to get dtype from attr %s of op %s", attr.c_str(), op.GetName().c_str());
  if (AttrUtils::GetDataType(op, attr, dtype)) {
    return GRAPH_SUCCESS;
  }
  int32_t numeric_dtype = -1;
  if (AttrUtils::GetInt(op, attr, numeric_dtype)) {
    GE_WARN_ASSERT(numeric_dtype >= 0 && numeric_dtype < DT_MAX,
                   "Invalid numeric dtype %d for sym %s of op %s, it should be in [0, %d)", numeric_dtype, attr.c_str(),
                   op.GetName().c_str(), DT_MAX);
    dtype = static_cast<DataType>(numeric_dtype);
    return GRAPH_SUCCESS;
  }
  GELOGW("Op %s has no attr named %s", op.GetName().c_str(), attr.c_str());
  return GRAPH_FAILED;
}

graphStatus GetListDtypeFromAttr(const OpDesc &op, const std::string &attr, std::vector<DataType> &dtypes) {
  GELOGI("Trying to get list-dtype from attr %s of op %s", attr.c_str(), op.GetName().c_str());
  if (AttrUtils::GetListDataType(op, attr, dtypes)) {
    return GRAPH_SUCCESS;
  }
  std::vector<int32_t> numeric_dtypes;
  if (AttrUtils::GetListInt(op, attr, numeric_dtypes)) {
    for (auto &numeric_dtype : numeric_dtypes) {
      GE_WARN_ASSERT(numeric_dtype >= 0 && numeric_dtype < DT_MAX,
                     "Invalid numeric dtype %d for sym %s of op %s, it should be in [0, %d)", numeric_dtype,
                     attr.c_str(), op.GetName().c_str(), DT_MAX);
      dtypes.push_back(static_cast<DataType>(numeric_dtype));
    }
    return GRAPH_SUCCESS;
  }
  GELOGW("Op %s has no attr named %s", op.GetName().c_str(), attr.c_str());
  return GRAPH_FAILED;
}

std::string ToString(const TensorType &types) {
  std::string s = "[";
  for (auto &dtype : types.tensor_type_impl_->GetMutableDateTypeSet()) {
    s += TypeUtils::DataTypeToSerialString(dtype);
    s += ",";
  }
  s += "]";
  return s;
}

const char *ToString(const IrInputType &type) {
  if (type == kIrInputRequired) {
    return "Required";
  }
  if (type == kIrInputOptional) {
    return "Optional";
  }
  if (type == kIrInputDynamic) {
    return "Dynamic";
  }
  return "Unknown";
}

graphStatus PromoteDtype(const DataType &left, const DataType &right, DataType &promoted_dtype) {
  GE_WARN_ASSERT(left >= 0 && left < DT_MAX, "Invalid left dtype %d, it should be in [0, %d)", left, DT_MAX);
  GE_WARN_ASSERT(right >= 0 && right < DT_MAX, "Invalid right dtype %d, it should be in [0, %d)", right, DT_MAX);

  promoted_dtype = opcommon::PromoteType(left, right);
  GELOGD("Promoted dtype %s from %s and %s", TypeUtils::DataTypeToSerialString(promoted_dtype).c_str(),
         TypeUtils::DataTypeToSerialString(left).c_str(), TypeUtils::DataTypeToSerialString(right).c_str());
  return GRAPH_SUCCESS;
}

graphStatus PromoteDtype(const TypeOrTypes &left, const TypeOrTypes &right, TypeOrTypes &promoted_dtype) {
  GE_WARN_ASSERT(left.IsListType() == right.IsListType(), "Trying to promote %s with %s", left.DebugString().c_str(),
                 right.DebugString().c_str());
  if (left.IsListType()) {
    std::vector<DataType> left_dtypes;
    std::vector<DataType> right_dtypes;

    GE_WARN_ASSERT_GRAPH_SUCCESS(left.GetTypes(left_dtypes));
    GE_WARN_ASSERT_GRAPH_SUCCESS(right.GetTypes(right_dtypes));

    GE_WARN_ASSERT(left_dtypes.size() == right_dtypes.size(), "Trying to promote %s with %s",
                   left.DebugString().c_str(), right.DebugString().c_str());

    std::vector<DataType> data_types;
    data_types.resize(left_dtypes.size());
    for (size_t i = 0U; i < left_dtypes.size(); i++) {
      GE_WARN_ASSERT_GRAPH_SUCCESS(PromoteDtype(left_dtypes[i], right_dtypes[i], data_types[i]));
    }
    promoted_dtype.SetTypes(data_types);
    return GRAPH_SUCCESS;
  }

  DataType left_dtype;
  DataType right_dtype;
  GE_WARN_ASSERT_GRAPH_SUCCESS(left.GetType(left_dtype));
  GE_WARN_ASSERT_GRAPH_SUCCESS(right.GetType(right_dtype));

  DataType dtype;
  GE_WARN_ASSERT_GRAPH_SUCCESS(PromoteDtype(left_dtype, right_dtype, dtype));
  promoted_dtype.SetType(dtype);
  return GRAPH_SUCCESS;
}
}  // namespace

graphStatus TypeOrTypes::GetType(DataType &type) const {
  if (!initialized_ || is_list_ || types_.empty()) {
    return GRAPH_FAILED;
  }
  type = types_[0];
  return GRAPH_SUCCESS;
}

graphStatus TypeOrTypes::GetTypes(std::vector<DataType> &types) const {
  if (!initialized_ || !is_list_) {
    return GRAPH_FAILED;
  }
  types = types_;
  return GRAPH_SUCCESS;
}

const DataType &TypeOrTypes::UnsafeGetType() const {
  if (!initialized_ || is_list_ || (types_.size() != 1)) {
    const static DataType kUndefined = DT_UNDEFINED;
    return kUndefined;
  }
  return types_[0];
}

const std::vector<DataType> &TypeOrTypes::UnsafeGetTypes() const {
  if (!initialized_ || !is_list_) {
    const static std::vector<DataType> kUndefined{};
    return kUndefined;
  }
  return types_;
}

void TypeOrTypes::SetType(const DataType &type) {
  initialized_ = true;
  is_list_ = false;
  types_.clear();
  types_.emplace_back(type);
}

void TypeOrTypes::SetTypes(const std::vector<DataType> &types) {
  initialized_ = true;
  is_list_ = true;
  types_ = types;
}

std::string TypeOrTypes::DebugString() const {
  if (!initialized_) {
    return "Uninitialized";
  }
  std::string ret = is_list_ ? "List[" : "";
  for (auto &type : types_) {
    ret += TypeUtils::DataTypeToSerialString(type) + ",";
  }
  if (is_list_) {
    ret += "]";
  }
  return ret;
}

// 不使用DATATYPE指定sym的取值范围时，sym的取值范围为所有数据类型
SymDtype::SymDtype(const std::string &id)
    : id_(id),
      is_legacy_(true),
      is_list_(false),
      tensor_type_({}),
      is_ordered_list_(false),
      ordered_tensor_type_list_({}) {}

const std::string &SymDtype::Id() const {
  return id_;
}

bool SymDtype::IsLegacy() const {
  return is_legacy_;
}

void SymDtype::BindIrInput(const std::string &ir_input, const IrInputType &input_type, size_t input_index) {
  ir_inputs_.emplace_back(ir_input, input_type, input_index);
}

void SymDtype::BindAllowedDtypes(const TensorType &types) {
  is_legacy_ = false;
  is_list_ = false;
  tensor_type_ = types;
}

void SymDtype::BindAllowedDtypes(const ListTensorType &types) {
  is_legacy_ = false;
  is_list_ = true;
  tensor_type_ = types.tensor_type;
}

void SymDtype::BindExpression(const std::shared_ptr<SymDtypeExpression> &expression) {
  is_legacy_ = false;
  expression_ = expression;
}

bool SymDtype::IsListType() const {
  if (expression_ != nullptr) {
    return expression_->IsListType();
  }
  return is_list_;
}

const std::string &SymDtype::DebugString() const {
  std::string ret = id_ + ":";
  ret += (is_list_ ? "List" : "Oneof");
  ret += ToString(tensor_type_);
  return id_;
}

std::vector<size_t> SymDtype::GetDirectIrInputIndexes() const {
  std::vector<size_t> ir_input_indexes;
  ir_input_indexes.reserve(ir_inputs_.size());
  for (const auto &ir_input : ir_inputs_) {
    ir_input_indexes.push_back(ir_input.index);
  }
  return ir_input_indexes;
}

std::vector<size_t> SymDtype::GetIrInputIndexes() const {
  if (expression_ == nullptr) {
    return GetDirectIrInputIndexes();
  } else {
    return expression_->GetIrInputIndexes();
  }
}

ExpressionType SymDtype::Type() const {
  if (expression_ == nullptr) {
    return ExpressionType::kSingle;
  } else {
    return expression_->Type();
  }
}

graphStatus SymDtype::Eval(const OpDesc &op, TypeOrTypes &type_or_types) const {
  GE_WARN_ASSERT(!is_legacy_, "Trying to eval legacy sym dtype %s", id_.c_str());
  if (expression_ != nullptr) {
    GELOGI("Eval sym dtype %s from expression of op %s", id_.c_str(), op.GetType().c_str());
    return expression_->Eval(op, type_or_types);
  }

  if (IsListType()) {
    std::vector<DataType> dtypes;
    GE_WARN_ASSERT_GRAPH_SUCCESS(Eval(op, dtypes));
    type_or_types.SetTypes(dtypes);
    return GRAPH_SUCCESS;
  }

  DataType single_dtype;
  GE_WARN_ASSERT_GRAPH_SUCCESS(Eval(op, single_dtype));
  type_or_types.SetType(single_dtype);
  return GRAPH_SUCCESS;
}

std::string SymDtype::SymBackend::DebugString() const {
  return std::string(ToString(type)) + "[" + std::to_string(index) + "] " + name;
}

graphStatus SymDtype::Eval(const OpDesc &op, DataType &dtype) const {
  if (ir_inputs_.empty()) {
    GELOGI("Trying to eval sym dtype from attr %s of op %s", id_.c_str(), op.GetType().c_str());
    if (AttrUtils::HasAttr(op, id_)) {
      GE_WARN_ASSERT_GRAPH_SUCCESS(GetDtypeFromAttr(op, id_, dtype));
      GE_WARN_ASSERT(tensor_type_.tensor_type_impl_->IsDataTypeInRange(dtype));
      return GRAPH_SUCCESS;
    }
    GE_WARN_ASSERT(tensor_type_.tensor_type_impl_->GetMutableDateTypeSet().size() == 1,
                   "Op %s has no attr %s and sym %s allowed dtypes size is %zu, expected 1", op.GetType().c_str(),
                   id_.c_str(), tensor_type_.tensor_type_impl_->GetMutableDateTypeSet().size());
    dtype = *tensor_type_.tensor_type_impl_->GetMutableDateTypeSet().begin();
    return GRAPH_SUCCESS;
  }

  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  GE_WARN_ASSERT_GRAPH_SUCCESS(GetIrInputRawDescRange(const_cast<OpDesc *>(&op)->shared_from_this(), ir_input_2_range));
  GE_WARN_ASSERT(ir_input_2_range.size() == op.GetIrInputsSize(),
                 "Failed to get input instance info of %s %s, ir_input range size:%zu, ir input size:%zu",
                 op.GetName().c_str(), op.GetType().c_str(), ir_input_2_range.size(), op.GetIrInputsSize());

  std::set<DataType> inferred_dtypes;
  for (auto &backend : ir_inputs_) {
    auto &input_range = ir_input_2_range[backend.index];
    size_t start = input_range.first;
    size_t end = input_range.first + input_range.second;
    GELOGD("Sym %s of %s backend %s mapping to input desc[%zu:%zu)", id_.c_str(), op.GetName().c_str(),
           backend.DebugString().c_str(), start, end);

    for (size_t i = start; i < end; i++) {
      auto desc = op.MutableInputDesc(i);
      GE_ASSERT_NOTNULL(desc);
      GELOGD("Get dtype %s from %s input %s:%zu of op %s",
             TypeUtils::DataTypeToSerialString(desc->GetDataType()).c_str(), ToString(backend.type),
             backend.name.c_str(), i - start, op.GetName().c_str());
      inferred_dtypes.insert(desc->GetDataType());
    }
  }

  GE_WARN_ASSERT(inferred_dtypes.size() == 1, "Infer dtype failed for op %s as %zu types inferred",
                 op.GetName().c_str(), inferred_dtypes.size());
  dtype = *inferred_dtypes.begin();
  if (!tensor_type_.tensor_type_impl_->IsDataTypeInRange(dtype)) {
    REPORT_INNER_ERR_MSG("EZ9999", "Sym %s of op %s %s inferred dtype %s not in range %s", id_.c_str(),
                         op.GetName().c_str(), op.GetType().c_str(), TypeUtils::DataTypeToSerialString(dtype).c_str(),
                         ToString(tensor_type_).c_str());
    GELOGW("Sym %s inferred dtype %s not in range %s", id_.c_str(), TypeUtils::DataTypeToSerialString(dtype).c_str(),
           ToString(tensor_type_).c_str());
    return ge::PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

graphStatus SymDtype::Eval(const OpDesc &op, std::vector<DataType> &dtypes) const {
  if (ir_inputs_.empty()) {
    GELOGI("Eval sym list-dtype from attr %s of op %s", id_.c_str(), op.GetType().c_str());
    GE_WARN_ASSERT_GRAPH_SUCCESS(GetListDtypeFromAttr(op, id_, dtypes));
    for (auto &dtype : dtypes) {
      GE_WARN_ASSERT(tensor_type_.tensor_type_impl_->IsDataTypeInRange(dtype),
                     "Sym %s inferred one of list-dtype %s not in range %s", id_.c_str(),
                     TypeUtils::DataTypeToSerialString(dtype).c_str(), ToString(tensor_type_).c_str());
    }
    return GRAPH_SUCCESS;
  }

  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  GE_WARN_ASSERT_GRAPH_SUCCESS(GetIrInputRawDescRange(const_cast<OpDesc *>(&op)->shared_from_this(), ir_input_2_range));
  GE_WARN_ASSERT(ir_input_2_range.size() == op.GetIrInputsSize(),
                 "Failed to get input instance info of %s %s, ir_input range size:%zu, ir input size:%zu",
                 op.GetName().c_str(), op.GetType().c_str(), ir_input_2_range.size(), op.GetIrInputsSize());

  for (auto &backend : ir_inputs_) {
    GE_WARN_ASSERT(backend.type == kIrInputDynamic, "List-type sym %s can not bind to %s input %s", id_.c_str(),
                   ToString(backend.type), backend.name.c_str());
    auto &input_range = ir_input_2_range[backend.index];
    size_t start = input_range.first;
    size_t end = input_range.first + input_range.second;
    GELOGD("Sym %s of %s backend %s mapping to input desc[%zu:%zu)", id_.c_str(), op.GetName().c_str(),
           backend.DebugString().c_str(), start, end);

    std::vector<DataType> input_dtypes;
    for (size_t i = start; i < end; i++) {
      auto desc = op.MutableInputDesc(i);
      GE_ASSERT_NOTNULL(desc);
      GELOGD("Get dtype %s from dynamic input %s:%zu of op %s",
             TypeUtils::DataTypeToSerialString(desc->GetDataType()).c_str(), backend.name.c_str(), i - start,
             op.GetName().c_str());
      input_dtypes.push_back(desc->GetDataType());
    }

    if (dtypes.empty()) {
      dtypes = input_dtypes;
    } else {
      GE_WARN_ASSERT(input_dtypes.size() == dtypes.size(), "Infer dtype size mismatch %zu vs. %zu", input_dtypes.size(),
                     dtypes.size());
      for (size_t i = 0U; i < input_dtypes.size(); i++) {
        GE_WARN_ASSERT(input_dtypes[i] == dtypes[i], "Sym %s list-dtype mismatch at index %zu: input=%s, expected=%s",
                       id_.c_str(), i, TypeUtils::DataTypeToSerialString(input_dtypes[i]).c_str(),
                       TypeUtils::DataTypeToSerialString(dtypes[i]).c_str());
      }
    }
  }

  for (auto &dtype : dtypes) {
    GE_WARN_ASSERT(tensor_type_.tensor_type_impl_->IsDataTypeInRange(dtype),
                   "Sym %s inferred list-dtype %s not in range %s", id_.c_str(),
                   TypeUtils::DataTypeToSerialString(dtype).c_str(), ToString(tensor_type_).c_str());
  }

  return GRAPH_SUCCESS;
}

void SymDtype::BindAllowedOrderedDtypes(const OrderedTensorTypeList &types) {
  is_legacy_ = false;
  is_list_ = true;
  is_ordered_list_ = true;
  ordered_tensor_type_list_ = types;
}

PromotionSymDtypeExpression::PromotionSymDtypeExpression(const std::vector<SymDtype *> &syms) : syms_(syms) {}

graphStatus PromotionSymDtypeExpression::Eval(const OpDesc &op, TypeOrTypes &type_or_types) const {
  GE_WARN_ASSERT(syms_.size() > 1U, "Trying to eval promotion sym with %zu syms", syms_.size());

  GE_WARN_ASSERT_GRAPH_SUCCESS(syms_[0]->Eval(op, type_or_types));
  GELOGI("Promoting start with %s from sym %s", type_or_types.DebugString().c_str(), syms_[0]->DebugString().c_str());

  TypeOrTypes next;
  for (size_t i = 1U; i < syms_.size(); i++) {
    GE_WARN_ASSERT_GRAPH_SUCCESS(syms_[i]->Eval(op, next));
    GELOGD("Promoting %s with %s from sym %s", type_or_types.DebugString().c_str(), next.DebugString().c_str(),
           syms_[i]->DebugString().c_str());
    GE_WARN_ASSERT_GRAPH_SUCCESS(PromoteDtype(type_or_types, next, type_or_types));
  }

  return GRAPH_SUCCESS;
}

ExpressionType PromotionSymDtypeExpression::Type() const {
  return ExpressionType::kPromote;
}

std::vector<size_t> PromotionSymDtypeExpression::GetIrInputIndexes() const {
  std::vector<size_t> ir_input_indexes;
  for (const auto &sym : syms_) {
    const auto sym_ir_input_indexs = sym->GetIrInputIndexes();
    for (const auto sym_ir_input : sym_ir_input_indexs) {
      ir_input_indexes.push_back(sym_ir_input);
    }
  }
  return ir_input_indexes;
}

namespace {
class DescEnv {
 public:
  DescEnv(const OpDescPtr &op, bool for_input) : op_(op), for_input_(for_input) {}
  ~DescEnv() = default;

  bool IsDescValid(uint32_t index) const {
    return for_input_ ? (op_->MutableInputDesc(index) != nullptr) : (op_->MutableOutputDesc(index) != nullptr);
  }

  size_t NumDescs() const {
    return for_input_ ? op_->GetAllInputsSize() : op_->GetOutputsSize();
  }

  std::string DebugString() const {
    std::string str = "Env for ";
    str += op_->GetName() + " ";
    str += op_->GetType() + " ";
    str += for_input_ ? "input" : "output";
    return str;
  }

 private:
  const OpDescPtr &op_;
  bool for_input_;
};
class IrIOSpec {
 public:
  IrIOSpec(const std::string &name, const IrInputType &type) {
    name_ = name;
    is_input_ = true;
    if (type == kIrInputDynamic) {
      is_dynamic_ = true;
    } else if (type == kIrInputOptional) {
      is_optional_ = true;
    } else if (type == kIrInputRequired) {
      is_required_ = true;
    } else {
      is_valid_ = false;
    }
  }

  IrIOSpec(const std::string &name, const IrOutputType &type) {
    name_ = name;
    if (type == kIrOutputDynamic) {
      is_dynamic_ = true;
    } else if (type == kIrOutputRequired) {
      is_required_ = true;
    } else {
      is_valid_ = false;
    }
  }
  ~IrIOSpec() = default;

  const std::string &GetName() const {
    return name_;
  }
  std::string DebugString() const {
    std::string str = (is_dynamic_ ? "Dynamic " : is_optional_ ? "Optional " : is_required_ ? "Required " : "Invalid ");
    str += is_input_ ? "input " : "output ";
    str += name_;
    return str;
  }
  bool IsValid() const {
    return is_valid_;
  }
  bool IsDynamic() const {
    return is_dynamic_;
  }
  bool IsOptional() const {
    return is_optional_;
  }
  bool IsRequired() const {
    return is_required_;
  }

 private:
  std::string name_;
  bool is_input_ = false;
  bool is_valid_ = true;
  bool is_dynamic_ = false;
  bool is_optional_ = false;
  bool is_required_ = false;
};

// 对于空的Dynamic输入和未传值的Optional输入，计算其起始index以展示更为友好
size_t GetIrDescStartIndex(std::map<size_t, std::pair<size_t, size_t>> &ir_2_range, size_t ir_index) {
  if (ir_index == 0U) {
    return 0U;
  }

  auto iter = ir_2_range.find(ir_index - 1U);
  if (iter == ir_2_range.end()) {
    return 0U;
  }
  return iter->second.first + iter->second.second;
}

graphStatus MappingDynamicIrDesc(const std::vector<IrIOSpec> &ir_specs, const DescEnv &desc_env,
                                 const std::map<std::string, uint32_t> &name2idx,
                                 std::map<size_t, std::pair<size_t, size_t>> &ir_2_range) {
  GELOGD("Start mapping dynamic ir desc for %s", desc_env.DebugString().c_str());
  for (size_t ir_io_idx = 0U; ir_io_idx < ir_specs.size(); ir_io_idx++) {
    const auto &ir_spec = ir_specs[ir_io_idx];
    GE_WARN_ASSERT(ir_spec.IsValid(), "Invalid ir spec %s", ir_spec.DebugString().c_str());
    if (!ir_spec.IsDynamic()) {  // 优先处理Dynamic类型的IR输入
      continue;
    }
    std::set<size_t> indexes;  // Dynamic类型的IR输入对应的多个index
    size_t num_instances = 0U;
    for (; num_instances < name2idx.size(); num_instances++) {
      auto iter = name2idx.find(ir_spec.GetName() + std::to_string(num_instances));
      if (iter == name2idx.end()) {
        break;
      }
      indexes.insert(iter->second);
    }
    // 校验Dynamic类型的IR IO对应的多个index连续
    GE_WARN_ASSERT((indexes.size() <= 1U) || (*indexes.rbegin() - *indexes.begin() == (indexes.size() - 1U)));
    if (indexes.empty()) {
      GELOGD("Dynamic ir spec %s has no instance", ir_spec.DebugString().c_str());
      ir_2_range.emplace(ir_io_idx, std::make_pair(GetIrDescStartIndex(ir_2_range, ir_io_idx), 0U));
    } else {
      ir_2_range.emplace(ir_io_idx, std::make_pair(*indexes.begin(), indexes.size()));
      GELOGD("Mapping %s to desc[%zu, %zu)", ir_spec.DebugString().c_str(), *indexes.begin(),
             *indexes.begin() + indexes.size());
    }
  }
  return GRAPH_SUCCESS;
}

void UpdateRawDescInstanceShifts(std::vector<size_t> &desc_instance_shifts, size_t elim_index) {
  if (elim_index >= desc_instance_shifts.size()) {
    return;
  }
  auto iter = desc_instance_shifts.begin() + elim_index + 1U;
  for (; iter != desc_instance_shifts.end(); iter++) {
    (*iter)++;
  }
}

graphStatus MappingNonDynamicIrDesc(const std::vector<IrIOSpec> &ir_specs, const DescEnv &desc_env,
                                    const std::vector<std::pair<std::string, uint32_t>> &name2index_left,
                                    const bool &require_raw_index,
                                    std::map<size_t, std::pair<size_t, size_t>> &ir_2_range) {
  GELOGD("Start mapping non-dynamic ir desc for %s", desc_env.DebugString().c_str());
  std::vector<size_t> desc_instance_shifts;
  desc_instance_shifts.resize(desc_env.NumDescs(), 0U);

  auto iter = name2index_left.begin();
  for (size_t ir_io_idx = 0U; ir_io_idx < ir_specs.size(); ir_io_idx++) {
    const auto &ir_spec = ir_specs[ir_io_idx];
    if (ir_spec.IsDynamic()) {  // 已经处理过Dynamic类型的IR输入
      continue;
    }

    if (iter == name2index_left.end()) {  // 只允许Optional的IR输入没有对应的desc，对应Optional在IR最后且没有Desc信息
      if (!ir_spec.IsOptional()) {
        GELOGW("No desc left for %s", ir_spec.DebugString().c_str());
        return GRAPH_SUCCESS;
      }
      ir_2_range.emplace(ir_io_idx, std::make_pair(GetIrDescStartIndex(ir_2_range, ir_io_idx), 0U));
      continue;
    }

    auto &name = iter->first;
    auto &index = iter->second;

    if (ir_spec.GetName() != name) {  // 如果当前名字和IR不一致，需要确保不是乱序，即没有与IR名字对应的Desc存在
      for (auto &name2index : name2index_left) {
        GE_WARN_ASSERT(ir_spec.GetName() != name2index.first, "Found desc for %s index %u, while current name is %s",
                       ir_spec.DebugString().c_str(), name2index.second, name.c_str());
      }
    }

    if (!ir_spec.IsOptional()) {  // 非可选，则认为是自行构造的非标IR
      iter++;
      ir_2_range.emplace(ir_io_idx, std::make_pair(index, 1U));
      GELOGD("Mapping %s to desc %zu named %s", ir_spec.DebugString().c_str(), index, name.c_str());
      continue;
    }

    if (name != ir_spec.GetName()) {  // 对应Optional不在尾部且未传入
      GELOGD("Ir spec %s has no instance as desc[%u] named %s", ir_spec.DebugString().c_str(), index, name.c_str());
      ir_2_range.emplace(ir_io_idx, std::make_pair(index, 0U));
      continue;
    }

    iter++;
    if (desc_env.IsDescValid(index)) {  // 对应Optional传入有效值
      GELOGD("Mapping %s desc[%zu]", ir_spec.DebugString().c_str(), index);
      ir_2_range.emplace(ir_io_idx, std::make_pair(index, 1U));
    } else {  // Optional传入无效值，对实例index进行调整（实例index只会保存非nullptr的输入）
      GELOGD("Skip mapping %s to invalid desc[%zu]", ir_spec.DebugString().c_str(), index);
      ir_2_range.emplace(ir_io_idx, std::make_pair(index, 0U));
      UpdateRawDescInstanceShifts(desc_instance_shifts, index);
    }
  }

  if (!require_raw_index) {
    for (auto &item : ir_2_range) {
      auto &start = item.second.first;
      auto &num = item.second.second;
      size_t shift = (start >= desc_instance_shifts.size() ? 0U : desc_instance_shifts[start]);
      start = (start > shift) ? (start - shift) : 0U;
      GELOGD("Re-mapping %s to desc[%zu, %zu) shift(-%zu)", ir_specs[item.first].DebugString().c_str(), start,
             start + num, shift);
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus GetIrDescRange(const std::vector<IrIOSpec> &ir_specs, const std::map<std::string, uint32_t> &name2idx,
                           const DescEnv &desc_env, const bool &require_raw_index,
                           std::map<size_t, std::pair<size_t, size_t>> &ir_2_range) {
  GELOGD("Start get desc range for %s", desc_env.DebugString().c_str());
  for (auto &ir_spec : ir_specs) {
    GELOGD("  Spec %s", ir_spec.DebugString().c_str());
  }

  std::map<uint32_t, std::string> idx2name;
  for (auto &item : name2idx) {
    GELOGD("  Desc name %s index %d", item.first.c_str(), item.second);
    idx2name.emplace(item.second, item.first);
  }
  GE_WARN_ASSERT(idx2name.size() == name2idx.size(), "Found %zu names, while %zu indexes", idx2name.size(),
                 name2idx.size());
  if (!idx2name.empty()) {
    GE_WARN_ASSERT(idx2name.rbegin()->first == idx2name.size() - 1U);  // 拦截index不连续
  }

  // 首先确定Dynamic类型的IR IO对应的index范围，对于IR构图场景，用户会通过create_dynmaic_xx接口创建多个输入Desc，
  // 但是Desc在所有desc中的位置，是受调用时的参数决定的，默认情况下，都向尾部追加，会出现先定义的IR输入或输出对应的desc，在后定义的之后
  GE_WARN_ASSERT_GRAPH_SUCCESS(MappingDynamicIrDesc(ir_specs, desc_env, name2idx, ir_2_range));

  std::vector<bool> index_consumed;  // index对应的desc是否已经决定对应关系
  index_consumed.resize(name2idx.size(), false);
  for (auto &item : ir_2_range) {
    auto &range = item.second;
    for (size_t i = range.first; i < range.first + range.second; i++) {
      index_consumed[i] = true;
    }
  }

  std::vector<std::pair<std::string, uint32_t>> name2index_left;
  for (size_t i = 0U; i < index_consumed.size(); i++) {
    if (!index_consumed[i]) {  // 未被使用的index顺序排列
      name2index_left.emplace_back(idx2name[i], static_cast<uint32_t>(i));
    }
  }

  // 确定非Dynamic类型的IR IO对应的index范围
  GE_WARN_ASSERT_GRAPH_SUCCESS(
      MappingNonDynamicIrDesc(ir_specs, desc_env, name2index_left, require_raw_index, ir_2_range));

  // 不校验所有的index都决定了对应的IR输入（存在算子追加非IR输入的场景，CCB裁决框架适配支持）

  return GRAPH_SUCCESS;
}

graphStatus GetIrInputDescRange(const OpDescPtr &op, const bool &require_raw_index,
                                std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range) {
  GE_ASSERT_NOTNULL(op);
  std::vector<IrIOSpec> ir_specs;
  for (auto &item : op->GetIrInputs()) {
    ir_specs.emplace_back(item.first, item.second);
  }
  const std::map<std::string, uint32_t> &name2idx = op->GetAllInputName();
  DescEnv desc_env(op, true);

  return GetIrDescRange(ir_specs, name2idx, desc_env, require_raw_index, ir_input_2_range);
}
}  // namespace

// 获取输入IR对应的实例Desc的index范围，实例Desc中会去除未传值的Optional输入Desc
graphStatus GetIrInputInstanceDescRange(const OpDescPtr &op,
                                        std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range) {
  return GetIrInputDescRange(op, false, ir_input_2_range);
}

// 获取输入IR对应的全部Desc的index范围，包含未传值的Optional输入Desc
graphStatus GetIrInputRawDescRange(const OpDescPtr &op, std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range) {
  return GetIrInputDescRange(op, true, ir_input_2_range);
}

graphStatus GetIrOutputDescRange(const OpDescPtr &op, std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range) {
  GE_ASSERT_NOTNULL(op);
  std::vector<IrIOSpec> ir_specs;
  for (auto &item : op->GetIrOutputs()) {
    ir_specs.emplace_back(item.first, item.second);
  }
  const std::map<std::string, uint32_t> &name2idx = op->GetAllOutputName();
  DescEnv desc_env(op, false);

  return GetIrDescRange(ir_specs, name2idx, desc_env, true, ir_output_2_range);
}
}  // namespace af
