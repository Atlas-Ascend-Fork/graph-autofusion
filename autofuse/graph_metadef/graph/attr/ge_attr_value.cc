/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ge_attr_value.h"

#include <google/protobuf/text_format.h>
#include "graph/graph.h"
#include "graph/utils/attr_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/model_serialize.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "graph/buffer/buffer_impl.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/serialization/attr_serializer_registry.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/math_util.h"

static std::string GetOverflowDescribeOfListint(const std::string &name, const size_t &index, const int64_t &value) {
  std::string reason = "When obtaining the attribute of " + name + ", the list_value[" + std::to_string(index) +
                       "] is " + std::to_string(value) +
                       ", which exceeds the maximum value of integer type and causes value overflow.";
  return reason;
}

static std::string GetOverflowDescribeOfInt(const std::string &name, const int64_t &value) {
  std::string reason = "When obtaining the attribute of " + name + ", the value is " + std::to_string(value) +
                       ", which exceeds the maximum value of integer type and causes value overflow.";
  return reason;
}

namespace af {
namespace {
const std::map<AnyValue::ValueType, std::string> kAttrTypesMap = {
    {AnyValue::VT_NONE, "VT_NONE"},
    {AnyValue::VT_STRING, "VT_STRING"},
    {AnyValue::VT_FLOAT, "VT_FLOAT"},
    {AnyValue::VT_BOOL, "VT_BOOL"},
    {AnyValue::VT_INT, "VT_INT"},
    {AnyValue::VT_TENSOR_DESC, "VT_TENSOR_DESC"},
    {AnyValue::VT_TENSOR, "VT_TENSOR"},
    {AnyValue::VT_BYTES, "VT_BYTES"},
    {AnyValue::VT_GRAPH, "VT_GRAPH"},
    {AnyValue::VT_NAMED_ATTRS, "VT_NAMED_ATTRS"},
    {AnyValue::VT_LIST_LIST_INT, "VT_LIST_LIST_INT"},
    {AnyValue::VT_DATA_TYPE, "VT_DATA_TYPE"},
    {AnyValue::VT_LIST_STRING, "VT_LIST_STRING"},
    {AnyValue::VT_LIST_FLOAT, "VT_LIST_FLOAT"},
    {AnyValue::VT_LIST_BOOL, "VT_LIST_BOOL"},
    {AnyValue::VT_LIST_INT, "VT_LIST_INT"},
    {AnyValue::VT_LIST_TENSOR_DESC, "VT_LIST_TENSOR_DESC"},
    {AnyValue::VT_LIST_TENSOR, "VT_LIST_TENSOR"},
    {AnyValue::VT_LIST_BYTES, "VT_LIST_BYTES"},
    {AnyValue::VT_GRAPH, "VT_GRAPH"},
    {AnyValue::VT_LIST_NAMED_ATTRS, "VT_LIST_NAMED_ATTRS"},
    {AnyValue::VT_LIST_DATA_TYPE, "VT_LIST_DATA_TYPE"},
};

const std::map<std::string, AnyValue::ValueType> kAttrStrTypesMap = {
    {"VT_NONE", AnyValue::VT_NONE},
    {"VT_STRING", AnyValue::VT_STRING},
    {"VT_FLOAT", AnyValue::VT_FLOAT},
    {"VT_BOOL", AnyValue::VT_BOOL},
    {"VT_INT", AnyValue::VT_INT},
    {"VT_TENSOR_DESC", AnyValue::VT_TENSOR_DESC},
    {"VT_TENSOR", AnyValue::VT_TENSOR},
    {"VT_BYTES", AnyValue::VT_BYTES},
    {"VT_GRAPH", AnyValue::VT_GRAPH},
    {"VT_NAMED_ATTRS", AnyValue::VT_NAMED_ATTRS},
    {"VT_LIST_LIST_INT", AnyValue::VT_LIST_LIST_INT},
    {"VT_DATA_TYPE", AnyValue::VT_DATA_TYPE},
    {"VT_LIST_STRING", AnyValue::VT_LIST_STRING},
    {"VT_LIST_FLOAT", AnyValue::VT_LIST_FLOAT},
    {"VT_LIST_BOOL", AnyValue::VT_LIST_BOOL},
    {"VT_LIST_INT", AnyValue::VT_LIST_INT},
    {"VT_LIST_TENSOR_DESC", AnyValue::VT_LIST_TENSOR_DESC},
    {"VT_LIST_TENSOR", AnyValue::VT_LIST_TENSOR},
    {"VT_LIST_BYTES", AnyValue::VT_LIST_BYTES},
    {"VT_GRAPH", AnyValue::VT_GRAPH},
    {"VT_LIST_NAMED_ATTRS", AnyValue::VT_LIST_NAMED_ATTRS},
    {"VT_LIST_DATA_TYPE", AnyValue::VT_LIST_DATA_TYPE},
};
}  // namespace
void NamedAttrs::SetName(const std::string &name) {
  name_ = name;
}

std::string NamedAttrs::GetName() const {
  return name_;
}

AnyValue NamedAttrs::GetItem(const std::string &key) const {
  AnyValue value;
  (void)GetAttr(key, value);
  return value;
}

ProtoAttrMap &NamedAttrs::MutableAttrMap() {
  return attrs_;
}

ConstProtoAttrMap &NamedAttrs::GetAttrMap() const {
  return attrs_;
}

bool AttrUtils::HasAttr(ConstAttrHolderAdapter &&obj, const std::string &name) {
  if (!obj) {
    return false;
  }
  return obj->HasAttr(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name, int32_t &value) {
  int64_t int64_val = 0;
  if (!AttrUtils::GetInt(std::move(obj), name, int64_val)) {
    return false;
  }
  if (!ge::IntegerChecker<int32_t>::Compat(int64_val)) {
    const std::string reason = GetOverflowDescribeOfInt(name, int64_val);
    REPORT_INNER_ERR_MSG("E18888", "%s", reason.c_str());
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] %s", reason.c_str());
    return false;
  }
  value = static_cast<int32_t>(int64_val);
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name, uint32_t &value) {
  int64_t int64_val = 0;
  if (!AttrUtils::GetInt(std::move(obj), name, int64_val)) {
    return false;
  }
  if (!ge::IntegerChecker<uint32_t>::Compat(int64_val)) {
    const std::string reason = GetOverflowDescribeOfInt(name, int64_val);
    REPORT_INNER_ERR_MSG("E18888", "%s", reason.c_str());
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] %s", reason.c_str());
    return false;
  }
  value = static_cast<uint32_t>(int64_val);
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr AttrUtils::CloneOpDesc(const ConstOpDescPtr &org_op_desc) {
  return GraphUtils::CloneOpDesc(org_op_desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr AttrUtils::CopyOpDesc(const ConstOpDescPtr &org_op_desc) {
  return GraphUtils::CopyOpDesc(org_op_desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListInt(AttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          const std::vector<int64_t> &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          std::vector<int64_t> &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetInt(AttrHolderAdapter &&obj, const std::string &name,
                                                                      const int64_t &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name, int64_t &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name, uint64_t &value) {
  if (!obj) {
    return false;
  }
  int64_t int64_val = 0;
  const bool ret = GetAttrValue(obj->GetAttrMap(), name, int64_val);
  if (ret) {
    value = static_cast<uint64_t>(int64_val);
  }
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetFloat(AttrHolderAdapter &&obj,
                                                                        const std::string &name,
                                                                        const float32_t &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetFloat(ConstAttrHolderAdapter &&obj,
                                                                        const std::string &name, float32_t &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListFloat(AttrHolderAdapter &&obj,
                                                                            const std::string &name,
                                                                            const std::vector<float32_t> &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListFloat(ConstAttrHolderAdapter &&obj,
                                                                            const std::string &name,
                                                                            std::vector<float32_t> &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetBool(AttrHolderAdapter &&obj, const std::string &name,
                                                                       const bool &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetBool(ConstAttrHolderAdapter &&obj,
                                                                       const std::string &name, bool &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListBool(AttrHolderAdapter &&obj,
                                                                           const std::string &name,
                                                                           const std::vector<bool> &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListBool(ConstAttrHolderAdapter &&obj,
                                                                           const std::string &name,
                                                                           std::vector<bool> &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetStr(AttrHolderAdapter &&obj, const std::string &name,
                                                                      const std::string &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetStr(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name, std::string &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetStr(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name1,
                                                                      const std::string &name2, std::string &value) {
  if (!obj) {
    return false;
  }
  if (!GetAttrValue(obj->GetAttrMap(), name1, value)) {
    return GetAttrValue(obj->GetAttrMap(), name2, value);
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::string *AttrUtils::GetStr(ConstAttrHolderAdapter &&obj,
                                                                                    const std::string &name) {
  if (!obj) {
    return nullptr;
  }
  return GetAttrValue<std::string>(obj->GetAttrMap(), name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListStr(AttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          const std::vector<std::string> &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListStr(ConstAttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          std::vector<std::string> &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetTensorDesc(AttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             const GeTensorDesc &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetTensorDesc(ConstAttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             GeTensorDesc &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListTensorDesc(
    AttrHolderAdapter &&obj, const std::string &name, const std::vector<GeTensorDesc> &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListTensorDesc(ConstAttrHolderAdapter &&obj,
                                                                                 const std::string &name,
                                                                                 std::vector<GeTensorDesc> &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetNamedAttrs(AttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             const NamedAttrs &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetNamedAttrs(ConstAttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             NamedAttrs &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListNamedAttrs(AttrHolderAdapter &&obj,
                                                                                 const std::string &name,
                                                                                 const std::vector<NamedAttrs> &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListNamedAttrs(ConstAttrHolderAdapter &&obj,
                                                                                 const std::string &name,
                                                                                 std::vector<NamedAttrs> &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetDataType(AttrHolderAdapter &&obj,
                                                                           const std::string &name,
                                                                           const DataType &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetDataType(ConstAttrHolderAdapter &&obj,
                                                                           const std::string &name, DataType &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListDataType(AttrHolderAdapter &&obj,
                                                                               const std::string &name,
                                                                               const std::vector<DataType> &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListDataType(ConstAttrHolderAdapter &&obj,
                                                                               const std::string &name,
                                                                               std::vector<DataType> &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListListInt(
    AttrHolderAdapter &&obj, const std::string &name, const std::vector<std::vector<int64_t>> &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListListInt(
    ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<std::vector<int64_t>> &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListListFloat(
    AttrHolderAdapter &&obj, const std::string &name, const std::vector<std::vector<float32_t>> &value) {
  if (!obj) {
    return false;
  }
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListListFloat(
    ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<std::vector<float32_t>> &value) {
  if (!obj) {
    return false;
  }
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListInt(AttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          const std::vector<uint32_t> &value) {
  if (!obj) {
    return false;
  }
  return SetListInt(std::move(obj), name, std::vector<int64_t>(value.begin(), value.end()));
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListInt(AttrUtils::AttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          const std::vector<int32_t> &value) {
  if (!obj) {
    return false;
  }
  return SetListInt(std::move(obj), name, std::vector<int64_t>(value.begin(), value.end()));
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListInt(AttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          std::initializer_list<int64_t> &&value) {
  if (!obj) {
    return false;
  }
  return SetListInt(std::move(obj), name, std::vector<int64_t>(value));
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          std::vector<int32_t> &value) {
  value.clear();
  std::vector<int64_t> int64_list;
  if (!GetListInt(std::move(obj), name, int64_list)) {
    return false;
  }

  for (size_t i = 0UL; i < int64_list.size(); ++i) {
    if (!ge::IntegerChecker<int32_t>::Compat(int64_list[i])) {
      const std::string reason = GetOverflowDescribeOfListint(name, i, int64_list[i]);
      REPORT_INNER_ERR_MSG("E18888", "%s", reason.c_str());
      GELOGE(ge::GRAPH_FAILED, "[Check][Param] %s", reason.c_str());
      return false;
    }
  }
  (void)value.insert(value.cbegin(), int64_list.cbegin(), int64_list.cend());
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          std::vector<uint32_t> &value) {
  value.clear();
  std::vector<int64_t> int64_list;
  if (!GetListInt(std::move(obj), name, int64_list)) {
    return false;
  }

  for (size_t i = 0UL; i < int64_list.size(); ++i) {
    if (!ge::IntegerChecker<uint32_t>::Compat(int64_list[i])) {
      const std::string reason = GetOverflowDescribeOfListint(name, i, int64_list[i]);
      REPORT_INNER_ERR_MSG("E18888", "%s", reason.c_str());
      GELOGE(ge::GRAPH_FAILED, "[Check][Param] %s", reason.c_str());
      return false;
    }
  }
  (void)value.insert(value.cbegin(), int64_list.cbegin(), int64_list.cend());
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetTensor(AttrUtils::AttrHolderAdapter &&obj,
                                                                         const std::string &name,
                                                                         const GeTensor &value) {
  if (!obj) {
    return false;
  }
  // 当前GeTensor的拷贝赋值、拷贝构造函数均不是深拷贝，因此无法使用默认的方法SetAttr
  if (!obj->MutableAttrMap().SetByName(name, GeTensor())) {
    return false;
  }
  const auto tensor = obj->MutableAttrMap().MutableGetByName<GeTensor>(name);
  if (tensor == nullptr) {
    return false;
  }
  TensorUtils::CopyTensor(value, *tensor);
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetShareTensor(AttrUtils::AttrHolderAdapter &&obj,
                                                                              const std::string &name,
                                                                              const GeTensor &value) {
  if (!obj) {
    return false;
  }
  // 当前GeTensor的拷贝赋值、拷贝构造函数均不是深拷贝，因此无法使用默认的方法SetAttr
  if (!obj->MutableAttrMap().SetByName(name, GeTensor())) {
    return false;
  }
  const auto tensor = obj->MutableAttrMap().MutableGetByName<GeTensor>(name);
  if (tensor == nullptr) {
    return false;
  }
  TensorUtils::ShareTensor(value, *tensor);
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetTensor(AttrHolderAdapter &&obj,
                                                                         const std::string &name,
                                                                         const GeTensorPtr &value) {
  if (!obj) {
    return false;
  }
  return SetTensor(std::move(obj), name, *value);
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetTensor(AttrHolderAdapter &&obj,
                                                                         const std::string &name,
                                                                         const ConstGeTensorPtr &value) {
  if (!obj) {
    return false;
  }
  return SetTensor(std::move(obj), name, *value);
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListTensor(AttrUtils::AttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             const std::vector<GeTensor> &value) {
  if (!obj) {
    return false;
  }
  std::vector<GeTensor> tensors(value.size());
  if (!obj->MutableAttrMap().SetByName(name, tensors)) {
    return false;
  }
  const auto attr_tensors = obj->MutableAttrMap().MutableGetByName<std::vector<GeTensor>>(name);
  if (attr_tensors == nullptr) {
    return false;
  }
  for (size_t i = 0UL; i < value.size(); ++i) {
    TensorUtils::CopyTensor(value[i], (*attr_tensors)[i]);
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListTensor(AttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             const std::vector<GeTensorPtr> &value) {
  if (!obj) {
    return false;
  }
  std::vector<ConstGeTensorPtr> tensors(value.size());
  (void)std::copy(value.begin(), value.end(), tensors.begin());
  return SetListTensor(std::move(obj), name, tensors);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListTensor(
    AttrHolderAdapter &&obj, const std::string &name, const std::vector<ConstGeTensorPtr> &value) {
  if (!obj) {
    return false;
  }
  std::vector<GeTensor> tensors(value.size());
  if (!obj->MutableAttrMap().SetByName(name, tensors)) {
    return false;
  }
  const auto attr_tensors = obj->MutableAttrMap().MutableGetByName<std::vector<GeTensor>>(name);
  if (attr_tensors == nullptr) {
    return false;
  }
  for (size_t i = 0UL; i < value.size(); ++i) {
    TensorUtils::CopyTensor(*(value[i]), (*attr_tensors)[i]);
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListTensor(
    AttrHolderAdapter &&obj, const std::string &name, std::initializer_list<ConstGeTensorPtr> &&value) {
  if (!obj) {
    return false;
  }
  return SetListTensor(std::move(obj), name, std::vector<ConstGeTensorPtr>(value));
}

// 所有权UT测试，不能把属性上的GeTensor给错误释放了
// 而且这里的行为与老版本是不一样的，老版本中，即使属性的owner生命周期结束析构了，通过本接口获取的value仍然是可用的
// 但是新接口中，owner没有转移，owner析构后，value指向的内存就被释放了，这里需要排查
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::MutableTensor(AttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             GeTensorPtr &value) {
  if (!obj) {
    return false;
  }
  const auto tensor = obj->MutableAttrMap().MutableGetByName<GeTensor>(name);
  if (tensor == nullptr) {
    return false;
  }
  value = std::shared_ptr<GeTensor>(tensor, [](const GeTensor *const ptr) { (void)ptr; });
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetTensor(ConstAttrHolderAdapter &&obj,
                                                                         const std::string &name,
                                                                         ConstGeTensorPtr &value) {
  if (!obj) {
    return false;
  }
  const auto tensor = obj->GetAttrMap().GetByName<GeTensor>(name);
  if (tensor == nullptr) {
    return false;
  }
  value = std::shared_ptr<const GeTensor>(tensor, [](const GeTensor *const ptr) { (void)ptr; });
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListTensor(ConstAttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             std::vector<ConstGeTensorPtr> &value) {
  if (!obj) {
    return false;
  }
  const auto tensors = obj->GetAttrMap().GetByName<std::vector<GeTensor>>(name);
  if (tensors == nullptr) {
    return false;
  }
  value.resize(tensors->size());
  for (size_t i = 0UL; i < tensors->size(); ++i) {
    value[i] = std::shared_ptr<const GeTensor>(&(*tensors)[i], [](const GeTensor *const ptr) { (void)ptr; });
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::MutableListTensor(AttrHolderAdapter &&obj,
                                                                                 const std::string &name,
                                                                                 std::vector<GeTensorPtr> &value) {
  if (!obj) {
    return false;
  }
  const auto tensors = obj->MutableAttrMap().MutableGetByName<std::vector<GeTensor>>(name);
  if (tensors == nullptr) {
    return false;
  }
  value.resize(tensors->size());
  for (size_t i = 0UL; i < tensors->size(); ++i) {
    value[i] = std::shared_ptr<GeTensor>(&(*tensors)[i], [](const GeTensor *const ptr) { (void)ptr; });
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetGraph(AttrUtils::AttrHolderAdapter &&obj,
                                                                        const std::string &name,
                                                                        const ComputeGraphPtr &value) {
  if (!obj) {
    return false;
  }
  proto::GraphDef *const graph_def = SetAndGetAttrValue(obj->MutableAttrMap(), name, proto::GraphDef());
  if (graph_def == nullptr) {
    return false;
  }
  const ModelSerializeImp imp;
  if (!imp.SerializeGraph(value, graph_def)) {
    REPORT_INNER_ERR_MSG("E18888", "SerializeGraph failed when add ComputeGraph to attr %s", name.c_str());
    GELOGE(ge::GRAPH_FAILED, "[Serialize][Graph] Failed when add ComputeGraph to attr %s", name.c_str());
    (void)obj->MutableAttrMap().Delete(name);
    return false;
  }
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListGraph(AttrUtils::AttrHolderAdapter &&obj,
                                                                            const std::string &name,
                                                                            const std::vector<ComputeGraphPtr> &value) {
  if (!obj) {
    return false;
  }
  std::vector<proto::GraphDef> graphs(value.size());
  if (!obj->MutableAttrMap().SetByName(name, graphs)) {
    return false;
  }
  const auto attr_graphs = obj->MutableAttrMap().MutableGetByName<std::vector<proto::GraphDef>>(name);
  if (attr_graphs == nullptr) {
    return false;
  }
  for (size_t i = 0UL; i < value.size(); ++i) {
    const ModelSerializeImp imp;
    if (!imp.SerializeGraph(value[i], &attr_graphs->at(i))) {
      REPORT_INNER_ERR_MSG("E18888", "SerializeGraph failed when add ComputeGraph to attr %s", name.c_str());
      GELOGE(ge::GRAPH_FAILED, "[Serialize][Graph] Failed when add ComputeGraph to attr %s", name.c_str());
      (void)obj->MutableAttrMap().Delete(name);
      return false;
    }
  }
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetGraph(AttrUtils::ConstAttrHolderAdapter &&obj,
                                                                        const std::string &name,
                                                                        ComputeGraphPtr &value) {
  if (!obj) {
    return false;
  }
  const auto attr_graph_def = obj->GetAttrMap().GetByName<proto::GraphDef>(name);
  if (attr_graph_def == nullptr) {
    return false;
  }
  // 这里延续了老代码实现，先拷贝构造一个ComputeGraph，然后做反序列化，感觉直接把attr_graph_def传进去应该就可以了?
  // 下一步对这里做整改，直接传入attr_graph_def，避免这一次拷贝
  const auto graph_def = ComGraphMakeShared<proto::GraphDef>(*attr_graph_def);
  if (graph_def == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create proto::GraphDef failed.");
    GELOGE(ge::GRAPH_FAILED, "[Create][GraphDef] proto::GraphDef make shared failed");
    return false;
  }

  ModelSerializeImp imp;
  imp.SetProtobufOwner(graph_def);
  if (!imp.UnserializeGraph(value, *graph_def)) {
    REPORT_INNER_ERR_MSG("E18888", "UnserializeGraph failed when get attr ComputeGraph by name %s", name.c_str());
    GELOGE(ge::GRAPH_FAILED, "[Unserialize][Graph] Failed when get attr ComputeGraph by name %s", name.c_str());
    return false;
  }

  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListGraph(AttrUtils::ConstAttrHolderAdapter &&obj,
                                                                            const std::string &name,
                                                                            std::vector<ComputeGraphPtr> &value) {
  if (!obj) {
    return false;
  }
  const auto graph_defs = obj->GetAttrMap().GetByName<std::vector<proto::GraphDef>>(name);
  if (graph_defs == nullptr) {
    return false;
  }

  value.resize(graph_defs->size());
  for (size_t i = 0UL; i < graph_defs->size(); ++i) {
    std::shared_ptr<proto::GraphDef> graph_def;
    graph_def = ComGraphMakeShared<proto::GraphDef>(graph_defs->at(i));
    if (graph_def == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "create proto::GraphDef failed.");
      GELOGE(ge::GRAPH_FAILED, "[Create][GraphDef] proto::GraphDef make shared failed");
      graph_def = nullptr;
      return false;
    } else {
      ComputeGraphPtr graph = nullptr;
      ModelSerializeImp imp;
      imp.SetProtobufOwner(static_cast<const ProtoMsgOwner &>(graph_def));
      if (!imp.UnserializeGraph(graph, *graph_def)) {
        REPORT_INNER_ERR_MSG("E18888", "UnserializeGraph failed, index[%zu], graph name[%s]", i,
                             graph_defs->at(i).name().c_str());
        GELOGE(ge::GRAPH_FAILED, "[Unserialize][Graph] Failed, index[%zu], graph name[%s]", i,
               graph_defs->at(i).name().c_str());
        return false;
      }
      value[i] = graph;
    }
  }
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetBytes(AttrUtils::AttrHolderAdapter &&obj,
                                                                        const std::string &name, const Buffer &value) {
  if (!obj) {
    return false;
  }
  const auto buffer = SetAndGetAttrValue(obj->MutableAttrMap(), name, Buffer());
  if (buffer == nullptr) {
    return false;
  }
  BufferUtils::CopyFrom(value, *buffer);
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetBytes(ConstAttrHolderAdapter &&obj,
                                                                        const std::string &name, Buffer &value) {
  if (!obj) {
    return false;
  }
  const auto buffer = obj->GetAttrMap().GetByName<Buffer>(name);
  if (buffer == nullptr) {
    return false;
  }
  BufferUtils::CopyFrom(*buffer, value);
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListBytes(AttrUtils::AttrHolderAdapter &&obj,
                                                                            const std::string &name,
                                                                            const std::vector<Buffer> &value) {
  if (!obj) {
    return false;
  }
  std::vector<Buffer> buffers(value.size());
  const auto attr_buffers = SetAndGetAttrValue(obj->MutableAttrMap(), name, buffers);
  if (attr_buffers == nullptr) {
    return false;
  }

  for (size_t i = 0UL; i < value.size(); ++i) {
    BufferUtils::CopyFrom(value[i], (*attr_buffers)[i]);
  }

  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListBytes(AttrUtils::ConstAttrHolderAdapter &&obj,
                                                                            const std::string &name,
                                                                            std::vector<Buffer> &value) {
  if (!obj) {
    return false;
  }
  const auto buffers = obj->GetAttrMap().GetByName<std::vector<Buffer>>(name);
  if (buffers == nullptr) {
    return false;
  }
  value.resize(buffers->size());
  for (size_t i = 0UL; i < buffers->size(); ++i) {
    BufferUtils::CopyFrom(buffers->at(i), value[i]);
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetZeroCopyBytes(AttrHolderAdapter &&obj,
                                                                                const std::string &name,
                                                                                Buffer &&buffer) {
  if (!obj) {
    return false;
  }
  // Value will be shared
  return SetAttrValue(obj->MutableAttrMap(), name, std::move(buffer));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetZeroCopyBytes(ConstAttrHolderAdapter &&obj,
                                                                                const std::string &name,
                                                                                Buffer &buffer) {
  if (!obj) {
    return false;
  }
  // Value will be shared
  return GetAttrValue<Buffer>(obj->GetAttrMap(), name, buffer);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetZeroCopyListBytes(AttrHolderAdapter &&obj,
                                                                                    const std::string &name,
                                                                                    std::vector<Buffer> &list_buffer) {
  if (!obj) {
    return false;
  }
  // Value will be shared
  return SetAttrValue(obj->MutableAttrMap(), name, list_buffer);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetZeroCopyListBytes(ConstAttrHolderAdapter &&obj,
                                                                                    const std::string &name,
                                                                                    std::vector<Buffer> &list_buffer) {
  if (!obj) {
    return false;
  }
  // Value will be shared
  return GetAttrValue<std::vector<Buffer>>(obj->GetAttrMap(), name, list_buffer);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::map<std::string, AnyValue> AttrUtils::GetAllAttrs(
    ConstAttrHolderAdapter &&obj) {
  return GetAllAttrsWithFilter(std::move(obj), nullptr);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::map<std::string, AnyValue> AttrUtils::GetAllAttrsWithFilter(
    ConstAttrHolderAdapter &&obj, const AttrNameFilter &attr_filter) {
  const auto holder = obj.get();
  if (holder == nullptr) {
    const std::map<std::string, AnyValue> empty;
    return empty;
  }
  return holder->GetAllAttrsWithFilter(attr_filter);
}

std::string AttrUtils::GetAttrsStrAfterRid(ConstAttrHolderAdapter &&obj,
                                           const std::set<std::string> &un_compute_attrs) {
  const std::map<std::string, AnyValue> attr_map = GetAllAttrs(std::move(obj));
  if (attr_map.empty()) {
    return "";
  }
  std::map<std::string, std::string> ordered_attrs;
  for (auto &attr : attr_map) {
    proto::AttrDef attr_def;
    auto *const value_serializer = AttrSerializerRegistry::GetInstance().GetSerializer(attr.second.GetValueTypeId());
    if ((value_serializer == nullptr) || (value_serializer->Serialize(attr.second, attr_def) != GRAPH_SUCCESS)) {
      ordered_attrs[attr.first] = "";
      continue;
    }

    ordered_attrs[attr.first] = attr_def.SerializeAsString();
  }

  std::stringstream str_stream;
  for (auto &attr : ordered_attrs) {
    if (un_compute_attrs.find(attr.first) != un_compute_attrs.end()) {
      continue;
    }
    str_stream << attr.first << ":" << attr.second << ";";
  }
  return str_stream.str();
}
std::string AttrUtils::GetAllAttrsStr(ConstAttrHolderAdapter &&obj) {
  const auto attr_map = GetAllAttrs(std::move(obj));
  if (attr_map.empty()) {
    return "";
  }
  return GetAllAttrsStr(attr_map);
}

std::string AttrUtils::GetAllAttrsStr(const std::map<std::string, AnyValue> &attr_map) {
  std::map<std::string, std::string> ordered_attrs;
  for (auto &attr : attr_map) {
    proto::AttrDef attr_def;
    auto *const value_serializer = AttrSerializerRegistry::GetInstance().GetSerializer(attr.second.GetValueTypeId());
    if ((value_serializer == nullptr) || (value_serializer->Serialize(attr.second, attr_def) != GRAPH_SUCCESS)) {
      ordered_attrs[attr.first] = "";
      continue;
    }

    if (attr_def.has_t()) {
      // print tensor desc message as an ordered string.
      std::string ordered_tensor_desc;
      (void)google::protobuf::TextFormat::PrintToString(attr_def.t().desc(), &ordered_tensor_desc);
      ordered_attrs[attr.first] = ordered_tensor_desc + attr_def.t().data();
    } else if (attr_def.has_td()) {
      // print tensor desc message as an ordered string.
      std::string ordered_attr;
      (void)google::protobuf::TextFormat::PrintToString(attr_def.td(), &ordered_attr);
      ordered_attrs[attr.first] = ordered_attr;
    } else {
      ordered_attrs[attr.first] = attr_def.SerializeAsString();
    }
  }

  std::stringstream ss;
  for (auto &attr : ordered_attrs) {
    ss << attr.first << ":" << attr.second << ";";
  }
  return ss.str();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::ClearAllAttrs(AttrHolderAdapter &&obj) {
  if (!obj) {
    return false;
  }
  obj->MutableAttrMap().Clear();
  return true;
}

std::string AttrUtils::ValueTypeToSerialString(const AnyValue::ValueType value_type) {
  const auto it = kAttrTypesMap.find(value_type);
  if (it != kAttrTypesMap.end()) {
    return it->second;
  } else {
    REPORT_INNER_ERR_MSG("E18888", "value_type %d is not supported", value_type);
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] value_type %d is not supported", value_type);
    return "";
  }
}

AnyValue::ValueType AttrUtils::SerialStringToValueType(const string &value_type_string) {
  const auto it = kAttrStrTypesMap.find(value_type_string);
  if (it != kAttrStrTypesMap.end()) {
    return it->second;
  } else {
    REPORT_INNER_ERR_MSG("E18888", "value_type_string %s is not supported", value_type_string.c_str());
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] value_type_string %s is not supported", value_type_string.c_str());
    return AnyValue::VT_NONE;
  }
}
}  // namespace af
