/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_INC_BASE_ATTR_ATTRS_TO_BUFFER_H_
#define METADEF_INC_BASE_ATTR_ATTRS_TO_BUFFER_H_
#include <cstring>
#include <securec.h>
#include "base/runtime/runtime_attrs_def.h"
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "graph/any_value.h"
#include "graph/def_types.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/types_af.h"
#include "graph/utils/math_util.h"

namespace gert {
namespace bg {

template <typename T, typename std::enable_if<std::is_fundamental<T>::value, int32_t>::type = 0>
bool AppendFundAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val = attr.Get<T>();
  GE_ASSERT_NOTNULL(val);
  std::vector<uint8_t> runtime_attr(sizeof(*val));
  GE_ASSERT_EOK(memcpy_s(runtime_attr.data(), sizeof(*val), val, sizeof(*val)));
  (void)attrs.emplace_back(std::move(runtime_attr));
  return true;
}
inline bool AppendStrAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto str = attr.Get<std::string>();
  GE_ASSERT_NOTNULL(str);
  std::vector<uint8_t> runtime_attr(str->size() + static_cast<size_t>(1));
  GE_ASSERT_EOK(strcpy_s(ge::PtrToPtr<uint8_t, ge::char_t>(runtime_attr.data()), str->size() + static_cast<size_t>(1),
                         str->c_str()));
  (void)attrs.emplace_back(std::move(runtime_attr));
  return true;
}
template <typename T, typename std::enable_if<std::is_fundamental<T>::value, int32_t>::type = 0>
bool AppendVectorAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val = attr.Get<std::vector<T>>();
  GE_ASSERT_NOTNULL(val);
  size_t total_size;
  if (ge::MulOverflow(val->size(), sizeof(T), total_size)) {
    return false;
  }
  if (ge::AddOverflow(total_size, sizeof(ContinuousVector), total_size)) {
    return false;
  }

  std::vector<uint8_t> buf(total_size);
  auto cv = new (buf.data()) ContinuousVector();
  GE_ASSERT_NOTNULL(cv);
  cv->Init(val->size());
  (void)cv->SetSize(val->size());

  if (!val->empty()) {
    const size_t copy_size = val->size() * sizeof(T);
    GE_ASSERT_EOK(memcpy_s(cv->MutableData(), cv->GetCapacity() * sizeof(T), val->data(), copy_size));
  }
  (void)attrs.emplace_back(std::move(buf));
  return true;
}

inline bool AppendVectorBoolAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val_bool_list = attr.Get<std::vector<bool>>();
  GE_ASSERT_NOTNULL(val_bool_list);
  std::vector<uint8_t> val_uint8_list;
  for (size_t i = 0U; i < val_bool_list->size(); ++i) {
    (void)val_uint8_list.emplace_back(static_cast<uint8_t>(val_bool_list->at(i)));
  }
  size_t total_size;
  if (ge::MulOverflow(val_uint8_list.size(), sizeof(uint8_t), total_size)) {
    return false;
  }
  if (ge::AddOverflow(total_size, sizeof(ContinuousVector), total_size)) {
    return false;
  }

  std::vector<uint8_t> buf(total_size);
  auto cv = new (buf.data()) ContinuousVector();
  GE_ASSERT_NOTNULL(cv);
  cv->Init(val_uint8_list.size());
  (void)cv->SetSize(val_uint8_list.size());

  if (!val_uint8_list.empty()) {
    const size_t copy_size = val_uint8_list.size() * sizeof(uint8_t);
    GE_ASSERT_EOK(memcpy_s(cv->MutableData(), cv->GetCapacity() * sizeof(uint8_t), val_uint8_list.data(), copy_size));
  }
  (void)attrs.emplace_back(std::move(buf));
  return true;
}

template <typename T, typename std::enable_if<std::is_fundamental<T>::value, int32_t>::type = 0>
bool AppendVectorVectorAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto vector_vector_list = attr.Get<std::vector<std::vector<T>>>();
  GE_ASSERT_NOTNULL(vector_vector_list);

  size_t total_size = ContinuousVectorVector::GetOverHeadLength(vector_vector_list->size());
  for (const auto &inner_vec : *vector_vector_list) {
    size_t inner_vec_length = 0U;
    if (ge::MulOverflow(inner_vec.size(), sizeof(T), inner_vec_length)) {
      return false;
    }
    if (ge::AddOverflow(inner_vec_length, sizeof(ContinuousVector), inner_vec_length)) {
      return false;
    }
    if (ge::AddOverflow(total_size, inner_vec_length, total_size)) {
      return false;
    }
  }
  std::vector<uint8_t> buf(total_size);
  auto cvv = new (buf.data()) ContinuousVectorVector();
  GE_ASSERT_NOTNULL(cvv);
  cvv->Init(vector_vector_list->size());

  for (const auto &inner_list : *vector_vector_list) {
    auto cv = cvv->Add<T>(inner_list.size());
    GE_ASSERT_NOTNULL(cv);
    if (!inner_list.empty()) {
      const size_t copy_size = inner_list.size() * sizeof(T);
      GE_ASSERT_EOK(memcpy_s(cv->MutableData(), cv->GetCapacity() * sizeof(T), inner_list.data(), copy_size));
    }
  }

  (void)attrs.emplace_back(std::move(buf));
  return true;
}
inline bool AppendDataTypeAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val = attr.Get<ge::DataType>();
  GE_ASSERT_NOTNULL(val);
  std::vector<uint8_t> runtime_attr(sizeof(*val));
  GE_ASSERT_EOK(memcpy_s(runtime_attr.data(), sizeof(*val), val, sizeof(*val)));
  (void)attrs.emplace_back(std::move(runtime_attr));
  return true;
}
inline bool AppendVectorDataTypeAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val = attr.Get<std::vector<ge::DataType>>();
  GE_ASSERT_NOTNULL(val);
  size_t total_size = 0U;
  GE_ASSERT_TRUE(!ge::MulOverflow(val->size(), sizeof(ge::DataType), total_size),
                 "Mul overflow vec size %zu, elem size %zu.", val->size(), sizeof(ge::DataType));
  GE_ASSERT_TRUE(!ge::AddOverflow(total_size, sizeof(ContinuousVector), total_size),
                 "Add overflow total size %zu, size of vec %zu.", total_size, sizeof(ContinuousVector));

  std::vector<uint8_t> buf(total_size);
  auto cv = new (buf.data()) ContinuousVector();
  GE_ASSERT_NOTNULL(cv);
  cv->Init(val->size());
  (void)cv->SetSize(val->size());

  if (!val->empty()) {
    const size_t copy_size = val->size() * sizeof(ge::DataType);
    GE_ASSERT_EOK(memcpy_s(cv->MutableData(), cv->GetCapacity() * sizeof(ge::DataType), val->data(), copy_size));
  }
  (void)attrs.emplace_back(std::move(buf));
  return true;
}
inline bool AppendVectorStrAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val = attr.Get<std::vector<std::string>>();
  GE_ASSERT_NOTNULL(val);

  size_t total_str_size = 0U;
  for (size_t i = 0U; i < (*val).size(); ++i) {
    const size_t ele_str_size = (*val)[i].size() + static_cast<size_t>(1);
    if (ge::AddOverflow(total_str_size, ele_str_size, total_str_size)) {
      GELOGW("Add overflow ele str size %zu, total_str_size %zu.", ele_str_size, total_str_size);
      return false;
    }
  }
  size_t total_size = 0U;
  if (ge::AddOverflow(total_str_size, sizeof(ContinuousVector), total_size)) {
    GELOGW("Add overflow ContinuousVector size %zu, total_str_size %zu.", sizeof(ContinuousVector), total_str_size);
    return false;
  }

  std::vector<uint8_t> buf(total_size);
  auto cv = new (buf.data()) ContinuousVector();
  GE_ASSERT_NOTNULL(cv);
  cv->Init(val->size());
  (void)cv->SetSize(val->size());
  size_t offset = 0U;
  for (size_t i = 0U; i < val->size(); ++i) {
    const size_t ele_str_size = (*val)[i].size() + 1U;
    GE_ASSERT_EOK(strcpy_s(ge::PtrToPtr<uint8_t, char>(ge::PtrAdd(ge::PtrToPtr<void, uint8_t>(cv->MutableData()),
                                                                  std::numeric_limits<size_t>::max(), offset)),
                           total_str_size, (*val)[i].c_str()));
    offset += ele_str_size;
  }
  (void)attrs.emplace_back(std::move(buf));
  return true;
}

inline bool AppendAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  switch (attr.GetValueType()) {
    case ge::AnyValue::VT_FLOAT:
      return AppendFundAttr<float>(attr, attrs);
    case ge::AnyValue::VT_BOOL:
      return AppendFundAttr<bool>(attr, attrs);
    case ge::AnyValue::VT_INT:
      return AppendFundAttr<int64_t>(attr, attrs);
    case ge::AnyValue::VT_DATA_TYPE:
      return AppendDataTypeAttr(attr, attrs);
    case ge::AnyValue::VT_STRING:
      return AppendStrAttr(attr, attrs);
    case ge::AnyValue::VT_LIST_BOOL:
      return AppendVectorBoolAttr(attr, attrs);
    case ge::AnyValue::VT_LIST_FLOAT:
      return AppendVectorAttr<float>(attr, attrs);
    case ge::AnyValue::VT_LIST_INT:
      return AppendVectorAttr<int64_t>(attr, attrs);
    case ge::AnyValue::VT_LIST_DATA_TYPE:
      return AppendVectorDataTypeAttr(attr, attrs);
    case ge::AnyValue::VT_LIST_STRING:
      return AppendVectorStrAttr(attr, attrs);
    case ge::AnyValue::VT_LIST_LIST_FLOAT:
      return AppendVectorVectorAttr<float>(attr, attrs);
    case ge::AnyValue::VT_LIST_LIST_INT:
      return AppendVectorVectorAttr<int64_t>(attr, attrs);
    default:
      GELOGE(af::FAILED,
             "Does not support the attr type now, attr type %d, supported types: VT_FLOAT, VT_BOOL, VT_INT, "
             "VT_DATA_TYPE, VT_STRING, VT_LIST_BOOL, VT_LIST_FLOAT, VT_LIST_INT, VT_LIST_DATA_TYPE, VT_LIST_STRING, "
             "VT_LIST_LIST_FLOAT, VT_LIST_LIST_INT",
             attr.GetValueType());
      return false;
  }
}

inline std::unique_ptr<uint8_t[]> CreateAttrBuffer(const std::vector<std::vector<uint8_t>> &attrs, size_t &total_size) {
  total_size = sizeof(RuntimeAttrsDef);
  size_t offset_size = 0U;
  if (ge::MulOverflow(sizeof(size_t), attrs.size(), offset_size)) {
    GELOGE(af::FAILED, "Failed to create attr buffer, total size overflow, attrs size may invalid %zu", attrs.size());
    return nullptr;
  }
  if (ge::AddOverflow(total_size, offset_size, total_size)) {
    GELOGE(af::FAILED, "Failed to create attr buffer, total size overflow, attrs offset may invalid %zu", offset_size);
    return nullptr;
  }
  for (const auto &attr : attrs) {
    if (ge::AddOverflow(total_size, attr.size(), total_size)) {
      GELOGE(af::FAILED,
             "Failed to create attr buffer, total size overflow, attr size may invalid %zu, current total size %zu",
             attr.size(), total_size);
      return nullptr;
    }
  }
  auto attr_holder = ge::ComGraphMakeUnique<uint8_t[]>(total_size);
  GE_ASSERT_NOTNULL(attr_holder);
  auto attr_def = ge::PtrToPtr<uint8_t, RuntimeAttrsDef>(attr_holder.get());
  attr_def->attr_num = attrs.size();
  GE_ASSERT_EQ(memset_s(attr_def->reserved_, sizeof(attr_def->reserved_), 0, sizeof(attr_def->reserved_)), EOK);
  size_t current_offset = sizeof(RuntimeAttrsDef) + sizeof(size_t) * attr_def->attr_num;
  auto attr_pos = attr_holder.get();
  for (size_t i = 0; i < attrs.size(); ++i) {
    attr_def->offset[i] = current_offset;
    const auto ret = ge::GeMemcpy(ge::PtrAdd(attr_pos, std::numeric_limits<size_t>::max(), current_offset),
                                  total_size - current_offset, attrs[i].data(), attrs[i].size());
    GE_ASSERT_TRUE((ret == af::SUCCESS), "memcpy_s failed, copy size is %zu, dst size is %zu", attrs[i].size(),
                   total_size - current_offset);
    current_offset += attrs[i].size();
  }
  return attr_holder;
}

inline std::unique_ptr<uint8_t[]> CreateAttrBufferWithAttrs(const std::vector<ge::AnyValue> &attrs, size_t &size) {
  std::vector<std::vector<uint8_t>> runtime_attrs;
  for (auto &attr : attrs) {
    (void)AppendAttr(attr, runtime_attrs);
  }
  return CreateAttrBuffer(runtime_attrs, size);
}
}  // namespace bg
}  // namespace gert
#endif
