/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/tensor_af.h"

#include <numeric>
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/ge_tensor.h"
#include "graph/debug/ge_attr_define.h"
#include "securec.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "common/checker.h"

namespace {
const int64_t UNKNOWN_DIM_SIZE = -1;
}  // namespace

namespace af {
// If not overflow return true
static bool Int64MulNotOverflow(const int64_t a, const int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT64_MAX / b)) {
        return false;
      }
    } else {
      if (b < (INT64_MIN / a)) {
        return false;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT64_MIN / b)) {
        return false;
      }
    } else {
      if ((a != 0) && (b < (INT64_MAX / a))) {
        return false;
      }
    }
  }
  return true;
}

class TensorDescValue {
 public:
  TensorDescValue() = default;
  ~TensorDescValue() = default;
  TensorDescValue(const TensorDescValue &other) {
    if ((other.const_data_len_ == 0U) || (other.const_data_buffer_ == nullptr)) {
      return;
    }
    if (!TensorDescValue::CloneValue(this->const_data_buffer_, other.const_data_buffer_, other.const_data_len_)) {
      return;
    }
    this->const_data_len_ = other.const_data_len_;
    return;
  }
  TensorDescValue &operator=(const TensorDescValue &other) {
    if ((&other == this) || (other.const_data_len_ == 0U) || (other.const_data_buffer_ == nullptr)) {
      return *this;
    }
    if (!TensorDescValue::CloneValue(this->const_data_buffer_, other.const_data_buffer_, other.const_data_len_)) {
      return *this;
    }
    this->const_data_len_ = other.const_data_len_;
    return *this;
  }

 private:
  std::unique_ptr<uint8_t[]> const_data_buffer_ = nullptr;
  size_t const_data_len_ = 0U;

  static bool CloneValue(std::unique_ptr<uint8_t[]> &dst, const std::unique_ptr<uint8_t[]> &src,
                         const std::size_t len) {
    dst = ComGraphMakeUnique<uint8_t[]>(len);
    if (dst == nullptr) {
      return false;
    }
    size_t remain_size = len;
    auto dst_addr = ge::PtrToValue(static_cast<void *>(dst.get()));
    auto src_addr = ge::PtrToValue(static_cast<void *>(src.get()));
    while (remain_size > SECUREC_MEM_MAX_LEN) {
      if (memcpy_s(ge::ValueToPtr(dst_addr), SECUREC_MEM_MAX_LEN, ge::ValueToPtr(src_addr), SECUREC_MEM_MAX_LEN) !=
          EOK) {
        return false;
      }
      remain_size -= SECUREC_MEM_MAX_LEN;
      dst_addr += SECUREC_MEM_MAX_LEN;
      src_addr += SECUREC_MEM_MAX_LEN;
    }
    if ((remain_size != 0U) &&
        (memcpy_s(ge::ValueToPtr(dst_addr), remain_size, ge::ValueToPtr(src_addr), remain_size) != EOK)) {
      return false;
    }
    return true;
  }
  friend class TensorDesc;
};

class TensorDescImpl {
 public:
  TensorDescImpl() = default;
  ~TensorDescImpl() = default;
  TensorDescImpl(const Shape &shape, const Format format, const DataType dt)
      : shape_(shape), format_(format), data_type_(dt) {}

 private:
  Shape shape_;
  std::vector<std::pair<int64_t, int64_t>> range_;
  Format format_ = FORMAT_ND;
  Format origin_format_ = FORMAT_ND;
  bool origin_format_is_set_ = false;
  DataType data_type_ = DT_FLOAT;
  Shape origin_shape_;
  bool origin_shape_is_set_ = false;
  int64_t size_ = 0;
  int64_t real_dim_cnt_ = 0;
  std::string name_;
  Placement placement_ = kPlacementHost;
  TensorDescValue tensor_desc_value_;
  std::string expand_dims_rule_;
  bool reuse_input_ = false;
  uint32_t reuse_input_index_ = 0U;

  friend class TensorDesc;
  friend class TensorAdapter;
};

class TensorImpl {
 public:
  TensorImpl() = default;
  ~TensorImpl() = default;

  explicit TensorImpl(const TensorDesc &tensor_desc) : ge_tensor(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc)) {}
  TensorImpl(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data)
      : ge_tensor(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc), data) {}
  TensorImpl(const TensorDesc &tensor_desc, const uint8_t *const data, const size_t size)
      : ge_tensor(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc), data, size) {}
  TensorImpl(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data)
      : ge_tensor(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc), std::move(data)) {}

  graphStatus SetData(const std::string &data) {
    if (!data.empty()) {
      /// Extra 16 bytes store string head
      /// Extra 1 byte store '\0'
      const size_t total_size = data.size() + sizeof(StringHead) + 1U;
      const std::unique_ptr<char_t[]> buff = ComGraphMakeUnique<char_t[]>(total_size);
      if (buff == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "allocate string raw data buff failed, size:%zu", total_size);
        GELOGE(ge::GRAPH_FAILED, "[New][Buffer] allocate string raw data buff failed");
        return ge::GRAPH_FAILED;
      }
      StringHead *const string_head = PtrToPtr<char_t, StringHead>(buff.get());
      // Front 8 bytes store pointer of string
      char_t *const raw_data = PtrToPtr<void, char_t>(
          ge::ValueToPtr(ge::PtrToValue(PtrToPtr<char_t, void>(buff.get())) + sizeof(*string_head)));
      string_head->addr = static_cast<int64_t>(sizeof(StringHead));
      string_head->len = static_cast<int64_t>(data.size());
      const int32_t memcpy_ret = memcpy_s(raw_data, total_size - sizeof(StringHead), data.c_str(), data.size() + 1U);
      if (memcpy_ret != EOK) {
        REPORT_INNER_ERR_MSG("E18888", "memcpy data failed, ret:%d, size:%zu.", memcpy_ret, data.size() + 1U);
        GELOGE(ge::GRAPH_FAILED, "[Copy][Data] failed, ret:%d", memcpy_ret);
        return ge::GRAPH_FAILED;
      }
      (void)ge_tensor.SetData(PtrToPtr<char_t, const uint8_t>(buff.get()), total_size);
      return GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
  }

  graphStatus SetData(const std::vector<std::string> &data) {
    if (data.empty()) {
      REPORT_INNER_ERR_MSG("E18888", "there is no data, please check the input variable");
      GELOGE(ge::GRAPH_FAILED, "[Check][Param] there is no data, please check the input variable");
      return ge::GRAPH_FAILED;
    }
    size_t total_size = 0U;
    total_size = std::accumulate(data.begin(), data.end(), total_size, [](size_t total, const std::string &str) {
      /// Extra 16 bytes store string head
      /// Extra 1 byte store '\0'
      total += str.size() + sizeof(StringHead) + 1U;
      return total;
    });

    const std::unique_ptr<char_t[]> buff = ComGraphMakeUnique<char_t[]>(total_size);
    if (buff == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "allocate string raw data buff failed, size:%zu", total_size);
      GELOGE(ge::GRAPH_FAILED, "[New][Buffer] allocate string raw data buff failed");
      return ge::GRAPH_FAILED;
    }
    // Front some bytes store head of each string
    StringHead *const string_head = PtrToPtr<char_t, StringHead>(buff.get());
    uint64_t raw_data = ge::PtrToValue(static_cast<void *>(buff.get())) + (data.size() * sizeof(*string_head));
    uint64_t ptr_size = data.size() * sizeof(StringHead);
    for (size_t i = 0U; i < data.size(); ++i) {
      PtrAdd<StringHead>(string_head, data.size(), i)->addr = static_cast<int64_t>(ptr_size);
      PtrAdd<StringHead>(string_head, data.size(), i)->len = static_cast<int64_t>(data[i].size());
      if (total_size < ptr_size) {
        REPORT_INNER_ERR_MSG("E18888", "Subtraction invalid, total_size:%zu, ptr_size:%" PRIu64, total_size, ptr_size);
        GELOGE(ge::GRAPH_FAILED, "[Check][Param] Subtraction invalid, total_size: %zu, ptr_size: %" PRIu64, total_size,
               ptr_size);
        return ge::GRAPH_FAILED;
      }
      const int32_t memcpy_ret =
          memcpy_s(ge::ValueToPtr(raw_data), total_size - ptr_size, data[i].c_str(), data[i].size() + 1U);
      GE_CHK_BOOL_RET_STATUS(memcpy_ret == EOK, GRAPH_FAILED, "copy data failed");
      raw_data += (data[i].size() + 1U);
      ptr_size += (data[i].size() + 1U);
    }

    (void)ge_tensor.SetData(PtrToPtr<char_t, const uint8_t>(buff.get()), total_size);
    return GRAPH_SUCCESS;
  }

 private:
  GeTensor ge_tensor;
  friend class Tensor;
  friend class TensorAdapter;
};

class ShapeImpl {
 public:
  ShapeImpl() = default;
  ~ShapeImpl() = default;
  explicit ShapeImpl(const std::vector<int64_t> &dims) {
    bool is_unknown_dim_num = false;
    for (const auto &dim : dims) {
      if (dim == UNKNOWN_DIM_NUM) {
        is_unknown_dim_num = true;
        break;
      }
    }
    dims_ = is_unknown_dim_num ? std::vector<int64_t>({UNKNOWN_DIM_NUM}) : dims;
  }

 private:
  std::vector<int64_t> dims_;
  friend class Shape;
};

Shape::Shape() {
  impl_ = ComGraphMakeShared<ShapeImpl>();
}

Shape::Shape(const std::vector<int64_t> &dims) {
  impl_ = ComGraphMakeShared<ShapeImpl>(dims);
}

size_t Shape::GetDimNum() const {
  if (impl_ != nullptr) {
    const bool is_dim_unknown = std::any_of(std::begin(impl_->dims_), std::end(impl_->dims_),
                                            [](const int64_t i) { return i == UNKNOWN_DIM_NUM; });
    if (is_dim_unknown) {
      GELOGI("Dim num is unknown, return 0U.");
      return 0U;
    }
    return impl_->dims_.size();
  }
  return 0U;
}

int64_t Shape::GetDim(size_t idx) const {
  if (impl_ != nullptr) {
    if (idx >= impl_->dims_.size()) {
      return 0;
    }
    return impl_->dims_[idx];
  }
  return 0;
}

graphStatus Shape::SetDim(size_t idx, int64_t value) {
  if (impl_ != nullptr) {
    if (idx >= impl_->dims_.size()) {
      return ge::GRAPH_FAILED;
    }
    impl_->dims_[idx] = value;
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

std::vector<int64_t> Shape::GetDims() const {
  const std::vector<int64_t> dims;
  if (impl_ != nullptr) {
    return impl_->dims_;
  }
  return dims;
}

int64_t Shape::GetShapeSize() const {
  if (impl_ != nullptr) {
    if (impl_->dims_.empty()) {
      return 0;
    }
    int64_t size = 1;
    for (const auto i : impl_->dims_) {
      if ((i == UNKNOWN_DIM_NUM) || (i == UNKNOWN_DIM)) {
        return UNKNOWN_DIM_SIZE;
      }

      if (!Int64MulNotOverflow(size, i)) {
        REPORT_INNER_ERR_MSG("E18888", "mul overflow: %" PRId64 ", %" PRId64, size, i);
        GELOGE(ge::GRAPH_FAILED, "[Check][Overflow] mul overflow: %" PRId64 ", %" PRId64, size, i);
        size = 0;
        return size;
      }
      size *= i;
    }
    return size;
  }
  return 0;
}

TensorDesc::TensorDesc() {
  impl = ComGraphMakeSharedAndThrow<TensorDescImpl>();
}

TensorDesc::TensorDesc(Shape shape, Format format, DataType dt) {
  impl = ComGraphMakeSharedAndThrow<TensorDescImpl>(shape, format, dt);
  SetRealDimCnt(static_cast<int64_t>(shape.GetDimNum()));
}

TensorDesc::TensorDesc(const TensorDesc &desc) {
  // Copy
  impl = ComGraphMakeShared<TensorDescImpl>();
  if ((desc.impl != nullptr) && (impl != nullptr)) {
    *impl = *desc.impl;
  }
}

TensorDesc::TensorDesc(TensorDesc &&desc) {
  // Move
  impl = std::move(desc.impl);
}

TensorDesc &TensorDesc::operator=(const TensorDesc &desc) {
  // Copy
  if (&desc != this) {
    impl = ComGraphMakeShared<TensorDescImpl>();
    if ((desc.impl != nullptr) && (impl != nullptr)) {
      *impl = *desc.impl;
    }
  }
  return *this;
}

TensorDesc &TensorDesc::operator=(TensorDesc &&desc) {
  if (&desc != this) {
    impl = std::move(desc.impl);
  }
  return *this;
}

void TensorDesc::Update(const Shape &shape, Format format, DataType dt) {
  if (impl != nullptr) {
    impl->shape_ = shape;
    impl->format_ = format;
    impl->data_type_ = dt;
  }
}

Shape TensorDesc::GetShape() const {
  if (impl != nullptr) {
    return impl->shape_;
  }
  return Shape();
}

void TensorDesc::SetShape(const Shape &shape) {
  if (impl != nullptr) {
    impl->shape_ = shape;
  }
}

// set shape with -2, it stand for unknown shape
graphStatus TensorDesc::SetUnknownDimNumShape() {
  if (impl != nullptr) {
    impl->shape_ = Shape({UNKNOWN_DIM_NUM});
    return GRAPH_SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E18888", "Set unknown shape failed, because no impl class!");
  GELOGE(ge::GRAPH_FAILED, "[Set][UnknownDimNumShape] failed, because no impl class!");
  return ge::GRAPH_FAILED;
}

// for unknown shape
graphStatus TensorDesc::SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range) {
  if (impl != nullptr) {
    impl->range_ = range;
    return GRAPH_SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E18888", "SetShapeRange failed! impl is nullptr!");
  GELOGE(ge::GRAPH_FAILED, "[Set][ShapeRange] failed! impl is nullptr!");
  return ge::GRAPH_FAILED;
}
graphStatus TensorDesc::GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const {
  if (impl != nullptr) {
    range = impl->range_;
    return GRAPH_SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E18888", "impl is nullptr! check invalid");
  GELOGE(ge::GRAPH_FAILED, "[Check][Param] impl is nullptr! check invalid");
  return ge::GRAPH_FAILED;
}

Shape TensorDesc::GetOriginShape() const {
  if (impl != nullptr) {
    return impl->origin_shape_;
  }
  return Shape();
}

void TensorDesc::SetOriginShape(const Shape &origin_shape) {
  if (impl != nullptr) {
    impl->origin_shape_ = origin_shape;
    impl->origin_shape_is_set_ = true;
  }
}

Format TensorDesc::GetFormat() const {
  if (impl != nullptr) {
    return impl->format_;
  }
  return FORMAT_RESERVED;
}

void TensorDesc::SetFormat(Format format) {
  if (impl != nullptr) {
    impl->format_ = format;
  }
}

Format TensorDesc::GetOriginFormat() const {
  if (impl != nullptr) {
    return impl->origin_format_;
  }
  return FORMAT_RESERVED;
}

void TensorDesc::SetOriginFormat(Format origin_format) {
  if (impl != nullptr) {
    impl->origin_format_ = origin_format;
    impl->origin_format_is_set_ = true;
  }
}

DataType TensorDesc::GetDataType() const {
  if (impl != nullptr) {
    return impl->data_type_;
  }
  return DT_UNDEFINED;
}

void TensorDesc::SetDataType(DataType dt) {
  if (impl != nullptr) {
    impl->data_type_ = dt;
  }
}

void TensorDesc::SetSize(int64_t size) {
  if (impl != nullptr) {
    impl->size_ = size;
  }
}

int64_t TensorDesc::GetSize() const {
  if (impl != nullptr) {
    return impl->size_;
  }
  return 0;
}

void TensorDesc::SetRealDimCnt(const int64_t real_dim_cnt) {
  if (impl != nullptr) {
    impl->real_dim_cnt_ = real_dim_cnt;
  }
}

int64_t TensorDesc::GetRealDimCnt() const {
  if (impl != nullptr) {
    return impl->real_dim_cnt_;
  }
  return 0;
}

std::string TensorDesc::GetName() const {
  if (impl != nullptr) {
    return impl->name_;
  }
  return "";
}

void TensorDesc::SetName(const std::string &name) {
  if (impl != nullptr) {
    impl->name_ = name;
  }
}

graphStatus TensorDesc::GetName(AscendString &name) {
  if (impl != nullptr) {
    name = AscendString(impl->name_.c_str());
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

graphStatus TensorDesc::GetName(AscendString &name) const {
  if (impl != nullptr) {
    name = AscendString(impl->name_.c_str());
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

void TensorDesc::SetName(const char_t *name) {
  if ((impl != nullptr) && (name != nullptr)) {
    impl->name_ = name;
  }
}

void TensorDesc::SetPlacement(Placement placement) {
  if (impl != nullptr) {
    impl->placement_ = placement;
  }
}

Placement TensorDesc::GetPlacement() const {
  if (impl != nullptr) {
    return impl->placement_;
  }
  return kPlacementHost;
}

void TensorDesc::SetConstData(std::unique_ptr<uint8_t[]> const_data_buffer, const size_t &const_data_len) {
  if (impl != nullptr) {
    impl->tensor_desc_value_.const_data_buffer_ = std::move(const_data_buffer);
    impl->tensor_desc_value_.const_data_len_ = const_data_len;
  }
  return;
}

bool TensorDesc::GetConstData(uint8_t **const_data_buffer, size_t &const_data_len) const {
  if (impl != nullptr) {
    *const_data_buffer = impl->tensor_desc_value_.const_data_buffer_.get();
    const_data_len = impl->tensor_desc_value_.const_data_len_;
    return true;
  }
  return false;
}

void TensorDesc::SetExpandDimsRule(const AscendString &expand_dims_rule) {
  if (impl != nullptr) {
    impl->expand_dims_rule_ = expand_dims_rule.GetString();
  }
}

graphStatus TensorDesc::GetExpandDimsRule(AscendString &expand_dims_rule) const {
  if (impl != nullptr) {
    expand_dims_rule = AscendString(impl->expand_dims_rule_.c_str());
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

void TensorDesc::SetReuseInputIndex(const uint32_t idx) {
  if (impl != nullptr) {
    impl->reuse_input_ = true;
    impl->reuse_input_index_ = idx;
  }
}

Tensor::Tensor() {
  impl = ComGraphMakeSharedAndThrow<TensorImpl>();
}

Tensor::Tensor(const TensorDesc &tensor_desc) {
  impl = ComGraphMakeSharedAndThrow<TensorImpl>(tensor_desc);
}

static void CheckTensorParam(const uint64_t shape_size, const DataType data_type, const size_t data_size) {
  uint32_t type_length;
  const bool ret = TypeUtils::GetDataTypeLength(data_type, type_length);
  if (!ret) {
    GELOGW("[Create][Tensor] Datatype %d not found.", data_type);
  }

  if (ret && ((shape_size != 0U) || (data_size != type_length))) {
    if ((type_length != 0U) && ((UINT64_MAX / type_length) < shape_size)) {
      GELOGW("[Create][Tensor] Calculate size failed, as mul overflow: %" PRIu64 " * %" PRIu32, shape_size,
             type_length);
    } else {
      if ((shape_size * type_length) != data_size) {
        GELOGW("[Create][Tensor] Tensor length not equal: shape_byte_size=%" PRIu64 ", dt_type=%s, data_size=%zu.",
               shape_size * type_length, TypeUtils::DataTypeToSerialString(data_type).c_str(), data_size);
      }
    }
  }
}

Tensor::Tensor(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data) {
  CheckTensorParam(static_cast<uint64_t>(tensor_desc.GetShape().GetShapeSize()), tensor_desc.GetDataType(),
                   data.size());
  impl = ComGraphMakeShared<TensorImpl>(tensor_desc, data);
}

Tensor::Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size) {
  CheckTensorParam(static_cast<uint64_t>(tensor_desc.GetShape().GetShapeSize()), tensor_desc.GetDataType(), size);
  impl = ComGraphMakeShared<TensorImpl>(tensor_desc, data, size);
}

Tensor::Tensor(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data) {
  CheckTensorParam(static_cast<uint64_t>(tensor_desc.GetShape().GetShapeSize()), tensor_desc.GetDataType(),
                   data.size());
  impl = ComGraphMakeShared<TensorImpl>(std::move(tensor_desc), std::move(data));
}

TensorDesc Tensor::GetTensorDesc() const {
  if (impl != nullptr) {
    return TensorAdapter::GeTensorDesc2TensorDesc(impl->ge_tensor.MutableTensorDesc());
  }
  return TensorDesc();
}

graphStatus Tensor::SetTensorDesc(const TensorDesc &tensor_desc) {
  if (impl != nullptr) {
    impl->ge_tensor.SetTensorDesc(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

const uint8_t *Tensor::GetData() const {
  if (impl != nullptr) {
    return impl->ge_tensor.GetData().data();
  }
  return nullptr;
}

uint8_t *Tensor::GetData() {
  if (impl != nullptr) {
    return impl->ge_tensor.MutableData().data();
  }
  return nullptr;
}

size_t Tensor::GetSize() const {
  if (impl != nullptr) {
    return impl->ge_tensor.GetData().size();
  }
  return 0U;
}

std::unique_ptr<uint8_t[], Tensor::DeleteFunc> Tensor::ResetData() {
  if (impl != nullptr) {
    auto aligned_ptr = impl->ge_tensor.GetAlignedPtr();
    if (aligned_ptr != nullptr) {
      return aligned_ptr->Reset();
    }
  }
  return nullptr;
}

graphStatus Tensor::SetData(std::vector<uint8_t> &&data) {
  if (impl != nullptr) {
    (void)impl->ge_tensor.SetData(data);
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::vector<uint8_t> &data) {
  if (impl != nullptr) {
    (void)impl->ge_tensor.SetData(data);
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

graphStatus Tensor::SetData(const uint8_t *data, size_t size) {
  if (impl != nullptr) {
    (void)impl->ge_tensor.SetData(data, size);
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::string &data) {
  if (impl != nullptr) {
    if (impl->SetData(data) != GRAPH_SUCCESS) {
      GELOGE(ge::GRAPH_FAILED, "[Set][Data] %s failed.", data.c_str());
      return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::vector<std::string> &data) {
  if (impl != nullptr) {
    if (impl->SetData(data) != GRAPH_SUCCESS) {
      GELOGE(ge::GRAPH_FAILED, "[Call][SetData] Tensor set vector data failed.");
      return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

graphStatus Tensor::SetData(const char_t *data) {
  if ((impl != nullptr) && (data != nullptr)) {
    const std::string tensor_data = data;
    if (impl->SetData(tensor_data) != GRAPH_SUCCESS) {
      GELOGE(ge::GRAPH_FAILED, "[Call][SetData] Tensor set data(%s) failed.", data);
      return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::vector<AscendString> &ascend_strings) {
  if (impl != nullptr) {
    std::vector<std::string> tensor_data;
    for (auto &data : ascend_strings) {
      if (data.GetString() == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "Data is nullptr. check invalid");
        GELOGE(ge::GRAPH_FAILED, "[Check][Param] Data is nullptr.");
        return ge::GRAPH_FAILED;
      }
      tensor_data.emplace_back(data.GetString());
    }
    if (impl->SetData(tensor_data) != GRAPH_SUCCESS) {
      GELOGE(ge::GRAPH_FAILED, "[Call][SetData] Tensor set vector data failed.");
      return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

graphStatus Tensor::SetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func) {
  if (impl != nullptr) {
    if (impl->ge_tensor.SetData(data, size, deleter_func) != GRAPH_SUCCESS) {
      GELOGE(ge::GRAPH_FAILED, "[Call][SetData] Tensor set data with deleter function failed");
      return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

graphStatus Tensor::IsValid() {
  const uint64_t shape_size = static_cast<uint64_t>(GetTensorDesc().GetShape().GetShapeSize());
  const DataType data_type = GetTensorDesc().GetDataType();
  uint32_t type_length;
  const bool ret = TypeUtils::GetDataTypeLength(data_type, type_length);
  if (!ret) {
    GELOGW("[Check][Tensor] Datatype %d not found.", data_type);
    return GRAPH_SUCCESS;
  }

  const size_t data_size = GetSize();
  if (data_type == DT_STRING) {
    return GRAPH_SUCCESS;
  }

  if ((shape_size != 0U) || (data_size != type_length)) {
    if ((type_length != 0U) && ((UINT64_MAX / type_length) < shape_size)) {
      GELOGW("[Check][Tensor] Calculate size failed, as mul overflow: %" PRIu64 " * %" PRIu32, shape_size, type_length);
    } else {
      if ((shape_size * type_length) != data_size) {
        GELOGE(ge::GRAPH_FAILED,
               "[Check][Tensor] Tensor length not equal: shape_byte_size=%" PRIu64 ", dt_type=%s, data_size=%zu.",
               shape_size * type_length, TypeUtils::DataTypeToSerialString(data_type).c_str(), data_size);
        return ge::GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus Tensor::SetOriginShapeDimNum(const size_t dim_num) {
  if (impl != nullptr) {
    impl->ge_tensor.MutableTensorDesc().MutableOriginShape().SetDimNum(dim_num);
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

size_t Tensor::GetOriginShapeDimNum() const {
  if (impl != nullptr) {
    return impl->ge_tensor.GetTensorDesc().GetOriginShape().GetDimNum();
  }
  return 0U;
}

graphStatus Tensor::SetOriginShapeDim(const size_t idx, const int64_t dim_value) {
  if (impl != nullptr) {
    return impl->ge_tensor.MutableTensorDesc().MutableOriginShape().SetDim(idx, dim_value);
  }
  return ge::GRAPH_FAILED;
}

int64_t Tensor::GetOriginShapeDim(const size_t idx) const {
  if (impl != nullptr) {
    return impl->ge_tensor.GetTensorDesc().GetOriginShape().GetDim(idx);
  }
  return 0;
}

graphStatus Tensor::SetOriginFormat(const ge::Format &format) {
  if (impl != nullptr) {
    impl->ge_tensor.MutableTensorDesc().SetOriginFormat(format);
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

ge::Format Tensor::GetOriginFormat() const {
  if (impl != nullptr) {
    return impl->ge_tensor.GetTensorDesc().GetOriginFormat();
  }
  return ge::FORMAT_RESERVED;
}

graphStatus Tensor::SetShapeDimNum(const size_t dim_num) {
  if (impl != nullptr) {
    impl->ge_tensor.MutableTensorDesc().MutableShape().SetDimNum(dim_num);
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

size_t Tensor::GetShapeDimNum() const {
  if (impl != nullptr) {
    return impl->ge_tensor.GetTensorDesc().GetShape().GetDimNum();
  }
  return 0U;
}

graphStatus Tensor::SetShapeDim(const size_t idx, const int64_t dim_value) {
  if (impl != nullptr) {
    return impl->ge_tensor.MutableTensorDesc().MutableShape().SetDim(idx, dim_value);
  }
  return ge::GRAPH_FAILED;
}

int64_t Tensor::GetShapeDim(const size_t idx) const {
  if (impl != nullptr) {
    return impl->ge_tensor.GetTensorDesc().GetShape().GetDim(idx);
  }
  return 0;
}

graphStatus Tensor::SetFormat(const ge::Format &format) {
  if (impl != nullptr) {
    impl->ge_tensor.MutableTensorDesc().SetFormat(format);
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

ge::Format Tensor::GetFormat() const {
  if (impl != nullptr) {
    return impl->ge_tensor.GetTensorDesc().GetFormat();
  }
  return ge::FORMAT_RESERVED;
}

graphStatus Tensor::SetDataType(const ge::DataType &dtype) {
  if (impl != nullptr) {
    impl->ge_tensor.MutableTensorDesc().SetDataType(dtype);
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

ge::DataType Tensor::GetDataType() const {
  if (impl != nullptr) {
    return impl->ge_tensor.GetTensorDesc().GetDataType();
  }
  return ge::DT_UNDEFINED;
}

graphStatus Tensor::SetPlacement(const ge::Placement &placement) {
  if (impl != nullptr) {
    impl->ge_tensor.MutableTensorDesc().SetPlacement(placement);
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

ge::Placement Tensor::GetPlacement() const {
  if (impl != nullptr) {
    return impl->ge_tensor.GetTensorDesc().GetPlacement();
  }
  return ge::Placement::kPlacementEnd;
}

graphStatus Tensor::SetExpandDimsRule(const AscendString &expand_dims_rule) {
  if (impl != nullptr) {
    impl->ge_tensor.MutableTensorDesc().SetExpandDimsRule(expand_dims_rule.GetString());
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

graphStatus Tensor::GetExpandDimsRule(AscendString &expand_dims_rule) const {
  if (impl != nullptr) {
    expand_dims_rule = AscendString(impl->ge_tensor.GetTensorDesc().GetExpandDimsRule().c_str());
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

graphStatus Tensor::ResetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func) {
  if (impl != nullptr) {
    if (impl->ge_tensor.ResetData(data, size, deleter_func) != GRAPH_SUCCESS) {
      GELOGE(ge::GRAPH_FAILED, "[Call][SetData] Tensor set data with deleter function failed");
      return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

Tensor Tensor::Clone() const {
  const Tensor tensor;
  if ((impl != nullptr) && (tensor.impl != nullptr)) {
    tensor.impl->ge_tensor = impl->ge_tensor.Clone();
  }
  return tensor;
}

GeTensorDesc TensorAdapter::TensorDesc2GeTensorDesc(const TensorDesc &tensor_desc) {
  GeTensorDesc ge_tensor_desc(GeShape(tensor_desc.GetShape().GetDims()), tensor_desc.GetFormat(),
                              tensor_desc.GetDataType());
  if (tensor_desc.impl->origin_format_is_set_) {
    (void)AttrUtils::SetBool(ge_tensor_desc, ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
  }
  if (tensor_desc.impl->origin_shape_is_set_) {
    ge_tensor_desc.SetOriginShape(GeShape(tensor_desc.GetOriginShape().GetDims()));
  }
  ge_tensor_desc.SetOriginFormat(tensor_desc.GetOriginFormat());
  ge_tensor_desc.SetExpandDimsRule(tensor_desc.impl->expand_dims_rule_);
  TensorUtils::SetReuseInput(ge_tensor_desc, tensor_desc.impl->reuse_input_);
  TensorUtils::SetReuseInputIndex(ge_tensor_desc, tensor_desc.impl->reuse_input_index_);

  AscendString name("");
  (void)tensor_desc.GetName(name);
  ge_tensor_desc.SetName(name.GetString());
  ge_tensor_desc.SetPlacement(tensor_desc.GetPlacement());
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  auto status = tensor_desc.GetShapeRange(shape_range);
  if (status != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Get shape range failed! ret:%u", status);
    GELOGE(ge::GRAPH_FAILED, "[Get][ShapeRange] failed! ret:%u", status);
    return ge_tensor_desc;
  }
  status = ge_tensor_desc.SetShapeRange(shape_range);
  if (status != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Set shape range failed! ret:%u", status);
    GELOGE(ge::GRAPH_FAILED, "[Set][ShapeRange] failed! ret:%u", status);
    return ge_tensor_desc;
  }
  const auto size = tensor_desc.GetSize();
  TensorUtils::SetSize(ge_tensor_desc, size);

  const auto real_dim_cnt = static_cast<uint32_t>(tensor_desc.GetRealDimCnt());
  TensorUtils::SetRealDimCnt(ge_tensor_desc, real_dim_cnt);
  return ge_tensor_desc;
}

TensorDesc TensorAdapter::GeTensorDesc2TensorDesc(const GeTensorDesc &ge_tensor_desc) {
  TensorDesc tensor_desc(Shape(ge_tensor_desc.GetShape().GetDims()), ge_tensor_desc.GetFormat(),
                         ge_tensor_desc.GetDataType());
  if (TensorUtils::IsOriginShapeInited(ge_tensor_desc)) {
    tensor_desc.SetOriginShape(Shape(ge_tensor_desc.GetOriginShape().GetDims()));
  }
  tensor_desc.SetOriginFormat(ge_tensor_desc.GetOriginFormat());
  tensor_desc.SetName(ge_tensor_desc.GetName().c_str());
  tensor_desc.SetPlacement(ge_tensor_desc.GetPlacement());
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  auto status = ge_tensor_desc.GetShapeRange(shape_range);
  if (status != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Get shape range failed! ret:%u", status);
    GELOGE(ge::GRAPH_FAILED, "[Get][ShapeRange] failed! ret:%u", status);
    return tensor_desc;
  }
  status = tensor_desc.SetShapeRange(shape_range);
  if (status != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Set shape range failed! ret:%u", status);
    GELOGE(ge::GRAPH_FAILED, "[Set][ShapeRange] failed! ret:%u", status);
    return tensor_desc;
  }
  int64_t size = 0;
  (void)TensorUtils::GetSize(ge_tensor_desc, size);
  tensor_desc.SetSize(size);

  uint32_t real_dim_cnt = 0U;
  (void)TensorUtils::GetRealDimCnt(ge_tensor_desc, real_dim_cnt);
  tensor_desc.SetRealDimCnt(static_cast<int64_t>(real_dim_cnt));

  tensor_desc.SetExpandDimsRule(AscendString(ge_tensor_desc.GetExpandDimsRule().c_str()));
  return tensor_desc;
}

Tensor TensorAdapter::GeTensor2Tensor(const ConstGeTensorPtr &ge_tensor) {
  const Tensor tensor;
  if ((ge_tensor != nullptr) && (tensor.impl != nullptr)) {
    tensor.impl->ge_tensor = ge_tensor->Clone();
  }
  return tensor;
}

ConstGeTensorPtr TensorAdapter::AsGeTensorPtr(const Tensor &tensor) {
  GeTensorPtr ge_tensor;
  if (tensor.impl != nullptr) {
    ge_tensor = ComGraphMakeShared<GeTensor>(tensor.impl->ge_tensor);
  }
  return ge_tensor;
}

GeTensorPtr TensorAdapter::AsGeTensorPtr(Tensor &tensor) {
  GeTensorPtr ge_tensor;
  if (tensor.impl != nullptr) {
    ge_tensor = ComGraphMakeShared<GeTensor>(tensor.impl->ge_tensor);
  }
  return ge_tensor;
}

const GeTensor TensorAdapter::AsGeTensor(const Tensor &tensor) {
  if (tensor.impl != nullptr) {
    return tensor.impl->ge_tensor;
  }
  return GeTensor();
}

const GeTensor *TensorAdapter::AsBareGeTensorPtr(const Tensor &tensor) {
  if (tensor.impl != nullptr) {
    return &(tensor.impl->ge_tensor);
  }
  return nullptr;
}

GeTensor TensorAdapter::AsGeTensorShared(const Tensor &tensor) {
  if (tensor.impl != nullptr) {
    // Construct new rvalue ge tensor to avoid call copy constructor
    return GeTensor(tensor.impl->ge_tensor.impl_);
  }
  return {};
}

GeTensor TensorAdapter::NormalizeGeTensor(const GeTensor &tensor) {
  auto normalized_tensor = tensor;
  auto &desc = normalized_tensor.MutableTensorDesc();
  NormalizeGeTensorDesc(desc);
  return normalized_tensor;
}

void TensorAdapter::NormalizeGeTensorDesc(GeTensorDesc &desc) {
  bool origin_format_is_set = false;
  if (AttrUtils::GetBool(desc, ATTR_NAME_ORIGIN_FORMAT_IS_SET, origin_format_is_set) && origin_format_is_set &&
      TensorUtils::IsOriginShapeInited(desc)) {
    (void)AttrUtils::SetInt(desc, ATTR_NAME_STORAGE_FORMAT, static_cast<int64_t>(desc.GetFormat()));
    (void)AttrUtils::SetListInt(desc, ATTR_NAME_STORAGE_SHAPE, desc.GetShape().GetDims());
    desc.SetFormat(desc.GetOriginFormat());
    desc.SetShape(desc.GetOriginShape());
    (void)AttrUtils::SetBool(desc, ATTR_NAME_ORIGIN_FORMAT_IS_SET, false);
  }
}

GeTensor TensorAdapter::AsGeTensor(Tensor &tensor) {
  if (tensor.impl != nullptr) {
    return tensor.impl->ge_tensor;
  }
  return GeTensor();
}

const Tensor TensorAdapter::AsTensor(const GeTensor &ge_tensor) {
  const Tensor tensor;
  if (tensor.impl != nullptr) {
    tensor.impl->ge_tensor = ge_tensor;
  }
  return tensor;
}

Tensor TensorAdapter::AsTensor(GeTensor &ge_tensor) {
  const Tensor tensor;
  if (tensor.impl != nullptr) {
    tensor.impl->ge_tensor = ge_tensor;
  }
  return tensor;
}
}  // namespace af
