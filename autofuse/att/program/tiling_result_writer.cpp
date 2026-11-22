/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tiling_result_writer.h"
#include <climits>
#include <algorithm>

namespace att {
namespace {
using Status = TilingExpressionStatus;
using Type = TilingProgramIRValueType;
uint32_t FieldBytes(Type type) {
  switch (type) {
    case Type::kBoolean:
    case Type::kInt8:
    case Type::kUint8:
      return 1;
    case Type::kInt16:
    case Type::kUint16:
      return sizeof(uint16_t);
    case Type::kFloat32:
    case Type::kInt32:
    case Type::kUint32:
      return sizeof(uint32_t);
    case Type::kFloat64:
    case Type::kInt64:
    case Type::kUint64:
      return sizeof(uint64_t);
    default:
      return 0;
  }
}
void StoreField(uint8_t *bytes, const TilingResultField &field, uint64_t bits) {
  const auto width = FieldBytes(field.type);
  for (uint32_t byte = 0; byte < width; ++byte) {
    bytes[field.byte_offset + byte] = static_cast<uint8_t>(bits >> (CHAR_BIT * byte));
  }
}

bool ConvertTile(int64_t value, const TilingResultField &field, TilingProgramIRValue &converted) {
  const auto type = field.type;
  if (!TilingResultLayout::IsTileField(field) || (value < 0 && (type == Type::kUint8 || type == Type::kUint16 ||
                                                                type == Type::kUint32 || type == Type::kUint64))) {
    return false;
  }
  const auto bits = static_cast<uint64_t>(value);
  TilingProgramIRValue round_trip;
  using ValueStatus = TilingProgramValueStatus;
  return CastTilingProgramIRValue({Type::kInt64, bits}, type, &converted) == ValueStatus::kOk &&
         CastTilingProgramIRValue(converted, Type::kInt64, &round_trip) == ValueStatus::kOk &&
         round_trip.raw_bits == bits;
}
}  // namespace

bool TilingResultLayout::IsTileField(const TilingResultField &field) {
  return FieldBytes(field.type) != 0 && field.type != Type::kBoolean && field.type != Type::kFloat32 &&
         field.type != Type::kFloat64;
}

bool TilingResultLayout::HasOnlyTileFields() const {
  return std::all_of(fields_.begin(), fields_.end(), IsTileField);
}

TilingExpressionStatus TilingResultLayout::Build(uint32_t payload_size, uint32_t value_count,
                                                 const std::vector<TilingResultField> &fields,
                                                 std::shared_ptr<const TilingResultLayout> *output) {
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (fields.size() > kTilingExpressionMaxItems || value_count > kTilingExpressionMaxItems) {
    return Status::kInvalidProgram;
  }
  auto layout = std::make_shared<TilingResultLayout>(TilingResultLayout::ConstructionKey{});
  layout->payload_size_ = payload_size;
  layout->value_count_ = value_count;
  layout->fields_ = fields;
  std::sort(layout->fields_.begin(), layout->fields_.end(),
            [](const auto &a, const auto &b) { return a.byte_offset < b.byte_offset; });
  uint32_t end = 0;
  std::vector<bool> bound(value_count);
  for (const auto &field : layout->fields_) {
    const auto width = FieldBytes(field.type);
    if (width == 0 || field.value_index >= value_count || field.byte_offset < end || field.byte_offset > payload_size ||
        width > payload_size - field.byte_offset) {
      return Status::kInvalidProgram;
    }
    end = field.byte_offset + width;
    bound[field.value_index] = true;
  }
  layout->all_values_bound_ = std::find(bound.begin(), bound.end(), false) == bound.end();
  *output = std::move(layout);
  return Status::kOk;
}

TilingExpressionStatus TilingResultLayout::Write(const TilingProgramIRValue *values, size_t value_count,
                                                 void *destination, size_t capacity) const {
  if (capacity < payload_size_) {
    return Status::kBufferTooSmall;
  }
  if (value_count != value_count_ || (value_count != 0 && values == nullptr) ||
      (payload_size_ != 0 && destination == nullptr)) {
    return Status::kInvalidArgument;
  }
  for (const auto &field : fields_) {
    const auto &value = values[field.value_index];
    if (value.type != field.type || ValidateTilingProgramIRValue(value) != TilingProgramValueStatus::kOk) {
      return Status::kInputError;
    }
  }
  auto *bytes = static_cast<uint8_t *>(destination);
  for (const auto &field : fields_) {
    StoreField(bytes, field, values[field.value_index].raw_bits);
  }
  return Status::kOk;
}
TilingExpressionStatus TilingResultLayout::WriteTiles(const int64_t *values, size_t value_count, void *destination,
                                                      size_t capacity) const {
  if (capacity < payload_size_) {
    return Status::kBufferTooSmall;
  }
  if (value_count != value_count_ || (value_count != 0 && values == nullptr) ||
      (payload_size_ != 0 && destination == nullptr)) {
    return Status::kInvalidArgument;
  }
  std::vector<TilingProgramIRValue> converted(fields_.size());
  for (size_t i = 0; i < fields_.size(); ++i) {
    const auto &field = fields_[i];
    if (!ConvertTile(values[field.value_index], field, converted[i])) {
      return Status::kInputError;
    }
  }
  auto *bytes = static_cast<uint8_t *>(destination);
  for (size_t i = 0; i < fields_.size(); ++i) {
    StoreField(bytes, fields_[i], converted[i].raw_bits);
  }
  return Status::kOk;
}
}  // namespace att
