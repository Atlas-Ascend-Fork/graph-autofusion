/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tiling_result_layout_codec.h"
#include "tiling_record_io.h"

namespace att {
namespace {
using Status = TilingExpressionStatus;
constexpr size_t kFieldBytes = 3 * tiling_record::kWordBytes;  // slot, offset and type
constexpr uint32_t kVersion = 1;
}  // namespace

TilingExpressionStatus EncodeTilingResultLayoutV1(uint32_t payload_size, uint32_t value_count,
                                                  const std::vector<TilingResultField> &fields,
                                                  std::vector<uint8_t> *output) {
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  std::shared_ptr<const TilingResultLayout> validated;
  const auto status = TilingResultLayout::Build(payload_size, value_count, fields, &validated);
  if (status != Status::kOk) {
    return status;
  }
  tiling_record::Writer writer;
  writer.Word(kVersion);
  writer.Word(payload_size);
  writer.Word(value_count);
  writer.Word(static_cast<uint32_t>(fields.size()));
  for (const auto &field : fields) {
    writer.Word(field.value_index);
    writer.Word(field.byte_offset);
    writer.Word(static_cast<uint32_t>(field.type));
  }
  *output = std::move(writer.bytes);
  return Status::kOk;
}

TilingExpressionStatus DecodeTilingResultLayoutV1(const void *data, size_t size,
                                                  std::shared_ptr<const TilingResultLayout> *output) {
  if (data == nullptr || output == nullptr) {
    return Status::kInvalidArgument;
  }
  tiling_record::Reader reader(data, size);
  uint32_t version, payload_size, value_count, count;
  if (!reader.Word(version)) {
    return Status::kInvalidProgram;
  }
  if (version != kVersion) {
    return Status::kUnsupported;
  }
  if (!reader.Word(payload_size) || !reader.Word(value_count) || !reader.Count(count, kFieldBytes) ||
      reader.Size() != static_cast<size_t>(count) * kFieldBytes) {
    return Status::kInvalidProgram;
  }
  std::vector<TilingResultField> fields(count);
  for (auto &field : fields) {
    if (!reader.Word(field.value_index) || !reader.Word(field.byte_offset) || !reader.ReadType(field.type)) {
      return Status::kInvalidProgram;
    }
  }
  return TilingResultLayout::Build(payload_size, value_count, fields, output);
}
}  // namespace att
