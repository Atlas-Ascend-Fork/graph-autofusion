/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_PROGRAM_TILING_RESULT_WRITER_H_
#define ATT_PROGRAM_TILING_RESULT_WRITER_H_
#include "tiling_expression.h"

namespace att {
struct TilingResultField {
  uint32_t value_index;
  uint32_t byte_offset;
  TilingProgramIRValueType type;
};

class TilingResultLayout final {
 public:
  static TilingExpressionStatus Build(uint32_t payload_size, uint32_t value_count,
                                      const std::vector<TilingResultField> &fields,
                                      std::shared_ptr<const TilingResultLayout> *output);

  uint32_t PayloadSize() const {
    return payload_size_;
  }
  uint32_t ValueCount() const {
    return value_count_;
  }
  const std::vector<TilingResultField> &Fields() const {
    return fields_;
  }
  bool HasAllValues() const {
    return all_values_bound_;
  }
  bool HasOnlyTileFields() const;
  static bool IsTileField(const TilingResultField &field);

  // Values and destination must not overlap. A failed write leaves the destination untouched.
  TilingExpressionStatus Write(const TilingProgramIRValue *values, size_t value_count, void *destination,
                               size_t capacity) const;
  TilingExpressionStatus WriteTiles(const int64_t *values, size_t value_count, void *destination,
                                    size_t capacity) const;

 private:
  struct ConstructionKey {
    explicit ConstructionKey() = default;
  };

 public:
  explicit TilingResultLayout(ConstructionKey) {}

 private:
  uint32_t payload_size_ = 0;
  uint32_t value_count_ = 0;
  bool all_values_bound_ = false;
  std::vector<TilingResultField> fields_;
};
}  // namespace att
#endif
