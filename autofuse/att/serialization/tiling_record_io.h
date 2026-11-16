/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_SERIALIZATION_TILING_RECORD_IO_H_
#define ATT_SERIALIZATION_TILING_RECORD_IO_H_
#include <limits>
#include "program/tiling_expression.h"

namespace att::tiling_record {
using Type = TilingProgramIRValueType;
constexpr size_t kWordBytes = sizeof(uint32_t);
constexpr unsigned kWordBits = std::numeric_limits<uint32_t>::digits;
constexpr unsigned kByteBits = std::numeric_limits<uint8_t>::digits;
constexpr size_t kValueBytes = 3 * kWordBytes;  // type, low word, high word
constexpr size_t kInstructionBytes = 8 * kWordBytes;
class Writer {
 public:
  std::vector<uint8_t> bytes;
  void Word(uint32_t value) {
    for (unsigned shift = 0; shift < kWordBits; shift += kByteBits) {
      bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
  }
  void Value(const TilingProgramIRValue &value) {
    Word(static_cast<uint32_t>(value.type));
    Word(static_cast<uint32_t>(value.raw_bits));
    Word(static_cast<uint32_t>(value.raw_bits >> kWordBits));
  }
};

class Reader {
 public:
  Reader(const void *data, size_t size) : cursor_(static_cast<const uint8_t *>(data)), remaining_(size) {}
  bool Word(uint32_t &value) {
    if (remaining_ < kWordBytes) {
      return false;
    }
    value = 0;
    for (unsigned shift = 0; shift < kWordBits; shift += kByteBits) {
      value |= static_cast<uint32_t>(*cursor_++) << shift;
    }
    remaining_ -= kWordBytes;
    return true;
  }
  bool Count(uint32_t &count, size_t minimum_bytes) {
    return Word(count) && count <= kTilingExpressionMaxItems && count <= remaining_ / minimum_bytes;
  }
  bool ReadType(Type &type) {
    uint32_t word;
    if (!Word(word) || word > UINT8_MAX) {
      return false;
    }
    type = static_cast<Type>(word);
    return ValidateTilingProgramIRValue({type, 0}) == TilingProgramValueStatus::kOk;
  }
  bool Value(TilingProgramIRValue &value) {
    uint32_t low, high;
    if (!ReadType(value.type) || !Word(low) || !Word(high)) {
      return false;
    }
    value.raw_bits = low | (static_cast<uint64_t>(high) << kWordBits);
    return true;
  }
  bool Instruction(TilingProgramIRInstruction &instruction) {
    uint32_t kind, opcode, boolean;
    if (!Word(kind) || kind > UINT8_MAX || !Word(opcode) || opcode > UINT16_MAX || !Word(instruction.constant_id) ||
        !Word(instruction.symbol_id) || !Word(instruction.jump) || !Word(instruction.arg_count) || !Word(boolean) ||
        boolean > 1 || !ReadType(instruction.value_type)) {
      return false;
    }
    instruction.kind = static_cast<TilingProgramIRInstructionKind>(kind);
    instruction.opcode = static_cast<TilingProgramIROpcode>(opcode);
    instruction.boolean_value = boolean != 0;
    return true;
  }
  bool Skip(size_t bytes) {
    if (bytes > remaining_) {
      return false;
    }
    cursor_ += bytes;
    remaining_ -= bytes;
    return true;
  }
  bool Done() const {
    return remaining_ == 0;
  }
  const uint8_t *Data() const {
    return cursor_;
  }
  size_t Size() const {
    return remaining_;
  }

 private:
  const uint8_t *cursor_;
  size_t remaining_;
};

}  // namespace att::tiling_record
#endif
