/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tiling_stage_codec.h"
#include "aftp_v1.h"
#include "tiling_record_io.h"

namespace att {
namespace {
using Status = TilingExpressionStatus;
using Type = TilingProgramIRValueType;
constexpr uint32_t kStageVersion = 1;

using tiling_record::kWordBytes;
using tiling_record::Reader;
using tiling_record::Writer;
constexpr size_t kMinExpressionBytes = 3 * kWordBytes;  // result type and two counts
constexpr size_t kConditionalReturnBytes = 2 * kWordBytes + tiling_record::kValueBytes;

bool ReadExpression(Reader &reader, TilingProgramIRExpressionCode &expression) {
  uint32_t count;
  if (!reader.ReadType(expression.result_type) || !reader.Count(count, tiling_record::kValueBytes)) {
    return false;
  }
  expression.constants.resize(count);
  for (auto &value : expression.constants) {
    if (!reader.Value(value)) {
      return false;
    }
  }
  if (!reader.Count(count, tiling_record::kInstructionBytes)) {
    return false;
  }
  expression.instructions.resize(count);
  for (auto &instruction : expression.instructions) {
    if (!reader.Instruction(instruction)) {
      return false;
    }
  }
  return true;
}
}  // namespace

TilingExpressionStatus EncodeTilingStageV1(const TilingProgramIRStageDefinition &definition,
                                           std::vector<uint8_t> *output) {
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  std::shared_ptr<const TilingProgramIRCompiledStage> validated;
  const auto error = BuildTilingProgramIRStage(definition, &validated);
  if (error.status != Status::kOk) {
    return error.status;
  }
  Writer writer;
  writer.Word(kStageVersion);
  writer.Word(static_cast<uint32_t>(definition.input_types.size()));
  for (auto type : definition.input_types) {
    writer.Word(static_cast<uint32_t>(type));
  }
  writer.Word(static_cast<uint32_t>(definition.expressions.size()));
  for (const auto &expression : definition.expressions) {
    writer.Word(static_cast<uint32_t>(expression.result_type));
    writer.Word(static_cast<uint32_t>(expression.constants.size()));
    for (const auto &value : expression.constants) {
      writer.Value(value);
    }
    writer.Word(static_cast<uint32_t>(expression.instructions.size()));
    for (const auto &instruction : expression.instructions) {
      writer.Word(static_cast<uint32_t>(instruction.kind));
      writer.Word(static_cast<uint32_t>(instruction.opcode));
      writer.Word(instruction.constant_id);
      writer.Word(instruction.symbol_id);
      writer.Word(instruction.jump);
      writer.Word(instruction.arg_count);
      writer.Word(instruction.boolean_value);
      writer.Word(static_cast<uint32_t>(instruction.value_type));
    }
  }
  writer.Word(static_cast<uint32_t>(definition.conditional_returns.size()));
  for (const auto &branch : definition.conditional_returns) {
    writer.Word(branch.condition_step);
    writer.Word(branch.value_step);
    writer.Value(branch.value);
  }
  *output = std::move(writer.bytes);
  return Status::kOk;
}

TilingExpressionStatus DecodeTilingStageV1(const void *data, size_t size, TilingProgramIRStageDefinition *output) {
  if (data == nullptr || output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (size > kAftpV1MaxBytes) {
    return Status::kInvalidProgram;
  }
  Reader reader(data, size);
  uint32_t version, count;
  if (!reader.Word(version)) {
    return Status::kInvalidProgram;
  }
  if (version != kStageVersion) {
    return Status::kUnsupported;
  }
  TilingProgramIRStageDefinition decoded;
  if (!reader.Count(count, kWordBytes)) {
    return Status::kInvalidProgram;
  }
  decoded.input_types.resize(count);
  for (auto &type : decoded.input_types) {
    if (!reader.ReadType(type)) {
      return Status::kInvalidProgram;
    }
  }
  if (!reader.Count(count, kMinExpressionBytes)) {
    return Status::kInvalidProgram;
  }
  auto visible = decoded.input_types;
  size_t remaining_items = kTilingExpressionMaxItems - visible.size();
  for (uint32_t i = 0; i < count; ++i) {
    TilingProgramIRExpressionCode expression;
    if (visible.size() > remaining_items || !ReadExpression(reader, expression)) {
      return Status::kInvalidProgram;
    }
    remaining_items -= visible.size();
    const size_t items = expression.constants.size() + expression.instructions.size();
    if (items > remaining_items) {
      return Status::kInvalidProgram;
    }
    remaining_items -= items;
    expression.input_types = visible;
    visible.push_back(expression.result_type);
    decoded.expressions.push_back(std::move(expression));
  }
  if (!reader.Count(count, kConditionalReturnBytes) || count > remaining_items) {
    return Status::kInvalidProgram;
  }
  decoded.conditional_returns.resize(count);
  for (auto &branch : decoded.conditional_returns) {
    if (!reader.Word(branch.condition_step) || !reader.Word(branch.value_step) || !reader.Value(branch.value)) {
      return Status::kInvalidProgram;
    }
  }
  if (!reader.Done()) {
    return Status::kInvalidProgram;
  }
  std::shared_ptr<const TilingProgramIRCompiledStage> validated;
  const auto error = BuildTilingProgramIRStage(decoded, &validated);
  if (error.status == Status::kOk) {
    *output = std::move(decoded);
  }
  return error.status;
}
}  // namespace att
