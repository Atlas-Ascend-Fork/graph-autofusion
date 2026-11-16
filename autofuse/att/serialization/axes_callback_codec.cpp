/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "axes_callback_codec.h"
#include "aftp_v1.h"
#include "tiling_record_io.h"
#include "tiling_stage_codec.h"

namespace att {
namespace {
using Status = TilingExpressionStatus;
constexpr size_t kSourceBytes = 2 * tiling_record::kWordBytes;  // kind and index
constexpr uint32_t kCallbackVersion = 1;
}  // namespace

TilingExpressionStatus EncodeAxesCallbackV1(const AxesProgramEvaluationDefinition &definition,
                                            std::vector<uint8_t> *output) {
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  std::shared_ptr<const AxesProgramEvaluation> validated;
  const auto error = BuildAxesProgramEvaluation(definition, &validated);
  if (error.status != Status::kOk) {
    return error.status;
  }
  std::vector<uint8_t> stage;
  const auto status = EncodeTilingStageV1(definition.stage, &stage);
  if (status != Status::kOk) {
    return status;
  }
  tiling_record::Writer writer;
  writer.Word(kCallbackVersion);
  writer.Word(static_cast<uint32_t>(definition.role));
  writer.Word(definition.related_tiling_count);
  writer.Word(definition.related_input_count);
  writer.Word(definition.parent_count);
  writer.Word(static_cast<uint32_t>(definition.inputs.size()));
  for (const auto &input : definition.inputs) {
    writer.Word(static_cast<uint32_t>(input.kind));
    writer.Word(input.index);
  }
  writer.bytes.insert(writer.bytes.end(), stage.begin(), stage.end());
  *output = std::move(writer.bytes);
  return Status::kOk;
}

TilingExpressionStatus DecodeAxesCallbackV1(const void *data, size_t size,
                                            std::shared_ptr<const AxesProgramEvaluation> *output) {
  if (data == nullptr || output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (size > kAftpV1MaxBytes) {
    return Status::kInvalidProgram;
  }
  tiling_record::Reader reader(data, size);
  uint32_t version, role, count;
  AxesProgramEvaluationDefinition definition;
  if (!reader.Word(version)) {
    return Status::kInvalidProgram;
  }
  if (version != kCallbackVersion) {
    return Status::kUnsupported;
  }
  if (!reader.Word(role) || role > static_cast<uint32_t>(AxesProgramEvaluationRole::kUpperBound) ||
      !reader.Word(definition.related_tiling_count) || !reader.Word(definition.related_input_count) ||
      !reader.Word(definition.parent_count) || !reader.Count(count, kSourceBytes)) {
    return Status::kInvalidProgram;
  }
  definition.role = static_cast<AxesProgramEvaluationRole>(role);
  definition.inputs.resize(count);
  for (auto &input : definition.inputs) {
    uint32_t kind;
    if (!reader.Word(kind) || kind > static_cast<uint32_t>(AxesProgramInputKind::kParentVariable) ||
        !reader.Word(input.index)) {
      return Status::kInvalidProgram;
    }
    input.kind = static_cast<AxesProgramInputKind>(kind);
  }
  const auto status = DecodeTilingStageV1(reader.Data(), reader.Size(), &definition.stage);
  if (status != Status::kOk) {
    return status;
  }
  return BuildAxesProgramEvaluation(definition, output).status;
}
}  // namespace att
