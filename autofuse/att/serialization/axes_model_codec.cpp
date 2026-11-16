/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "axes_model_codec.h"
#include "aftp_v1.h"
#include "tiling_record_io.h"
#include "tiling_stage_codec.h"

namespace att {
namespace {
using Status = TilingExpressionStatus;
constexpr size_t kSourceBytes = 2 * tiling_record::kWordBytes;  // kind and index
constexpr uint32_t kModelVersion = 1;
constexpr std::array<AxesModelFunction, kAxesModelMethodCount> kMethodOrder = {
    AxesModelFunction::kUsedCoreNum, AxesModelFunction::kRealUsedCoreNum,
    AxesModelFunction::kUbThreshold, AxesModelFunction::kUbCacheLine,
    AxesModelFunction::kPerf,
};
}  // namespace

TilingExpressionStatus EncodeAxesModelV1(const AxesModelEvaluationDefinition &definition,
                                         std::vector<uint8_t> *output) {
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  std::shared_ptr<const AxesModelEvaluation> validated;
  const auto error = BuildAxesModelEvaluation(definition, &validated);
  if (error.status != Status::kOk) {
    return error.status;
  }
  tiling_record::Writer writer;
  writer.Word(kModelVersion);
  writer.Word(static_cast<uint32_t>(definition.function));
  writer.Word(definition.input_count);
  writer.Word(definition.pure_mc_count);
  writer.Word(definition.local_buffer_count);
  writer.Word(definition.writes_output);
  writer.Word(static_cast<uint32_t>(definition.inputs.size()));
  for (const auto &input : definition.inputs) {
    writer.Word(static_cast<uint32_t>(input.kind));
    writer.Word(input.index);
  }
  if (definition.writes_output) {
    std::vector<uint8_t> stage;
    const auto status = EncodeTilingStageV1(definition.stage, &stage);
    if (status != Status::kOk) {
      return status;
    }
    writer.bytes.insert(writer.bytes.end(), stage.begin(), stage.end());
  }
  *output = std::move(writer.bytes);
  return Status::kOk;
}

TilingExpressionStatus DecodeAxesModelV1(const void *data, size_t size,
                                         std::shared_ptr<const AxesModelEvaluation> *output) {
  if (data == nullptr || output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (size > kAftpV1MaxBytes) {
    return Status::kInvalidProgram;
  }
  tiling_record::Reader reader(data, size);
  uint32_t version, function, writes, count;
  AxesModelEvaluationDefinition definition;
  if (!reader.Word(version)) {
    return Status::kInvalidProgram;
  }
  if (version != kModelVersion) {
    return Status::kUnsupported;
  }
  if (!reader.Word(function) || function > static_cast<uint32_t>(AxesModelFunction::kPerf) ||
      !reader.Word(definition.input_count) || !reader.Word(definition.pure_mc_count) ||
      !reader.Word(definition.local_buffer_count) || !reader.Word(writes) || writes > 1 ||
      !reader.Count(count, kSourceBytes)) {
    return Status::kInvalidProgram;
  }
  definition.function = static_cast<AxesModelFunction>(function);
  definition.writes_output = writes != 0;
  definition.inputs.resize(count);
  for (auto &input : definition.inputs) {
    uint32_t kind;
    if (!reader.Word(kind) || kind > static_cast<uint32_t>(AxesModelInputKind::kCacheLocalValue) ||
        !reader.Word(input.index)) {
      return Status::kInvalidProgram;
    }
    input.kind = static_cast<AxesModelInputKind>(kind);
  }
  if (definition.writes_output) {
    const auto status = DecodeTilingStageV1(reader.Data(), reader.Size(), &definition.stage);
    if (status != Status::kOk) {
      return status;
    }
  } else if (!reader.Done()) {
    return Status::kInvalidProgram;
  }
  return BuildAxesModelEvaluation(definition, output).status;
}
TilingExpressionStatus EncodeAxesModelMethodsV1(const AxesModelDefinitions &definitions, std::vector<uint8_t> *output) {
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  tiling_record::Writer writer;
  writer.Word(kModelVersion);
  const auto &first = definitions.front();
  if (first.writes_output != definitions[1].writes_output) {
    return Status::kInvalidProgram;
  }
  for (size_t i = 0; i < definitions.size(); ++i) {
    const auto &method = definitions[i];
    if (method.function != kMethodOrder[i] || method.input_count != first.input_count ||
        method.pure_mc_count != first.pure_mc_count || method.local_buffer_count != first.local_buffer_count) {
      return Status::kInvalidProgram;
    }
    std::vector<uint8_t> record;
    const auto status = EncodeAxesModelV1(method, &record);
    if (status != Status::kOk) {
      return status;
    }
    writer.Word(static_cast<uint32_t>(record.size()));
    writer.bytes.insert(writer.bytes.end(), record.begin(), record.end());
  }
  *output = std::move(writer.bytes);
  return Status::kOk;
}

TilingExpressionStatus DecodeAxesModelMethodsV1(const void *data, size_t size, AxesModelMethods *output) {
  if (data == nullptr || output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (size > kAftpV1MaxBytes) {
    return Status::kInvalidProgram;
  }
  tiling_record::Reader reader(data, size);
  uint32_t version;
  if (!reader.Word(version)) {
    return Status::kInvalidProgram;
  }
  if (version != kModelVersion) {
    return Status::kUnsupported;
  }
  AxesModelMethods methods;
  for (size_t i = 0; i < methods.size(); ++i) {
    uint32_t length;
    if (!reader.Word(length) || length > reader.Size()) {
      return Status::kInvalidProgram;
    }
    const auto status = DecodeAxesModelV1(reader.Data(), length, &methods[i]);
    if (status != Status::kOk) {
      return status;
    }
    reader.Skip(length);
    const auto &method = *methods[i];
    const auto &first = *methods.front();
    if (method.Function() != kMethodOrder[i] || method.InputCount() != first.InputCount() ||
        method.PureMcCount() != first.PureMcCount() || method.LocalBufferCount() != first.LocalBufferCount()) {
      return Status::kInvalidProgram;
    }
  }
  if (!reader.Done() || methods[0]->WritesOutput() != methods[1]->WritesOutput()) {
    return Status::kInvalidProgram;
  }
  *output = std::move(methods);
  return Status::kOk;
}
}  // namespace att
