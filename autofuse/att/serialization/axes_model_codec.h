/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_SERIALIZATION_AXES_MODEL_CODEC_H_
#define ATT_SERIALIZATION_AXES_MODEL_CODEC_H_
#include <array>
#include "program/axes_model_evaluation.h"

namespace att {
TilingExpressionStatus EncodeAxesModelV1(const AxesModelEvaluationDefinition &definition, std::vector<uint8_t> *output);
TilingExpressionStatus DecodeAxesModelV1(const void *data, size_t size,
                                         std::shared_ptr<const AxesModelEvaluation> *output);
constexpr size_t kAxesModelMethodCount = 5;
using AxesModelDefinitions = std::array<AxesModelEvaluationDefinition, kAxesModelMethodCount>;
using AxesModelMethods = std::array<std::shared_ptr<const AxesModelEvaluation>, kAxesModelMethodCount>;
TilingExpressionStatus EncodeAxesModelMethodsV1(const AxesModelDefinitions &definitions, std::vector<uint8_t> *output);
TilingExpressionStatus DecodeAxesModelMethodsV1(const void *data, size_t size, AxesModelMethods *output);
}  // namespace att
#endif
