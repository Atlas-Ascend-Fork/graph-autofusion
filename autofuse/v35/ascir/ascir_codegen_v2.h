/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCIR_CODEGEN_V2_H
#define ASCIR_CODEGEN_V2_H

#include "ascir_registry.h"

namespace af {
namespace ascir {
class AscIrCodegenV2 : public AscIrCodegen {
 public:
  // 返回api call类的名称
  virtual std::string GetMicroApiCallName() const {
    return "";
  }

  // 返回api的名称
  virtual std::string GetMicroApiName() const {
    return "";
  }

  // 微指令api包含的微指令的条数, 如果支持vector function, 该接口返回值才有意义
  virtual uint32_t GetMicroInstNum() const {
    return 1U;
  }

  virtual std::vector<std::vector<ge::DataType>> GetOutputConversionPath(const AscNode &node) {
    (void)node;
    return {};
  }

  virtual bool IsSimtScalarSupported(const AscNode &node) const {
    (void)node;
    return false;
  }

  virtual ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                 std::string &expr) const {
    (void)node;
    (void)inputs;
    (void)expr;
    return af::FAILED;
  }
};
}  // namespace ascir
}  // namespace af
#endif  // ASCIR_CODEGEN_V2_H
