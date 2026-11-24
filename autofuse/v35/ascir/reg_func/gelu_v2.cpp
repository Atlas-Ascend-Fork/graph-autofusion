/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "default_reg_func_v2.h"

namespace af {
namespace ascir {
std::vector<std::unique_ptr<TmpBufDesc>> CalcGeluTmpSizeV2(const AscNode &node) {
  constexpr uint32_t GELU_CALC_PROC = 3;
  constexpr uint32_t GELU_ONE_REPEAT_BYTE_SIZE = 256;
  uint32_t calcTmpBuf = GELU_CALC_PROC * GELU_ONE_REPEAT_BYTE_SIZE;
  GELOGD("Node %s[%s] temp buffer size: %u bytes", node.GetTypePtr(), node.GetNamePtr(), calcTmpBuf);
  Expression TmpSize = Symbol(calcTmpBuf);
  TmpBufDesc desc = {TmpSize, -1};
  std::vector<std::unique_ptr<TmpBufDesc>> tmpBufDescs;
  tmpBufDescs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmpBufDescs;
}
}  // namespace ascir
}  // namespace af
