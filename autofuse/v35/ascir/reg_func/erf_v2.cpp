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
std::vector<std::unique_ptr<TmpBufDesc>> CalcErfTmpSizeV2(const AscNode &node) {
  constexpr uint32_t ERF_HALF_CALC_PROC = 8;
  constexpr uint32_t ERF_FLOAT_CALC_PROC = 3;
  constexpr uint32_t ERF_ONE_REPEAT_BYTE_SIZE = 256;
  auto node_inputs = node.inputs;
  GE_ASSERT_TRUE(node_inputs.Size() > 0, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
  uint32_t calcTmpBuf = ERF_ONE_REPEAT_BYTE_SIZE *
                        (node_inputs[0].attr.dtype == ge::DT_FLOAT16 ? ERF_HALF_CALC_PROC : ERF_FLOAT_CALC_PROC);
  GELOGD("Node %s[%s] temp buffer size: %u bytes", node.GetTypePtr(), node.GetNamePtr(), calcTmpBuf);
  Expression TmpSize = Symbol(calcTmpBuf);
  TmpBufDesc desc = {TmpSize, -1};
  std::vector<std::unique_ptr<TmpBufDesc>> tmpBufDescs;
  tmpBufDescs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmpBufDescs;
}
}  // namespace ascir
}  // namespace af
