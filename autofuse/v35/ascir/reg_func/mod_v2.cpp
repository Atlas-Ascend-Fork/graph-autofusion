/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
std::vector<std::unique_ptr<TmpBufDesc>> CalcModTmpSizeV2(const AscNode &node) {
  constexpr uint32_t MOD_FLOAT_TMP_SIZE = 256;
  constexpr uint32_t MOD_FLOAT16_TMP_SIZE = 2048;
  std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_FLOAT, DT_FLOAT},  {DT_FLOAT16, DT_FLOAT16},
                                                               {DT_BF16, DT_FLOAT},   {DT_INT16, DT_FLOAT},
                                                               {DT_INT8, DT_FLOAT16}, {DT_UINT8, DT_FLOAT16}};
  auto node_inputs = node.inputs;
  GE_ASSERT_TRUE(node_inputs.Size() > 0, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
  uint32_t calcTmpBuf =
      ((dtype_conversion_map[node_inputs[0].attr.dtype] == ge::DT_FLOAT16) ? MOD_FLOAT16_TMP_SIZE : MOD_FLOAT_TMP_SIZE);
  GELOGD("Node %s[%s] temp buffer size: %u bytes", node.GetTypePtr(), node.GetNamePtr(), calcTmpBuf);
  Expression TmpSize = Symbol(calcTmpBuf);
  TmpBufDesc desc = {TmpSize, -1};
  std::vector<std::unique_ptr<TmpBufDesc>> tmpBufDescs;
  tmpBufDescs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmpBufDescs;
}
}  // namespace ascir
}  // namespace af
