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
std::vector<std::unique_ptr<TmpBufDesc>> CalcIgammaTmpSizeV2(const AscNode &node) {
  constexpr uint32_t kIgammaOneRepeatByteSize = 256U;
  constexpr uint32_t kFloatCalcProc = 8U;
  auto node_inputs = node.inputs;
  GE_ASSERT_TRUE(node_inputs.Size() > 0, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
  uint32_t calc_tmp_buf = kIgammaOneRepeatByteSize * kFloatCalcProc;
  GELOGD("Node %s[%s] temp buffer size: %u bytes", node.GetTypePtr(), node.GetNamePtr(), calc_tmp_buf);
  Expression tmp_size = Symbol(calc_tmp_buf);
  TmpBufDesc desc = {tmp_size, -1};
  std::vector<std::unique_ptr<TmpBufDesc>> tmp_buf_descs;
  tmp_buf_descs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmp_buf_descs;
}
}  // namespace ascir
}  // namespace af
