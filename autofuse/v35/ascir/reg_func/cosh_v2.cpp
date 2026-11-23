/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "default_reg_func_v2.h"

namespace af {
namespace ascir {
std::vector<std::unique_ptr<TmpBufDesc>> CalcCoshTmpSizeV2(const AscNode &node) {
  auto node_inputs = node.inputs;
  GELOGD("Node %s[%s] inputs[0] data type size is: %d", node.GetTypePtr(), node.GetNamePtr(),
         GetSizeByDataType(node_inputs[0].attr.dtype));
  constexpr uint32_t COSH_FLOAT_CALC_FAC = 2;
  constexpr uint32_t COSH_HALF_CALC_FAC = 6;
  constexpr uint32_t COSH_ONE_REPEAT_BYTE_SIZE = 256;
  Expression buf_nums =
      ((node_inputs[0].attr.dtype == ge::DT_FLOAT) ? Symbol(COSH_FLOAT_CALC_FAC) : Symbol(COSH_HALF_CALC_FAC));
  Expression total_size = buf_nums * Symbol(COSH_ONE_REPEAT_BYTE_SIZE);
  GELOGD("Get temp buffer size: %s bytes", total_size.Str().get());
  TmpBufDesc desc = {total_size, -1};
  std::vector<std::unique_ptr<TmpBufDesc>> tmp_buf_descs;
  tmp_buf_descs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmp_buf_descs;
}
}  // namespace ascir
}  // namespace af
