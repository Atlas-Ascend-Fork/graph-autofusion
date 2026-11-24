/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "default_reg_func_v2.h"

namespace af {
namespace ascir {
std::vector<std::unique_ptr<TmpBufDesc>> CalcCeilTmpSizeV2(const AscNode &node) {
  auto node_inputs = node.inputs;
  GELOGD("Node %s[%s] inputs[0] data type size is: %d", node.GetTypePtr(), node.GetNamePtr(),
         GetSizeByDataType(node_inputs[0].attr.dtype));
  constexpr uint32_t CEIL_FLOAT_CALC_FAC = 1;
  constexpr uint32_t CEIL_HALF_CALC_FAC = 2;
  constexpr uint32_t CEIL_ONE_REPEAT_BYTE_SIZE = 256;
  Expression buf_nums =
      ((node_inputs[0].attr.dtype == ge::DT_FLOAT) ? Symbol(CEIL_FLOAT_CALC_FAC) : Symbol(CEIL_HALF_CALC_FAC));
  Expression total_size = buf_nums * Symbol(CEIL_ONE_REPEAT_BYTE_SIZE);
  GELOGD("Get temp buffer size: %s bytes", total_size.Str().get());
  TmpBufDesc desc = {total_size, -1};
  std::vector<std::unique_ptr<TmpBufDesc>> tmp_buf_descs;
  tmp_buf_descs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmp_buf_descs;
}
}  // namespace ascir
}  // namespace af
