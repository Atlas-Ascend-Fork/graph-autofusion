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
#include "graph/symbolizer/symbolic_utils.h"

namespace af {
namespace ascir {
std::vector<std::unique_ptr<af::TmpBufDesc>> CalcBucketizeTmpSizeV2(const af::AscNode &node) {
  constexpr uint32_t BUCKETIZE_MIN_TMP_SIZE = 32;
  constexpr uint32_t BUCKETIZE_B8_PROMO_SIZE = 2;
  constexpr size_t BUCKETIZE_MIN_INPUT_COUNT = 2;
  auto node_inputs = node.inputs;
  GE_ASSERT_TRUE(node_inputs.Size() > 0, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
  auto src_dtype = node_inputs[0].attr.dtype;
  auto src_type_size = GetSizeByDataType(src_dtype);
  uint32_t calc_tmp_buf = BUCKETIZE_MIN_TMP_SIZE;
  if (src_type_size == 1 && node_inputs.Size() >= BUCKETIZE_MIN_INPUT_COUNT) {
    // b8 types (int8/uint8) need scratch space for b8→b16 promotion.
    // PromoteB8 writes CeilDiv(count, vlB8) * GetVecLen() * 2 bytes per input,
    // where vlB8 = GetVecLen() / sizeof(T) = 256 elements (for b8 sizeof=1).
    // So each input requires AlignUp(count, vecLen) * sizeof(PromoT) bytes.
    // Total tmp = AlignUp(calCount, VEC_LEN) * 2 + AlignUp(boundCount, VEC_LEN) * 2
    constexpr uint32_t BUCKETIZE_VEC_LEN = 256;  // GetVecLen() / sizeof(T) for b8 types
    Expression cal_count = af::sym::kSymbolOne;
    for (int32_t i = 0; i < static_cast<int32_t>(node_inputs[0].attr.repeats.size()); i++) {
      cal_count = sym::Mul(cal_count, node_inputs[0].attr.repeats[i]);
    }
    Expression bound_count = af::sym::kSymbolOne;
    for (int32_t i = 0; i < static_cast<int32_t>(node_inputs[1].attr.repeats.size()); i++) {
      bound_count = sym::Mul(bound_count, node_inputs[1].attr.repeats[i]);
    }
    Expression total_size =
        sym::Mul(sym::Add(sym::Align(cal_count, BUCKETIZE_VEC_LEN), sym::Align(bound_count, BUCKETIZE_VEC_LEN)),
                 af::Symbol(BUCKETIZE_B8_PROMO_SIZE))
            .Simplify();
    GELOGD("Node %s[%s] temp buffer size: %s", node.GetTypePtr(), node.GetNamePtr(), total_size.Str().get());
    af::TmpBufDesc desc = {total_size, -1};
    std::vector<std::unique_ptr<af::TmpBufDesc>> bucketize_tmp_buf_descs;
    bucketize_tmp_buf_descs.emplace_back(std::make_unique<af::TmpBufDesc>(desc));
    return bucketize_tmp_buf_descs;
  }
  GELOGD("Node %s[%s] temp buffer size: %u bytes", node.GetTypePtr(), node.GetNamePtr(), calc_tmp_buf);
  Expression tmp_size = af::Symbol(calc_tmp_buf);
  af::TmpBufDesc desc = {tmp_size, -1};
  std::vector<std::unique_ptr<af::TmpBufDesc>> bucketize_tmp_buf_descs;
  bucketize_tmp_buf_descs.emplace_back(std::make_unique<af::TmpBufDesc>(desc));
  return bucketize_tmp_buf_descs;
}
}  // namespace ascir
}  // namespace af
