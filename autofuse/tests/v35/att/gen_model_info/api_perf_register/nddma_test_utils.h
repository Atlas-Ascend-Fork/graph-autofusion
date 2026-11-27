/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_V35_ATT_GEN_MODEL_INFO_API_PERF_REGISTER_NDDMA_TEST_UTILS_H_
#define AUTOFUSE_TESTS_V35_ATT_GEN_MODEL_INFO_API_PERF_REGISTER_NDDMA_TEST_UTILS_H_

#include <string>

#include "gtest/gtest.h"

#include "api_perf_register/api_perf.h"

namespace att {

inline TensorShapeInfo Make1DNddmaShape(const std::string &dtype, int32_t dtype_size, const Expr &dim,
                                        const Expr &input_stride, const Expr &output_stride) {
  TensorShapeInfo shape;
  shape.data_type = dtype;
  shape.data_type_size = dtype_size;
  shape.dims = {dim};
  shape.repeats = {dim};
  shape.gm_strides = {input_stride};
  shape.strides = {output_stride};
  return shape;
}

inline double GetNddmaConstCycles(Expr cycles) {
  cycles.Simplify();
  int64_t integer = 0;
  if (cycles.GetConstValue(integer)) {
    return static_cast<double>(integer);
  }
  double value = 0.0;
  EXPECT_TRUE(cycles.GetConstValue(value));
  return value;
}

}  // namespace att

#endif  // AUTOFUSE_TESTS_V35_ATT_GEN_MODEL_INFO_API_PERF_REGISTER_NDDMA_TEST_UTILS_H_
