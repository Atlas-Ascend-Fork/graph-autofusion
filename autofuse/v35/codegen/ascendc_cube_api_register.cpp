/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_api_registry.h"

namespace codegen {
namespace {
class Register {
 public:
  Register();
};

Register::Register() {
  const std::string kAscendcMatmulStr = {
#include "matmul_str.h"

  };
  const std::string kAscendcmat_mul_include_headers = {
#include "matmul_include_headers_str.h"

  };
  const std::string kAscendcmat_mul_tiling_key = {
#include "mat_mul_tiling_key_str.h"

  };

  const std::string kAscendcbatch_mat_mul_v3_tiling_key = {
#include "batch_mat_mul_v3_tiling_key_str.h"

  };
  const std::string kAscendcmat_mul_pingpong_basic_cmct = {
#include "mat_mul_pingpong_basic_cmct_str.h"

  };
  const std::string kAscendcbatch_matmul = {
#include "batch_matmul_str.h"

  };
  const std::string kAscendcbatch_matmul_include_headers = {
#include "batch_matmul_include_headers_str.h"

  };
  const std::string kAscendcconv2d_include_headers = {
#include "conv2d_include_headers_str.h"

  };
  const std::string kAscendcconv2d_v2_tilingkey = {
#include "conv2d_v2_tilingkey_cv_str.h"

  };
  const std::string kAscendcconv2d = {
#include "conv2d_str.h"

  };
  const std::string kAscendcConv_pingpong_basic_atcos = {
#include "conv_pingpong_basic_atcos_str.h"

  };
  const std::string kAscendcMatmul_dynamic = {
#include "matmul_dynamic_str.h"

  };
  const std::string kAscendcmat_mul_tiling_key_dynamic = {
#include "mat_mul_tiling_key_dynamic_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_tiling_key_dynamic = {
#include "batch_mat_mul_v3_tiling_key_dynamic_str.h"

  };
  const std::string kAscendcmat_mul_pingpong_basic_cmct_dynamic = {
#include "mat_mul_pingpong_basic_cmct_dynamic_str.h"

  };
  const std::string kAscendcbatch_matmul_dynamic = {
#include "batch_matmul_dynamic_str.h"

  };

  std::unordered_map<std::string, std::string> api_to_file{
      {"matmul.h", kAscendcMatmulStr},
      {"matmul_include_headers.h", kAscendcmat_mul_include_headers},
      {"mat_mul_tiling_key.h", kAscendcmat_mul_tiling_key},
      {"batch_mat_mul_v3_tiling_key.h", kAscendcbatch_mat_mul_v3_tiling_key},
      {"mat_mul_pingpong_basic_cmct.h", kAscendcmat_mul_pingpong_basic_cmct},
      {"batch_matmul.h", kAscendcbatch_matmul},
      {"batch_matmul_include_headers.h", kAscendcbatch_matmul_include_headers},
      {"conv2d_include_headers.h", kAscendcconv2d_include_headers},
      {"conv2d_v2_tilingkey_cv.h", kAscendcconv2d_v2_tilingkey},
      {"conv_pingpong_basic_atcos.h", kAscendcConv_pingpong_basic_atcos},
      {"conv2d.h", kAscendcconv2d},
      {"matmul_dynamic.h", kAscendcMatmul_dynamic},
      {"mat_mul_tiling_key_dynamic.h", kAscendcmat_mul_tiling_key_dynamic},
      {"batch_mat_mul_v3_tiling_key_dynamic.h", kAscendcbatch_mat_mul_v3_tiling_key_dynamic},
      {"mat_mul_pingpong_basic_cmct_dynamic.h", kAscendcmat_mul_pingpong_basic_cmct_dynamic},
      {"batch_matmul_dynamic.h", kAscendcbatch_matmul_dynamic}};

  AscendCApiRegistry::GetInstance().RegisterApi(api_to_file);
}

Register __attribute__((unused)) cube_api_register;
}  // namespace
}  // namespace codegen
