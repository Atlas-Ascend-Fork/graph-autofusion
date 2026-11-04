/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
  const std::string kAscendcBitwise_andStr = {
#include "bitwise_and_str.h"

  };
  const std::string kAscendcDuplicateStr = {
#include "duplicate_str.h"

  };
  const std::string kAscendcBroadcastStr = {
#include "broadcast_str.h"

  };
  const std::string kAscendcCastStr = {
#include "cast_str.h"

  };
  const std::string kAscendcClipbyvalueStr = {
#include "clipbyvalue_str.h"

  };
  const std::string kAscendcCompareStr = {
#include "compare_str.h"

  };
  const std::string kAscendcCompareV2Str = {
#include "compare_v2_str.h"

  };
  const std::string kAscendcConcatStr = {
#include "concat_str.h"

  };
  const std::string kAscendcDatacopyStr = {
#include "datacopy_str.h"

  };
  const std::string kAscendcIsfiniteStr = {
#include "isfinite_str.h"

  };
  const std::string kAscendcIsnanStr = {
#include "isnan_str.h"

  };
  const std::string kAscendcIsInfStr = {
#include "is_inf_str.h"

  };
  const std::string kAscendcLogical_notStr = {
#include "logical_not_str.h"

  };
  const std::string kAscendcLogicalStr = {
#include "logical_str.h"

  };
  const std::string kAscendcPowStr = {
#include "pow_str.h"

  };
  const std::string kAscendcAxpyStr = {
#include "axpy_str.h"

  };
  const std::string kAscendcReciprocalStr = {
#include "reciprocal_str.h"

  };
  const std::string kAscendcReduce_initStr = {
#include "reduce_init_str.h"

  };
  const std::string kAscendcRemovePadStr = {
#include "removepad_str.h"

  };
  const std::string kAscendcReduce_prodStr = {
#include "reduce_prod_str.h"

  };
  const std::string kAscendcReduceStr = {
#include "reduce_str.h"

  };
  const std::string kAscendcRsqrtStr = {
#include "rsqrt_str.h"

  };
  const std::string kAscendcScalar_divStr = {
#include "scalar_div_str.h"

  };
  const std::string kAscendcFloorDivStr = {
#include "floor_div_str.h"

  };

  const std::string kAscendcSigmoidStr = {
#include "sigmoid_str.h"

  };
  const std::string kAscendcSignStr = {
#include "sign_str.h"

  };
  const std::string kAscendcWhereStr = {
#include "where_str.h"

  };

  const std::string kAscendcGatherStr = {
#include "gather_str.h"

  };
  const std::string kAscendcScalarSubStr = {
#include "scalar_sub_str.h"

  };
  const std::string kAscendcTranposeBaseTypeStr = {
#include "transpose_base_type_str.h"

  };
  const std::string kAscendcTranposeStr = {
#include "transpose_str.h"

  };
  const std::string kAscendcScalarAddStr = {
#include "scalar_add_str.h"
  };
  const std::string kAscendcScalarMulStr = {
#include "scalar_mul_str.h"
  };
  const std::string kAscendcArgmax_with_valueStr = {
#include "argmax_with_value_str.h"
  };
  const std::string kAscendcArgmaxStr = {
#include "argmax_str.h"
  };
  const std::string kAscendcScalarMaximumStr = {
#include "scalar_maximum_str.h"
  };
  const std::string kAscendcScalarMinimumStr = {
#include "scalar_minimum_str.h"
  };
  const std::string kAscendcAbsStr = {
#include "abs_str.h"
  };
  const std::string kAscendcTrueDivStr = {
#include "true_div_str.h"
  };
  const std::string kAscendcRemainderStr = {
#include "remainder_str.h"
  };
  const std::string kAscendcNegStr = {
#include "neg_str.h"
  };
  // utils.h 不声明任何 API，它是其余扩展头的前导（KernelUtils / MAX_REPEAT_TIME）。
  // legacy 走的是 IncludeAndDefines 那条无条件内联的路，不查这张表；MLIR 路径按
  // op 声明的前导名来查，所以这里也得有一份。
  const std::string kAscendcUtilsStr = {
#include "utils_str.h"
  };
  std::unordered_map<std::string, std::string> api_to_file{{"bitwise_and.h", kAscendcBitwise_andStr},
                                                           {"duplicate.h", kAscendcDuplicateStr},
                                                           {"broadcast.h", kAscendcBroadcastStr},
                                                           {"cast.h", kAscendcCastStr},
                                                           {"clipbyvalue.h", kAscendcClipbyvalueStr},
                                                           {"compare.h", kAscendcCompareStr},
                                                           {"argmax_with_value.h", kAscendcArgmax_with_valueStr},
                                                           {"argmax.h", kAscendcArgmaxStr},
                                                           {"compare_v2.h", kAscendcCompareV2Str},
                                                           {"concat.h", kAscendcConcatStr},
                                                           {"datacopy.h", kAscendcDatacopyStr},
                                                           {"isfinite.h", kAscendcIsfiniteStr},
                                                           {"isnan.h", kAscendcIsnanStr},
                                                           {"is_inf.h", kAscendcIsInfStr},
                                                           {"logical_not.h", kAscendcLogical_notStr},
                                                           {"logical.h", kAscendcLogicalStr},
                                                           {"pow.h", kAscendcPowStr},
                                                           {"axpy.h", kAscendcAxpyStr},
                                                           {"reciprocal.h", kAscendcReciprocalStr},
                                                           {"reduce_init.h", kAscendcReduce_initStr},
                                                           {"removepad.h", kAscendcRemovePadStr},
                                                           {"reduce_prod.h", kAscendcReduce_prodStr},
                                                           {"reduce.h", kAscendcReduceStr},
                                                           {"rsqrt.h", kAscendcRsqrtStr},
                                                           {"scalar_div.h", kAscendcScalar_divStr},
                                                           {"floor_div.h", kAscendcFloorDivStr},
                                                           {"sigmoid.h", kAscendcSigmoidStr},
                                                           {"sign.h", kAscendcSignStr},
                                                           {"where.h", kAscendcWhereStr},
                                                           {"gather.h", kAscendcGatherStr},
                                                           {"scalar_sub.h", kAscendcScalarSubStr},
                                                           {"scalar_add.h", kAscendcScalarAddStr},
                                                           {"scalar_mul.h", kAscendcScalarMulStr},
                                                           {"scalar_maximum.h", kAscendcScalarMaximumStr},
                                                           {"scalar_minimum.h", kAscendcScalarMinimumStr},
                                                           {"transpose_base_type.h", kAscendcTranposeBaseTypeStr},
                                                           {"transpose.h", kAscendcTranposeStr},
                                                           {"abs.h", kAscendcAbsStr},
                                                           {"true_div.h", kAscendcTrueDivStr},
                                                           {"remainder.h", kAscendcRemainderStr},
                                                           {"neg.h", kAscendcNegStr},
                                                           {"utils.h", kAscendcUtilsStr}};

  AscendCApiRegistry::GetInstance().RegisterApi(api_to_file);
}

Register __attribute__((unused)) api_register;
}  // namespace

AscendCApiRegistry &AscendCApiRegistry::GetInstance() {
  static AscendCApiRegistry instance;
  return instance;
}

const std::string &AscendCApiRegistry::GetFileContent(const std::string &api_name) {
  static const std::string kEmpty;
  auto it = api_to_file_content_.find(api_name);
  return it != api_to_file_content_.end() ? it->second : kEmpty;
}

void AscendCApiRegistry::RegisterApi(const std::unordered_map<std::string, std::string> &api_to_file_content) {
  api_to_file_content_.insert(api_to_file_content.cbegin(), api_to_file_content.cend());
}
}  // namespace codegen
