/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSOR_ASSIGN_H
#define TENSOR_ASSIGN_H

#include <cinttypes>
#include <vector>
#include "graph/ge_tensor.h"
#include "graph/def_types.h"
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "register/register_error_codes.h"
#include "utils/extern_math_util.h"
#include "proto/tensorflow/tensor.pb.h"

namespace domi {
using GeTensorPtr = std::shared_ptr<ge::GeTensor>;
using Status = uint32_t;
constexpr int64_t kComplexWidth = 2;

class TensorAssign {
 public:
  static Status SetGeTensor(const domi::tensorflow::TensorProto &tensor, GeTensorPtr &weight);

  static Status SetGeTensorDataType(const int64_t data_type, GeTensorPtr &weight);

  static ge::DataType ConvertTensorflowDataType(const uint32_t tf_data_type);

 private:
  static bool CheckBoolVal(const tensorflow::DataType data_type);

  static bool CheckHalfVal(const tensorflow::DataType data_type);

  static bool CheckFloatVal(const tensorflow::DataType data_type);

  static bool CheckDoubleVal(const tensorflow::DataType data_type);

  static bool CheckComplex32Val(const tensorflow::DataType data_type);

  static bool CheckComplex64Val(const tensorflow::DataType data_type);

  static bool CheckComplex128Val(const tensorflow::DataType data_type);

  static bool CheckStringVal(const tensorflow::DataType data_type);

  static bool CheckByte(const tensorflow::DataType data_type);

  static bool CheckDoubleByte(const tensorflow::DataType data_type);

  static bool CheckSignedFourByte(const tensorflow::DataType data_type);

  static bool CheckUnsignedFourByte(const tensorflow::DataType data_type);

  static bool CheckSignedEightByte(const tensorflow::DataType data_type);

  static bool CheckUnsignedEightByte(const tensorflow::DataType data_type);

  static Status GetDoubleByteVal(const int64_t val_size,
                                 const google::protobuf::RepeatedField<google::protobuf::int32> &val_vector,
                                 const int64_t count, GeTensorPtr &weight);
  static Status GetByteVal(const int64_t val_size,
                           const google::protobuf::RepeatedField<google::protobuf::int32> &val_vector,
                           const int64_t count, GeTensorPtr &weight);

  static Status GetStringVal(const int64_t val_size, const google::protobuf::RepeatedPtrField<std::string> &val_vector,
                             const int64_t count, GeTensorPtr &weight);

  static void SetGeTensorWeightData(const domi::tensorflow::TensorProto &tensor, const int64_t val_size,
                                    const int64_t count, GeTensorPtr &weight);

  static void SetWeightData(const tensorflow::DataType data_type, const int64_t count,
                            const std::string &tensor_content, GeTensorPtr &weight);

  template <typename T>
  static Status GetVal(const int64_t val_size, const google::protobuf::RepeatedField<T> &val_vector,
                       const int64_t count, GeTensorPtr &weight, const bool is_complex = false) {
    // val_size must be even, and complex value should be an integer multiple of 2
    if (is_complex && ((val_size % kComplexWidth) != 0)) {
      GELOGE(FAILED, "complex value size:%" PRId64 " should be an integer multiple of %" PRId64 ".", val_size,
             kComplexWidth);
      return FAILED;
    }
    const std::unique_ptr<T[]> addr(new (std::nothrow) T[static_cast<size_t>(count)]());  // Zero init default value
    GE_CHECK_NOTNULL(addr);
    if (val_size == 0) {
      (void)weight->SetData(ge::PtrToPtr<T, uint8_t>(addr.get()), static_cast<size_t>(count) * sizeof(T));
      return SUCCESS;
    }
    // Complex numbers are made up of real and imaginary numbers
    const bool zerosLike = ((count != val_size) && ((val_size == 1) || (is_complex && (val_size == 2))));
    if ((!zerosLike) && (val_size <= count)) {
      for (size_t i = 0UL; i < static_cast<size_t>(val_size); i++) {
        addr[i] = val_vector.Get(static_cast<int32_t>(i));
      }
      const int64_t value_r = val_size - 1;
      GE_ASSERT_EQ(ge::IntegerChecker<int32_t>::Compat(value_r), true);
      if (is_complex) {
        // val_vector format is real value, complex value..., here is getting the corresponding value.
        // real value and complex value are stored spaced apart, so use 2 and 1 to store in the correct addr.
        const int64_t value_l = val_size - kComplexWidth;
        GE_ASSERT_EQ(ge::IntegerChecker<int32_t>::Compat(value_l), true);
        for (int64_t i = val_size; i < count; i += kComplexWidth) {
          addr[static_cast<size_t>(i)] = val_vector.Get(static_cast<int32_t>(value_l));
          addr[static_cast<size_t>(i) + 1UL] = val_vector.Get(static_cast<int32_t>(value_r));
        }
      } else {
        for (int64_t i = val_size; i < count; i++) {
          addr[static_cast<size_t>(i)] = val_vector.Get(static_cast<int32_t>(value_r));
        }
      }
    } else {
      if (is_complex) {
        for (int64_t i = 0; i < count; i += kComplexWidth) {
          addr[static_cast<size_t>(i)] = val_vector.Get(0);
          addr[static_cast<size_t>(i) + 1UL] = val_vector.Get(1);
        }
      } else {
        for (int64_t i = 0; i < count; i++) {
          addr[static_cast<size_t>(i)] = val_vector.Get(0);
        }
      }
    }
    (void)weight->SetData(ge::PtrToPtr<T, uint8_t>(addr.get()), static_cast<size_t>(count) * sizeof(T));
    return SUCCESS;
  }
};
}  // namespace domi
#endif  // TENSOR_ASSIGN_H
