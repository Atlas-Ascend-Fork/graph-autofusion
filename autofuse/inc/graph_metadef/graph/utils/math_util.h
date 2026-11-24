/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_GRAPH_UTILS_MATH_UTIL_H_
#define METADEF_CXX_INC_GRAPH_UTILS_MATH_UTIL_H_

#include <securec.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <iostream>

#include "ge_common_af/ge_api_error_codes_af.h"
#include "graph/def_types.h"
#include "utils/extern_math_util.h"
#include "common/ge_common/debug/log.h"
#include "graph_metadef/common/ge_common/util.h"

namespace af {
constexpr uint32_t kDiv16RightShiftBits = 4U;
constexpr uint32_t kDiv32RightShiftBits = 5U;
/**
 * @ingroup domi_calibration
 * @brief  Initializes an input array to a specified value
 * @param [in]  n        array initialization length
 * @param [in]  alpha    initialization value
 * @param [out]  output  array to be initialized
 * @return      Status
 */
template <typename Dtype>
Status NnSet(const int32_t n, const Dtype alpha, Dtype *const output) {
  GE_CHECK_NOTNULL(output);

  if (std::equal_to<Dtype>()(alpha, static_cast<Dtype>(0))) {
    if ((sizeof(Dtype) * static_cast<size_t>(n)) < SECUREC_MEM_MAX_LEN) {
      const errno_t err =
          memset_s(output, sizeof(Dtype) * static_cast<size_t>(n), 0, sizeof(Dtype) * static_cast<size_t>(n));
      GE_CHK_BOOL_RET_STATUS(err == EOK, ge::PARAM_INVALID, "memset_s err, dst:%p, size:%zu, err:%d", output,
                             sizeof(Dtype) * static_cast<size_t>(n), err);
    } else {
      const uint64_t size = static_cast<uint64_t>(sizeof(Dtype) * static_cast<size_t>(n));
      const uint64_t step = SECUREC_MEM_MAX_LEN - (SECUREC_MEM_MAX_LEN % sizeof(Dtype));
      const uint64_t times = size / step;
      const uint64_t remainder = size % step;
      uint64_t i = 0U;
      while (i < times) {
        const errno_t err = memset_s(ValueToPtr(PtrToValue(output) + (i * (step / sizeof(Dtype)))), step, 0, step);
        GE_CHK_BOOL_RET_STATUS(err == EOK, ge::PARAM_INVALID, "memset_s err, dst:%p, step:%" PRIu64 ", err:%d",
                               ValueToPtr(PtrToValue(output) + (i * (step / sizeof(Dtype)))), step, err);
        i++;
      }
      if (remainder != 0U) {
        const errno_t err =
            memset_s(ValueToPtr(PtrToValue(output) + (i * (step / sizeof(Dtype)))), remainder, 0, remainder);
        GE_CHK_BOOL_RET_STATUS(err == EOK, ge::PARAM_INVALID, "memset_s err, dst:%p, remainder:%" PRIu64 ", err:%d",
                               ValueToPtr(PtrToValue(output) + (i * (step / sizeof(Dtype)))), remainder, err);
      }
    }
  }

  for (int32_t i = 0; i < n; ++i) {
    output[i] = alpha;
  }
  return af::SUCCESS;
}

template <typename T, typename TR>
bool RoundUpOverflow(T value, T multiple_of, TR &ret) {
  if (multiple_of == 0) {
    ret = 0;
    return true;
  }
  auto remainder = value % multiple_of;
  if (remainder == 0) {
    if (!ge::IntegerChecker<TR>::Compat(value)) {
      return true;
    }
    ret = static_cast<TR>(value);
    return false;
  }
  return ge::AddOverflow(value - remainder, multiple_of, ret);
}
template <typename T>
T CeilDiv16(const T n) {
  if (n & 0xF) {
    return (n >> kDiv16RightShiftBits) + 1;
  } else {
    return n >> kDiv16RightShiftBits;
  }
}

template <typename T>
T CeilDiv32(const T n) {
  if (n & 0x1F) {
    return (n >> kDiv32RightShiftBits) + 1;
  } else {
    return n >> kDiv32RightShiftBits;
  }
}

template <typename T>
T CeilDiv(const T n1, const T n2) {
  if (n1 == 0) {
    return 0;
  }
  return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

template <typename T>
T FloorDiv(const T u_value, const T d_value) {
  if (d_value == 0) {
    return u_value;
  }
  return u_value / d_value;
}

inline uint64_t RoundUp(const uint64_t origin_value, const uint64_t multiple_of) {
  uint64_t ret = 0U;
  if (RoundUpOverflow(origin_value, multiple_of, ret)) {
    return 0U;
  }
  return ret;
}

inline Status GeMemcpy(uint8_t *dst_ptr, size_t dst_size, const uint8_t *src_ptr, const size_t src_size) {
  GE_CHK_BOOL_RET_STATUS((dst_size >= src_size), ge::PARAM_INVALID, "memcpy_s verify fail, src size %zu, dst size %zu",
                         src_size, dst_size);
  size_t offset = 0U;
  size_t remain_size = src_size;
  do {
    size_t copy_size = (remain_size > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : remain_size;
    const auto err = memcpy_s((dst_ptr + offset), copy_size, (src_ptr + offset), copy_size);
    GE_CHK_BOOL_RET_STATUS(err == EOK, ge::PARAM_INVALID,
                           "memcpy_s err, src ptr %p size %zu, dst ptr %p size %zu, offset %zu, err %d", src_ptr,
                           copy_size, dst_ptr, (dst_size - offset), offset, err);
    offset += copy_size;
    remain_size -= copy_size;
  } while (remain_size > 0U);
  return af::SUCCESS;
}
}  // namespace af

#define REQUIRE_COMPAT(T, v)                                                                                 \
  do {                                                                                                       \
    if (!IntegerChecker<T>::Compat((v))) {                                                                   \
      std::stringstream ss;                                                                                  \
      ss << #v << " value " << (v) << " out of " << #T << " range [" << std::numeric_limits<T>::min() << "," \
         << std::numeric_limits<T>::max() << "]";                                                            \
      GELOGE(af::FAILED, "%s", ss.str().c_str());                                                            \
      return af::FAILED;                                                                                     \
    }                                                                                                        \
  } while (false)

#define REQUIRE_COMPAT_FOR_CHAR(T, v)                                                    \
  do {                                                                                   \
    if (!IntegerChecker<T>::Compat((v))) {                                               \
      std::stringstream ss;                                                              \
      ss << #v << " value " << static_cast<int32_t>(v) << " out of " << #T << " range [" \
         << static_cast<int32_t>(std::numeric_limits<T>::min()) << ","                   \
         << static_cast<int32_t>(std::numeric_limits<T>::max()) << "]";                  \
      GELOGE(af::FAILED, "%s", ss.str().c_str());                                        \
      return af::FAILED;                                                                 \
    }                                                                                    \
  } while (false)

#define REQUIRE_COMPAT_INT8(v) REQUIRE_COMPAT_FOR_CHAR(int8_t, (v))
#define REQUIRE_COMPAT_UINT8(v) REQUIRE_COMPAT_FOR_CHAR(uint8_t, (v))
#define REQUIRE_COMPAT_INT16(v) REQUIRE_COMPAT(int16_t, (v))
#define REQUIRE_COMPAT_UINT16(v) REQUIRE_COMPAT(uint16_t, (v))
#define REQUIRE_COMPAT_INT32(v) REQUIRE_COMPAT(int32_t, (v))
#define REQUIRE_COMPAT_UINT32(v) REQUIRE_COMPAT(uint32_t, (v))
#define REQUIRE_COMPAT_INT64(v) REQUIRE_COMPAT(int64_t, (v))
#define REQUIRE_COMPAT_UINT64(v) REQUIRE_COMPAT(uint64_t, (v))
#define REQUIRE_COMPAT_SIZE_T(v) REQUIRE_COMPAT(size_t, (v))

#endif  // METADEF_CXX_INC_GRAPH_UTILS_MATH_UTIL_H_
