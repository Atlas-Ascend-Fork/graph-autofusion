/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AF_COMMON_GRAPH_DEBUG_GE_UTIL_H_
#define AF_COMMON_GRAPH_DEBUG_GE_UTIL_H_

#include "graph_metadef/common/ge_common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_error_codes_af.h"
namespace af {
template <typename T, typename... Args>
static inline std::shared_ptr<T> ComGraphMakeShared(Args &&...args) {
  using T_nc = typename std::remove_const<T>::type;
  std::shared_ptr<T> ret = nullptr;
  try {
    ret = std::make_shared<T_nc>(std::forward<Args>(args)...);
  } catch (const std::bad_alloc &) {
    ret = nullptr;
#ifdef AUTOFUSE_USE_GE_METADEF
    GELOGE(af::FAILED, "Make shared failed, type size:%zu.", sizeof(T_nc));
#else
    GELOGE(af::FAILED, "Make shared failed, type size:%zu.", sizeof(T_nc));
#endif
  }
  return ret;
}
template <typename T, typename... Args>
static inline std::shared_ptr<T> ComGraphMakeSharedAndThrow(Args &&...args) {
  using T_nc = typename std::remove_const<T>::type;
  return std::make_shared<T_nc>(std::forward<Args>(args)...);
}
template <typename T>
struct ComGraphMakeUniq {
  using unique_object = std::unique_ptr<T>;
};

template <typename T>
struct ComGraphMakeUniq<T[]> {
  using unique_array = std::unique_ptr<T[]>;
};

template <typename T, size_t B>
struct ComGraphMakeUniq<T[B]> {
  struct invalid_type {};
};

template <typename T, typename... Args>
static inline typename ComGraphMakeUniq<T>::unique_object ComGraphMakeUnique(Args &&...args) {
  using T_nc = typename std::remove_const<T>::type;
  return std::unique_ptr<T>(new (std::nothrow) T_nc(std::forward<Args>(args)...));
}

template <typename T>
static inline typename ComGraphMakeUniq<T>::unique_array ComGraphMakeUnique(const size_t num) {
  return std::unique_ptr<T>(new (std::nothrow) typename std::remove_extent<T>::type[num]());
}

template <typename T, typename... Args>
static inline typename ComGraphMakeUniq<T>::invalid_type ComGraphMakeUnique(Args &&...) = delete;
}  // namespace af

#endif  // AF_COMMON_GRAPH_DEBUG_GE_UTIL_H_
