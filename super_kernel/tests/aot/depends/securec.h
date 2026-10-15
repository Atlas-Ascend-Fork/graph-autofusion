/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <stddef.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void SkUtSetSecurecMemcpyFailOnCall(int hitOnCall);
void SkUtSetSecurecMemsetFailOnCall(int hitOnCall);
int SkUtSecurecShouldFailMemcpy();
int SkUtSecurecShouldFailMemset();

// Define errno_t type for secure functions
#ifndef _ERRNO_T_DEFINED
#define _ERRNO_T_DEFINED
typedef int errno_t;
#endif

#define SECUREC_INLINE static inline

SECUREC_INLINE errno_t memcpy_s(void *dest, size_t destMax, const void *src, size_t count) {
  if (SkUtSecurecShouldFailMemcpy() != 0) {
    return -1;
  }
  if (dest == NULL || src == NULL || destMax < count) {
    return -1;
  }
  memcpy(dest, src, count);
  return 0;
}

SECUREC_INLINE errno_t memset_s(void *dest, size_t destMax, int c, size_t count) {
  if (SkUtSecurecShouldFailMemset() != 0) {
    return -1;
  }
  if (dest == NULL || destMax < count) {
    return -1;
  }
  memset(dest, c, count);
  return 0;
}

SECUREC_INLINE int vsnprintf_s(char *dest, size_t destMax, size_t count, const char *fmt, va_list arglist) {
  if (dest == NULL || fmt == NULL || destMax == 0) {
    return -1;
  }
  (void)count;
  return vsnprintf(dest, destMax, fmt, arglist);
}

SECUREC_INLINE int snprintf_s(char *dest, size_t destMax, size_t count, const char *fmt, ...) {
  if (dest == NULL || fmt == NULL || destMax == 0) {
    return -1;
  }
  (void)count;
  va_list arglist;
  va_start(arglist, fmt);
  int ret = vsnprintf(dest, destMax, fmt, arglist);
  va_end(arglist);
  return ret;
}

#ifdef __cplusplus
}
#endif
