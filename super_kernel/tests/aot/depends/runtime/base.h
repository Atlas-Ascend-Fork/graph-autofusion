/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __RUNTIME_BASE_H_STUB__
#define __RUNTIME_BASE_H_STUB__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t rtError_t;

static const int32_t RT_ERROR_NONE = 0;

typedef enum tagRtCoreType {
  RT_CORE_TYPE_AIC = 0,
  RT_CORE_TYPE_AIV,
} rtCoreType_t;

typedef void *rtBinHandle;

#define RT_ERR_REG_NUMS 64

typedef struct tagRtExceptionErrRegInfo {
  uint32_t coreId;
  uint32_t coreType;
  uint64_t startPC;
  uint64_t currentPC;
  uint32_t errReg[RT_ERR_REG_NUMS];
} rtExceptionErrRegInfo_t;

rtError_t rtGetExceptionRegInfo(const void *exception, rtExceptionErrRegInfo_t **errRegInfo, uint32_t *coreNum);

// Exception expand type
typedef enum tagRtExceptionExpandType {
  RT_EXCEPTION_AICORE = 0,
  RT_EXCEPTION_FFTS_PLUS = 1,
  RT_EXCEPTION_FUSION = 2,
} rtExceptionExpandType_t;

#ifdef __cplusplus
}
#endif

#endif  // __RUNTIME_BASE_H_STUB__
