/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __RUNTIME_RT_MODEL_H_STUB__
#define __RUNTIME_RT_MODEL_H_STUB__

#include "base.h"

extern "C" {
typedef void *rtTask_t;
rtError_t rtModelTaskCopyArgsDeviceBuffer(rtTask_t task, void **devBufferAddr, size_t *size);
}

#endif  // __RUNTIME_RT_MODEL_H_STUB__
