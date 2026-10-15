/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __RT_DATA_H__
#define __RT_DATA_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// #include "rt_sk_intf.h"

#define K_TYPE_AICORE 1
#define K_TYPE_AIC 2
#define K_TYPE_AIV 3
#define K_TYPE_MIX_AIC_MAIN 4
#define K_TYPE_MIX_AIV_MAIN 5
#define K_TYPE_AIC_ROLLBACK 6
#define K_TYPE_AIV_ROLLBACK 7

typedef enum rt_task_type {
  RT_TASK_DEFAULT = 0,
  RT_TASK_KERNEL = 1,
  RT_TASK_EVENT_RECORD = 2,
  RT_TASK_EVENT_WAIT = 3,
  RT_TASK_EVENT_RESET = 4,
  RT_TASK_VALUE_WRITE = 5,
  RT_TASK_VALUE_WAIT = 6
} rt_task_type;

struct rt_info {
  size_t arg_size;
  void *arg_data;
  const char *func_name;
  void *bin_hdl;
  uint64_t stream_id;
  uint32_t numBlocks;
  uint32_t kernel_type;
  uint32_t task_ratio[2];
  uint8_t legacy;
  uint8_t reserve[7];

  // 以下字段为新添加，用于支持事件和内存值相关的任务参数
  uint32_t task_type;
  uint32_t event_type;
  uint32_t event_id;
  void *event_addr;
  uint64_t value;
  uint32_t value_size;
  uint32_t wait_flag;
  void *value_addr;
};

typedef struct rtFunctionInfo {
  void *pcAddr;
  uint32_t prefetchCnt;
  uint8_t mixType;  // 0:NO_MIX; 1:MIX_AIC; 2:MIX_AIV; 3:MIX_AIC_AIV
  uint8_t reserved[3];
} rtFunctionInfo_t;

typedef struct tagRtKernelInfo {
  uint8_t functionInfoNum;
  uint8_t reserved[3];
  rtFunctionInfo_t functionInfo[2];
} rtKernelDetailInfo_t;

void rt_start_capture(int dry_run);
void rt_stop_capture(void);
void rt_get_kernel_info(struct rt_info **info, size_t *size);

int rtKernelGetAddrAndPrefCntV2(void *hdl, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                                rtKernelDetailInfo_t *kernelInfo);
int rtGetFunctionByName(const char *name, void **fun_hdl);
int rtGetAddrByFun(const void *fun_hdl, void **addr);
int rtQueryFunctionRegistered(const char *name);
int rtFunctionRegister(void *bin_hdl, const void *fun, const char *name, const void *kernel_info, uint32_t mode);

#ifdef __cplusplus
}
#endif

#endif  // __RT_DATA_H__
