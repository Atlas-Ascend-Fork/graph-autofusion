/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __Autofuse_Tiling_Data_H__
#define __Autofuse_Tiling_Data_H__
#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"
#define BEGIN_TILING_DATA_DEF_T(name) struct name {
#define TILING_DATA_FIELD_DEF_T(type, name) \
  type name;                                \
  inline void set_##name(type value) {      \
    name = value;                           \
  }                                         \
  inline type get_##name() {                \
    return name;                            \
  }                                         \
  inline type *get_addr_##name() {          \
    return &name;                           \
  }
#define END_TILING_DATA_DEF_T \
  }                           \
  ;
#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) struct_type filed_name;

BEGIN_TILING_DATA_DEF_T(AutofuseTilingData)
TILING_DATA_FIELD_DEF_T(uint16_t, block_dim);
TILING_DATA_FIELD_DEF_T(uint16_t, tiling_key);
TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);
TILING_DATA_FIELD_DEF_T(int64_t, s2);
TILING_DATA_FIELD_DEF_T(int64_t, s3);
TILING_DATA_FIELD_DEF_T(uint32_t, z0z1t_size);
TILING_DATA_FIELD_DEF_T(uint32_t, z0z1Tb_size);
TILING_DATA_FIELD_DEF_T(uint32_t, q0_size);
TILING_DATA_FIELD_DEF_T(uint32_t, q1_size);
TILING_DATA_FIELD_DEF_T(uint32_t, q2_size);
TILING_DATA_FIELD_DEF_T(uint32_t, b0_size);
END_TILING_DATA_DEF_T;

struct AutofuseTilingDataPerf {
  AutofuseTilingData tiling_data;
  double best_perf;
};
#endif
