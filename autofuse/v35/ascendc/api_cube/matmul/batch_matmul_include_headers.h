/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BATCH_MATMUL_INCLUDE_HEADERS_H
#define BATCH_MATMUL_INCLUDE_HEADERS_H

#include "arch35/mat_mul_v3_full_load_kernel_helper.h"
#include "arch35/batch_mat_mul_v3_asw_block_advanced.h"
#include "arch35/batch_mat_mul_v3_asw_kernel_advanced.h"
#include "arch35/batch_mat_mul_v3_asw_broadcast.h"
#include "arch35/batch_mat_mul_v3_asw_al1_full_load_kernel_advanced.h"
#include "arch35/batch_mat_mul_v3_asw_bl1_full_load_kernel_advanced.h"
#include "arch35/batch_mat_mul_v3_iterbatch_block_advanced.h"
#include "arch35/batch_mat_mul_v3_iterbatch_kernel_advanced.h"
#include "arch35/batch_mat_mul_v3_iterbatch_basicapi_block_scheduler.h"
#include "arch35/batch_mat_mul_v3_iterbatch_basicapi_cmct.h"
#include "arch35/batch_mat_mul_v3_mergebatch_basicapi_block_scheduler.h"
#include "arch35/batch_mat_mul_v3_mergebatch_basicapi_cmct.h"
#include "arch35/block_scheduler_aswt.h"
#include "arch35/batch_mat_mul_v3_matmul2mul_block_scheduler.h"
#include "arch35/batch_mat_mul_v3_matmul2mul_cmct.h"
#include "arch35/mat_mul_input_k_eq_zero_clear_output.h"

#endif  // BATCH_MATMUL_INCLUDE_HEADERS_H
