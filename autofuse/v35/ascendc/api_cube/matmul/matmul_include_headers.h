/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef MATMUL_INCLUDE_HEADERS_H
#define MATMUL_INCLUDE_HEADERS_H

#include "arch35/mat_mul_tiling_data.h"
#include "mat_mul_v3_common.h"
#include "arch35/mat_mul_asw_block.h"
#include "arch35/mat_mul_asw_kernel.h"
#include "arch35/mat_mul_stream_k_block.h"
#include "arch35/mat_mul_stream_k_kernel.h"
#include "arch35/mat_mul_v3_full_load_kernel_helper.h"
#include "arch35/mat_mul_full_load.h"
#include "arch35/mm_extension_interface/mm_copy_cube_out.h"
#include "arch35/mm_extension_interface/mm_custom_mm_policy.h"
#include "arch35/mat_mul_fixpipe_opti.h"
#include "arch35/block_scheduler_aswt.h"
#include "arch35/block_scheduler_streamk.h"
#include "arch35/mat_mul_streamk_basic_cmct.h"
#include "arch35/mat_mul_fixpipe_opti_basic_cmct.h"
#include "arch35/mat_mul_input_k_eq_zero_clear_output.h"
#include "arch35/mat_mul_to_mul_cmct.h"
#include "arch35/mat_mul_to_multi_mul_cmct.h"

#endif  // MATMUL_INCLUDE_HEADERS_H
