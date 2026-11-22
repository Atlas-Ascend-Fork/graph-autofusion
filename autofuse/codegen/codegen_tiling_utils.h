/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CODEGEN_CODEGEN_TILING_UTILS_H_
#define AUTOFUSE_CODEGEN_CODEGEN_TILING_UTILS_H_

#include <sstream>
#include <string>
#include "ascir.h"
#include "schedule_result.h"

namespace codegen {

// A [K, B, N] storage exposed as a zero-offset [B, K, N] B input.
__attribute__((visibility("hidden"))) ge::AscNodePtr GetBmmInputView(const ge::AscNodePtr &node);
__attribute__((visibility("hidden"))) ge::AscNodePtr FindBmmInputView(const ascir::FusedScheduledResult &fused);
__attribute__((visibility("hidden"))) bool IsRegisterBmmInputPackSupported();
__attribute__((visibility("hidden"))) bool CanPackBmmInput(const ge::AscNodePtr &view);

__attribute__((visibility("hidden"))) void AppendCvBaseAlignHelperDefs(std::stringstream &ss);
__attribute__((visibility("hidden"))) void AppendCvPlatformSelectionDefs(std::stringstream &ss, bool avoid_tensor_api,
                                                                         bool a_trans, bool b_trans);
__attribute__((visibility("hidden"))) void AppendCvSafetyMixModeHelperDefs(std::stringstream &ss, bool is_batch,
                                                                           bool is_device = false);
__attribute__((visibility("hidden"))) void AppendCvUbSupportedKeyHelperDef(std::stringstream &ss,
                                                                           const std::string &tensor_b_full_load,
                                                                           bool is_device = false);

}  // namespace codegen

#endif  // AUTOFUSE_CODEGEN_CODEGEN_TILING_UTILS_H_
