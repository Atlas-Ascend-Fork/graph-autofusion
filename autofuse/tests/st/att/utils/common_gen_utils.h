/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_AUTOFUSE_ST_ATT_UTILS_COMMON_GEN_UTILS_H_
#define TESTS_AUTOFUSE_ST_ATT_UTILS_COMMON_GEN_UTILS_H_

#include "../../../common/att/common_gen_utils.h"

namespace att {
namespace test {
// Arguments are supplied by the test and may contain shell globs and redirections.
int CompileGeneratedTiling(const std::string &arguments);
}  // namespace test
}  // namespace att

#endif  // TESTS_AUTOFUSE_ST_ATT_UTILS_COMMON_GEN_UTILS_H_
