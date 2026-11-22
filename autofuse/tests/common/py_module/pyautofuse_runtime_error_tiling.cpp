/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * http://www.huawei.com
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include <map>
#include <stdexcept>
#include <string>

namespace ascir {
struct FusedScheduledResult;
}

extern "C" bool CodegenTiling(const std::string &op_name, const ascir::FusedScheduledResult &fused_schedule_result,
                              std::map<std::string, std::string> &options,
                              std::map<std::string, std::string> &tiling_func) {
  (void)op_name;
  (void)fused_schedule_result;
  (void)options;
  (void)tiling_func;
  throw std::runtime_error("pyautofuse runtime error test");
}
