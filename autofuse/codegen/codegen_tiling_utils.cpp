/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling_utils.h"

namespace codegen {
namespace {
void AppendCvSafetyAivOnlyModeDef(std::stringstream &ss, bool is_batch) {
  ss << "static inline bool is_cv_safety_aiv_only_mode(int64_t tiling_key) {" << std::endl;
  ss << "    switch (tiling_key) {" << std::endl;
  if (is_batch) {
    for (auto key : {513, 8193}) {
      ss << "      case " << key << ":" << std::endl;
    }
  } else {
    for (auto key : {8193, 8209, 8257, 8273, 12289, 12305, 12353, 12369, 16449}) {
      ss << "      case " << key << ":" << std::endl;
    }
  }
  ss << "        return true;" << std::endl;
  ss << "      default:" << std::endl;
  ss << "        return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "}" << std::endl;
}

void AppendCvSafetyBlockIdxScheduledModeDef(std::stringstream &ss, bool is_batch) {
  ss << "static inline bool is_cv_safety_blockidx_scheduled_mode(int64_t tiling_key) {" << std::endl;
  ss << "    switch (tiling_key) {" << std::endl;
  if (is_batch) {
    for (auto key : {257, 273, 321, 337, 769, 785, 833, 849, 2097409, 2097425, 2097473, 2097489}) {
      ss << "      case " << key << ":" << std::endl;
    }
  }
  ss << "        return true;" << std::endl;
  ss << "      default:" << std::endl;
  ss << "        return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "}" << std::endl;
}

void AppendCvSafetyMixModeDef(std::stringstream &ss, bool is_batch) {
  ss << "static inline bool is_cv_safety_mix_mode(int64_t tiling_key) {" << std::endl;
  ss << "    switch (tiling_key) {" << std::endl;
  if (is_batch) {
    for (auto key :
         {513, 4097, 4113, 4161, 4177, 8193, 2097409, 2097425, 2097473, 2097489, 2101249, 2101265, 2101313, 2101329}) {
      ss << "      case " << key << ":" << std::endl;
    }
  } else {
    for (auto key : {4097,    4113,    4161,    4177,    8193,    8209,    8257,    8273,    12289,   12305,   12353,
                     12369,   16449,   28673,   28689,   28737,   28753,   1048577, 1048593, 1048641, 1048657, 1179649,
                     1179665, 1179713, 1179729, 2097153, 2097169, 2097217, 2097233, 2101249, 2101265, 2101313, 2101329,
                     2125825, 2125841, 2125889, 2125905, 2228225, 2228241, 2228289, 2228305}) {
      ss << "      case " << key << ":" << std::endl;
    }
  }
  ss << "        return true;" << std::endl;
  ss << "      default:" << std::endl;
  ss << "        return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "}" << std::endl;
}
}  // namespace

void AppendCvBaseAlignHelperDefs(std::stringstream &ss) {
  ss << std::endl;
  ss << "static int32_t g_basen_basem_align = 0;" << std::endl << std::endl;
  ss << "int32_t get_g_basen_basem_align() {" << std::endl;
  ss << "  return g_basen_basem_align;" << std::endl;
  ss << "}" << std::endl << std::endl;
  ss << "void set_g_basen_basem_align(int32_t value) {" << std::endl;
  ss << "  g_basen_basem_align = value;" << std::endl;
  ss << "}" << std::endl;
}

void AppendCvSafetyMixModeHelperDefs(std::stringstream &ss, bool is_batch) {
  AppendCvSafetyAivOnlyModeDef(ss, is_batch);
  AppendCvSafetyBlockIdxScheduledModeDef(ss, is_batch);
  AppendCvSafetyMixModeDef(ss, is_batch);
}

}  // namespace codegen
