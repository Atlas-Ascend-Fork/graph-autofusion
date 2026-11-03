/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_FINAL_TILING_SELECTION_SNAPSHOT_H
#define AUTOFUSE_FINAL_TILING_SELECTION_SNAPSHOT_H

#include <string>

namespace ascgen_utils {

inline std::string GenFinalTilingSelectionSnapshotSnippet() {
  return R"(#ifndef AUTOFUSE_FINAL_TILING_SELECTION_SNAPSHOT_DEFINED
#define AUTOFUSE_FINAL_TILING_SELECTION_SNAPSHOT_DEFINED
struct FinalTilingSelectionSnapshot {
  bool valid{false};
  uint32_t graph{0U};
  uint32_t result{0U};
  uint32_t group{0U};
  uint32_t tiling_key{0U};
  int32_t case_id{-1};
  int32_t sub_case_id{-1};
};
#endif
)";
}

}  // namespace ascgen_utils

#endif  // AUTOFUSE_FINAL_TILING_SELECTION_SNAPSHOT_H
