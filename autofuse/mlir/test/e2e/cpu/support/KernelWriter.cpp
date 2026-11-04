/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include "test/e2e/cpu/support/E2eCase.h"

namespace mlir::afir {
std::vector<uint32_t> g_e2eRuntimeShapeArgs;

namespace {
std::string MakeStandaloneTilingSource(const std::string &tiling) {
  std::stringstream input(tiling);
  std::stringstream output;
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("#include \"autofuse_tiling_func_") != std::string::npos ||
        line.find("#include \"cube_kernel_tiling_wrapper.h\"") != std::string::npos) {
      continue;
    }
    output << line << '\n';
  }
  return output.str();
}
}  // namespace

void WriteSources(const std::string &kernel, const std::string &tilingData, const std::string &tiling) {
  constexpr char kHostLaunchMarker[] = "extern \"C\" int64_t AutofuseLaunch";
  const auto hostLaunchOffset = kernel.find(kHostLaunchMarker);
  ASSERT_NE(hostLaunchOffset, std::string::npos);

  std::ofstream kernelFile(AF_MLIR_E2E_CPU_KERNEL_SOURCE);
  std::ofstream tilingDataFile(AF_MLIR_E2E_CPU_TILING_HEADER);
  std::ofstream tilingFile(AF_MLIR_E2E_CPU_TILING_SOURCE);
  ASSERT_TRUE(kernelFile.is_open());
  ASSERT_TRUE(tilingDataFile.is_open());
  ASSERT_TRUE(tilingFile.is_open());
  kernelFile << kernel.substr(0, hostLaunchOffset);
  tilingDataFile << tilingData;
  tilingFile << MakeStandaloneTilingSource(tiling);
  ASSERT_TRUE(kernelFile.good());
  ASSERT_TRUE(tilingDataFile.good());
  ASSERT_TRUE(tilingFile.good());
}
}  // namespace mlir::afir
