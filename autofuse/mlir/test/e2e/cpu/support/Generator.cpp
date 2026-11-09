/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/support/E2eCase.h"
#include "codegen.h"
#include "common/autofuse_config/auto_fuse_config.h"
#include "common/platform_context.h"
#include "optimize.h"

namespace mlir::afir {

namespace {
#ifdef AF_MLIR_E2E_CPU_NPU_ARCH
// 换平台的第 2 处：**NpuArch 字符串**。legacy 那一半全部按它分流——`REG_ASC_IR` 的
// `.Impl(v1_soc_versions, ...)` 注册的键是 `"2201"`，v2 那份是 `"3510"`，optimizer 的
// `PlatformFactory` 与 ATT 的 tmp 尺寸函数也都查它。
//
// 只改 `CodegenOptions::soc_version` 是不够的：那一处只管 MLIR 侧选哪一侧的
// implementation，调度算出来的 tmp buffer 仍然是 v1 的口径，两边对不上，
// `checkTmpBuffer` 当场判 unsupported。
//
// 走 `SetPlatformInfo` 而不是改 runtime stub：stub 由整个测试套件共用，把它的返回值
// 改成可配会牵连到所有用例。**必须用 `SetPlatformInfo` 而不是 `SetPlatform`**——后者
// 只写 `soc_ver` 就把 `initialized_` 置了位，`aiv_num` / `ub_size` / `l2_size` 全留在 0，
// 调度拿到的是一台没有 UB 的机器。
void OverridePlatformForCase() {
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = AF_MLIR_E2E_CPU_NPU_ARCH;
  platform_info.aiv_num = AF_MLIR_E2E_CPU_AIV_NUM;
  platform_info.ub_size = AF_MLIR_E2E_CPU_UB_SIZE;
  platform_info.l2_size = AF_MLIR_E2E_CPU_L2_SIZE;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);
}
#else
// 默认平台由 runtime stub 给（Ascend910B1 / NpuArch 2201），一个字都不改。
void OverridePlatformForCase() {}
#endif

std::string AppendE2eTilingEntry(const std::string &tiling, const std::string &tilingData) {
  // 复用生成的 ATT 入口，把 device kernel 要的那份 tiling data 初始化出来。
  //
  // 动态形状的用例在 `tiling` 之前多出每个运行期 size var 一个 `uint32_t` 形参，值由
  // 用例经 `g_e2eRuntimeShapeArgs` 给。静态形状的用例那个表是空的，拼出来与从前逐字相同。
  std::string arguments;
  for (uint32_t extent : g_e2eRuntimeShapeArgs) {
    arguments += std::to_string(extent) + ", ";
  }
  std::string resourceSetup;
  std::string resourceArgument = "nullptr";
#ifdef AF_MLIR_E2E_CPU_AIV_LIMIT
  resourceSetup = "  ResLimit resources = {1, " + std::to_string(AF_MLIR_E2E_CPU_AIV_LIMIT) + ", 0, " +
                  std::to_string(AF_MLIR_E2E_CPU_UB_LIMIT) + ", {}};\n";
  resourceArgument = "&resources";
#endif
  // Verify the executed ATT choice, not merely the presence of a kernel branch.
  const auto &config = att::AutoFuseConfig::GetAttStrategyConfig();
  std::string selectionCheck;
  if (config.set_force_schedule_result && config.force_schedule_result >= 0) {
    const auto plan = std::to_string(config.force_schedule_result);
    const bool hasGraphTilingKey = tilingData.find("graph0_tiling_key") != std::string::npos;
    if (hasGraphTilingKey) {
      selectionCheck = "  if (status == 0 && tiling->get_graph0_tiling_key() != " + plan + ") return -1;\n";
      selectionCheck += "  std::printf(\"ATT_SELECTED plan=%u\\n\", tiling->get_graph0_tiling_key());\n";
    } else {
      // Flat ATT data has one implicit plan (0); its root tiling_key selects a template.
      selectionCheck = "  if (status == 0 && " + plan + " != 0) return -1;\n";
      selectionCheck += "  std::printf(\"ATT_SELECTED plan=0\\n\");\n";
    }
    if (config.set_force_tiling_case && !config.force_tiling_case.empty() &&
        config.force_tiling_case.find_first_not_of("0123456789") == std::string::npos) {
      const auto field = hasGraphTilingKey ? "tiling->graph0_result" + plan + "_g0_tiling_data.get_tiling_key()"
                                           : "tiling->get_tiling_key()";
      selectionCheck += "  if (status == 0 && " + field + " != " + config.force_tiling_case + ") return -1;\n";
      selectionCheck += "  std::printf(\"ATT_SELECTED case=%u\\n\", " + field + ");\n";
    }
  }
  return tiling + "\n#include <cstdio>\n" +
         "\nextern \"C\" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, "
         "uint32_t *blockDim) {\n" +
         resourceSetup + "  const auto status = AutofuseTiling(" + arguments + "tiling, workspaceSize, blockDim, " +
         resourceArgument + ");\n" + selectionCheck + "  return status;\n}\n";
}
}  // namespace

TEST(MlirE2eCpuGenerator, GeneratesSources) {
  OverridePlatformForCase();
  auto graph = BuildGraph();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fusedScheduleResult;
  ASSERT_EQ(optimizer.Optimize(graph, fusedScheduleResult), af::SUCCESS);

  codegen::Codegen codegen(codegen::CodegenOptions{.tiling_lib_path = AF_MLIR_E2E_CPU_TILING_LIBRARY,
                                                   .tiling_lib_codegen_symbol = "CodegenTiling",
                                                   .using_att_calc_qbt_size = false,
                                                   .soc_version = AF_MLIR_E2E_CPU_SOC_VERSION});
  codegen::CodegenResult result;
  ASSERT_EQ(codegen.GenerateForInductor(fusedScheduleResult, result), af::SUCCESS);
  CheckGeneratedKernel(result.kernel);
  ASSERT_FALSE(HasFailure());
  WriteSources(result.kernel, result.tiling_data, AppendE2eTilingEntry(result.tiling, result.tiling_data));
}
}  // namespace mlir::afir
