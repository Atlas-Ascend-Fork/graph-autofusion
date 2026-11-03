/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- CodegenOrchestration.h - MLIR codegen entry for Codegen -*- C++ -*-===//
//
// Runs the MLIR codegen path for one fused schedule result: select plan
// candidates, lower each through addPlanCandidateLoweringPipeline, aggregate
// into one entry module, and emit kernel/tiling artifacts.
//
// Hooked at codegen.cpp's Codegen::GenerateForInductor/Generate, gated by
// ENABLE_AUTOFUSE_MLIR and AUTOFUSE_DFX_FLAGS. The default configuration is
// none, which keeps the legacy behavior unchanged. The `mlir` configuration
// is also a test-only measurement switch: it makes a capability shortfall
// fail instead of falling back, see MlirCodegenGate / ReportStrictVerdict.
// AUTOFUSE_DFX_FLAGS also controls the file-only dump through
// ascir::utils::GetDumpFilePrefix(), alongside the other Autofuse diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_BRIDGE_CODEGEN_ORCHESTRATION_CODEGEN_ORCHESTRATION_H
#define AUTOFUSE_MLIR_BRIDGE_CODEGEN_ORCHESTRATION_CODEGEN_ORCHESTRATION_H

#include <string>

#include "codegen.h"
#include "schedule_result.h"  // ascir::FusedScheduledResult
#include "llvm/ADT/StringRef.h"

namespace mlir::bridge {

inline constexpr llvm::StringLiteral kMlirCodegenConfigOptionName = "mlir_codegen_config";

enum class MlirCodegenConfig {
  kNone,
  kMlir,
  kFallback,
  kInvalid,
};

struct MlirCodegenOptions {
  MlirCodegenConfig config = MlirCodegenConfig::kNone;
  std::string dump_root;
};

// Read all process-level configuration used by the MLIR codegen path once.
// The dump root is supplied by the shared legacy DFX configuration; no new
// environment variable is introduced for MLIR dumps.
MlirCodegenOptions ReadMlirCodegenOptions();

enum class MlirCodegenStatus { kNotSupported, kFailed, kSucceeded };
enum class MlirCallerScene { kInductor, kTensorFlow };

// 这张图在哪一道闸门上停下的。**按闸门分，不按 op 分**——同一道闸门拦下不同 op 是一件事，
// 不同闸门拦下同一个 op 是两件事（与 `test/probe/CapabilityBoundaryProbe.cpp` 的分桶同一
// 口径）。
//
// 只在严格模式（`options.config == MlirCodegenConfig::kMlir`）下用：取值随那一行标签打到
// stderr，供「接管表」按它聚合。**结论不看它**——状态仍然由 `classify()` 给，这里只是给同一次
// 结论配一个可机读的理由。
enum class MlirCodegenGate {
  kNone,                  // 一道都没拦下：这张图是 MLIR 发的
  kCubeFusion,            // Cube 融合，明确不在这条路的范围
  kSocVersion,            // soc_version 解析不出来
  kCandidatePreparation,  // 候选选择 / ATT tiling
  kPipelineSession,       // 这颗芯片没有 TargetProfile
  kLowering,              // bridge、能力层或 pattern
  kArtifactAssembly,      // kernel / tiling 文本装配
  kException,             // 抛了异常
};

// 严格模式打在 stderr 上那行标签的前缀。产生方（`RunMlirCodegen`）与消费方
// （`test/probe/legacy_takeover_table.sh`）共用这一个常量。
inline constexpr llvm::StringLiteral kMlirCodegenStrictMarker = "AF_MLIR_STRICT";

// Runs the MLIR codegen orchestration path for one fused schedule result:
// select plan candidates, lower each through addPlanCandidateLoweringPipeline,
// aggregate into one entry module, and emit kernel/tiling artifacts into
// `result`. With AUTOFUSE_DFX_FLAGS="--mlir_codegen_config=none" (the default),
// returns kNotSupported so the caller keeps the legacy path. With
// "--mlir_codegen_config=mlir", any unsupported or failed result becomes
// kFailed. With "--mlir_codegen_config=fallback", unsupported and failed MLIR
// attempts return kNotSupported so the caller can use legacy. Invalid values
// return kFailed.
// When `soc_version` is empty, the current SoC name is resolved through
// aclrtGetSocName(); under the fallback configuration, a resolution failure is
// also handled according to the configured policy.
MlirCodegenStatus RunMlirCodegen(const ascir::FusedScheduledResult &fused_schedule_result, MlirCallerScene scene,
                                 const std::string &soc_version, codegen::CodegenResult &result);

}  // namespace mlir::bridge

#endif
