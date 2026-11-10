/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_COMMON_FINGERPRINT_SCHEDULE_FINGERPRINT_H
#define AUTOFUSE_MLIR_COMMON_FINGERPRINT_SCHEDULE_FINGERPRINT_H

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

#include "common/schedule_result.h"

namespace mlir::common {

// Changing either format version intentionally invalidates previously
// generated artifact keys.
inline constexpr char kScheduleFingerprintFormatVersion[] = "autofuse.schedule.fingerprint.v1";
inline constexpr char kArtifactCacheKeyFormatVersion[] = "autofuse.artifact.cache.v1";

// A structural fingerprint is deliberately separate from an artifact key.  It
// describes the schedule shape only; toolchain and pipeline versions belong to
// ArtifactVersionSet below.
struct ScheduleFingerprint {
  std::string value;

  bool operator==(const ScheduleFingerprint &other) const {
    return value == other.value;
  }
  bool operator!=(const ScheduleFingerprint &other) const {
    return !(*this == other);
  }
};

struct ArtifactVersionSet {
  std::string target_profile_version;
  std::string cann_toolchain_version;
  std::string bridge_version;
  std::string att_schema_version;
  std::string codegen_version;
  std::string pass_pipeline_version;
  std::string entry_abi_version;
  // Whether this artifact targets the dynamic-shape dispatch-by-tiling_key
  // path or one concrete (static) shape (see FR4/3.2.5 in the design doc:
  // dynamic generation keeps every candidate, static generation keeps
  // only the ATT winner). A dynamic-mode and a static-mode artifact are never
  // interchangeable even for the same schedule structure, so this must be
  // part of the versioned key, not left to be inferred from the structural
  // fingerprint alone.
  bool is_dynamic_shape{false};
  // The concrete shape symbol -> value bindings a static-shape artifact was
  // compiled against (mirrors mlir::bridge::ConcreteShape; duplicated
  // here rather than reused by type to keep this header free of a
  // mlir::bridge dependency). Always empty when is_dynamic_shape is true.
  // Two static artifacts for the same schedule structure but different
  // concrete shapes are different compiled programs and must not collide on
  // one cache key.
  std::map<std::string, int64_t> static_shape_constants;
};

struct CompileOptions {
  std::map<std::string, std::string> values;
};

struct ArtifactCacheKey {
  std::string value;

  bool operator==(const ArtifactCacheKey &other) const {
    return value == other.value;
  }
  bool operator!=(const ArtifactCacheKey &other) const {
    return !(*this == other);
  }
};

// NOTE on scope (see FR6/3.2.7 in the design doc): the ideal input is a
// pre-Schedule FusionGraph snapshot, which would let a cache hit skip both
// Schedule and Codegen. This function's actual input is FusedScheduledResult
// -- Schedule's OUTPUT, not its input -- because Schedule does not yet
// expose a stable pre-Schedule structural snapshot. That means a cache hit
// against this fingerprint can only skip Codegen; Schedule itself still runs
// every time. Do not present this fingerprint as having skipped Schedule.
ScheduleFingerprint ComputeScheduleFingerprint(const ascir::FusedScheduledResult &fused_result);

ArtifactCacheKey BuildArtifactCacheKey(const ScheduleFingerprint &structure, const ArtifactVersionSet &versions,
                                       const CompileOptions &options);

}  // namespace mlir::common

#endif  // AUTOFUSE_MLIR_COMMON_FINGERPRINT_SCHEDULE_FINGERPRINT_H
