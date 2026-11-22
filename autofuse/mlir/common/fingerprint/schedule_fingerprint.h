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

// Changing either format version invalidates existing artifact keys.
inline constexpr char kScheduleFingerprintFormatVersion[] = "autofuse.schedule.fingerprint.v1";
inline constexpr char kArtifactCacheKeyFormatVersion[] = "autofuse.artifact.cache.v1";

// Describes schedule structure only. Artifact keys also include the toolchain
// and pipeline versions in ArtifactVersionSet.
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
  // Shape specialization is part of the artifact cache key. This metadata
  // does not imply compile-time ATT winner selection by PlanSelector.
  bool is_dynamic_shape{false};
  // Concrete shape bindings used for static compilation; empty in dynamic mode.
  // Different static shapes must produce different cache keys for the same schedule.
  // Kept local to the fingerprint contract to avoid a bridge dependency.
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

// Fingerprints Schedule output, so a cache hit skips Codegen but not Schedule.
// Skipping both would require a stable pre-Schedule FusionGraph snapshot, which
// Schedule does not yet expose (design doc FR6/3.2.7).
ScheduleFingerprint ComputeScheduleFingerprint(const ascir::FusedScheduledResult &fused_result);

ArtifactCacheKey BuildArtifactCacheKey(const ScheduleFingerprint &structure, const ArtifactVersionSet &versions,
                                       const CompileOptions &options);

}  // namespace mlir::common

#endif  // AUTOFUSE_MLIR_COMMON_FINGERPRINT_SCHEDULE_FINGERPRINT_H
