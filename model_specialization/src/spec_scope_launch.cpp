/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "spec_scope_launch.h"

#include <array>
#include <cstdint>
#include "sk_log.h"

extern "C" void spec_scope_kernel_begin_do_dav_2201(void *stream);
extern "C" void spec_scope_kernel_end_do_dav_2201(void *stream);
extern "C" void spec_scope_kernel_begin_do_dav_3510(void *stream);
extern "C" void spec_scope_kernel_end_do_dav_3510(void *stream);

namespace model_spec {
namespace {
using MarkerLaunch = void (*)(void *);
struct ScopeKernelImpls {
  int64_t arch;
  MarkerLaunch begin;
  MarkerLaunch end;
  const char *beginName;
  const char *endName;
};
const std::array<ScopeKernelImpls, 2> SCOPE_KERNEL_IMPLS = {{
    {2201, spec_scope_kernel_begin_do_dav_2201, spec_scope_kernel_end_do_dav_2201, SPEC_SCOPE_BEGIN_DAV_2201,
     SPEC_SCOPE_END_DAV_2201},
    {3510, spec_scope_kernel_begin_do_dav_3510, spec_scope_kernel_end_do_dav_3510, SPEC_SCOPE_BEGIN_DAV_3510,
     SPEC_SCOPE_END_DAV_3510},
}};
}  // namespace

aclError LaunchSpecScopeMarker(aclrtStream stream, SpecScopeMarkerKind kind) {
  if (stream == nullptr || (kind != SpecScopeMarkerKind::Begin && kind != SpecScopeMarkerKind::End)) {
    SK_DLOGE("Invalid Spec scope marker request: stream=%p, kind=%d", stream, static_cast<int>(kind));
    return ACL_ERROR_INVALID_PARAM;
  }
  const char *markerType = kind == SpecScopeMarkerKind::Begin ? "Begin" : "End";
  aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI model = nullptr;
  aclError result = aclmdlRICaptureGetInfo(stream, &status, &model);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("Spec scope capture query failed: markerType=%s, stream=%p, ret=%d", markerType, stream, result);
    return result;
  }
  switch (status) {
    case ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE:
      break;
    case ACL_MODEL_RI_CAPTURE_STATUS_NONE:
      SK_DLOGE("Spec scope stream is not captured: markerType=%s, stream=%p, status=%d", markerType, stream,
               static_cast<int>(status));
      return ACL_ERROR_RT_STREAM_NOT_CAPTURED;
    case ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED:
      SK_DLOGE("Spec scope capture is invalidated: markerType=%s, stream=%p, status=%d", markerType, stream,
               static_cast<int>(status));
      return ACL_ERROR_RT_STREAM_CAPTURE_INVALIDATED;
    default:
      // Unknown Runtime output is a contract failure, not a non-captured stream.
      SK_DLOGE("Unknown Spec scope capture status: markerType=%s, stream=%p, status=%d", markerType, stream,
               static_cast<int>(status));
      return ACL_ERROR_FAILURE;
  }
  int32_t deviceId = 0;
  result = aclrtGetDevice(&deviceId);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("Spec scope current device query failed: markerType=%s, stream=%p, ret=%d", markerType, stream, result);
    return result;
  }
  int64_t arch = 0;
  result = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId), ACL_DEV_ATTR_NPU_ARCH, &arch);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("Spec scope architecture query failed: markerType=%s, stream=%p, device=%d, ret=%d", markerType, stream,
             deviceId, result);
    return result;
  }
  for (const auto &impl : SCOPE_KERNEL_IMPLS) {
    if (impl.arch == arch) {
      const bool isBegin = kind == SpecScopeMarkerKind::Begin;
      const char *kernelName = isBegin ? impl.beginName : impl.endName;
      SK_DLOGI("Dispatch Spec scope marker: markerType=%s, stream=%p, device=%d, arch=%lld, kernelName=%s", markerType,
               stream, deviceId, static_cast<long long>(arch), kernelName);
      (isBegin ? impl.begin : impl.end)(stream);
      // ASC's generated void wrapper reports synchronous Runtime errors.
      // Preserve SK dispatch semantics; this return does not report device completion.
      return ACL_SUCCESS;
    }
  }
  SK_DLOGE("Unsupported Spec scope marker arch: markerType=%s, stream=%p, device=%d, arch=%lld", markerType, stream,
           deviceId, static_cast<long long>(arch));
  return ACL_ERROR_FEATURE_UNSUPPORTED;
}
}  // namespace model_spec
