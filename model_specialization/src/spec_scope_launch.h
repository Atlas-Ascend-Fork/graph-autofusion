/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MODEL_SPEC_SPEC_SCOPE_LAUNCH_H
#define MODEL_SPEC_SPEC_SCOPE_LAUNCH_H

#include "acl/acl.h"

namespace model_spec {
enum class SpecScopeMarkerKind { Begin, End };

// Full device names are shared by dispatch and subsequent marker recognition.
inline constexpr char SPEC_SCOPE_BEGIN_DAV_2201[] = "spec_scope_kernel_begin_dav_2201";
inline constexpr char SPEC_SCOPE_END_DAV_2201[] = "spec_scope_kernel_end_dav_2201";
inline constexpr char SPEC_SCOPE_BEGIN_DAV_3510[] = "spec_scope_kernel_begin_dav_3510";
inline constexpr char SPEC_SCOPE_END_DAV_3510[] = "spec_scope_kernel_end_dav_3510";

aclError LaunchSpecScopeMarker(aclrtStream stream, SpecScopeMarkerKind kind);
}  // namespace model_spec

#endif
