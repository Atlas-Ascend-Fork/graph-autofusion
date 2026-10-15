/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLMODEL_SPECIALIZATION_H
#define ACLMODEL_SPECIALIZATION_H

#include <cstdint>
#include "acl/acl.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY _declspec(dllexport)
#else
#define ACL_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define ACL_FUNC_VISIBILITY
#endif
#endif

// Internal input contract until the public ACL header is integrated.
struct aclmdlRISpecCompileOption {
  const char *kernelName;
  uint64_t compileOptionCount;
  const char *const *compileOptions;
};

struct aclmdlRISpecSKFeature {
  const char *feature;
  uint64_t kernelNameCount;
  const char *const *kernelNames;
};

struct aclmdlRISpecOptions {
  uint64_t jobs;  // Accepted but not yet honoured: kernels are always compiled serially.
  uint64_t specCompileOptionCount;
  aclmdlRISpecCompileOption *specCompileOptions;
  bool enableSK;
  uint64_t specSKFeatureCount;
  aclmdlRISpecSKFeature *specSKFeatures;
  char reserved[128];
};

/**
 * @brief Specialize eligible kernels in an idle model, preserving their launch parameters.
 *
 * The caller must exclude concurrent execution, modification and destruction, and keep the
 * specialized values invariant. options may be null for defaults; its referenced data must remain
 * valid and unchanged during this call. Unsupported kernels retain their original functions.
 *
 * @retval ACL_SUCCESS Optimization completed, including when every kernel was skipped because it
 * could not be specialized.
 * @retval ACL_ERROR_INVALID_PARAM Invalid model or configuration.
 * @retval ACL_ERROR_BAD_ALLOC Insufficient resources before the model was modified.
 * A specialized binary that cannot be loaded, or that lacks its kernel entry, is reported rather
 * than skipped: it indicates a broken artifact or an exhausted resource, and the model is left
 * untouched because loading precedes any modification.
 * A failed commit is not rolled back, so some tasks may already use specialized functions; their
 * binaries stay loaded until the model is destroyed. The caller keeps ownership of the model and
 * decides what to do with it.
 */
ACL_FUNC_VISIBILITY aclError aclmdlRISpecOptimize(aclmdlRI modelRI, aclmdlRISpecOptions *options);
ACL_FUNC_VISIBILITY aclError aclmdlRISpecScopeBegin(aclrtStream stream);
ACL_FUNC_VISIBILITY aclError aclmdlRISpecScopeEnd(aclrtStream stream);
ACL_FUNC_VISIBILITY aclError aclmdlRISpecOptimizeByFlag(aclmdlRI modelRI, aclmdlRISpecOptions *options);

#ifdef __cplusplus
}
#endif

#endif
