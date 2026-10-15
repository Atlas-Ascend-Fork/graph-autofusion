/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclmodel_specialization.cpp
 * \brief Public model specialization entry point.
 */

#include "aclmodel_specialization.h"

#include <new>
#include <stdexcept>

#include "acl_spec_optimizer.h"
#include "feature_manager.h"
#include "sk_log.h"

using model_spec::AclSpecOptimizer;
using model_spec::FeatureManager;

aclError aclmdlRISpecOptimize(aclmdlRI modelRI, aclmdlRISpecOptions *options) {
  try {
    FeatureManager featureManager;
    if (!featureManager.Init(options)) {
      SK_DLOGE("aclmdlRISpecOptimize invalid configuration: model=%p", modelRI);
      return ACL_ERROR_INVALID_PARAM;
    }
    AclSpecOptimizer optimizer;
    const aclError result = optimizer.Init(modelRI);
    if (result != ACL_SUCCESS) {
      return result;
    }
    return optimizer.Optimize(modelRI, featureManager);
  } catch (const std::bad_alloc &) {
    SK_DLOGE("aclmdlRISpecOptimize allocation exception: model=%p", modelRI);
    return ACL_ERROR_BAD_ALLOC;
  } catch (const std::length_error &) {
    SK_DLOGE("aclmdlRISpecOptimize length exception: model=%p", modelRI);
    return ACL_ERROR_BAD_ALLOC;
  } catch (const std::invalid_argument &) {
    SK_DLOGE("aclmdlRISpecOptimize invalid argument exception: model=%p", modelRI);
    return ACL_ERROR_INVALID_PARAM;
  } catch (const std::exception &error) {
    SK_DLOGE("aclmdlRISpecOptimize exception: model=%p reason=%s", modelRI, error.what());
    return ACL_ERROR_FAILURE;
  } catch (...) {
    SK_DLOGE("aclmdlRISpecOptimize unknown exception: model=%p", modelRI);
    return ACL_ERROR_FAILURE;
  }
}
