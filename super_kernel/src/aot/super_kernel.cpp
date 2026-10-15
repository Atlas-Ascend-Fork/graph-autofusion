/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <nlohmann/json.hpp>

#include "super_kernel.h"
#include "sk_log.h"
#include "sk_options_manager.h"
#include "sk_optimizer.h"
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_dump_json.h"
#include "sk_dfx_exception_handler.h"
#include "sk_lock_detector.h"
#include "sk_common.h"
#include "sk_resource_manager.h"
#include "sk_scope_launch.h"
#include "sk_scope_verify.h"
#include "sk_event_recorder.h"

namespace {
class CurrentModelGuard {
 public:
  explicit CurrentModelGuard(aclmdlRI model) {
    SkResourceManager::SetCurrentModel(model);
  }

  ~CurrentModelGuard() {
    SkResourceManager::SetCurrentModel(nullptr);
  }
};

/**
 * @brief Dump graph to JSON file for debugging
 * @param model Model RI handle
 * @param metaDir Meta directory path
 * @param suffix Filename suffix (e.g., "before" or "after")
 * @return aclError status
 */
aclError DumpMdlJson(aclmdlRI model, const std::string &metaDir, const std::string &suffix) {
  if (!sk::logger::FileLogger::Instance().IsEnabled()) {
    return ACL_SUCCESS;  // Kernel meta save is disabled, skip
  }
  std::string jsonPath = metaDir + "/" + suffix + ".json";
  aclError ret = aclmdlRIDebugJsonPrint(model, jsonPath.c_str(), 0);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to print json file: %s", jsonPath.c_str());
    return ACL_ERROR_FAILURE;
  }
  return ACL_SUCCESS;
}

/**
 * @brief Create meta directory for graph dumping
 * @param metaDir Output meta directory path
 * @return aclError status
 */
aclError PrepareGraphDumpEnv(const std::string &modelId, std::string &metaDir) {
  if (!sk::logger::FileLogger::Instance().IsEnabled()) {
    return ACL_SUCCESS;  // Kernel meta save is disabled, skip directory creation
  }
  metaDir = CreateSkMetaDirectory(modelId);
  return ACL_SUCCESS;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclError aclskOptimize(aclmdlRI model, aclskOptions *options) {
  std::string modelId;
  aclError ret = SkResourceManager::GenerateModelId(model, modelId);
  if (ret != ACL_SUCCESS) {
    return ret;
  }

  // Initialize logger first (controlled by environment variable ASCEND_OP_COMPILE_SAVE_KERNEL_META)
  InitSkLogger(modelId);
  // Init device socname, corenum, TICK_US_MULTIPLIER
  InitSkRuntimeConfig();

  std::string metaDir;
  ret = PrepareGraphDumpEnv(modelId, metaDir);
  if (ret != ACL_SUCCESS) {
    return ret;
  }

  SK_LOGI("Start dump tasks by use rts api to JSON...");
  ret = DumpMdlJson(model, metaDir, "sk_mdl_origin");
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  SK_LOGI("End dump tasks by use rts api to JSON...");
  CurrentModelGuard modelGuard(model);
  ret = SkResourceManager::CallbackRegister(model, modelId);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  ret = aclrtExceptionInfoCallbackRegister(SuperKernelExceptionCallBackFunc);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to register exception callback.");
    return ACL_ERROR_FAILURE;
  }
  if (Adx::AdumpRegExceptionDumpCallback) {
    SK_LOGI("Register exception dump callback.");
    ret = Adx::AdumpRegExceptionDumpCallback(ExceptionDumpInfoCallBack);
    if (ret != ACL_SUCCESS) {
      SK_LOGE("Failed to register exception dump callback.");
      return ACL_ERROR_FAILURE;
    }
    SK_LOGI("Register exception dump callback success.");
  }
  SK_LOGI("Begin aclskOptimize");
  SK_LOGI("Start parse sk options...");
  SuperKernelOptionsManager opts;
  opts.ParseOptions(options);
  SK_LOGI("End parse sk options");

  SK_LOGI("Start init sk graph...");
  SuperKernelGraph graph(model, opts, modelId);
  if (!graph.InitSKGraph()) {
    return ACL_ERROR_FAILURE;
  }

  // Dump the initialized graph directly before optimization.
  SK_LOGI("Start dump raw tasks from graph to JSON...");
  if (!DumpGraphJson(graph, opts, metaDir, "sk_graph_origin")) {
    return ACL_ERROR_FAILURE;
  }

  ret = LockDetector::GetDeviceCores();
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  SK_LOGI("End init sk graph");

  SK_LOGI("Start init sk time profiling event recorder...");
  SkEventRecorder::Instance().Init();  // Initialize event recorder (if ASCEND_PROF_SK_ON=1 environment variable is set)
  SK_LOGI("End init sk time profiling event recorder");

  SK_LOGI("Start optimize sk graph...");
  SuperKernelOptimizer optimizer(opts);
  if (!optimizer.Process(graph)) {
    SK_LOGE("aclskOptimize failed: optimize sk graph failed");
    return ACL_ERROR_FAILURE;
  }
  SK_LOGI("End optimize sk graph");

  SK_LOGI("Start update graph...");
  ret = graph.Update();
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to update graph, ret=%d", ret);
    return ret;
  }
  SK_LOGI("End update graph");

  SK_LOGI("Start dump raw tasks after update from modelRI to JSON...");
  const auto &scopeInfos = optimizer.GetScopeInfos();
  if (!DumpGraphJson(model, modelId, opts, metaDir, "sk_graph_updated", &scopeInfos)) {
    return ACL_ERROR_FAILURE;
  }
  SK_LOGI("End dump raw tasks after update from modelRI to JSON...");

  SK_LOGI("Start dump kernel binaries...");
  if (sk::logger::FileLogger::Instance().IsEnabled()) {
    std::string binPath = CreateSkMetaDirectory(graph.GetModelId());
    if (!DumpKernelBinaries(graph, binPath)) {
      SK_LOGE("Failed to dump kernel binaries: %s/bin_files", binPath.c_str());
      return ACL_ERROR_FAILURE;
    }
  }
  SK_LOGI("End dump kernel binaries");

  SK_LOGI("Start dump tasks after update by use rts api to JSON...");
  ret = DumpMdlJson(model, metaDir, "sk_mdl_updated");
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  SK_LOGI("End dump tasks after update by use rts api to JSON");
  return ACL_SUCCESS;
}

aclError aclskScopeBegin(const char *scopeName, aclrtStream stream) {
  InitSkRuntimeConfig();
  if (scopeName != nullptr && scopeName[0] == '\0') {
    SK_DLOGE("Invalid scopeName: name is empty.");
    return ACL_ERROR_INVALID_PARAM;
  }
  return LaunchScopeKernel(scopeName, stream, true);
}

aclError aclskScopeEnd(const char *scopeName, aclrtStream stream) {
  InitSkRuntimeConfig();
  if (scopeName != nullptr && scopeName[0] == '\0') {
    SK_DLOGE("Invalid scopeName: name is empty.");
    return ACL_ERROR_INVALID_PARAM;
  }
  return LaunchScopeKernel(scopeName, stream, false);
}

aclError aclskScopeVerify(const aclskScopeVerifyGraphInfo *verifyGraph, size_t maxSplitResultCount,
                          aclskScopeVerifySplitResult *splitResults, size_t *realSplitResultCount) {
  SK_DLOGI("aclskScopeVerify start");
  sk::logger::FileLogSuppressGuard fileLogSuppressGuard;
  LogScopeVerifyInput(verifyGraph, maxSplitResultCount);
  return RunScopeVerify(verifyGraph, maxSplitResultCount, splitResults, realSplitResultCount);
}

#ifdef __cplusplus
}
#endif
