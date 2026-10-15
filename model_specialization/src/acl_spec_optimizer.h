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
 * \file acl_spec_optimizer.h
 * \brief Internal interfaces for request collection, kernel specialization, task update, and
 *        ownership of the specialized binaries.
 */

#ifndef ACL_SPEC_OPTIMIZER_H
#define ACL_SPEC_OPTIMIZER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_rt_compile.h"
#include "feature_manager.h"

namespace model_spec {

class KernelSpecRequest {
 public:
  static void UnloadBinary(aclrtBinHandle binary);
  using Binary = std::unique_ptr<void, decltype(&UnloadBinary)>;

  void BindApiRequest();
  aclmdlRITask task = nullptr;
  uint32_t streamIndex = 0;
  uint32_t taskIndex = 0;
  aclmdlRITaskParams originalParams{};
  std::string resourceId;
  std::string kernelEntry;
  std::string outputElfPath;
  std::vector<unsigned char> hostArgs;
  std::vector<const void *> argumentAddresses;
  std::vector<uint64_t> argumentBytes;
  std::vector<std::string> basicOptionStorage;
  std::vector<const char *> basicOptionPointers;
  std::vector<std::string> skOptionStorage;
  std::vector<const char *> skOptionPointers;
  aclrtcKernelSpecRequest apiRequest{};
  Binary binary{nullptr, UnloadBinary};
  aclrtFuncHandle specFuncHandle = nullptr;
};

class SpecOutputDirectory final {
 public:
  SpecOutputDirectory() = default;
  ~SpecOutputDirectory();
  SpecOutputDirectory(const SpecOutputDirectory &) = delete;
  SpecOutputDirectory &operator=(const SpecOutputDirectory &) = delete;

  aclError Create(uint32_t modelId);
  const std::string &Path() const {
    return path_;
  }

 private:
  std::string path_;
  bool created_ = false;
};

class AclSpecOptimizer final {
 public:
  aclError Init(aclmdlRI model);
  aclError Optimize(aclmdlRI model, const model_spec::FeatureManager &featureManager) const;

 private:
  aclError Collect(aclmdlRI model, const model_spec::FeatureManager &featureManager, std::string_view outputDirectory,
                   std::vector<std::unique_ptr<KernelSpecRequest>> &requests) const;
  bool TryBuildRequest(aclmdlRITask task, uint32_t streamIndex, uint32_t taskIndex,
                       const model_spec::FeatureManager &featureManager, std::string_view outputDirectory,
                       KernelSpecRequest &request) const;
  void SpecializeKernels(std::vector<std::unique_ptr<KernelSpecRequest>> &requests) const;
  aclError LoadKernels(std::vector<std::unique_ptr<KernelSpecRequest>> &requests) const;
  aclError UpdateTasks(aclmdlRI model, const std::vector<std::unique_ptr<KernelSpecRequest>> &requests) const;

  SpecOutputDirectory outputDirectory_;
};

class ModelSpecResourceManager final {
 public:
  static ModelSpecResourceManager &GetInstance();

  aclError RetainBinariesForModel(aclmdlRI model, const std::vector<std::unique_ptr<KernelSpecRequest>> &requests);

  ModelSpecResourceManager(const ModelSpecResourceManager &) = delete;
  ModelSpecResourceManager &operator=(const ModelSpecResourceManager &) = delete;

 private:
  ModelSpecResourceManager() = default;
  ~ModelSpecResourceManager() = default;

  aclError TryTakeOwnership(aclmdlRI model, const std::vector<std::unique_ptr<KernelSpecRequest>> &requests);
  static void OnModelDestroy(void *userData);

  std::mutex mutex_;
  std::unordered_map<aclmdlRI, std::vector<aclrtBinHandle>> modelBinaries_;
};

inline constexpr std::string_view SPEC_RESOURCE_ID_SECTION_NAME = ".ascend.meta";
inline constexpr uint16_t SPEC_RESOURCE_ID_TLV_TYPE = 6;
inline constexpr uint16_t SPEC_RESOURCE_ID_TLV_LENGTH = 64;

bool ReadSpecResourceId(const void *elfBytes, size_t elfSize, std::string &resourceId);

}  // namespace model_spec

#endif
