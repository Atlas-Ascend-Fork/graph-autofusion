/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/opsproto_manager.h"
#include <cstdlib>
#include <functional>
#include "framework/common/debug/ge_log.h"
#include "graph/types_af.h"
#include "graph/def_types.h"
#include "graph/operator_factory_impl.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "graph/operator_factory_impl.h"

namespace af {
OpsProtoManager *OpsProtoManager::Instance() {
  static OpsProtoManager instance;
  return &instance;
}

bool OpsProtoManager::Initialize(const std::map<std::string, std::string> &options) {
  const std::lock_guard<std::mutex> lock(mutex_);

  if (is_init_) {
    GELOGI("OpsProtoManager is already initialized.");
    return true;
  }

  const std::map<std::string, std::string>::const_iterator iter = options.find("ge.opsProtoLibPath");
  if (iter == options.end()) {
    GELOGW("[Initialize][CheckOption] Option \"ge.opsProtoLibPath\" not set");
    return false;
  }

  pluginPath_ = iter->second;
  LoadBuiltinOpsPluginSo(pluginPath_);

  is_init_ = true;

  return true;
}

void OpsProtoManager::Finalize() {
  const std::lock_guard<std::mutex> lock(mutex_);

  if (!is_init_) {
    GELOGI("OpsProtoManager is not initialized.");
    return;
  }

  for (const auto handle : handles_) {
    if (handle != nullptr) {
      if (mmDlclose(handle) != 0) {
        const char_t *error = mmDlerror();
        error = (error == nullptr) ? "" : error;
        GELOGW("[Finalize][CloseHandle] close handle unsuccessfully, reason:%s", error);
        continue;
      }
      GELOGI("close opsprotomanager handler success");
    } else {
      GELOGW("[Finalize][CheckHandle] handler is null");
    }
  }

  is_init_ = false;
}

OpsProtoManager::~OpsProtoManager() {
  OperatorFactoryImpl::ReleaseRegInfo();
  Finalize();
}

static std::vector<std::string> SplitStr(const std::string &str, const char_t delim) {
  std::vector<std::string> elems;
  if (str.empty()) {
    elems.emplace_back("");
    return elems;
  }

  std::stringstream str_stream(str);
  std::string item;

  while (getline(str_stream, item, delim)) {
    elems.push_back(item);
  }

  const auto str_size = str.size();
  if ((str_size > 0UL) && (str[str_size - 1UL] == delim)) {
    elems.emplace_back("");
  }

  return elems;
}

static void GetOpsProtoSoFileList(const std::string &path, std::vector<std::string> &file_list) {
  // Support multi lib directory with ":" as delimiter
  const std::vector<std::string> v_path = SplitStr(path, ':');

  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);

  for (size_t i = 0UL; i < v_path.size(); ++i) {
    const std::string new_path = v_path[i] + "lib/" + os_type + "/" + cpu_type + "/";
    char_t resolved_path[MMPA_MAX_PATH] = {};
    const INT32 result = mmRealPath(new_path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH);
    if (result == EN_OK) {
      std::vector<std::string> file_list_unfiltered;
      PluginManager::GetFileListWithSuffix(new_path, ".so", file_list_unfiltered);
      std::for_each(file_list_unfiltered.begin(), file_list_unfiltered.end(), [&file_list](const std::string &file) {
        if (!PluginManager::IsEndWith(file, "rt2.0.so") && !PluginManager::IsEndWith(file, "rt.so")) {
          file_list.emplace_back(file);
        }
      });
    } else {
      GELOGW("[FindSo][Check] Get path with os&cpu type [%s] unsuccessfully, reason:%s", new_path.c_str(),
             strerror(errno));
      PluginManager::GetFileListWithSuffix(v_path[i], ".so", file_list);
    }
  }
}

void OpsProtoManager::LoadOpsProtoPluginSo(const std::string &path) {
  if (path.empty()) {
    REPORT_INNER_ERR_MSG("E18888", "filePath is empty. please check your text file.");
    GELOGE(GRAPH_FAILED, "[Check][Param] filePath is empty. please check your text file.");
    return;
  }
  GELOGW(
      "[LoadSo][Check] Shared library will not be checked. Please make sure that the source of shared library is "
      "trusted.");
  OperatorFactoryImpl::SetRegisterOverridable(true);
  void *const handle = mmDlopen(path.c_str(), static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
                                                                   static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
  OperatorFactoryImpl::SetRegisterOverridable(false);
  if (handle == nullptr) {
    const char_t *error = mmDlerror();
    error = (error == nullptr) ? "" : error;
    GELOGW("[LoadSo][Open] OpsProtoManager dlopen unsuccessfully, plugin name:%s. Message(%s).", path.c_str(), error);
    return;
  }
  GELOGI("OpsProtoManager plugin load %s successfully.", path.c_str());
  handles_.push_back(handle);
}

void OpsProtoManager::LoadBuiltinOpsPluginSo(const std::string &path_list) {
  if (path_list.empty()) {
    REPORT_INNER_ERR_MSG("E18888", "filePath is empty. please check your text file.");
    GELOGE(GRAPH_FAILED, "[Check][Param] filePath is empty. please check your text file.");
    return;
  }
  std::vector<std::string> file_list;

  // If there is .so file in the lib path
  GetOpsProtoSoFileList(path_list, file_list);

  // Not found any .so file in the lib path
  if (file_list.empty()) {
    GELOGW("[LoadSo][Check] OpsProtoManager can not find any plugin file in pluginPath: %s", path_list.c_str());
    return;
  }
  // Warning message
  GELOGW(
      "[LoadSo][Check] Shared library will not be checked. Please make sure that the source of shared library is "
      "trusted.");

  // Load .so file
  for (const auto &elem : file_list) {
    LoadOpsProtoPluginSo(elem);
  }
}
}  // namespace af
