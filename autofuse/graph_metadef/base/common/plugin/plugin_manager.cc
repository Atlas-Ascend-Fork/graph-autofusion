/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_metadef/common/plugin/plugin_manager.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <regex>
#include <string>
#include <sys/stat.h>

#include "common/checker.h"
#include "common/ge_common/debug/ge_log.h"
#include "graph_metadef/common/ge_common/util.h"
#include "graph/def_types.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "common/ge_common/string_util.h"

namespace af {
namespace {
constexpr int32_t kMaxNumOfSo = 64;
constexpr int32_t kMaxSizeOfSo = 838860800;         // = 800M(unit is Byte)
constexpr int32_t kMaxSizeOfLoadedSo = 1048576000;  // = 1000M(unit is Byte)
const std::string kExt = ".so";                     // supported extension of shared object
const std::string kBuiltIn = "built-in";            // opp built-in directory name
const std::string kVendors = "vendors";             // opp vendors directory name
const std::string kConfig = "config.ini";           // opp vendors config file name
constexpr size_t kVendorConfigPartsCount = 2U;
const std::string kHostCpuLibRelativePathV01 = "/op_impl/built-in/host_cpu";
const std::string kHostCpuLibRelativePathV02 = "/built-in/op_impl/host_cpu";
constexpr size_t kMaxErrorStrLen = 128U;
constexpr uint32_t kLibFirstLayer = 0U;
constexpr uint32_t kLibSecondLayer = 1U;
const std::string kOppPath = "opp/";
const std::string kRuntimePath = "runtime/";
const std::string kCompilerPath = "compiler/";
const std::string kScene = "scene.info";
constexpr size_t kSceneValueCount = 2U;
constexpr size_t kSceneKeyIndex = 0U;
constexpr size_t kSceneValueIndex = 1U;
const std::string kSceneOs = "os";
const std::string kSceneArch = "arch";
const std::string kRtSoSuffix = "rt.so";
const std::string kRequiredOppAbiVersion = "required_opp_abi_version=";
const std::string kCompilerVersion = "compiler_version=";
const std::string kVersionInfo = "/version.info";
const std::string kOppVersion = "Version=";
constexpr size_t kEffectiveVersionNum = 2U;
const std::string kOpMasterDeviceLib = "/op_impl/ai_core/tbe/op_master_device/lib/";
const std::string kOpTilingDeviceLib = "/op_impl/ai_core/tbe/op_tiling_device/lib/";
const std::string kOppLatest = "opp_latest";

std::string TransRequiredOppAbiVersionToString(const std::vector<std::pair<uint32_t, uint32_t>> &required_version) {
  std::stringstream ss;
  for (const auto &it : required_version) {
    ss << "[" << it.first << ", " << it.second << "]";
  }
  return ss.str();
}

bool IsVersionWithInRequiredRange(const uint32_t effective_version,
                                  const std::vector<std::pair<uint32_t, uint32_t>> &required_version) {
  for (const auto &it : required_version) {
    if ((effective_version >= it.first) && (effective_version <= it.second)) {
      GELOGD("[ValidVersion] Effective version:%u within the required range.", effective_version);
      return true;
    }
  }

  GELOGW("[InvalidVersion] Effective version:%u not within the required range.", effective_version);
  return false;
}

// GCC has a bug with std::regex in versions after 12, so we use std::string::find instead of std::regex.
// See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105562
std::string ReplaceFirst(const std::string &str, const std::string &from, const std::string &to) {
  const size_t start_pos = str.find(from);
  if (start_pos == std::string::npos) {
    return str;  // 如果找不到 %s，就返回原字符串
  }
  return str.substr(0, start_pos) + to + str.substr(start_pos + from.length());
}

// 过滤函数：过滤掉 . 和 .. 目录，并且只保留目录条目
int32_t FilterDirectories(const mmDirent *entry) {
  if ((strcmp(entry->d_name, ".") == 0) || (strcmp(entry->d_name, "..") == 0)) {
    return 0;
  }
  return 1;
}

int32_t FilterSoFiles(const mmDirent *entry) {
  std::string name = entry->d_name;
  const std::string so_extension = ".so";
  if (name.size() > so_extension.size() && name.substr(name.size() - so_extension.size()) == so_extension) {
    return 1;
  }
  if (name != "." && name != "..") {
    GELOGI("Skipping file: %s", name.c_str());
  }
  return 0;
}

void ProcessSubdirectoryAndSoFiles(const std::string &directory, const mmDirent *dir_ent,
                                   std::vector<std::string> &so_files) {
  const std::string subdir = directory + "/" + std::string(dir_ent->d_name);
  const std::string custom_fusion_passes_dir = subdir + "/custom_fusion_passes";
  if (mmIsDir(custom_fusion_passes_dir.c_str()) != EN_OK) {
    GELOGI("The custom fusion passes directory: %s does not exist or has no permission.",
           custom_fusion_passes_dir.c_str());
    return;
  }

  mmDirent **so_entries = nullptr;
  const int32_t so_dir_num = mmScandir(custom_fusion_passes_dir.c_str(), &so_entries, FilterSoFiles, alphasort);
  GELOGI("Scanning subdirectory: %s, subdirectory num: %d", custom_fusion_passes_dir.c_str(), so_dir_num);
  if (so_dir_num <= 0) {
    return;
  }

  for (int32_t j = 0; j < so_dir_num; ++j) {
    const mmDirent *const so_dir_ent = so_entries[static_cast<size_t>(j)];
    if (so_dir_ent == nullptr) {
      continue;
    }
    const std::string path = custom_fusion_passes_dir + "/" + std::string(so_dir_ent->d_name);
    so_files.push_back(path);
    GELOGI("Found custom pass library file: %s", path.c_str());
  }
  mmScandirFree(so_entries, so_dir_num);
}
}  // namespace

void PluginManager::ClearHandles_() noexcept {
  for (const auto &handle : handles_) {
    if (mmDlclose(handle.second) != 0) {
      const char_t *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGW("Failed to close handle of %s, errmsg:%s", handle.first.c_str(), error);
    }
  }
  handles_.clear();
}

PluginManager::~PluginManager() {
  ClearHandles_();
}

Status PluginManager::GetOppPath(std::string &opp_path) {
  GELOGI("Enter get opp path schedule");
  const char_t *path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, path_env);
  if ((path_env != nullptr) && (strlen(path_env) > 0U)) {
    opp_path = path_env;
    std::string file_path = RealPath(opp_path.c_str());
    if (file_path.empty()) {
      GELOGW("[Call][RealPath] File path %s is invalid.", opp_path.c_str());
    } else {
      GELOGI("Get opp path from env: %s", opp_path.c_str());
    }
    if (opp_path.back() != '/') {
      opp_path += '/';
    }
  }
  if (opp_path.empty()) {
    opp_path = GetModelPath();
    GELOGI("Get opp path from model path, value is %s", opp_path.c_str());
    opp_path = opp_path.substr(0, opp_path.rfind('/'));
    opp_path = opp_path.substr(0, opp_path.rfind('/') + 1U);
    opp_path += "ops/";
  }
  GELOGD("Leave get opp path schedule, opp_path:%s", opp_path.c_str());
  return SUCCESS;
}

Status PluginManager::GetUpgradedOppPath(std::string &opp_path) {
  GELOGI("Enter get upgraded opp path schedule");
  const char_t *path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, path_env);
  if ((path_env != nullptr) && (strlen(path_env) > 0U)) {
    opp_path = path_env;
    opp_path += "/opp_latest/";
    std::string file_path = RealPath(opp_path.c_str());
    if (!file_path.empty()) {
      GELOGI("Get upgraded opp path from env: %s", opp_path.c_str());
      return SUCCESS;
    }
    GELOGW("[Call][RealPath] Opp path %s is invalid.", opp_path.c_str());
  }
  return FAILED;
}

bool PluginManager::IsNewOppPathStruct(const std::string &opp_path) {
  return mmIsDir((opp_path + kBuiltIn).c_str()) == EN_OK;
}

Status PluginManager::GetOppPluginVendors(const std::string &vendors_config, std::vector<std::string> &vendors) {
  GELOGI("Enter get opp plugin config file schedule, config file is '%s'", vendors_config.c_str());
  std::ifstream config(vendors_config);
  if (!config.good()) {
    GELOGI("Can not open file '%s'!", vendors_config.c_str());
    return FAILED;
  }
  std::string content;
  (void)std::getline(config, content);
  config.close();
  GE_ASSERT_TRUE(!content.empty(), "Content of file '%s' is empty!", vendors_config.c_str());
  std::vector<std::string> v_parts;
  SplitPath(content, v_parts, '=');
  GE_ASSERT_TRUE(v_parts.size() == kVendorConfigPartsCount,
                 "Content of file '%s' has invalid format, expect 'vendor=path', actual: '%s'.", vendors_config.c_str(),
                 content.c_str());
  SplitPath(v_parts[1], vendors, ',');
  GE_ASSERT_TRUE(!vendors.empty(), "Format of file content is invalid!");
  (void)for_each(vendors.begin(), vendors.end(), &StringUtils::Trim);
  return SUCCESS;
}

Status PluginManager::ReversePathString(std::string &path_str) {
  GELOGI("Enter ReversePathString schedule");
  if (path_str.empty() || (path_str.find(":") == std::string::npos)) {
    return SUCCESS;
  }
  std::vector<std::string> path_vec;
  SplitPath(path_str, path_vec, ':');
  GE_ASSERT_TRUE(!path_vec.empty(), "The vector path_vec should not be empty!");
  auto it = path_vec.crbegin();
  path_str = *(it++);
  while (it != path_vec.crend()) {
    path_str += ":" + *(it++);
  }
  GELOGI("path_str is '%s'", path_str.c_str());
  return SUCCESS;
}

// 当前自定义算子路径不能直接从环境变量获取，需要通过op_register.cc中PreProcessForCustomOp初始化后
// 才可以由对外接口aclGetCustomOpLibPath获取。
// 而aclGetCustomOpLibPath当前是编译在libregister.so中，plugin_manager.cc则在更基础的libgraph_base.so中，
// 所以此处采用一个static string保存，避免反向依赖。
std::string PluginManager::custom_op_lib_path_ = {};
void PluginManager::SetCustomOpLibPath(const std::string &custom_op_Lib_path) {
  custom_op_lib_path_ = custom_op_Lib_path;
}

void PluginManager::GetPluginPathFromCustomOppPath(const std::string &sub_path, std::string &plugin_path) {
  plugin_path = "";
  if (custom_op_lib_path_.empty()) {
    GELOGI("custom_op_lib_path_ is empty.");
    return;
  }
  GELOGI("value of env custom_op_lib_path_ is %s.", custom_op_lib_path_.c_str());
  std::vector<std::string> custom_paths = StringUtils::Split(custom_op_lib_path_, ':');
  for (const auto &custom_path : custom_paths) {
    if ((!custom_path.empty()) && (mmIsDir((custom_path + "/" + sub_path).c_str()) == EN_OK)) {
      if (IsVendorVersionValid(custom_path)) {
        plugin_path += custom_path + "/" + sub_path + ":";
        GELOGI("custom_path '%s' is valid.", custom_path.c_str());
      }
    } else {
      GELOGI("custom_path '%s' is invalid, which is skipped.", custom_path.c_str());
    }
  }
  GELOGI("Run GetPluginPathFromCustomOppPath finished, current plugin_path is %s.", plugin_path.c_str());
}

Status PluginManager::GetOppPluginPathOld(const std::string &opp_path, const std::string &path_fmt,
                                          std::string &plugin_path, const std::string &path_fmt_custom) {
  GELOGI("Enter get opp plugin path old schedule");
  const std::string &fmt_custom = path_fmt_custom.empty() ? path_fmt : path_fmt_custom;
  plugin_path = (opp_path + ReplaceFirst(fmt_custom, "%s", "custom") + ":") +
                (opp_path + ReplaceFirst(path_fmt, "%s", "built-in"));
  GELOGI("plugin_path is '%s'", plugin_path.c_str());
  return SUCCESS;
}

bool PluginManager::GetRequiredOppAbiVersion(std::vector<std::pair<uint32_t, uint32_t>> &required_opp_abi_version) {
  std::string model_path = GetModelPath();
  GELOGD("Current lib path is:%s", model_path.c_str());
  model_path = model_path.substr(0, model_path.rfind('/'));
  model_path = model_path.substr(0, model_path.rfind('/'));
  model_path = model_path.substr(0, model_path.rfind('/') + 1);
  GELOGD("Run package path is:%s", model_path.c_str());

  std::string version_path;
  if (mmIsDir((model_path + kCompilerPath).c_str()) == EN_OK) {
    version_path = model_path + kCompilerPath + kVersionInfo;
  } else if (mmIsDir((model_path + kRuntimePath).c_str()) == EN_OK) {
    version_path = model_path + kRuntimePath + kVersionInfo;
  } else {
    GELOGW("compiler and runtime do not exist");
    return true;
  }
  GELOGI("extract required opp abi version info from %s", version_path.c_str());

  std::string version;
  if (!PluginManager::GetVersionFromPathWithName(version_path, version, kRequiredOppAbiVersion)) {
    GELOGW("Failed to get required_opp_abi_version from path:%s", version_path.c_str());
    return true;
  }

  // valid required_opp_abi_version: ">=6.3, <=6.4, 6.4"
  version = StringUtils::ReplaceAll(version, "\"", "");
  (void)StringUtils::Trim(version);
  std::queue<std::string> split_version;
  for (auto &it : StringUtils::Split(version, ',')) {
    (void)split_version.emplace(StringUtils::Trim(it));
  }
  while (!split_version.empty()) {
    auto first = split_version.front();
    split_version.pop();
    if (StringUtils::StartWith(first, ">=")) {
      if (split_version.empty() || !StringUtils::StartWith(split_version.front(), "<=")) {
        GELOGW("Format of required_opp_abi_version [%s] is invalid, start with >= but not end with <=",
               version.c_str());
        return false;
      }
      auto second = split_version.front();
      split_version.pop();
      first = first.substr(kEffectiveVersionNum, first.size() - kEffectiveVersionNum);
      second = second.substr(kEffectiveVersionNum, second.size() - kEffectiveVersionNum);
      uint32_t first_num = 0U;
      if (!GetEffectiveVersion(first, first_num)) {
        GELOGW("[InvalidVersion] Format of required_opp_abi_version [%s] is invalid", version.c_str());
        return false;
      }
      uint32_t second_num = 0U;
      if (!GetEffectiveVersion(second, second_num)) {
        GELOGW("[InvalidVersion] Format of required_opp_abi_version [%s] is invalid", version.c_str());
        return false;
      }
      (void)required_opp_abi_version.emplace_back(first_num, second_num);
    } else {
      uint32_t tmp_num = 0U;
      if (!GetEffectiveVersion(first, tmp_num)) {
        GELOGW("[InvalidVersion] Format of required_opp_abi_version [%s] is invalid", version.c_str());
        return false;
      }
      (void)required_opp_abi_version.emplace_back(tmp_num, tmp_num);
    }
  }
  return true;
}

bool PluginManager::GetEffectiveVersion(const std::string &opp_version, uint32_t &effective_version) {
  auto split_version = StringUtils::Split(opp_version, '.');
  GE_ASSERT_TRUE(split_version.size() >= kEffectiveVersionNum);
  std::stringstream ss;
  ss << split_version[0];  // Cann version
  const size_t c_effective_version_max_num = 5;
  const size_t c_effective_version_size = split_version[1].size();
  if (c_effective_version_max_num > c_effective_version_size) {
    // C version 的数字个数要拉齐，不然配 3.20~9.0 之类的会取到空集
    split_version[1].append(c_effective_version_max_num - c_effective_version_size, '0');
  } else {
    GELOGW(
        "c_effective_version_size:%zu, will reach or exceed max num:%zu, Ensure that the range of valid version is not "
        "empty",
        c_effective_version_size, c_effective_version_max_num);
  }
  ss << split_version[1];  // C version
  ss >> effective_version;
  if (ss.fail() || !ss.eof()) {
    GELOGW("Can not convert [%s] to number from %s", ss.str().c_str(), opp_version.c_str());
    return false;
  }
  GELOGD("Get effective version:%u from %s", effective_version, opp_version.c_str());
  return true;
}

bool PluginManager::IsVendorVersionValid(const std::string &vendor_path) {
  // opp_kernel包支持独立升级，不进行版本号校验
  GELOGD("CheckVersionValid, vendor_path:%s", vendor_path.c_str());
  if (vendor_path.find(kOppLatest) != std::string::npos) {
    GELOGW("Will not verify version for [%s] as the opp kernel is upgraded independently.", vendor_path.c_str());
    return true;
  }

  // 获取opp包版本号：内置算子包字段为opp_version， 自定义算子包字段为compiler_version
  std::string opp_version;
  std::string compiler_version;
  GetOppAndCompilerVersion(vendor_path, opp_version, compiler_version);
  if (opp_version.empty() && compiler_version.empty()) {
    GELOGW("[NotVerification] Will not verify version as the opp version and compiler version are not set");
    return true;
  }
  return IsVendorVersionValid(opp_version, compiler_version);
}

bool PluginManager::IsVendorVersionValid(const std::string &opp_version, const std::string &compiler_version) {
  // 获取compiler或者runtime包支持的版本号范围
  std::vector<std::pair<uint32_t, uint32_t>> required_opp_abi_version;
  GE_ASSERT_TRUE(GetRequiredOppAbiVersion(required_opp_abi_version), "Get required opp abi version failed.");
  if (required_opp_abi_version.empty()) {
    GELOGW("[NotVerification] Will not verify version as the required_opp_abi_version are not set");
    return true;
  }

  // 校验版本号是否在支持的版本号列表范围内
  return CheckOppAndCompilerVersions(opp_version, compiler_version, required_opp_abi_version);
}

void PluginManager::GetOppAndCompilerVersion(const std::string &vendor_path, std::string &opp_version,
                                             std::string &compiler_version) {
  std::string version_path;
  if (vendor_path.find(kBuiltIn) != std::string::npos) {
    version_path = vendor_path.substr(0, vendor_path.rfind("/")) + kVersionInfo;
    (void)PluginManager::GetVersionFromPathWithName(version_path, opp_version, kOppVersion);
    GELOGD("Get opp_version:%s", opp_version.c_str());
  } else {
    version_path = vendor_path + kVersionInfo;
    (void)PluginManager::GetVersionFromPathWithName(version_path, compiler_version, kCompilerVersion);
    GELOGD("Get compiler_version:%s", compiler_version.c_str());
  }
  return;
}

bool PluginManager::CheckOppAndCompilerVersions(const std::string &opp_version, const std::string &compiler_version,
                                                const std::vector<std::pair<uint32_t, uint32_t>> &required_version) {
  if (!opp_version.empty()) {
    uint32_t effective_opp_version = 0U;
    if (!GetEffectiveVersion(opp_version, effective_opp_version)) {
      GELOGW("[InvalidVersion] Format of opp version [%s] is invalid", opp_version.c_str());
      return false;
    }
    if (!IsVersionWithInRequiredRange(effective_opp_version, required_version)) {
      GELOGW("opp_version:%s is not within required_opp_abi_version:%s", opp_version.c_str(),
             TransRequiredOppAbiVersionToString(required_version).c_str());
      return false;
    }
  }

  if (!compiler_version.empty()) {
    for (const auto &it : StringUtils::Split(compiler_version, ',')) {
      uint32_t effective_compiler_version = 0U;
      if (!GetEffectiveVersion(it, effective_compiler_version)) {
        GELOGW("[InvalidVersion] Format of compiler version [%s] is invalid", it.c_str());
        return false;
      }
      if (!IsVersionWithInRequiredRange(effective_compiler_version, required_version)) {
        GELOGW("compiler version:%s is not within required_opp_abi_version:%s", it.c_str(),
               TransRequiredOppAbiVersionToString(required_version).c_str());
        return false;
      }
    }
  }

  GELOGD("[ValidVersion] opp version:%s and compiler version:%s are within the required range:%s", opp_version.c_str(),
         compiler_version.c_str(), TransRequiredOppAbiVersionToString(required_version).c_str());
  return true;
}

// 需要打包进om的so优先级：
// 1. ASCEND_CUSTOM_OPP_PATH
// 2. ASCEND_OPP_PATH + vendors + 厂商名
// 3. ASCEND_OPP_PATH + built-in
void PluginManager::GetPackageSoPath(std::vector<std::string> &vendors) {
  std::string custom_opp_path;
  PluginManager::GetPluginPathFromCustomOppPath("", custom_opp_path);
  if (!custom_opp_path.empty()) {
    std::vector<std::string> split_custom_opp_path = StringUtils::Split(custom_opp_path, ':');
    (void)vendors.insert(vendors.end(), split_custom_opp_path.begin(), split_custom_opp_path.end());
  }

  std::string opp_path;
  if (PluginManager::GetOppPath(opp_path) == SUCCESS) {
    std::string vendors_path;
    (void)PluginManager::GetOppPluginPathNew(opp_path, "%s", vendors_path, "");
    if (!vendors_path.empty()) {
      auto split_vendors_path = StringUtils::Split(vendors_path, ':');
      (void)vendors.insert(vendors.end(), split_vendors_path.begin(), split_vendors_path.end());
    }
  }
  return;
}

Status PluginManager::GetOppPluginPathNew(const std::string &opp_path, const std::string &path_fmt,
                                          std::string &plugin_path, const std::string &old_custom_path,
                                          const std::string &path_fmt_custom) {
  GELOGI("Enter get opp plugin path new schedule");
  const std::string vendors_config = opp_path + kVendors + "/" + kConfig;
  std::vector<std::string> vendors;
  if (GetOppPluginVendors(vendors_config, vendors) != SUCCESS) {
    GELOGI("Can not get opp plugin vendors!");
    plugin_path += opp_path + old_custom_path + ":";
  } else {
    const std::string &fmt_custom = path_fmt_custom.empty() ? path_fmt : path_fmt_custom;
    for (const auto &vendor : vendors) {
      if (IsVendorVersionValid(opp_path + kVendors + "/" + vendor)) {
        plugin_path += opp_path + kVendors + "/" + ReplaceFirst(fmt_custom, "%s", vendor) + ":";
      }
    }
  }
  if (IsVendorVersionValid(opp_path + "/" + kBuiltIn)) {
    plugin_path += opp_path + ReplaceFirst(path_fmt, "%s", "built-in");
  }
  GELOGI("plugin_path is '%s'", plugin_path.c_str());
  return SUCCESS;
}

bool PluginManager::IsSplitOpp() {
  std::string opp_path;
  if (GetOppPath(opp_path) != SUCCESS) {
    GELOGE(af::FAILED, "Failed to get opp path:[%s]", opp_path.c_str());
    return false;
  }
  std::string os_type;
  std::string cpu_type;
  GetCurEnvPackageOsAndCpuType(os_type, cpu_type);

  std::vector<std::string> so_list;
  std::vector<std::string> path_vec;
  opp_path += "built-in/";
  (void)path_vec.emplace_back(opp_path + "op_proto/lib/" + os_type + "/" + cpu_type + "/");
  (void)path_vec.emplace_back(opp_path + "op_impl/ai_core/tbe/op_tiling/lib/" + os_type + "/" + cpu_type + "/");
  for (const auto &path : path_vec) {
    GetFileListWithSuffix(path, kRtSoSuffix, so_list);
    if (so_list.empty()) {
      GELOGI("Split opp package:[%s] does not exist", path.c_str());
      return false;
    }
    so_list.clear();
  }
  return true;
}

/**
 * 获取`ops proto`目录，按照优先级，中间使用冒号隔开返回，目录和优先级规则：
 *
 * 有两种目录格式，新目录和老目录，
 *
 * **老目录的规范：**
 *
 * 1. 内置算子包路径： `<opp-path>/op_proto/built-in/`
 * 2. 自定义算子包路径：`<opp-path>/op_proto/custom/`
 *
 * **新目录的规范：**
 *
 * 1. 内置算子包路径： `<opp-path>/built-in/op_proto`
 * 2. 自定义算子包路径： `<opp-path>/vendors/<name>/op_proto`
 *
 * 在新目录规范中，`<name>` 是自定义算子包的名称，可以是任意字符串，但是仅限一层，即不能包含 `/` 字符。
 * 新目录自定义算子包有两种配置方式：
 *
 * 1 配置在`<opp-path>/vendors/config.ini`文件中，
 *   在文件中，`<name>`列表按照**优先级**、以逗号分隔，例如该文件内容为：
 *
 * ```ini
 * load_priority=customize,mdc,lhisi
 * ```
 *
 * 代表有三个自定义算子包目录，按照优先级由高到底，分别为
 *
 * 1. `<opp-path>/vendors/customize/op_proto`
 * 2. `<opp-path>/vendors/mdc/op_proto`
 * 3. `<opp-path>/vendors/lhisi/op_proto`
 *
 * 2 配置在`ASCEND_CUSTOM_OPP_PATH`环境变量中，
 *   在环境变量中，`<custom-opp-path>`列表按照**优先级**、以冒号分隔，例如该环境变量为：
 *
 * ```shell
 * ASCEND_CUSTOM_OPP_PATH=/path/to/customize:/path/to/mdc:/path/to/lhisi
 * ```
 * 当前最新的自定义算子工程交付分为run包交付和so交付（新做的）两种形式：
 * 新的so交付的形式下：export ASCEND_CUSTOM_OPP_PATH=/path/to/customize:/path/to/mdc:/path/to/lhisi
 * 三个目录下都只有一个libcust_opapi.so
 *
 * 老的run包交付的形式下：export ASCEND_CUSTOM_OPP_PATH=/path/to/customize:/path/to/mdc:/path/to/lhisi
 * 三个目录下都有完整的算子子目录，如op_proto,op_impl子目录等
 *
 * 当前不支持两种交付方式混用，混用校验报错，具体逻辑考op_lib_regiser.cc中PreProcessForCustomOp实现
 *
 * 代表有三个自定义算子包目录，按照优先级由高到底，分别为
 *
 * 1. `/path/to/customize/op_proto`
 * 2. `/path/to/mdc/op_proto`
 * 3. `/path/to/lhisi/op_proto`
 *
 * 不论新老哪种目录规范，内置算子包的优先级永远最低。
 *
 * **新老目录、多种配置方式混合场景：**
 *
 * 支持新老风格目录的算子包混合使用，新目录自定义算子包支持环境变量和配置文件两种方式混用。
 * 混用存在场景约束：**内置算子包必须为新目录风格**，如果出现老内置算子包，新目录的自定义算子包会被忽略。
 * 因此，可混用的范围有：新内置、新配置文件的自定义、新环境变量的自定义、老自定义四种情况。
 * 这四种情况中，新配置文件的自定义和老自定义两种二选一，不可同时存在，具体来说，如果新配置文件的自定义存在，老自定义会被忽略。
 *
 * 以上四种混用时，优先级由高到底，规则依次为：
 *
 * 1. 如果有新环境变量自定义算子包，本组优先级最高，组内按照`ASCEND_CUSTOM_OPP_PATH`中顺序排序
 * 2. 如果有新配置文件自定义算子包，本组优先级第二，组内按照`<opp-path>/vendors/config.ini`中的优先级排序
 * 3. 如果没有新配置文件自定义算子包、而且老自定义算子包存在，老自定义算子包优先级第二
 * 4. 新内置算子包的优先级最低
 * @param opsproto_path
 * @return
 */
Status PluginManager::GetOpsProtoPath(std::string &opsproto_path) {
  Status ret = SUCCESS;
  std::string opp_path;
  GE_ASSERT_TRUE(GetOppPath(opp_path) == SUCCESS, "Failed to get opp path!");
  if (!IsNewOppPathStruct(opp_path)) {
    ret = GetOppPluginPathOld(opp_path, "op_proto/%s/", opsproto_path);
    GELOGI("Opp plugin path structure is old version! opsproto_path:%s", opsproto_path.c_str());
    return ret;
  } else {
    GetPluginPathFromCustomOppPath("op_proto/", opsproto_path);
    ret = GetOppPluginPathNew(opp_path, "%s/op_proto/", opsproto_path, "op_proto/custom/");
    GELOGI("Opp plugin path structure is new version! opsproto_path:%s", opsproto_path.c_str());
    return ret;
  }
}

Status PluginManager::GetUpgradedOpsProtoPath(std::string &opsproto_path) {
  GELOGI("Start to get upgraded ops proto path");
  std::string upgraded_opp_path;
  if (GetUpgradedOppPath(upgraded_opp_path) != af::SUCCESS) {
    GELOGW("Failed to get upgraded opp path!");
    return af::FAILED;
  }
  return GetOppPluginPathNew(upgraded_opp_path, "%s/op_proto", opsproto_path, "");
}

Status PluginManager::GetUpgradedOpMasterPath(std::string &op_tiling_path) {
  GELOGI("Start to get upgraded op tiling path");
  std::string upgraded_opp_path;
  if (GetUpgradedOppPath(upgraded_opp_path) != af::SUCCESS) {
    GELOGW("Failed to get upgraded opp path!");
    return af::FAILED;
  }
  return GetOppPluginPathNew(upgraded_opp_path, "%s/op_impl/ai_core/tbe", op_tiling_path, "");
}

Status PluginManager::GetCustomOpPath(const std::string &fmk_type, std::string &customop_path) {
  GELOGI("Enter GetCustomOpPath schedule");
  std::string opp_path;
  GE_ASSERT_TRUE(GetOppPath(opp_path) == SUCCESS, "Failed to get opp path!");
  if (!IsNewOppPathStruct(opp_path)) {
    GELOGI("Opp plugin path structure is old version!");
    return GetOppPluginPathOld(opp_path, "framework/%s/" + fmk_type + "/", customop_path, "framework/%s/");
  } else {
    GELOGI("Opp plugin path structure is new version!");
    GetPluginPathFromCustomOppPath("framework/", customop_path);
    return GetOppPluginPathNew(opp_path, "%s/framework/" + fmk_type + "/", customop_path, "framework/custom/",
                               "%s/framework/");
  }
}

Status PluginManager::GetCustomCaffeProtoPath(std::string &customcaffe_path) {
  GELOGD("Enter GetCustomCaffeProtoPath schedule");
  std::string opp_path;
  GE_ASSERT_TRUE(GetOppPath(opp_path) == SUCCESS, "Failed to get opp path!");
  if (!IsNewOppPathStruct(opp_path)) {
    customcaffe_path = opp_path + "framework/custom/caffe/";
    GELOGI("Opp plugin path structure is old version! customcaffe_path is '%s'", customcaffe_path.c_str());
    return SUCCESS;
  } else {
    GELOGI("Opp plugin path structure is new version!");
    GetPluginPathFromCustomOppPath("framework/caffe/", customcaffe_path);
    const std::string vendors_config = opp_path + kVendors + "/" + kConfig;
    std::vector<std::string> vendors;
    if (GetOppPluginVendors(vendors_config, vendors) != SUCCESS) {
      GELOGI("Can not get opp plugin vendors!");
      customcaffe_path += opp_path + "framework/custom/caffe/";
    } else {
      for (const auto &vendor : vendors) {
        if ((!customcaffe_path.empty()) && (customcaffe_path.back() != ':')) {
          customcaffe_path += ":";
        }
        customcaffe_path += opp_path + kVendors + "/" + vendor + "/framework/caffe/";
      }
    }
    GELOGI("customcaffe_path is '%s'", customcaffe_path.c_str());
    return SUCCESS;
  }
}

Status PluginManager::GetOpTilingForwardOrderPath(std::string &op_tiling_path) {
  GELOGI("Enter GetOpTilingPath schedule");
  std::string opp_path;
  GE_ASSERT_TRUE(GetOppPath(opp_path) == SUCCESS, "Failed to get opp path!");
  if (!IsNewOppPathStruct(opp_path)) {
    GELOGI("Opp plugin path structure is old version!");
    GE_ASSERT_TRUE(GetOppPluginPathOld(opp_path, "op_impl/%s/ai_core/tbe/", op_tiling_path) == SUCCESS,
                   "GetOppPluginPathOld failed!");
  } else {
    GELOGI("Opp plugin path structure is new version!");
    GetPluginPathFromCustomOppPath("op_impl/ai_core/tbe/", op_tiling_path);
    GE_ASSERT_TRUE(GetOppPluginPathNew(opp_path, "%s/op_impl/ai_core/tbe/", op_tiling_path,
                                       "op_impl/custom/ai_core/tbe/") == SUCCESS,
                   "GetOppPluginPathNew failed!");
  }
  return SUCCESS;
}

Status PluginManager::GetOpTilingPath(std::string &op_tiling_path) {
  GE_ASSERT_SUCCESS(GetOpTilingForwardOrderPath(op_tiling_path));
  return ReversePathString(op_tiling_path);
}

Status PluginManager::GetConstantFoldingOpsPath(const std::string &path_base, std::string &constant_folding_ops_path) {
  GELOGI("Enter GetConstantFoldingOpsPath schedule");
  std::string opp_path;
  const Status ret = GetOppPath(opp_path);
  if (ret != SUCCESS) {
    GELOGW("Failed to get opp path from env and so file! use path_base as opp path");
    opp_path = path_base;
  }
  GE_ASSERT_TRUE(!opp_path.empty(), "[Check]Value of opp_path should not be empty here!");
  if (!IsNewOppPathStruct(opp_path)) {
    constant_folding_ops_path = opp_path + kHostCpuLibRelativePathV01;
  } else {
    constant_folding_ops_path = opp_path + kHostCpuLibRelativePathV02;
  }
  return SUCCESS;
}

void PluginManager::SplitPath(const std::string &mutil_path, std::vector<std::string> &path_vec, const char sep) {
  const std::string tmp_string = mutil_path + sep;
  std::string::size_type start_pos = 0U;
  std::string::size_type cur_pos = tmp_string.find(sep, 0U);
  while (cur_pos != std::string::npos) {
    const std::string path = tmp_string.substr(start_pos, cur_pos - start_pos);
    if (!path.empty()) {
      path_vec.push_back(path);
    }
    start_pos = cur_pos + 1U;
    cur_pos = tmp_string.find(sep, start_pos);
  }
}

Status PluginManager::LoadSo(const std::string &path, const std::vector<std::string> &func_check_list) {
  constexpr int32_t flags =
      static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(MMPA_RTLD_GLOBAL));
  return LoadSoWithFlags(path, flags, func_check_list);
}

Status PluginManager::LoadSoWithFlags(const std::string &path, const int32_t flags,
                                      const std::vector<std::string> &func_check_list) {
  uint32_t num_of_loaded_so = 0U;
  int64_t size_of_loaded_so = 0;
  so_list_.clear();
  ClearHandles_();

  std::vector<std::string> path_vec;
  SplitPath(path, path_vec);
  for (const auto &single_path : path_vec) {
    GE_IF_BOOL_EXEC(single_path.length() >= static_cast<ULONG>(MMPA_MAX_PATH),
                    GELOGE(PARAM_INVALID, "The shared library file path is too long! path:%s", single_path.c_str());
                    continue);
    // load break when number of loaded so reach maximum
    if (num_of_loaded_so >= static_cast<uint32_t>(kMaxNumOfSo)) {
      GELOGW(
          "The number of dynamic libraries loaded exceeds the kMaxNumOfSo,"
          " and only the first %d shared libraries will be loaded.",
          kMaxNumOfSo);
      break;
    }

    std::string file_name = single_path.substr(single_path.rfind('/') + 1U, std::string::npos);
    const std::string file_path_dlopen = RealPath(single_path.c_str());
    if (file_path_dlopen.empty()) {
      GELOGW("Failed to get realpath of %s!", single_path.c_str());
      continue;
    }

    int64_t file_size = 0;
    if (ValidateSo(file_path_dlopen, size_of_loaded_so, file_size) != SUCCESS) {
      GELOGW("Failed to validate the shared library: %s", file_path_dlopen.c_str());
      continue;
    }

    GELOGI("dlopen path: %s, flags is %d", file_path_dlopen.c_str(), flags);

    // load continue when dlopen is failed
    const auto start = std::chrono::high_resolution_clock::now();
    const auto handle = mmDlopen(file_path_dlopen.c_str(), flags);
    const auto end = std::chrono::high_resolution_clock::now();
    GELOGI("[GEPERFTRACE] The time cost of PluginManager::Dlopen[%s] is [%lu] micro seconds.",
           (file_path_dlopen.c_str()), std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    if (handle == nullptr) {
      const char_t *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGW("[DLOpen][SharedLibraryPath]Failed, path[%s]. Message[%s]!", file_path_dlopen.c_str(), error);
      continue;
    }

    // load continue when so is invalid
    bool is_valid = true;
    for (const auto &func_name : func_check_list) {
      const auto real_fn = reinterpret_cast<void (*)()>(mmDlsym(handle, func_name.c_str()));
      if (real_fn == nullptr) {
        const char_t *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        REPORT_INNER_ERR_MSG("E19999", "[Check][So]%s is skipped since function %s does not exist! errmsg:%s",
                             file_name.c_str(), func_name.c_str(), error);
        GELOGE(PARAM_INVALID, "[Check][So]%s is skipped since function %s does not exist! errmsg:%s", file_name.c_str(),
               func_name.c_str(), error);
        is_valid = false;
        break;
      }
    }
    if (!is_valid) {
      if (mmDlclose(handle) != 0) {
        const char_t *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        GELOGE(FAILED, "[DLClose][Handle]Failed. errmsg:%s", error);
      }
      continue;
    }

    // add file to list
    size_of_loaded_so += file_size;
    (void)so_list_.emplace_back(file_name);
    handles_[std::string(file_name)] = handle;
    num_of_loaded_so++;
  }

  GELOGI("The total number of shared libraries loaded: %u", num_of_loaded_so);
  for (const auto &name : so_list_) {
    GELOGD("load %s successfully", name.c_str());
  }

  if (num_of_loaded_so == 0U) {
    GELOGW("No loadable shared library found in the path: %s", path.c_str());
    return SUCCESS;
  }

  return SUCCESS;
}

Status PluginManager::ValidateSo(const std::string &file_path, const int64_t size_of_loaded_so,
                                 int64_t &file_size) const {
  // read file size
  struct stat stat_buf;
  if (stat(file_path.c_str(), &stat_buf) != 0) {
    GELOGW("The shared library file check failed: %s", file_path.c_str());
    return FAILED;
  }

  // load continue when the size itself reaches maximum
  file_size = stat_buf.st_size;
  if (stat_buf.st_size > kMaxSizeOfSo) {
    GELOGW("The %s is skipped since its size exceeds maximum! (size: %ldB, maximum: %dB)", file_path.c_str(), file_size,
           kMaxSizeOfSo);
    return FAILED;
  }

  // load continue if the total size of so reaches maximum when it is loaded
  if ((size_of_loaded_so + file_size) > kMaxSizeOfLoadedSo) {
    GELOGW(
        "%s is skipped because the size of loaded shared library reaches maximum if it is loaded! "
        "(size: %ldB, size of loaded shared library: %ldB, maximum: %dB)",
        file_path.c_str(), file_size, size_of_loaded_so, kMaxSizeOfLoadedSo);
    return FAILED;
  }

  return SUCCESS;
}

Status PluginManager::Load(const std::string &path, const std::vector<std::string> &func_check_list) {
  constexpr int32_t flags =
      static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(MMPA_RTLD_GLOBAL));
  return LoadWithFlags(path, flags, func_check_list);
}

Status PluginManager::LoadWithFlags(const std::string &path, const int32_t flags,
                                    const std::vector<std::string> &func_check_list) {
  uint32_t num_of_loaded_so = 0U;
  int64_t size_of_loaded_so = 0;
  constexpr uint32_t is_folder = 0x4U;
  const std::string ext = kExt;
  so_list_.clear();
  ClearHandles_();

  char_t err_buf[kMaxErrorStrLen + 1U] = {};
  char_t canonical_path[MMPA_MAX_PATH] = {};
  if (mmRealPath(path.c_str(), &canonical_path[0], MMPA_MAX_PATH) != EN_OK) {
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLen);
    GELOGW("Failed to get realpath of %s, errmsg:%s", path.c_str(), err_msg);
    return SUCCESS;
  }

  const int32_t is_dir = mmIsDir(&canonical_path[0]);
  // Lib plugin path does not exist
  if (is_dir != EN_OK) {
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLen);
    GELOGW("Invalid path for load: %s, errmsg:%s", path.c_str(), err_msg);
    return SUCCESS;
  }

  mmDirent **entries = nullptr;
  const auto ret = mmScandir(&canonical_path[0], &entries, nullptr, nullptr);
  if (ret < EN_OK) {
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLen);
    GELOGW("scan dir failed. path = %s, ret = %d, errmsg = %s", &canonical_path[0], ret, err_msg);
    return FAILED;
  }
  for (int32_t i = 0; i < ret; ++i) {
    mmDirent *const entry = entries[i];
    // read fileName and fileType
    std::string file_name = entry->d_name;
    const auto file_type = static_cast<uint32_t>(entry->d_type);

    // ignore folder
    const bool invalid_file = ((file_type == is_folder) ||
                               // ignore file whose name length is less than 3
                               (file_name.size() <= ext.size()) ||
                               // ignore file whose extension is not so
                               (file_name.compare(file_name.size() - ext.size(), ext.size(), ext) != 0));
    if (invalid_file) {
      continue;
    }

    // load break when number of loaded so reach maximum
    if (num_of_loaded_so >= static_cast<uint32_t>(kMaxNumOfSo)) {
      GELOGW(
          "The number of dynamic libraries loaded exceeds the kMaxNumOfSo,"
          " and only the first %d shared libraries will be loaded.",
          kMaxNumOfSo);
      break;
    }
    const std::string canonical_path_str = (std::string(canonical_path) + "/" + file_name);
    const std::string file_path_dlopen = RealPath(canonical_path_str.c_str());
    if (file_path_dlopen.empty()) {
      GELOGW("failed to get realpath of %s", canonical_path_str.c_str());
      continue;
    }

    int64_t file_size = 0;
    if (ValidateSo(file_path_dlopen, size_of_loaded_so, file_size) != SUCCESS) {
      GELOGW("Failed to validate the shared library: %s", canonical_path_str.c_str());
      continue;
    }

    GELOGI("Dlopen path: %s. flags is %d, ensure that the source of the shared library is trusted",
           file_path_dlopen.c_str(), flags);

    // load continue when dlopen is failed
    const auto handle = mmDlopen(file_path_dlopen.c_str(), flags);
    if (handle == nullptr) {
      const char_t *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGW("Failed in dlopen %s!", error);
      continue;
    }

    // load continue when so is invalid
    bool is_valid = true;
    for (const auto &func_name : func_check_list) {
      const auto real_fn = reinterpret_cast<void (*)()>(mmDlsym(handle, func_name.c_str()));
      if (real_fn == nullptr) {
        const char_t *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        GELOGW("The %s is skipped since function %s does not exist! errmsg:%s", file_name.c_str(), func_name.c_str(),
               error);
        is_valid = false;
        break;
      }
    }
    if (!is_valid) {
      if (mmDlclose(handle) != 0) {
        const char_t *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        GELOGE(FAILED, "[DLClose][Handle]Failed. errmsg:%s", error);
      }
      continue;
    }

    // add file to list
    size_of_loaded_so += file_size;
    (void)so_list_.emplace_back(file_name);
    handles_[std::string(file_name)] = handle;
    num_of_loaded_so++;
  }
  mmScandirFree(entries, ret);
  if (num_of_loaded_so == 0U) {
    GELOGW("No loadable shared library found in the path: %s", path.c_str());
    return SUCCESS;
  }

  return SUCCESS;
}

void PluginManager::GetOppSupportedOsAndCpuType(
    std::unordered_map<std::string, std::unordered_set<std::string>> &opp_supported_os_cpu, std::string opp_path,
    std::string os_name, uint32_t layer) {
  if (layer > kLibSecondLayer) {
    GELOGW("The lib structure of the current opp package has only 2 layers");
    return;
  }
  GELOGD("Enter GetOppSupportedOsAndCpuType schedule");

  if (opp_path.empty()) {
    (void)GetOppPath(opp_path);
    opp_path += "built-in/op_proto/lib/";
    if (opp_path.size() >= static_cast<size_t>(MMPA_MAX_PATH)) {
      GELOGW("param path size:%zu >= max path:%d", opp_path.size(), MMPA_MAX_PATH);
      return;
    }
  }

  char_t real_path[MMPA_MAX_PATH] = {};
  if (mmRealPath(opp_path.c_str(), &(real_path[0U]), MMPA_MAX_PATH) != EN_OK) {
    GELOGW("Can not get real path:%s, it may be an old version", opp_path.c_str());
    return;
  }

  if (mmIsDir(&(real_path[0U])) != EN_OK) {
    GELOGW("Path %s is not directory, it may be an old version", real_path);
    return;
  }

  mmDirent **entries = nullptr;
  const auto ret = Scandir(&(real_path[0U]), &entries, nullptr, nullptr);
  if (ret < EN_OK) {
    GELOGW("Can not open directory %s, it may be an old version, ret = %d", real_path, ret);
    return;
  }
  for (int32_t i = 0; i < ret; ++i) {
    const mmDirent *const dir_ent = *ge::PtrAdd<mmDirent *>(entries, static_cast<size_t>(ret), static_cast<size_t>(i));
    if (dir_ent != nullptr && static_cast<int32_t>(dir_ent->d_type) == DT_DIR) {
      std::string dir_name = dir_ent->d_name;
      if ((dir_name.compare(".") == 0) || (dir_name.compare("..") == 0)) {
        continue;
      }
      if ((layer == kLibFirstLayer) && (opp_supported_os_cpu.find(dir_name) == opp_supported_os_cpu.end())) {
        opp_supported_os_cpu[dir_name] = {};
        GetOppSupportedOsAndCpuType(opp_supported_os_cpu, opp_path + dir_name, dir_name, 1U);
      }
      if (layer == kLibSecondLayer) {
        (void)opp_supported_os_cpu[os_name].emplace(dir_name);
        GELOGD("Get supported os[%s] -> cpu[%s]", os_name.c_str(), dir_name.c_str());
      }
    }
  }
  mmScandirFree(entries, ret);
  return;
}

void PluginManager::GetCurEnvPackageOsAndCpuType(std::string &host_env_os, std::string &host_env_cpu) {
  GELOGD("Enter GetCurEnvPackageOsAndCpuType schedule");
  std::string model_path = GetModelPath();
  GELOGD("Current lib path is:%s", model_path.c_str());
  model_path = model_path.substr(0, model_path.rfind('/'));
  model_path = model_path.substr(0, model_path.rfind('/'));
  model_path = model_path.substr(0, model_path.rfind('/') + 1);
  GELOGD("Run package path is:%s", model_path.c_str());

  std::string scene;
  if (mmAccess2((model_path + kOppPath + kScene).c_str(), M_R_OK) == EN_OK) {
    scene = model_path + kOppPath + kScene;
  } else if (mmAccess2((model_path + kRuntimePath + kScene).c_str(), M_R_OK) == EN_OK) {
    scene = model_path + kRuntimePath + kScene;
  } else {
    GELOGW("opp and runtime do not exist");
    return;
  }
  GELOGI("extract os and cpu info from %s", scene.c_str());
  std::ifstream ifs(scene);
  if (!ifs.good()) {
    GELOGW("Can not open file:%s", scene.c_str());
    return;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    line = StringUtils::Trim(line);
    std::vector<std::string> value = StringUtils::Split(line, '=');
    if (value.size() != kSceneValueCount) {
      continue;
    }
    if (value[kSceneKeyIndex].compare(kSceneOs) == 0) {
      host_env_os = value[kSceneValueIndex];
      GELOGI("Get os:%s", host_env_os.c_str());
    }
    if (value[kSceneKeyIndex].compare(kSceneArch) == 0) {
      host_env_cpu = value[kSceneValueIndex];
      GELOGI("Get cpu:%s", host_env_cpu.c_str());
    }
  }
  return;
}

bool PluginManager::GetVersionFromPathWithName(const std::string &file_path, std::string &version,
                                               const std::string version_name) {
  // Normalize the path
  std::string resolved_file_path = RealPath(file_path.c_str());
  if (resolved_file_path.empty()) {
    GELOGW("Invalid input file path [%s], make sure that the file path is correct.", file_path.c_str());
    return false;
  }
  std::ifstream fs(resolved_file_path, std::ifstream::in);
  if (!fs.is_open()) {
    GELOGW("Open %s failed.", file_path.c_str());
    return false;
  }

  std::string line;
  while (std::getline(fs, line)) {
    if (ParseVersion(line, version, version_name)) {
      GELOGD("Parse version success. content is [%s].", line.c_str());
      fs.close();
      return true;
    }
  }

  GELOGW("No version information found in the file path:%s", file_path.c_str());
  fs.close();
  return false;
}

bool PluginManager::GetVersionFromPath(const std::string &file_path, std::string &version) {
  return GetVersionFromPathWithName(file_path, version, kOppVersion);
}

// Parsing the command line
bool PluginManager::ParseVersion(std::string &line, std::string &version, const std::string version_name) {
  line = StringUtils::Trim(line);
  if (line.empty()) {
    GELOGW("line is empty.");
    return false;
  }

  const std::string::size_type pos = line.find(version_name);
  if (pos == std::string::npos) {
    GELOGW("Incorrect line [%s], it must include [%s].", line.c_str(), version_name.c_str());
    return false;
  }

  if (line.size() == version_name.size()) {
    GELOGW("version information is empty. %s", line.c_str());
    return false;
  }

  version = line.substr(pos + version_name.size());

  return true;
}

bool PluginManager::IsEndWith(const std::string &path, const std::string &suff) {
  return (path.size() >= suff.size()) && (path.compare(path.size() - suff.size(), suff.size(), suff) == 0);
}

void PluginManager::GetFileListWithSuffix(const std::string &path, const std::string &so_suff,
                                          std::vector<std::string> &file_list) {
  if (path.empty()) {
    GELOGI("realPath is empty");
    return;
  }
  if (path.size() >= static_cast<size_t>(MMPA_MAX_PATH)) {
    REPORT_INNER_ERR_MSG("E18888", "param path size:%zu >= max path:%d", path.size(), MMPA_MAX_PATH);
    GELOGE(FAILED, "param path size:%zu >= max path:%d", path.size(), MMPA_MAX_PATH);
    return;
  }

  char_t resolved_path[MMPA_MAX_PATH] = {};

  // Nullptr is returned when the path does not exist or there is no permission
  // Return absolute path when path is accessible
  if (mmRealPath(path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH) != EN_OK) {
    GELOGW("[FindSo][Check] Cannot get real_path for file %s, reason:%s", path.c_str(), strerror(errno));
    return;
  }

  const INT32 is_dir = mmIsDir(&(resolved_path[0U]));
  if (is_dir != EN_OK) {
    GELOGW("[FindSo][Check] Open directory %s failed, maybe it does not exist or is not a dir, errmsg:%s",
           &(resolved_path[0U]), strerror(errno));
    return;
  }

  mmDirent **entries = nullptr;
  const auto file_num = mmScandir(&(resolved_path[0U]), &entries, nullptr, alphasort);
  if ((file_num < 0) || (entries == nullptr)) {
    GELOGW("[FindSo][Scan] Scan directory %s failed, ret:%d, reason:%s", &(resolved_path[0U]), file_num,
           strerror(errno));
    return;
  }
  for (int32_t i = 0; i < file_num; ++i) {
    const mmDirent *const dir_ent = entries[static_cast<size_t>(i)];
    const std::string name = std::string(dir_ent->d_name);
    if ((strcmp(name.c_str(), ".") == 0) || (strcmp(name.c_str(), "..") == 0)) {
      continue;
    }

    if ((static_cast<int32_t>(dir_ent->d_type) != DT_DIR) && IsEndWith(name, so_suff)) {
      const std::string full_name = path + "/" + name;
      file_list.push_back(full_name);
      GELOGD("PluginManager Parse full name = %s.", full_name.c_str());
    }
  }
  mmScandirFree(entries, file_num);
  GELOGI("Found %zu libs.", file_list.size());
}

void PluginManager::FindSoFilesInCustomPassDirs(const std::string &directory, std::vector<std::string> &so_files) {
  char_t resolved_path[MMPA_MAX_PATH] = {};
  if (mmRealPath(directory.c_str(), resolved_path, MMPA_MAX_PATH) != EN_OK) {
    GELOGW("[FindDirs][Check] Get real_path for directory %s failed, reason:%s", directory.c_str(), strerror(errno));
    return;
  }

  const INT32 is_dir = mmIsDir(resolved_path);
  if (is_dir != EN_OK) {
    GELOGW("[FindDirs][Check] Open directory %s failed, maybe it does not exist or is not a dir, errmsg:%s",
           resolved_path, strerror(errno));
    return;
  }

  mmDirent **entries = nullptr;

  const int32_t dir_num = mmScandir(resolved_path, &entries, FilterDirectories, alphasort);
  GELOGI("Scanning directory: %s, directory num: %d", resolved_path, dir_num);
  if (dir_num <= 0) {
    return;
  }

  for (int32_t i = 0; i < dir_num; ++i) {
    const mmDirent *const dir_ent = entries[static_cast<size_t>(i)];
    if (dir_ent == nullptr) {
      continue;
    }
    ProcessSubdirectoryAndSoFiles(directory, dir_ent, so_files);
  }

  mmScandirFree(entries, dir_num);
}

Status PluginManager::GetOpMasterDeviceSoPath(std::string &op_master_device_path) {
  // op_master_device打包在opp和opp_kernel中，两个路径下的so都需要读取
  // op_master_device没有先后顺序，按照算子KernelDef中的so_name选择对应的so
  std::string opp_path;
  GE_ASSERT_TRUE(GetOppPath(opp_path) == SUCCESS, "Failed to get opp path!");
  GetPluginPathFromCustomOppPath(kOpMasterDeviceLib, op_master_device_path);
  std::string sub_pkg_builtin_path = opp_path + "built-in" + kOpTilingDeviceLib;
  if (mmAccess(sub_pkg_builtin_path.c_str()) == EN_OK) {
    GELOGI("[GetOpMasterDeviceSoPath] Sub pkg builtin path exists.");
    GE_ASSERT_SUCCESS(
        GetOppPluginPathNew(opp_path, "%s" + kOpTilingDeviceLib, op_master_device_path, "", "%s" + kOpMasterDeviceLib));
  } else {
    // 兼容老CANN包
    GELOGI("[GetOpMasterDeviceSoPath] Sub pkg builtin path does not exist.");
    GE_ASSERT_SUCCESS(
        GetOppPluginPathNew(opp_path, "%s" + kOpMasterDeviceLib, op_master_device_path, "", "%s" + kOpMasterDeviceLib));
  }
  std::string opp_kernel_path;
  (void)GetUpgradedOppPath(opp_kernel_path);
  if (!opp_kernel_path.empty()) {
    op_master_device_path = opp_kernel_path + kBuiltIn + kOpMasterDeviceLib + ":" + op_master_device_path;
  }
  GELOGI("Get op master device so path is %s", op_master_device_path.c_str());
  return SUCCESS;
}

std::string PluginManager::GetSoPackageName(const std::string &path) {
  // 新的自定义算子包路径：<opp-path>/vendors/<name>/op_proto/
  const std::string vendors_str = "vendors/";
  auto pos = path.find(vendors_str);
  if (pos != std::string::npos) {
    // vendors/ 后面一级目录
    pos += vendors_str.size();
    auto end_pos = path.find('/', pos);
    if (end_pos == std::string::npos) {
      end_pos = path.size();
    }
    return path.substr(pos, end_pos - pos);
  }

  // 老的自定义算子包路径：<opp-path>/op_proto/custom/
  pos = path.find("custom/");
  if (pos != std::string::npos) {
    return "custom";
  }

  // 内置算子包名
  return "built-in";
}

/**
 * 获取内置算子的子包的so是否存在，如果存在子包so就加载子包so，否则加载整包so：
 * 有两种目录格式，整包目录和子包目录，
 * **整包的规范：**
 * 内置算子包路径：
 *                <opp-path>/built-in/op_proto/lib/os_type/cpu_type/
 *                <opp-path>/built-in/op_impl/ai_core/tbe/op_tiling/lib/os_type/cpu_type/
 *
 * **子包的规范：**
 * 内置算子包路径：
 *                <opp-path>/built-in/op_graph/lib/os_type/cpu_type/
 *                <opp-path>/built-in/op_impl/ai_core/tbe/op_host/lib/os_type/cpu_type/
 */
std::string PluginManager::GetOppPkgPath(const std::string &opp_built_in_path, const std::string &whole_pkg_path,
                                         const std::string &sub_pkg_path, const std::string &os_cpu_type,
                                         bool &is_sub_pkg) {
  const bool built_in_op = (opp_built_in_path.find(kBuiltIn) != std::string::npos);
  is_sub_pkg = false;
  if (built_in_op) {
    std::string opp_sub_pkg_path = opp_built_in_path + sub_pkg_path + os_cpu_type;
    std::vector<std::string> so_list;
    PluginManager::GetFileListWithSuffix(opp_sub_pkg_path, kExt, so_list);
    if (!so_list.empty()) {
      is_sub_pkg = true;
      return opp_sub_pkg_path;
    }
    GELOGI("Sub opp package:[%s] does not exist.", opp_sub_pkg_path.c_str());
  }
  return opp_built_in_path + whole_pkg_path + os_cpu_type;
}
}  // namespace af
