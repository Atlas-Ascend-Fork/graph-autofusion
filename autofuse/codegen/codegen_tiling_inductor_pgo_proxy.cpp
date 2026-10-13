/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling.h"

namespace codegen {

namespace {

void AppendInductorPgoProxySystemIncludes(std::stringstream &ss) {
  ss << R"(
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <dlfcn.h>
#include <climits>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include "acl/acl.h"
#ifndef AUTOFUSE_PGO_GENERATION
#define AUTOFUSE_PGO_GENERATION ""
#endif
#ifndef AUTOFUSE_PGO_RUNNER_TIMEOUT_SECONDS
#define AUTOFUSE_PGO_RUNNER_TIMEOUT_SECONDS 1800
#endif
extern char **environ;
)";
}

void AppendInductorPgoProxyDeclarations(std::stringstream &ss, const std::string &tiling) {
  ss << "extern \"C\" int64_t GenerateTopnSolutions("
     << "const std::vector<std::map<std::string, std::string>> &, int64_t, std::vector<" << tiling
     << "> &, std::vector<int64_t> &, std::vector<int64_t> &, ResLimit *);" << std::endl;
  ss << "namespace inductor_pgo_fallback {" << std::endl;
  ss << "static int64_t GenerateModeledFallbackTopnSolutions("
     << "const std::vector<std::map<std::string, std::string>> &, int64_t, std::vector<" << tiling
     << "> &, std::vector<int64_t> &, std::vector<int64_t> &, ResLimit *);" << std::endl;
  ss << "}  // namespace inductor_pgo_fallback" << std::endl;
  ss << R"(
namespace {
constexpr uint32_t kPgoBundleSchemaVersion = 2U;
constexpr int64_t kMaxPgoTopn = 1024;
constexpr size_t kMaxPgoManifestSize = 1024U * 1024U;
constexpr size_t kMaxPgoArtifactSize = 512U * 1024U * 1024U;
constexpr size_t kMaxPgoResultSize = 256U * 1024U * 1024U;
constexpr char kPgoKernelFormat[] = "aicore_binary_elf_v1";
constexpr char kPgoTopnMagic[] = "AUTOFUSE_PGO_TOPN_V2";
constexpr uint32_t kPgoTopnProtocolVersion = 2U;
constexpr uint32_t kPgoTopnRecordHeaderSize = 32U;
constexpr char kPgoGeneration[] = AUTOFUSE_PGO_GENERATION;
static_assert(std::is_trivially_copyable<AutofuseTilingData>::value);
static_assert(std::is_standard_layout<AutofuseTilingData>::value);
struct InductorPgoArtifacts {
  std::string tiling_so; std::string generation_dir; std::string runner;
  std::string kernel; std::string manifest; std::string ld_preload;
};
)";
}

}  // namespace

void TilingLib::GenInductorPgoProxyIncludes(std::stringstream &ss, const std::string &tiling) const {
  AppendInductorPgoProxySystemIncludes(ss);
  AppendInductorPgoProxyDeclarations(ss, tiling);
}

void TilingLib::GenInductorPgoProxyFileHelpers(std::stringstream &ss) const {
  ss << R"(
bool ReadPgoFile(const std::string &path, size_t limit, std::vector<uint8_t> &data) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) { return false; }
  const auto size = file.tellg();
  if (size < 0 || static_cast<uint64_t>(size) > limit) { return false; }
  data.resize(static_cast<size_t>(size)); file.seekg(0, std::ios::beg);
  return data.empty() || static_cast<bool>(file.read(reinterpret_cast<char *>(data.data()), size));
}

std::string PgoBaseName(const std::string &path) {
  const auto pos = path.find_last_of('/');
  return pos == std::string::npos ? path : path.substr(pos + 1U);
}

bool GetJsonString(const std::string &json, const std::string &key, size_t begin, std::string &value) {
  const std::string token = "\"" + key + "\"";
  const auto key_pos = json.find(token, begin);
  const auto colon = key_pos == std::string::npos ? key_pos : json.find(':', key_pos + token.size());
  const auto quote = colon == std::string::npos ? colon : json.find('"', colon + 1U);
  const auto end = quote == std::string::npos ? quote : json.find('"', quote + 1U);
  if (end == std::string::npos) { return false; }
  value = json.substr(quote + 1U, end - quote - 1U); return true;
}

bool GetJsonUint(const std::string &json, const std::string &key, uint32_t &value) {
  const std::string token = "\"" + key + "\""; const auto key_pos = json.find(token);
  const auto colon = key_pos == std::string::npos ? key_pos : json.find(':', key_pos + token.size());
  if (colon == std::string::npos) { return false; }
  char *end = nullptr; errno = 0; const auto parsed = std::strtoul(json.c_str() + colon + 1U, &end, 10);
  if (errno != 0 || end == json.c_str() + colon + 1U || parsed > UINT32_MAX) { return false; }
  value = static_cast<uint32_t>(parsed); return true;
}
)";
}

void TilingLib::GenInductorPgoProxySha256(std::stringstream &ss) const {
  ss << R"(
bool ComputeFileSha256(const std::string &path, std::string &hex) {
  std::vector<uint8_t> data;
  if (!ReadPgoFile(path, kMaxPgoArtifactSize, data)) { return false; }
  void *crypto = dlopen("libcrypto.so.3", RTLD_NOW | RTLD_LOCAL);
  if (crypto == nullptr) { crypto = dlopen("libcrypto.so", RTLD_NOW | RTLD_LOCAL); }
  if (crypto == nullptr) { return false; }
  using Sha256Fn = unsigned char *(*)(const unsigned char *, size_t, unsigned char *);
  auto sha256 = reinterpret_cast<Sha256Fn>(dlsym(crypto, "SHA256"));
  unsigned char digest[32] = {};
  const bool valid = sha256 != nullptr && sha256(data.data(), data.size(), digest) != nullptr;
  dlclose(crypto);
  if (!valid) { return false; }
  constexpr char digits[] = "0123456789abcdef"; hex.resize(64U);
  for (size_t i = 0; i < 32U; ++i) {
    hex[2U * i] = digits[digest[i] >> 4U]; hex[2U * i + 1U] = digits[digest[i] & 0xFU];
  }
  return true;
}

bool ValidateArtifactHash(const std::string &json, const std::string &name, const std::string &path) {
  const auto section = json.find("\"" + name + "\"");
  std::string file_name; std::string expected_hash; std::string actual_hash;
  return section != std::string::npos && GetJsonString(json, "file", section, file_name) &&
      GetJsonString(json, "sha256", section, expected_hash) && file_name == PgoBaseName(path) &&
      expected_hash.size() == 64U && ComputeFileSha256(path, actual_hash) && actual_hash == expected_hash;
}
)";
}

namespace {

void AppendInductorPgoManifestValidation(std::stringstream &ss) {
  ss << R"(
bool ValidateInductorPgoManifest(InductorPgoArtifacts &artifacts) {
  std::vector<uint8_t> bytes;
  if (!ReadPgoFile(artifacts.manifest, kMaxPgoManifestSize, bytes)) {
    OP_LOGE(OP_NAME, "Read Inductor PGO manifest failed: %s", artifacts.manifest.c_str());
    return false;
  }
  const std::string json(bytes.begin(), bytes.end());
  std::string generation; std::string ld_preload;
  uint32_t bundle_schema_version = 0; uint32_t result_protocol_version = 0;
  if (!GetJsonUint(json, "bundle_schema_version", bundle_schema_version) ||
      !GetJsonString(json, "generation", 0U, generation) ||
      !GetJsonUint(json, "result_protocol_version", result_protocol_version) ||
      !GetJsonString(json, "ld_preload", 0U, ld_preload)) {
    OP_LOGE(OP_NAME, "Invalid Inductor PGO manifest fields: %s", artifacts.manifest.c_str());
    return false;
  }
  const bool valid = bundle_schema_version == kPgoBundleSchemaVersion && generation == kPgoGeneration &&
      result_protocol_version == kPgoTopnProtocolVersion &&
      ValidateArtifactHash(json, "tiling_so", artifacts.tiling_so) &&
      ValidateArtifactHash(json, "runner", artifacts.runner) &&
      ValidateArtifactHash(json, "kernel", artifacts.kernel);
  if (!valid) {
    OP_LOGE(OP_NAME, "Inductor PGO manifest validation failed: %s", artifacts.manifest.c_str());
    return false;
  }
  artifacts.ld_preload = ld_preload;
  return true;
}
)";
}

void AppendInductorPgoArtifactResolution(std::stringstream &ss) {
  ss << R"(
bool ResolveInductorPgoArtifactsUncached(InductorPgoArtifacts &artifacts) {
  Dl_info info = {};
  if (kPgoGeneration[0] == '\0' ||
      dladdr(reinterpret_cast<const void *>(&ResolveInductorPgoArtifactsUncached), &info) == 0 ||
      info.dli_fname == nullptr) {
    OP_LOGE(OP_NAME, "Resolve Inductor PGO artifacts failed: invalid generation or tiling so path");
    return false;
  }
  char real_path[PATH_MAX] = {};
  if (realpath(info.dli_fname, real_path) == nullptr) {
    OP_LOGE(OP_NAME, "Resolve Inductor PGO tiling so realpath failed: %s", info.dli_fname);
    return false;
  }
  const std::string tiling_entry = real_path;
  artifacts.generation_dir = tiling_entry + ".pgo." + kPgoGeneration;
  const std::string base = PgoBaseName(tiling_entry);
  artifacts.tiling_so = artifacts.generation_dir + "/" + base;
  artifacts.runner = artifacts.generation_dir + "/" + base + ".pgo_runner";
  artifacts.kernel = artifacts.generation_dir + "/" + base + ".pgo_kernel." + kPgoKernelFormat;
  artifacts.manifest = artifacts.generation_dir + "/manifest.json";
  if (access(artifacts.tiling_so.c_str(), R_OK) != 0 || access(artifacts.runner.c_str(), X_OK) != 0 ||
      access(artifacts.kernel.c_str(), R_OK) != 0) {
    OP_LOGE(OP_NAME, "Inductor PGO sidecar missing: tiling=%s runner=%s kernel=%s", artifacts.tiling_so.c_str(),
            artifacts.runner.c_str(), artifacts.kernel.c_str());
    return false;
  }
  return ValidateInductorPgoManifest(artifacts);
}
)";
}

void AppendInductorPgoArtifactCache(std::stringstream &ss) {
  ss << R"(
bool ResolveInductorPgoArtifacts(InductorPgoArtifacts &artifacts) {
  static std::once_flag validation_once;
  static InductorPgoArtifacts cached_artifacts;
  static bool valid = false;
  std::call_once(validation_once, []() {
    InductorPgoArtifacts resolved_artifacts;
    valid = ResolveInductorPgoArtifactsUncached(resolved_artifacts);
    if (valid) { cached_artifacts = std::move(resolved_artifacts); }
  });
  if (!valid) { return false; }
  artifacts = cached_artifacts;
  return true;
}
)";
}

}  // namespace

void TilingLib::GenInductorPgoProxyManifest(std::stringstream &ss) const {
  AppendInductorPgoManifestValidation(ss);
  AppendInductorPgoArtifactResolution(ss);
  AppendInductorPgoArtifactCache(ss);
}

void TilingLib::GenInductorPgoProxyResultParser(std::stringstream &ss, const std::string &tiling) const {
  ss << R"(
template <typename T>
bool ReadPgoResultValue(const std::vector<uint8_t> &data, size_t &offset, T &value) {
  if (offset > data.size() || sizeof(T) > data.size() - offset) { return false; }
  std::memcpy(&value, data.data() + offset, sizeof(T)); offset += sizeof(T); return true;
}

uint64_t HashProxyTiling(const void *data, size_t size) {
  const auto *bytes = static_cast<const uint8_t *>(data); uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0; i < size; ++i) { hash = (hash ^ bytes[i]) * 1099511628211ULL; }
  return hash;
}
)";
  ss << "bool ParseInductorPgoResult(const std::string &path, int64_t topn, std::vector<" << tiling
     << "> &tiling_datas, std::vector<int64_t> &workspaces, std::vector<int64_t> &block_dims) {" << std::endl;
  ss << R"(  std::vector<uint8_t> data;
  if (!ReadPgoFile(path, kMaxPgoResultSize, data) || data.size() < 40U ||
      std::memcmp(data.data(), kPgoTopnMagic, 20U) != 0) { return false; }
  size_t offset = 20U; uint32_t version = 0; uint32_t flags = 0; uint32_t count = 0;
  uint32_t tiling_size = 0; uint32_t record_header_size = 0;
  if (!ReadPgoResultValue(data, offset, version) || !ReadPgoResultValue(data, offset, flags) ||
      !ReadPgoResultValue(data, offset, count) || !ReadPgoResultValue(data, offset, tiling_size) ||
      !ReadPgoResultValue(data, offset, record_header_size) || version != kPgoTopnProtocolVersion || flags != 0U ||
      count == 0U || count > static_cast<uint64_t>(topn) || tiling_size != sizeof(AutofuseTilingData) ||
      record_header_size != kPgoTopnRecordHeaderSize) { return false; }
)";
  ss << "  std::vector<" << tiling << "> parsed_tilings; std::vector<uint64_t> parsed_workspaces;" << std::endl;
  ss << R"(  std::vector<int64_t> parsed_block_dims; parsed_tilings.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint64_t repr_len = 0; uint64_t workspace = 0; int64_t block_dim = 0; uint64_t tiling_hash = 0;
    AutofuseTilingData tiling_data = {};
    if (!ReadPgoResultValue(data, offset, repr_len) || !ReadPgoResultValue(data, offset, workspace) ||
        !ReadPgoResultValue(data, offset, block_dim) || !ReadPgoResultValue(data, offset, tiling_hash) ||
        !ReadPgoResultValue(data, offset, tiling_data) || repr_len == 0U || repr_len > 16U * 1024U * 1024U ||
        repr_len > data.size() - offset || workspace > static_cast<uint64_t>(INT64_MAX) || block_dim <= 0 ||
        block_dim > UINT32_MAX) { return false; }
    const std::string repr(reinterpret_cast<const char *>(data.data() + offset), repr_len); offset += repr_len;
    uint64_t expected_workspace = 0;
    if (HashProxyTiling(&tiling_data, sizeof(tiling_data)) != tiling_hash ||
        GetTilingDataRepr(&tiling_data) != repr || !GetWorkspaceSize(tiling_data, expected_workspace) ||
        expected_workspace != workspace ||
        static_cast<int64_t>(tiling_data.get_block_dim()) != block_dim) { return false; }
    parsed_tilings.push_back(tiling_data); parsed_workspaces.push_back(workspace); parsed_block_dims.push_back(block_dim);
  }
  if (offset != data.size()) { return false; }
  tiling_datas.swap(parsed_tilings); block_dims.swap(parsed_block_dims);
  workspaces.resize(parsed_workspaces.size());
  for (size_t i = 0; i < parsed_workspaces.size(); ++i) {
    workspaces[i] = static_cast<int64_t>(parsed_workspaces[i]);
  }
  return true;
}
)";
}

namespace {

void AppendInductorPgoResultPathAndWait(std::stringstream &ss) {
  ss << R"(
bool MakeInductorPgoResultPath(std::string &path) {
  char result_dir_template[] = "/tmp/autofuse_inductor_pgo_XXXXXX";
  if (mkdtemp(result_dir_template) == nullptr) { return false; }
  path = std::string(result_dir_template) + "/result.bin"; return true;
}

std::string PgoParentPath(const std::string &path) {
  const auto pos = path.find_last_of('/');
  return pos == std::string::npos ? "." : path.substr(0U, pos);
}

void RemoveInductorPgoResultPath(const std::string &path) {
  unlink(path.c_str()); unlink((path + ".tmp").c_str());
  rmdir(PgoParentPath(path).c_str());
}

int WaitInductorPgoRunner(pid_t pid) {
  constexpr auto timeout = std::chrono::seconds(AUTOFUSE_PGO_RUNNER_TIMEOUT_SECONDS);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  int status = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t wait_ret = waitpid(pid, &status, WNOHANG);
    if (wait_ret == pid) { return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : -1; }
    if (wait_ret < 0 && errno != EINTR) { return -1; }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  kill(pid, SIGKILL); while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  return -1;
}
)";
}

void AppendInductorPgoRunnerEnvironment(std::stringstream &ss) {
  ss << R"(
std::vector<std::string> BuildInductorPgoRunnerEnv(const std::string &ld_preload) {
  std::vector<std::string> env_strings;
  bool found_ld_preload = false;
  for (char **env = environ; env != nullptr && *env != nullptr; ++env) {
    std::string item(*env);
    if (item.rfind("LD_PRELOAD=", 0U) != 0U) {
      env_strings.push_back(std::move(item));
      continue;
    }
    found_ld_preload = true;
    if (ld_preload.empty()) {
      env_strings.push_back(std::move(item));
      continue;
    }
    const std::string old_preload = item.substr(std::strlen("LD_PRELOAD="));
    env_strings.push_back("LD_PRELOAD=" + ld_preload + (old_preload.empty() ? "" : ":" + old_preload));
  }
  if (!found_ld_preload && !ld_preload.empty()) {
    env_strings.push_back("LD_PRELOAD=" + ld_preload);
  }
  return env_strings;
}

std::vector<char *> BuildInductorPgoEnvp(std::vector<std::string> &env_strings) {
  std::vector<char *> envp;
  envp.reserve(env_strings.size() + 1U);
  for (auto &env : env_strings) {
    envp.push_back(const_cast<char *>(env.c_str()));
  }
  envp.push_back(nullptr);
  return envp;
}
)";
}

void AppendInductorPgoRunnerSpawn(std::stringstream &ss) {
  ss << R"(
int SpawnInductorPgoRunner(const InductorPgoArtifacts &artifacts, int32_t device_id, const ResLimit &limit,
                           int64_t topn, const std::string &result_path) {
  const std::string device = std::to_string(device_id); const std::string aiv = std::to_string(limit.aiv_num);
  const std::string ub = std::to_string(limit.ub_size); const std::string topn_arg = std::to_string(topn);
  std::vector<char *> argv = {const_cast<char *>(artifacts.runner.c_str()), const_cast<char *>(device.c_str()),
      const_cast<char *>(aiv.c_str()), const_cast<char *>(ub.c_str()), const_cast<char *>("autofuse"),
      const_cast<char *>(artifacts.tiling_so.c_str()), const_cast<char *>(artifacts.kernel.c_str()),
      const_cast<char *>(result_path.c_str()), const_cast<char *>(topn_arg.c_str()), nullptr};
  auto env_strings = BuildInductorPgoRunnerEnv(artifacts.ld_preload);
  auto envp = BuildInductorPgoEnvp(env_strings);
  pid_t pid = -1;
  const int spawn_ret = posix_spawn(&pid, artifacts.runner.c_str(), nullptr, nullptr, argv.data(), envp.data());
  return spawn_ret == 0 ? WaitInductorPgoRunner(pid) : -1;
}
)";
}

}  // namespace

void TilingLib::GenInductorPgoProxySpawn(std::stringstream &ss) const {
  AppendInductorPgoResultPathAndWait(ss);
  AppendInductorPgoRunnerEnvironment(ss);
  AppendInductorPgoRunnerSpawn(ss);
}

void TilingLib::GenInductorPgoProxyFunction(std::stringstream &ss, const std::string &tiling) const {
  ss << "int64_t FallbackToInductorModeledTopn("
     << "const std::vector<std::map<std::string, std::string>> &input_configs, int64_t topn, "
     << "const ResLimit &limit, std::vector<" << tiling
     << "> &tiling_datas, std::vector<int64_t> &workspaces, std::vector<int64_t> &block_dims) {" << std::endl;
  ss << R"(  OP_LOGW(OP_NAME, "Inductor PGO failed, fallback to modeled TopN");
  ResLimit modeled_limit = limit;
  return inductor_pgo_fallback::GenerateModeledFallbackTopnSolutions(
      input_configs, topn, tiling_datas, workspaces, block_dims, &modeled_limit);
}

)";
  ss << "static int64_t RunInductorPgoProxy("
     << "const std::vector<std::map<std::string, std::string>> &input_configs, int64_t topn, "
     << "std::vector<" << tiling << "> &tiling_datas, std::vector<int64_t> &workspaces, "
     << "std::vector<int64_t> &block_dims, ResLimit *res_limit) {" << std::endl;
  ss << R"(  tiling_datas.clear(); workspaces.clear(); block_dims.clear();
  if (topn <= 0 || topn > kMaxPgoTopn) { return -1; }
  const ResLimit effective_res_limit = GetResLimit(res_limit);
  const ResLimit *limit = &effective_res_limit;
  if (limit->aiv_num == 0U || limit->ub_size <= 256U) { return -1; }
  InductorPgoArtifacts artifacts;
  if (!ResolveInductorPgoArtifacts(artifacts)) {
    return FallbackToInductorModeledTopn(input_configs, topn, *limit, tiling_datas, workspaces, block_dims);
  }
  int32_t device_id = -1;
  if (aclrtGetDevice(&device_id) != ACL_SUCCESS || device_id < 0) {
    OP_LOGW(OP_NAME, "Get current device failed for Inductor PGO");
    return FallbackToInductorModeledTopn(input_configs, topn, *limit, tiling_datas, workspaces, block_dims);
  }
  std::string result_path;
  if (!MakeInductorPgoResultPath(result_path)) {
    OP_LOGW(OP_NAME, "Create Inductor PGO result path failed");
    return FallbackToInductorModeledTopn(input_configs, topn, *limit, tiling_datas, workspaces, block_dims);
  }
  const int runner_ret = SpawnInductorPgoRunner(artifacts, device_id, *limit, topn, result_path);
  const bool parsed = runner_ret == 0 &&
      ParseInductorPgoResult(result_path, topn, tiling_datas, workspaces, block_dims);
  RemoveInductorPgoResultPath(result_path);
  if (!parsed) {
    OP_LOGW(OP_NAME, "Inductor PGO runner or result parsing failed, runner_ret=%d", runner_ret);
    return FallbackToInductorModeledTopn(input_configs, topn, *limit, tiling_datas, workspaces, block_dims);
  }
  return 0;
}
}  // namespace
)";
}

void TilingLib::GenInductorPgoProxyEntry(std::stringstream &ss, const std::string &tiling) const {
  GenInductorPgoProxyIncludes(ss, tiling);
  GenInductorPgoProxyFileHelpers(ss);
  GenInductorPgoProxySha256(ss);
  GenInductorPgoProxyManifest(ss);
  GenInductorPgoProxyResultParser(ss, tiling);
  GenInductorPgoProxySpawn(ss);
  GenInductorPgoProxyFunction(ss, tiling);
  ss << "extern \"C\" int64_t GenerateTopnSolutions("
     << "const std::vector<std::map<std::string, std::string>> &input_configs, int64_t topn, "
     << "std::vector<" << tiling << "> &tiling_datas, std::vector<int64_t> &workspaces, "
     << "std::vector<int64_t> &block_dims, ResLimit *res_limit) {" << std::endl;
  ss << "  return RunInductorPgoProxy(input_configs, topn, tiling_datas, workspaces, block_dims," << std::endl;
  ss << "                             res_limit);" << std::endl;
  ss << "}" << std::endl;
}

}  // namespace codegen
