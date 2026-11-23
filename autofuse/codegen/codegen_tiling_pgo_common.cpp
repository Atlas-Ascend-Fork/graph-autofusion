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

#include <array>
#include <cstring>

#include "backend/backend_spec.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"

namespace codegen {
using namespace ascgen_utils;
void AppendPgoDlopenFlags(std::stringstream &ss) {
  ss << R"(
#if defined(RTLD_NODELETE)
// Device launches can outlive the host wrapper call.  Keep the kernel DSO
// mapped until the short-lived PGO runner exits; the call guard only covers
// host-side calls and cannot observe device-side references.
constexpr bool kPgoDlopenNodelete = true;
constexpr int kPgoDlopenFlags = RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE;
#else
constexpr bool kPgoDlopenNodelete = true;
constexpr int kPgoDlopenFlags = RTLD_NOW | RTLD_LOCAL;
#endif
)" << std::endl;
}

void AppendPgoDsoCallGuard(std::stringstream &ss, const char *handle_name, const char *initialized_name,
                           const char *closing_name, const char *active_calls_name, const char *mutex_name,
                           const char *condition_name) {
  std::string code = R"(
class PgoDsoCallGuard {
 public:
  PgoDsoCallGuard() {
    std::lock_guard<std::mutex> lock(MUTEX);
    if (CLOSING.load(std::memory_order_acquire) || HANDLE == nullptr || !INITIALIZED) { return; }
    ACTIVE.fetch_add(1, std::memory_order_release);
    valid_ = true;
  }
  ~PgoDsoCallGuard() {
    if (valid_) {
      ACTIVE.fetch_sub(1, std::memory_order_release);
      CONDITION.notify_all();
    }
  }
  explicit operator bool() const { return valid_; }
  PgoDsoCallGuard(const PgoDsoCallGuard &) = delete;
  PgoDsoCallGuard &operator=(const PgoDsoCallGuard &) = delete;
 private:
  bool valid_ = false;
};
)";
  const std::array<std::pair<const char *, const char *>, 6> replacements = {{{"HANDLE", handle_name},
                                                                              {"INITIALIZED", initialized_name},
                                                                              {"CLOSING", closing_name},
                                                                              {"ACTIVE", active_calls_name},
                                                                              {"MUTEX", mutex_name},
                                                                              {"CONDITION", condition_name}}};
  for (const auto &[from, to] : replacements) {
    size_t pos = 0U;
    while ((pos = code.find(from, pos)) != std::string::npos) {
      code.replace(pos, std::strlen(from), to);
      pos += std::strlen(to);
    }
  }
  ss << code << std::endl;
}
namespace {
bool IsNeedFfts() {
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  return backend_spec->pgo_spec.need_ffts;
}

void AppendPgoLogDefs(std::stringstream &ss, bool direct_link) {
  ss << "constexpr int32_t PGO_MODULE_NAME = static_cast<int32_t>(" << GE_MODULE_NAME << ");" << std::endl;
  ss << "#define PGO_LOG_PREFIX \"%\" PRIu64 \" %s:[PGO][\" PGO_GRAPH_NAME \"] \"" << std::endl;
  ss << "#define DLOGD(fmt, ...) do { dlog_debug(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], "
        "##__VA_ARGS__); } while (false)"
     << std::endl;
  ss << "#define DLOGI(fmt, ...) do { dlog_info(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], "
        "##__VA_ARGS__); } while (false)"
     << std::endl;
  ss << "#define DLOGW(fmt, ...) do { dlog_warn(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], "
        "##__VA_ARGS__); } while (false)"
     << std::endl;
  // Standalone GE runners execute optional candidates. The GE compile
  // orchestration owns the final profiling error after both search stages.
  if (direct_link) {
    ss << "#define DLOGE(fmt, ...) do { dlog_error(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), "
          "&__FUNCTION__[0U], ##__VA_ARGS__); } while (false)"
       << std::endl;
  } else {
    ss << "#define DLOGE(fmt, ...) DLOGW(fmt, ##__VA_ARGS__)" << std::endl;
  }
}

void AppendDynamicLibraryState(std::stringstream &ss) {
  ss << R"(
static void *handle = nullptr;
static bool initialized = false;
static std::atomic<uint32_t> active_calls{0};
static std::atomic<bool> closing{false};
static std::mutex dso_mutex;
static std::condition_variable dso_cv;
)" << std::endl;
  AppendPgoDlopenFlags(ss);
  AppendPgoDsoCallGuard(ss, "handle", "initialized", "closing", "active_calls", "dso_mutex", "dso_cv");
}

void AppendDynamicLibraryInit(std::stringstream &ss) {
  ss << R"(
__attribute__((constructor)) void Init() {
  std::lock_guard<std::mutex> lock(dso_mutex);
  if (initialized) return;
  closing.store(false, std::memory_order_release);
  handle = dlopen(kernel_file, kPgoDlopenFlags);
  if (!handle) {
    DLOGE("Failed to load %s: %s", kernel_file, dlerror());
    return;
  }
  DLOGD("Kernel api lib %s loaded successfully", kernel_file);
  initialized = true;
})" << std::endl;
}

void AppendDynamicLibraryDeinit(std::stringstream &ss) {
  ss << R"(
__attribute__((destructor)) void DeInit() {
  std::unique_lock<std::mutex> lock(dso_mutex);
  closing.store(true, std::memory_order_release);
  dso_cv.wait(lock, [] { return active_calls.load(std::memory_order_acquire) == 0; });
  if (handle) {
    if (!kPgoDlopenNodelete) { dlclose(handle); }
    handle = nullptr;
  }
  initialized = false;
})" << std::endl;
}

void AppendDynamicLibraryLookup(std::stringstream &ss) {
  ss << R"(
inline void *GetFunc(const char *func_name) {
  if (handle == nullptr) {
    return nullptr;
  }
  void *func = dlsym(handle, func_name);
  if (func == nullptr) {
    DLOGE("Failed to load wrapper api func: %s", dlerror());
  }
  return func;
})" << std::endl;
}
}  // namespace

void TilingLib::GenPgoHeaders(std::stringstream &ss, bool direct_link) const {
  ss << "#include <cinttypes>" << std::endl;
  ss << "#include <unistd.h>" << std::endl;
  ss << "#include <fcntl.h>" << std::endl;
  ss << "#include <sys/file.h>" << std::endl;
  ss << "#include <sys/syscall.h>" << std::endl;
  ss << "#include <sys/wait.h>" << std::endl;
  ss << "#include <dlfcn.h>" << std::endl << std::endl;

  ss << "#include <algorithm>" << std::endl;
  ss << "#include <chrono>" << std::endl;
  ss << "#include <cfloat>" << std::endl;
  ss << "#include <cstdint>" << std::endl;
  ss << "#include <cstdlib>" << std::endl;
  ss << "#include <atomic>" << std::endl;
  ss << "#include <cerrno>" << std::endl;
  ss << "#include <cstring>" << std::endl;
  ss << "#include <securec.h>" << std::endl;
  ss << "#include <fstream>" << std::endl;
  if (direct_link) {
    ss << "#include <cctype>" << std::endl;
    ss << "#include <climits>" << std::endl;
    ss << "#include <cstdio>" << std::endl;
    ss << "#include <limits>" << std::endl;
    ss << "#include <type_traits>" << std::endl;
  }
  ss << "#include <map>" << std::endl;
  ss << "#include <memory>" << std::endl;
  ss << "#include <mutex>" << std::endl;
  ss << "#include <condition_variable>" << std::endl;
  ss << "#include <string>" << std::endl;
  ss << "#include <thread>" << std::endl;
  ss << "#include <utility>" << std::endl;
  ss << "#include <unordered_map>" << std::endl;
  ss << "#include <unordered_set>" << std::endl;
  ss << "#include <vector>" << std::endl << std::endl;

  ss << "#include \"acl/acl.h\"" << std::endl;
  ss << "#include \"dlog_pub.h\"" << std::endl;
  ss << "#include \"tiling/platform/platform_ascendc.h\"" << std::endl << std::endl;

  ss << "#include \"autofuse_tiling_data.h\"" << std::endl << std::endl;
  ss << "extern \"C\" int AutofusePgoCollectorCreate(uint32_t, void *, void **);" << std::endl;
  ss << "extern \"C\" int AutofusePgoCollectorBegin(void *, const char *);" << std::endl;
  ss << "extern \"C\" int AutofusePgoCollectorBeginCandidate(void *, const char *, const char *, uint32_t, uint32_t);"
     << std::endl;
  ss << "extern \"C\" int AutofusePgoCollectorRecordLaunch(void *, uint64_t);" << std::endl;
  ss << "extern \"C\" int AutofusePgoCollectorEnd(void *);" << std::endl;
  ss << "extern \"C\" int AutofusePgoCollectorAbort(void *);" << std::endl;
  ss << "extern \"C\" int AutofusePgoCollectorFinalizeBatch(void *);" << std::endl;
  ss << "extern \"C\" int AutofusePgoCollectorGetCandidateDurationNs(void *, uint32_t, uint64_t *);" << std::endl;
  ss << "extern \"C\" void AutofusePgoCollectorDestroy(void *);" << std::endl << std::endl;
  ss << PGOTensorArgsDef();
}

void TilingLib::GenDynamicLibraryLoaderCode(std::stringstream &ss) const {
  AppendDynamicLibraryState(ss);
  AppendDynamicLibraryInit(ss);
  AppendDynamicLibraryDeinit(ss);
  AppendDynamicLibraryLookup(ss);
}

void TilingLib::GenPgoCardLock(std::stringstream &ss) const {
  ss << R"(
class CardLock {
public:
  CardLock(const char *path) {
    fd_ = open(path, O_RDWR | O_CREAT, 0666);
    if (fd_ == -1) {
      DLOGW("open lock file %s: %s", path, std::strerror(errno));
      std::exit(1);
    }
    if (flock(fd_, LOCK_EX) == -1) {
      DLOGW("flock LOCK_EX: %s", std::strerror(errno));
      std::exit(1);
    }
  }

  ~CardLock() {
    if (fd_ != -1) {
      if (flock(fd_, LOCK_UN) == -1) {
        DLOGW("flock LOCK_UN: %s", std::strerror(errno));
      }
      close(fd_);
    }
  }

  CardLock(const CardLock&) = delete;
  CardLock& operator=(const CardLock&) = delete;

private:
  int fd_{-1};
};
)" << std::endl;
}

void TilingLib::GenPgoSaveTilingKey(std::stringstream &ss) const {
  ss << R"(void PgoSaveTilingKey(const AutofuseTilingData &tiling_data, double best_perf, std::ofstream &out_file) {
  const size_t tiling_bytes = sizeof(tiling_data);
  const size_t tiling_bytes_align = (tiling_bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
  std::vector<int32_t> tiling_i32(tiling_bytes_align, 0);
  memcpy_s(tiling_i32.data(), tiling_i32.size() * sizeof(int32_t), &tiling_data, tiling_bytes);
  for (size_t idx = 0; idx < tiling_i32.size(); ++idx) {
    out_file << tiling_i32[idx] << " ";
  }
  out_file << "# " << best_perf << std::endl;
})" << std::endl;
}

void TilingLib::GenPgoAppendSearchTilingData(std::stringstream &ss) const {
  GenPgoSaveTilingKey(ss);
  ss << "void AppendPgoSearchTilingData(const AutofuseTilingData &tiling_data, double best_perf, std::ios::openmode "
        "mode = std::ios::app) {"
     << std::endl;
  ss << "  const std::string result_file = std::string(search_file) + \".retest\";" << std::endl;
  ss << "  std::ofstream out_file(result_file, mode);" << std::endl;
  ss << "  if (!out_file.is_open()) {" << std::endl;
  ss << "    DLOGW(\"Failed to open file:%s\", result_file.c_str());" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  PgoSaveTilingKey(tiling_data, best_perf, out_file);" << std::endl;
  ss << "  out_file.close();" << std::endl;
  ss << std::endl;

  ss << "  int fd = ::open(result_file.c_str(), O_WRONLY);" << std::endl;
  ss << "  if (fd < 0) {" << std::endl;
  ss << "    DLOGW(\"Failed to open file:%s\", result_file.c_str());" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (::fsync(fd) < 0) {" << std::endl;
  ss << "    DLOGW(\"Failed to fsync file:%s\", result_file.c_str());" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ::close(fd);" << std::endl;
  ss << std::endl;
  ss << "  return;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoKernelLaunchOpArgs(const ascir::FusedScheduledResult &fused_schedule_result,
                                         std::stringstream &ss, bool direct_link) const {
  ss << "struct AivKernelLaunchOpArgs {" << std::endl;
  ss << PGOSearchStructInputOutputDef(fused_schedule_result);
  ss << "  uint64_t workspace_addr;" << std::endl;
  ss << (direct_link ? "  AutofuseTilingData tiling_data;" : "  uint64_t tiling_addr;") << std::endl;
  ss << "};" << std::endl;

  ss << "struct MixKernelLaunchOpArgs {" << std::endl;
  if (IsNeedFfts()) {
    ss << "  uint64_t ffts;" << std::endl;
  }
  ss << PGOSearchStructInputOutputDef(fused_schedule_result);
  ss << "  uint64_t workspace_addr;" << std::endl;
  ss << (direct_link ? "  AutofuseTilingData tiling_data;" : "  uint64_t tiling_addr;") << std::endl;
  ss << "};" << std::endl;

  ss << "void *g_workspace = nullptr;" << std::endl;
}

void TilingLib::GenPgoCheckTilingIsMix(const ascir::FusedScheduledResult &fused_schedule_result,
                                       std::stringstream &ss) const {
  ss << "bool IsMixTiling(const AutofuseTilingData &t) {" << std::endl;
  ss << "  if constexpr (!g_is_mix_operator) {" << std::endl;
  ss << "    return false;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (!g_is_static_kernel) {" << std::endl;
  ss << "    return true;" << std::endl;
  ss << "  }" << std::endl;
  if (!ascgen_utils::IsSingleGroup(fused_schedule_result)) {
    for (size_t graph_id = 0U; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
      ss << "  if (!g_mix_graph" << graph_id << "_tiling_keys.empty() && std::find(g_mix_graph" << graph_id
         << "_tiling_keys.begin(), g_mix_graph" << graph_id << "_tiling_keys.end(), t.graph" << graph_id
         << "_tiling_key) != g_mix_graph" << graph_id << "_tiling_keys.end()) {" << std::endl;
      ss << "    return true;" << std::endl;
      ss << "  }" << std::endl;
    }
  }
  ss << "  return false;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoLaunchParamsInit(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss,
                                       bool direct_link) const {
  ss << "aclError LaunchParamsInit(PgoTensorArgs *tensor_args) {" << std::endl;
  ss << "  static void *ffts = nullptr;" << std::endl;
  ss << "  aclError ret = ACL_SUCCESS;" << std::endl;
  ss << "  constexpr uint32_t kPgoInputCount = " << fused_schedule_result.input_nodes.size() << "U;" << std::endl;
  ss << "  constexpr uint32_t kPgoOutputCount = " << PGOSearchFuncGetOutputCount(fused_schedule_result) << "U;"
     << std::endl;
  ss << "  if (tensor_args == nullptr || tensor_args->input_num < kPgoInputCount ||" << std::endl;
  ss << "      tensor_args->output_num < kPgoOutputCount ||" << std::endl;
  ss << "      (kPgoInputCount > 0U && tensor_args->inputs == nullptr) ||" << std::endl;
  ss << "      (kPgoOutputCount > 0U && tensor_args->outputs == nullptr)) {" << std::endl;
  ss << "    DLOGE(\"invalid pgo tensor args: args=%p, input_num=%u (expected >= %u), output_num=%u "
        "(expected >= %u), inputs=%p, outputs=%p\", static_cast<void *>(tensor_args), "
        "tensor_args == nullptr ? 0U : tensor_args->input_num, kPgoInputCount, "
        "tensor_args == nullptr ? 0U : tensor_args->output_num, kPgoOutputCount, "
        "tensor_args == nullptr ? nullptr : static_cast<const void *>(tensor_args->inputs), "
        "tensor_args == nullptr ? nullptr : static_cast<const void *>(tensor_args->outputs));"
     << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << PGOSearchFuncInputOutputStructAssignDef(fused_schedule_result, "  g_launch_params.aiv_args");
  if (!direct_link) {
    ss << "  g_launch_params.aiv_args.tiling_addr = reinterpret_cast<uint64_t>(g_tiling_device_addr);" << std::endl;
  }
  if (IsNeedFfts()) {
    ss << "  ret = aclrtGetHardwareSyncAddr(&ffts);" << std::endl;
    ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
    ss << "    DLOGE(\"acl get hardware sync addr failed, ERROR: %d\", ret);" << std::endl;
    ss << "    return FAILED;" << std::endl;
    ss << "  }" << std::endl;
    ss << "  g_launch_params.mix_args.ffts = reinterpret_cast<uint64_t>(ffts);" << std::endl;
  }
  ss << PGOSearchFuncInputOutputStructAssignDef(fused_schedule_result, "  g_launch_params.mix_args");
  if (!direct_link) {
    ss << "  g_launch_params.mix_args.tiling_addr = reinterpret_cast<uint64_t>(g_tiling_device_addr);" << std::endl;
  }
  ss << "  ret = aclrtMalloc(&g_launch_params.aiv_args_device, sizeof(AivKernelLaunchOpArgs), "
        "ACL_MEM_MALLOC_HUGE_FIRST);"
     << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl malloc aiv args device failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtMalloc(&g_launch_params.mix_args_device, sizeof(MixKernelLaunchOpArgs), "
        "ACL_MEM_MALLOC_HUGE_FIRST);"
     << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl malloc mix args device failed, ERROR: %d\", ret);" << std::endl;
  ss << "    (void)aclrtFree(g_launch_params.aiv_args_device);" << std::endl;
  ss << "    g_launch_params.aiv_args_device = nullptr;" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return ACL_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoLaunchParamsDeInit(std::stringstream &ss) const {
  ss << "void LaunchParamsDeInit() {" << std::endl;
  ss << "  if (g_launch_params.aiv_args_device != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(g_launch_params.aiv_args_device);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGW(\"acl free aiv args device failed, ERROR: %d\", ret);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_launch_params.aiv_args_device = nullptr;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (g_launch_params.mix_args_device != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(g_launch_params.mix_args_device);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGW(\"acl free mix args device failed, ERROR: %d\", ret);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_launch_params.mix_args_device = nullptr;" << std::endl;
  ss << "  }" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoCopyLaunchArgs(std::stringstream &ss, const std::string &kernel_type,
                                     const std::string &assignment) const {
  ss << "    " << assignment << "aclrtMemcpy(g_launch_params." << kernel_type << "_args_device, sizeof(g_launch_params."
     << kernel_type << "_args), (void *)&g_launch_params." << kernel_type << "_args, sizeof(g_launch_params."
     << kernel_type << "_args), ACL_MEMCPY_HOST_TO_DEVICE);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"memcpy " << kernel_type << "_args to device failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
}

void TilingLib::GenPgoUpdateLaunchParams(std::stringstream &ss) const {
  ss << "aclError UpdateLaunchParam(const AutofuseTilingData &tiling_data) {" << std::endl;
  ss << "  if (IsMixTiling(tiling_data)) {" << std::endl;
  ss << "    auto ret = aclrtMemcpy((void *)g_launch_params.mix_args.tiling_addr, sizeof(AutofuseTilingData), (void "
        "*)&tiling_data, "
     << "sizeof(AutofuseTilingData), ACL_MEMCPY_HOST_TO_DEVICE);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"memcpy tiling data to device failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_launch_params.mix_args.workspace_addr = reinterpret_cast<uint64_t>(g_workspace);" << std::endl;
  GenPgoCopyLaunchArgs(ss, "mix", "ret = ");
  ss << "  } else {" << std::endl;
  ss << "    auto ret = aclrtMemcpy((void *)g_launch_params.aiv_args.tiling_addr, sizeof(AutofuseTilingData), (void "
        "*)&tiling_data, "
     << "sizeof(AutofuseTilingData), ACL_MEMCPY_HOST_TO_DEVICE);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"memcpy tiling data to device failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_launch_params.aiv_args.workspace_addr = reinterpret_cast<uint64_t>(g_workspace);" << std::endl;
  GenPgoCopyLaunchArgs(ss, "aiv", "ret = ");
  ss << "  }" << std::endl;
  ss << "  return ACL_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenInductorPgoUpdateLaunchParams(std::stringstream &ss) const {
  ss << "aclError UpdateLaunchParam(const AutofuseTilingData &tiling_data) {" << std::endl;
  ss << "  if (IsMixTiling(tiling_data)) {" << std::endl;
  ss << "    g_launch_params.mix_args.tiling_data = tiling_data;" << std::endl;
  ss << "    g_launch_params.mix_args.workspace_addr = reinterpret_cast<uint64_t>(g_workspace);" << std::endl;
  GenPgoCopyLaunchArgs(ss, "mix", "auto ret = ");
  ss << "  } else {" << std::endl;
  ss << "    g_launch_params.aiv_args.tiling_data = tiling_data;" << std::endl;
  ss << "    g_launch_params.aiv_args.workspace_addr = reinterpret_cast<uint64_t>(g_workspace);" << std::endl;
  GenPgoCopyLaunchArgs(ss, "aiv", "auto ret = ");
  ss << "  }" << std::endl;
  ss << "  return ACL_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoLaunchParams(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss,
                                   bool direct_link) const {
  ss << "struct LaunchParams {" << std::endl;
  ss << "  AivKernelLaunchOpArgs aiv_args;" << std::endl;
  ss << "  void *aiv_args_device;" << std::endl;
  ss << "  MixKernelLaunchOpArgs mix_args;" << std::endl;
  ss << "  void *mix_args_device;" << std::endl;
  ss << "} g_launch_params;" << std::endl;

  GenPgoLaunchParamsInit(fused_schedule_result, ss, direct_link);
  GenPgoLaunchParamsDeInit(ss);
  if (direct_link) {
    GenInductorPgoUpdateLaunchParams(ss);
  } else {
    GenPgoUpdateLaunchParams(ss);
  }
}

void TilingLib::GenPgoToolDeclarations(const ascir::FusedScheduledResult &fused_schedule_result,
                                       const std::string &pgo_dir, std::stringstream &ss, bool direct_link) const {
  const std::string graph_name = CamelToLowerSneak(GenValidName(fused_schedule_result.fused_graph_name.GetString()));
  ss << "namespace {" << std::endl;
  ss << "constexpr bool g_is_mix_operator = " << (IsMixKernelTaskType(fused_schedule_result) ? "true;" : "false;")
     << std::endl;
  ss << "static bool g_is_static_kernel = false;" << std::endl;
  GenPgoMixTilingTable(fused_schedule_result, ss);
  GenPgoCheckTilingIsMix(fused_schedule_result, ss);
  if (direct_link) {
    ss << "constexpr char kInductorPgoKernelName[] = \"" << graph_name << "\";" << std::endl;
  } else {
    ss << "static std::string g_kernel_name;" << std::endl;
  }
  ss << "static std::string g_kernel_o_file;" << std::endl;
  ss << "static std::string g_npu_lock_file;" << std::endl;
  ss << "#define PGO_GRAPH_NAME \"" << graph_name << "\"" << std::endl;
  if (!direct_link) {
    ss << "const char *pgo_dir = \"" << pgo_dir << "\";" << std::endl;
    ss << "const char *config_file = \"" << pgo_dir << "/" << graph_name << "_config.txt" << "\";" << std::endl;
    ss << "const char *search_file = \"" << pgo_dir << "/" << graph_name << "_search.txt" << "\";" << std::endl;
    ss << "const char *kernel_file = \"" << pgo_dir << "/lib" << graph_name << ".so" << "\";" << std::endl;
    ss << "static bool g_pgo_attempt_recorded = false;" << std::endl;
    ss << "static int g_pgo_preflight_status = 0;" << std::endl;
  }
  ss << "#define SUCCESS 0" << std::endl;
  ss << "#define FAILED 1" << std::endl;

  ss << "inline uint64_t PgoGetTid() {" << std::endl;
  ss << "  return static_cast<uint64_t>(syscall(__NR_gettid));" << std::endl;
  ss << "}" << std::endl;
  AppendPgoLogDefs(ss, direct_link);

  GenPgoCardLock(ss);
  if (!direct_link) {
    GenPgoAppendSearchTilingData(ss);
  }
  GenPgoKernelLaunchOpArgs(fused_schedule_result, ss, direct_link);

  if (!direct_link) {
    GenDynamicLibraryLoaderCode(ss);
  }
}

void TilingLib::GenPgoToolFunction(const ascir::FusedScheduledResult &fused_schedule_result, const std::string &pgo_dir,
                                   std::stringstream &ss, bool direct_link) const {
  GenPgoToolDeclarations(fused_schedule_result, pgo_dir, ss, direct_link);

  ss << (direct_link ? "aclrtStream g_stream = nullptr;" : "aclrtStream g_stream;") << std::endl;
  ss << PGOSearchTensorInputOutputDef(fused_schedule_result) << std::endl;
  if (direct_link) {
    ss << "bool g_acl_initialized = false;" << std::endl;
    ss << "bool g_device_set = false;" << std::endl;
    ss << "int32_t g_device_id = -1;" << std::endl;
  } else {
    ss << "void *g_tiling_device_addr = nullptr;" << std::endl;
  }
  ss << "uint32_t g_pgo_device_id = 0;" << std::endl;

  GenPgoLaunchParams(fused_schedule_result, ss, direct_link);

  ss << "struct ResLimit {" << std::endl;
  ss << "  uint32_t valid_num = 0;" << std::endl;
  ss << "  uint32_t aiv_num = 0;" << std::endl;
  ss << "  uint32_t aic_num = 0;" << std::endl;
  ss << "  uint32_t ub_size = 0;" << std::endl;
  ss << "  uint32_t resv[10];" << std::endl;
  ss << "};" << std::endl;
  ss << (direct_link ? "ResLimit g_res_limit = {1, 0, 0, 0, {}};" : "ResLimit g_res_limit = {1, {}};") << std::endl;
  ss << "inline bool IsEqual(double a, double b) {" << std::endl;
  ss << "  const double epsilon = 1e-8;" << std::endl;
  ss << "  double abs = (a > b) ? (a - b) : (b - a);" << std::endl;
  ss << "  return abs < epsilon;" << std::endl;
  ss << "}" << std::endl;
  ss << "} // namespace" << std::endl;
}

void TilingLib::GenPgoWrapperParmCall(const ascir::FusedScheduledResult &fused_schedule_result,
                                      std::stringstream &ss) const {
  ss << "  if (tiling_data == nullptr) {" << std::endl;
  ss << "    DLOGE(\"tiling_data is null\");" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  uint32_t block_dim = tiling_data->block_dim;" << std::endl;
  ss << "  aclError ret = ACL_SUCCESS;" << std::endl;
  ss << "  int64_t tiling_key = 0;" << std::endl;
  if (CanUseTilingKey(fused_schedule_result)) {
    ss << "  if (find_best_tiling_key_fn != nullptr) {" << std::endl;
    ss << "    tiling_key = find_best_tiling_key_fn(*tiling_data);" << std::endl;
    ss << "    if (tiling_key < 0 || static_cast<uint64_t>(tiling_key) >= tiling_key_count) {" << std::endl;
    ss << "      DLOGE(\"find best tiling key failed, tiling_key=%ld, valid range=[0,%lu)\", tiling_key, "
          "tiling_key_count);"
       << std::endl;
    ss << "      return FAILED;" << std::endl;
    ss << "    }" << std::endl;
    ss << "  } else {" << std::endl;
    ss << "    DLOGE(\"find best tiling key func is null\");" << std::endl;
    ss << "    return FAILED;" << std::endl;
    ss << "  }" << std::endl;
  }
}

void TilingLib::GenPgoWrapperKernelLaunch(std::stringstream &ss) const {
  ss << "  if (IsMixTiling(*tiling_data)) {" << std::endl;
  const auto backend_spce = optimize::BackendSpec::GetInstance();
  const bool use_local_memory = (backend_spce != nullptr && backend_spce->set_local_memory_size > 0);
  ss << (use_local_memory ? "    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, "
                            "g_launch_params.mix_args_device, sizeof(g_launch_params.mix_args), &kernel_cfg, g_stream);"
                          : "    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, "
                            "g_launch_params.mix_args_device, sizeof(g_launch_params.mix_args), nullptr, g_stream);")
     << std::endl;
  ss << "  } else {" << std::endl;
  ss << (use_local_memory ? "    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, "
                            "g_launch_params.aiv_args_device, sizeof(g_launch_params.aiv_args), &kernel_cfg, g_stream);"
                          : "    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, "
                            "g_launch_params.aiv_args_device, sizeof(g_launch_params.aiv_args), nullptr, g_stream);")
     << std::endl;
  ss << "  }" << std::endl;
  ss << "  auto ret_async = aclrtSynchronizeStream(g_stream);" << std::endl;
}

void TilingLib::GenPgoWrapperInit(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss,
                                  bool direct_link) const {
  ss << "typedef uint64_t (*GetTilingKeyCountType)(void);" << std::endl;
  ss << "GetTilingKeyCountType get_tiling_key_count_fn = "
     << (direct_link ? "nullptr;" : "reinterpret_cast<GetTilingKeyCountType>(GetFunc(\"GetTilingKeyCount\"));")
     << std::endl;
  ss << "typedef int64_t (*FindBestTilingKeyType)(AutofuseTilingData &t);" << std::endl;
  const bool should_load_find_best_tiling_key = !direct_link && CanUseTilingKey(fused_schedule_result);
  ss << "FindBestTilingKeyType find_best_tiling_key_fn = "
     << (should_load_find_best_tiling_key ? "reinterpret_cast<FindBestTilingKeyType>(GetFunc(\"FindBestTilingKey\"));"
                                          : "nullptr;")
     << std::endl;
  if (direct_link) {
    ss << "static aclrtBinHandle g_pgo_bin_handle = nullptr;" << std::endl;
  }
  ss << "int WrapperOnlyLaunch(uint64_t workspace_size, AutofuseTilingData *tiling_data) {" << std::endl;
  if (!direct_link) {
    ss << "  PgoDsoCallGuard dso_guard;" << std::endl;
    ss << "  if (!dso_guard) { return FAILED; }" << std::endl;
  }
  if (direct_link) {
    ss << "  (void)workspace_size;" << std::endl;
  }
  ss << "  static bool inited = false;" << std::endl;
  if (!direct_link) {
    ss << "  static aclrtBinHandle bin_handle = nullptr;" << std::endl;
  }
  const auto backend_spce = optimize::BackendSpec::GetInstance();
  if (backend_spce != nullptr && backend_spce->set_local_memory_size > 0) {
    ss << "  static aclrtLaunchKernelCfg kernel_cfg{};" << std::endl;
    ss << "  static aclrtLaunchKernelAttr local_memory_size_attr{};" << std::endl;
  }
  ss << "  if (get_tiling_key_count_fn == nullptr) {" << std::endl;
  ss << "    DLOGE(\"get_tiling_key_count_fn is nullptr\");" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  static uint64_t tiling_key_count = get_tiling_key_count_fn();" << std::endl;
  ss << "  static std::vector<aclrtFuncHandle> func_handles(tiling_key_count);" << std::endl;
}

void TilingLib::GenPgoWrapper(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss,
                              bool direct_link) const {
  GenPgoWrapperInit(fused_schedule_result, ss, direct_link);

  GenPgoWrapperParmCall(fused_schedule_result, ss);
  GenPgoLaunchKernelInit(ss, direct_link);
  GenPgoWrapperKernelLaunch(ss);
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"aclrtLaunchKernelV2 failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (ret_async != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"aclrtSynchronizeStream failed, ERROR: %d\", ret_async);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return ret;" << std::endl;
  ss << "}" << std::endl << std::endl;
  if (direct_link) {
    ss << "void PgoBinaryDeInit() {" << std::endl;
    ss << "  if (g_pgo_bin_handle == nullptr) { return; }" << std::endl;
    ss << "  auto ret = aclrtBinaryUnLoad(g_pgo_bin_handle);" << std::endl;
    ss << "  if (ret != ACL_SUCCESS) { DLOGW(\"acl unload binary failed, ERROR: %d\", ret); }" << std::endl;
    ss << "  g_pgo_bin_handle = nullptr;" << std::endl;
    ss << "}" << std::endl << std::endl;
  }
}

void TilingLib::GenPgoAclProfilingBeginCandidate(std::stringstream &ss, bool direct_link) const {
  if (direct_link) {
    ss << R"(
int PgoBeginCandidate(void *collector, AutofuseTilingData *tiling_data, uint64_t workspace_size) {
  if (get_tiling_data_repr_fn == nullptr || tiling_data == nullptr) { return FAILED; }
  const std::string candidate_repr = get_tiling_data_repr_fn(tiling_data);
  if (candidate_repr.empty()) { return FAILED; }
  return AutofusePgoCollectorBeginCandidate(collector, PGO_GRAPH_NAME, candidate_repr.c_str(), workspace_size,
                                            tiling_data->block_dim);
}
)";
  } else {
    ss << R"(
int PgoBeginCandidate(void *collector) {
  return AutofusePgoCollectorBegin(collector, PGO_GRAPH_NAME);
}
)";
  }
}

void TilingLib::GenPgoAclProfilingConstants(std::stringstream &ss) const {
  ss << R"(
static constexpr uint64_t kPgoMeasureLoop = 20U;
static constexpr size_t kPgoBatchGroupSize = 1000U;
static constexpr int kPgoRetryCount = 5;
static constexpr int kPgoProfileUnsupported = -13;
static constexpr int kPgoReporterBusy = -12;
static constexpr int kPgoStopFailed = -6;
static double best_perf = DBL_MAX;
static std::vector<uint64_t> g_acl_durations;
static void *g_pgo_collector = nullptr;

void ResetProfilingRound() {
  g_acl_durations.clear();
}
)";
}

void TilingLib::GenPgoAclProfilingInit(std::stringstream &ss, bool direct_link) const {
  ss << R"(
int InitAclPgoEvents() {
  const int status = AutofusePgoCollectorCreate(g_pgo_device_id, g_stream, &g_pgo_collector);
  if (status != 0) { DLOGI("InitAclPgoEvents failed, status: %d", status); }
)";
  if (!direct_link) {
    ss << "  if (status == kPgoProfileUnsupported || status == kPgoReporterBusy) { g_pgo_preflight_status = status; }"
       << std::endl;
  }
  ss << R"(
  return status;
}
)";
}

void TilingLib::GenPgoAclProfilingDestroy(std::stringstream &ss) const {
  ss << R"(
void DestroyAclPgoEvents() {
  if (g_pgo_collector != nullptr) {
    AutofusePgoCollectorDestroy(g_pgo_collector);
    g_pgo_collector = nullptr;
  }
}
)";
}

void TilingLib::GenPgoAclProfilingMeasure(std::stringstream &ss, bool direct_link) const {
  ss << R"(
int MeasureAclPgoLaunch(uint64_t workspace_size, AutofuseTilingData *tiling_data) {
  if (g_pgo_collector == nullptr) {
    DLOGI("PGO collector is null");
    return FAILED;
  }
  if (tiling_data == nullptr || tiling_data->block_dim == 0U) {
    DLOGI("PGO candidate is invalid");
    return FAILED;
  }
)";
  GenPgoAclProfilingBeginStatus(ss, direct_link);
  GenPgoAclProfilingAttempt(ss, direct_link);
  ss << R"(
  for (uint64_t launch_count = 0U; launch_count < kPgoMeasureLoop; ++launch_count) {
    const int record_status = AutofusePgoCollectorRecordLaunch(g_pgo_collector, launch_count);
    if (record_status != 0) {
      DLOGE("AutofusePgoCollectorRecordLaunch failed, status: %d", record_status);
      AutofusePgoCollectorAbort(g_pgo_collector);
      return FAILED;
    }
    const int launch_status = WrapperOnlyLaunch(workspace_size, tiling_data);
    if (launch_status != ACL_SUCCESS) {
      DLOGE("WrapperOnlyLaunch failed, ERROR: %d", launch_status);
      AutofusePgoCollectorAbort(g_pgo_collector);
      return FAILED;
    }
  }
  const int end_status = AutofusePgoCollectorEnd(g_pgo_collector);
  if (end_status != 0) {
    DLOGE("AutofusePgoCollectorEnd failed, status: %d", end_status);
    AutofusePgoCollectorAbort(g_pgo_collector);
    return FAILED;
  }
  return 0;
}
)" << std::endl;
}

void TilingLib::GenPgoAclProfilingBeginStatus(std::stringstream &ss, bool direct_link) const {
  if (direct_link) {
    ss << "  const int begin_status = PgoBeginCandidate(g_pgo_collector, tiling_data, workspace_size);" << std::endl;
  } else {
    ss << "  const int begin_status = PgoBeginCandidate(g_pgo_collector);" << std::endl;
  }
  ss << R"(
  if (begin_status != 0) {
    DLOGI("PgoBeginCandidate failed, status: %d", begin_status);
    return begin_status;
  }
)";
}

void TilingLib::GenPgoAclProfilingAttempt(std::stringstream &ss, bool direct_link) const {
  if (direct_link) {
    return;
  }
  ss << R"(
  if (!g_pgo_attempt_recorded) {
    std::ofstream attempted(std::string(search_file) + ".attempted", std::ios::app);
    g_pgo_attempt_recorded = attempted.is_open();
    if (!g_pgo_attempt_recorded) {
      DLOGW("Unable to record that PGO profiling started");
    }
  }
)";
}

void TilingLib::GenPgoAclProfiling(std::stringstream &ss, bool direct_link) const {
  GenPgoAclProfilingBeginCandidate(ss, direct_link);
  GenPgoAclProfilingConstants(ss);
  GenPgoAclProfilingInit(ss, direct_link);
  GenPgoAclProfilingDestroy(ss);
  GenPgoAclProfilingMeasure(ss, direct_link);
}

void TilingLib::GenPgoBatchProcess(std::stringstream &ss) const {
  ss << R"(int ProfilingBatchProcess(uint64_t workspace_size, std::vector<AutofuseTilingDataPerf>::iterator begin,
                          std::vector<AutofuseTilingDataPerf>::iterator end) {
  for (auto it = begin; it != end; ++it) {
    const auto update_status = UpdateLaunchParam(it->tiling_data);
    if (update_status != ACL_SUCCESS) {
      DLOGE("UpdateLaunchParam failed, ERROR: %d", update_status);
      AutofusePgoCollectorAbort(g_pgo_collector);
      return FAILED;
    }
    const int measure_status = MeasureAclPgoLaunch(workspace_size, &it->tiling_data);
    if (measure_status != 0) {
      AutofusePgoCollectorAbort(g_pgo_collector);
      return measure_status;
    }
  }
  const int finalize_status = AutofusePgoCollectorFinalizeBatch(g_pgo_collector);
  if (finalize_status != 0) {
    DLOGE("AutofusePgoCollectorFinalizeBatch failed, status: %d", finalize_status);
    AutofusePgoCollectorAbort(g_pgo_collector);
    return finalize_status;
  }
  uint32_t candidate_index = 0U;
  for (auto it = begin; it != end; ++it, ++candidate_index) {
    uint64_t elapsed_ns = 0U;
    const int duration_status = AutofusePgoCollectorGetCandidateDurationNs(g_pgo_collector, candidate_index, &elapsed_ns);
    if (duration_status != 0) {
      DLOGE("AutofusePgoCollectorGetCandidateDurationNs failed, candidate_index: %u, status: %d", candidate_index,
            duration_status);
      AutofusePgoCollectorAbort(g_pgo_collector);
      return FAILED;
    }
    it->best_perf = static_cast<double>(elapsed_ns);
    if (best_perf > it->best_perf) { best_perf = it->best_perf; }
  }
  return 0;
}

)";
}

void TilingLib::GenPgoProfilingBatchSetup(std::stringstream &ss) const {
  ss << "  (void)tensor_args;" << std::endl;
  ss << "  (void)stream;" << std::endl;
  ss << "  if (profiles == nullptr) { return FAILED; }" << std::endl;
  ss << "  int case_num = profiles->size();" << std::endl;
  ss << "  DLOGI(\"PGOGetProfilingBatch case_num:%d\", case_num);" << std::endl;
  ss << "  ResetProfilingRound();" << std::endl;
  ss << "  if (workspace_size > 0) {" << std::endl;
  ss << "    if (workspace_size > SIZE_MAX) {" << std::endl;
  ss << "      DLOGE(\"workspace size exceeds SIZE_MAX: %\" PRIu64, workspace_size);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    auto ret = aclrtMalloc(&g_workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"malloc workspace failed, size: %\" PRIu64 \", ERROR: %d\", workspace_size, ret);" << std::endl;
  ss << "      DestroyAclPgoEvents();" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoGetProfilingBatch(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss,
                                        bool direct_link) const {
  ss << "extern \"C\" long int PGOGetProfilingBatch(" << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "void* stream, uint64_t workspace_size, std::vector<AutofuseTilingDataPerf> *profiles) {" << std::endl;
  if (!direct_link) {
    ss << "  PgoDsoCallGuard dso_guard;" << std::endl;
    ss << "  if (!dso_guard) { return FAILED; }" << std::endl;
  }
  GenPgoProfilingBatchSetup(ss);
  GenPgoProfilingBatchRetry(ss);
  GenPgoProfilingBatchFailure(ss);
  GenPgoProfilingBatchCleanup(ss);
}

void TilingLib::GenPgoProfilingBatchRetry(std::stringstream &ss) const {
  ss << "  int64_t result = 0;" << std::endl;
  ss << "  auto it = profiles->begin();" << std::endl;
  ss << "  while (it != profiles->end()) {" << std::endl;
  ss << "    auto end_it = it + std::min<size_t>(kPgoBatchGroupSize, std::distance(it, profiles->end()));" << std::endl;
  ss << "    size_t start_index = std::distance(profiles->begin(), it);" << std::endl;
  ss << "    for (int i = 0; i < kPgoRetryCount; i++) {" << std::endl;
  ss << "      auto init_status = InitAclPgoEvents();" << std::endl;
  ss << "      if (init_status == kPgoProfileUnsupported) {" << std::endl;
  ss << "        result = kPgoProfileUnsupported;" << std::endl;
  ss << "        DLOGI(\"PGO profiling unsupported on this device; fallback to non-PGO path\");" << std::endl;
  ss << "        break;" << std::endl;
  ss << "      }" << std::endl;
  ss << "      if (init_status == kPgoReporterBusy) { result = init_status; break; }" << std::endl;
  ss << "      if (init_status != 0) {" << std::endl;
  ss << "        result = FAILED;" << std::endl;
  ss << "        DLOGD(\"InitAclPgoEvents failed at start_index:%zu retry time:%d\", start_index, i);" << std::endl;
  ss << "        continue;" << std::endl;
  ss << "      }" << std::endl;
  ss << "      result = ProfilingBatchProcess(workspace_size, it, end_it);" << std::endl;
  ss << "      DestroyAclPgoEvents();" << std::endl;
  ss << "      if (result == kPgoStopFailed || result == kPgoReporterBusy) { break; }" << std::endl;
  ss << "      if (result != 0) {" << std::endl;
  ss << "        DLOGD(\"ProfilingBatchProcess failed at start_index:%zu retry time:%d\", start_index, i);"
     << std::endl;
  ss << "      } else {" << std::endl;
  ss << "        break;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
}

void TilingLib::GenPgoProfilingBatchFailure(std::stringstream &ss) const {
  ss << "    if (result != 0) {" << std::endl;
  ss << "      DestroyAclPgoEvents();" << std::endl;
  ss << "      if (g_workspace != nullptr) {" << std::endl;
  ss << "        aclrtFree(g_workspace);" << std::endl;
  ss << "        g_workspace = nullptr;" << std::endl;
  ss << "      }" << std::endl;
  ss << "      return result;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    it = end_it;" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoProfilingBatchCleanup(std::stringstream &ss) const {
  ss << "  if (g_workspace != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(g_workspace);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"free workspace failed, ERROR: %d\", ret);" << std::endl;
  ss << "      DestroyAclPgoEvents();" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_workspace = nullptr;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  DestroyAclPgoEvents();" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoDirectProfilingCallback(std::stringstream &ss) const {
  ss << R"(  if (g_acl_durations.size() != 1U) {
    DLOGI("PGO measured duration count is invalid, count: %zu", g_acl_durations.size());
    return -1;
  }
  *outCostTime = static_cast<double>(g_acl_durations.front());
  if (best_perf > *outCostTime) { best_perf = *outCostTime; }
  DestroyAclPgoEvents();
)";
}

void TilingLib::GenPgoProfilingCallback(std::stringstream &ss) const {
  GenPgoDirectProfilingCallback(ss);
}

void TilingLib::GenPgoProfilingSetup(std::stringstream &ss) const {
  ss << "  (void)tensor_args;" << std::endl;
  ss << "  (void)stream;" << std::endl;
  ss << "  if (workspace_size > 0) {" << std::endl;
  ss << "    if (workspace_size > SIZE_MAX) {" << std::endl;
  ss << "      DLOGE(\"workspace size exceeds SIZE_MAX: %\" PRIu64, workspace_size);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    auto ret = aclrtMalloc(&g_workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"malloc workspace failed, size: %\" PRIu64 \", ERROR: %d\", workspace_size, ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ResetProfilingRound();" << std::endl;
  ss << "  auto init_status = InitAclPgoEvents();" << std::endl;
  ss << "  if (init_status == kPgoProfileUnsupported) {" << std::endl;
  ss << "    if (g_workspace != nullptr) {" << std::endl;
  ss << "      aclrtFree(g_workspace);" << std::endl;
  ss << "      g_workspace = nullptr;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    return kPgoProfileUnsupported;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (init_status != 0) {" << std::endl;
  ss << "    if (g_workspace != nullptr) {" << std::endl;
  ss << "      aclrtFree(g_workspace);" << std::endl;
  ss << "      g_workspace = nullptr;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    return init_status;" << std::endl;
  ss << "  }" << std::endl << std::endl;
  ss << "  int64_t result = -1;" << std::endl;
  ss << "  *outCostTime = DBL_MAX;" << std::endl;
  ss << "  static int64_t count = 0;" << std::endl;
  ss << "  count++;" << std::endl << std::endl;
}

void TilingLib::GenPgoProfilingLaunch(std::stringstream &ss) const {
  ss << "  const auto update_status = UpdateLaunchParam(*tiling_data);" << std::endl;
  ss << "  if (update_status != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"UpdateLaunchParam failed, ERROR: %d\", update_status);" << std::endl;
  ss << "    DestroyAclPgoEvents();" << std::endl;
  ss << "    if (g_workspace != nullptr) {" << std::endl;
  ss << "      aclrtFree(g_workspace);" << std::endl;
  ss << "      g_workspace = nullptr;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  result = MeasureAclPgoLaunch(workspace_size, tiling_data);" << std::endl;
  ss << "  if (result != 0) {" << std::endl;
  ss << "    DestroyAclPgoEvents();" << std::endl;
  ss << "    if (g_workspace != nullptr) {" << std::endl;
  ss << "      aclrtFree(g_workspace);" << std::endl;
  ss << "      g_workspace = nullptr;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    return result;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  const int finalize_status = AutofusePgoCollectorFinalizeBatch(g_pgo_collector);" << std::endl;
  ss << "  if (finalize_status != 0) {" << std::endl;
  ss << "    DLOGE(\"AutofusePgoCollectorFinalizeBatch failed, status: %d\", finalize_status);" << std::endl;
  ss << "    AutofusePgoCollectorAbort(g_pgo_collector);" << std::endl;
  ss << "    DestroyAclPgoEvents();" << std::endl;
  ss << "    if (g_workspace != nullptr) {" << std::endl;
  ss << "      aclrtFree(g_workspace);" << std::endl;
  ss << "      g_workspace = nullptr;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    return finalize_status;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  uint64_t elapsed_ns = 0U;" << std::endl;
  ss << "  const int duration_status = AutofusePgoCollectorGetCandidateDurationNs(g_pgo_collector, 0U, &elapsed_ns);"
     << std::endl;
  ss << "  if (duration_status != 0) {" << std::endl;
  ss << "    DLOGE(\"AutofusePgoCollectorGetCandidateDurationNs failed, candidate_index: %u, status: %d\", 0U, "
        "duration_status);"
     << std::endl;
  ss << "    AutofusePgoCollectorAbort(g_pgo_collector);" << std::endl;
  ss << "    DestroyAclPgoEvents();" << std::endl;
  ss << "    if (g_workspace != nullptr) {" << std::endl;
  ss << "      aclrtFree(g_workspace);" << std::endl;
  ss << "      g_workspace = nullptr;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  g_acl_durations.push_back(elapsed_ns);" << std::endl << std::endl;
}

void TilingLib::GenPgoProfilingWorkspaceCleanup(std::stringstream &ss) const {
  ss << "  if (g_workspace != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(g_workspace);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"free workspace failed, ERROR: %d\", ret);" << std::endl;
  ss << "      DestroyAclPgoEvents();" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_workspace = nullptr;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  DestroyAclPgoEvents();" << std::endl;
}

void TilingLib::GenPgoGetProfiling(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss,
                                   bool direct_link) const {
  ss << "extern \"C\" long int PGOGetProfiling(" << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "void *stream, uint64_t workspace_size, AutofuseTilingData *tiling_data, double *outCostTime) {" << std::endl;
  if (!direct_link) {
    ss << "  PgoDsoCallGuard dso_guard;" << std::endl;
    ss << "  if (!dso_guard) { return FAILED; }" << std::endl;
  }
  GenPgoProfilingSetup(ss);
  GenPgoProfilingLaunch(ss);
  GenPgoProfilingWorkspaceCleanup(ss);
  GenPgoProfilingCallback(ss);
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoFunc(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "int pgo() {" << std::endl;
  ss << "  PgoDsoCallGuard dso_guard;" << std::endl;
  ss << "  if (!dso_guard) { return FAILED; }" << std::endl;
  ss << "  AutofuseTilingData tiling_data = {0};" << std::endl;
  ss << "  PgoTensorArgs *tensor_args = &g_pgo_tensor_args;" << std::endl;
  ss << "  uint64_t workspace_size = 0;" << std::endl;
  ss << "  uint32_t block_dim = 0;" << std::endl;
  ss << "  if (pgo_search_fn == nullptr) {" << std::endl;
  ss << "    DLOGE(\"pgo search func not found\");" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  int64_t result = pgo_search_fn((char*)search_file, (char *)config_file, &tiling_data, &workspace_size, "
        "&block_dim, &g_res_limit,"
     << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "&g_stream, reinterpret_cast<void*>(PGOGetProfiling), reinterpret_cast<void*>(PGOGetProfilingBatch));"
     << std::endl;
  ss << "  if (result != 0) {" << std::endl;
  ss << "    DLOGW(\"pgo search failed, status: %\" PRId64, result);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoStaticFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                 std::stringstream &ss) const {
  ss << "int static_pgo(const char* config_file) {" << std::endl;
  ss << "  PgoDsoCallGuard dso_guard;" << std::endl;
  ss << "  if (!dso_guard) { return FAILED; }" << std::endl;
  ss << "  if (autofuse_tiling_with_config_fn == nullptr) {" << std::endl;
  ss << "    DLOGE(\"autofuse tiling with config func not found\");" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  AutofuseTilingData tiling_data = {0};" << std::endl;
  ss << "  PgoTensorArgs *tensor_args = &g_pgo_tensor_args;" << std::endl;
  ss << "  uint64_t workspace_size = 0;" << std::endl;
  ss << "  uint32_t block_dim = 0;" << std::endl;
  ss << "  int64_t result = autofuse_tiling_with_config_fn(config_file, &tiling_data, &workspace_size, &block_dim, "
        "&g_res_limit, -1);"
     << std::endl;
  ss << "  if (result != 0) {" << std::endl;
  ss << "    DLOGW(\"autofuse tiling with config failed, status: %\" PRId64, result);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  double out_cost = DBL_MAX;" << std::endl;
  ss << "  for (int i = 0; i < kPgoRetryCount; i++) {" << std::endl;
  ss << "    result = PGOGetProfiling(" << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "g_stream, workspace_size, &tiling_data, &out_cost);" << std::endl;
  ss << "    if (result == kPgoProfileUnsupported || result == kPgoReporterBusy || result == kPgoStopFailed) {"
     << std::endl;
  ss << "      return static_cast<int>(result);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    if (result != 0 || IsEqual(out_cost, DBL_MAX)) {" << std::endl;
  ss << "      DLOGD(\"get profiling failed, status: %\" PRId64, result);" << std::endl;
  ss << "      if (std::getenv(\"AF_PGO_STARS_DIAGNOSTICS\") != nullptr) { break; }" << std::endl;
  ss << "    } else {" << std::endl;
  ss << "      break;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  AppendPgoSearchTilingData(tiling_data, out_cost);" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoProfiling(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "typedef int64_t (*PGOSearchType)(char *search_file, char *config_file, AutofuseTilingData *tiling_data, "
        "uint64_t *workspace_size, uint32_t *blockDim, void *resource_limit, "
     << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "void *stream, void *prof_callback, void *prof_batch_callback);" << std::endl;
  ss << "static PGOSearchType pgo_search_fn = reinterpret_cast<PGOSearchType>(GetFunc(\"PgoTilingSearch\"));"
     << std::endl;
  GenPgoFunc(fused_schedule_result, ss);
  ss << "typedef int64_t (*AutofuseTilingWithConfigType)(const char *config_file, AutofuseTilingData *tiling, "
        "uint64_t *workspace_size, uint32_t *blockDim, ResLimit *res_limit, int32_t tiling_case_id);"
     << std::endl;
  ss << "static AutofuseTilingWithConfigType autofuse_tiling_with_config_fn = "
     << "reinterpret_cast<AutofuseTilingWithConfigType>(GetFunc(\"AutofuseTilingWithConfig\"));" << std::endl;
  GenPgoStaticFunc(fused_schedule_result, ss);
}

void TilingLib::GenPgoMain(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "int main(int argc, char *argv[]) {" << std::endl;
  ss << "  if (argc != 6 && argc != 7) {" << std::endl;
  ss << "    DLOGE(\"Usage: %s <type> <device_id> <aiv_num> <ub_size> <kernel_name> [lock_dir]\", argv[0]);"
     << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  int32_t type = static_cast<int32_t>(atoi(argv[1]));" << std::endl;
  ss << "  int32_t device_id = static_cast<int32_t>(atoi(argv[2]));" << std::endl;
  ss << "  g_pgo_device_id = static_cast<uint32_t>(device_id);" << std::endl;
  ss << "  int32_t aiv_num = static_cast<int32_t>(atoi(argv[3]));" << std::endl;
  ss << "  int32_t ub_size = static_cast<int32_t>(atoi(argv[4]));" << std::endl;
  ss << "  g_kernel_name = argv[5];" << std::endl;
  ss << "  DLOGI(\"execute info : type: %d, device_id: %d, kernel_name: %s\", type, device_id, g_kernel_name.c_str());"
     << std::endl;
  ss << "  DLOGI(\"execute limit: aiv_num is %d, ub_size is %d bytes\", aiv_num, ub_size);" << std::endl;
  ss << "  const char *lock_dir = argc == 7 ? argv[6] : pgo_dir;" << std::endl;
  ss << "  g_npu_lock_file = std::string(lock_dir) + \"/npu_lock_\" + std::to_string(device_id) + \".lock\";"
     << std::endl;
  ss << "  g_kernel_o_file = std::string(pgo_dir) + \"/\" + g_kernel_name + \".o\";" << std::endl;
  ss << "  CardLock lock(g_npu_lock_file.c_str());" << std::endl;
  GenPgoEnvInit(fused_schedule_result, ss);
  ss << "  if (type == 0) {" << std::endl;
  ss << "    ret = pgo();" << std::endl;
  ss << "  } else if (type == 1) {" << std::endl;
  ss << "    g_is_static_kernel = true;" << std::endl;
  ss << "    ret = static_pgo(config_file);" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    DLOGE(\"Invalid type: %d, expected 0 or 1\", type);" << std::endl;
  ss << "    ret = -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (!g_pgo_attempt_recorded && g_pgo_preflight_status != 0) { ret = g_pgo_preflight_status; }" << std::endl;
  GenPgoDeinit(fused_schedule_result, ss);
  ss << "  return ret;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoEnvInit(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "  auto pgo_env_cleanup = [&]() {" << std::endl;
  ss << "    aclError ret = ACL_SUCCESS;" << std::endl;
  ss << "    LaunchParamsDeInit();" << std::endl;
  ss << PGOSearchTensorFreeDef(fused_schedule_result);
  ss << "    if (g_tiling_device_addr != nullptr) {" << std::endl;
  ss << "      ret = aclrtFree(g_tiling_device_addr);" << std::endl;
  ss << "      g_tiling_device_addr = nullptr;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    if (g_stream != nullptr) {" << std::endl;
  ss << "      ret = aclrtDestroyStream(g_stream);" << std::endl;
  ss << "      g_stream = nullptr;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    (void)ret;" << std::endl;
  ss << "    (void)aclrtResetDevice(device_id);" << std::endl;
  ss << "    (void)aclFinalize();" << std::endl;
  ss << "  };" << std::endl;
  ss << "  g_res_limit.aiv_num = aiv_num;" << std::endl;
  ss << "  g_res_limit.ub_size = ub_size;" << std::endl;
  ss << "  auto ret = aclInit(nullptr);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl init failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtSetDevice(device_id);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl set device failed, device id: %d, ERROR: %d\", device_id, ret);" << std::endl;
  ss << "    aclFinalize();" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtCreateStream(&g_stream);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl create stream failed, ERROR: %d\", ret);" << std::endl;
  ss << "    aclrtResetDevice(device_id);" << std::endl;
  ss << "    aclFinalize();" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << PGOSearchTensorMallocDef(fused_schedule_result, "pgo_env_cleanup();") << std::endl;
  ss << PGOSearchTensorArgsUpdateDef(fused_schedule_result);
  ss << "  ret = aclrtMalloc(&g_tiling_device_addr, sizeof(AutofuseTilingData), ACL_MEM_MALLOC_HUGE_FIRST);"
     << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl malloc tiling data failed, ERROR: %d\", ret);" << std::endl;
  ss << "    pgo_env_cleanup();" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = LaunchParamsInit(&g_pgo_tensor_args);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    pgo_env_cleanup();" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenInductorPgoKernelFunctionInit(std::stringstream &ss) const {
  ss << "    aclrtFuncHandle func_handle = nullptr;" << std::endl;
  ss << "    ret = aclrtBinaryGetFunction(g_pgo_bin_handle, kInductorPgoKernelName, &func_handle);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"acl get function failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    std::fill(func_handles.begin(), func_handles.end(), func_handle);" << std::endl;
}

void TilingLib::GenPgoKernelFunctionsInit(const std::string &bin_handle, std::stringstream &ss) const {
  ss << "    if (g_is_static_kernel) {" << std::endl;
  ss << "      aclrtFuncHandle func_handle = nullptr;" << std::endl;
  ss << "      ret = aclrtBinaryGetFunction(" << bin_handle
     << ", (g_kernel_name + \"_\" + std::to_string(tiling_key)).c_str(), &func_handle);" << std::endl;
  ss << "      if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "        DLOGE(\"acl get function failed, ERROR: %d\", ret);" << std::endl;
  ss << "        return FAILED;" << std::endl;
  ss << "      }" << std::endl;
  ss << "      func_handles[tiling_key] = func_handle;" << std::endl;
  ss << "    } else {" << std::endl;
  ss << "      for (uint64_t i = 0; i < tiling_key_count; ++i) {" << std::endl;
  ss << "        aclrtFuncHandle func_handle = nullptr;" << std::endl;
  ss << "        ret = aclrtBinaryGetFunction(" << bin_handle
     << ", (g_kernel_name + \"_\" + std::to_string(i)).c_str(), &func_handle);" << std::endl;
  ss << "        if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "          DLOGE(\"acl get function failed, ERROR: %d\", ret);" << std::endl;
  ss << "          return FAILED;" << std::endl;
  ss << "        }" << std::endl;
  ss << "        func_handles[i] = func_handle;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
}

void TilingLib::GenPgoLaunchKernelInit(std::stringstream &ss, bool direct_link) const {
  const std::string bin_handle = direct_link ? "g_pgo_bin_handle" : "bin_handle";
  ss << "  if (!inited) {" << std::endl;
  ss << "    auto ret = aclrtBinaryLoadFromFile(g_kernel_o_file.c_str(), nullptr, &" << bin_handle << ");" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"acl load binary from file %s failed, ERROR: %d\", g_kernel_o_file.c_str(), ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  if (direct_link) {
    GenInductorPgoKernelFunctionInit(ss);
  } else {
    GenPgoKernelFunctionsInit(bin_handle, ss);
  }
  const auto backend_spce = optimize::BackendSpec::GetInstance();
  if (backend_spce != nullptr && backend_spce->set_local_memory_size > 0) {
    ss << "    local_memory_size_attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;" << std::endl;
    ss << "    local_memory_size_attr.value.dynUBufSize = " << backend_spce->set_local_memory_size << ";" << std::endl;
    ss << "    kernel_cfg.numAttrs = 1;" << std::endl;
    ss << "    kernel_cfg.attrs = &local_memory_size_attr;" << std::endl;
  }
  ss << "    inited = true;" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoDeinit(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "  LaunchParamsDeInit();" << std::endl;
  ss << PGOSearchTensorFreeDef(fused_schedule_result) << std::endl;
  ss << "  if (g_tiling_device_addr != nullptr) {" << std::endl;
  ss << "    ret = aclrtFree(g_tiling_device_addr);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"acl free tiling data failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_tiling_device_addr = nullptr;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtDestroyStream(g_stream);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl destroy stream failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtResetDevice(device_id);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl reset device failed, device id: %d, ERROR: %d\", device_id, ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclFinalize();" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl finalize failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  DeInit();" << std::endl;
}

}  // namespace codegen
