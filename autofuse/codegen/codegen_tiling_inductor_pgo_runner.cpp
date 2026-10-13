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
void AppendInductorPgoLoaderState(std::stringstream &ss) {
  ss << R"(
void *g_pgo_tiling_handle = nullptr;
std::atomic<uint32_t> g_pgo_active_calls{0};
std::atomic<bool> g_pgo_closing{false};
std::mutex g_pgo_dso_mutex;
std::condition_variable g_pgo_dso_cv;
)" << std::endl;
  AppendPgoDlopenFlags(ss);
  AppendPgoDsoCallGuard(ss, "g_pgo_tiling_handle", "true", "g_pgo_closing", "g_pgo_active_calls", "g_pgo_dso_mutex",
                        "g_pgo_dso_cv");
}

void AppendInductorPgoLoaderFunctions(std::stringstream &ss) {
  ss << R"(
GenerateMeasuredTopnSolutionsType generate_measured_topn_solutions_fn = nullptr;
SetTopnPgoContextType set_topn_pgo_context_fn = nullptr;
ClearTopnPgoContextType clear_topn_pgo_context_fn = nullptr;
GetTilingDataReprType get_tiling_data_repr_fn = nullptr;

template <typename T>
bool LoadInductorPgoSymbol(T &target, const char *name) {
  target = reinterpret_cast<T>(dlsym(g_pgo_tiling_handle, name));
  if (target == nullptr) { DLOGE("dlsym %s failed: %s", name, dlerror()); return false; }
  return true;
}
)" << std::endl;
}

void AppendInductorPgoLoadUnload(std::stringstream &ss) {
  ss << R"(
int LoadInductorPgoHost(const InductorPgoRunnerArgs &args) {
  std::lock_guard<std::mutex> lock(g_pgo_dso_mutex);
  g_pgo_closing.store(false, std::memory_order_release);
  g_pgo_tiling_handle = dlopen(args.tiling_file.c_str(), kPgoDlopenFlags);
  if (g_pgo_tiling_handle == nullptr) { DLOGE("dlopen tiling failed: %s", dlerror()); return FAILED; }
  bool valid = LoadInductorPgoSymbol(generate_measured_topn_solutions_fn, "GenerateMeasuredTopnSolutions") &&
      LoadInductorPgoSymbol(set_topn_pgo_context_fn, "SetTopnPgoContext") &&
      LoadInductorPgoSymbol(clear_topn_pgo_context_fn, "ClearTopnPgoContext") &&
      LoadInductorPgoSymbol(get_tiling_data_repr_fn, "GetTilingDataRepr") &&
      LoadInductorPgoSymbol(get_tiling_key_count_fn, "GetTilingKeyCount") &&
      LoadInductorPgoSymbol(find_best_tiling_key_fn, "FindBestTilingKey");
  if (!valid) {
    generate_measured_topn_solutions_fn = nullptr;
    set_topn_pgo_context_fn = nullptr;
    clear_topn_pgo_context_fn = nullptr;
    get_tiling_data_repr_fn = nullptr;
    get_tiling_key_count_fn = nullptr;
    find_best_tiling_key_fn = nullptr;
    if (!kPgoDlopenNodelete) { dlclose(g_pgo_tiling_handle); }
    g_pgo_tiling_handle = nullptr;
    g_pgo_closing.store(true, std::memory_order_release);
    return FAILED;
  }
  return valid ? SUCCESS : FAILED;
}

void UnloadInductorPgoHost() {
  std::unique_lock<std::mutex> lock(g_pgo_dso_mutex);
  g_pgo_closing.store(true, std::memory_order_release);
  g_pgo_dso_cv.wait(lock, [] { return g_pgo_active_calls.load(std::memory_order_acquire) == 0; });
  generate_measured_topn_solutions_fn = nullptr; set_topn_pgo_context_fn = nullptr;
  clear_topn_pgo_context_fn = nullptr; get_tiling_data_repr_fn = nullptr;
  get_tiling_key_count_fn = nullptr; find_best_tiling_key_fn = nullptr;
  if (g_pgo_tiling_handle != nullptr) {
    if (!kPgoDlopenNodelete) { dlclose(g_pgo_tiling_handle); }
    g_pgo_tiling_handle = nullptr;
  }
}
)" << std::endl;
}
}  // namespace

void TilingLib::GenInductorPgoHostLoader(std::stringstream &ss) const {
  AppendInductorPgoLoaderState(ss);
  AppendInductorPgoLoaderFunctions(ss);
  AppendInductorPgoLoadUnload(ss);
}

void TilingLib::GenInductorPgoResultProtocol(std::stringstream &ss) const {
  GenInductorPgoRecordWriter(ss);
  GenInductorPgoResultWriter(ss);
  GenPgoSaveTilingKey(ss);
  GenInductorPgoSearchWriter(ss);
  GenInductorPgoArgValidators(ss);
  GenInductorPgoArgParser(ss);
  GenInductorPgoContextGuard(ss);
}

void TilingLib::GenInductorPgoResultTypes(std::stringstream &ss) const {
  ss << R"(
namespace {
constexpr char kPgoTopnMagic[] = "AUTOFUSE_PGO_TOPN_V2";
constexpr size_t kPgoTopnMagicSize = 20U;
constexpr uint32_t kPgoTopnProtocolVersion = 2U;
constexpr uint32_t kPgoTopnProtocolFlags = 0U;
constexpr uint32_t kPgoTopnRecordHeaderSize = 32U;
constexpr size_t kMaxPgoReprSize = 16U * 1024U * 1024U;
constexpr int64_t kMaxPgoTopn = 1024;
static_assert(sizeof(kPgoTopnMagic) - 1U == kPgoTopnMagicSize);
static_assert(std::is_trivially_copyable<AutofuseTilingData>::value);
static_assert(std::is_standard_layout<AutofuseTilingData>::value);

using GenerateMeasuredTopnSolutionsType = int64_t (*)(
    const std::vector<std::map<std::string, std::string>> &, int64_t,
    std::vector<AutofuseTilingData> &, std::vector<int64_t> &, std::vector<int64_t> &, ResLimit *);
using InductorPgoProfilingCallback = long int (*)(
    PgoTensorArgs *, void *, uint64_t, AutofuseTilingData *, double *);
using InductorPgoProfilingBatchCallback = long int (*)(
    PgoTensorArgs *, void *, uint64_t, std::vector<AutofuseTilingDataPerf> *);
using SetTopnPgoContextType = int64_t (*)(
    PgoTensorArgs *, void *, InductorPgoProfilingCallback, InductorPgoProfilingBatchCallback,
    std::vector<AutofuseTilingDataPerf> *);
using ClearTopnPgoContextType = void (*)(void);
using GetTilingDataReprType = std::string (*)(const AutofuseTilingData *);

struct InductorPgoRunnerArgs {
  int32_t device_id = -1;
  uint32_t aiv_num = 0;
  uint32_t ub_size = 0;
  std::string kernel_name;
  std::string tiling_file;
  std::string kernel_file;
  std::string result_file;
  int64_t topn = 0;
};
)" << std::endl;
}

void TilingLib::GenInductorPgoRecordWriter(std::stringstream &ss) const {
  ss << R"(
template <typename T>
bool WritePgoValue(std::ofstream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
  return out.good();
}

uint64_t HashPgoBytes(const void *data, size_t size) {
  const auto *bytes = static_cast<const uint8_t *>(data);
  uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0; i < size; ++i) { hash = (hash ^ bytes[i]) * 1099511628211ULL; }
  return hash;
}

bool WritePgoRecord(std::ofstream &out, const AutofuseTilingData &tiling_data,
                    uint64_t workspace, int64_t block_dim) {
  PgoDsoCallGuard dso_guard;
  if (!dso_guard) { return false; }
  if (get_tiling_data_repr_fn == nullptr) { return false; }
  const std::string repr = get_tiling_data_repr_fn(&tiling_data);
  if (repr.empty() || repr.size() > kMaxPgoReprSize ||
      workspace > static_cast<uint64_t>(INT64_MAX) || block_dim <= 0 || block_dim > UINT32_MAX) {
    return false;
  }
  const uint64_t repr_len = repr.size();
  const uint64_t tiling_hash = HashPgoBytes(&tiling_data, sizeof(tiling_data));
  return WritePgoValue(out, repr_len) && WritePgoValue(out, workspace) &&
         WritePgoValue(out, block_dim) && WritePgoValue(out, tiling_hash) &&
         WritePgoValue(out, tiling_data) &&
         static_cast<bool>(out.write(repr.data(), static_cast<std::streamsize>(repr.size())));
}
)" << std::endl;
}

void TilingLib::GenInductorPgoResultWriter(std::stringstream &ss) const {
  ss << R"(
int WritePgoTopnResult(const std::string &path, const std::vector<AutofuseTilingData> &tiling_datas,
                       const std::vector<int64_t> &workspaces, const std::vector<int64_t> &block_dims) {
  if (tiling_datas.empty() || tiling_datas.size() != workspaces.size() || tiling_datas.size() != block_dims.size() ||
      tiling_datas.size() > std::numeric_limits<uint32_t>::max()) {
    return FAILED;
  }
  const std::string tmp_path = path + ".tmp";
  std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
  const uint32_t count = static_cast<uint32_t>(tiling_datas.size());
  const uint32_t tiling_size = sizeof(AutofuseTilingData);
  out.write(kPgoTopnMagic, kPgoTopnMagicSize);
  if (!out.good() || !WritePgoValue(out, kPgoTopnProtocolVersion) || !WritePgoValue(out, kPgoTopnProtocolFlags) ||
      !WritePgoValue(out, count) || !WritePgoValue(out, tiling_size) ||
      !WritePgoValue(out, kPgoTopnRecordHeaderSize)) {
    out.close(); std::remove(tmp_path.c_str()); return FAILED;
  }
  for (size_t i = 0; i < tiling_datas.size(); ++i) {
    if (workspaces[i] < 0 ||
        !WritePgoRecord(out, tiling_datas[i], static_cast<uint64_t>(workspaces[i]), block_dims[i])) {
      out.close(); std::remove(tmp_path.c_str()); return FAILED;
    }
  }
  out.flush();
  if (!out.good()) { out.close(); std::remove(tmp_path.c_str()); return FAILED; }
  out.close();
  const int fd = ::open(tmp_path.c_str(), O_RDONLY);
  if (fd < 0 || ::fsync(fd) != 0) {
    if (fd >= 0) { ::close(fd); }
    std::remove(tmp_path.c_str()); return FAILED;
  }
  ::close(fd);
  if (std::rename(tmp_path.c_str(), path.c_str()) != 0) { std::remove(tmp_path.c_str()); return FAILED; }
  return SUCCESS;
}
)" << std::endl;
}

void TilingLib::GenInductorPgoSearchWriter(std::stringstream &ss) const {
  ss << R"(
std::string PgoParentPath(const std::string &path) {
  const auto pos = path.find_last_of('/');
  return pos == std::string::npos ? "." : path.substr(0U, pos);
}

int PublishPgoSearchFile(const std::string &tmp_path, const std::string &path) {
  const int fd = ::open(tmp_path.c_str(), O_RDONLY);
  if (fd < 0 || ::fsync(fd) != 0) {
    if (fd >= 0) { ::close(fd); }
    std::remove(tmp_path.c_str());
    return FAILED;
  }
  ::close(fd);
  if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
    std::remove(tmp_path.c_str());
    return FAILED;
  }
  return SUCCESS;
}

int WritePgoSearchResult(const InductorPgoRunnerArgs &args,
                         const std::vector<AutofuseTilingDataPerf> &measured_candidates) {
  if (measured_candidates.empty()) { return FAILED; }
  const std::string path = PgoParentPath(args.kernel_file) + "/" + std::string(PGO_GRAPH_NAME) + "_search.txt";
  const std::string tmp_path = path + ".tmp." + std::to_string(getpid());
  std::ofstream out(tmp_path, std::ios::out | std::ios::trunc);
  if (!out.is_open()) { return FAILED; }
  for (const auto &candidate : measured_candidates) {
    PgoSaveTilingKey(candidate.tiling_data, candidate.best_perf, out);
    if (!out.good()) { out.close(); std::remove(tmp_path.c_str()); return FAILED; }
  }
  out.flush();
  if (!out.good()) { out.close(); std::remove(tmp_path.c_str()); return FAILED; }
  out.close();
  return PublishPgoSearchFile(tmp_path, path);
}
)" << std::endl;
}

void TilingLib::GenInductorPgoArgValidators(std::stringstream &ss) const {
  ss << R"(
bool ParseRunnerInteger(const char *text, int64_t min_value, int64_t max_value, int64_t &value) {
  if (text == nullptr || *text == '\0') { return false; }
  errno = 0;
  char *end = nullptr;
  const long long parsed = std::strtoll(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) { return false; }
  value = static_cast<int64_t>(parsed);
  return true;
}

bool IsValidKernelName(const std::string &name) {
  return !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char c) {
    return std::isalnum(c) != 0 || c == '_';
  });
}
)" << std::endl;
}

void TilingLib::GenInductorPgoArgParser(std::stringstream &ss) const {
  ss << R"(
int ParseInductorPgoRunnerArgs(int argc, char *argv[], InductorPgoRunnerArgs &args) {
  if (argc != 9) {
    DLOGE("Usage: %s <device_id> <aiv_num> <ub_size> <kernel_name> <tiling_so> <dynamic_kernel> <result> <topn>",
          argv[0]);
    return FAILED;
  }
  int64_t device_id = 0;
  int64_t aiv_num = 0;
  int64_t ub_size = 0;
  if (!ParseRunnerInteger(argv[1], 0, INT32_MAX, device_id) ||
      !ParseRunnerInteger(argv[2], 1, UINT32_MAX, aiv_num) ||
      !ParseRunnerInteger(argv[3], 257, UINT32_MAX, ub_size) ||
      !ParseRunnerInteger(argv[8], 1, kMaxPgoTopn, args.topn)) {
    DLOGE("invalid numeric runner argument"); return FAILED;
  }
  args.device_id = static_cast<int32_t>(device_id);
  args.aiv_num = static_cast<uint32_t>(aiv_num);
  args.ub_size = static_cast<uint32_t>(ub_size);
  args.kernel_name = argv[4]; args.tiling_file = argv[5];
  args.kernel_file = argv[6]; args.result_file = argv[7];
  if (!IsValidKernelName(args.kernel_name) || args.tiling_file.empty() || args.kernel_file.empty() ||
      args.result_file.empty() || ::access(args.tiling_file.c_str(), R_OK) != 0 ||
      ::access(args.kernel_file.c_str(), R_OK) != 0) {
    DLOGE("invalid runner path or kernel name"); return FAILED;
  }
  return SUCCESS;
}
)" << std::endl;
}

void TilingLib::GenInductorPgoContextGuard(std::stringstream &ss) const {
  ss << R"(
class PgoContextGuard {
 public:
  explicit PgoContextGuard(std::vector<AutofuseTilingDataPerf> *measured_candidates) {
    dso_guard_ = std::make_unique<PgoDsoCallGuard>();
    if (*dso_guard_ && set_topn_pgo_context_fn != nullptr) {
      valid_ = set_topn_pgo_context_fn(&g_pgo_tensor_args, g_stream, PGOGetProfiling, PGOGetProfilingBatch,
                                       measured_candidates) == 0;
    }
  }
  ~PgoContextGuard() {
    if (valid_ && clear_topn_pgo_context_fn != nullptr) { clear_topn_pgo_context_fn(); }
  }
  bool IsValid() const { return valid_; }
  PgoContextGuard(const PgoContextGuard &) = delete;
  PgoContextGuard &operator=(const PgoContextGuard &) = delete;
 private:
  bool valid_ = false;
  std::unique_ptr<PgoDsoCallGuard> dso_guard_;
};
}  // namespace
)" << std::endl;
}

void TilingLib::GenInductorPgoRuntime(const ascir::FusedScheduledResult &fused_schedule_result,
                                      std::stringstream &ss) const {
  GenInductorPgoAclRuntime(ss);
  GenInductorPgoMemoryRuntime(fused_schedule_result, ss);
  GenInductorPgoDeinitRuntime(fused_schedule_result, ss);
}

void TilingLib::GenInductorPgoAclRuntime(std::stringstream &ss) const {
  ss << R"(
int InitInductorPgoAcl(const InductorPgoRunnerArgs &args) {
  g_res_limit.aiv_num = args.aiv_num;
  g_res_limit.ub_size = args.ub_size;
  auto ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) { DLOGE("acl init failed, ERROR: %d", ret); return FAILED; }
  g_acl_initialized = true;
  ret = aclrtSetDevice(args.device_id);
  if (ret != ACL_SUCCESS) { DLOGE("acl set device failed, ERROR: %d", ret); return FAILED; }
  g_device_id = args.device_id;
  g_device_set = true;
  ret = aclrtCreateStream(&g_stream);
  if (ret != ACL_SUCCESS) { DLOGE("acl create stream failed, ERROR: %d", ret); return FAILED; }
  return SUCCESS;
}
)" << std::endl;
}

void TilingLib::GenInductorPgoMemoryRuntime(const ascir::FusedScheduledResult &fused_schedule_result,
                                            std::stringstream &ss) const {
  ss << R"(
int InitInductorPgoMemory() {
  aclError ret = ACL_SUCCESS;
)";
  ss << PGOSearchTensorMallocDef(fused_schedule_result);
  ss << PGOSearchTensorArgsUpdateDef(fused_schedule_result);
  ss << R"(  ret = LaunchParamsInit(&g_pgo_tensor_args);
  if (ret != ACL_SUCCESS) { return FAILED; }
  return SUCCESS;
}
)" << std::endl;
}

void TilingLib::GenInductorPgoDeinitRuntime(const ascir::FusedScheduledResult &fused_schedule_result,
                                            std::stringstream &ss) const {
  ss << R"(
void DeInitInductorPgoMemory() {
  aclError ret = ACL_SUCCESS;
  if (g_workspace != nullptr) {
    ret = aclrtFree(g_workspace);
    if (ret != ACL_SUCCESS) { DLOGW("acl free workspace failed, ERROR: %d", ret); }
    g_workspace = nullptr;
  }
  PgoBinaryDeInit();
  LaunchParamsDeInit();
)";
  ss << PGOSearchTensorFreeDef(fused_schedule_result);
  ss << R"(}

void DeInitInductorPgoAcl() {
  if (g_stream != nullptr) {
    auto ret = aclrtDestroyStream(g_stream);
    if (ret != ACL_SUCCESS) { DLOGW("acl destroy stream failed, ERROR: %d", ret); }
    g_stream = nullptr;
  }
  if (g_device_set) {
    auto ret = aclrtResetDevice(g_device_id);
    if (ret != ACL_SUCCESS) { DLOGW("acl reset device failed, ERROR: %d", ret); }
    g_device_set = false;
  }
  if (g_acl_initialized) {
    auto ret = aclFinalize();
    if (ret != ACL_SUCCESS) { DLOGW("acl finalize failed, ERROR: %d", ret); }
    g_acl_initialized = false;
  }
}

void DeInitInductorPgoRuntime() {
  DeInitInductorPgoMemory();
  DeInitInductorPgoAcl();
}
)" << std::endl;
}

void TilingLib::GenInductorPgoMain(std::stringstream &ss) const {
  ss << R"(
int RunInductorPgo(const InductorPgoRunnerArgs &args) {
  PgoDsoCallGuard dso_guard;
  if (!dso_guard) { return FAILED; }
  std::vector<AutofuseTilingDataPerf> measured_candidates;
  PgoContextGuard context_guard(&measured_candidates);
  if (!context_guard.IsValid() || generate_measured_topn_solutions_fn == nullptr) { return FAILED; }
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;
  const auto ret = generate_measured_topn_solutions_fn(
      {}, args.topn, tiling_datas, workspaces, block_dims, &g_res_limit);
  if (ret != 0) { DLOGE("GenerateMeasuredTopnSolutions failed, ERROR: %" PRId64, ret); return FAILED; }
  if (tiling_datas.empty()) { return FAILED; }
  if (WritePgoTopnResult(args.result_file, tiling_datas, workspaces, block_dims) != SUCCESS) {
    DLOGE("Write PGO TopN result failed"); return FAILED;
  }
  if (WritePgoSearchResult(args, measured_candidates) != SUCCESS) {
    DLOGW("Write PGO search result failed");
  }
  return SUCCESS;
}

int main(int argc, char *argv[]) {
  InductorPgoRunnerArgs args;
  if (ParseInductorPgoRunnerArgs(argc, argv, args) != SUCCESS) { return FAILED; }
  g_kernel_o_file = args.kernel_file;
  DLOGI("execute info: device_id: %d, graph_name: %s", args.device_id, args.kernel_name.c_str());
  const char *tmp_dir = std::getenv("TMPDIR");
  const std::string lock_dir = (tmp_dir != nullptr && *tmp_dir != '\0') ? tmp_dir : "/tmp";
  g_npu_lock_file = lock_dir + "/autofuse_pgo_npu_lock_" + std::to_string(args.device_id) + ".lock";
  CardLock lock(g_npu_lock_file.c_str());
  int ret = LoadInductorPgoHost(args);
  if (ret == SUCCESS) { ret = InitInductorPgoAcl(args); }
  if (ret == SUCCESS) { ret = InitInductorPgoMemory(); }
  if (ret == SUCCESS) { ret = RunInductorPgo(args); }
  DeInitInductorPgoRuntime();
  UnloadInductorPgoHost();
  return ret;
}
)" << std::endl;
}

}  // namespace codegen
